import { renderHook, act } from "@testing-library/react-native";

/**
 * useAudioStream — mid-call naming ("that Speaker B is Mom") and the
 * pleasantness scoreboard, driven through the real FastLoop with an energy
 * VAD, a fake recognizer, a fake on-device LLM and a scripted embedder.
 */
const mockMic = {
  onBuffer: null as
    | ((buffer: { data: ArrayBuffer; sampleRate: number; channels: number; timestamp: number }) => void)
    | null,
  start: jest.fn<Promise<void>, []>(),
  stop: jest.fn(),
  requestPermissions: jest.fn<Promise<{ status: string; granted: boolean }>, []>(),
  setAudioMode: jest.fn<Promise<void>, [unknown]>(),
};

jest.mock("expo-audio", () => ({
  __esModule: true,
  requestRecordingPermissionsAsync: () => mockMic.requestPermissions(),
  setAudioModeAsync: (mode: unknown) => mockMic.setAudioMode(mode),
  useAudioStream: (options?: { onBuffer?: (buffer: never) => void }) => {
    mockMic.onBuffer = (options?.onBuffer ?? null) as typeof mockMic.onBuffer;
    return {
      stream: { id: "mock-stream", sampleRate: 16000, channels: 1, isStreaming: false, start: mockMic.start, stop: mockMic.stop },
      isStreaming: false,
    };
  },
}));

import { useAudioStream } from "../src/hooks/useAudioStream";
import { FastLoop } from "../src/live/fastLoop";
import { EnergyVad } from "../src/live/vad";
import { FakeSpeechRecognizer } from "../src/live/stt";
import { ProviderChain, parseSuggestionJson } from "../src/live/localLlm";
import { SpeakerLabeler, type Embedder } from "../src/live/speakerId";
import type { FastLoopHandlers } from "../src/live/defaultDeps";
import type { LiveSessionBody } from "../src/api/liveSessions";
import { silenceInt16, toneInt16, unitVector } from "../src/live/testing/synth";

class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];
  url: string;
  readyState = FakeWebSocket.OPEN;
  sent: string[] = [];
  onopen: ((e: unknown) => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;
  onclose: ((e: unknown) => void) | null = null;
  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }
  send(data: string | ArrayBuffer | ArrayBufferView) {
    if (typeof data === "string") this.sent.push(data);
  }
  close() {
    this.readyState = 3;
    this.onclose?.({});
  }
  emitOpen() {
    this.onopen?.({});
  }
  emitServer(obj: unknown) {
    this.onmessage?.({ data: JSON.stringify(obj) });
  }
  sentJson() {
    return this.sent.map((s) => JSON.parse(s));
  }
}

const DIM = 8;
class ScriptedEmbedder implements Embedder {
  constructor(private readonly queue: Float32Array[]) {}
  async embed(): Promise<Float32Array> {
    const next = this.queue.shift();
    if (!next) throw new Error("no scripted embedding left");
    return next;
  }
}

// Tone reads alternate so the two voices score differently.
const TONES = [
  '{"suggestion":"Say: I miss you too.","tone":{"warmth":80,"defensiveness":10,"sarcasm":0,"frustration":10,"label":"warm"}}',
  '{"suggestion":"Say: I hear you.","tone":{"warmth":20,"defensiveness":70,"sarcasm":40,"frustration":60,"label":"defensive"}}',
];

function makeFakeFastLoop(embeddings: Float32Array[]) {
  const rec = new FakeSpeechRecognizer();
  let n = 0;
  const build = {
    rec,
    loop: null as FastLoop | null,
    async make(handlers: FastLoopHandlers) {
      build.loop = new FastLoop({
        ...handlers,
        vad: new EnergyVad(-45, 0.032),
        embedder: new ScriptedEmbedder(embeddings),
        labeler: new SpeakerLabeler([]),
        recognizer: rec,
        llm: new ProviderChain([
          { name: "os", isAvailable: async () => true, suggest: async () => parseSuggestionJson(TONES[n++ % 2]) },
        ]),
        sttGraceMs: 100,
        pollMs: 5,
        selfSpeakerFallback: null,
      });
      return {
        loop: build.loop,
        status: "energy VAD · speaker-ID on (0 enrolled) · LLM os",
        capabilities: {
          vad: "energy" as const,
          speakerId: { active: true, reason: "test", enrolled: 0, model: null, droppedForModel: 0 },
          llm: ["os"],
        },
      };
    },
  };
  return build;
}

function feed(pcm: Int16Array) {
  for (let off = 0; off < pcm.length; off += 1600) {
    const chunk = pcm.subarray(off, Math.min(off + 1600, pcm.length));
    const f32 = new Float32Array(chunk.length);
    for (let i = 0; i < chunk.length; i++) f32[i] = chunk[i] / 32768;
    mockMic.onBuffer?.({ data: f32.buffer, sampleRate: 16000, channels: 1, timestamp: 0 });
  }
}

const flush = () => new Promise((r) => setTimeout(r, 30));
let logSpy: jest.SpyInstance;

beforeEach(() => {
  FakeWebSocket.instances = [];
  (globalThis as Record<string, unknown>).WebSocket = FakeWebSocket;
  mockMic.onBuffer = null;
  mockMic.start.mockReset().mockResolvedValue(undefined);
  mockMic.stop.mockReset();
  mockMic.requestPermissions.mockReset().mockResolvedValue({ status: "granted", granted: true });
  mockMic.setAudioMode.mockReset().mockResolvedValue(undefined);
  logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
});

afterEach(() => {
  logSpy.mockRestore();
});

/** Start a live session, speak `turns` (seconds, words) alternating A/B
 *  embeddings, and return the hook + socket + fake loop. */
async function session(opts: {
  embeddings: Float32Array[];
  enroll?: jest.Mock;
  patchLabels?: jest.Mock;
  postSession?: (body: LiveSessionBody) => Promise<{ status: "created"; episodeId: string; sharedWith: string[] } | { status: "unsupported" }>;
}) {
  const fake = makeFakeFastLoop(opts.embeddings);
  const posted: LiveSessionBody[] = [];
  const hook = await renderHook(() =>
    useAudioStream({
      capability: { capable: true, reason: "ok" },
      makeFastLoop: fake.make,
      postSession: async (body) => {
        posted.push(body);
        return opts.postSession ? opts.postSession(body) : { status: "unsupported" as const };
      },
      enrollSpeaker: opts.enroll,
      patchLabels: opts.patchLabels,
    }),
  );
  await act(async () => {
    await hook.result.current.startSession("live-9", 50);
  });
  const ws = FakeWebSocket.instances[0];
  await act(() => ws.emitOpen());
  await act(async () => {
    await flush();
  });
  const speak = async (seconds: number, text: string) => {
    await act(async () => {
      feed(toneInt16(seconds, -20));
      fake.rec.emit({ text, isFinal: true });
      feed(silenceInt16(0.5));
      await fake.loop!.settle();
    });
    await act(async () => {
      await flush();
    });
  };
  return { hook, ws, fake, posted, speak };
}

describe("useAudioStream — mid-call naming", () => {
  it("names a cluster for the rest of the call: transcript, wire, server and session record", async () => {
    const a = unitVector(DIM, 0);
    const b = unitVector(DIM, 1);
    const enroll = jest.fn().mockResolvedValue({ enrollCount: 1, seconds: 3.5 });
    // Script: A, B, B, (the pooled-audio embedding the enrollment takes), B.
    const s = await session({ embeddings: [a, b, b, b, b], enroll });
    await s.speak(2.0, "hi there");
    await s.speak(2.0, "you never call");
    expect(s.hook.result.current.transcript.map((t) => t.speaker)).toEqual(["Speaker A", "Speaker B"]);
    expect(s.hook.result.current.transcript[1].speakerId).toBe("Speaker B");
    expect(s.hook.result.current.displayNameOf("Speaker B")).toBe("Speaker B");

    // Pooled audio for Speaker B is only 2 s: name her as a NEW person and
    // the flow binds at once but honestly can't learn the voice yet.
    let outcome = await act(() =>
      s.hook.result.current.labelSpeaker("Speaker B", { personId: "mom", displayName: "Mom", isSelf: false, isNew: true }),
    );
    expect(outcome.enrolled).toBe(false);
    expect(outcome.text).toMatch(/Mom is labeled for the rest of this call/);
    expect(outcome.text).toMatch(/Only 2 s of Mom’s voice so far/);
    expect(enroll).not.toHaveBeenCalled();
    // Instantly on screen …
    expect(s.hook.result.current.transcript.map((t) => t.speaker)).toEqual(["Speaker A", "Mom"]);
    expect(s.hook.result.current.speakerNames["Speaker B"]).toEqual({ personId: "mom", displayName: "Mom", isSelf: false });
    expect(s.hook.result.current.displayNameOf("Speaker B")).toBe("Mom");
    // … and on the server's running session.
    const labelMsgs = s.ws.sentJson().filter((m) => m.type === "speaker_label");
    expect(labelMsgs).toEqual([
      { type: "speaker_label", session_id: "live-9", speaker: "Speaker B", person_id: null, display_name: "Mom", is_self: false },
    ]);

    // Two more turns from her (now ≥ 3 s pooled): naming again learns the voice.
    await s.speak(2.0, "that's not fair");
    expect(s.hook.result.current.transcript[2]).toMatchObject({ speaker: "Mom", speakerId: "Speaker B" });
    const turnLocals = s.ws.sentJson().filter((m) => m.type === "turn_local");
    // The wire label stays raw; the binding rides as the person the user chose.
    expect(turnLocals[2]).toMatchObject({ speaker: "Speaker B", speaker_person_id: "mom", is_self: false });
    // The stored-label map only attaches the id once the person exists.
    expect(s.hook.result.current.speakerNames["Speaker B"].personId).toBe("mom");

    outcome = await act(() =>
      s.hook.result.current.labelSpeaker("Speaker B", { personId: "mom", displayName: "Mom", isSelf: false, isNew: true }),
    );
    expect(enroll).toHaveBeenCalledTimes(1);
    expect(enroll.mock.calls[0][0].length).toBeGreaterThanOrEqual(3 * 16000);
    expect(enroll.mock.calls[0][1]).toEqual({ personId: "mom", displayName: "Mom" });
    expect(outcome.enrolled).toBe(true);
    expect(outcome.text).toMatch(/Learned 3.5 s of Mom’s voice/);
    // A voiceprint now exists on the server: later turns carry the person id.
    await s.speak(2.0, "fine");
    const after = s.ws.sentJson().filter((m) => m.type === "turn_local");
    expect(after[3]).toMatchObject({ speaker: "Speaker B", speaker_person_id: "mom", is_self: false });
    // The server's own late identity verdict never overrides the user's name.
    await act(() =>
      s.ws.emitServer({ type: "speaker_identity", session_id: "live-9", speaker: "Speaker B", person_id: null, display_name: "Someone", is_self: false, score: 0.2 }),
    );
    expect(s.hook.result.current.transcript[1].speaker).toBe("Mom");

    await act(async () => {
      await s.hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));
    expect(s.posted).toHaveLength(1);
    const body = s.posted[0];
    expect(body.speaker_labels).toEqual({ "Speaker B": { display_name: "Mom", person_id: "mom", is_self: false } });
    // Earlier turns on that label were rewritten in the record.
    expect(body.turns.filter((t) => t.speaker === "Speaker B").map((t) => t.speaker_person_id)).toEqual(["mom", "mom", "mom"]);
    expect(body.turns.filter((t) => t.speaker === "Speaker A").map((t) => t.speaker_person_id)).toEqual([null]);
  });

  it("naming a voice as 'me' switches side-aware coaching and the self chip", async () => {
    const a = unitVector(DIM, 0);
    const b = unitVector(DIM, 1);
    const s = await session({ embeddings: [a, b, b] });
    await s.speak(2.0, "one");
    await s.speak(2.0, "two");
    await act(() =>
      s.hook.result.current.labelSpeaker("Speaker B", { personId: "self", displayName: "You", isSelf: true, isNew: false }),
    );
    expect(s.hook.result.current.selfSpeaker).toBe("Speaker B");
    expect(s.ws.sentJson().filter((m) => m.type === "speaker_label")[0]).toMatchObject({ is_self: true, person_id: "self" });
    await s.speak(2.0, "three");
    const turnLocals = s.ws.sentJson().filter((m) => m.type === "turn_local");
    expect(turnLocals[2]).toMatchObject({ speaker: "Speaker B", speaker_person_id: "self", is_self: true });
    // A self turn gets a nudge, not a response.
    expect(s.hook.result.current.suggestions[0].kind).toBe("nudge");
    await act(async () => {
      await s.hook.result.current.stopSession();
    });
  });

  it("after the session, naming PATCHes the stored episode like 'Who is this?' does", async () => {
    const a = unitVector(DIM, 0);
    const b = unitVector(DIM, 1);
    const patchLabels = jest.fn().mockResolvedValue({});
    const s = await session({
      embeddings: [a, b],
      patchLabels,
      postSession: async () => ({ status: "created", episodeId: "ep-1", sharedWith: ["t@x.com"] }),
    });
    await s.speak(2.0, "one");
    await s.speak(2.0, "two");
    await act(async () => {
      await s.hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));
    await act(async () => {
      await flush();
    });
    expect(s.hook.result.current.lastEpisode?.episodeId).toBe("ep-1");
    const outcome = await act(() =>
      s.hook.result.current.labelSpeaker("Speaker A", { personId: "dad", displayName: "Dad", isSelf: false, isNew: false }),
    );
    expect(patchLabels).toHaveBeenCalledWith("ep-1", { "Speaker A": "Dad" }, { "Speaker A": "dad" });
    expect(outcome.text).toMatch(/Saved to the session record/);
    expect(s.hook.result.current.transcript[0].speaker).toBe("Dad");
  });

  it("scores every on-device turn onto the scoreboard, keyed by raw label", async () => {
    const a = unitVector(DIM, 0);
    const b = unitVector(DIM, 1);
    const s = await session({ embeddings: [a, b, a, b] });
    expect(s.hook.result.current.scoreboard).toBeNull();
    await s.speak(2.0, "hi");
    await s.speak(2.0, "you never call");
    await s.speak(2.0, "i'm sorry");
    await s.speak(2.0, "whatever");
    const board = s.hook.result.current.scoreboard!;
    expect(board.people.map((p) => p.speaker)).toEqual(["Speaker A", "Speaker B"]);
    expect(board.people[0].scoredTurns).toBe(2);
    expect(board.people[1].scoredTurns).toBe(2);
    // Warm tone reads on A's turns, defensive ones on B's → A leads.
    expect(board.people[0].current!).toBeGreaterThan(board.people[1].current!);
    expect(board.lead?.speaker).toBe("Speaker A");
    expect(board.people[0].series).toHaveLength(2);
    await act(async () => {
      await s.hook.result.current.stopSession();
    });
    // A new session starts a fresh board.
    await act(async () => {
      await s.hook.result.current.startSession("live-10", 50);
    });
    expect(s.hook.result.current.scoreboard).toBeNull();
    expect(s.hook.result.current.speakerNames).toEqual({});
    await act(async () => {
      await s.hook.result.current.stopSession();
    });
  });
});
