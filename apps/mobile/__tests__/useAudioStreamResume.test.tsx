import { renderHook, act } from "@testing-library/react-native";

/**
 * Session resume (server/session_resume.py) on the phone: a live session
 * survives the network dropping mid-conversation.
 *
 * The real shape of the failure is that the WebSocket dies while the user is
 * still talking. The fast loop keeps finalizing turns, and before resume they
 * went nowhere — the cloud coach (and, in a call, everyone else's screen)
 * silently missed them. Here the same drop is forced with the same FastLoop /
 * energy VAD / fake recognizer rig the other live tests use, and the
 * assertion is exactly the promise: across the drop the server sees every
 * turn, in order, exactly once.
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
import { FastLoop, newTurnUidPrefix } from "../src/live/fastLoop";
import { EnergyVad } from "../src/live/vad";
import { FakeSpeechRecognizer } from "../src/live/stt";
import { ProviderChain, parseSuggestionJson } from "../src/live/localLlm";
import type { FastLoopHandlers } from "../src/live/defaultDeps";
import type { TurnLocalEvent } from "../src/live/types";
import { silenceInt16, toneInt16 } from "../src/live/testing/synth";

/** The hook's reconnect backoff (useAudioStream RECONNECT_DELAY_MS). */
const RECONNECT_DELAY_MS = 2000;

class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];
  url: string;
  /** CONNECTING until the server accepts, like a real one — a reconnect that
   *  has not completed must not look sendable. */
  readyState = 0;
  sent: string[] = [];
  sentBinary: ArrayBuffer[] = [];
  /** Set when the socket is "dead" the way a real dropped connection is:
   *  still OPEN as far as the app can see, but every send throws. */
  broken = false;
  onopen: ((e: unknown) => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;
  onclose: ((e: unknown) => void) | null = null;
  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }
  send(data: string | ArrayBuffer | ArrayBufferView) {
    if (this.broken) throw new Error("network is gone");
    if (typeof data === "string") this.sent.push(data);
    else if (ArrayBuffer.isView(data)) this.sentBinary.push(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer);
    else this.sentBinary.push(data);
  }
  close() {
    this.readyState = 3;
    this.onclose?.({});
  }
  emitOpen() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.({});
  }
  emitServer(obj: unknown) {
    this.onmessage?.({ data: JSON.stringify(obj) });
  }
  sentJson() {
    return this.sent.map((s) => JSON.parse(s));
  }
  turnLocals(): TurnLocalEvent[] {
    return this.sentJson().filter((m) => m.type === "turn_local");
  }
  /** The network dies: sends start failing, then the socket closes. */
  drop() {
    this.broken = true;
    this.readyState = 3;
    this.onclose?.({});
  }
}

const SUGGESTION = '{"suggestion":"Say: I hear you.","tone":{"warmth":40,"label":"hurt"}}';

function makeFakeFastLoop() {
  const rec = new FakeSpeechRecognizer();
  const build = {
    rec,
    loop: null as FastLoop | null,
    async make(handlers: FastLoopHandlers) {
      build.loop = new FastLoop({
        ...handlers,
        vad: new EnergyVad(-45, 0.032),
        embedder: null,
        labeler: null,
        recognizer: rec,
        llm: new ProviderChain([
          { name: "os", isAvailable: async () => true, suggest: async () => parseSuggestionJson(SUGGESTION) },
        ]),
        sttGraceMs: 100,
        pollMs: 5,
      });
      return {
        loop: build.loop,
        status: "energy VAD · speaker-ID off · LLM os",
        capabilities: {
          vad: "energy" as const,
          speakerId: { active: false, reason: "test", enrolled: 0, model: null, droppedForModel: 0 },
          llm: ["os"],
        },
      };
    },
  };
  return build;
}

/** Feed int16 PCM to the hook as float32 expo-audio buffers (100 ms each). */
function feed(pcm: Int16Array) {
  for (let off = 0; off < pcm.length; off += 1600) {
    const chunk = pcm.subarray(off, Math.min(off + 1600, pcm.length));
    const f32 = new Float32Array(chunk.length);
    for (let i = 0; i < chunk.length; i++) f32[i] = chunk[i] / 32768;
    mockMic.onBuffer?.({ data: f32.buffer, sampleRate: 16000, channels: 1, timestamp: 0 });
  }
}

const flush = () => new Promise((r) => setTimeout(r, 30));
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

let logSpy: jest.SpyInstance;
let warnSpy: jest.SpyInstance;

beforeEach(() => {
  FakeWebSocket.instances = [];
  (globalThis as Record<string, unknown>).WebSocket = FakeWebSocket;
  mockMic.onBuffer = null;
  mockMic.start.mockReset().mockResolvedValue(undefined);
  mockMic.stop.mockReset();
  mockMic.requestPermissions.mockReset().mockResolvedValue({ status: "granted", granted: true });
  mockMic.setAudioMode.mockReset().mockResolvedValue(undefined);
  logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
  warnSpy = jest.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  logSpy.mockRestore();
  warnSpy.mockRestore();
});

describe("turn_uid", () => {
  it("is unique per session and stable per turn", () => {
    const a = newTurnUidPrefix(() => 0.1);
    const b = newTurnUidPrefix(() => 0.9);
    expect(a).not.toBe(b);
    // The server's shape bound: [A-Za-z0-9._:-]{1,64}.
    expect(`${a}-1`).toMatch(/^[A-Za-z0-9._:-]{1,64}$/);
  });
});

describe("useAudioStream across a network drop", () => {
  async function startLiveSession({ live = true }: { live?: boolean } = {}) {
    const fake = makeFakeFastLoop();
    const hook = await renderHook(() =>
      useAudioStream({
        capability: live ? { capable: true, reason: "ok" } : { capable: false, reason: "no on-device STT" },
        makeFastLoop: fake.make,
        postSession: async () => ({ status: "unsupported" as const }),
      }),
    );
    await act(async () => {
      await hook.result.current.startSession("live-resume", 60);
    });
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    await act(async () => {
      await flush();
    });
    return { fake, hook, ws };
  }

  /** One finalized turn: a second of speech, the words, then silence. */
  async function speak(fake: ReturnType<typeof makeFakeFastLoop>, text: string) {
    await act(async () => {
      feed(toneInt16(1.0, -20));
      fake.rec.emit({ text, isFinal: true });
      feed(silenceInt16(0.5));
      await fake.loop!.settle();
    });
    await act(async () => {
      await flush();
    });
  }

  it("loses no turn and duplicates none: the offline ones flush on resume", async () => {
    const { fake, hook, ws } = await startLiveSession();

    await speak(fake, "you never call me");
    expect(ws.turnLocals()).toHaveLength(1);
    const firstUid = ws.turnLocals()[0].turn_uid;
    expect(firstUid).toBeTruthy();

    // The network dies. The user keeps talking; the phone keeps hearing.
    await act(() => ws.drop());
    await speak(fake, "I am busy at work");
    await speak(fake, "please stop shouting at me");
    // Nothing more reached the dead socket…
    expect(ws.turnLocals()).toHaveLength(1);
    // …and the session is not pretending to be live.
    expect(hook.result.current.connectionStatus).toBe("disconnected");

    // The hook reconnects on its own backoff.
    await act(async () => {
      await wait(RECONNECT_DELAY_MS + 200);
    });
    const ws2 = FakeWebSocket.instances[1];
    expect(ws2).toBeTruthy();
    await act(() => ws2.emitOpen());
    await act(async () => {
      await flush();
    });

    const frames = ws2.sentJson();
    // Order matters: config (auth) → resume (clock + de-dup) → the flush.
    expect(frames[0].type).toBe("config");
    expect(frames[1].type).toBe("resume");
    expect(frames[1].session_id).toBe("live-resume");
    expect(frames[1].since_seq).toBe(0);          // solo session: no merged seq
    // The phone's capture clock, not zero: it kept capturing through the drop.
    expect(frames[1].last_local_time).toBeGreaterThan(1);
    expect(frames.slice(2).every((m) => m.type === "turn_local")).toBe(true);

    // Every turn reached the server exactly once, in the order it was said.
    const delivered = [...ws.turnLocals(), ...ws2.turnLocals()];
    expect(delivered.map((t) => t.text)).toEqual([
      "you never call me",
      "I am busy at work",
      "please stop shouting at me",
    ]);
    const uids = delivered.map((t) => t.turn_uid);
    expect(new Set(uids).size).toBe(uids.length);
    expect(uids[0]).toBe(firstUid);
    // …and the on-screen transcript is the same three turns, un-duplicated.
    expect(hook.result.current.transcript.map((t) => t.text)).toEqual(
      delivered.map((t) => t.text),
    );

    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[1].emitServer({ type: "session_complete" }));
  }, 20000);

  it("keeps the turn ids stable so a re-sent turn can be recognised", async () => {
    const { fake, ws } = await startLiveSession();
    await speak(fake, "one");
    const sent = ws.turnLocals()[0];
    // The session record and the wire carry the SAME id — which is what lets
    // the server ignore a copy that was in flight when the socket died.
    expect(sent.turn_uid).toMatch(/^[A-Za-z0-9._:-]{1,64}$/);
    expect(sent.session_id).toBe("live-resume");
  }, 20000);

  it("drops the oldest turns when the outage outlasts the queue, and says so", async () => {
    const { fake, hook, ws } = await startLiveSession();
    await act(() => ws.drop());
    // 52 turns with a 50-deep queue: the two oldest fall off the bottom.
    for (let i = 0; i < 52; i++) await speak(fake, `turn ${i}`);
    await act(async () => {
      await wait(RECONNECT_DELAY_MS + 200);
    });
    const ws2 = FakeWebSocket.instances[1];
    await act(() => ws2.emitOpen());
    await act(async () => {
      await flush();
    });
    const flushed = ws2.turnLocals();
    expect(flushed).toHaveLength(50);
    expect(flushed[0].text).toBe("turn 2");        // 0 and 1 were dropped
    expect(flushed[49].text).toBe("turn 51");
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("dropped 2 oldest turn(s)"));
    // The dropped turns are still on the session record — the gap is
    // reported, never silently passed off as silence.
    expect(hook.result.current.transcript).toHaveLength(52);
  }, 60000);

  it("resumes at the merged seq it has, and renders the replayed turns", async () => {
    // Server-transcript client (no on-device loop), so the merged turns the
    // server pushes are the ones on screen — the seq bookkeeping and the
    // resume frame are the same either way.
    const { hook, ws } = await startLiveSession({ live: false });
    await act(() =>
      ws.emitServer({
        type: "transcript", call_id: "c-1", speaker: "Speaker B", display_name: "Mom",
        text: "I called twice", start_time: 0, end_time: 1, seq: 4, participant_uid: "u-mom",
      }),
    );

    await act(() => ws.drop());
    await act(async () => {
      await wait(RECONNECT_DELAY_MS + 200);
    });
    const ws2 = FakeWebSocket.instances[1];
    await act(() => ws2.emitOpen());
    const resume = ws2.sentJson().find((m) => m.type === "resume");
    expect(resume.since_seq).toBe(4);       // exactly what we had rendered

    // The server replays what we missed; a replayed turn renders like any
    // other, and moves the high-water mark on.
    await act(() =>
      ws2.emitServer({ type: "resume_replay", call_id: "c-1", since_seq: 4, replayed: 1, dropped: 0 }),
    );
    await act(() =>
      ws2.emitServer({
        type: "transcript", call_id: "c-1", speaker: "Speaker B", display_name: "Mom",
        text: "and you did not pick up", start_time: 2, end_time: 3, seq: 5,
        participant_uid: "u-mom", replay: true,
      }),
    );
    expect(hook.result.current.transcript.map((t) => t.text)).toEqual([
      "I called twice",
      "and you did not pick up",
    ]);
    // A second drop resumes from the replayed turn, not the pre-drop one.
    await act(() => ws2.drop());
    await act(async () => {
      await wait(RECONNECT_DELAY_MS + 200);
    });
    const ws3 = FakeWebSocket.instances[2];
    await act(() => ws3.emitOpen());
    expect(ws3.sentJson().find((m) => m.type === "resume").since_seq).toBe(5);
  }, 30000);

  it("does not send a resume frame on the first socket of a session", async () => {
    const { ws } = await startLiveSession();
    expect(ws.sentJson().some((m) => m.type === "resume")).toBe(false);
  }, 20000);
});
