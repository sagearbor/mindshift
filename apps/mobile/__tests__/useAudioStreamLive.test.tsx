import { renderHook, act } from "@testing-library/react-native";

/**
 * useAudioStream in ON-DEVICE live mode (Track 3): the real FastLoop
 * orchestrator wired through the hook's `makeFastLoop` seam with an energy
 * VAD, a fake recognizer and a fake LLM provider, driven by synthetic PCM
 * through the same expo-audio mock the legacy tests use.
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

import * as Speech from "expo-speech";
import { cloudAnswersOpenMoment, useAudioStream } from "../src/hooks/useAudioStream";
import { FastLoop } from "../src/live/fastLoop";
import { EnergyVad } from "../src/live/vad";
import { FakeSpeechRecognizer } from "../src/live/stt";
import { cloudProvider, ProviderChain, parseSuggestionJson } from "../src/live/localLlm";
import { SpeakerLabeler } from "../src/live/speakerId";
import type { FastLoopHandlers } from "../src/live/defaultDeps";
import type { LiveSessionBody } from "../src/api/liveSessions";
import { silenceInt16, toneInt16, unitVector } from "../src/live/testing/synth";

const speakMock = Speech.speak as jest.Mock;

class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];
  url: string;
  readyState = FakeWebSocket.OPEN;
  sent: string[] = [];
  sentBinary: ArrayBuffer[] = [];
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
    else if (ArrayBuffer.isView(data)) this.sentBinary.push(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer);
    else this.sentBinary.push(data);
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

const GOOD = '{"suggestion":"Say: I miss our calls too, Mom.","tone":{"warmth":40,"label":"hurt"}}';

function makeFakeFastLoop(opts: { provider?: "ok" | "cloud" } = {}) {
  const rec = new FakeSpeechRecognizer();
  const build = {
    rec,
    loop: null as FastLoop | null,
    calls: 0,
    async make(handlers: FastLoopHandlers) {
      build.calls += 1;
      const providers =
        opts.provider === "cloud"
          ? [cloudProvider()]
          : [{ name: "os", isAvailable: async () => true, suggest: async () => parseSuggestionJson(GOOD) }, cloudProvider()];
      build.loop = new FastLoop({
        ...handlers,
        vad: new EnergyVad(-45, 0.032),
        embedder: null,
        labeler: null,
        recognizer: rec,
        llm: new ProviderChain(providers),
        sttGraceMs: 100,
        pollMs: 5,
      });
      return {
        loop: build.loop,
        status: "energy VAD · speaker-ID off · LLM os → cloud",
        capabilities: {
          vad: "energy" as const,
          speakerId: { active: false, reason: "test", enrolled: 0, model: null, droppedForModel: 0 },
          llm: ["os", "cloud"],
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

/** The latency report is console.log'd at session end; keep test output
 *  quiet without jest.restoreAllMocks (which, on Jest 29, also wipes the
 *  jest.fn() implementations in jest-setup's module mocks). */
let logSpy: jest.SpyInstance;

beforeEach(() => {
  FakeWebSocket.instances = [];
  (globalThis as Record<string, unknown>).WebSocket = FakeWebSocket;
  mockMic.onBuffer = null;
  mockMic.start.mockReset().mockResolvedValue(undefined);
  mockMic.stop.mockReset();
  mockMic.requestPermissions.mockReset().mockResolvedValue({ status: "granted", granted: true });
  mockMic.setAudioMode.mockReset().mockResolvedValue(undefined);
  speakMock.mockReset();
  logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
});

afterEach(() => {
  logSpy.mockRestore();
});

describe("cloudAnswersOpenMoment", () => {
  const answered = { text: "you never call me", hadSuggestion: true };
  const open = { text: "well I am busy", hadSuggestion: false };
  it("voices only the answer to the latest, still-open turn", () => {
    expect(cloudAnswersOpenMoment([answered, open], "well I am busy")).toBe(true);
    expect(cloudAnswersOpenMoment([answered, open], "you never call me")).toBe(false); // earlier turn
    expect(cloudAnswersOpenMoment([open, answered], "well I am busy")).toBe(false); // earlier turn
    expect(cloudAnswersOpenMoment([answered], "you never call me")).toBe(false); // already answered locally
  });
  it("treats words the phone never reported (or no text) as the only voice there is", () => {
    expect(cloudAnswersOpenMoment([], "anything")).toBe(true);
    expect(cloudAnswersOpenMoment([], null)).toBe(true);
    expect(cloudAnswersOpenMoment([answered, open], "something the VAD missed")).toBe(true);
    expect(cloudAnswersOpenMoment([open, answered], "something the VAD missed")).toBe(false);
    expect(cloudAnswersOpenMoment([open], null)).toBe(true);
    expect(cloudAnswersOpenMoment([answered], null)).toBe(false);
  });
});

describe("useAudioStream live mode", () => {
  it("is off (legacy path, no tts config) when the device isn't capable", async () => {
    const fake = makeFakeFastLoop();
    const hook = await renderHook(() =>
      useAudioStream({ capability: { capable: false, reason: "no on-device STT" }, makeFastLoop: fake.make }),
    );
    expect(hook.result.current.liveCapable).toBe(false);
    expect(hook.result.current.liveMode).toBe(false);
    expect(hook.result.current.liveCapabilityReason).toBe("no on-device STT");
    await act(async () => {
      await hook.result.current.startSession("live-1", 50);
    });
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    expect(fake.calls).toBe(0);
    expect(ws.sentJson()[0].tts).toBeUndefined();
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));
  });

  it("runs the fast loop: local turns fill the transcript + feed, speak, send turn_local, then post the session", async () => {
    const fake = makeFakeFastLoop();
    const posted: LiveSessionBody[] = [];
    const hook = await renderHook(() =>
      useAudioStream({
        capability: { capable: true, reason: "ok" },
        makeFastLoop: fake.make,
        postSession: async (body) => {
          posted.push(body);
          return { status: "unsupported" as const };
        },
      }),
    );
    expect(hook.result.current.liveCapable).toBe(true);
    expect(hook.result.current.liveMode).toBe(true);
    await act(() => {
      hook.result.current.setSpeechEnabled(true);
      hook.result.current.setSessionMode("speaker");
    });

    await act(async () => {
      await hook.result.current.startSession("live-2", 70);
    });
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    await act(async () => {
      await flush();
    });
    expect(fake.calls).toBe(1);
    expect(fake.rec.started).toBe(true);
    expect(hook.result.current.liveStatus).toMatch(/^On-device: energy VAD/);
    // The server is told the phone does its own TTS.
    expect(ws.sentJson().some((m) => m.type === "config" && m.tts === "on-device")).toBe(true);

    // 1 s of "Mom" talking, the recognizer's words, then silence => a turn.
    await act(async () => {
      feed(toneInt16(1.0, -20));
      fake.rec.emit({ text: "you never call me", isFinal: true });
      feed(silenceInt16(0.5));
      await fake.loop!.settle();
    });
    await act(async () => {
      await flush();
    });

    expect(hook.result.current.transcript).toHaveLength(1);
    expect(hook.result.current.transcript[0]).toMatchObject({ speaker: "Unknown", text: "you never call me" });
    expect(hook.result.current.transcript[0].startTime).toBeCloseTo(0, 1);
    expect(hook.result.current.suggestions[0]).toMatchObject({
      kind: "response",
      texts: ["Say: I miss our calls too, Mom."],
      source: "on-device",
      muted: false,
    });
    expect(speakMock).toHaveBeenCalledWith("Say: I miss our calls too, Mom.", expect.anything());
    // PCM still streams to the server exactly as before …
    expect(ws.sentBinary.length).toBeGreaterThan(0);
    // … plus one turn_local per finalized turn.
    const turnLocal = ws.sentJson().filter((m) => m.type === "turn_local");
    expect(turnLocal).toHaveLength(1);
    expect(turnLocal[0]).toMatchObject({
      session_id: "live-2",
      text: "you never call me",
      transcript_source: "on-device",
      suggestion_source: "on-device",
      tts_source: "on-device",
    });

    // The server's transcript event is NOT rendered while the phone owns the
    // transcript, but its cloud suggestion augments the feed unspoken (the
    // phone already answered this turn).
    speakMock.mockClear();
    await act(() => {
      ws.emitServer({ type: "transcript", speaker: "Speaker A", text: "duplicate", start_time: 0, end_time: 1 });
      ws.emitServer({
        type: "suggestion",
        session_id: "live-2",
        speaker: "Speaker A",
        utterance_text: "you never call me",
        suggestions: ["Cloud: tell her you love her."],
        empathy_slider: 70,
        suggestion_source: "cloud",
      });
    });
    expect(hook.result.current.transcript).toHaveLength(1);
    expect(hook.result.current.suggestions[0]).toMatchObject({ source: "cloud", texts: ["Cloud: tell her you love her."] });
    expect(speakMock).not.toHaveBeenCalled();

    // Identity + tone events render additively.
    await act(() => {
      ws.emitServer({ type: "speaker_identity", session_id: "live-2", speaker: "Unknown", person_id: "p-mom", display_name: "Mom", is_self: false, score: 0.8 });
      ws.emitServer({ type: "tone_flag", session_id: "live-2", speaker: "Mom", start_time: 0, end_time: 1, source: "text", scores: { sadness: 70 }, label: "hurt", confidence: 0.7 });
    });
    expect(hook.result.current.transcript[0].speaker).toBe("Mom");
    // The header label follows the LAST event's speaker (the cloud suggestion
    // above said "Speaker A"); an identity for "Unknown" doesn't touch it.
    expect(hook.result.current.speakerLabel).toBe("Speaker A");
    expect(hook.result.current.toneFlags[0]).toMatchObject({ label: "hurt", source: "text" });

    // A server-computed nudge (call mode's sustained talking-over) renders
    // like an on-device one: flash + escalation count.
    await act(() => {
      ws.emitServer({ type: "nudge", channel: "A", level: 1, t: 47, vectors: ["interrupting"], detail: "self talked over the other party for 3.0s" });
    });
    expect(hook.result.current.nudgeFlash).toMatchObject({ channel: "A", level: 1, vectors: ["interrupting"] });
    expect(hook.result.current.escalationCount).toBe(1);

    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));
    expect(fake.rec.stopped).toBe(true);
    expect(hook.result.current.latencySummary).toMatch(/1 turns/);
    expect(posted).toHaveLength(1);
    expect(posted[0]).toMatchObject({ session_id: "live-2", mode: "speaker" });
    expect(posted[0].turns).toHaveLength(1);
    expect(posted[0].speaker_identities).toHaveLength(1);
    expect(posted[0].tone_flags).toHaveLength(1);
    expect(posted[0].started_at <= posted[0].ended_at).toBe(true);
    // The stop handshake still happens after the loop drained.
    expect(ws.sentJson().some((m) => m.type === "stop")).toBe(true);
  });

  it("renders a streaming partial preview dimmed and replaces it with the final", async () => {
    const fake = makeFakeFastLoop({ provider: "cloud" });
    const hook = await renderHook(() =>
      useAudioStream({ capability: { capable: true, reason: "ok" }, makeFastLoop: fake.make, postSession: async () => ({ status: "unsupported" as const }) }),
    );
    await act(() => hook.result.current.setSpeechEnabled(true));
    await act(async () => {
      await hook.result.current.startSession("live-p", 50);
    });
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    // Local-first config asks the server for its latency report too.
    expect(ws.sentJson().some((m) => m.type === "config" && m.report_latency === true)).toBe(true);
    await act(() => {
      ws.emitServer({ type: "suggestion", session_id: "live-p", speaker: "Speaker A", utterance_text: "hello", suggestions: ["Prev"], empathy_slider: 50, speak: false, partial: true });
    });
    expect(hook.result.current.suggestions).toHaveLength(1);
    expect(hook.result.current.suggestions[0]).toMatchObject({ partial: true, muted: true, texts: ["Prev"] });
    expect(speakMock).not.toHaveBeenCalled();
    await act(() => {
      ws.emitServer({ type: "suggestion", session_id: "live-p", speaker: "Speaker A", utterance_text: "hello", suggestions: ["Previewed final."], empathy_slider: 50 });
    });
    expect(hook.result.current.suggestions).toHaveLength(1);
    expect(hook.result.current.suggestions[0]).toMatchObject({ texts: ["Previewed final."], muted: false });
    expect(hook.result.current.suggestions[0].partial).toBeUndefined();
    expect(speakMock).toHaveBeenCalledWith("Previewed final.", expect.anything());
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => ws.emitServer({ type: "session_complete", latency_summary: { turns: 1 } }));
  });

  it("voices the cloud suggestion when the phone's providers fell through", async () => {
    const fake = makeFakeFastLoop({ provider: "cloud" });
    const hook = await renderHook(() =>
      useAudioStream({ capability: { capable: true, reason: "ok" }, makeFastLoop: fake.make, postSession: async () => ({ status: "unsupported" as const }) }),
    );
    await act(() => hook.result.current.setSpeechEnabled(true));
    await act(async () => {
      await hook.result.current.startSession("live-3", 50);
    });
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    await act(async () => {
      feed(toneInt16(1.0, -20));
      fake.rec.emit({ text: "hello", isFinal: true });
      feed(silenceInt16(0.5));
      await fake.loop!.settle();
      await flush();
    });
    expect(hook.result.current.suggestions).toHaveLength(0);
    await act(() => {
      ws.emitServer({ type: "suggestion", session_id: "live-3", speaker: "Speaker A", utterance_text: "hello", suggestions: ["Cloud answer."], empathy_slider: 50 });
    });
    expect(speakMock).toHaveBeenCalledWith("Cloud answer.", expect.anything());
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));
  });

  it("switching live mode off restores the legacy path", async () => {
    const fake = makeFakeFastLoop();
    const hook = await renderHook(() =>
      useAudioStream({ capability: { capable: true, reason: "ok" }, makeFastLoop: fake.make }),
    );
    await act(() => hook.result.current.setLiveMode(false));
    expect(hook.result.current.liveMode).toBe(false);
    await act(async () => {
      await hook.result.current.startSession("live-4", 50);
    });
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    expect(fake.calls).toBe(0);
    await act(() => {
      ws.emitServer({ type: "transcript", speaker: "Speaker A", text: "server words", start_time: 0, end_time: 1 });
    });
    expect(hook.result.current.transcript[0].text).toBe("server words");
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));
  });

  it("a fast loop that fails to build leaves the server path running and says why", async () => {
    const hook = await renderHook(() =>
      useAudioStream({
        capability: { capable: true, reason: "ok" },
        makeFastLoop: async () => {
          throw new Error("models missing");
        },
      }),
    );
    await act(async () => {
      await hook.result.current.startSession("live-5", 50);
    });
    expect(hook.result.current.sessionActive).toBe(true);
    expect(hook.result.current.liveStatus).toMatch(/unavailable \(models missing\)/);
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    await act(() => {
      ws.emitServer({ type: "transcript", speaker: "Speaker A", text: "still works", start_time: 0, end_time: 1 });
    });
    expect(hook.result.current.transcript[0].text).toBe("still works");
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));
  });

  it("clears the nudge flash on request and reports STT failure honestly", async () => {
    const fake = makeFakeFastLoop();
    const hook = await renderHook(() =>
      useAudioStream({ capability: { capable: true, reason: "ok" }, makeFastLoop: fake.make, postSession: async () => ({ status: "unsupported" as const }) }),
    );
    await act(async () => {
      await hook.result.current.startSession("live-6", 50);
    });
    await act(async () => {
      await flush();
    });
    await act(() => fake.rec.emitError("service-not-allowed", "gone"));
    expect(hook.result.current.liveStatus).toMatch(/speech recognition failed \(service-not-allowed: gone\)/);
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    // With on-device STT gone, the server's transcript is accepted again.
    await act(() => {
      ws.emitServer({ type: "transcript", speaker: "Speaker A", text: "server fallback", start_time: 0, end_time: 1 });
    });
    expect(hook.result.current.transcript[0].text).toBe("server fallback");
    await act(() => hook.result.current.clearNudgeFlash());
    expect(hook.result.current.nudgeFlash).toBeNull();
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));
  });

  it("a recognizer that fails to START is reported in the status line and the server transcript is accepted", async () => {
    const fake = makeFakeFastLoop();
    fake.rec.start = async () => {
      throw new Error("speech recognition permission denied");
    };
    const hook = await renderHook(() =>
      useAudioStream({ capability: { capable: true, reason: "ok" }, makeFastLoop: fake.make, postSession: async () => ({ status: "unsupported" as const }) }),
    );
    await act(async () => {
      await hook.result.current.startSession("live-7", 50);
    });
    await act(async () => {
      await flush();
    });
    // Not "On-device: energy VAD · …" as if everything worked.
    expect(hook.result.current.liveStatus).toMatch(/^On-device: energy VAD/);
    expect(hook.result.current.liveStatus).toMatch(/speech recognition failed \(start-failed: speech recognition permission denied\) — transcript from the server/);
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    await act(() => {
      ws.emitServer({ type: "transcript", speaker: "Speaker A", text: "server words", start_time: 0, end_time: 1 });
    });
    expect(hook.result.current.transcript[0].text).toBe("server words");
    // A VAD turn with no words claims nothing from the server.
    await act(async () => {
      feed(toneInt16(1.0, -20));
      feed(silenceInt16(0.5));
      await fake.loop!.settle();
      await flush();
    });
    expect(ws.sentJson().filter((m) => m.type === "turn_local")).toHaveLength(0);
    // …and the server's suggestion for those words is voiced as on the legacy path.
    await act(() => hook.result.current.setSpeechEnabled(true));
    await act(() => {
      ws.emitServer({ type: "suggestion", session_id: "live-7", speaker: "Speaker A", utterance_text: "server words", suggestions: ["Cloud only."], empathy_slider: 50 });
    });
    expect(speakMock).toHaveBeenCalledWith("Cloud only.", expect.anything());
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));
  });

  it("a late cloud answer to an already-coached earlier turn is shown but not voiced; the answer to the latest unanswered turn is", async () => {
    // The local provider answers only turns that mention "call".
    const fake = makeFakeFastLoop();
    const originalMake = fake.make;
    fake.make = async (handlers: FastLoopHandlers) => {
      const build = await originalMake(handlers);
      const selective = {
        name: "os",
        isAvailable: async () => true,
        suggest: async (input: { text: string }) => (input.text.includes("call") ? parseSuggestionJson(GOOD) : null),
      };
      fake.loop = new FastLoop({
        ...handlers,
        vad: new EnergyVad(-45, 0.032),
        embedder: null,
        labeler: null,
        recognizer: fake.rec,
        llm: new ProviderChain([selective]),
        sttGraceMs: 100,
        pollMs: 5,
      });
      return { ...build, loop: fake.loop };
    };
    const hook = await renderHook(() =>
      useAudioStream({ capability: { capable: true, reason: "ok" }, makeFastLoop: fake.make, postSession: async () => ({ status: "unsupported" as const }) }),
    );
    await act(() => hook.result.current.setSpeechEnabled(true));
    await act(async () => {
      await hook.result.current.startSession("live-8", 50);
    });
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    const turn = async (text: string) => {
      await act(async () => {
        feed(toneInt16(1.0, -20));
        fake.rec.emit({ text, isFinal: true });
        feed(silenceInt16(0.5));
        await fake.loop!.settle();
        await flush();
      });
    };
    await turn("you never call me"); // answered locally (spoken)
    expect(speakMock).toHaveBeenCalledTimes(1);
    await turn("well I am busy"); // local providers passed: waiting on the cloud
    speakMock.mockClear();
    // The cloud's (slow) answer to turn 1 arrives now — turn 1 was already coached.
    await act(() => {
      ws.emitServer({ type: "suggestion", session_id: "live-8", speaker: "Unknown", utterance_text: "you never call me", suggestions: ["Cloud: late answer to turn 1."], empathy_slider: 50 });
    });
    expect(hook.result.current.suggestions[0].texts).toEqual(["Cloud: late answer to turn 1."]);
    expect(speakMock).not.toHaveBeenCalled();
    // The cloud's answer to turn 2 — the latest, unanswered turn — is voiced.
    await act(() => {
      ws.emitServer({ type: "suggestion", session_id: "live-8", speaker: "Unknown", utterance_text: "well I am busy", suggestions: ["Cloud: answer to turn 2."], empathy_slider: 50 });
    });
    expect(speakMock).toHaveBeenCalledWith("Cloud: answer to turn 2.", expect.anything());
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));
  });

  it("therapist mode never voices a cloud suggestion either (on-screen only)", async () => {
    const fake = makeFakeFastLoop({ provider: "cloud" });
    const hook = await renderHook(() =>
      useAudioStream({ capability: { capable: true, reason: "ok" }, makeFastLoop: fake.make, postSession: async () => ({ status: "unsupported" as const }) }),
    );
    await act(() => {
      hook.result.current.setSpeechEnabled(true);
      hook.result.current.setSessionMode("therapist");
    });
    await act(async () => {
      await hook.result.current.startSession("live-9", 50);
    });
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    await act(async () => {
      feed(toneInt16(1.0, -20));
      fake.rec.emit({ text: "hello", isFinal: true });
      feed(silenceInt16(0.5));
      await fake.loop!.settle();
      await flush();
    });
    await act(() => {
      ws.emitServer({ type: "suggestion", session_id: "live-9", speaker: "Speaker A", utterance_text: "hello", suggestions: ["Cloud answer."], empathy_slider: 50 });
    });
    expect(hook.result.current.suggestions[0]).toMatchObject({ source: "cloud", texts: ["Cloud answer."] });
    expect(speakMock).not.toHaveBeenCalled();
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));
  });

  it("a cloud suggestion arriving mid-speech is held until the VAD goes quiet (never over live speech)", async () => {
    const fake = makeFakeFastLoop({ provider: "cloud" });
    const hook = await renderHook(() =>
      useAudioStream({ capability: { capable: true, reason: "ok" }, makeFastLoop: fake.make, postSession: async () => ({ status: "unsupported" as const }) }),
    );
    await act(() => {
      hook.result.current.setSpeechEnabled(true);
      hook.result.current.setSessionMode("speaker");
    });
    await act(async () => {
      await hook.result.current.startSession("live-10", 50);
    });
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    await act(async () => {
      feed(toneInt16(1.0, -20));
      fake.rec.emit({ text: "hello", isFinal: true });
      feed(silenceInt16(0.5));
      await fake.loop!.settle();
      await flush();
    });
    // The other person starts talking again, THEN the cloud answers.
    await act(async () => {
      feed(toneInt16(0.6, -20));
      await fake.loop!.settle();
    });
    await act(() => {
      ws.emitServer({ type: "suggestion", session_id: "live-10", speaker: "Speaker A", utterance_text: "hello", suggestions: ["Cloud answer."], empathy_slider: 50 });
    });
    expect(speakMock).not.toHaveBeenCalled();
    await act(async () => {
      feed(silenceInt16(0.5));
      await fake.loop!.settle();
    });
    expect(speakMock).toHaveBeenCalledWith("Cloud answer.", expect.anything());
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));
  });

  it("the You: A ⇄ B chip reaches the on-device loop (nudge vs response flips)", async () => {
    const fake = makeFakeFastLoop();
    const originalMake = fake.make;
    fake.make = async (handlers: FastLoopHandlers) => {
      const build = await originalMake(handlers);
      const D = 192;
      const queue = [unitVector(D, 0), unitVector(D, 5, 0.05, 3)];
      fake.loop = new FastLoop({
        ...handlers,
        vad: new EnergyVad(-45, 0.032),
        embedder: { embed: async () => queue.shift() ?? unitVector(D, 9) },
        labeler: new SpeakerLabeler([]),
        recognizer: fake.rec,
        llm: new ProviderChain([{ name: "os", isAvailable: async () => true, suggest: async () => parseSuggestionJson(GOOD) }]),
        sttGraceMs: 100,
        pollMs: 5,
      });
      return { ...build, loop: fake.loop };
    };
    const hook = await renderHook(() =>
      useAudioStream({ capability: { capable: true, reason: "ok" }, makeFastLoop: fake.make, postSession: async () => ({ status: "unsupported" as const }) }),
    );
    await act(async () => {
      await hook.result.current.startSession("live-11", 50);
    });
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    const turn = async (text: string) => {
      await act(async () => {
        // >= MIN_CLUSTER_SECONDS: long enough to found an unknown cluster.
        feed(toneInt16(2.0, -20));
        fake.rec.emit({ text, isFinal: true });
        feed(silenceInt16(0.5));
        await fake.loop!.settle();
        await flush();
      });
    };
    await turn("first voice"); // Speaker A = you by convention -> nudge
    expect(hook.result.current.suggestions[0].kind).toBe("nudge");
    await act(() => hook.result.current.setSelfSpeaker("Speaker B"));
    expect(ws.sentJson().some((m) => m.type === "config" && m.self_speaker === "Speaker B")).toBe(true);
    await turn("second voice"); // Speaker B is now you -> nudge (not a response)
    expect(hook.result.current.transcript[1].speaker).toBe("Speaker B");
    expect(hook.result.current.suggestions[0].kind).toBe("nudge");
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));
  });
});

// --- Kept session audio (live/liveAudioKeeper.ts + POST /sessions/{id}/audio) ---
import { createLiveAudioKeeper, type LiveAudioKeeperOpen } from "../src/live/liveAudioKeeper";
import { MemoryFs } from "../src/recorder/memoryFs";
import type { LiveSessionAudioResult } from "../src/api/liveSessions";

const KEEP_DIR = "file:///cache/live-audio";

/** The hook's keeper seam over MemoryFs, flushing on every append. */
function keeperSeam() {
  const fs = new MemoryFs();
  const opened: string[] = [];
  const open = (sessionId: string): LiveAudioKeeperOpen => {
    opened.push(sessionId);
    return { keeper: createLiveAudioKeeper({ fs, dir: KEEP_DIR, sessionId, flushMs: 0 }), reason: null };
  };
  return { fs, opened, open };
}

const audioOk = (recordingId: string): LiveSessionAudioResult => ({
  recording_id: recordingId,
  media_type: "audio",
  duration_seconds: 1.5,
  size_bytes: 48044,
  stored_variants: ["wav"],
});

describe("useAudioStream keeps the session audio", () => {
  // 1.0 s of tone + 0.5 s of silence = 24 000 samples at 16 kHz.
  const SAMPLES = 24000;
  const EXPECTED_BYTES = 44 + SAMPLES * 2;

  async function runSession(
    hook: { result: { current: ReturnType<typeof useAudioStream> } },
    id: string,
    fake: ReturnType<typeof makeFakeFastLoop>,
  ) {
    await act(async () => {
      await hook.result.current.startSession(id, 50);
    });
    const ws = FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
    await act(() => ws.emitOpen());
    await act(async () => {
      await flush();
    });
    await act(async () => {
      feed(toneInt16(1.0, -20));
      fake.rec.emit({ text: "hello", isFinal: true });
      feed(silenceInt16(0.5));
      await fake.loop!.settle();
      await flush();
    });
    return ws;
  }

  it("by default keeps a WAV of the mic frames and attaches it to the saved episode, then deletes it", async () => {
    const fake = makeFakeFastLoop();
    const seam = keeperSeam();
    const uploads: { recordingId: string; uri: string; bytes: number; onDisk: number | null }[] = [];
    const hook = await renderHook(() =>
      useAudioStream({
        capability: { capable: true, reason: "ok" },
        makeFastLoop: fake.make,
        postSession: async () => ({ status: "created" as const, episodeId: "ep-1", sharedWith: [] }),
        openAudioKeeper: seam.open,
        postSessionAudio: async (recordingId, uri, bytes) => {
          uploads.push({ recordingId, uri, bytes, onDisk: seam.fs.sizeOf(uri) });
          return audioOk(recordingId);
        },
      }),
    );
    expect(hook.result.current.audioKeep).toEqual({ status: "off" });
    const ws = await runSession(hook, "live-keep", fake);
    expect(seam.opened).toEqual(["live-keep"]);
    expect(hook.result.current.audioKeep).toEqual({ status: "keeping" });
    const uri = `${KEEP_DIR}/live-keep.wav`;
    expect(seam.fs.sizeOf(uri)).toBe(EXPECTED_BYTES);

    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => ws.emitServer({ type: "session_complete" }));

    expect(hook.result.current.lastEpisode).toMatchObject({ episodeId: "ep-1", postStatus: "created" });
    expect(uploads).toEqual([{ recordingId: "ep-1", uri, bytes: EXPECTED_BYTES, onDisk: EXPECTED_BYTES }]);
    // The server has it: the local copy is gone.
    expect(seam.fs.exists(uri)).toBe(false);
    expect(hook.result.current.audioKeep).toEqual({ status: "kept", bytes: EXPECTED_BYTES, seconds: 1.5 });
  });

  it("keepAudio: false writes nothing and uploads nothing", async () => {
    const fake = makeFakeFastLoop();
    const seam = keeperSeam();
    const upload = jest.fn(async (id: string) => audioOk(id));
    const hook = await renderHook(() =>
      useAudioStream({
        capability: { capable: true, reason: "ok" },
        makeFastLoop: fake.make,
        keepAudio: false,
        postSession: async () => ({ status: "created" as const, episodeId: "ep-2", sharedWith: [] }),
        openAudioKeeper: seam.open,
        postSessionAudio: upload,
      }),
    );
    const ws = await runSession(hook, "live-nokeep", fake);
    expect(hook.result.current.audioKeep).toEqual({ status: "off" });
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => ws.emitServer({ type: "session_complete" }));
    expect(hook.result.current.lastEpisode).toMatchObject({ episodeId: "ep-2", postStatus: "created" });
    expect(seam.opened).toEqual([]);
    expect(seam.fs.listFileNames(KEEP_DIR)).toEqual([]);
    expect(upload).not.toHaveBeenCalled();
    expect(hook.result.current.audioKeep).toEqual({ status: "off" });
  });

  it("an upload failure leaves the session saved and the file on the phone, with a retry that works", async () => {
    const fake = makeFakeFastLoop();
    const seam = keeperSeam();
    const upload = jest
      .fn<Promise<LiveSessionAudioResult>, [string, string, number]>()
      .mockRejectedValueOnce(Object.assign(new Error("API error: 503"), { status: 503 }))
      .mockImplementation(async (id) => audioOk(id));
    const warn = jest.spyOn(console, "warn").mockImplementation(() => {});
    const hook = await renderHook(() =>
      useAudioStream({
        capability: { capable: true, reason: "ok" },
        makeFastLoop: fake.make,
        postSession: async () => ({ status: "created" as const, episodeId: "ep-3", sharedWith: [] }),
        openAudioKeeper: seam.open,
        postSessionAudio: upload,
      }),
    );
    const ws = await runSession(hook, "live-fail", fake);
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => ws.emitServer({ type: "session_complete" }));
    const uri = `${KEEP_DIR}/live-fail.wav`;
    expect(hook.result.current.lastEpisode).toMatchObject({ episodeId: "ep-3", postStatus: "created" });
    expect(upload).toHaveBeenCalledTimes(1);
    expect(hook.result.current.audioKeep).toMatchObject({
      status: "failed",
      bytes: EXPECTED_BYTES,
      seconds: 1.5,
      error: "API error: 503",
    });
    expect(typeof hook.result.current.audioKeep.retry).toBe("function");
    expect(seam.fs.exists(uri)).toBe(true);

    await act(async () => {
      await hook.result.current.audioKeep.retry!();
    });
    expect(upload).toHaveBeenCalledTimes(2);
    expect(upload.mock.calls[1]).toEqual(["ep-3", uri, EXPECTED_BYTES]);
    expect(hook.result.current.audioKeep).toEqual({ status: "kept", bytes: EXPECTED_BYTES, seconds: 1.5 });
    expect(seam.fs.exists(uri)).toBe(false);
    warn.mockRestore();
  });

  it("a session the server didn't store (unsupported / no episode id) discards the file and uploads nothing", async () => {
    const fake = makeFakeFastLoop();
    const seam = keeperSeam();
    const upload = jest.fn(async (id: string) => audioOk(id));
    const hook = await renderHook(() =>
      useAudioStream({
        capability: { capable: true, reason: "ok" },
        makeFastLoop: fake.make,
        postSession: async () => ({ status: "unsupported" as const }),
        openAudioKeeper: seam.open,
        postSessionAudio: upload,
      }),
    );
    const ws = await runSession(hook, "live-unsaved", fake);
    const uri = `${KEEP_DIR}/live-unsaved.wav`;
    expect(seam.fs.sizeOf(uri)).toBe(EXPECTED_BYTES);
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => ws.emitServer({ type: "session_complete" }));
    expect(upload).not.toHaveBeenCalled();
    expect(seam.fs.listFileNames(KEEP_DIR)).toEqual([]);
    expect(hook.result.current.audioKeep).toMatchObject({ status: "off", error: expect.stringMatching(/wasn't saved/) });
  });

  it("unmounting mid-session discards the file without uploading", async () => {
    const fake = makeFakeFastLoop();
    const seam = keeperSeam();
    const upload = jest.fn(async (id: string) => audioOk(id));
    const hook = await renderHook(() =>
      useAudioStream({
        capability: { capable: true, reason: "ok" },
        makeFastLoop: fake.make,
        postSession: async () => ({ status: "created" as const, episodeId: "ep-5", sharedWith: [] }),
        openAudioKeeper: seam.open,
        postSessionAudio: upload,
      }),
    );
    await runSession(hook, "live-unmount", fake);
    expect(seam.fs.exists(`${KEEP_DIR}/live-unmount.wav`)).toBe(true);
    await act(async () => {
      hook.unmount();
      await flush();
    });
    expect(seam.fs.listFileNames(KEEP_DIR)).toEqual([]);
    expect(upload).not.toHaveBeenCalled();
  });
});
