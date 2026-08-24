import { renderHook, act } from "@testing-library/react-native";

/**
 * useAudioStream's two-sided additions: the pre-flight probe, the
 * end-of-session summary (from the transcript + latency log + nudges), the
 * server's record of the session (episode id + who it was auto-shared
 * with), and the optimistic episode handed to Your Day.
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
import { cloudProvider, ProviderChain, parseSuggestionJson } from "../src/live/localLlm";
import type { FastLoopHandlers, FastLoopCapabilities } from "../src/live/defaultDeps";
import type { LiveSessionBody } from "../src/api/liveSessions";
import { useLiveEpisodeStore } from "../src/store/liveEpisodeStore";
import { silenceInt16, toneInt16 } from "../src/live/testing/synth";

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
}

const GOOD = '{"suggestion":"Say: I miss our calls too, Mom.","tone":{"warmth":40,"label":"hurt"}}';

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
          { name: "os", isAvailable: async () => true, suggest: async () => parseSuggestionJson(GOOD) },
          cloudProvider(),
        ]),
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
  useLiveEpisodeStore.getState().clear();
  logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
});

afterEach(() => {
  logSpy.mockRestore();
});

describe("useAudioStream — preflight + session summary", () => {
  it("runPreflight probes the capabilities (and is a no-op when the device can't run the loop)", async () => {
    const caps: FastLoopCapabilities = {
      vad: "silero",
      speakerId: { active: true, reason: "model cached", enrolled: 2, model: "cached", droppedForModel: 0 },
      llm: ["os", "cloud"],
    };
    const probe = jest.fn().mockResolvedValue(caps);
    const hook = await renderHook(() =>
      useAudioStream({ capability: { capable: true, reason: "ok" }, probeCapabilities: probe }),
    );
    expect(hook.result.current.preflight).toBeNull();
    await act(async () => {
      await hook.result.current.runPreflight();
    });
    expect(probe).toHaveBeenCalledTimes(1);
    expect(hook.result.current.preflight).toEqual({ status: "ready", capabilities: caps });

    const failing = jest.fn().mockRejectedValue(new Error("ONNX session failed"));
    const hook2 = await renderHook(() =>
      useAudioStream({ capability: { capable: true, reason: "ok" }, probeCapabilities: failing }),
    );
    await act(async () => {
      await hook2.result.current.runPreflight();
    });
    expect(hook2.result.current.preflight).toEqual({ status: "failed", reason: "ONNX session failed" });

    const never = jest.fn();
    const hook3 = await renderHook(() =>
      useAudioStream({ capability: { capable: false, reason: "no STT" }, probeCapabilities: never }),
    );
    await act(async () => {
      await hook3.result.current.runPreflight();
    });
    expect(never).not.toHaveBeenCalled();
    expect(hook3.result.current.preflight).toBeNull();
  });

  it("after a live session: summary from the transcript + latency log, the episode record, and Your Day's optimistic row", async () => {
    const fake = makeFakeFastLoop();
    const posted: LiveSessionBody[] = [];
    const hook = await renderHook(() =>
      useAudioStream({
        capability: { capable: true, reason: "ok" },
        makeFastLoop: fake.make,
        postSession: async (body) => {
          posted.push(body);
          return { status: "created" as const, episodeId: "ep-1", sharedWith: ["mom@example.com"] };
        },
      }),
    );
    expect(hook.result.current.sessionSummary).toBeNull();
    expect(hook.result.current.lastEpisode).toBeNull();

    await act(async () => {
      hook.result.current.setSessionMode("speaker");
      await hook.result.current.startSession("live-1", 50);
    });
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    await act(async () => {
      await flush();
    });

    // One other-person turn: 1.2 s of tone, then silence to close it.
    await act(async () => {
      feed(toneInt16(220, 1.2, -20));
      fake.rec.emit({ text: "You never call me back.", isFinal: true });
      feed(silenceInt16(0.5));
      await fake.loop!.settle();
      await flush();
    });
    expect(hook.result.current.transcript.length).toBeGreaterThanOrEqual(1);

    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));

    expect(posted).toHaveLength(1);
    expect(hook.result.current.lastEpisode).toEqual({
      episodeId: "ep-1",
      postStatus: "created",
      sharedWith: ["mom@example.com"],
    });
    const summary = hook.result.current.sessionSummary;
    expect(summary).not.toBeNull();
    expect(summary!.totalTurns).toBeGreaterThanOrEqual(1);
    expect(summary!.durationMs).not.toBeNull();
    expect(summary!.turnsBySpeaker[0].turns).toBeGreaterThanOrEqual(1);
    // Spoken suggestion → a measured first-words latency.
    expect(summary!.spokenTurns).toBeGreaterThanOrEqual(1);
    expect(summary!.firstWordsMedianMs).not.toBeNull();
    expect(summary!.topProvider).toBe("os");

    const recent = useLiveEpisodeStore.getState().recent;
    expect(recent).toHaveLength(1);
    expect(recent[0]).toMatchObject({ episodeId: "ep-1", sessionId: "live-1", mode: "speaker", sharedWith: ["mom@example.com"] });

    // A new session clears the previous card.
    await act(async () => {
      await hook.result.current.startSession("live-2", 50);
    });
    expect(hook.result.current.sessionSummary).toBeNull();
    expect(hook.result.current.lastEpisode).toBeNull();
    expect(hook.result.current.escalationCount).toBe(0);
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[1].emitServer({ type: "session_complete" }));
  });

  it("legacy path: a summary still lands (no latency), the server's nudges count as escalations, nothing is POSTed", async () => {
    const postSession = jest.fn();
    const hook = await renderHook(() =>
      useAudioStream({ capability: { capable: false, reason: "no STT" }, postSession }),
    );
    await act(async () => {
      await hook.result.current.startSession("legacy-1", 50);
    });
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    await act(() =>
      ws.emitServer({ type: "transcript", speaker: "Speaker A", text: "I said no!", start_time: 0, end_time: 1 }),
    );
    await act(() =>
      ws.emitServer({ type: "suggestion", speaker: "Speaker A", suggestions: ["ease up"], kind: "nudge", empathy_slider: 50 }),
    );
    expect(hook.result.current.escalationCount).toBe(1);
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));
    expect(postSession).not.toHaveBeenCalled();
    expect(hook.result.current.lastEpisode).toBeNull();
    const s = hook.result.current.sessionSummary!;
    expect(s.escalations).toBe(1);
    expect(s.turnsBySpeaker).toEqual([{ speaker: "Speaker A", turns: 1 }]);
    expect(s.firstWordsMedianMs).toBeNull();
    expect(useLiveEpisodeStore.getState().recent).toHaveLength(0);
  });
});
