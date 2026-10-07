import { renderHook, act } from "@testing-library/react-native";

/**
 * Live Coach safety contract after the 2026-10-07 dinner test (Pixel 10,
 * 1.18.0 preview, earpiece mode, dx-NCRN-SAQE):
 *
 *   A. STOP means stop — nothing is spoken once Stop is pressed, including
 *      a cloud suggestion that lands while the session is still winding
 *      down and the loop's own last in-flight turn.
 *
 * Same seams as useAudioStreamLive.test.tsx: the real FastLoop with an
 * energy VAD, a fake recognizer and a fake LLM chain, a fake WebSocket, and
 * the expo-audio mic mock fed synthetic PCM.
 */
const mockMic = {
  onBuffer: null as
    | ((buffer: { data: ArrayBuffer; sampleRate: number; channels: number; timestamp: number }) => void)
    | null,
  start: jest.fn<Promise<void>, []>(),
  stop: jest.fn(),
};

jest.mock("expo-audio", () => ({
  __esModule: true,
  requestRecordingPermissionsAsync: async () => ({ status: "granted", granted: true }),
  setAudioModeAsync: async () => undefined,
  useAudioStream: (options?: { onBuffer?: (buffer: never) => void }) => {
    mockMic.onBuffer = (options?.onBuffer ?? null) as typeof mockMic.onBuffer;
    return {
      stream: { id: "mock-stream", sampleRate: 16000, channels: 1, isStreaming: false, start: mockMic.start, stop: mockMic.stop },
      isStreaming: false,
    };
  },
}));

import * as Speech from "expo-speech";
import { useAudioStream, type UseAudioStreamOptions } from "../src/hooks/useAudioStream";
import { FastLoop } from "../src/live/fastLoop";
import { EnergyVad } from "../src/live/vad";
import { FakeSpeechRecognizer } from "../src/live/stt";
import { cloudProvider, ProviderChain, parseSuggestionJson } from "../src/live/localLlm";
import type { FastLoopHandlers } from "../src/live/defaultDeps";
import type { LiveSessionBody, PostLiveSessionResult } from "../src/api/liveSessions";
import { silenceInt16, toneInt16 } from "../src/live/testing/synth";

const speakMock = Speech.speak as jest.Mock;
const speechStopMock = Speech.stop as jest.Mock;

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

const GOOD = '{"suggestion":"Say: I hear you.","tone":{"warmth":40,"label":"hurt"}}';

function makeFakeFastLoop(opts: { provider?: "ok" | "cloud"; llmDelayMs?: number } = {}) {
  const rec = new FakeSpeechRecognizer();
  const build = {
    rec,
    loop: null as FastLoop | null,
    async make(handlers: FastLoopHandlers) {
      const ok = {
        name: "os",
        isAvailable: async () => true,
        suggest: async () => {
          if (opts.llmDelayMs) await new Promise((r) => setTimeout(r, opts.llmDelayMs));
          return parseSuggestionJson(GOOD);
        },
      };
      const providers = opts.provider === "cloud" ? [cloudProvider()] : [ok, cloudProvider()];
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

const flush = (ms = 30) => new Promise((r) => setTimeout(r, ms));

function cloudSuggestion(text: string, utterance: string | null = null) {
  return {
    type: "suggestion",
    session_id: "s",
    speaker: "Speaker A",
    ...(utterance ? { utterance_text: utterance } : {}),
    suggestions: [text],
    empathy_slider: 50,
    suggestion_source: "cloud",
  };
}

async function startEarpieceSession(options: UseAudioStreamOptions, sessionId = "safety-1") {
  const hook = await renderHook(() => useAudioStream({ capability: { capable: true, reason: "ok" }, ...options }));
  await act(() => {
    hook.result.current.setSpeechEnabled(true);
    hook.result.current.setSessionMode("earpiece");
  });
  await act(async () => {
    await hook.result.current.startSession(sessionId, 50);
  });
  const ws = FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
  await act(() => ws.emitOpen());
  await act(async () => {
    await flush();
  });
  return { hook, ws };
}

let logSpy: jest.SpyInstance;
let warnSpy: jest.SpyInstance;

beforeEach(() => {
  FakeWebSocket.instances = [];
  (globalThis as Record<string, unknown>).WebSocket = FakeWebSocket;
  mockMic.onBuffer = null;
  mockMic.start.mockReset().mockResolvedValue(undefined);
  mockMic.stop.mockReset();
  speakMock.mockReset();
  speechStopMock.mockClear();
  logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
  warnSpy = jest.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  logSpy.mockRestore();
  warnSpy.mockRestore();
});

describe("A. STOP means stop", () => {
  it("a cloud suggestion arriving after Stop — while the session is still winding down — is never spoken", async () => {
    const fake = makeFakeFastLoop();
    // The session POST hangs: Stop is pressed but the wind-down (loop drain
    // + POST) has not finished. This is the window the dinner test hit.
    let releasePost: (r: PostLiveSessionResult) => void = () => {};
    const posted: LiveSessionBody[] = [];
    const { hook, ws } = await startEarpieceSession({
      makeFastLoop: fake.make,
      postSession: (body) => {
        posted.push(body);
        return new Promise((r) => {
          releasePost = r;
        });
      },
    });

    // Sanity: before Stop the loop's own suggestion IS voiced.
    await act(async () => {
      feed(toneInt16(1.0, -20));
      fake.rec.emit({ text: "you never listen", isFinal: true });
      feed(silenceInt16(0.5));
      await fake.loop!.settle();
      await flush();
    });
    expect(speakMock).toHaveBeenCalledTimes(1);
    speakMock.mockClear();
    speechStopMock.mockClear();

    let stopping: Promise<void> = Promise.resolve();
    await act(async () => {
      stopping = hook.result.current.stopSession();
      await flush();
    });
    // Speech was cut immediately, before the wind-down finished.
    expect(speechStopMock).toHaveBeenCalled();
    expect(posted).toHaveLength(1); // POST in flight, not resolved

    await act(async () => {
      ws.emitServer(cloudSuggestion("Late cloud line.", "something new"));
      ws.emitServer(cloudSuggestion("Another late line."));
      await flush();
    });
    expect(speakMock).not.toHaveBeenCalled();

    await act(async () => {
      releasePost({ status: "unsupported" });
      await stopping;
    });
    // …and during the drain window afterwards, still nothing.
    await act(async () => {
      ws.emitServer(cloudSuggestion("Drain-window line."));
      await flush();
    });
    expect(speakMock).not.toHaveBeenCalled();
    await act(() => ws.emitServer({ type: "session_complete" }));
    expect(speakMock).not.toHaveBeenCalled();
  });

  it("the loop's own last in-flight turn finishing during Stop is never spoken", async () => {
    const fake = makeFakeFastLoop({ llmDelayMs: 60 });
    const { hook } = await startEarpieceSession({
      makeFastLoop: fake.make,
      postSession: async () => ({ status: "unsupported" as const }),
    });
    await act(async () => {
      feed(toneInt16(1.0, -20));
      fake.rec.emit({ text: "fine, whatever", isFinal: true });
      feed(silenceInt16(0.5));
      // Do NOT settle: the turn's LLM call is still running when Stop lands.
    });
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(async () => {
      await flush(120);
    });
    expect(speakMock).not.toHaveBeenCalled();
  });
});
