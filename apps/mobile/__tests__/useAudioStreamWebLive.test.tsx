import { renderHook, act } from "@testing-library/react-native";
import { Platform } from "react-native";

/**
 * useAudioStream on the WEB build with the on-device fast loop: the real
 * FastLoop (energy VAD, fake recognizer, fake provider) built through the
 * `makeFastLoop` seam, fed by the mocked WebAudioCapture. Proves the one
 * thing that matters for iOS Safari: the frames the worklet delivers reach
 * BOTH the WebSocket (as 16 kHz int16) AND the loop, the recognizer primed
 * inside the Start gesture is the one the loop runs, the loop's build
 * progress lands in liveStatus, and therapist mode never speaks — not even
 * for the cloud's suggestion event.
 */
const mockWeb = {
  onBuffer: null as ((b: unknown) => void) | null,
  onTrackEnded: null as ((r: "ended" | "muted") => void) | null,
  start: jest.fn<Promise<void>, []>(),
  stop: jest.fn<Promise<void>, []>(),
};

jest.mock("../src/utils/webAudioCapture", () => ({
  __esModule: true,
  WebCaptureError: class extends Error {},
  isWebAudioCaptureSupported: () => true,
  WebAudioCapture: class {
    constructor(opts: { onBuffer: (b: unknown) => void; onTrackEnded?: (r: "ended" | "muted") => void }) {
      mockWeb.onBuffer = opts.onBuffer;
      mockWeb.onTrackEnded = opts.onTrackEnded ?? null;
    }
    start() {
      return mockWeb.start();
    }
    stop() {
      return mockWeb.stop();
    }
  },
}));

const primedMock = { instance: null as FakeSpeechRecognizer | null, calls: 0 };
jest.mock("../src/live/webDeps", () => ({
  __esModule: true,
  createWebFastLoop: jest.fn(),
  primeWebRecognizer: () => {
    primedMock.calls += 1;
    return primedMock.instance;
  },
}));

const mockUnlock = jest.fn(() => true);
jest.mock("../src/utils/webSpeech", () => ({
  __esModule: true,
  unlockWebSpeechSynthesis: () => mockUnlock(),
  webSpeechSynthesisAvailable: () => true,
}));

import * as Speech from "expo-speech";
import { useAudioStream } from "../src/hooks/useAudioStream";
import { FastLoop } from "../src/live/fastLoop";
import { EnergyVad } from "../src/live/vad";
import { FakeSpeechRecognizer } from "../src/live/stt";
import { cloudProvider, ProviderChain, parseSuggestionJson } from "../src/live/localLlm";
import type { FastLoopHandlers } from "../src/live/defaultDeps";
import { silenceInt16, toneInt16 } from "../src/live/testing/synth";

const speakMock = Speech.speak as jest.Mock;

class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];
  readyState = FakeWebSocket.OPEN;
  sent: string[] = [];
  sentBinary: ArrayBuffer[] = [];
  onopen: ((e: unknown) => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;
  onclose: ((e: unknown) => void) | null = null;
  constructor(public url: string) {
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

const GOOD = '{"suggestion":"Say: I hear you.","tone":{"warmth":40,"label":"hurt"}}';

function fakeBuild(opts: { provider?: "ok" | "cloud"; status?: string[] } = {}) {
  const build = {
    loop: null as FastLoop | null,
    handlers: null as FastLoopHandlers | null,
    async make(handlers: FastLoopHandlers) {
      build.handlers = handlers;
      handlers.onStatus?.("Downloading voice model (one time) … 42 %");
      const providers =
        opts.provider === "cloud"
          ? [cloudProvider()]
          : [{ name: "os", isAvailable: async () => true, suggest: async () => parseSuggestionJson(GOOD) }, cloudProvider()];
      const { recognizer, onStatus: _s, ...rest } = handlers;
      void _s;
      build.loop = new FastLoop({
        ...rest,
        vad: new EnergyVad(-45, 0.032),
        embedder: null,
        labeler: null,
        recognizer: recognizer ?? new FakeSpeechRecognizer(),
        llm: new ProviderChain(providers),
        sttGraceMs: 100,
        pollMs: 5,
      });
      return {
        loop: build.loop,
        status: "energy VAD · speaker-ID off (test) · LLM cloud · browser speech recognition",
        capabilities: {
          vad: "energy" as const,
          speakerId: { active: false, reason: "test", enrolled: 0, model: null, droppedForModel: 0 },
          llm: ["cloud"],
        },
      };
    },
  };
  return build;
}

/** Push 48 kHz float32 worklet batches (what WebAudioCapture posts). */
function feed48k(pcm16k: Int16Array) {
  // Upsample 16k -> 48k by repetition: the hook's resampler brings it back.
  for (let off = 0; off < pcm16k.length; off += 1600) {
    const chunk = pcm16k.subarray(off, Math.min(off + 1600, pcm16k.length));
    const f32 = new Float32Array(chunk.length * 3);
    for (let i = 0; i < chunk.length; i++) {
      const v = chunk[i] / 32768;
      f32[i * 3] = v;
      f32[i * 3 + 1] = v;
      f32[i * 3 + 2] = v;
    }
    mockWeb.onBuffer?.({ data: f32.buffer, sampleRate: 48000, channels: 1, timestamp: 0 });
  }
}

const flush = () => new Promise((r) => setTimeout(r, 30));
const originalOS = Platform.OS;
let logSpy: jest.SpyInstance;

beforeEach(() => {
  Object.defineProperty(Platform, "OS", { value: "web", configurable: true });
  FakeWebSocket.instances = [];
  (globalThis as Record<string, unknown>).WebSocket = FakeWebSocket;
  mockWeb.onBuffer = null;
  mockWeb.start.mockReset().mockResolvedValue(undefined);
  mockWeb.stop.mockReset().mockResolvedValue(undefined);
  primedMock.instance = new FakeSpeechRecognizer();
  primedMock.calls = 0;
  mockUnlock.mockClear();
  speakMock.mockClear();
  logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
});

afterEach(() => {
  Object.defineProperty(Platform, "OS", { value: originalOS, configurable: true });
  logSpy.mockRestore();
});

const capable = { capable: true, reason: "browser speech recognition available" };
const postSession = async () => ({ status: "unsupported" as const });

describe("useAudioStream — web build with the fast loop", () => {
  it("primes STT + TTS in the Start gesture, feeds one frame stream to both the WS and the loop", async () => {
    const build = fakeBuild();
    const hook = await renderHook(() =>
      useAudioStream({ capability: capable, makeFastLoop: build.make, postSession }),
    );
    expect(hook.result.current.liveCapable).toBe(true);
    expect(hook.result.current.liveMode).toBe(true);
    await act(() => hook.result.current.setSpeechEnabled(true));

    await act(async () => {
      await hook.result.current.startSession("sess-web-live", 60);
    });
    // Gesture-bound work happened, and the primed recognizer is the loop's.
    expect(mockUnlock).toHaveBeenCalledTimes(1);
    expect(primedMock.calls).toBe(1);
    expect(build.handlers?.recognizer).toBe(primedMock.instance);
    expect(primedMock.instance!.started).toBe(true);
    // Build progress surfaced, then the final status.
    expect(hook.result.current.liveStatus).toMatch(/^On-device: energy VAD/);

    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    // The socket learned the phone speaks for itself.
    expect(ws.sentJson().some((m) => m.type === "config" && m.tts === "on-device")).toBe(true);

    // One second of tone, STT words, then silence to close the turn.
    await act(async () => {
      feed48k(toneInt16(1.0, -20));
      primedMock.instance!.emit({
        text: "you never listen",
        isFinal: true,
        segments: [
          { startTimeMillis: 100, endTimeMillis: 400, segment: "you" },
          { startTimeMillis: 400, endTimeMillis: 700, segment: "never" },
          { startTimeMillis: 700, endTimeMillis: 950, segment: "listen" },
        ],
      });
      feed48k(silenceInt16(0.6));
      await build.loop!.settle();
      await flush();
    });

    // The WS got the same 16 kHz int16 frames the loop heard: 1.6 s ≈ 16 frames.
    expect(ws.sentBinary.length).toBeGreaterThanOrEqual(15);
    expect(ws.sentBinary[0].byteLength).toBe(3200);
    // The loop finalized the turn from those frames and spoke its suggestion.
    expect(hook.result.current.transcript.map((t) => t.text)).toEqual(["you never listen"]);
    const local = ws.sentJson().filter((m) => m.type === "turn_local");
    expect(local).toHaveLength(1);
    expect(local[0].text).toBe("you never listen");
    expect(local[0].transcript_source).toBe("on-device");
    expect(speakMock).toHaveBeenCalledWith("Say: I hear you.", expect.anything());

    await act(async () => {
      await hook.result.current.stopSession();
    });
    expect(mockWeb.stop).toHaveBeenCalled();
    expect(primedMock.instance!.stopped).toBe(true);
  });

  it("therapist mode: on-screen only — neither the loop nor the cloud event is voiced", async () => {
    const build = fakeBuild({ provider: "cloud" });
    const hook = await renderHook(() =>
      useAudioStream({ capability: capable, makeFastLoop: build.make, postSession }),
    );
    await act(() => {
      hook.result.current.setSpeechEnabled(true);
      hook.result.current.setSessionMode("therapist");
    });
    await act(async () => {
      await hook.result.current.startSession("sess-therapist", 50);
    });
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    await act(async () => {
      feed48k(toneInt16(0.8, -20));
      primedMock.instance!.emit({ text: "I feel unheard", isFinal: true });
      feed48k(silenceInt16(0.6));
      await build.loop!.settle();
      await flush();
    });
    // The cloud answers the turn_local: rendered, never spoken in therapist mode.
    await act(() =>
      ws.emitServer({
        type: "suggestion",
        speaker: "Speaker A",
        suggestions: ["Try: what would help right now?"],
        suggestion_source: "cloud",
      }),
    );
    expect(hook.result.current.suggestions.map((s) => s.texts[0])).toContain("Try: what would help right now?");
    expect(speakMock).not.toHaveBeenCalled();
    await act(async () => {
      await hook.result.current.stopSession();
    });
  });

  it("the browser releasing the mic mid-session is reported, not silently ignored", async () => {
    const build = fakeBuild();
    const hook = await renderHook(() =>
      useAudioStream({ capability: capable, makeFastLoop: build.make, postSession }),
    );
    await act(async () => {
      await hook.result.current.startSession("sess-lock", 50);
    });
    await act(() => mockWeb.onTrackEnded?.("ended"));
    expect(hook.result.current.micError).toMatch(/released the microphone/);
    await act(async () => {
      await hook.result.current.stopSession();
    });
  });

  it("live mode off: no priming, no loop, the legacy server path exactly as before", async () => {
    const build = fakeBuild();
    const makeSpy = jest.fn(build.make);
    const hook = await renderHook(() =>
      useAudioStream({ capability: capable, makeFastLoop: makeSpy, postSession }),
    );
    await act(() => hook.result.current.setLiveMode(false));
    await act(async () => {
      await hook.result.current.startSession("sess-legacy", 50);
    });
    expect(primedMock.calls).toBe(0);
    expect(makeSpy).not.toHaveBeenCalled();
    expect(hook.result.current.isRecording).toBe(true);
    await act(async () => {
      await hook.result.current.stopSession();
    });
  });
});
