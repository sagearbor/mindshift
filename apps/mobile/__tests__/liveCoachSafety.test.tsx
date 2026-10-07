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
import { silenceInt16, toneInt16, unitVector } from "../src/live/testing/synth";
import { SpeakerLabeler } from "../src/live/speakerId";
import { useDiagnosticsStore } from "../src/diagnostics/diagnostics";

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

/** A route probe the test flips like a headset being plugged/unplugged. */
function switchableRoute(initial: "private" | "public" | "unknown" = "private") {
  const state = { route: initial as "private" | "public" | "unknown", checks: 0 };
  return {
    state,
    make: () => ({
      check: () => {
        state.checks += 1;
        return state.route;
      },
      dispose: () => {},
    }),
  };
}

const LEGACY = { capability: { capable: false, reason: "test: server path" } } as const;

describe("B. Earpiece mode never falls back to the loudspeaker", () => {
  it("headset removed mid-session: speech is cut, nothing more is spoken, and the screen state says why; reconnecting restores it", async () => {
    {
      const route = switchableRoute("private");
      const { hook, ws } = await startEarpieceSession({ ...LEGACY, makeAudioRouteProbe: route.make });
      expect(hook.result.current.privateAudioRoute).toBe("private");

      await act(() => ws.emitServer(cloudSuggestion("Breathe first.")));
      expect(speakMock).toHaveBeenCalledTimes(1);
      speechStopMock.mockClear();

      // The earpiece comes out (Bluetooth disconnects) while it is talking.
      route.state.route = "public";
      await act(async () => {
        await flush(600); // one route poll (ROUTE_POLL_MS)
      });
      expect(speechStopMock).toHaveBeenCalled(); // the line in flight is cut
      expect(hook.result.current.privateAudioRoute).toBe("public");

      await act(() => ws.emitServer(cloudSuggestion("Say it calmly.")));
      expect(speakMock).toHaveBeenCalledTimes(1); // silent, not the speaker
      // Still shown on screen.
      expect(hook.result.current.suggestions[0].texts[0]).toBe("Say it calmly.");

      // Headset back: speech resumes.
      route.state.route = "private";
      await act(async () => {
        await flush(600);
      });
      expect(hook.result.current.privateAudioRoute).toBe("private");
      await act(() => ws.emitServer(cloudSuggestion("Ask what they need.")));
      expect(speakMock).toHaveBeenCalledTimes(2);
      expect(speakMock.mock.calls[1][0]).toBe("Ask what they need.");

      await act(async () => {
        await hook.result.current.stopSession();
      });
    }
  });

  it("the route is checked right before every utterance, not only on the poll", async () => {
    const route = switchableRoute("private");
    const { hook, ws } = await startEarpieceSession({ ...LEGACY, makeAudioRouteProbe: route.make });
    route.state.route = "public"; // disconnect lands between two polls
    await act(() => ws.emitServer(cloudSuggestion("Not out loud.")));
    expect(speakMock).not.toHaveBeenCalled();
    expect(hook.result.current.privateAudioRoute).toBe("public");
    await act(async () => {
      await hook.result.current.stopSession();
    });
  });

  it("a platform that cannot tell the route ('unknown') is treated as no headset", async () => {
    const route = switchableRoute("unknown");
    const { hook, ws } = await startEarpieceSession({ ...LEGACY, makeAudioRouteProbe: route.make });
    await act(() => ws.emitServer(cloudSuggestion("Not out loud.")));
    expect(speakMock).not.toHaveBeenCalled();
    expect(hook.result.current.privateAudioRoute).toBe("unknown");
    await act(async () => {
      await hook.result.current.stopSession();
    });
  });

  it("the on-device loop's own lines obey the same gate", async () => {
    const route = switchableRoute("public");
    const fake = makeFakeFastLoop();
    const { hook } = await startEarpieceSession({
      makeFastLoop: fake.make,
      makeAudioRouteProbe: route.make,
      postSession: async () => ({ status: "unsupported" as const }),
    });
    await act(async () => {
      feed(toneInt16(1.0, -20));
      fake.rec.emit({ text: "you always do this", isFinal: true });
      feed(silenceInt16(0.5));
      await fake.loop!.settle();
      await flush();
    });
    expect(hook.result.current.suggestions[0].source).toBe("on-device");
    expect(speakMock).not.toHaveBeenCalled();
    await act(async () => {
      await hook.result.current.stopSession();
    });
  });

  it("In-person (speaker) mode speaks aloud by design and is not gated", async () => {
    const route = switchableRoute("public");
    const hook = await renderHook(() => useAudioStream({ ...LEGACY, makeAudioRouteProbe: route.make }));
    await act(() => {
      hook.result.current.setSpeechEnabled(true);
      hook.result.current.setSessionMode("speaker");
    });
    await act(async () => {
      await hook.result.current.startSession("speaker-1", 50);
    });
    const ws = FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
    await act(() => ws.emitOpen());
    await act(() => ws.emitServer(cloudSuggestion("Out loud is fine here.")));
    expect(speakMock).toHaveBeenCalledTimes(1);
    expect(hook.result.current.privateAudioRoute).toBeNull();
    await act(async () => {
      await hook.result.current.stopSession();
    });
  });
});

describe("C. Mid-stream identity: no 'Speaker A' shortcut", () => {
  it("starts unconfirmed (coaching still runs) and tells the server once the on-device voiceprint matches the wearer", async () => {
    const D = 192;
    const fake = makeFakeFastLoop();
    const originalMake = fake.make;
    fake.make = async (handlers: FastLoopHandlers) => {
      const build = await originalMake(handlers);
      // The son speaks first (an unenrolled voice), then the wearer.
      const queue = [unitVector(D, 7), unitVector(D, 0)];
      fake.loop = new FastLoop({
        ...handlers,
        vad: new EnergyVad(-45, 0.032),
        embedder: { embed: async () => queue.shift() ?? unitVector(D, 9) },
        labeler: new SpeakerLabeler([{ personId: "self", displayName: "Sage", isSelf: true, embedding: unitVector(D, 0) }]),
        recognizer: fake.rec,
        llm: new ProviderChain([{ name: "os", isAvailable: async () => true, suggest: async () => parseSuggestionJson(GOOD) }]),
        sttGraceMs: 100,
        pollMs: 5,
      });
      return { ...build, loop: fake.loop };
    };
    const { hook, ws } = await startEarpieceSession({
      makeFastLoop: fake.make,
      postSession: async () => ({ status: "unsupported" as const }),
    });
    const first = ws.sentJson().find((m) => m.type === "config" && "self_speaker" in m);
    expect(first).toMatchObject({ self_speaker: null, wearer_voice_confirmed: false, wearer_identity: "unconfirmed" });
    expect(JSON.stringify(ws.sentJson())).not.toContain("Speaker A");

    const turn = async (text: string) => {
      await act(async () => {
        feed(toneInt16(2.0, -20));
        fake.rec.emit({ text, isFinal: true });
        feed(silenceInt16(0.5));
        await fake.loop!.settle();
        await flush();
      });
    };
    // Mid-argument switch-on: the first voice is NOT the wearer by default,
    // and an unconfirmed session still coaches (a response card).
    await turn("you never let me finish");
    expect(hook.result.current.wearerVoiceConfirmed).toBe(false);
    expect(hook.result.current.suggestions[0].kind).toBe("response");
    expect(ws.sentJson().filter((m) => m.wearer_voice_confirmed === true)).toHaveLength(0);

    await turn("okay, I hear you");
    expect(hook.result.current.wearerVoiceConfirmed).toBe(true);
    const confirmations = ws.sentJson().filter((m) => m.type === "config" && m.wearer_voice_confirmed === true);
    expect(confirmations).toEqual([{ type: "config", wearer_voice_confirmed: true, wearer_identity: "voiceprint" }]);
    await act(async () => {
      await hook.result.current.stopSession();
    });
  });
});

// --- Cold start: the loop comes up ~8 s after Start ---------------------------
import * as nodePath from "path";
import { readWav16kMono } from "../src/live/replay/wav";

/** A makeFastLoop whose build only finishes when the test says so — the
 *  ≈8.5 s model load the Pixel 10 showed on 2026-10-07. */
function slowBuild(fake: ReturnType<typeof makeFakeFastLoop>) {
  let release: () => void = () => {};
  const gate = new Promise<void>((r) => {
    release = r;
  });
  const make = async (handlers: FastLoopHandlers) => {
    await gate;
    return fake.make(handlers);
  };
  return { make, release: () => release() };
}

async function startWithSlowLoop(fake: ReturnType<typeof makeFakeFastLoop>, posted: LiveSessionBody[]) {
  const slow = slowBuild(fake);
  const hook = await renderHook(() =>
    useAudioStream({
      capability: { capable: true, reason: "ok" },
      makeFastLoop: slow.make,
      postSession: async (body) => {
        posted.push(body);
        return { status: "unsupported" as const };
      },
    }),
  );
  await act(() => {
    hook.result.current.setSpeechEnabled(true);
    hook.result.current.setSessionMode("earpiece");
  });
  let starting: Promise<void> = Promise.resolve();
  await act(async () => {
    starting = hook.result.current.startSession("cold-1", 50);
    await flush();
  });
  const ws = FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
  await act(() => ws.emitOpen());
  return { hook, ws, slow, starting: () => starting };
}

describe("Cold start: audio from Start reaches the loop once it is up", () => {
  it("speech at t=0 and t=3 s, loop up at 8 s: both become loop turns on the capture clock, not claimed as on-device words", async () => {
    const fake = makeFakeFastLoop();
    const posted: LiveSessionBody[] = [];
    const { hook, ws, slow, starting } = await startWithSlowLoop(fake, posted);

    // 8 s of a heated opening while the models load: 1 s, pause, 1 s, pause.
    await act(async () => {
      feed(toneInt16(1.0, -20));
      feed(silenceInt16(2.0));
      feed(toneInt16(1.0, -20));
      feed(silenceInt16(4.0));
    });
    expect(fake.loop).toBeNull(); // still building
    await act(async () => {
      slow.release();
      await starting();
      await fake.loop!.settle();
      await flush();
    });

    // After the loop is up, a turn the recognizer does hear.
    await act(async () => {
      feed(toneInt16(1.0, -20));
      fake.rec.emit({ text: "can we slow down", isFinal: true });
      feed(silenceInt16(0.5));
      await fake.loop!.settle();
      await flush();
    });

    const turns = fake.loop!.turnsSoFar;
    const early = turns.filter((t) => t.endTime <= 8);
    expect(early.length).toBe(2);
    expect(early[0].startTime).toBeCloseTo(0, 0);
    expect(early[1].startTime).toBeCloseTo(3, 0);
    expect(early.every((t) => t.beforeLoopUp === true && t.text === "")).toBe(true);
    const late = turns.filter((t) => t.startTime >= 8);
    expect(late).toHaveLength(1);
    expect(late[0].text).toBe("can we slow down");
    expect(late[0].startTime).toBeCloseTo(8, 0); // capture clock, not loop clock

    // Only the turn the phone actually heard is claimed as on-device words.
    const turnLocal = ws.sentJson().filter((m) => m.type === "turn_local");
    expect(turnLocal.map((m) => m.text)).toEqual(["can we slow down"]);

    await act(async () => {
      await hook.result.current.stopSession();
    });
    // started_at is Start; loop_up_at is when the loop came up.
    expect(posted).toHaveLength(1);
    expect(typeof posted[0].loop_up_at).toBe("string");
    expect(posted[0].started_at <= posted[0].loop_up_at!).toBe(true);

    // The diagnostics record carries the loop's own health counters.
    await act(async () => {
      ws.emitServer({ type: "session_complete" });
      await flush();
    });
    const dx = useDiagnosticsStore.getState().lastSession!;
    expect(dx.loopUpAt).toBe(posted[0].loop_up_at);
    expect(dx.loop).toMatchObject({
      turns: 3,
      localTurns: 1,
      spansClosed: 3,
      spansDroppedShort: 0,
    });
    expect(dx.loop!.prerollSeconds).toBeGreaterThanOrEqual(7.9);
    expect(dx.loop!.vadFrames).toBeGreaterThan(0);
    expect(dx.loop!.vadSpeechFrames).toBeGreaterThan(0);
    expect(dx.loop!.inputDbfs).not.toBeNull();
    expect(dx.loop!.inputDbfs!).toBeLessThan(0);
    expect(typeof dx.loop!.startupMs).toBe("number");
    expect(dx.loop!.sttRestartCodes).toEqual({});
    expect(dx.loop!.liveSeconds).toBeGreaterThanOrEqual(0);
  });

  it("replays the real family recording (fixtures/pcm16k_family_real_6s.wav ×2) with the loop up at 8 s: the first 8 s still yields turns", async () => {
    const wav = readWav16kMono(nodePath.join(__dirname, "fixtures/pcm16k_family_real_6s.wav"));
    const scene = new Int16Array(wav.length * 2);
    scene.set(wav, 0);
    scene.set(wav, wav.length);
    const fake = makeFakeFastLoop();
    const posted: LiveSessionBody[] = [];
    const { hook, slow, starting } = await startWithSlowLoop(fake, posted);
    const cut = 8 * 16000;
    await act(async () => {
      feed(scene.subarray(0, cut));
    });
    await act(async () => {
      slow.release();
      await starting();
    });
    await act(async () => {
      feed(scene.subarray(cut));
      feed(silenceInt16(1.0));
      await fake.loop!.settle();
      await flush();
    });
    const turns = fake.loop!.turnsSoFar;
    expect(turns.length).toBeGreaterThan(0);
    expect(turns.some((t) => t.startTime < 8)).toBe(true);
    expect(Math.max(...turns.map((t) => t.endTime))).toBeGreaterThan(8);
    await act(async () => {
      await hook.result.current.stopSession();
    });
  });
});

// --- D. A loop that hears nothing while the transcript fills ---------------
describe("D. Zero local turns while the transcript shows speech = a broken loop", () => {
  it("replay: the real family recording is transcribed by the server, the loop yields no turns — nothing is voiced, the status says why, diagnostics report it", async () => {
    const wav = readWav16kMono(nodePath.join(__dirname, "fixtures/pcm16k_family_real_6s.wav"));
    // A loop whose VAD never fires (the dinner symptom: latency log empty,
    // providers={}), fed the real recording.
    const rec = new FakeSpeechRecognizer();
    let loop: FastLoop | null = null;
    const make = async (handlers: FastLoopHandlers) => {
      loop = new FastLoop({
        ...handlers,
        vad: new EnergyVad(6, 0.032), // +6 dBFS: never speech
        embedder: null,
        labeler: null,
        recognizer: rec,
        llm: new ProviderChain([cloudProvider()]),
        sttGraceMs: 100,
        pollMs: 5,
      });
      return {
        loop,
        status: "energy VAD · speaker-ID off · LLM cloud",
        capabilities: {
          vad: "energy" as const,
          speakerId: { active: false, reason: "test", enrolled: 0, model: null, droppedForModel: 0 },
          llm: ["cloud"],
        },
      };
    };
    const posted: LiveSessionBody[] = [];
    const { hook, ws } = await startEarpieceSession({
      makeFastLoop: make,
      postSession: async (body) => {
        posted.push(body);
        return { status: "unsupported" as const };
      },
    });
    await act(async () => {
      feed(wav);
      await loop!.settle();
    });
    // The server transcribes what was said and coaches every utterance.
    const lines = ["you never help", "I did the dishes", "that was yesterday", "fine whatever", "can we not do this now"];
    for (const [i, text] of lines.entries()) {
      await act(async () => {
        ws.emitServer({ type: "transcript", speaker: i % 2 ? "Speaker B" : "Speaker A", text, start_time: i, end_time: i + 0.8 });
        ws.emitServer(cloudSuggestion(`Cloud line ${i}`, text));
        await flush(5);
      });
    }
    expect(loop!.turnsSoFar).toHaveLength(0);
    // Shown on screen, never in the earpiece — not even the first ones.
    expect(hook.result.current.suggestions.map((sg) => sg.texts[0])).toContain("Cloud line 4");
    expect(speakMock).not.toHaveBeenCalled();
    expect(hook.result.current.liveStatus).toMatch(/on-device listening stalled/);

    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(async () => {
      ws.emitServer({ type: "session_complete" });
      await flush();
    });
    const dx = useDiagnosticsStore.getState().lastSession!;
    expect(dx.errors).toContain("on-device loop finalized 0 turns while transcript had 5");
    expect(dx.loop).toMatchObject({ turns: 0, localTurns: 0, spansClosed: 0, vadSpeechFrames: 0 });
    expect(dx.loop!.vadFrames).toBeGreaterThan(100); // it got audio; the VAD never fired
  });

  it("a loop that resumes producing turns clears the stall (cloud lines can be voiced again)", async () => {
    const fake = makeFakeFastLoop({ provider: "cloud" });
    const { hook, ws } = await startEarpieceSession({
      makeFastLoop: fake.make,
      postSession: async () => ({ status: "unsupported" as const }),
    });
    for (let i = 0; i < 3; i++) {
      await act(async () => {
        ws.emitServer({ type: "transcript", speaker: "Speaker A", text: `missed ${i}`, start_time: i, end_time: i + 0.5 });
      });
    }
    expect(hook.result.current.liveStatus).toMatch(/stalled/);
    await act(async () => {
      feed(toneInt16(1.0, -20));
      fake.rec.emit({ text: "okay let me listen", isFinal: true });
      feed(silenceInt16(0.5));
      await fake.loop!.settle();
      await flush();
    });
    expect(hook.result.current.liveStatus).not.toMatch(/stalled/);
    // The phone had nothing to say for its latest turn: the cloud's answer to it is voiced.
    await act(async () => {
      ws.emitServer(cloudSuggestion("Thank them for waiting.", "okay let me listen"));
      await flush();
    });
    expect(speakMock).toHaveBeenCalledWith("Thank them for waiting.", expect.anything());
    await act(async () => {
      await hook.result.current.stopSession();
    });
  });
});
