import { renderHook, act } from "@testing-library/react-native";

/**
 * The praise lane's phone half: a DELIVERED positive is sent up the live
 * session socket as `{type:"positive", code, t}` so the server can relay it to
 * a paired watch.
 *
 * Why it exists: the watch relay carried alert vectors only, so a wearer with
 * the phone in a pocket felt every complaint and no praise — a pure complaint
 * channel, which is worse than having no positives at all. The phone stays the
 * only DETECTOR (its two-minute cap has already run by the time this fires),
 * so the flash on the screen and the buzz on the wrist can never disagree.
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
import type { FastLoopHandlers } from "../src/live/defaultDeps";
import type { PositiveNudge } from "../src/live/positiveNudges";

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
  sentJson() {
    return this.sent.map((s) => JSON.parse(s));
  }
}

const GOOD = '{"suggestion":"Say: I hear you.","tone":{"warmth":50,"label":"calm"}}';

/** Captures the handlers the hook binds, so a positive can be delivered
 *  directly — detection itself is covered by positiveNudges' own vectors. */
function captureHandlers() {
  const captured: { handlers: FastLoopHandlers | null } = { handlers: null };
  const make = async (handlers: FastLoopHandlers) => {
    captured.handlers = handlers;
    return {
      loop: new FastLoop({
        ...handlers,
        vad: new EnergyVad(-45, 0.032),
        embedder: null,
        labeler: null,
        recognizer: new FakeSpeechRecognizer(),
        llm: new ProviderChain([
          { name: "os", isAvailable: async () => true, suggest: async () => parseSuggestionJson(GOOD) },
        ]),
        sttGraceMs: 100,
        pollMs: 5,
      }),
      status: "test",
      capabilities: {
        vad: "energy" as const,
        speakerId: { active: false, reason: "test", enrolled: 0, model: null, droppedForModel: 0 },
        llm: ["os"],
      },
    };
  };
  return { captured, make };
}

function positive(over: Partial<PositiveNudge> = {}): PositiveNudge {
  return { code: "E", t: 41.5, turnIndex: 3, delivered: true, detail: "you let them finish", ...over };
}

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
  for (const r of rendered.splice(0)) r.unmount();
  logSpy.mockRestore();
});

/** Rendered hooks are unmounted in afterEach: an unmounted tree left running
 *  timers is what used to crash a whole Jest worker in this suite. */
const rendered: { unmount: () => void }[] = [];

async function startedSession() {
  const { captured, make } = captureHandlers();
  const hook = await renderHook(() =>
    useAudioStream({
      capability: { capable: true, reason: "ok" },
      makeFastLoop: make,
      postSession: async () => ({ status: "unsupported" as const }),
    }),
  );
  rendered.push(hook);
  await act(async () => {
    await hook.result.current.startSession("praise-1", 50);
  });
  const ws = FakeWebSocket.instances[0];
  await act(() => ws.emitOpen());
  return { hook, ws, captured };
}

describe("delivered positives are relayed to the wrist", () => {
  it("sends one positive frame carrying the vocabulary code and session clock", async () => {
    const { ws, captured, hook } = await startedSession();
    await act(async () => {
      captured.handlers!.onPositiveNudge?.(positive());
    });
    expect(ws.sentJson()).toContainEqual({ type: "positive", code: "E", t: 41.5 });
    // It still does its local job: the flash and the tally.
    expect(hook.result.current.positiveFlash?.code).toBe("E");
    expect(hook.result.current.positiveCounts.E).toBe(1);
  });

  it("does not send a positive the two-minute cap withheld", async () => {
    const { ws, captured, hook } = await startedSession();
    await act(async () => {
      captured.handlers!.onPositiveNudge?.(positive({ delivered: false }));
    });
    // The cap governs the INTERRUPTION on every device that has one. A wrist
    // buzz for something the phone deliberately stayed quiet about would be
    // the cap leaking out the side.
    expect(ws.sentJson().filter((m) => m.type === "positive")).toEqual([]);
    expect(hook.result.current.positiveFlash).toBeNull();
    // ...but the achievement is still counted, exactly as on the phone.
    expect(hook.result.current.positiveCounts.E).toBe(1);
  });

  it("one frame per positive, in order, with its own code", async () => {
    const { ws, captured } = await startedSession();
    await act(async () => {
      captured.handlers!.onPositiveNudge?.(positive({ code: "D", t: 10 }));
      captured.handlers!.onPositiveNudge?.(positive({ code: "R", t: 20, turnIndex: 5 }));
    });
    expect(ws.sentJson().filter((m) => m.type === "positive")).toEqual([
      { type: "positive", code: "D", t: 10 },
      { type: "positive", code: "R", t: 20 },
    ]);
  });

  it("survives a closed socket — praise is never worth a crash", async () => {
    const { ws, captured, hook } = await startedSession();
    ws.readyState = 3;
    await act(async () => {
      captured.handlers!.onPositiveNudge?.(positive());
    });
    expect(ws.sentJson().filter((m) => m.type === "positive")).toEqual([]);
    expect(hook.result.current.positiveFlash?.code).toBe("E");
  });
});
