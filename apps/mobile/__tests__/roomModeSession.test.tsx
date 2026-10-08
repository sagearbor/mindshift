import { renderHook, act } from "@testing-library/react-native";

/**
 * Room mode through the real useAudioStream hook (fake WebSocket, mocked
 * mic): the phone asks the server for the room pipeline, never brings up the
 * personal coaching loop, speaks ONLY the consent line and addressed answers
 * — on the loudspeaker, because room mode's route rule never consults the
 * private-route probe — shows cards silently, and Stop silences everything
 * (the PR #192 speech gate).
 */
const mockMic = {
  onBuffer: null as unknown,
  start: jest.fn<Promise<void>, []>(),
  stop: jest.fn(),
};

jest.mock("expo-audio", () => ({
  __esModule: true,
  requestRecordingPermissionsAsync: async () => ({ status: "granted", granted: true }),
  setAudioModeAsync: async () => undefined,
  useAudioStream: (options?: { onBuffer?: unknown }) => {
    mockMic.onBuffer = options?.onBuffer ?? null;
    return {
      stream: { id: "mock-stream", sampleRate: 16000, channels: 1, isStreaming: false, start: mockMic.start, stop: mockMic.stop },
      isStreaming: false,
    };
  },
}));

import * as Speech from "expo-speech";
import { useAudioStream } from "../src/hooks/useAudioStream";
import type { LiveMode } from "../src/live/localLlm";
import { ROOM_CONSENT_LINE } from "../src/live/roomMode";
import type { AudioRouteProbe } from "../src/live/audioRoute";

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
  emitServer(obj: unknown) {
    this.onmessage?.({ data: JSON.stringify(obj) });
  }
  sentJson() {
    return this.sent.map((s) => JSON.parse(s));
  }
}

const flush = (ms = 20) => new Promise((r) => setTimeout(r, ms));

/** Only the phone's own speaker: no headset anywhere. */
const PUBLIC_ROUTE = (): AudioRouteProbe => ({ check: () => "public", dispose: () => {} });

const ACME_CARD = {
  type: "room_card",
  id: "c1",
  title: "Acme · Q3",
  fact: "Q3 2026: Acme bought 120 seats of the Pro plan for $48,000.",
  source_item_id: "item-acme",
  source_title: "Acme account",
  t: 8,
};
const ACME_ANSWER = {
  type: "room_answer",
  question: "what did Acme buy last quarter?",
  text: "Acme bought 120 seats of the Pro plan for $48,000 in Q3 2026.",
  known: true,
  source_item_ids: ["item-acme"],
  t: 33,
  speak: true,
};

async function start(mode: LiveMode) {
  const makeFastLoop = jest.fn();
  const hook = await renderHook(() =>
    useAudioStream({
      capability: { capable: true, reason: "ok" },
      makeFastLoop,
      makeAudioRouteProbe: PUBLIC_ROUTE,
      postSession: async () => ({ status: "unsupported" as const }),
    }),
  );
  await act(() => {
    hook.result.current.setSpeechEnabled(true);
    hook.result.current.setSessionMode(mode);
  });
  await act(async () => {
    await hook.result.current.startSession(`room-${mode}`, 50);
  });
  const ws = FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
  await act(() => ws.onopen?.({}));
  await act(async () => {
    await flush();
  });
  return { hook, ws, makeFastLoop };
}

let logSpy: jest.SpyInstance;
let warnSpy: jest.SpyInstance;

beforeEach(() => {
  FakeWebSocket.instances = [];
  (globalThis as Record<string, unknown>).WebSocket = FakeWebSocket;
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

describe("room mode session", () => {
  it("asks for the room pipeline and never starts the coaching loop", async () => {
    const { ws, makeFastLoop } = await start("room");
    const config = ws.sentJson().find((f) => f.type === "config");
    expect(config.mode).toBe("room");
    expect(makeFastLoop).not.toHaveBeenCalled();
  });

  it("speaks the consent line once, then only addressed answers — on the loudspeaker", async () => {
    const { hook, ws } = await start("room");
    await act(() => ws.emitServer({ type: "config_ack", mode: "room" }));
    await act(() => ws.emitServer({ type: "config_ack", mode: "room" }));
    expect(speakMock.mock.calls.map((c) => c[0])).toEqual([ROOM_CONSENT_LINE]);
    expect(hook.result.current.room.serverConfirmed).toBe(true);
    speakMock.mockClear();

    // A card is on screen, never spoken.
    await act(() => ws.emitServer(ACME_CARD));
    expect(speakMock).not.toHaveBeenCalled();
    expect(hook.result.current.room.cards.map((c) => c.fact)).toEqual([ACME_CARD.fact]);
    expect(hook.result.current.room.cards[0].sourceTitle).toBe("Acme account");

    // The addressed answer IS spoken, though no headset is connected.
    await act(() => ws.emitServer(ACME_ANSWER));
    expect(speakMock.mock.calls.map((c) => c[0])).toEqual([ACME_ANSWER.text]);
    expect(hook.result.current.room.answers[0].text).toBe(ACME_ANSWER.text);
    speakMock.mockClear();

    // Coaching never speaks in room mode.
    await act(() =>
      ws.emitServer({ type: "suggestion", speaker: "Speaker A", suggestions: ["Slow down."], empathy_slider: 50 }),
    );
    expect(speakMock).not.toHaveBeenCalled();
  });

  it("Stop silences immediately and nothing is spoken after it", async () => {
    const { hook, ws } = await start("room");
    await act(() => ws.emitServer(ACME_ANSWER));
    expect(speakMock).toHaveBeenCalledTimes(1);
    speakMock.mockClear();
    speechStopMock.mockClear();
    await act(async () => {
      void hook.result.current.stopSession();
      await flush();
    });
    expect(speechStopMock).toHaveBeenCalled();
    await act(async () => {
      ws.emitServer({ ...ACME_ANSWER, text: "A late answer.", t: 70 });
      ws.emitServer({ type: "config_ack", mode: "room" });
      await flush();
    });
    expect(speakMock).not.toHaveBeenCalled();
  });

  it("muting answers (speak aloud off) keeps them on screen only", async () => {
    const { hook, ws } = await start("room");
    await act(() => hook.result.current.setSpeechEnabled(false));
    await act(() => ws.emitServer(ACME_ANSWER));
    expect(speakMock).not.toHaveBeenCalled();
    expect(hook.result.current.room.answers).toHaveLength(1);
  });
});

describe("the same frames in earpiece mode", () => {
  it("send no room mode, ignore room frames, and stay silent on a public route", async () => {
    const { hook, ws } = await start("earpiece");
    const config = ws.sentJson().find((f) => f.type === "config");
    expect(config.mode).toBeUndefined();
    await act(() => ws.emitServer(ACME_ANSWER));
    await act(() =>
      ws.emitServer({ type: "suggestion", speaker: "Speaker A", suggestions: ["Say: I hear you."], empathy_slider: 50 }),
    );
    expect(speakMock).not.toHaveBeenCalled();
    expect(hook.result.current.room.answers).toHaveLength(0);
  });
});
