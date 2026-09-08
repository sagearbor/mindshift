import { renderHook, act } from "@testing-library/react-native";

/**
 * Tier B: the server's `watch_connected` frames (a paired watch's companion
 * socket appeared on / left the watch relay) drive the hook's `watchConnected`
 * flag, which Live Coach renders as "watch connected — nudges on your wrist".
 * Same expo-audio + FakeWebSocket harness as useAudioStreamLive.test.tsx.
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

describe("watch_connected frames", () => {
  it("tracks the server's watch_connected frames and resets per session", async () => {
    const hook = await renderHook(() =>
      useAudioStream({ capability: { capable: false, reason: "test" } }),
    );
    expect(hook.result.current.watchConnected).toBe(false);

    await act(async () => {
      await hook.result.current.startSession("watch-1", 50);
    });
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());

    await act(() => ws.emitServer({ type: "watch_connected", connected: true }));
    expect(hook.result.current.watchConnected).toBe(true);

    await act(() => ws.emitServer({ type: "watch_connected", connected: false }));
    expect(hook.result.current.watchConnected).toBe(false);

    // Malformed "connected" is treated as not-connected, never truthy-coerced.
    await act(() => ws.emitServer({ type: "watch_connected", connected: "yes" }));
    expect(hook.result.current.watchConnected).toBe(false);

    // Flag on, then end the session: the NEXT session must start clean — the
    // watch state belongs to the server session that reported it.
    await act(() => ws.emitServer({ type: "watch_connected", connected: true }));
    expect(hook.result.current.watchConnected).toBe(true);
    await act(async () => {
      await hook.result.current.stopSession();
    });
    await act(() => FakeWebSocket.instances[0].emitServer({ type: "session_complete" }));

    await act(async () => {
      await hook.result.current.startSession("watch-2", 50);
    });
    expect(hook.result.current.watchConnected).toBe(false);
  });
});
