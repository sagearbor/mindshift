import { renderHook, act } from "@testing-library/react-native";
import { Platform } from "react-native";

/**
 * Web-path integration for useAudioStream. On web (Platform.OS === "web")
 * expo-audio has no recorder, so the hook drives our WebAudioCapture backend
 * instead. This suite mocks that backend to prove the hook: opens the SAME
 * WebSocket protocol, feeds captured frames through the SAME resample/int16/
 * batching pipeline as native, and reports honest states (permission denied,
 * unsupported browser). The heavy Web Audio API is exercised separately in
 * webAudioCapture.test.ts.
 */

// Controllable WebAudioCapture mock: captures the onBuffer the hook registers
// and lets tests drive start()'s outcome.
const mockWeb = {
  onBuffer: null as ((b: unknown) => void) | null,
  supported: true,
  startImpl: null as null | (() => Promise<void>),
  start: jest.fn<Promise<void>, []>(),
  stop: jest.fn<Promise<void>, []>(),
};

jest.mock("../src/utils/webAudioCapture", () => {
  class WebCaptureError extends Error {
    kind: string;
    constructor(kind: string, message: string) {
      super(message);
      this.kind = kind;
    }
  }
  return {
    __esModule: true,
    WebCaptureError,
    isWebAudioCaptureSupported: () => mockWeb.supported,
    WebAudioCapture: class {
      constructor(opts: { onBuffer: (b: unknown) => void }) {
        mockWeb.onBuffer = opts.onBuffer;
      }
      start() {
        return mockWeb.start();
      }
      stop() {
        return mockWeb.stop();
      }
    },
  };
});

import { useAudioStream } from "../src/hooks/useAudioStream";
import { WebCaptureError } from "../src/utils/webAudioCapture";

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
    else if (ArrayBuffer.isView(data))
      this.sentBinary.push(
        data.buffer.slice(
          data.byteOffset,
          data.byteOffset + data.byteLength,
        ) as ArrayBuffer,
      );
    else this.sentBinary.push(data);
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

function makePcmBuffer(sampleCount: number, sampleRate = 48000, value = 0.25) {
  const samples = new Float32Array(sampleCount).fill(value);
  return { data: samples.buffer, sampleRate, channels: 1, timestamp: 0 };
}

const originalOS = Platform.OS;

beforeEach(() => {
  // Platform.OS is a plain data property; override it to drive the web branch.
  Object.defineProperty(Platform, "OS", {
    value: "web",
    configurable: true,
  });
  FakeWebSocket.instances = [];
  // @ts-expect-error — install fake WebSocket
  global.WebSocket = FakeWebSocket;
  mockWeb.onBuffer = null;
  mockWeb.supported = true;
  mockWeb.start.mockReset().mockResolvedValue(undefined);
  mockWeb.stop.mockReset().mockResolvedValue(undefined);
});

afterEach(() => {
  Object.defineProperty(Platform, "OS", {
    value: originalOS,
    configurable: true,
  });
});

describe("useAudioStream — web capture path", () => {
  it("captures via WebAudioCapture and streams identical 16kHz int16 frames", async () => {
    const hook = await renderHook(() => useAudioStream());
    await act(async () => {
      await hook.result.current.startSession("sess-web", 40);
    });

    // Capture started, session opened on the same WS protocol.
    expect(mockWeb.start).toHaveBeenCalledTimes(1);
    expect(FakeWebSocket.instances).toHaveLength(1);
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    expect(hook.result.current.connectionStatus).toBe("live");
    expect(hook.result.current.isRecording).toBe(true);
    expect(
      ws.sentJson().filter((m) => m.type === "config" && m.empathy_slider === 40),
    ).toHaveLength(1);

    // 4800 samples @ 48kHz = 100ms -> one 1600-sample (3200-byte) 16kHz frame.
    await act(() => mockWeb.onBuffer!(makePcmBuffer(4800, 48000)));
    expect(ws.sentBinary).toHaveLength(1);
    expect(ws.sentBinary[0].byteLength).toBe(3200);
    const samples = new Int16Array(ws.sentBinary[0]);
    expect(samples).toHaveLength(1600);
    // 0.25 * 32767 -> 8192, same conversion as native.
    expect(samples[0]).toBe(8192);
  });

  it("permission denied: honest banner, no session, no audio", async () => {
    mockWeb.start.mockRejectedValue(
      new WebCaptureError("permission-denied", "denied"),
    );
    const hook = await renderHook(() => useAudioStream());
    await act(async () => {
      await hook.result.current.startSession("sess-web", 50);
    });

    expect(hook.result.current.micError).toMatch(/permission denied/i);
    expect(hook.result.current.isRecording).toBe(false);
    expect(hook.result.current.connectionStatus).toBe("idle");
    expect(FakeWebSocket.instances).toHaveLength(0);
    // The half-started capture was released.
    expect(mockWeb.stop).toHaveBeenCalled();
  });

  it("unsupported browser: honest banner, session still runs without audio", async () => {
    mockWeb.supported = false;
    const hook = await renderHook(() => useAudioStream());
    await act(async () => {
      await hook.result.current.startSession("sess-web", 50);
    });

    expect(hook.result.current.micError).toMatch(/can't capture audio/i);
    // Session runs so the server can still report its own state.
    expect(FakeWebSocket.instances).toHaveLength(1);
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    expect(hook.result.current.connectionStatus).toBe("live");
    expect(hook.result.current.sessionActive).toBe(true);
    expect(hook.result.current.isRecording).toBe(false);
    expect(mockWeb.start).not.toHaveBeenCalled();
    expect(ws.sentBinary).toHaveLength(0);
  });

  it("stopSession releases the web mic and runs the stop handshake", async () => {
    const hook = await renderHook(() => useAudioStream());
    await act(async () => {
      await hook.result.current.startSession("sess-web", 50);
    });
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());

    await act(async () => {
      await hook.result.current.stopSession();
    });

    // Same graceful-stop handshake as native, plus the web mic is released.
    expect(ws.sentJson().some((m) => m.type === "stop")).toBe(true);
    expect(mockWeb.stop).toHaveBeenCalled();
    expect(hook.result.current.isRecording).toBe(false);
  });
});
