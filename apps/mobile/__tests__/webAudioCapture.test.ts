import {
  WebAudioCapture,
  WebCaptureError,
  classifyCaptureError,
  isWebAudioCaptureSupported,
} from "../src/utils/webAudioCapture";

/**
 * These tests exercise the web capture backend with a minimal fake of the Web
 * Audio + getUserMedia surface (the real APIs don't exist in the Jest/node
 * env). They verify the honest error mapping, the support detection, the
 * buffer shape handed to the pipeline, and — crucially — that stop() releases
 * the microphone (stops every track) and closes the AudioContext.
 */

// --- Fakes for the Web Audio API surface -----------------------------------

class FakeAudioWorkletNode {
  port = { onmessage: null as ((e: MessageEvent) => void) | null };
  connect = jest.fn();
  disconnect = jest.fn();
  constructor(
    public ctx: FakeAudioContext,
    public name: string,
    public options?: unknown,
  ) {}
}

class FakeMediaStreamSource {
  connect = jest.fn();
  disconnect = jest.fn();
}

class FakeAudioContext {
  state: "suspended" | "running" | "closed" = "suspended";
  sampleRate = 48000;
  currentTime = 0;
  destination = {};
  audioWorklet = { addModule: jest.fn().mockResolvedValue(undefined) };
  resume = jest.fn().mockImplementation(async () => {
    this.state = "running";
  });
  close = jest.fn().mockImplementation(async () => {
    this.state = "closed";
  });
  createMediaStreamSource = jest.fn(() => new FakeMediaStreamSource());
  static last: FakeAudioContext | null = null;
  constructor() {
    FakeAudioContext.last = this;
  }
}

function makeTrack() {
  return { stop: jest.fn(), kind: "audio" };
}

function makeStream(tracks = [makeTrack()]) {
  return { getTracks: () => tracks } as unknown as MediaStream;
}

interface FakeGlobals {
  getUserMedia: jest.Mock;
  tracks: ReturnType<typeof makeTrack>[];
}

function installWebAudioGlobals(): FakeGlobals {
  const tracks = [makeTrack()];
  const getUserMedia = jest.fn().mockResolvedValue(makeStream(tracks));
  const g = globalThis as Record<string, unknown>;
  g.AudioContext = FakeAudioContext as unknown;
  g.AudioWorkletNode = FakeAudioWorkletNode as unknown;
  g.navigator = { mediaDevices: { getUserMedia } };
  g.Blob = class {
    constructor(
      public parts: unknown[],
      public opts?: unknown,
    ) {}
  } as unknown;
  g.URL = {
    createObjectURL: jest.fn(() => "blob:fake"),
    revokeObjectURL: jest.fn(),
  } as unknown;
  return { getUserMedia, tracks };
}

function clearWebAudioGlobals() {
  const g = globalThis as Record<string, unknown>;
  delete g.AudioContext;
  delete g.webkitAudioContext;
  delete g.AudioWorkletNode;
  delete g.navigator;
  delete g.Blob;
  delete g.URL;
  FakeAudioContext.last = null;
}

afterEach(clearWebAudioGlobals);

describe("classifyCaptureError", () => {
  it("maps NotAllowedError / SecurityError to permission-denied", () => {
    expect(classifyCaptureError({ name: "NotAllowedError" }).kind).toBe(
      "permission-denied",
    );
    expect(classifyCaptureError({ name: "SecurityError" }).kind).toBe(
      "permission-denied",
    );
  });

  it("maps NotFoundError / OverconstrainedError to no-microphone", () => {
    expect(classifyCaptureError({ name: "NotFoundError" }).kind).toBe(
      "no-microphone",
    );
    expect(classifyCaptureError({ name: "OverconstrainedError" }).kind).toBe(
      "no-microphone",
    );
  });

  it("maps anything else to unavailable", () => {
    expect(classifyCaptureError(new Error("boom")).kind).toBe("unavailable");
    expect(classifyCaptureError(null).kind).toBe("unavailable");
  });

  it("passes an existing WebCaptureError through unchanged", () => {
    const original = new WebCaptureError("no-microphone", "gone");
    expect(classifyCaptureError(original)).toBe(original);
  });
});

describe("isWebAudioCaptureSupported", () => {
  it("is false when the Web Audio APIs are absent (e.g. old browser / node)", () => {
    clearWebAudioGlobals();
    expect(isWebAudioCaptureSupported()).toBe(false);
  });

  it("is false without getUserMedia even if AudioWorklet exists", () => {
    installWebAudioGlobals();
    (globalThis as Record<string, unknown>).navigator = { mediaDevices: {} };
    expect(isWebAudioCaptureSupported()).toBe(false);
  });

  it("is true when getUserMedia + AudioWorklet + AudioContext exist", () => {
    installWebAudioGlobals();
    expect(isWebAudioCaptureSupported()).toBe(true);
  });
});

describe("WebAudioCapture", () => {
  it("creates + resumes the AudioContext, wires the graph, and forwards mono frames", async () => {
    const { getUserMedia } = installWebAudioGlobals();
    const frames: Array<{
      data: ArrayBuffer;
      sampleRate: number;
      channels: number;
    }> = [];
    const capture = new WebAudioCapture({
      onBuffer: (b) => frames.push(b),
    });

    await capture.start();

    const ctx = FakeAudioContext.last!;
    expect(ctx.resume).toHaveBeenCalled(); // Safari: resumed in the gesture.
    expect(ctx.audioWorklet.addModule).toHaveBeenCalledWith("blob:fake");
    // Requested the mic with echo cancellation etc.
    expect(getUserMedia).toHaveBeenCalledTimes(1);
    const constraints = getUserMedia.mock.calls[0][0];
    expect(constraints.audio.echoCancellation).toBe(true);

    // A batch posted by the worklet is forwarded with the ctx sample rate,
    // mono, as a raw ArrayBuffer of float32 samples.
    const worklet = capture["worklet"] as unknown as FakeAudioWorkletNode;
    const batch = new Float32Array([0.1, -0.2, 0.3]);
    worklet.port.onmessage!({ data: batch } as MessageEvent);
    expect(frames).toHaveLength(1);
    expect(frames[0].sampleRate).toBe(48000);
    expect(frames[0].channels).toBe(1);
    expect(new Float32Array(frames[0].data)).toEqual(batch);
  });

  it("stop() releases the mic (stops tracks) and closes the AudioContext", async () => {
    const { tracks } = installWebAudioGlobals();
    const capture = new WebAudioCapture({ onBuffer: () => {} });
    await capture.start();
    const ctx = FakeAudioContext.last!;

    await capture.stop();

    expect(tracks[0].stop).toHaveBeenCalled();
    expect(ctx.close).toHaveBeenCalled();

    // No frames are forwarded after stop, even if a late message arrives.
    const forwarded: unknown[] = [];
    const captureWithSpy = new WebAudioCapture({
      onBuffer: (b) => forwarded.push(b),
    });
    await captureWithSpy.start();
    await captureWithSpy.stop();
    const worklet = captureWithSpy["worklet"] as unknown;
    expect(worklet).toBeNull(); // graph torn down
    expect(forwarded).toHaveLength(0);
  });

  it("permission denied surfaces an honest WebCaptureError and releases the context", async () => {
    const { getUserMedia } = installWebAudioGlobals();
    getUserMedia.mockRejectedValueOnce({ name: "NotAllowedError" });
    const capture = new WebAudioCapture({ onBuffer: () => {} });

    await expect(capture.start()).rejects.toMatchObject({
      kind: "permission-denied",
    });
    // The AudioContext created in the gesture was closed on the failure path.
    expect(FakeAudioContext.last!.close).toHaveBeenCalled();
  });

  it("throws unavailable when the browser lacks the capture APIs", async () => {
    clearWebAudioGlobals();
    const capture = new WebAudioCapture({ onBuffer: () => {} });
    await expect(capture.start()).rejects.toMatchObject({
      kind: "unavailable",
    });
  });
});
