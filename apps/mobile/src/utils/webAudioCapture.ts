/**
 * Web microphone capture for the live-audio pipeline.
 *
 * expo-audio ships no web recorder, so on the web build (desktop Chrome /
 * Firefox / Edge, Android Chrome, iOS Safari) we capture the microphone here
 * with the Web Audio API and feed the SAME pipeline the native path uses: this
 * module hands the caller raw Float32 frames plus the actual sample rate, and
 * the hook downmixes / resamples them to 16 kHz int16 mono and streams them
 * over the identical WebSocket. The backend cannot tell native from web.
 *
 * Why these choices:
 *  - AudioWorklet, not the deprecated ScriptProcessorNode and not
 *    MediaRecorder (which only emits webm/opus — the wrong wire format). The
 *    worklet pulls raw Float32 at the AudioContext's native rate and posts
 *    mono frames to the main thread.
 *  - The worklet module is loaded from a runtime Blob URL, so there is no
 *    Metro / bundler asset wiring to get wrong — it behaves identically in
 *    every browser and under `expo export -p web`.
 *  - iOS Safari: the AudioContext must be created AND resumed synchronously
 *    inside the user gesture (Safari creates it "suspended" and refuses to
 *    resume it later otherwise). `start()` does both at the top, and it is
 *    reached synchronously from the Start button's onPress — no macrotask
 *    boundary intervenes. Safari also picks its own sample rate (often
 *    44.1 kHz and it ignores a requested rate), so we report the real
 *    `ctx.sampleRate` and let the downstream resampler handle any rate.
 *
 * All access to browser globals is lazy (inside methods/functions), so this
 * module is import-safe in a Node/Jest environment where those globals are
 * absent.
 */

/**
 * One captured PCM frame handed to the caller. Structurally identical to
 * expo-audio's `AudioStreamBuffer` so the same `onBuffer` handler consumes
 * both native and web frames: `data` is float32 samples (mono here), and
 * `sampleRate` is the hardware/context rate — NOT assumed to be 16 kHz.
 */
export interface WebCaptureBuffer {
  /** Mono float32 PCM samples' backing ArrayBuffer, values in [-1, 1]. */
  data: ArrayBuffer;
  /** The AudioContext's actual sample rate (e.g. 44100 or 48000). */
  sampleRate: number;
  /** Always 1 — the worklet downmixes to mono before posting. */
  channels: number;
  /** Seconds since capture started (best-effort; unused downstream). */
  timestamp: number;
}

export type WebCaptureErrorKind =
  | "permission-denied"
  | "no-microphone"
  | "unavailable";

/** A capture failure with a classified, honest reason (never a fake success). */
export class WebCaptureError extends Error {
  readonly kind: WebCaptureErrorKind;
  constructor(kind: WebCaptureErrorKind, message: string) {
    super(message);
    this.name = "WebCaptureError";
    this.kind = kind;
  }
}

/**
 * Classify a getUserMedia / Web Audio failure into an honest state. Uses the
 * DOMException `name` (portable across browsers) rather than `instanceof`,
 * which is unreliable across realms.
 */
export function classifyCaptureError(err: unknown): WebCaptureError {
  const name =
    err && typeof err === "object" && "name" in err
      ? String((err as { name: unknown }).name)
      : "";
  const message =
    err && typeof err === "object" && "message" in err
      ? String((err as { message: unknown }).message)
      : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return new WebCaptureError(
      "permission-denied",
      message || "Microphone permission denied",
    );
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return new WebCaptureError(
      "no-microphone",
      message || "No microphone available",
    );
  }
  if (err instanceof WebCaptureError) return err;
  return new WebCaptureError(
    "unavailable",
    message || "Microphone capture is unavailable",
  );
}

/**
 * Whether this browser can capture audio: it needs both `getUserMedia` and
 * `AudioWorklet`. Older browsers (or non-secure contexts, where
 * `navigator.mediaDevices` is undefined) fail this and get an honest
 * "can't capture audio" message instead of a broken session.
 */
export function isWebAudioCaptureSupported(): boolean {
  if (typeof navigator === "undefined" || !navigator.mediaDevices) return false;
  if (typeof navigator.mediaDevices.getUserMedia !== "function") return false;
  if (typeof AudioWorkletNode === "undefined") return false;
  const g = globalThis as Record<string, unknown>;
  return (
    typeof g.AudioContext !== "undefined" ||
    typeof g.webkitAudioContext !== "undefined"
  );
}

/**
 * AudioWorklet processor source, loaded at runtime via a Blob URL. It runs in
 * the audio-rendering realm (128-sample quanta), downmixes to mono, batches
 * to ~BATCH_SIZE samples to keep postMessage traffic modest (~50 msgs/s at
 * 48 kHz), and posts each batch as a Float32Array to the main thread.
 */
const WORKLET_SOURCE = `
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const batchSize =
      (options && options.processorOptions && options.processorOptions.batchSize) || 2048;
    this._batch = new Float32Array(batchSize);
    this._n = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channelCount = input.length;
    const frames = input[0].length;
    for (let i = 0; i < frames; i++) {
      let sum = 0;
      for (let c = 0; c < channelCount; c++) sum += input[c][i];
      this._batch[this._n++] = sum / channelCount;
      if (this._n === this._batch.length) {
        // Copy out (slice) so we hand ownership of a fresh buffer to the main
        // thread and keep filling this._batch without a race.
        this.port.postMessage(this._batch.slice(0, this._n));
        this._n = 0;
      }
    }
    return true;
  }
}
registerProcessor('pcm-capture', PcmCaptureProcessor);
`;

const BATCH_SIZE = 2048;

interface WebAudioCaptureOptions {
  onBuffer: (buffer: WebCaptureBuffer) => void;
  /** Overridable for tests; production uses the real getUserMedia constraints. */
  constraints?: MediaStreamConstraints;
}

/**
 * Captures the microphone on the web and streams mono Float32 batches to
 * `onBuffer`. Lifecycle: `start()` (must be called from a user gesture) then
 * `stop()`. `stop()` is idempotent and releases the mic (stops every track)
 * synchronously, then closes the AudioContext.
 */
export class WebAudioCapture {
  private readonly onBuffer: (buffer: WebCaptureBuffer) => void;
  private readonly constraints: MediaStreamConstraints;
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private worklet: AudioWorkletNode | null = null;
  /** Set the instant stop() is called so late async steps in start() bail. */
  private stopped = false;

  constructor(options: WebAudioCaptureOptions) {
    this.onBuffer = options.onBuffer;
    this.constraints = options.constraints ?? {
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
      video: false,
    };
  }

  /**
   * Begin capture. MUST be invoked synchronously from a user gesture: the
   * AudioContext is created and resumed first (the Safari requirement) before
   * any await. Throws a {@link WebCaptureError} on permission denial, missing
   * hardware, or an unsupported browser.
   */
  async start(): Promise<void> {
    if (!isWebAudioCaptureSupported()) {
      throw new WebCaptureError(
        "unavailable",
        "This browser cannot capture audio (needs getUserMedia + AudioWorklet).",
      );
    }

    // 1. Create + resume the AudioContext synchronously in the gesture. Safari
    //    starts it "suspended" and will not resume it outside a gesture.
    const g = globalThis as Record<string, unknown>;
    const Ctor = (g.AudioContext ?? g.webkitAudioContext) as {
      new (): AudioContext;
    };
    const ctx = new Ctor();
    this.ctx = ctx;
    // Kick off resume() from within the gesture; await it later.
    const resumePromise: Promise<void> =
      ctx.state === "suspended" ? ctx.resume() : Promise.resolve();

    // 2. Prompt for and acquire the microphone.
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(this.constraints);
    } catch (err) {
      await this.stop();
      throw classifyCaptureError(err);
    }
    if (this.stopped) {
      // stop() landed while the permission prompt was open: release and bail.
      stream.getTracks().forEach((t) => t.stop());
      await this.stop();
      return;
    }
    this.stream = stream;

    // 3. Load the worklet module (runtime Blob URL) and wire the graph.
    try {
      const blob = new Blob([WORKLET_SOURCE], {
        type: "application/javascript",
      });
      const url = URL.createObjectURL(blob);
      try {
        await ctx.audioWorklet.addModule(url);
      } finally {
        URL.revokeObjectURL(url);
      }
      await resumePromise;
      if (this.stopped) {
        await this.stop();
        return;
      }

      const source = ctx.createMediaStreamSource(stream);
      const worklet = new AudioWorkletNode(ctx, "pcm-capture", {
        processorOptions: { batchSize: BATCH_SIZE },
      });
      worklet.port.onmessage = (event: MessageEvent) => {
        if (this.stopped) return;
        const samples = event.data as Float32Array;
        this.onBuffer({
          data: samples.buffer as ArrayBuffer,
          sampleRate: ctx.sampleRate,
          channels: 1,
          timestamp: ctx.currentTime,
        });
      };
      source.connect(worklet);
      // The worklet writes no output (returns silence), but connecting it to
      // the destination guarantees the graph is pulled so process() runs. No
      // audible feedback results because nothing is copied to the output.
      worklet.connect(ctx.destination);
      this.source = source;
      this.worklet = worklet;
    } catch (err) {
      await this.stop();
      throw new WebCaptureError(
        "unavailable",
        err instanceof Error && err.message
          ? err.message
          : "Failed to start the audio capture graph.",
      );
    }
  }

  /**
   * Stop capture and release the microphone. Idempotent. Tracks are stopped
   * synchronously (the mic indicator goes off immediately) before the async
   * AudioContext close, so callers that fire-and-forget still release the mic
   * at once.
   */
  async stop(): Promise<void> {
    this.stopped = true;
    if (this.worklet) {
      try {
        this.worklet.port.onmessage = null;
        this.worklet.disconnect();
      } catch {
        // Node may already be disconnected.
      }
      this.worklet = null;
    }
    if (this.source) {
      try {
        this.source.disconnect();
      } catch {
        // Node may already be disconnected.
      }
      this.source = null;
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        try {
          track.stop();
        } catch {
          // Track may already be stopped.
        }
      }
      this.stream = null;
    }
    if (this.ctx) {
      const ctx = this.ctx;
      this.ctx = null;
      try {
        await ctx.close();
      } catch {
        // Context may already be closed.
      }
    }
  }
}
