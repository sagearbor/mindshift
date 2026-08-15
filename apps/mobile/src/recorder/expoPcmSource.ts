import { AudioModule } from "expo-audio";
import type { AudioStream, AudioStreamBuffer } from "expo-audio";
import { downmixToMono, float32ToInt16 } from "../utils/audio";
import type { PcmFrame, PcmSource } from "./pcmSource";

/** The rate we ASK for — the server's native format. The hardware may
 *  deliver another rate; frames report the true one and the engine records
 *  it honestly (see PcmFrame.sampleRate). */
const REQUESTED_SAMPLE_RATE = 16000;

/**
 * PcmSource over expo-audio's realtime AudioStream — the same native capture
 * engine the live-coach WebSocket path uses (via its hook), driven here
 * imperatively so the recording engine owns the lifecycle without React.
 *
 * Normalization mirrors the live path: interleaved multi-channel buffers are
 * downmixed to mono and float32 is converted to int16. Unlike the live path
 * there is NO resampling — a recording must keep every sample the hardware
 * gave us at the rate it actually used, with the header telling the truth.
 */
export class ExpoPcmSource implements PcmSource {
  private stream: AudioStream | null = null;
  private subscriptions: { remove(): void }[] = [];
  /** Last status reported by the native audioStreamStatus event. */
  private streaming = false;

  async start(onFrame: (frame: PcmFrame) => void): Promise<void> {
    const stream = new AudioModule.AudioStream({
      sampleRate: REQUESTED_SAMPLE_RATE,
      channels: 1,
      encoding: "float32",
    });
    this.subscriptions.push(
      stream.addListener("audioStreamBuffer", (buffer: AudioStreamBuffer) => {
        let samples: Float32Array = new Float32Array(buffer.data);
        if (buffer.channels > 1) {
          samples = downmixToMono(samples, buffer.channels);
        }
        onFrame({
          samples: float32ToInt16(samples),
          sampleRate: buffer.sampleRate,
        });
      }),
    );
    this.subscriptions.push(
      stream.addListener("audioStreamStatus", (status) => {
        this.streaming = status.isStreaming;
      }),
    );
    this.stream = stream;
    await stream.start();
    this.streaming = true;
  }

  stop(): void {
    for (const sub of this.subscriptions) {
      try {
        sub.remove();
      } catch {
        // Listener already gone.
      }
    }
    this.subscriptions = [];
    const stream = this.stream;
    this.stream = null;
    this.streaming = false;
    if (stream) {
      try {
        stream.stop();
      } catch {
        // A stream invalidated by the OS may throw on stop; nothing to free.
      }
    }
  }

  isCapturing(): boolean {
    const stream = this.stream;
    if (!stream) return false;
    // Prefer the live native property; fall back to the last status event
    // if the property is unavailable on some platform version.
    return typeof stream.isStreaming === "boolean"
      ? stream.isStreaming
      : this.streaming;
  }
}
