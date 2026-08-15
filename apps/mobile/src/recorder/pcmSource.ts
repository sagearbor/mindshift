/**
 * The capture seam of the v2 (gapless) recorder engine.
 *
 * A PcmSource is a continuous microphone PCM stream: started once, it delivers
 * mono int16 frames until stopped — it is never stopped/restarted to rotate
 * files (that was v1's design and the origin of its ~0.2 s per-rotation gap).
 * Production implements this over expo-audio's realtime AudioStream (the same
 * native capture the live-coach path uses); tests implement it with a fake
 * that pushes known signals.
 */

/** One captured chunk of audio: mono int16 samples at the TRUE rate the
 *  hardware delivered. The rate is a per-frame fact, not an assumption —
 *  if the device ignores our 16 kHz request the engine records honestly at
 *  whatever this says (and stamps it into the WAV header). */
export interface PcmFrame {
  samples: Int16Array;
  sampleRate: number;
}

export interface PcmSource {
  /** Begin capturing; `onFrame` fires for every buffer until stop(). */
  start(onFrame: (frame: PcmFrame) => void): Promise<void>;
  /** Stop capturing and release native resources. Must never throw. */
  stop(): void;
  /** Whether the platform still reports the stream as live — polled by the
   *  engine to detect the OS silently taking the microphone. */
  isCapturing(): boolean;
}

/** A fresh source per capture run (one at session start, one per resume
 *  after an interruption), mirroring v1's RecorderFactory. */
export type PcmSourceFactory = () => PcmSource;
