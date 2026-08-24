/**
 * Voice activity detection for the on-device fast loop.
 *
 * Two detectors share one `FrameVad` contract so the segmenter doesn't care
 * which is running:
 *
 * - `SileroVad` — the production detector: silero_vad.onnx (v6, MIT, bundled
 *   at assets/models/) through the `OnnxSession` seam, with the official I/O
 *   (input [1, 576] = 64 context samples + 512 new samples @ 16 kHz, state
 *   [2, 1, 128], `sr` int64) and the reference hysteresis (on at 0.5, off at
 *   0.35 — Silero's `threshold` / `neg_threshold = threshold - 0.15`).
 * - `EnergyVad` — the server's energy VAD (server/watch/diarize.py +
 *   watch/vectors.rms_dbfs) ported verbatim. It is what the golden vectors in
 *   server/tests/fixtures/policy_vectors/vad_segments.json pin, so it stays
 *   here as the parity reference and as the fallback when the ONNX model
 *   can't load.
 */
import type { OnnxSession } from "./ort";
import { float32Tensor, int64Scalar } from "./ort";

export const SILERO_SAMPLE_RATE = 16000;
/** Samples of NEW audio per Silero call at 16 kHz (32 ms). */
export const SILERO_CHUNK_SAMPLES = 512;
/** Samples of the previous chunk prepended as context (the v5+ model input
 *  is [1, 64 + 512]; feeding only 512 silently degrades accuracy). */
export const SILERO_CONTEXT_SAMPLES = 64;
export const SILERO_STATE_DIMS = [2, 1, 128] as const;
export const SILERO_SPEECH_ON = 0.5;
export const SILERO_SPEECH_OFF = 0.35;

/** A detector the segmenter can drive: fixed frame size, one verdict per frame. */
export interface FrameVad {
  /** Samples per frame at 16 kHz. */
  readonly frameSamples: number;
  /** Speech / not-speech for one frame of exactly `frameSamples` samples. */
  isSpeech(frame: Float32Array): Promise<boolean>;
  /** Forget all state (session boundary). */
  reset(): void;
}

/**
 * Two-threshold hysteresis over a probability stream: enters "speech" when
 * p >= on, leaves when p < off. Pure — tested on its own.
 */
export class SpeechGate {
  private speaking = false;
  constructor(
    private readonly on = SILERO_SPEECH_ON,
    private readonly off = SILERO_SPEECH_OFF,
  ) {}
  update(p: number): boolean {
    if (this.speaking) {
      if (p < this.off) this.speaking = false;
    } else if (p >= this.on) {
      this.speaking = true;
    }
    return this.speaking;
  }
  reset() {
    this.speaking = false;
  }
}

export class SileroVad implements FrameVad {
  readonly frameSamples = SILERO_CHUNK_SAMPLES;
  private state = new Float32Array(2 * 128);
  private context = new Float32Array(SILERO_CONTEXT_SAMPLES);
  private readonly input = new Float32Array(
    SILERO_CONTEXT_SAMPLES + SILERO_CHUNK_SAMPLES,
  );
  private readonly gate: SpeechGate;
  /** The most recent raw probability (for logging/UI). */
  lastProbability = 0;

  constructor(
    private readonly session: OnnxSession,
    gate: SpeechGate = new SpeechGate(),
  ) {
    this.gate = gate;
  }

  /** Raw speech probability for one 512-sample chunk (state carried). */
  async probability(chunk: Float32Array): Promise<number> {
    if (chunk.length !== SILERO_CHUNK_SAMPLES) {
      throw new Error(
        `SileroVad: expected ${SILERO_CHUNK_SAMPLES} samples, got ${chunk.length}`,
      );
    }
    this.input.set(this.context, 0);
    this.input.set(chunk, SILERO_CONTEXT_SAMPLES);
    const out = await this.session.run({
      input: float32Tensor(this.input, [1, this.input.length]),
      state: float32Tensor(this.state, SILERO_STATE_DIMS),
      sr: int64Scalar(SILERO_SAMPLE_RATE),
    });
    const stateN = out.stateN ?? out[this.session.outputNames[1]];
    if (stateN) this.state = Float32Array.from(stateN.data as Float32Array);
    // Next call's context = the tail of this chunk.
    this.context = chunk.slice(chunk.length - SILERO_CONTEXT_SAMPLES);
    const prob = out.output ?? out[this.session.outputNames[0]];
    this.lastProbability = prob ? Number(prob.data[0]) : 0;
    return this.lastProbability;
  }

  async isSpeech(frame: Float32Array): Promise<boolean> {
    return this.gate.update(await this.probability(frame));
  }

  reset() {
    this.state = new Float32Array(2 * 128);
    this.context = new Float32Array(SILERO_CONTEXT_SAMPLES);
    this.gate.reset();
    this.lastProbability = 0;
  }
}

// ---------------------------------------------------------------------------
// Energy VAD — port of server/watch/diarize.py's per-frame decision.
// ---------------------------------------------------------------------------

/** Same silence floor the streaming VectorEngine and diarize.py use. */
export const SILENCE_FLOOR_DBFS = -45.0;
export const ENERGY_FRAME_SECONDS = 0.25;
export const INT16_FULL_SCALE = 32768;

/** `20*log10(rms/32768)` for int16-scaled PCM; -Infinity for silence
 *  (watch/vectors.py::rms_dbfs). Accepts int16 values in any numeric array. */
export function rmsDbfsInt16(samples: ArrayLike<number>): number {
  if (samples.length === 0) return -Infinity;
  let acc = 0;
  for (let i = 0; i < samples.length; i++) acc += samples[i] * samples[i];
  const rms = Math.sqrt(acc / samples.length);
  if (rms <= 0) return -Infinity;
  return 20 * Math.log10(rms / INT16_FULL_SCALE);
}

/** Strict `>`: a frame AT the floor is silence (fixture contract). */
export function energyIsSpeech(
  int16Frame: ArrayLike<number>,
  floorDbfs = SILENCE_FLOOR_DBFS,
): boolean {
  return rmsDbfsInt16(int16Frame) > floorDbfs;
}

/** FrameVad over the energy rule, for float [-1, 1] frames. */
export class EnergyVad implements FrameVad {
  readonly frameSamples: number;
  constructor(
    private readonly floorDbfs = SILENCE_FLOOR_DBFS,
    frameSeconds = ENERGY_FRAME_SECONDS,
    sampleRate = SILERO_SAMPLE_RATE,
  ) {
    this.frameSamples = Math.max(1, Math.round(frameSeconds * sampleRate));
  }
  async isSpeech(frame: Float32Array): Promise<boolean> {
    // Scale to int16 units so the dBFS math is byte-identical to the server.
    const scaled = new Float32Array(frame.length);
    for (let i = 0; i < frame.length; i++) scaled[i] = frame[i] * INT16_FULL_SCALE;
    return energyIsSpeech(scaled, this.floorDbfs);
  }
  reset() {
    // Stateless.
  }
}
