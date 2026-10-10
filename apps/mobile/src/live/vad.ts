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
// Input gain normalisation (AGC) in front of the VAD.
// ---------------------------------------------------------------------------
//
// Silero has a level floor: a talker far from the phone (quiet room, phone on
// the table) stays under its threshold however long they talk. Measured
// 2026-10-10 on the recording corpus (frame-level Silero vs human ground-truth
// speech times, overnight tuning agent D): SBC033 (DEV) 64% -> 86% of speech
// detected; the same clip attenuated 18 dB 0% -> 83%; louder DEV clips +1-3
// points; false speech outside ground truth +0.00-0.04 min per 5 min, and
// none at all on synthetic pink/white/hum noise at -60..-35 dBFS (Silero
// rejects amplified stationary noise). The two quiet held-out recordings
// (SBC042 at -36 dBFS, AMI ES2003a at -46 dBFS) were 15% / 23% detected —
// a real VAD miss, not reference-STT sparsity: lowering the threshold to
// 0.25 alone only reached 24% / 25%.
//
// Only the VAD sees the gained audio: speaker-ID and prosody keep the raw
// frames (loudness drives the heat lane and the alert buzz).

export interface AgcConfig {
  /** Level the speech envelope is lifted to (dBFS RMS). */
  targetDbfs: number;
  /** Ceiling on the gain (dB) — bounds how much room noise is amplified. */
  maxGainDb: number;
  /** Envelope attack: fraction of the way to a louder frame per frame. */
  attack: number;
  /** Envelope release (dB per second) after the talker goes quiet. */
  releaseDbPerSec: number;
}

export const VAD_AGC_DEFAULTS: AgcConfig = {
  targetDbfs: -16,
  maxGainDb: 30,
  attack: 0.5,
  releaseDbPerSec: 3,
};

/** Production switch for the gain stage in front of the VAD (buildVad).
 *  DARK for now: turning it on changes every pinned replay fixture
 *  (maggiano3's baseline included), which needs the owner's sign-off.
 *  Replays measure it with MINDSHIFT_VAD_AGC=1 (replay/tuning.ts). */
export const VAD_AGC_ENABLED = false;

/**
 * Streaming peak-envelope AGC over fixed frames: the envelope rises quickly to
 * a louder frame and falls slowly (`releaseDbPerSec`); gain = target /
 * envelope, clamped to [1, maxGain] — it only ever amplifies, never attenuates.
 */
export class GainNormalizer {
  private env: number;
  private g = 1;
  private readonly target: number;
  private readonly maxGain: number;
  constructor(
    private readonly cfg: AgcConfig = VAD_AGC_DEFAULTS,
    private readonly frameSeconds = SILERO_CHUNK_SAMPLES / SILERO_SAMPLE_RATE,
  ) {
    this.target = 10 ** (cfg.targetDbfs / 20);
    this.maxGain = 10 ** (cfg.maxGainDb / 20);
    this.env = this.target;
  }
  /** Current linear gain (1 = unity). */
  get gain(): number {
    return this.g;
  }
  process(frame: Float32Array): Float32Array {
    let acc = 0;
    for (let i = 0; i < frame.length; i++) acc += frame[i] * frame[i];
    const r = Math.max(Math.sqrt(acc / Math.max(1, frame.length)), 1e-7);
    if (r > this.env) {
      this.env += this.cfg.attack * (r - this.env);
    } else {
      const rel = 10 ** ((-this.cfg.releaseDbPerSec * this.frameSeconds * frame.length) / SILERO_CHUNK_SAMPLES / 20);
      this.env = Math.max(this.env * rel, r);
    }
    this.g = Math.min(Math.max(this.target / this.env, 1), this.maxGain);
    const out = new Float32Array(frame.length);
    for (let i = 0; i < frame.length; i++) {
      const v = frame[i] * this.g;
      out[i] = v > 1 ? 1 : v < -1 ? -1 : v;
    }
    return out;
  }
  reset() {
    this.env = this.target;
    this.g = 1;
  }
}

/** A FrameVad that runs its input through a {@link GainNormalizer} first. */
export class AgcVad implements FrameVad {
  readonly frameSamples: number;
  private readonly agc: GainNormalizer;
  constructor(
    readonly inner: FrameVad,
    cfg: AgcConfig = VAD_AGC_DEFAULTS,
  ) {
    this.frameSamples = inner.frameSamples;
    this.agc = new GainNormalizer(cfg, inner.frameSamples / SILERO_SAMPLE_RATE);
  }
  get gain(): number {
    return this.agc.gain;
  }
  /** Pass-through of the inner detector's probability (fast loop logging). */
  get lastProbability(): number | undefined {
    return (this.inner as { lastProbability?: number }).lastProbability;
  }
  isSpeech(frame: Float32Array): Promise<boolean> {
    return this.inner.isSpeech(this.agc.process(frame));
  }
  reset() {
    this.agc.reset();
    this.inner.reset();
  }
}

/** The detector under any wrappers (capability reporting). */
export function baseVad(vad: FrameVad): FrameVad {
  let v = vad;
  while (v instanceof AgcVad) v = v.inner;
  return v;
}

/** Production wrapping: the gain stage when {@link VAD_AGC_ENABLED}. The
 *  energy fallback is left alone (its fixed dBFS floor IS a level rule). */
export function withVadAgc(vad: FrameVad, enabled = VAD_AGC_ENABLED): FrameVad {
  return enabled && vad instanceof SileroVad ? new AgcVad(vad) : vad;
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
