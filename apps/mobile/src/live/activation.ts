/**
 * Vocal activation — "how worked-up does the voice sound" — the CANDOR
 * paper's vocal-intensity classifier (Reece et al. 2023: logistic regression
 * over prosody features, trained on RAVDESS normal-vs-strong intensity),
 * retrained 2026-09-05 on the public RAVDESS speech set with OUR feature
 * definitions (server/prosody.py frame F0 + rms energy, 40 ms / 10 ms
 * frames; tmp/ravdess/analysis/activation_model.json). Grouped
 * cross-validation by actor (no voice in both train and test): ROC-AUC
 * 0.82, accuracy 0.75.
 *
 * Features are measured over a FIXED-length window — the last
 * ACTIVATION_WINDOW_SECONDS of the turn — because the two strongest
 * coefficients are voiced/unvoiced DURATION: trained on ~3.7 s clips they
 * partly encode clip length, so a fixed window keeps the phone's inputs on
 * the training scale. Honest caveat: RAVDESS is acted speech; the ladder
 * thresholds below are first guesses to tune from on-device telemetry.
 *
 * Ships DARK: the probability/level is recorded per turn and shown in
 * Developer mode; it drives the nudge policy only when the fast loop is
 * built with `activationNudges: true`.
 */
import { FRAME_MS, HOP_MS, frameF0, rmsEnergy } from "./prosody";

/**
 * The MODEL's inputs (v2, 2026-09-07) — every one invariant to recording gain
 * and to clip length. v1 fed the model absolute energy and absolute duration,
 * and those turned out to be the two biggest terms in its logit: RAVDESS sits
 * at about -67 dBFS while a phone turn sits at about -19, worth +3.96 of logit
 * on its own, and a real turn is nearly all speech while a RAVDESS clip
 * carries a second of silence at each end, worth another +2.32. It was reading
 * the microphone, not the person, which is why it called the owner's calm
 * narration "worked up" on his own recording.
 */
export const ACTIVATION_FEATURE_ORDER = [
  "f0_sd",
  "f0_range_ratio",
  "energy_dynamic_range",
  "energy_db_sd",
  "voiced_fraction",
] as const;

export type ActivationFeatureName = (typeof ACTIVATION_FEATURE_ORDER)[number];
export type ActivationFeatures = Record<ActivationFeatureName, number>;

/** The RAW per-window measurements. Kept in full because they are what
 *  Developer mode shows and what any future re-fit would train on; the model
 *  itself only ever sees the invariant combinations derived from them. */
export interface ActivationRaw {
  f0_mean: number;
  f0_max: number;
  f0_sd: number;
  energy_db_mean: number;
  energy_db_max: number;
  energy_db_sd: number;
  voiced_duration_s: number;
  unvoiced_duration_s: number;
  voiced_fraction: number;
}

/**
 * Raw measurements -> the model's invariant inputs. Gain cancels in
 * `energy_dynamic_range` (a difference of two dB values) and in
 * `energy_db_sd`; length cancels in `voiced_fraction` and in the pitch ratio.
 */
export function activationVector(raw: ActivationRaw): ActivationFeatures {
  return {
    f0_sd: raw.f0_sd,
    f0_range_ratio: raw.f0_mean > 1e-6 ? raw.f0_max / raw.f0_mean : 0,
    energy_dynamic_range: raw.energy_db_max - raw.energy_db_mean,
    energy_db_sd: raw.energy_db_sd,
    voiced_fraction: raw.voiced_fraction,
  };
}

/** Standardization + weights exported by tmp/ravdess/scripts/train.py
 *  (feature order as above). `p = sigmoid(dot(coef, (x - mean) / sd) + b)`. */
export const ACTIVATION_MODEL = {
  mean: [80.3987837390671, 1.65198407418653, 38.811056812646385, 26.899925983084394, 0.5657735200324732],
  sd: [36.566149487819395, 0.47999201455683604, 7.009310117337278, 5.633570889380242, 0.16417802740665846],
  coefficients: [0.13472634877186535, -1.898031972204167, 0.6580116865795932, 1.2722818199187464, 0.7878529610142222],
  intercept: -1.4243854172238426,
} as const;

/** ≈ the RAVDESS clip length the durations were trained on (voiced 2.0 s +
 *  unvoiced 1.66 s of frames). */
export const ACTIVATION_WINDOW_SECONDS = 3.7;

/**
 * Probability -> level 1..3 (descending thresholds, like YELLING_LEVELS).
 *
 * Rung 1 is not a taste call: it sits ABOVE the highest score any CALM clip
 * produced in out-of-fold cross-validation, so "never flags a calm turn" is a
 * property of the threshold rather than a hope. That safety costs reach — it
 * catches the clearest quarter of heated turns and lets the rest go — and that
 * is the right trade for a signal that rides alongside loudness, which already
 * fires on its own.
 */
export const ACTIVATION_LEVELS: [number, number][] = [[0.97, 3], [0.95, 2], [0.91, 1]];

/** Frames between event-loop yields in the async variant. */
export const ACTIVATION_YIELD_EVERY_FRAMES = 50;

/** The python extractor's per-frame log-energy: 20·log10(max(rms, 1e-6)). */
function frameEnergyDb(frame: Float32Array): number {
  return 20 * Math.log10(Math.max(rmsEnergy(frame), 1e-6));
}

function stats(values: number[]): { mean: number; max: number; sd: number } {
  if (values.length === 0) return { mean: 0, max: 0, sd: 0 };
  let sum = 0;
  let max = -Infinity;
  for (const v of values) {
    sum += v;
    if (v > max) max = v;
  }
  const mean = sum / values.length;
  let acc = 0;
  for (const v of values) acc += (v - mean) * (v - mean);
  return { mean, max, sd: Math.sqrt(acc / values.length) }; // population SD (numpy default)
}

function assemble(f0s: number[], energies: number[], nFrames: number, hopSeconds: number): ActivationRaw {
  const f = stats(f0s);
  const e = stats(energies);
  return {
    f0_mean: f.mean,
    f0_max: f.max,
    f0_sd: f.sd,
    energy_db_mean: e.mean,
    energy_db_max: e.max,
    energy_db_sd: e.sd,
    voiced_duration_s: f0s.length * hopSeconds,
    unvoiced_duration_s: (nFrames - f0s.length) * hopSeconds,
    voiced_fraction: nFrames > 0 ? f0s.length / nFrames : 0,
  };
}

/** The RAW measurements over the given samples (no windowing here). Null when
 *  shorter than one frame. `activationVector` turns these into the model's
 *  invariant inputs. */
export function activationRaw(samples: Float32Array, sr: number): ActivationRaw | null {
  const frameLen = Math.max(1, Math.floor((sr * FRAME_MS) / 1000));
  const hop = Math.max(1, Math.floor((sr * HOP_MS) / 1000));
  if (sr <= 0 || samples.length < frameLen) return null;
  const f0s: number[] = [];
  const energies: number[] = [];
  let nFrames = 0;
  for (let start = 0; start + frameLen <= samples.length; start += hop) {
    const frame = samples.subarray(start, start + frameLen);
    nFrames += 1;
    energies.push(frameEnergyDb(frame));
    const f0 = frameF0(frame, sr);
    if (f0 !== null) f0s.push(f0);
  }
  return assemble(f0s, energies, nFrames, hop / sr);
}

/** The model's invariant inputs over the given samples; null when unmeasurable. */
export function activationFeatures(samples: Float32Array, sr: number): ActivationFeatures | null {
  const raw = activationRaw(samples, sr);
  return raw ? activationVector(raw) : null;
}

export function activationProbability(features: ActivationFeatures): number {
  let logit = ACTIVATION_MODEL.intercept;
  ACTIVATION_FEATURE_ORDER.forEach((name, i) => {
    const z = (features[name] - ACTIVATION_MODEL.mean[i]) / ACTIVATION_MODEL.sd[i];
    logit += ACTIVATION_MODEL.coefficients[i] * z;
  });
  return 1 / (1 + Math.exp(-logit));
}

export function activationLevel(probability: number): number {
  for (const [threshold, level] of ACTIVATION_LEVELS) {
    if (probability >= threshold) return level;
  }
  return 0;
}

export interface TurnActivation {
  probability: number;
  level: number;
  /** The model's invariant inputs. */
  features: ActivationFeatures;
  /** The raw per-window measurements behind them — what Developer mode shows
   *  and what any future re-fit would train on. */
  raw: ActivationRaw;
}

export interface AsyncActivationOptions {
  yieldEvery?: number;
  sleep?: (ms: number) => Promise<void>;
  /** Window (seconds from the END of the turn); default ACTIVATION_WINDOW_SECONDS. */
  windowSeconds?: number;
}

const defaultSleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/**
 * `activationFeatures` + model over the LAST `windowSeconds` of the turn,
 * cooperative (yields every few dozen frames — the per-frame F0
 * autocorrelation is the one heavy loop, shared with the mic thread).
 * Same numbers as the sync path.
 */
export async function turnActivationAsync(
  samples: Float32Array,
  sr: number,
  opts: AsyncActivationOptions = {},
): Promise<TurnActivation | null> {
  const window = opts.windowSeconds ?? ACTIVATION_WINDOW_SECONDS;
  const capped =
    Number.isFinite(window) && window > 0 && samples.length > window * sr
      ? samples.subarray(samples.length - Math.round(window * sr))
      : samples;
  const frameLen = Math.max(1, Math.floor((sr * FRAME_MS) / 1000));
  const hop = Math.max(1, Math.floor((sr * HOP_MS) / 1000));
  if (sr <= 0 || capped.length < frameLen) return null;
  const yieldEvery = Math.max(1, opts.yieldEvery ?? ACTIVATION_YIELD_EVERY_FRAMES);
  const sleep = opts.sleep ?? defaultSleep;
  const f0s: number[] = [];
  const energies: number[] = [];
  let nFrames = 0;
  for (let start = 0; start + frameLen <= capped.length; start += hop) {
    const frame = capped.subarray(start, start + frameLen);
    nFrames += 1;
    energies.push(frameEnergyDb(frame));
    const f0 = frameF0(frame, sr);
    if (f0 !== null) f0s.push(f0);
    if (nFrames % yieldEvery === 0) await sleep(0);
  }
  const raw = assemble(f0s, energies, nFrames, hop / sr);
  const features = activationVector(raw);
  const probability = activationProbability(features);
  return { probability, level: activationLevel(probability), features, raw };
}
