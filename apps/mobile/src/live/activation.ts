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

export const ACTIVATION_FEATURE_ORDER = [
  "f0_mean",
  "f0_max",
  "f0_sd",
  "energy_db_mean",
  "energy_db_max",
  "energy_db_sd",
  "voiced_duration_s",
  "unvoiced_duration_s",
] as const;

export type ActivationFeatureName = (typeof ACTIVATION_FEATURE_ORDER)[number];
export type ActivationFeatures = Record<ActivationFeatureName, number>;

/** Standardization + weights exported by tmp/ravdess/scripts/train.py
 *  (feature order as above). `p = sigmoid(dot(coef, (x - mean) / sd) + b)`. */
export const ACTIVATION_MODEL = {
  mean: [250.77502358385598, 385.9955771203342, 79.15507121651372, -67.15615811543704, -28.615591991404326, 26.783605241498552, 2.003527777777778, 1.663048611111108],
  sd: [53.59427832691743, 38.11306566810214, 33.073245855416374, 8.929168684081308, 9.336374872676995, 5.438160601420517, 0.6053191759592208, 0.6259820709832797],
  coefficients: [0.37395166803627944, -0.12784657757543236, 0.009041577652365455, 0.7346941538685611, 0.32470324990355787, 0.26952791045736174, 1.176260064190532, 1.2818648499002512],
  intercept: -0.137409482878309,
} as const;

/** ≈ the RAVDESS clip length the durations were trained on (voiced 2.0 s +
 *  unvoiced 1.66 s of frames). */
export const ACTIVATION_WINDOW_SECONDS = 3.7;

/** Probability -> level 1..3 (descending thresholds, like YELLING_LEVELS).
 *  Deliberately conservative; RAVDESS "strong" is stage-level intensity. */
export const ACTIVATION_LEVELS: [number, number][] = [
  [0.96, 3],
  [0.88, 2],
  [0.75, 1],
];

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

function assemble(f0s: number[], energies: number[], nFrames: number, hopSeconds: number): ActivationFeatures {
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
  };
}

/** Mirror of tmp/ravdess/scripts/extract_features.py::extract_features over
 *  the given samples (no windowing here). Null when shorter than one frame. */
export function activationFeatures(samples: Float32Array, sr: number): ActivationFeatures | null {
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
  features: ActivationFeatures;
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
  const features = assemble(f0s, energies, nFrames, hop / sr);
  const probability = activationProbability(features);
  return { probability, level: activationLevel(probability), features };
}
