/**
 * INSTANT TIER — angry vs happy-loud, on the phone, from 2 seconds of audio.
 *
 * The shipped instant signal is loudness over the speaker's own baseline. It
 * fires on a delighted shout exactly as hard as on an angry one: cross-corpus
 * (CREMA-D -> RAVDESS) angry-vs-happy AUC 0.713. That is the failure that
 * matters — a coach that buzzes when you are happy is a coach you turn off.
 *
 * openSMILE's 88 eGeMAPSv02 functionals with a linear probe reach 0.832 on the
 * same split, but openSMILE is a native dependency we will not ship and 88
 * descriptors is not something anyone hand-maintains. So we asked how few of
 * them we actually need, restricted to descriptors that are cheap and
 * well-defined to write from scratch (scripts/instant_tier_select.py): a
 * greedy-16 subset selected purely on CREMA-D speaker-grouped CV reaches
 * cross-corpus 0.842 — above the full set, because 88 correlated descriptors
 * fit CREMA-D's rooms rather than anger.
 *
 * This module is that subset, computed from raw PCM with nothing but
 * arithmetic: no native modules, no ONNX, no neural net. ~10 ms for a 2 s
 * window in node (see scripts/instant_tier_eval.ts --bench).
 *
 *   25 ms frames / 10 ms hop   energy, spectrum, MFCC, band slopes
 *   25 ms YIN / 20 ms hop      F0 and the voiced/unvoiced decision
 *   512-point radix-2 FFT      the app's first; pure TS, below
 *
 * WHAT THIS IS NOT: numerically identical to openSMILE. The frame windows,
 * the auditory model behind eGeMAPS "loudness", the mel filterbank and the
 * pitch tracker all differ, so the openSMILE fit's coefficients cannot simply
 * be lifted. instantTier.model.json instead holds a ridge fit onto that
 * reference model's LOGIT, computed from what this code measures over the
 * same corpora — the phone reproducing the model the bench chose, not
 * learning anger a second time. (Fitting the labels directly was measured
 * too: same angry-vs-happy AUC, worse agreement with the reference.)
 *
 * What we hold ourselves to is RANK agreement with that reference
 * (__tests__/instantTier.parity.test.ts, Spearman 0.942 over 147 clips) and
 * the real-audio cross-corpus AUC gate in __tests__/instantTier.eval.test.ts
 * (0.795 on 384 RAVDESS clips, floor 0.78).
 *
 * Honesty rule, same as prosody.ts: a feature that could not be measured is
 * `null`, never a fabricated number. A window with no voiced frame has no F0
 * percentile, and the scorer falls back to the training mean for that one
 * feature rather than inventing a pitch.
 */

import MODEL from "./instantTier.model.json";

// --- frame geometry --------------------------------------------------------
export const FRAME_MS = 25;
export const HOP_MS = 10;
export const PITCH_FRAME_MS = 25;
export const PITCH_HOP_MS = 20;
/** Shortest window we will score at all — below this the functionals
 *  (percentiles, rising/falling slopes) are noise. */
export const MIN_WINDOW_SEC = 0.5;
/** The window the model was fitted on. */
export const WINDOW_SEC = 2;

// --- pitch -----------------------------------------------------------------
export const F0_MIN_HZ = 55;
export const F0_MAX_HZ = 500;
/** YIN's absolute threshold on the cumulative mean normalized difference. */
export const YIN_THRESHOLD = 0.15;
/** Semitone reference eGeMAPS uses, kept so the numbers are comparable. */
export const SEMITONE_REF_HZ = 27.5;

// --- spectrum --------------------------------------------------------------
export const MEL_BANDS = 26;
export const MEL_LO_HZ = 20;
export const MEL_HI_HZ = 8000;
/** Intensity-to-loudness power law, as in the auditory spectrum eGeMAPS
 *  sums for its `loudness` descriptor. */
export const LOUDNESS_COMPRESSION = 0.33;

const EPS = 1e-10;

// ---------------------------------------------------------------------------
// Radix-2 FFT. Iterative, in-place, twiddles and the bit-reversal permutation
// precomputed once per size and cached — a 2 s window runs 200 of these.
// ---------------------------------------------------------------------------
class Fft {
  readonly n: number;
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;
  private readonly rev: Uint32Array;

  constructor(n: number) {
    if (n < 2 || (n & (n - 1)) !== 0) throw new Error(`FFT size must be a power of two, got ${n}`);
    this.n = n;
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((-2 * Math.PI * i) / n);
      this.sin[i] = Math.sin((-2 * Math.PI * i) / n);
    }
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
      this.rev[i] = r;
    }
  }

  /** In-place forward transform of (re, im), both length n. */
  forward(re: Float64Array, im: Float64Array): void {
    const { n, rev, cos, sin } = this;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const c = cos[k];
          const s = sin[k];
          const tre = re[j + half] * c - im[j + half] * s;
          const tim = re[j + half] * s + im[j + half] * c;
          re[j + half] = re[j] - tre;
          im[j + half] = im[j] - tim;
          re[j] += tre;
          im[j] += tim;
        }
      }
    }
  }
}

const fftCache = new Map<number, Fft>();
function fftFor(n: number): Fft {
  let f = fftCache.get(n);
  if (!f) {
    f = new Fft(n);
    fftCache.set(n, f);
  }
  return f;
}

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

// ---------------------------------------------------------------------------
// Mel filterbank (triangular, on the power spectrum), cached per geometry.
// ---------------------------------------------------------------------------
interface MelBank {
  /** For each band: the first bin, and the weight per bin from there. */
  start: Int32Array;
  weights: Float64Array[];
}
const melCache = new Map<string, MelBank>();

const hzToMel = (hz: number): number => 2595 * Math.log10(1 + hz / 700);
const melToHz = (mel: number): number => 700 * (10 ** (mel / 2595) - 1);

function melBank(sr: number, nfft: number): MelBank {
  const key = `${sr}:${nfft}`;
  const hit = melCache.get(key);
  if (hit) return hit;
  const bins = nfft / 2 + 1;
  const lo = hzToMel(MEL_LO_HZ);
  const hi = hzToMel(Math.min(MEL_HI_HZ, sr / 2));
  const edges: number[] = [];
  for (let i = 0; i < MEL_BANDS + 2; i++) {
    edges.push((melToHz(lo + ((hi - lo) * i) / (MEL_BANDS + 1)) * nfft) / sr);
  }
  const start = new Int32Array(MEL_BANDS);
  const weights: Float64Array[] = [];
  for (let b = 0; b < MEL_BANDS; b++) {
    const [l, c, r] = [edges[b], edges[b + 1], edges[b + 2]];
    const from = Math.max(0, Math.ceil(l));
    const to = Math.min(bins - 1, Math.floor(r));
    const w = new Float64Array(Math.max(0, to - from + 1));
    for (let k = from; k <= to; k++) {
      w[k - from] = k <= c ? (c > l ? (k - l) / (c - l) : 1) : r > c ? (r - k) / (r - c) : 1;
    }
    start[b] = from;
    weights.push(w);
  }
  const bank = { start, weights };
  melCache.set(key, bank);
  return bank;
}

// --- DCT-II matrix for the MFCCs (we only ever need c1..c4) ----------------
const DCT_ORDER = 4;
const dctCache = new Map<number, Float64Array[]>();
function dctBasis(bands: number): Float64Array[] {
  const hit = dctCache.get(bands);
  if (hit) return hit;
  const rows: Float64Array[] = [];
  for (let k = 1; k <= DCT_ORDER; k++) {
    const row = new Float64Array(bands);
    for (let b = 0; b < bands; b++) row[b] = Math.cos((Math.PI * k * (b + 0.5)) / bands);
    rows.push(row);
  }
  const scaled = rows.map((r) => {
    const out = new Float64Array(r.length);
    for (let i = 0; i < r.length; i++) out[i] = r[i] * Math.sqrt(2 / bands);
    return out;
  });
  dctCache.set(bands, scaled);
  return scaled;
}

// ---------------------------------------------------------------------------
// Small statistics helpers. All of them return `null` on an empty input
// rather than NaN or 0 — a functional over nothing is not zero.
// ---------------------------------------------------------------------------
function mean(xs: ArrayLike<number>): number | null {
  if (xs.length === 0) return null;
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s / xs.length;
}

function stddev(xs: ArrayLike<number>): number | null {
  if (xs.length < 2) return null;
  const m = mean(xs) as number;
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += (xs[i] - m) ** 2;
  return Math.sqrt(s / xs.length);
}

/** Linear-interpolated percentile over an already-sorted copy. */
function percentile(sorted: number[], p: number): number | null {
  if (sorted.length === 0) return null;
  if (sorted.length === 1) return sorted[0];
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/** eGeMAPS smooths every low-level descriptor with a 3-frame moving average
 *  before the functionals ("sma3"); contours here get the same treatment so
 *  the slope functionals see the same degree of jitter. */
function sma3(xs: number[]): number[] {
  if (xs.length < 3) return xs.slice();
  const out = new Array<number>(xs.length);
  for (let i = 0; i < xs.length; i++) {
    const a = xs[Math.max(0, i - 1)];
    const b = xs[i];
    const c = xs[Math.min(xs.length - 1, i + 1)];
    out[i] = (a + b + c) / 3;
  }
  return out;
}

interface Slopes {
  risingMean: number | null;
  risingStd: number | null;
  fallingMean: number | null;
  fallingStd: number | null;
}

/**
 * Rising / falling slopes of a contour, the way eGeMAPS means them: find the
 * local extrema and take the slope of each minimum-to-peak (rising) and
 * peak-to-minimum (falling) run, in units per second. Returns nulls when a
 * contour has no turning points at all (a flat or monotone window).
 */
function contourSlopes(xs: number[], hopSec: number): Slopes {
  const ext: number[] = [];
  for (let i = 1; i < xs.length - 1; i++) {
    const up = xs[i] > xs[i - 1] && xs[i] >= xs[i + 1];
    const down = xs[i] < xs[i - 1] && xs[i] <= xs[i + 1];
    if (up || down) ext.push(i);
  }
  const rising: number[] = [];
  const falling: number[] = [];
  for (let k = 1; k < ext.length; k++) {
    const a = ext[k - 1];
    const b = ext[k];
    const slope = (xs[b] - xs[a]) / ((b - a) * hopSec);
    if (xs[b] > xs[a]) rising.push(slope);
    else if (xs[b] < xs[a]) falling.push(slope);
  }
  return {
    risingMean: mean(rising),
    risingStd: stddev(rising),
    fallingMean: mean(falling),
    fallingStd: stddev(falling),
  };
}

// ---------------------------------------------------------------------------
// Pitch: YIN's cumulative mean normalized difference function with an
// absolute threshold and parabolic refinement. Time domain — for a 25 ms
// integration window and lags up to 55 Hz that is ~100k multiply-adds per
// pitch frame, which measures faster than three FFTs of the padded frame.
// ---------------------------------------------------------------------------
function yinF0(x: Float64Array, offset: number, w: number, sr: number): number | null {
  const tauMin = Math.max(2, Math.floor(sr / F0_MAX_HZ));
  const tauMax = Math.floor(sr / F0_MIN_HZ);
  if (offset + w + tauMax > x.length) return null;

  let power = 0;
  for (let j = 0; j < w; j++) power += x[offset + j] * x[offset + j];
  if (power <= EPS) return null;

  const d = new Float64Array(tauMax + 1);
  for (let tau = tauMin; tau <= tauMax; tau++) {
    let s = 0;
    for (let j = 0; j < w; j++) {
      const diff = x[offset + j] - x[offset + j + tau];
      s += diff * diff;
    }
    d[tau] = s;
  }

  // cumulative mean normalized difference
  const cmnd = new Float64Array(tauMax + 1);
  cmnd[0] = 1;
  let run = 0;
  for (let tau = tauMin; tau <= tauMax; tau++) {
    run += d[tau];
    cmnd[tau] = run === 0 ? 1 : (d[tau] * (tau - tauMin + 1)) / run;
  }

  // first dip below the absolute threshold; else the global minimum, but
  // only if it is convincing enough to call the frame voiced at all
  let tau = -1;
  for (let t = tauMin; t <= tauMax; t++) {
    if (cmnd[t] < YIN_THRESHOLD) {
      while (t + 1 <= tauMax && cmnd[t + 1] < cmnd[t]) t++;
      tau = t;
      break;
    }
  }
  if (tau < 0) return null;

  // parabolic interpolation around the dip
  let better = tau;
  if (tau > tauMin && tau < tauMax) {
    const a = cmnd[tau - 1];
    const b = cmnd[tau];
    const c = cmnd[tau + 1];
    const denom = 2 * (2 * b - a - c);
    if (denom !== 0) better = tau + (c - a) / denom;
  }
  const f0 = sr / better;
  return f0 >= F0_MIN_HZ && f0 <= F0_MAX_HZ ? f0 : null;
}

// ---------------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------------

/** The 16 descriptors the shipped model reads, in model order. */
export const INSTANT_FEATURE_NAMES = [
  "loudMean",
  "loudP20",
  "loudPctlRange",
  "loudStddevNorm",
  "loudRiseSlopeMean",
  "loudRiseSlopeStd",
  "loudFallSlopeMean",
  "loudFallSlopeStd",
  "f0SemitoneP80",
  "spectralFluxMean",
  "mfcc2Mean",
  "mfcc4Mean",
  "alphaRatioV",
  "alphaRatioUV",
  "hammarbergUV",
  "slopeV0to500",
] as const;

export type InstantFeatureName = (typeof INSTANT_FEATURE_NAMES)[number];
export type InstantFeatures = Record<InstantFeatureName, number | null>;

export interface InstantHeat {
  /** P(angry) in [0, 1], or null when the window was too short/too quiet to
   *  measure anything. NOT a nudge level — the policy owns that. */
  score: number | null;
  features: InstantFeatures;
  /** Diagnostics a replay report can show without re-deriving them. */
  frames: number;
  voicedFrames: number;
}

function toFloat64(pcm: Float32Array | Int16Array | Float64Array): Float64Array {
  const out = new Float64Array(pcm.length);
  if (pcm instanceof Int16Array) {
    for (let i = 0; i < pcm.length; i++) out[i] = pcm[i] / 32768;
  } else {
    for (let i = 0; i < pcm.length; i++) out[i] = pcm[i];
  }
  return out;
}

const emptyFeatures = (): InstantFeatures =>
  Object.fromEntries(INSTANT_FEATURE_NAMES.map((n) => [n, null])) as InstantFeatures;

interface Measured {
  features: InstantFeatures;
  frames: number;
  voicedFrames: number;
}

/**
 * Every descriptor the model reads, measured over one window of PCM.
 * Nulls where a descriptor had nothing to measure (no voiced frames, no
 * unvoiced frames, a contour with no turning point).
 */
export function extractInstantFeatures(
  pcm: Float32Array | Int16Array | Float64Array,
  sr: number,
): InstantFeatures {
  return measure(pcm, sr).features;
}

function measure(pcm: Float32Array | Int16Array | Float64Array, sr: number): Measured {
  const f = emptyFeatures();
  const x = toFloat64(pcm);
  const frameLen = Math.round((FRAME_MS / 1000) * sr);
  const hop = Math.round((HOP_MS / 1000) * sr);
  if (x.length < Math.max(frameLen, MIN_WINDOW_SEC * sr)) return { features: f, frames: 0, voicedFrames: 0 };

  const nfft = nextPow2(frameLen);
  const fft = fftFor(nfft);
  const bank = melBank(sr, nfft);
  const dct = dctBasis(MEL_BANDS);
  const bins = nfft / 2 + 1;
  const binHz = sr / nfft;

  // Hamming window, precomputed for this frame length.
  const win = new Float64Array(frameLen);
  for (let i = 0; i < frameLen; i++) win[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (frameLen - 1));

  const nFrames = Math.floor((x.length - frameLen) / hop) + 1;

  // --- pitch contour on its own (coarser) grid --------------------------
  const pitchW = Math.round((PITCH_FRAME_MS / 1000) * sr);
  const pitchHop = Math.round((PITCH_HOP_MS / 1000) * sr);
  const nPitch = Math.max(0, Math.floor((x.length - pitchW - Math.floor(sr / F0_MIN_HZ)) / pitchHop) + 1);
  const f0 = new Float64Array(Math.max(0, nPitch)); // 0 = unvoiced
  for (let p = 0; p < nPitch; p++) {
    const hz = yinF0(x, p * pitchHop, pitchW, sr);
    f0[p] = hz ?? 0;
  }
  const pitchAt = (frameIdx: number): number => {
    if (nPitch === 0) return 0;
    const centre = frameIdx * hop + frameLen / 2;
    const p = Math.min(nPitch - 1, Math.max(0, Math.round((centre - pitchW / 2) / pitchHop)));
    return f0[p];
  };

  // --- frame loop ---------------------------------------------------------
  const re = new Float64Array(nfft);
  const im = new Float64Array(nfft);
  const power = new Float64Array(bins);
  const melE = new Float64Array(MEL_BANDS);
  const logMel = new Float64Array(MEL_BANDS);
  let prevMag: Float64Array | null = null;
  const curMag = new Float64Array(bins);

  const loud: number[] = [];
  const flux: number[] = [];
  const mfcc2: number[] = [];
  const mfcc4: number[] = [];
  const alphaV: number[] = [];
  const alphaUV: number[] = [];
  const hammUV: number[] = [];
  const slope0to500V: number[] = [];
  const f0Semitones: number[] = [];

  // band edges as bin indices, computed once
  const bin = (hz: number) => Math.min(bins - 1, Math.max(0, Math.round(hz / binHz)));
  const b50 = bin(50);
  const b1k = bin(1000);
  const b2k = bin(2000);
  const b5k = bin(5000);
  const b500 = bin(500);

  let voicedFrames = 0;

  for (let t = 0; t < nFrames; t++) {
    const off = t * hop;
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < frameLen; i++) re[i] = x[off + i] * win[i];
    fft.forward(re, im);

    for (let k = 0; k < bins; k++) power[k] = re[k] * re[k] + im[k] * im[k];

    // --- loudness: compressed auditory (mel) spectrum, summed ------------
    let loudness = 0;
    for (let b = 0; b < MEL_BANDS; b++) {
      const w = bank.weights[b];
      const s = bank.start[b];
      let e = 0;
      for (let i = 0; i < w.length; i++) e += power[s + i] * w[i];
      melE[b] = e;
      logMel[b] = Math.log(e + EPS);
      loudness += (e + EPS) ** LOUDNESS_COMPRESSION;
    }
    loud.push(loudness);

    // --- MFCC c2 and c4 ---------------------------------------------------
    let c2 = 0;
    let c4 = 0;
    for (let b = 0; b < MEL_BANDS; b++) {
      c2 += logMel[b] * dct[1][b];
      c4 += logMel[b] * dct[3][b];
    }
    mfcc2.push(c2);
    mfcc4.push(c4);

    // --- spectral flux: L2 distance between consecutive magnitude spectra.
    // On the RAW magnitudes, not energy-normalised ones: normalising the
    // frame first drops the Spearman against openSMILE's spectralFlux from
    // 0.999 to 0.17, because most of what the descriptor carries is how
    // violently the loudness itself is moving.
    for (let k = 0; k < bins; k++) curMag[k] = Math.sqrt(power[k]);
    if (prevMag) {
      let s = 0;
      for (let k = 0; k < bins; k++) {
        const d = curMag[k] - prevMag[k];
        s += d * d;
      }
      flux.push(Math.sqrt(s));
    }
    if (!prevMag) prevMag = new Float64Array(bins);
    prevMag.set(curMag);

    // --- band ratios, split by voicing -----------------------------------
    const hz = pitchAt(t);
    const voiced = hz > 0;
    if (voiced) {
      voicedFrames++;
      f0Semitones.push(12 * Math.log2(hz / SEMITONE_REF_HZ));
    }

    let eLow = 0;
    for (let k = b50; k <= b1k; k++) eLow += power[k];
    let eHigh = 0;
    for (let k = b1k + 1; k <= b5k; k++) eHigh += power[k];
    const alpha = 10 * Math.log10((eHigh + EPS) / (eLow + EPS));

    if (voiced) {
      alphaV.push(alpha);
      // Spectral slope 0-500 Hz: least-squares fit of the spectrum on
      // frequency, in dB. Fitting linear magnitude instead tracks openSMILE
      // at only rho 0.31; in dB it reaches 0.91 — the descriptor is about
      // tilt, and tilt lives in the log domain.
      let sx = 0;
      let sy = 0;
      let sxy = 0;
      let sxx = 0;
      let n = 0;
      for (let k = 0; k <= b500; k++) {
        const fx = k * binHz;
        const fy = 10 * Math.log10(power[k] + EPS);
        sx += fx;
        sy += fy;
        sxy += fx * fy;
        sxx += fx * fx;
        n++;
      }
      const denom = n * sxx - sx * sx;
      if (denom !== 0) slope0to500V.push((n * sxy - sx * sy) / denom);
    } else {
      alphaUV.push(alpha);
      let peakLow = 0;
      for (let k = 0; k <= b2k; k++) if (power[k] > peakLow) peakLow = power[k];
      let peakHigh = 0;
      for (let k = b2k + 1; k <= b5k; k++) if (power[k] > peakHigh) peakHigh = power[k];
      hammUV.push(10 * Math.log10((peakLow + EPS) / (peakHigh + EPS)));
    }
  }

  if (loud.length === 0) return { features: f, frames: 0, voicedFrames: 0 };

  // --- functionals --------------------------------------------------------
  const loudS = sma3(loud);
  const loudSorted = [...loudS].sort((a, b) => a - b);
  const m = mean(loudS);
  const sd = stddev(loudS);
  f.loudMean = m;
  f.loudP20 = percentile(loudSorted, 20);
  const p80 = percentile(loudSorted, 80);
  f.loudPctlRange = p80 !== null && f.loudP20 !== null ? p80 - f.loudP20 : null;
  f.loudStddevNorm = m !== null && sd !== null && Math.abs(m) > EPS ? sd / m : null;

  const slopes = contourSlopes(loudS, HOP_MS / 1000);
  f.loudRiseSlopeMean = slopes.risingMean;
  f.loudRiseSlopeStd = slopes.risingStd;
  f.loudFallSlopeMean = slopes.fallingMean;
  f.loudFallSlopeStd = slopes.fallingStd;

  if (f0Semitones.length > 0) {
    const st = sma3(f0Semitones).sort((a, b) => a - b);
    f.f0SemitoneP80 = percentile(st, 80);
  }

  f.spectralFluxMean = mean(sma3(flux));
  f.mfcc2Mean = mean(sma3(mfcc2));
  f.mfcc4Mean = mean(sma3(mfcc4));
  f.alphaRatioV = mean(alphaV);
  f.alphaRatioUV = mean(alphaUV);
  f.hammarbergUV = mean(hammUV);
  f.slopeV0to500 = mean(slope0to500V);

  return { features: f, frames: nFrames, voicedFrames };
}

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------
interface InstantModel {
  fit_date: string;
  features: string[];
  mean: number[];
  sd: number[];
  coef: number[];
  intercept: number;
  metrics: Record<string, number>;
}
const model = MODEL as unknown as InstantModel;

export const INSTANT_MODEL_FIT_DATE = model.fit_date;
export const INSTANT_MODEL_METRICS = model.metrics;

/** P(angry) from an already-extracted feature set. A null feature falls back
 *  to the training mean (z = 0) — it contributes nothing rather than pulling
 *  the score toward an invented value. */
export function scoreInstantFeatures(features: InstantFeatures): number {
  let z = model.intercept;
  for (let i = 0; i < model.features.length; i++) {
    const v = features[model.features[i] as InstantFeatureName];
    if (v === null || v === undefined || !Number.isFinite(v)) continue;
    z += model.coef[i] * ((v - model.mean[i]) / model.sd[i]);
  }
  return 1 / (1 + Math.exp(-z));
}

/**
 * The instant tier, end to end: 2 s of 16 kHz PCM in, P(angry) out.
 *
 * `score` is null when the window was too short to measure — the caller must
 * treat that as "no opinion", not as "calm".
 */
export function instantHeatScore(
  pcm: Float32Array | Int16Array | Float64Array,
  sr: number,
): InstantHeat {
  const { features, frames, voicedFrames } = measure(pcm, sr);
  if (frames === 0) return { score: null, features, frames, voicedFrames };
  return { score: scoreInstantFeatures(features), features, frames, voicedFrames };
}
