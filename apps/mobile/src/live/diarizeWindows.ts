/**
 * Approach B on the phone — a pure-TypeScript port of the server's
 * transcript-free window pass + spectral clustering (the SHIPPED semantics:
 * `server/diarize_local.py::_WindowPass` + `server/diarize_sliding_window.py`
 * `refine_affinity` / `eigengap_k` / `spectral_labels` / `mode_filter` /
 * `window_label_runs`; bake-off: docs/research/2026-08-29-voice-separation/
 * B-sliding-window/). Run post-hoc over a stored recording's 16 kHz PCM:
 *
 *   speech gate (noise-floor-relative RMS, 30 ms frames)
 *     -> 1.5 s / 0.25 s window grid, ≤ MAX_WINDOWS with hop auto-widening
 *     -> one ECAPA embedding per speech window (injected `EmbedBatch`)
 *     -> refined cosine affinity (Wang et al. 2018) -> eigengap k
 *     -> k-way spectral clustering (seeded k-means++, numpy-identical draws)
 *     -> mode filter over ±2 hops -> runs ≥ 0.5 s -> [[start, end, label]]
 *
 * Every step mirrors the numpy line by line (same percentile interpolation,
 * same tie-breaking, same RNG — `numpyRandom.ts`), so a phone result can be
 * compared 1:1 with the server's on the same embeddings
 * (`__tests__/diarizeWindows.parity.test.ts`). The only piece with no numpy
 * twin is the symmetric eigensolver (`symmetricEigen`: Householder
 * tridiagonalisation + implicit QL, the classic EISPACK tred2/tql2 pair) —
 * eigenvalues agree to ~1e-12 and the k-means on the top-k eigenvectors is
 * invariant to the basis chosen within a degenerate eigenspace.
 *
 * No dependencies; nothing here touches ORT, the network or React.
 */
import { l2Normalize } from "./speakerId";
import { NumpyGenerator } from "./numpyRandom";

// --- the shipped constants (server/diarize_local.py, speaker_id.py) ---------
export const WINDOW_SECONDS = 1.5;
export const HOP_SECONDS = 0.25;
export const MAX_WINDOWS = 600;
export const MIN_SPEECH_FRAC = 0.3;
export const EMBED_BATCH = 12;
export const SPEECH_FRAME_MS = 30;
export const SPEECH_RMS_FLOOR = 0.003;
export const SPEECH_RMS_FLOOR_MULT = 1.5;
export const SPEECH_NOISE_FLOOR_PERCENTILE = 10;
export const SPEECH_RMS_GATE_CEILING = 0.03;
export const SPECTRAL_PERCENTILE = 0.8;
export const SPECTRAL_SMOOTH_HOPS = 2;
export const SPECTRAL_MIN_RUN_SECONDS = 0.5;
export const KMEANS_RESTARTS = 10;
/** B's eigengap search range (run_b.py SPEC_MAX_K); production's k-selection
 *  narrows it to MAX_SPEAKERS_LOCAL (6) with a floor of 2 — pass
 *  `{ maxSpeakers: 6, minSpeakers: 2 }` for that behaviour. */
export const MAX_SPEAKERS = 8;

export type Segment = [start: number, end: number, label: number];

/** Embeds a batch of equal-length 16 kHz PCM chunks → one vector each. */
export type EmbedBatch = (chunks: Float32Array[], sampleRate: number) => Promise<ArrayLike<number>[]>;

export interface DiarizeProgress {
  stage: "gate" | "embed" | "cluster" | "smooth";
  done: number;
  total: number;
}

export interface ClusterOptions {
  percentile?: number;
  maxSpeakers?: number;
  minSpeakers?: number;
  smoothHops?: number;
  minRunSeconds?: number;
}

export interface DiarizeWindowsOptions extends ClusterOptions {
  windowSeconds?: number;
  hopSeconds?: number;
  maxWindows?: number;
  minSpeechFrac?: number;
  batchSize?: number;
  onProgress?: (p: DiarizeProgress) => void;
  /** Checked between embedding batches; `aborted` → an AbortError. */
  signal?: { readonly aborted: boolean } | null;
}

export interface ClusterResult {
  /** Spectral labels before smoothing (one per window). */
  rawLabels: number[];
  /** After the ±smoothHops mode filter. */
  labels: number[];
  k: number;
  kEigengap: number;
  eigenvalues: number[];
}

export interface DiarizeWindowsResult extends ClusterResult {
  segments: Segment[];
  /** Start second of every SPEECH window that was embedded. */
  starts: number[];
  /** Its L2-normalised embedding (index-aligned with `starts`). */
  embeddings: Float32Array[];
  windows: number;
  totalWindows: number;
  windowSeconds: number;
  hopSeconds: number;
  gate: number;
  speechSeconds: number;
  durationSeconds: number;
  embedMs: number[];
  timings: { gateMs: number; embedMs: number; clusterMs: number; smoothMs: number; totalMs: number };
}

// ---------------------------------------------------------------------------
// numpy-faithful helpers
// ---------------------------------------------------------------------------

/** Python's round() (half to even) on a non-negative float. */
function pyRound(x: number): number {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff > 0.5) return f + 1;
  if (diff < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/**
 * `np.percentile(values, q * 100)` with the default "linear" method — the
 * exact virtual-index and lerp arithmetic of numpy 2.x so a threshold that
 * lands ON a sample compares identically on both sides.
 */
export function percentileLinear(values: ArrayLike<number>, qPercent: number): number {
  const n = values.length;
  if (n === 0) return NaN;
  const sorted = Float64Array.from(values as ArrayLike<number>).sort();
  const q = qPercent / 100;
  // _compute_virtual_index(n, q, alpha=1, beta=1) = n*q + (1 + q*(1-2)) - 1
  const virtual = n * q + (1 + q * (1 - 1 - 1)) - 1;
  let prev = Math.floor(virtual);
  let gamma = virtual - prev;
  let next = prev + 1;
  if (prev < 0) {
    prev = 0;
    gamma = 0;
  }
  if (prev > n - 1) prev = n - 1;
  if (next > n - 1) next = n - 1;
  const a = sorted[prev];
  const b = sorted[next];
  const diff = b - a;
  return gamma >= 0.5 ? b - diff * (1 - gamma) : a + diff * gamma;
}

/** RMS of each full `frameMs` frame (partial tail dropped), float64. */
export function frameRms(pcm: Float32Array, sampleRate: number, frameMs = SPEECH_FRAME_MS): Float64Array {
  if (pcm.length === 0 || sampleRate <= 0) return new Float64Array(0);
  const frame = Math.max(1, Math.trunc((sampleRate * frameMs) / 1000));
  const n = Math.floor(pcm.length / frame);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    const base = i * frame;
    for (let j = 0; j < frame; j++) {
      const v = pcm[base + j];
      acc += v * v;
    }
    out[i] = Math.sqrt(acc / frame);
  }
  return out;
}

/** speaker_id.speech_rms_threshold: max(floor, min(ceiling, mult × p10)). */
export function speechRmsThreshold(
  rms: ArrayLike<number>,
  rmsFloor = SPEECH_RMS_FLOOR,
  floorMult = SPEECH_RMS_FLOOR_MULT,
  floorPercentile = SPEECH_NOISE_FLOOR_PERCENTILE,
  ceiling = SPEECH_RMS_GATE_CEILING,
): number {
  if (rms.length === 0) return rmsFloor;
  const relative = floorMult * percentileLinear(rms, floorPercentile);
  return Math.max(rmsFloor, Math.min(ceiling, relative));
}

export interface SpeechMask {
  mask: Uint8Array;
  threshold: number;
  frameSeconds: number;
}

/** speaker_id.speech_mask — one flag per 30 ms frame. */
export function speechMask(pcm: Float32Array, sampleRate: number): SpeechMask {
  const rms = frameRms(pcm, sampleRate);
  const threshold = speechRmsThreshold(rms);
  const mask = new Uint8Array(rms.length);
  for (let i = 0; i < rms.length; i++) mask[i] = rms[i] >= threshold ? 1 : 0;
  return { mask, threshold, frameSeconds: SPEECH_FRAME_MS / 1000 };
}

export interface WindowGrid {
  /** Start SAMPLE of every window on the clip-anchored grid. */
  starts: number[];
  windowSamples: number;
  hopSamples: number;
  /** The grid size at the requested hop, before any widening. */
  totalAtRequestedHop: number;
  widened: boolean;
}

/** _WindowPass._grid over the whole clip + run_global's hop widening. */
export function windowGrid(
  nSamples: number,
  sampleRate: number,
  windowSeconds = WINDOW_SECONDS,
  hopSeconds = HOP_SECONDS,
  maxWindows = MAX_WINDOWS,
): WindowGrid {
  const windowSamples = Math.trunc(pyRound(windowSeconds * sampleRate));
  let hopSamples = Math.trunc(pyRound(hopSeconds * sampleRate));
  if (windowSamples <= 0 || hopSamples <= 0) throw new Error("diarizeWindows: window and hop must be positive");
  const count = (hop: number) => (nSamples - windowSamples + 1 > 0 ? Math.ceil((nSamples - windowSamples + 1) / hop) : 0);
  const totalAtRequestedHop = count(hopSamples);
  let widened = false;
  if (totalAtRequestedHop > maxWindows) {
    const factor = Math.ceil(totalAtRequestedHop / maxWindows);
    hopSamples *= factor;
    widened = true;
  }
  const starts: number[] = [];
  for (let s = 0; s + windowSamples <= nSamples; s += hopSamples) starts.push(s);
  return { starts, windowSamples, hopSamples, totalAtRequestedHop, widened };
}

/** _WindowPass._is_speech for one window. */
export function windowIsSpeech(
  speech: SpeechMask,
  startSample: number,
  windowSamples: number,
  sampleRate: number,
  minSpeechFrac = MIN_SPEECH_FRAC,
): boolean {
  const a = Math.trunc(startSample / sampleRate / speech.frameSeconds);
  const b = Math.max(a + 1, Math.trunc((startSample + windowSamples) / sampleRate / speech.frameSeconds));
  const hi = Math.min(b, speech.mask.length);
  if (hi <= a) return false;
  let sum = 0;
  for (let i = a; i < hi; i++) sum += speech.mask[i];
  return sum / (hi - a) >= minSpeechFrac;
}

// ---------------------------------------------------------------------------
// Affinity + eigengap
// ---------------------------------------------------------------------------

/** Row-major N×N float64 matrix. */
export interface Matrix {
  n: number;
  data: Float64Array;
}

/**
 * diarize_sliding_window.refine_affinity: cosine affinity of L2-normalised
 * embeddings, diagonal = row max, row-wise percentile damping (×0.01 under
 * the row's p-quantile), symmetrise by max, diffusion A·Aᵀ, row-max
 * normalisation.
 */
export function refineAffinity(embs: ArrayLike<number>[], p = SPECTRAL_PERCENTILE): Matrix {
  const n = embs.length;
  const d = n > 0 ? embs[0].length : 0;
  const a = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    const ei = embs[i];
    for (let j = i; j < n; j++) {
      const ej = embs[j];
      let dot = 0;
      for (let t = 0; t < d; t++) dot += ei[t] * ej[t];
      a[i * n + j] = dot;
      a[j * n + i] = dot;
    }
  }
  // diag = 0, then diag = row max (the zero participates in the max).
  for (let i = 0; i < n; i++) {
    a[i * n + i] = 0;
    let m = -Infinity;
    for (let j = 0; j < n; j++) m = Math.max(m, a[i * n + j]);
    a[i * n + i] = m;
  }
  // Row-wise percentile threshold; entries under it damped ×0.01.
  const row = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) row[j] = a[i * n + j];
    const thr = percentileLinear(row, p * 100);
    for (let j = 0; j < n; j++) {
      const v = a[i * n + j];
      if (!(v >= thr)) a[i * n + j] = v * 0.01;
    }
  }
  // Symmetrise by max.
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const m = Math.max(a[i * n + j], a[j * n + i]);
      a[i * n + j] = m;
      a[j * n + i] = m;
    }
  }
  // Diffusion: A @ A.T (A symmetric now, so this is A²).
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    const ri = i * n;
    for (let j = 0; j < n; j++) {
      const rj = j * n;
      let s = 0;
      for (let t = 0; t < n; t++) s += a[ri + t] * a[rj + t];
      out[ri + j] = s;
    }
  }
  // Row-max normalise.
  for (let i = 0; i < n; i++) {
    let m = -Infinity;
    for (let j = 0; j < n; j++) m = Math.max(m, out[i * n + j]);
    const denom = Math.max(m, 1e-9);
    for (let j = 0; j < n; j++) out[i * n + j] /= denom;
  }
  return { n, data: out };
}

export interface EigenResult {
  /** Ascending. */
  values: Float64Array;
  /** vectors[k] is the k-th ROW of V; column j of V is the eigenvector of values[j]. */
  vectors: Float64Array[];
}

/**
 * Eigen-decomposition of a real symmetric matrix (Householder tridiagonal
 * reduction + implicit QL with eigenvector accumulation — EISPACK tred2 /
 * tql2, the algorithm behind JAMA's symmetric EigenvalueDecomposition).
 * O(n³); n ≤ 600 takes well under a second in JS.
 */
export function symmetricEigen(m: Matrix): EigenResult {
  const n = m.n;
  const V: Float64Array[] = [];
  for (let i = 0; i < n; i++) V.push(m.data.slice(i * n, (i + 1) * n));
  const d = new Float64Array(n);
  const e = new Float64Array(n);
  if (n === 0) return { values: d, vectors: V };

  // --- tred2 ---------------------------------------------------------------
  for (let j = 0; j < n; j++) d[j] = V[n - 1][j];
  for (let i = n - 1; i > 0; i--) {
    let scale = 0;
    let h = 0;
    for (let k = 0; k < i; k++) scale += Math.abs(d[k]);
    if (scale === 0) {
      e[i] = d[i - 1];
      for (let j = 0; j < i; j++) {
        d[j] = V[i - 1][j];
        V[i][j] = 0;
        V[j][i] = 0;
      }
    } else {
      for (let k = 0; k < i; k++) {
        d[k] /= scale;
        h += d[k] * d[k];
      }
      let f = d[i - 1];
      let g = Math.sqrt(h);
      if (f > 0) g = -g;
      e[i] = scale * g;
      h -= f * g;
      d[i - 1] = f - g;
      for (let j = 0; j < i; j++) e[j] = 0;
      for (let j = 0; j < i; j++) {
        f = d[j];
        V[j][i] = f;
        g = e[j] + V[j][j] * f;
        for (let k = j + 1; k <= i - 1; k++) {
          g += V[k][j] * d[k];
          e[k] += V[k][j] * f;
        }
        e[j] = g;
      }
      f = 0;
      for (let j = 0; j < i; j++) {
        e[j] /= h;
        f += e[j] * d[j];
      }
      const hh = f / (h + h);
      for (let j = 0; j < i; j++) e[j] -= hh * d[j];
      for (let j = 0; j < i; j++) {
        f = d[j];
        g = e[j];
        for (let k = j; k <= i - 1; k++) V[k][j] -= f * e[k] + g * d[k];
        d[j] = V[i - 1][j];
        V[i][j] = 0;
      }
    }
    d[i] = h;
  }
  for (let i = 0; i < n - 1; i++) {
    V[n - 1][i] = V[i][i];
    V[i][i] = 1;
    const h = d[i + 1];
    if (h !== 0) {
      for (let k = 0; k <= i; k++) d[k] = V[k][i + 1] / h;
      for (let j = 0; j <= i; j++) {
        let g = 0;
        for (let k = 0; k <= i; k++) g += V[k][i + 1] * V[k][j];
        for (let k = 0; k <= i; k++) V[k][j] -= g * d[k];
      }
    }
    for (let k = 0; k <= i; k++) V[k][i + 1] = 0;
  }
  for (let j = 0; j < n; j++) {
    d[j] = V[n - 1][j];
    V[n - 1][j] = 0;
  }
  V[n - 1][n - 1] = 1;
  e[0] = 0;

  // --- tql2 ----------------------------------------------------------------
  for (let i = 1; i < n; i++) e[i - 1] = e[i];
  e[n - 1] = 0;
  let f = 0;
  let tst1 = 0;
  const eps = Math.pow(2, -52);
  for (let l = 0; l < n; l++) {
    tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
    let m2 = l;
    while (m2 < n) {
      if (Math.abs(e[m2]) <= eps * tst1) break;
      m2++;
    }
    if (m2 > l) {
      let iter = 0;
      do {
        iter++;
        if (iter > 200) throw new Error("symmetricEigen: QL iteration did not converge");
        let g = d[l];
        let p = (d[l + 1] - g) / (2 * e[l]);
        let r = Math.hypot(p, 1);
        if (p < 0) r = -r;
        d[l] = e[l] / (p + r);
        d[l + 1] = e[l] * (p + r);
        const dl1 = d[l + 1];
        let h = g - d[l];
        for (let i = l + 2; i < n; i++) d[i] -= h;
        f += h;
        p = d[m2];
        let c = 1;
        let c2 = c;
        let c3 = c;
        const el1 = e[l + 1];
        let s = 0;
        let s2 = 0;
        for (let i = m2 - 1; i >= l; i--) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[i];
          h = c * p;
          r = Math.hypot(p, e[i]);
          e[i + 1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);
          for (let k = 0; k < n; k++) {
            const row = V[k];
            h = row[i + 1];
            row[i + 1] = s * row[i] + c * h;
            row[i] = c * row[i] - s * h;
          }
        }
        p = (-s * s2 * c3 * el1 * e[l]) / dl1;
        e[l] = s * p;
        d[l] = c * p;
      } while (Math.abs(e[l]) > eps * tst1);
    }
    d[l] += f;
    e[l] = 0;
  }
  // Sort ascending (selection sort, swapping eigenvector columns).
  for (let i = 0; i < n - 1; i++) {
    let k = i;
    let p = d[i];
    for (let j = i + 1; j < n; j++) {
      if (d[j] < p) {
        k = j;
        p = d[j];
      }
    }
    if (k !== i) {
      d[k] = d[i];
      d[i] = p;
      for (let j = 0; j < n; j++) {
        const t = V[j][i];
        V[j][i] = V[j][k];
        V[j][k] = t;
      }
    }
  }
  return { values: d, vectors: V };
}

function symmetrised(m: Matrix): Matrix {
  const n = m.n;
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) out[i * n + j] = (m.data[i * n + j] + m.data[j * n + i]) / 2;
  }
  return { n, data: out };
}

/**
 * diarize_sliding_window.eigengap_k: k = argmax over 1..maxK of λk/λk+1 on
 * the descending eigenvalues (clipped at 1e-9). Returns the eigenvalues it
 * looked at (the first maxK + 1, descending).
 */
export function eigengapK(affinity: Matrix, maxK: number, eigen?: EigenResult): { k: number; eigenvalues: number[] } {
  const n = affinity.n;
  if (n < 2) return { k: 1, eigenvalues: [] };
  const ev = eigen ?? symmetricEigen(symmetrised(affinity));
  const w: number[] = [];
  for (let i = n - 1; i >= 0; i--) w.push(Math.max(ev.values[i], 1e-9));
  const mk = Math.max(1, Math.min(Math.trunc(maxK), n - 1));
  let best = -Infinity;
  let bestI = 0;
  for (let i = 0; i < mk; i++) {
    const ratio = w[i] / w[i + 1];
    if (ratio > best) {
      best = ratio;
      bestI = i;
    }
  }
  return { k: bestI + 1, eigenvalues: w.slice(0, mk + 1) };
}

// ---------------------------------------------------------------------------
// Seeded k-means++ (numpy-identical draws) + spectral labels
// ---------------------------------------------------------------------------

/** diarize_sliding_window._kmeans: best of `restarts` seeded k-means++ runs. */
export function kmeansSeeded(feats: Float64Array[], k: number, restarts = KMEANS_RESTARTS, seed = 0, iters = 100): number[] {
  const n = feats.length;
  if (n === 0) return [];
  const dim = feats[0].length;
  const kk = Math.max(1, Math.min(Math.trunc(k), n));
  const rng = new NumpyGenerator(seed);
  let bestLabels = new Array<number>(n).fill(0);
  let bestInertia = Infinity;
  const sq = (x: Float64Array, c: Float64Array): number => {
    let s = 0;
    for (let t = 0; t < dim; t++) {
      const diff = x[t] - c[t];
      s += diff * diff;
    }
    return s;
  };
  for (let restart = 0; restart < restarts; restart++) {
    const centers: Float64Array[] = [];
    centers.push(Float64Array.from(feats[rng.integers(n)]));
    const d2 = new Float64Array(n);
    for (let i = 0; i < n; i++) d2[i] = sq(feats[i], centers[0]);
    for (let j = 1; j < kk; j++) {
      let tot = 0;
      for (let i = 0; i < n; i++) tot += d2[i];
      let idx: number;
      if (tot <= 0) idx = rng.integers(n);
      else {
        const p = new Float64Array(n);
        for (let i = 0; i < n; i++) p[i] = d2[i] / tot;
        idx = rng.choice(p);
      }
      centers.push(Float64Array.from(feats[idx]));
      for (let i = 0; i < n; i++) d2[i] = Math.min(d2[i], sq(feats[i], centers[j]));
    }
    let labels = new Array<number>(n).fill(0);
    const assign = (): number[] => {
      const out = new Array<number>(n);
      for (let i = 0; i < n; i++) {
        let bi = 0;
        let bd = sq(feats[i], centers[0]);
        for (let j = 1; j < kk; j++) {
          const dd = sq(feats[i], centers[j]);
          if (dd < bd) {
            bd = dd;
            bi = j;
          }
        }
        out[i] = bi;
      }
      return out;
    };
    for (let it = 0; it < iters; it++) {
      const next = assign();
      let same = true;
      for (let i = 0; i < n; i++) {
        if (next[i] !== labels[i]) {
          same = false;
          break;
        }
      }
      if (same && it > 0) break;
      labels = next;
      for (let j = 0; j < kk; j++) {
        const mean = new Float64Array(dim);
        let count = 0;
        for (let i = 0; i < n; i++) {
          if (labels[i] !== j) continue;
          count++;
          for (let t = 0; t < dim; t++) mean[t] += feats[i][t];
        }
        if (count > 0) {
          for (let t = 0; t < dim; t++) mean[t] /= count;
          centers[j] = mean;
        }
      }
    }
    let inertia = 0;
    for (let i = 0; i < n; i++) inertia += sq(feats[i], centers[labels[i]]);
    if (inertia < bestInertia) {
      bestInertia = inertia;
      bestLabels = labels.slice();
    }
  }
  return bestLabels;
}

/**
 * diarize_sliding_window.spectral_labels: k-means on the row-normalised
 * top-k eigenvectors of the symmetrised affinity. k ≤ 1 → all zeros.
 */
export function spectralLabels(affinity: Matrix, k: number, eigen?: EigenResult): number[] {
  const n = affinity.n;
  const kk = Math.max(1, Math.min(Math.trunc(k), n));
  if (kk === 1) return new Array<number>(n).fill(0);
  const ev = eigen ?? symmetricEigen(symmetrised(affinity));
  const feats: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(kk);
    let norm = 0;
    for (let j = 0; j < kk; j++) {
      const v = ev.vectors[i][n - 1 - j];
      row[j] = v;
      norm += v * v;
    }
    norm = Math.max(Math.sqrt(norm), 1e-9);
    for (let j = 0; j < kk; j++) row[j] /= norm;
    feats.push(row);
  }
  return kmeansSeeded(feats, kk);
}

// ---------------------------------------------------------------------------
// Smoothing + runs
// ---------------------------------------------------------------------------

/** diarize_sliding_window.mode_filter — mode over temporal neighbours within
 *  `radius` hops; ties keep the window's own label, else the smallest label. */
export function modeFilter(labels: ArrayLike<number>, starts: ArrayLike<number>, hop: number, radius = SPECTRAL_SMOOTH_HOPS): number[] {
  const n = labels.length;
  const out = new Array<number>(n);
  const reach = radius * hop + 1e-6;
  for (let i = 0; i < n; i++) {
    const counts = new Map<number, number>();
    for (let j = 0; j < n; j++) {
      if (Math.abs(starts[j] - starts[i]) <= reach) counts.set(labels[j], (counts.get(labels[j]) ?? 0) + 1);
    }
    let max = 0;
    for (const c of counts.values()) max = Math.max(max, c);
    const own = labels[i];
    if (counts.get(own) === max) {
      out[i] = own;
      continue;
    }
    let best = Infinity;
    for (const [lab, c] of counts) if (c === max && lab < best) best = lab;
    out[i] = best;
  }
  return out;
}

/**
 * diarize_sliding_window.window_label_runs: window labels → runs covering
 * [lo, hi] at 10 ms resolution by nearest window CENTRE (gaps inherit),
 * adjacent same-label frames merged, runs shorter than `minRun` absorbed
 * into the longer neighbour shortest-first.
 */
export function windowLabelRuns(
  labels: ArrayLike<number>,
  starts: ArrayLike<number>,
  windowSeconds: number,
  lo: number,
  hi: number,
  minRun = SPECTRAL_MIN_RUN_SECONDS,
  step = 0.01,
): Segment[] {
  const n = Math.max(1, Math.trunc(pyRound((hi - lo) / step)));
  if (starts.length === 0) return [[lo, hi, 0]];
  const nw = starts.length;
  const centres = new Float64Array(nw);
  for (let j = 0; j < nw; j++) centres[j] = starts[j] + windowSeconds / 2;
  // Nearest centre per frame (first index on an exact tie), sweeping.
  const frameLabels = new Int32Array(n);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const t = lo + (i + 0.5) * step;
    while (j + 1 < nw && Math.abs(t - centres[j + 1]) < Math.abs(t - centres[j])) j++;
    frameLabels[i] = labels[j];
  }
  let runs: Segment[] = [];
  let s = 0;
  for (let i = 1; i <= n; i++) {
    if (i === n || frameLabels[i] !== frameLabels[s]) {
      runs.push([lo + s * step, i === n ? hi : lo + i * step, frameLabels[s]]);
      s = i;
    }
  }
  while (runs.length > 1) {
    let iMin = 0;
    let lenMin = Infinity;
    for (let r = 0; r < runs.length; r++) {
      const len = runs[r][1] - runs[r][0];
      if (len < lenMin) {
        lenMin = len;
        iMin = r;
      }
    }
    if (lenMin >= minRun) break;
    const cand = [iMin - 1, iMin + 1].filter((c) => c >= 0 && c < runs.length);
    let best = cand[0];
    for (const c of cand) if (runs[c][1] - runs[c][0] > runs[best][1] - runs[best][0]) best = c;
    runs[iMin][2] = runs[best][2];
    const merged: Segment[] = [];
    for (const [b, e2, lab] of runs) {
      if (merged.length > 0 && merged[merged.length - 1][2] === lab) merged[merged.length - 1][1] = e2;
      else merged.push([b, e2, lab]);
    }
    runs = merged;
  }
  return runs;
}

// ---------------------------------------------------------------------------
// Composition
// ---------------------------------------------------------------------------

/** Affinity → eigengap k → spectral labels → mode filter (no embedding). */
export function clusterWindows(embs: ArrayLike<number>[], starts: number[], hopSeconds: number, opts: ClusterOptions = {}): ClusterResult {
  const n = embs.length;
  const maxSpeakers = opts.maxSpeakers ?? MAX_SPEAKERS;
  const minSpeakers = opts.minSpeakers ?? 1;
  if (n < 3) {
    const zeros = new Array<number>(n).fill(0);
    return { rawLabels: zeros, labels: zeros.slice(), k: n > 0 ? 1 : 0, kEigengap: 1, eigenvalues: [] };
  }
  const affinity = refineAffinity(embs, opts.percentile ?? SPECTRAL_PERCENTILE);
  const eigen = symmetricEigen(symmetrised(affinity));
  const { k: kEigengap, eigenvalues } = eigengapK(affinity, maxSpeakers, eigen);
  const k = Math.max(minSpeakers, Math.min(maxSpeakers, kEigengap, n));
  const rawLabels = spectralLabels(affinity, k, eigen);
  const labels = modeFilter(rawLabels, starts, hopSeconds, opts.smoothHops ?? SPECTRAL_SMOOTH_HOPS);
  return { rawLabels, labels, k: new Set(rawLabels).size, kEigengap, eigenvalues };
}

function abortError(): Error {
  const err = new Error("diarizeWindows: cancelled");
  err.name = "AbortError";
  return err;
}

/** The whole pass over one clip's 16 kHz float PCM. */
export async function diarizeWindows(pcm: Float32Array, sampleRate: number, embedBatch: EmbedBatch, opts: DiarizeWindowsOptions = {}): Promise<DiarizeWindowsResult> {
  const t0 = now();
  const windowSeconds = opts.windowSeconds ?? WINDOW_SECONDS;
  const hopSeconds = opts.hopSeconds ?? HOP_SECONDS;
  const progress = opts.onProgress ?? (() => {});
  const checkAbort = () => {
    if (opts.signal?.aborted) throw abortError();
  };

  progress({ stage: "gate", done: 0, total: 1 });
  const speech = speechMask(pcm, sampleRate);
  let speechFrames = 0;
  for (let i = 0; i < speech.mask.length; i++) speechFrames += speech.mask[i];
  const grid = windowGrid(pcm.length, sampleRate, windowSeconds, hopSeconds, opts.maxWindows ?? MAX_WINDOWS);
  const keep = grid.starts.filter((s) => windowIsSpeech(speech, s, grid.windowSamples, sampleRate, opts.minSpeechFrac ?? MIN_SPEECH_FRAC));
  const gateMs = now() - t0;
  progress({ stage: "gate", done: 1, total: 1 });
  checkAbort();

  const tEmbed = now();
  const batchSize = Math.max(1, opts.batchSize ?? EMBED_BATCH);
  const embs: Float32Array[] = [];
  const embedMs: number[] = [];
  progress({ stage: "embed", done: 0, total: keep.length });
  for (let i = 0; i < keep.length; i += batchSize) {
    checkAbort();
    const batch = keep.slice(i, i + batchSize);
    const chunks = batch.map((s) => pcm.subarray(s, s + grid.windowSamples));
    const tb = now();
    const vecs = await embedBatch(chunks, sampleRate);
    const per = (now() - tb) / batch.length;
    if (vecs.length !== batch.length) throw new Error(`diarizeWindows: embedBatch returned ${vecs.length} vectors for ${batch.length} windows`);
    for (const v of vecs) {
      embs.push(l2Normalize(v));
      embedMs.push(per);
    }
    progress({ stage: "embed", done: Math.min(keep.length, i + batch.length), total: keep.length });
  }
  const embedTotalMs = now() - tEmbed;
  checkAbort();

  const tCluster = now();
  progress({ stage: "cluster", done: 0, total: 1 });
  const starts = keep.map((s) => s / sampleRate);
  const actualHop = grid.hopSamples / sampleRate;
  const clustered = clusterWindows(embs, starts, actualHop, opts);
  const clusterMs = now() - tCluster;
  progress({ stage: "cluster", done: 1, total: 1 });

  const tSmooth = now();
  progress({ stage: "smooth", done: 0, total: 1 });
  const durationSeconds = pcm.length / sampleRate;
  const segments = windowLabelRuns(clustered.labels, starts, windowSeconds, 0, durationSeconds, opts.minRunSeconds ?? SPECTRAL_MIN_RUN_SECONDS);
  const smoothMs = now() - tSmooth;
  progress({ stage: "smooth", done: 1, total: 1 });

  return {
    ...clustered,
    segments,
    starts,
    embeddings: embs,
    windows: keep.length,
    totalWindows: grid.totalAtRequestedHop,
    windowSeconds,
    hopSeconds: actualHop,
    gate: speech.threshold,
    speechSeconds: speechFrames * speech.frameSeconds,
    durationSeconds,
    embedMs,
    timings: { gateMs, embedMs: embedTotalMs, clusterMs, smoothMs, totalMs: now() - t0 },
  };
}

function now(): number {
  return typeof performance !== "undefined" && typeof performance.now === "function" ? performance.now() : Date.now();
}
