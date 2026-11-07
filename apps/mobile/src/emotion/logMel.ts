/**
 * Whisper log-mel frontend for the Emotion dev tester's two ONNX models
 * (eval/codex/infer.py + input_contract.json are the reference).
 *
 * Matches HF `WhisperFeatureExtractor(feature_size=80)` called as
 * `padding="max_length", max_length=96000` (6 s) and then cut to 600 frames:
 *
 *   - 16 kHz mono float32, NO gain normalisation (the contract says so);
 *   - zero-pad to 96000 samples; periodic Hann(400), hop 160, centre padding
 *     = 200 reflected samples each side; power spectrum;
 *   - 80 Slaney mel filters, 0..8000 Hz, Slaney-normalised;
 *   - log10(max(mel, 1e-10)); drop the last STFT frame (601 -> 600);
 *     max(x, max(x) - 8); (x + 4) / 4;
 *   - valid_frames = min(300, ceil(n / 320)).
 *
 * Parity against the Python reference is pinned by
 * __tests__/logMel.test.ts over a golden vector (scripts/emotion_golden.py).
 *
 * Cost: the 400-point DFT is a mixed-radix FFT (400 = 2^4 * 5^2), and frames
 * that lie entirely in the zero padding are not transformed at all (their
 * value is exactly the clamp, log10(1e-10) = -10) — a 2 s utterance costs a
 * third of a 6 s one.
 */

export const SAMPLE_RATE = 16000;
export const N_FFT = 400;
export const HOP = 160;
export const N_MELS = 80;
export const PAD_SAMPLES = 96000;
export const N_FRAMES = 600;
export const MAX_VALID_FRAMES = 300;
const N_FREQ = N_FFT / 2 + 1;
const SILENCE_RMS = 1e-5;

export interface LogMel {
  /** Row-major [80][600] float32, the model's `log_mel` input (batch dim added by the caller). */
  data: Float32Array;
  validFrames: number;
}

/** Why a waveform must not be fed to the model (contract: reject empty /
 *  non-finite / silent input), or null when it is fine. */
export function waveformProblem(x: Float32Array): string | null {
  if (x.length < SAMPLE_RATE / 10) return "shorter than 0.1 s";
  if (x.length > PAD_SAMPLES) return "longer than 6 s (segment explicitly)";
  let sum = 0;
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    if (!Number.isFinite(v)) return "non-finite samples";
    sum += v * v;
  }
  if (Math.sqrt(sum / x.length) < SILENCE_RMS) return "silence (below -100 dBFS)";
  return null;
}

// ---------------------------------------------------------------------------
// Tables (built once, lazily)
// ---------------------------------------------------------------------------

function hzToMel(f: number): number {
  const fSp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fSp;
  const logstep = Math.log(6.4) / 27;
  return f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / logstep : f / fSp;
}

function melToHz(m: number): number {
  const fSp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fSp;
  const logstep = Math.log(6.4) / 27;
  return m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fSp * m;
}

/** HF `mel_filter_bank(201, 80, 0, 8000, 16000, norm="slaney", mel_scale="slaney")`,
 *  stored [mel][freq]. */
export function buildMelFilters(): Float64Array {
  const melMin = hzToMel(0);
  const melMax = hzToMel(SAMPLE_RATE / 2);
  const filterFreqs = new Float64Array(N_MELS + 2);
  for (let i = 0; i < N_MELS + 2; i++) {
    filterFreqs[i] = melToHz(melMin + ((melMax - melMin) * i) / (N_MELS + 1));
  }
  const fb = new Float64Array(N_MELS * N_FREQ);
  for (let m = 0; m < N_MELS; m++) {
    const lo = filterFreqs[m];
    const mid = filterFreqs[m + 1];
    const hi = filterFreqs[m + 2];
    const enorm = 2 / (hi - lo);
    for (let k = 0; k < N_FREQ; k++) {
      const f = (k * (SAMPLE_RATE / 2)) / (N_FREQ - 1);
      const down = -(lo - f) / (mid - lo); // (f - lo) / (mid - lo)
      const up = (hi - f) / (hi - mid);
      fb[m * N_FREQ + k] = Math.max(0, Math.min(down, up)) * enorm;
    }
  }
  return fb;
}

let melFilters: Float64Array | null = null;
let hann: Float64Array | null = null;

function tables(): { mel: Float64Array; win: Float64Array } {
  if (!melFilters) melFilters = buildMelFilters();
  if (!hann) {
    // Periodic Hann: np.hanning(N + 1)[:-1] == 0.5 - 0.5 cos(2 pi n / N).
    hann = new Float64Array(N_FFT);
    for (let n = 0; n < N_FFT; n++) hann[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);
  }
  return { mel: melFilters, win: hann };
}

// ---------------------------------------------------------------------------
// Mixed-radix FFT, N = 400 = 4 * 4 * 5 * 5 (any factor list works)
// ---------------------------------------------------------------------------

const FACTORS = [4, 4, 5, 5];
let twRe: Float64Array | null = null;
let twIm: Float64Array | null = null;

function twiddles(): { re: Float64Array; im: Float64Array } {
  if (!twRe || !twIm) {
    twRe = new Float64Array(N_FFT);
    twIm = new Float64Array(N_FFT);
    for (let k = 0; k < N_FFT; k++) {
      twRe[k] = Math.cos((2 * Math.PI * k) / N_FFT);
      twIm[k] = -Math.sin((2 * Math.PI * k) / N_FFT);
    }
  }
  return { re: twRe, im: twIm };
}

/** Recursive decimation-in-time FFT over `n` points read at `stride` from the
 *  input, written contiguously to the output. */
function fftRec(
  inRe: Float64Array,
  inIm: Float64Array,
  inOff: number,
  stride: number,
  n: number,
  fi: number,
  outRe: Float64Array,
  outIm: Float64Array,
  outOff: number,
  tw: { re: Float64Array; im: Float64Array },
): void {
  if (n === 1) {
    outRe[outOff] = inRe[inOff];
    outIm[outOff] = inIm[inOff];
    return;
  }
  const p = FACTORS[fi];
  const m = n / p;
  for (let r = 0; r < p; r++) {
    fftRec(inRe, inIm, inOff + r * stride, stride * p, m, fi + 1, outRe, outIm, outOff + r * m, tw);
  }
  // Butterflies: X[k + q m] = sum_r W_n^{r (k + q m)} Y_r[k]
  const twStep = N_FFT / n; // W_n^j = W_N^{j * N/n}
  const tmpRe = new Float64Array(p);
  const tmpIm = new Float64Array(p);
  for (let k = 0; k < m; k++) {
    for (let r = 0; r < p; r++) {
      const idx = ((r * k) % n) * twStep;
      const yr = outRe[outOff + r * m + k];
      const yi = outIm[outOff + r * m + k];
      tmpRe[r] = yr * tw.re[idx] - yi * tw.im[idx];
      tmpIm[r] = yr * tw.im[idx] + yi * tw.re[idx];
    }
    for (let q = 0; q < p; q++) {
      let sr = 0;
      let si = 0;
      for (let r = 0; r < p; r++) {
        const idx = ((r * q * m) % n) * twStep;
        sr += tmpRe[r] * tw.re[idx] - tmpIm[r] * tw.im[idx];
        si += tmpRe[r] * tw.im[idx] + tmpIm[r] * tw.re[idx];
      }
      outRe[outOff + q * m + k] = sr;
      outIm[outOff + q * m + k] = si;
    }
  }
}

/** Power spectrum (201 bins) of one windowed 400-sample frame. */
export function powerSpectrum(frame: Float64Array): Float64Array {
  const tw = twiddles();
  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);
  fftRec(frame, new Float64Array(N_FFT), 0, 1, N_FFT, 0, re, im, 0, tw);
  const out = new Float64Array(N_FREQ);
  for (let k = 0; k < N_FREQ; k++) out[k] = re[k] * re[k] + im[k] * im[k];
  return out;
}

// ---------------------------------------------------------------------------
// The frontend
// ---------------------------------------------------------------------------

/** Log-mel for up to 6 s of 16 kHz mono float32. Throws on a waveform the
 *  contract rejects (callers gate on {@link waveformProblem} first). */
export function computeLogMel(audio: Float32Array): LogMel {
  const problem = waveformProblem(audio);
  if (problem) throw new Error(`invalid waveform: ${problem}`);
  const { mel, win } = tables();
  const n = audio.length;

  // x padded to 6 s, then reflect-padded 200 each side (np.pad "reflect").
  const half = N_FFT / 2;
  const padded = new Float64Array(PAD_SAMPLES + N_FFT);
  for (let i = 0; i < n; i++) padded[half + i] = audio[i];
  for (let i = 1; i <= half; i++) {
    padded[half - i] = padded[half + i]; // left reflect around x[0]
    padded[half + PAD_SAMPLES - 1 + i] = padded[half + PAD_SAMPLES - 1 - i]; // right reflect
  }
  // Highest padded index holding signal energy (frames starting beyond it are all zero).
  let lastNonZero = -1;
  for (let i = padded.length - 1; i >= 0; i--) {
    if (padded[i] !== 0) {
      lastNonZero = i;
      break;
    }
  }

  const logSpec = new Float64Array(N_MELS * N_FRAMES);
  const frame = new Float64Array(N_FFT);
  let maxVal = -Infinity;
  for (let t = 0; t < N_FRAMES; t++) {
    const start = t * HOP;
    if (start > lastNonZero) {
      for (let m = 0; m < N_MELS; m++) logSpec[m * N_FRAMES + t] = -10; // log10(1e-10)
      if (-10 > maxVal) maxVal = -10;
      continue;
    }
    for (let i = 0; i < N_FFT; i++) frame[i] = padded[start + i] * win[i];
    const power = powerSpectrum(frame);
    for (let m = 0; m < N_MELS; m++) {
      let acc = 0;
      const row = m * N_FREQ;
      for (let k = 0; k < N_FREQ; k++) acc += mel[row + k] * power[k];
      const v = Math.log10(Math.max(acc, 1e-10));
      logSpec[m * N_FRAMES + t] = v;
      if (v > maxVal) maxVal = v;
    }
  }
  const floor = maxVal - 8;
  const out = new Float32Array(N_MELS * N_FRAMES);
  for (let i = 0; i < out.length; i++) out[i] = (Math.max(logSpec[i], floor) + 4) / 4;
  return { data: out, validFrames: Math.min(MAX_VALID_FRAMES, Math.ceil(n / 320)) };
}

/** int16 PCM (what the mic path holds) -> the float32 the models expect. */
export function int16ToFloat32(pcm: Int16Array): Float32Array {
  const out = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) out[i] = pcm[i] / 32768;
  return out;
}
