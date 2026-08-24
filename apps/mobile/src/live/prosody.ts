/**
 * Per-turn prosody, on the phone — a straight port of server/prosody.py's
 * feature extraction (RMS energy, autocorrelation pitch, speech-rate proxy).
 *
 * Only the MEASUREMENT half is ported. The server's relative labelling
 * (tertiles over a whole recording's turns) needs the whole recording; the
 * realtime path reports raw numbers in `TurnLocalEvent.prosody` and lets the
 * server/episode record label them later. Same honesty rule as the server:
 * pitch is `null` when a turn is mostly unvoiced — never a fabricated F0.
 *
 * Pure functions over Float32Array — no native seam, fully Jest-testable on
 * synthetic tones (see __tests__/liveProsody.test.ts, which mirrors
 * server/tests/test_prosody.py's expectations).
 */

// Frame-wise F0 params — identical to server/prosody.py so both runtimes
// measure the same thing.
export const FRAME_MS = 40.0;
export const HOP_MS = 10.0;
export const F0_MIN_HZ = 60.0;
export const F0_MAX_HZ = 400.0;
export const VOICED_AUTOCORR_THRESHOLD = 0.35;
export const MIN_VOICED_FRACTION = 0.2;

/** Root-mean-square amplitude (0 for an empty window). */
export function rmsEnergy(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let acc = 0;
  for (let i = 0; i < samples.length; i++) acc += samples[i] * samples[i];
  return Math.sqrt(acc / samples.length);
}

/** RMS loudness in dB relative to full scale for float PCM in [-1, 1];
 *  `-Infinity` for digital silence (matches watch.vectors.rms_dbfs, which
 *  divides int16 by 32768 — same number for the same signal). */
export function rmsDbfs(samples: Float32Array): number {
  const rms = rmsEnergy(samples);
  if (rms <= 0) return -Infinity;
  return 20 * Math.log10(rms);
}

/**
 * One frame's F0 via normalized autocorrelation, or `null` when unvoiced.
 * Mean-removed, autocorrelated over the 60–400 Hz lag window, normalized by
 * the zero-lag energy; a peak below VOICED_AUTOCORR_THRESHOLD is "not
 * periodic enough". Same decision rule as server/prosody.py::_frame_f0 —
 * only the lags actually needed are computed (the numpy version correlates
 * over every lag and then slices).
 */
export function frameF0(frame: Float32Array, sr: number): number | null {
  const n = frame.length;
  if (n === 0) return null;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += frame[i];
  mean /= n;
  const x = new Float64Array(n);
  let energy = 0;
  for (let i = 0; i < n; i++) {
    x[i] = frame[i] - mean;
    energy += x[i] * x[i];
  }
  if (energy <= 0) return null;
  const minLag = Math.floor(sr / F0_MAX_HZ);
  let maxLag = Math.floor(sr / F0_MIN_HZ);
  maxLag = Math.min(maxLag, n - 1);
  if (maxLag <= minLag) return null;
  let peakLag = -1;
  let peak = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let c = 0;
    for (let i = 0; i + lag < n; i++) c += x[i] * x[i + lag];
    // Strict > keeps the FIRST maximum, matching numpy argmax on ties.
    if (c > peak) {
      peak = c;
      peakLag = lag;
    }
  }
  if (peakLag <= 0) return null;
  if (peak / energy < VOICED_AUTOCORR_THRESHOLD) return null;
  return sr / peakLag;
}

export interface PitchEstimate {
  /** Median F0 over voiced frames; null when < MIN_VOICED_FRACTION voiced. */
  f0Median: number | null;
  /** Population std-dev of the voiced F0s (numpy's default ddof=0); null with f0Median. */
  f0Std: number | null;
  voicedFraction: number;
}

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function pitchFromF0s(f0s: number[], nFrames: number): PitchEstimate {
  if (nFrames === 0) return { f0Median: null, f0Std: null, voicedFraction: 0 };
  const voicedFraction = f0s.length / nFrames;
  if (voicedFraction < MIN_VOICED_FRACTION || f0s.length === 0) {
    return { f0Median: null, f0Std: null, voicedFraction };
  }
  const mean = f0s.reduce((a, b) => a + b, 0) / f0s.length;
  const variance = f0s.reduce((a, b) => a + (b - mean) * (b - mean), 0) / f0s.length;
  return { f0Median: median(f0s), f0Std: Math.sqrt(variance), voicedFraction };
}

/** Port of server/prosody.py::estimate_pitch. */
export function estimatePitch(samples: Float32Array, sr: number): PitchEstimate {
  if (sr <= 0 || samples.length === 0) {
    return { f0Median: null, f0Std: null, voicedFraction: 0 };
  }
  const frameLen = Math.max(1, Math.floor((sr * FRAME_MS) / 1000));
  const hop = Math.max(1, Math.floor((sr * HOP_MS) / 1000));
  if (samples.length < frameLen) {
    return { f0Median: null, f0Std: null, voicedFraction: 0 };
  }
  const f0s: number[] = [];
  let nFrames = 0;
  for (let start = 0; start + frameLen <= samples.length; start += hop) {
    nFrames += 1;
    const f0 = frameF0(samples.subarray(start, start + frameLen), sr);
    if (f0 !== null) f0s.push(f0);
  }
  return pitchFromF0s(f0s, nFrames);
}

/** Frames of pitch analysis between yields to the event loop in
 *  `estimatePitchAsync` (~0.5 s of audio; a few ms of work on a JIT, tens
 *  on Hermes). */
export const PITCH_YIELD_EVERY_FRAMES = 50;

export interface AsyncPitchOptions {
  /** Yield after this many frames (default PITCH_YIELD_EVERY_FRAMES). */
  yieldEvery?: number;
  /** How to yield (default: a 0 ms timer, i.e. a macrotask boundary). */
  sleep?: (ms: number) => Promise<void>;
}

const defaultSleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/**
 * `estimatePitch`, numerically identical, but cooperative: the per-frame
 * autocorrelation is the one O(n·lags) loop in the live path, and on the
 * phone it shares the JS thread with the mic callbacks, the WebSocket and
 * the UI. Yielding every few dozen frames keeps audio flowing while a long
 * turn is measured.
 */
export async function estimatePitchAsync(
  samples: Float32Array,
  sr: number,
  opts: AsyncPitchOptions = {},
): Promise<PitchEstimate> {
  if (sr <= 0 || samples.length === 0) {
    return { f0Median: null, f0Std: null, voicedFraction: 0 };
  }
  const frameLen = Math.max(1, Math.floor((sr * FRAME_MS) / 1000));
  const hop = Math.max(1, Math.floor((sr * HOP_MS) / 1000));
  if (samples.length < frameLen) {
    return { f0Median: null, f0Std: null, voicedFraction: 0 };
  }
  const yieldEvery = Math.max(1, opts.yieldEvery ?? PITCH_YIELD_EVERY_FRAMES);
  const sleep = opts.sleep ?? defaultSleep;
  const f0s: number[] = [];
  let nFrames = 0;
  for (let start = 0; start + frameLen <= samples.length; start += hop) {
    nFrames += 1;
    const f0 = frameF0(samples.subarray(start, start + frameLen), sr);
    if (f0 !== null) f0s.push(f0);
    if (nFrames % yieldEvery === 0) await sleep(0);
  }
  return pitchFromF0s(f0s, nFrames);
}

/** The numbers `TurnLocalEvent.prosody` carries (server/models/audio.py
 *  TurnProsody). Every field nullable: on-device measurement is best-effort. */
export interface TurnProsody {
  rms_dbfs: number | null;
  pitch_hz: number | null;
  speech_rate: number | null;
}

/**
 * Measure one finalized turn. `text` (from on-device STT) and the turn's
 * duration give the speech-rate proxy (words / second — the same words-over-
 * duration definition server/prosody.py::label_turns uses); null when there is
 * no text or no duration to divide by, never 0-as-a-guess.
 */
export function turnProsody(
  samples: Float32Array,
  sr: number,
  text: string,
  durationSeconds: number,
): TurnProsody {
  const dbfs = rmsDbfs(samples);
  const { f0Median } = estimatePitch(samples, sr);
  return assembleProsody(dbfs, f0Median, text, durationSeconds);
}

function assembleProsody(
  dbfs: number,
  f0Median: number | null,
  text: string,
  durationSeconds: number,
): TurnProsody {
  const words = text.trim() === "" ? 0 : text.trim().split(/\s+/).length;
  const speechRate =
    words > 0 && durationSeconds > 0 ? words / durationSeconds : null;
  return {
    rms_dbfs: Number.isFinite(dbfs) ? Math.round(dbfs * 100) / 100 : null,
    pitch_hz: f0Median === null ? null : Math.round(f0Median * 100) / 100,
    speech_rate: speechRate === null ? null : Math.round(speechRate * 1000) / 1000,
  };
}

/** Pitch is measured over at most the LAST this-many seconds of a turn in
 *  the live path: the per-frame autocorrelation costs ~100k multiply-adds a
 *  frame (100 frames/s), so an unbounded monologue would stall the JS
 *  thread for seconds; the median F0 of the last few seconds is the same
 *  delivery signal. Loudness is still over the whole turn (O(n)). */
export const LIVE_MAX_PITCH_SECONDS = 6;

export interface AsyncProsodyOptions extends AsyncPitchOptions {
  /** Cap on the pitch window (seconds from the END of the turn);
   *  undefined/Infinity = whole turn. */
  maxPitchSeconds?: number;
}

/** `turnProsody` for the live loop: same numbers, cooperative and bounded. */
export async function turnProsodyAsync(
  samples: Float32Array,
  sr: number,
  text: string,
  durationSeconds: number,
  opts: AsyncProsodyOptions = {},
): Promise<TurnProsody> {
  const dbfs = rmsDbfs(samples);
  const cap = opts.maxPitchSeconds;
  const window =
    cap !== undefined && Number.isFinite(cap) && cap > 0 && samples.length > cap * sr
      ? samples.subarray(samples.length - Math.round(cap * sr))
      : samples;
  const { f0Median } = await estimatePitchAsync(window, sr, opts);
  return assembleProsody(dbfs, f0Median, text, durationSeconds);
}
