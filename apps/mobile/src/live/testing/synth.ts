/**
 * Shared synthetic-PCM + fixture helpers for the src/live Jest suites. Lives
 * under src/live/testing so jest never mistakes it for a suite; nothing in the app graph imports it.
 */
import * as fs from "fs";
import * as path from "path";

export const SR = 16000;
export const INT16_FULL_SCALE = 32768;

export const FIXTURES_DIR = path.resolve(
  __dirname,
  "../../../../../server/tests/fixtures/policy_vectors",
);

export function loadFixture<T = unknown>(name: string): T {
  return JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, name), "utf8")) as T;
}

/** server/tests/fixtures/audio — the real recordings the parity tests use. */
export const AUDIO_FIXTURES_DIR = path.resolve(FIXTURES_DIR, "../audio");

/**
 * Read a canonical 16 kHz mono int16 RIFF wav into float32 in [-1, 1) via
 * `/ 32768` — the same arithmetic scripts/export_ecapa_onnx.py's reference
 * writer applies, so the two sides embed identical samples. Throws on any
 * other format (the fixtures are checked in that shape; a resampler here
 * would only hide a fixture regression).
 */
export function loadWav16k(file: string): Float32Array {
  const buf = fs.readFileSync(file);
  if (buf.toString("ascii", 0, 4) !== "RIFF" || buf.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error(`${file}: not a RIFF/WAVE file`);
  }
  let offset = 12;
  let channels = 0;
  let sampleRate = 0;
  let bits = 0;
  let data: Buffer | null = null;
  while (offset + 8 <= buf.length) {
    const id = buf.toString("ascii", offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    const body = offset + 8;
    if (id === "fmt ") {
      channels = buf.readUInt16LE(body + 2);
      sampleRate = buf.readUInt32LE(body + 4);
      bits = buf.readUInt16LE(body + 14);
    } else if (id === "data") {
      data = buf.subarray(body, Math.min(body + size, buf.length));
    }
    offset = body + size + (size % 2);
  }
  if (!data || channels !== 1 || sampleRate !== SR || bits !== 16) {
    throw new Error(`${file}: expected 16 kHz mono int16 wav (got ${channels}ch ${sampleRate}Hz ${bits}-bit)`);
  }
  const out = new Float32Array(Math.floor(data.length / 2));
  for (let i = 0; i < out.length; i++) out[i] = data.readInt16LE(i * 2) / INT16_FULL_SCALE;
  return out;
}

export interface VadStretch {
  seconds: number;
  dbfs: number | null;
}

export interface VadCase {
  name: string;
  sample_rate: number;
  tone_hz?: number;
  config: {
    floor_dbfs: number;
    frame_seconds: number;
    merge_gap_seconds: number;
    min_seconds: number;
  };
  signal: VadStretch[];
  expected: { start_s: number; end_s: number }[];
  tolerance_s: number;
}

/**
 * Bit-for-bit port of server/tests/test_vad_vectors.py::synthesize — int16
 * mono; per stretch a 150 Hz (default) sine with phase 0, amplitude
 * 32768 * 10^(dbfs/20) * sqrt(2), truncated toward zero like numpy's
 * astype(int16); `null` dbfs is digital silence. Sample count per stretch is
 * floor(seconds * sr), as Python's int() truncates.
 */
export function synthesizeCase(c: VadCase): Int16Array {
  const sr = c.sample_rate;
  const toneHz = c.tone_hz ?? 150.0;
  const parts: Int16Array[] = [];
  for (const s of c.signal) {
    const n = Math.floor(s.seconds * sr);
    const buf = new Int16Array(n);
    if (s.dbfs !== null) {
      const amp = INT16_FULL_SCALE * Math.pow(10, s.dbfs / 20) * Math.SQRT2;
      for (let i = 0; i < n; i++) {
        buf[i] = Math.trunc(Math.sin((2 * Math.PI * toneHz * i) / sr) * amp);
      }
    }
    parts.push(buf);
  }
  const total = parts.reduce((a, p) => a + p.length, 0);
  const out = new Int16Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

/** Float32 sine in [-1, 1]. */
export function sineF32(freq: number, seconds: number, amp = 0.5, sr = SR): Float32Array {
  const n = Math.floor(sr * seconds);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = amp * Math.sin((2 * Math.PI * freq * i) / sr);
  return out;
}

/** Int16 sine at a given RMS dBFS (same formula as the fixture generator). */
export function toneInt16(seconds: number, dbfs: number, hz = 150, sr = SR): Int16Array {
  return synthesizeCase({
    name: "adhoc",
    sample_rate: sr,
    tone_hz: hz,
    config: { floor_dbfs: -45, frame_seconds: 0.25, merge_gap_seconds: 0.3, min_seconds: 0.6 },
    signal: [{ seconds, dbfs }],
    expected: [],
    tolerance_s: 0.01,
  });
}

export function silenceInt16(seconds: number, sr = SR): Int16Array {
  return new Int16Array(Math.floor(seconds * sr));
}

/** Deterministic pseudo-random generator (mulberry32) for noise fixtures. */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Approximately Gaussian noise (sum of 12 uniforms). */
export function noiseF32(seconds: number, amp: number, seed = 0, sr = SR): Float32Array {
  const rnd = seededRandom(seed);
  const n = Math.floor(sr * seconds);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < 12; k++) s += rnd();
    out[i] = amp * (s - 6);
  }
  return out;
}

/** A unit vector along one axis, optionally perturbed (for voiceprint tests). */
export function unitVector(dim: number, axis: number, noise = 0, seed = 1): Float32Array {
  const rnd = seededRandom(seed);
  const v = new Float32Array(dim);
  v[axis % dim] = 1;
  if (noise > 0) for (let i = 0; i < dim; i++) v[i] += noise * (rnd() - 0.5);
  return v;
}
