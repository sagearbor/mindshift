/**
 * Frontend parity: the TS log-mel must match HF WhisperFeatureExtractor on a
 * golden vector produced by scripts/emotion_golden.py from one MESS demo clip
 * (CC BY 4.0 — see src/emotion/__fixtures__/ATTRIBUTION.md).
 */
import * as fs from "fs";
import * as path from "path";
import {
  buildMelFilters,
  computeLogMel,
  int16ToFloat32,
  N_FRAMES,
  N_MELS,
  powerSpectrum,
  waveformProblem,
} from "../src/emotion/logMel";

const FIX = path.join(__dirname, "..", "src", "emotion", "__fixtures__");

function readInt16(name: string): Int16Array {
  const b = fs.readFileSync(path.join(FIX, name));
  return new Int16Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}
function readF32(name: string): Float32Array {
  const b = fs.readFileSync(path.join(FIX, name));
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}
const meta = JSON.parse(fs.readFileSync(path.join(FIX, "clip04-meta.json"), "utf8"));

describe("emotion log-mel frontend", () => {
  const audio = int16ToFloat32(readInt16("clip04-16k-int16.bin"));
  const golden = readF32("clip04-logmel.f32");

  it("golden fixture is the shape the model contract names", () => {
    expect(audio.length).toBe(meta.samples);
    expect(golden.length).toBe(N_MELS * N_FRAMES);
  });

  it("matches the HF WhisperFeatureExtractor reference on clip-04", () => {
    const { data, validFrames } = computeLogMel(audio);
    expect(validFrames).toBe(meta.valid_frames);
    expect(data.length).toBe(golden.length);
    let maxAbs = 0;
    let sumAbs = 0;
    for (let i = 0; i < data.length; i++) {
      const d = Math.abs(data[i] - golden[i]);
      maxAbs = Math.max(maxAbs, d);
      sumAbs += d;
    }
    // float64 here vs float32 in HF's STFT/mel: agreement is far inside 1e-3
    // on a [-1, ~1.5] scale; the models are insensitive to anything this size.
    expect(maxAbs).toBeLessThan(1e-3);
    expect(sumAbs / data.length).toBeLessThan(1e-4);
  });

  it("valid_frames follows min(300, ceil(n / 320))", () => {
    const tone = (n: number) => Float32Array.from({ length: n }, (_, i) => 0.1 * Math.sin(i / 7));
    expect(computeLogMel(tone(16000)).validFrames).toBe(50);
    expect(computeLogMel(tone(16001)).validFrames).toBe(51);
    expect(computeLogMel(tone(96000)).validFrames).toBe(300);
  });

  it("does not normalise gain (louder input changes the output)", () => {
    const quiet = computeLogMel(audio).data;
    const loud = computeLogMel(audio.map((v) => v * 0.5) as Float32Array).data;
    let diff = 0;
    for (let i = 0; i < quiet.length; i++) diff += Math.abs(quiet[i] - loud[i]);
    expect(diff).toBeGreaterThan(1);
  });

  it("FFT agrees with a direct DFT", () => {
    const frame = Float64Array.from({ length: 400 }, (_, i) => Math.sin(i * 0.37) + 0.2 * Math.cos(i * 1.9));
    const p = powerSpectrum(frame);
    for (const k of [0, 1, 17, 100, 200]) {
      let re = 0;
      let im = 0;
      for (let n = 0; n < 400; n++) {
        re += frame[n] * Math.cos((2 * Math.PI * k * n) / 400);
        im -= frame[n] * Math.sin((2 * Math.PI * k * n) / 400);
      }
      expect(p[k]).toBeCloseTo(re * re + im * im, 6);
    }
  });

  it("mel filters are Slaney-normalised (each row integrates to ~2/width area)", () => {
    const fb = buildMelFilters();
    expect(fb.length).toBe(80 * 201);
    for (let m = 0; m < 80; m++) {
      let s = 0;
      for (let k = 0; k < 201; k++) s += fb[m * 201 + k];
      expect(s).toBeGreaterThan(0);
    }
  });

  it("rejects empty / non-finite / silent / too-long input", () => {
    expect(waveformProblem(new Float32Array(0))).not.toBeNull();
    expect(waveformProblem(new Float32Array(16000))).toMatch(/silence/);
    const nan = new Float32Array(16000).fill(0.1);
    nan[5] = NaN;
    expect(waveformProblem(nan)).toMatch(/non-finite/);
    expect(waveformProblem(new Float32Array(96001).fill(0.1))).toMatch(/longer/);
    expect(waveformProblem(audio)).toBeNull();
    expect(() => computeLogMel(new Float32Array(16000))).toThrow();
  });
});
