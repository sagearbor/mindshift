import {
  concatInt16,
  downmixToMono,
  float32ToInt16,
  StreamingResampler,
} from "../src/utils/audio";

describe("float32ToInt16", () => {
  it("scales [-1, 1] floats to int16 sample values", () => {
    const input = new Float32Array([0, 0.25, -0.25, 1, -1]);
    const out = float32ToInt16(input);
    // 0.25 * 32767 = 8191.75 -> rounds to 8192
    expect(Array.from(out)).toEqual([0, 8192, -8192, 32767, -32767]);
  });

  it("clamps out-of-range values instead of wrapping", () => {
    const input = new Float32Array([2, -2, 1.0001, -1.0001]);
    const out = float32ToInt16(input);
    expect(Array.from(out)).toEqual([32767, -32767, 32767, -32767]);
  });

  it("produces 2 bytes per sample", () => {
    const out = float32ToInt16(new Float32Array(1600));
    expect(out.byteLength).toBe(3200);
  });
});

describe("StreamingResampler", () => {
  it("passes input through unchanged when rates match", () => {
    const rs = new StreamingResampler(16000, 16000);
    const input = new Float32Array([0.1, 0.2, 0.3]);
    expect(rs.process(input)).toBe(input);
    expect(rs.flush()).toHaveLength(0);
  });

  it("downsamples 48kHz -> 16kHz by taking every 3rd position (linear interp)", () => {
    const rs = new StreamingResampler(48000, 16000);
    const out = rs.process(new Float32Array([1, 2, 3, 4, 5, 6]));
    // Read positions 0 and 3 land exactly on input samples.
    expect(Array.from(out)).toEqual([1, 4]);
    expect(rs.flush()).toHaveLength(0); // position 6 is past the input
  });

  it("upsamples 8kHz -> 16kHz by linear interpolation, tail on flush", () => {
    const rs = new StreamingResampler(8000, 16000);
    const out = rs.process(new Float32Array([0, 1]));
    expect(Array.from(out)).toEqual([0, 0.5]);
    // Position 1.0 needs no right neighbour — flush emits it (clamped).
    expect(Array.from(rs.flush())).toEqual([1]);
  });

  it("carries the fractional read position across process() calls", () => {
    // 40kHz -> 16kHz is ratio 2.5: global read positions 0, 2.5, 5, ...
    const rs = new StreamingResampler(40000, 16000);
    const first = rs.process(new Float32Array([0, 1, 2]));
    expect(Array.from(first)).toEqual([0]);
    // Position 2.5 straddles the chunk boundary: interpolates 2 and 3.
    const second = rs.process(new Float32Array([3, 4, 5, 6, 7]));
    expect(Array.from(second)).toEqual([2.5, 5]);
    // A stateless resampler restarting phase at 0 would have emitted 3, 5.5.
  });

  it("flush() returns the held-back tail samples", () => {
    const rs = new StreamingResampler(32000, 16000); // ratio 2
    const out = rs.process(new Float32Array([0, 1, 2, 3, 4]));
    expect(Array.from(out)).toEqual([0, 2]);
    expect(Array.from(rs.flush())).toEqual([4]);
    // After flush the resampler is reset.
    expect(rs.flush()).toHaveLength(0);
  });

  it("44.1kHz -> 16kHz: total output length within ±1 of round(total/ratio) across varying buffer sizes", () => {
    const ratio = 44100 / 16000; // 2.75625 — non-integer
    const rs = new StreamingResampler(44100, 16000);
    const totalInput = 44100; // 1 s
    const sizes = [997, 1601, 4410, 12345, 3200, 1]; // deliberately irregular
    let fed = 0;
    let totalOutput = 0;
    let i = 0;
    while (fed < totalInput) {
      const size = Math.min(sizes[i % sizes.length], totalInput - fed);
      totalOutput += rs.process(new Float32Array(size)).length;
      fed += size;
      i += 1;
    }
    totalOutput += rs.flush().length;
    expect(Math.abs(totalOutput - Math.round(totalInput / ratio))).toBeLessThanOrEqual(1);
  });

  it("44.1kHz -> 16kHz: resampling a ramp is continuous across buffer boundaries (no phase restart)", () => {
    const ratio = 44100 / 16000;
    const rs = new StreamingResampler(44100, 16000);
    const totalInput = 44100;
    // Global ramp x[n] = n * SLOPE, chopped into irregular chunks. Linear
    // interpolation of a linear signal is exact, so the output must be the
    // ramp k * ratio * SLOPE — any per-buffer phase restart shows up as an
    // oversized step (dropped fractional samples) at a chunk boundary.
    // SLOPE keeps values small so float32 quantization stays negligible.
    const SLOPE = 0.001;
    const sizes = [1000, 4410, 733, 8000, 2205, 999];
    const produced: number[] = [];
    let offset = 0;
    let i = 0;
    while (offset < totalInput) {
      const size = Math.min(sizes[i % sizes.length], totalInput - offset);
      const chunk = new Float32Array(size);
      for (let j = 0; j < size; j++) chunk[j] = (offset + j) * SLOPE;
      for (const v of rs.process(chunk)) produced.push(v);
      offset += size;
      i += 1;
    }
    for (const v of rs.flush()) produced.push(v);

    expect(produced.length).toBeGreaterThan(0);
    let minStep = Infinity;
    let maxStep = -Infinity;
    for (let k = 1; k < produced.length; k++) {
      const step = produced[k] - produced[k - 1];
      minStep = Math.min(minStep, step);
      maxStep = Math.max(maxStep, step);
    }
    // Monotonic, and every step is exactly one output period — a stateless
    // phase restart injects steps of ~2x the period at chunk boundaries.
    expect(minStep).toBeGreaterThan(0);
    expect(maxStep).toBeLessThanOrEqual(ratio * SLOPE * 1.05);
    // No cumulative drift: the k-th output sits at read position k * ratio.
    const last = produced.length - 1;
    expect(produced[last]).toBeCloseTo(last * ratio * SLOPE, 4);
  });

  it("rejects nonsensical sample rates", () => {
    expect(() => new StreamingResampler(0, 16000)).toThrow();
    expect(() => new StreamingResampler(16000, -1)).toThrow();
    expect(() => new StreamingResampler(NaN, 16000)).toThrow();
  });
});

describe("downmixToMono", () => {
  it("averages interleaved stereo frames", () => {
    const input = new Float32Array([0.2, 0.4, 0.6, 0.8]);
    const out = downmixToMono(input, 2);
    expect(out.length).toBe(2);
    expect(out[0]).toBeCloseTo(0.3);
    expect(out[1]).toBeCloseTo(0.7);
  });

  it("passes mono through untouched", () => {
    const input = new Float32Array([0.1, 0.2]);
    expect(downmixToMono(input, 1)).toBe(input);
  });
});

describe("concatInt16", () => {
  it("joins two buffers in order", () => {
    const out = concatInt16(new Int16Array([1, 2]), new Int16Array([3]));
    expect(Array.from(out)).toEqual([1, 2, 3]);
  });

  it("returns the non-empty side when one is empty", () => {
    const a = new Int16Array([1]);
    expect(concatInt16(a, new Int16Array(0))).toBe(a);
    expect(concatInt16(new Int16Array(0), a)).toBe(a);
  });
});
