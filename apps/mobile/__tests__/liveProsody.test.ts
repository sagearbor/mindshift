/**
 * src/live/prosody.ts — the phone-side port of server/prosody.py. The
 * expectations mirror server/tests/test_prosody.py's key-free layer so a
 * drift between the two runtimes shows up here.
 */
import {
  estimatePitch,
  frameF0,
  MIN_VOICED_FRACTION,
  rmsDbfs,
  rmsEnergy,
  turnProsody,
} from "../src/live/prosody";
import { noiseF32, sineF32, SR } from "../src/live/testing/synth";

describe("estimatePitch", () => {
  it("finds a 200 Hz tone within ±15 Hz, almost entirely voiced, barely varying", () => {
    const { f0Median, f0Std, voicedFraction } = estimatePitch(sineF32(200, 1.0), SR);
    expect(f0Median).not.toBeNull();
    expect(Math.abs((f0Median as number) - 200)).toBeLessThanOrEqual(15);
    expect(voicedFraction).toBeGreaterThan(0.9);
    expect(f0Std as number).toBeLessThan(5);
  });

  it("reports an honest null for broadband noise", () => {
    const { f0Median, f0Std, voicedFraction } = estimatePitch(noiseF32(1.0, 0.3, 7), SR);
    expect(f0Median).toBeNull();
    expect(f0Std).toBeNull();
    expect(voicedFraction).toBeLessThan(MIN_VOICED_FRACTION);
  });

  it("reports null for silence and for too-short input", () => {
    expect(estimatePitch(new Float32Array(SR), SR)).toEqual({
      f0Median: null,
      f0Std: null,
      voicedFraction: 0,
    });
    expect(estimatePitch(new Float32Array(100), SR).f0Median).toBeNull();
  });

  it("orders a lower tone below a higher one", () => {
    const low = estimatePitch(sineF32(110, 0.5), SR).f0Median as number;
    const high = estimatePitch(sineF32(300, 0.5), SR).f0Median as number;
    expect(low).toBeLessThan(high);
    expect(Math.abs(low - 110)).toBeLessThanOrEqual(15);
    expect(Math.abs(high - 300)).toBeLessThanOrEqual(15);
  });
});

describe("frameF0", () => {
  it("is null for an all-zero or DC-only frame (no energy after mean removal)", () => {
    expect(frameF0(new Float32Array(640), SR)).toBeNull();
    expect(frameF0(new Float32Array(640).fill(0.3), SR)).toBeNull();
  });
});

describe("energy", () => {
  it("rmsEnergy of a sine is amp/√2 and rmsDbfs follows", () => {
    const rms = rmsEnergy(sineF32(200, 1.0, 0.5));
    expect(Math.abs(rms - 0.5 / Math.SQRT2)).toBeLessThan(0.002);
    expect(Math.abs(rmsDbfs(sineF32(200, 1.0, 0.5)) - 20 * Math.log10(0.5 / Math.SQRT2))).toBeLessThan(0.05);
    expect(rmsDbfs(new Float32Array(10))).toBe(-Infinity);
    expect(rmsEnergy(new Float32Array(0))).toBe(0);
  });

  it("a louder window measures louder", () => {
    expect(rmsDbfs(sineF32(200, 0.5, 0.5))).toBeGreaterThan(rmsDbfs(sineF32(200, 0.5, 0.05)));
  });
});

describe("turnProsody", () => {
  it("speech rate is words / duration, rounded to 3 places", () => {
    const p = turnProsody(sineF32(200, 2.0), SR, "one two three four", 2.0);
    expect(p.speech_rate).toBe(2);
    expect(p.pitch_hz).not.toBeNull();
    expect(p.rms_dbfs).not.toBeNull();
  });

  it("never fabricates: null rate without text, null pitch/loudness for silence", () => {
    const p = turnProsody(new Float32Array(SR), SR, "", 1.0);
    expect(p).toEqual({ rms_dbfs: null, pitch_hz: null, speech_rate: null });
  });
});
