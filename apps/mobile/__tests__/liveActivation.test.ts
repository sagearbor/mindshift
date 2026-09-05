/**
 * src/live/activation.ts — the vocal-activation classifier. Parity with the
 * Python extractor that trained it: __tests__/fixtures/activation_parity.json
 * was produced by server/prosody.py + tmp/ravdess/scripts/extract_features.py's
 * feature code on a deterministic synthetic signal; the TS port must land on
 * the same features and probability.
 */
import fixture from "./fixtures/activation_parity.json";
import {
  ACTIVATION_LEVELS,
  ACTIVATION_MODEL,
  ACTIVATION_WINDOW_SECONDS,
  activationFeatures,
  activationLevel,
  activationProbability,
  turnActivationAsync,
} from "../src/live/activation";

/** Rebuild the fixture's signal exactly as numpy did: amp·sin(2π·f·i/sr)
 *  with a running sample index across segments, cast to float32. */
function synth(): { samples: Float32Array; sr: number } {
  const sr = fixture.signal.sr as number;
  const segs = fixture.signal.segments as [number, number, number][];
  const total = segs.reduce((n, [, , sec]) => n + Math.floor(sec * sr), 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const [f, amp, sec] of segs) {
    const n = Math.floor(sec * sr);
    for (let k = 0; k < n; k++) {
      const i = offset + k;
      out[i] = f > 0 ? amp * Math.sin((2 * Math.PI * f * i) / sr) : 0;
    }
    offset += n;
  }
  return { samples: out, sr };
}

describe("activation features — parity with the Python extractor", () => {
  it("lands on the fixture's features (F0 stats, log-energy stats, durations)", () => {
    const { samples, sr } = synth();
    const f = activationFeatures(samples, sr)!;
    const exp = fixture.features as Record<string, number>;
    expect(f.voiced_duration_s).toBeCloseTo(exp.voiced_duration_s, 6);
    expect(f.unvoiced_duration_s).toBeCloseTo(exp.unvoiced_duration_s, 6);
    expect(f.energy_db_mean).toBeCloseTo(exp.energy_db_mean, 2);
    expect(f.energy_db_max).toBeCloseTo(exp.energy_db_max, 2);
    expect(f.energy_db_sd).toBeCloseTo(exp.energy_db_sd, 2);
    expect(Math.abs(f.f0_mean - exp.f0_mean)).toBeLessThan(0.5);
    expect(Math.abs(f.f0_max - exp.f0_max)).toBeLessThan(0.5);
    expect(Math.abs(f.f0_sd - exp.f0_sd)).toBeLessThan(1.0);
  });

  it("applies the exported weights to the same probability", () => {
    const { samples, sr } = synth();
    const p = activationProbability(activationFeatures(samples, sr)!);
    expect(Math.abs(p - (fixture.probability as number))).toBeLessThan(0.02);
    // The fixture model IS the shipped model.
    expect(ACTIVATION_MODEL.intercept).toBeCloseTo(fixture.model.intercept as number, 10);
  });

  it("async/windowed variant matches the sync path on a short turn and yields", async () => {
    const { samples, sr } = synth(); // 2.2 s < the 3.7 s window: no cropping
    let yields = 0;
    const r = await turnActivationAsync(samples, sr, { sleep: async () => { yields++; } });
    expect(r).not.toBeNull();
    expect(r!.probability).toBeCloseTo(activationProbability(activationFeatures(samples, sr)!), 10);
    expect(yields).toBeGreaterThan(0);
    expect(ACTIVATION_WINDOW_SECONDS).toBe(3.7);
  });

  it("windows a long turn to its last ACTIVATION_WINDOW_SECONDS", async () => {
    const sr = 16000;
    const long = new Float32Array(10 * sr); // 10 s of digital silence...
    for (let i = long.length - 2 * sr; i < long.length; i++) long[i] = 0.3 * Math.sin((2 * Math.PI * 200 * i) / sr); // ...then 2 s of tone at the end
    const r = await turnActivationAsync(long, sr, { sleep: async () => {} });
    // The window covers 3.7 s: 2 s voiced + 1.7 s unvoiced, never the 8 s of silence before it.
    expect(r!.features.voiced_duration_s).toBeCloseTo(2.0, 1);
    expect(r!.features.unvoiced_duration_s).toBeLessThan(1.8);
  });
});

describe("activationLevel ladder", () => {
  it("is conservative and monotone", () => {
    expect(activationLevel(0.5)).toBe(0);
    expect(activationLevel(0.75)).toBe(1);
    expect(activationLevel(0.9)).toBe(2);
    expect(activationLevel(0.99)).toBe(3);
    expect(ACTIVATION_LEVELS.map(([t]) => t)).toEqual([0.96, 0.88, 0.75]);
  });
});
