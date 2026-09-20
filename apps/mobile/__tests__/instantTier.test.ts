/**
 * The instant tier's arithmetic, on signals whose answer we know in advance.
 * The corpus-level claims live in instantTier.parity.test.ts (does it agree
 * with openSMILE?) and instantTier.eval.test.ts (does it tell angry from
 * happy on real audio?); this file is the floor those two stand on.
 */
import {
  F0_MAX_HZ,
  F0_MIN_HZ,
  FRAME_MS,
  HOP_MS,
  INSTANT_FEATURE_NAMES,
  INSTANT_MODEL_METRICS,
  MIN_WINDOW_SEC,
  SEMITONE_REF_HZ,
  extractInstantFeatures,
  instantHeatScore,
  scoreInstantFeatures,
  type InstantFeatures,
} from "../src/live/instantTier";

const SR = 16000;

function tone(hz: number, seconds: number, amp = 0.3, sr = SR): Float32Array {
  const out = new Float32Array(Math.round(seconds * sr));
  for (let i = 0; i < out.length; i++) out[i] = amp * Math.sin((2 * Math.PI * hz * i) / sr);
  return out;
}

/** A crude glottal buzz: a few harmonics, so the pitch tracker sees something
 *  closer to speech than a pure sine. */
function buzz(f0: number, seconds: number, amp = 0.3, sr = SR): Float32Array {
  const out = new Float32Array(Math.round(seconds * sr));
  for (let i = 0; i < out.length; i++) {
    let v = 0;
    for (let h = 1; h <= 6; h++) v += Math.sin((2 * Math.PI * f0 * h * i) / sr) / h;
    out[i] = amp * v;
  }
  return out;
}

function noise(seconds: number, amp = 0.2, sr = SR): Float32Array {
  const out = new Float32Array(Math.round(seconds * sr));
  // deterministic LCG — a flaky test is worse than a weak one
  let s = 12345;
  for (let i = 0; i < out.length; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    out[i] = amp * (s / 0x3fffffff - 1);
  }
  return out;
}

describe("instant tier — measurement", () => {
  it("measures a 2 s window with the geometry the model was fitted on", () => {
    const heat = instantHeatScore(buzz(140, 2), SR);
    // 2 s at 25 ms frames / 10 ms hop
    expect(heat.frames).toBe(Math.floor((2 * SR - 0.025 * SR) / (0.01 * SR)) + 1);
    expect(FRAME_MS).toBe(25);
    expect(HOP_MS).toBe(10);
    expect(heat.score).not.toBeNull();
    expect(heat.score as number).toBeGreaterThan(0);
    expect(heat.score as number).toBeLessThan(1);
  });

  it("has no opinion about a window too short to measure", () => {
    const heat = instantHeatScore(buzz(140, MIN_WINDOW_SEC / 2), SR);
    expect(heat.score).toBeNull();
    expect(heat.frames).toBe(0);
    for (const name of INSTANT_FEATURE_NAMES) expect(heat.features[name]).toBeNull();
  });

  it("reports null rather than a fabricated pitch when nothing is voiced", () => {
    const f = extractInstantFeatures(noise(2), SR);
    expect(f.f0SemitoneP80).toBeNull();
    // the voiced-only spectral descriptors go with it
    expect(f.alphaRatioV).toBeNull();
    expect(f.slopeV0to500).toBeNull();
    // and the unvoiced ones are the ones that DO have something to say
    expect(f.alphaRatioUV).not.toBeNull();
    expect(f.hammarbergUV).not.toBeNull();
  });

  it("still scores a window whose pitch it could not measure", () => {
    // a null feature falls back to the training mean (z = 0) rather than
    // dragging the score toward an invented value
    const heat = instantHeatScore(noise(2), SR);
    expect(heat.features.f0SemitoneP80).toBeNull();
    expect(heat.score).not.toBeNull();
    expect(Number.isFinite(heat.score as number)).toBe(true);
  });

  it("recovers the pitch of a synthetic buzz to within a semitone", () => {
    for (const f0 of [90, 140, 220]) {
      const f = extractInstantFeatures(buzz(f0, 2), SR);
      expect(f.f0SemitoneP80).not.toBeNull();
      const expected = 12 * Math.log2(f0 / SEMITONE_REF_HZ);
      expect(f.f0SemitoneP80 as number).toBeCloseTo(expected, 0);
    }
  });

  it("refuses pitches outside the range it claims to track", () => {
    // 30 Hz is below F0_MIN_HZ; whatever it does, it must not report it
    const below = extractInstantFeatures(tone(30, 2), SR);
    if (below.f0SemitoneP80 !== null) {
      const hz = SEMITONE_REF_HZ * 2 ** ((below.f0SemitoneP80 as number) / 12);
      expect(hz).toBeGreaterThanOrEqual(F0_MIN_HZ - 1);
      expect(hz).toBeLessThanOrEqual(F0_MAX_HZ + 1);
    }
  });

  it("hears a louder window as louder", () => {
    const quiet = extractInstantFeatures(buzz(140, 2, 0.05), SR);
    const loud = extractInstantFeatures(buzz(140, 2, 0.5), SR);
    expect(loud.loudMean as number).toBeGreaterThan(quiet.loudMean as number);
    expect(loud.loudP20 as number).toBeGreaterThan(quiet.loudP20 as number);
  });

  it("hears a steady window as steadier than one that swings", () => {
    const steady = buzz(140, 2, 0.3);
    const swinging = buzz(140, 2, 0.3);
    for (let i = 0; i < swinging.length; i++) {
      swinging[i] *= 0.5 + 0.5 * Math.sin((2 * Math.PI * 3 * i) / SR);
    }
    const a = extractInstantFeatures(steady, SR);
    const b = extractInstantFeatures(swinging, SR);
    expect(b.loudStddevNorm as number).toBeGreaterThan(a.loudStddevNorm as number);
    expect(b.loudPctlRange as number).toBeGreaterThan(a.loudPctlRange as number);
  });

  it("hears a bright signal as spectrally tilted towards the top", () => {
    // Both tones read as voiced — a sine is periodic, and a 3 kHz one is
    // periodic at the 500 Hz lag too — so the voiced alpha ratio is the one
    // with something to say here.
    const low = extractInstantFeatures(tone(200, 2), SR);
    const high = extractInstantFeatures(tone(3000, 2), SR);
    // alphaRatio here is 10*log10(E[1-5k] / E[50-1000])
    expect(high.alphaRatioV as number).toBeGreaterThan(low.alphaRatioV as number);
  });

  it("is deterministic", () => {
    const pcm = buzz(160, 2);
    const a = instantHeatScore(pcm, SR);
    const b = instantHeatScore(pcm, SR);
    expect(b.features).toEqual(a.features);
    expect(b.score).toBe(a.score);
  });

  it("reads Int16 PCM the same way it reads Float32", () => {
    const f32 = buzz(150, 2);
    const i16 = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) i16[i] = Math.round(f32[i] * 32768);
    const a = instantHeatScore(f32, SR).score as number;
    const b = instantHeatScore(i16, SR).score as number;
    expect(b).toBeCloseTo(a, 3);
  });
});

describe("instant tier — the model", () => {
  it("ships a model fitted on the features this file's extractor produces", () => {
    expect(INSTANT_MODEL_METRICS.xcorp_auc_angry_vs_happy).toBeGreaterThanOrEqual(0.78);
    // the signal it replaces: loudness alone manages 0.72 on the same split
    expect(INSTANT_MODEL_METRICS.xcorp_auc_angry_vs_happy).toBeGreaterThan(0.72);
    expect(INSTANT_MODEL_METRICS.n_train).toBeGreaterThan(1000);
    expect(INSTANT_MODEL_METRICS.n_test).toBeGreaterThan(1000);
  });

  it("treats every unmeasurable feature as no evidence, not as zero", () => {
    const all: InstantFeatures = Object.fromEntries(
      INSTANT_FEATURE_NAMES.map((n) => [n, null]),
    ) as InstantFeatures;
    const base = scoreInstantFeatures(all);
    // with nothing measured the score is the model's prior, exactly
    expect(base).toBeGreaterThan(0);
    expect(base).toBeLessThan(1);
    // and a NaN is treated the same as a null, never propagated
    const poisoned = { ...all, loudMean: NaN } as InstantFeatures;
    expect(scoreInstantFeatures(poisoned)).toBe(base);
  });

  it("stays inside the per-window latency budget", () => {
    const pcm = buzz(150, 2);
    for (let i = 0; i < 5; i++) instantHeatScore(pcm, SR); // warm up
    const t0 = Date.now();
    const runs = 20;
    for (let i = 0; i < runs; i++) instantHeatScore(pcm, SR);
    const ms = (Date.now() - t0) / runs;
    // budget is 40 ms per 2 s window on a phone-class engine; node on a dev
    // Mac measures ~11 ms. A generous ceiling here catches an algorithmic
    // regression without failing on a loaded CI box.
    expect(ms).toBeLessThan(120);
  });
});
