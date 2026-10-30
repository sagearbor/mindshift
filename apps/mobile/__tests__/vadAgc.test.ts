/**
 * Input gain normalisation (AGC) in front of the VAD: quiet rooms (a phone
 * far from the talker) left Silero under its threshold — measured on the
 * corpus, the -18 dB-attenuated DEV clip SBC033 went from 0% to 62% of its
 * speech detected with the gain stage. Only the VAD's input is scaled; the
 * embedder / prosody keep the raw audio (loudness feeds the heat lane).
 */
import { AgcVad, GainNormalizer, VAD_AGC_DEFAULTS, type FrameVad } from "../src/live/vad";

const N = 512;
function tone(amp: number): Float32Array {
  const f = new Float32Array(N);
  for (let i = 0; i < N; i++) f[i] = amp * Math.sin((2 * Math.PI * 220 * i) / 16000);
  return f;
}
const rms = (f: Float32Array) => Math.sqrt(f.reduce((a, x) => a + x * x, 0) / f.length);
const db = (x: number) => 20 * Math.log10(x);

describe("GainNormalizer", () => {
  it("leaves speech already at the target level alone (gain never below 1)", () => {
    const g = new GainNormalizer();
    const loud = tone(0.5); // ~ -9 dBFS rms, above the -16 dBFS target
    let out = loud;
    for (let i = 0; i < 50; i++) out = g.process(loud);
    expect(g.gain).toBe(1);
    expect(rms(out)).toBeCloseTo(rms(loud), 6);
  });

  it("lifts quiet speech toward the target, capped at the max gain", () => {
    const g = new GainNormalizer();
    const quiet = tone(0.002); // ~ -57 dBFS rms: needs more than the cap
    let out = quiet;
    for (let i = 0; i < 500; i++) out = g.process(quiet); // release 3 dB/s: ~10 s to reach the cap
    expect(db(g.gain)).toBeCloseTo(VAD_AGC_DEFAULTS.maxGainDb, 1);
    expect(db(rms(out)) - db(rms(quiet))).toBeCloseTo(VAD_AGC_DEFAULTS.maxGainDb, 1);
  });

  it("a moderately quiet talker is brought to the target", () => {
    const g = new GainNormalizer();
    const q = tone(0.05); // ~ -29 dBFS
    let out = q;
    for (let i = 0; i < 500; i++) out = g.process(q);
    expect(db(rms(out))).toBeCloseTo(VAD_AGC_DEFAULTS.targetDbfs, 0);
  });

  it("a loud burst pulls the gain down at once, then it recovers slowly", () => {
    const g = new GainNormalizer();
    for (let i = 0; i < 500; i++) g.process(tone(0.01));
    g.process(tone(0.5));
    g.process(tone(0.5));
    expect(g.gain).toBeLessThan(1.5);
    for (let i = 0; i < 94; i++) g.process(tone(0.01)); // ~3 s
    const after = db(g.gain);
    expect(after).toBeGreaterThan(0);
    expect(after).toBeLessThan(VAD_AGC_DEFAULTS.releaseDbPerSec * 3);
  });

  it("never clips", () => {
    const g = new GainNormalizer();
    for (let i = 0; i < 500; i++) g.process(tone(0.01));
    const spike = tone(0.01);
    spike[10] = 0.9;
    const out = g.process(spike);
    expect(Math.max(...Array.from(out).map(Math.abs))).toBeLessThanOrEqual(1);
  });

  it("does not mutate its input and resets to unity", () => {
    const g = new GainNormalizer();
    const q = tone(0.01);
    const copy = Float32Array.from(q);
    for (let i = 0; i < 20; i++) g.process(q);
    expect(Array.from(q)).toEqual(Array.from(copy));
    g.reset();
    expect(g.gain).toBe(1);
  });
});

describe("AgcVad", () => {
  it("hands the inner VAD the gained frame and keeps its frame size", async () => {
    const seen: number[] = [];
    const inner: FrameVad = {
      frameSamples: N,
      isSpeech: async (f) => {
        seen.push(rms(f));
        return rms(f) > 0.05;
      },
      reset: () => undefined,
    };
    const vad = new AgcVad(inner);
    expect(vad.frameSamples).toBe(N);
    let last = false;
    for (let i = 0; i < 500; i++) last = await vad.isSpeech(tone(0.01));
    expect(last).toBe(true);
    expect(seen[seen.length - 1]).toBeGreaterThan(rms(tone(0.01)) * 10);
  });

  it("reset clears both the gain and the inner VAD", () => {
    let resets = 0;
    const inner: FrameVad = { frameSamples: N, isSpeech: async () => false, reset: () => void resets++ };
    const vad = new AgcVad(inner);
    vad.reset();
    expect(resets).toBe(1);
    expect(vad.gain).toBe(1);
  });
});
