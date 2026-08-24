/**
 * src/live/prosody.ts — the cooperative, bounded variant the live loop uses
 * must produce the same numbers as the synchronous port (which the parity
 * tests in liveProsody.test.ts pin against server/prosody.py).
 */
import {
  estimatePitch,
  estimatePitchAsync,
  LIVE_MAX_PITCH_SECONDS,
  turnProsody,
  turnProsodyAsync,
} from "../src/live/prosody";
import { noiseF32, sineF32 } from "../src/live/testing/synth";

const SR = 16000;

function voiced(seconds: number, hz: number): Float32Array {
  const tone = sineF32(hz, seconds, 0.4);
  const noise = noiseF32(seconds, 0.01, 7);
  const out = new Float32Array(tone.length);
  for (let i = 0; i < out.length; i++) out[i] = tone[i] + noise[i];
  return out;
}

describe("estimatePitchAsync", () => {
  it("matches estimatePitch exactly and yields to the event loop while working", async () => {
    const pcm = voiced(2.0, 180);
    let yields = 0;
    const sleep = async () => {
      yields += 1;
    };
    const sync = estimatePitch(pcm, SR);
    const async = await estimatePitchAsync(pcm, SR, { yieldEvery: 25, sleep });
    expect(async).toEqual(sync);
    // 2 s at a 10 ms hop = ~197 frames -> 7 yields at every 25.
    expect(yields).toBeGreaterThanOrEqual(7);
  });

  it("handles the degenerate inputs like the sync version", async () => {
    expect(await estimatePitchAsync(new Float32Array(0), SR)).toEqual(estimatePitch(new Float32Array(0), SR));
    expect(await estimatePitchAsync(new Float32Array(100), SR)).toEqual(estimatePitch(new Float32Array(100), SR));
  });
});

describe("turnProsodyAsync", () => {
  it("equals turnProsody when the turn is within the pitch window", async () => {
    const pcm = voiced(1.5, 200);
    expect(await turnProsodyAsync(pcm, SR, "one two three", 1.5, { maxPitchSeconds: LIVE_MAX_PITCH_SECONDS })).toEqual(
      turnProsody(pcm, SR, "one two three", 1.5),
    );
  });

  it("measures pitch over the LAST maxPitchSeconds of a long turn, loudness over all of it", async () => {
    // 4 s at 120 Hz then 2 s at 240 Hz: the tail wins for pitch.
    const head = voiced(4.0, 120);
    const tail = voiced(2.0, 240);
    const pcm = new Float32Array(head.length + tail.length);
    pcm.set(head, 0);
    pcm.set(tail, head.length);
    const bounded = await turnProsodyAsync(pcm, SR, "words", 6, { maxPitchSeconds: 2 });
    expect(bounded.pitch_hz).toBeGreaterThan(200);
    expect(bounded.rms_dbfs).toEqual(turnProsody(pcm, SR, "words", 6).rms_dbfs);
    const whole = await turnProsodyAsync(pcm, SR, "words", 6, {});
    expect(whole).toEqual(turnProsody(pcm, SR, "words", 6));
    expect(whole.pitch_hz).toBeLessThan(200); // median over the 120 Hz majority
  });
});
