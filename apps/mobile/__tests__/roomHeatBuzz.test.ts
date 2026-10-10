/**
 * The identity-free room-heat buzz (src/live/roomHeatBuzz.ts) and its DARK
 * wiring into the fast loop (FastLoopDeps.roomHeatBuzz, default off).
 */
jest.mock("../src/live/instantTier", () => {
  const actual = jest.requireActual("../src/live/instantTier");
  return {
    ...actual,
    // Every 2 s window reads as very hot: lets the loop test drive the buzz
    // without depending on what synthetic tones score.
    instantHeatScore: () => ({ score: 0.97, frames: 198, voicedFrames: 60 }),
  };
});

import { RoomHeatBuzz, ROOM_HEAT_BUZZ_DEFAULTS } from "../src/live/roomHeatBuzz";
import { FastLoop } from "../src/live/fastLoop";
import { EnergyVad } from "../src/live/vad";
import { cloudProvider, ProviderChain, parseSuggestionJson } from "../src/live/localLlm";
import { silenceInt16, toneInt16 } from "../src/live/testing/synth";

const w = (t: number, score: number, voicedFrames = 50) => ({ t, score, voicedFrames });

describe("RoomHeatBuzz", () => {
  it("ships the DEV-tuned defaults (landscape Z-002)", () => {
    expect(ROOM_HEAT_BUZZ_DEFAULTS).toEqual({ meanWindows: 5, threshold: 0.9, cooldownS: 60, minVoicedFrames: 10 });
  });

  it("buzzes when the rolling mean of the last 5 windows reaches the threshold", () => {
    const b = new RoomHeatBuzz({ cooldownS: 0 });
    const fired = [0.5, 0.95, 0.95, 0.95, 0.95, 0.95].map((s, i) => b.observe(w(i + 2, s)));
    // the mean first reaches >= 0.9 once the 0.5 window rolls out of the 5
    expect(fired).toEqual([false, false, false, false, false, true]);
  });

  it("holds a cooldown after each buzz", () => {
    const b = new RoomHeatBuzz({ cooldownS: 60 });
    const times: number[] = [];
    for (let t = 2; t < 200; t++) if (b.observe(w(t, 0.99))) times.push(t);
    expect(times).toEqual([2, 62, 122, 182]);
  });

  it("treats a window without enough voicing as cold (never a fabricated score)", () => {
    const b = new RoomHeatBuzz();
    for (let t = 2; t < 20; t++) expect(b.observe(w(t, 0.99, 3))).toBe(false);
    expect(b.observe(w(20, Number.NaN))).toBe(false);
  });

  it("never buzzes on a calm stream", () => {
    const b = new RoomHeatBuzz();
    for (let t = 2; t < 600; t++) expect(b.observe(w(t, 0.6 + 0.25 * Math.sin(t)))).toBe(false);
  });
});

describe("FastLoop roomHeatBuzz wiring", () => {
  function run(roomHeatBuzz: boolean | undefined) {
    const haptic: { level: number; code?: string | null }[] = [];
    const loop = new FastLoop({
      vad: new EnergyVad(-45, 0.032),
      embedder: null,
      labeler: null,
      overlapProbe: false,
      instantHeat: true,
      ...(roomHeatBuzz === undefined ? {} : { roomHeatBuzz }),
      recognizer: null,
      llm: new ProviderChain([
        { name: "os", isAvailable: async () => true, suggest: async () => parseSuggestionJson('{"suggestion":"ok"}') },
        cloudProvider(),
      ]),
      speak: () => {},
      send: () => {},
      haptics: { nudge: async (level, code) => { haptic.push({ level, code }); } },
      sttGraceMs: 20,
      pollMs: 5,
    });
    return { loop, haptic };
  }

  async function feed(loop: FastLoop, seconds: number) {
    await loop.start({ sessionId: "s", mode: "earpiece", empathy: 60 });
    const pcm = toneInt16(seconds, -20);
    for (let off = 0; off < pcm.length; off += 1600) loop.pushSamples(pcm.subarray(off, Math.min(off + 1600, pcm.length)));
    const s = silenceInt16(0.5);
    for (let off = 0; off < s.length; off += 1600) loop.pushSamples(s.subarray(off, Math.min(off + 1600, s.length)));
    await loop.settle();
  }

  it("is OFF by default: hot windows alone never buzz", async () => {
    const { loop, haptic } = run(undefined);
    await feed(loop, 8);
    expect(haptic).toEqual([]);
  });

  it("when enabled, buzzes once (level 1, H) on a hot room with no identity at all", async () => {
    const { loop, haptic } = run(true);
    await feed(loop, 8);
    expect(haptic).toEqual([{ level: 1, code: "H" }]);
  });
});
