/**
 * src/live/floorTick.ts — the mid-turn wearer floor tracker (exp_FT's best
 * no-LLM rule, run every ~1 s while a turn is still open). Pure timelines:
 * one observation per tick, each saying whether the trailing window had
 * speech and who the live wearer estimate says it was.
 */
import {
  FLOOR_CUE_TEXT,
  FLOOR_TICK_DEFAULTS,
  FloorTracker,
  type FloorFire,
  type WearerEstimate,
} from "../src/live/floorTick";

/** One observation per second over [from, to): `who(t)` = the window's
 *  wearer estimate, or "silent" for a window with no speech. */
function run(
  tracker: FloorTracker,
  from: number,
  to: number,
  who: (t: number) => WearerEstimate | "silent",
): FloorFire[] {
  const fires: FloorFire[] = [];
  for (let t = from; t < to; t += 1) {
    const w = who(t);
    const f = tracker.observe({
      t,
      windowStart: t - 2,
      speaking: w !== "silent",
      wearer: w === "silent" ? null : w,
    });
    if (f) fires.push(f);
  }
  return fires;
}

describe("FloorTracker", () => {
  it("ships exp_FT's best live cell: 45 s, 30 s gap, the fixed line", () => {
    expect(FLOOR_TICK_DEFAULTS.thresholdSeconds).toBe(45);
    expect(FLOOR_TICK_DEFAULTS.minGapSeconds).toBe(30);
    expect(FLOOR_CUE_TEXT).toBe("You've had the floor a while. Check in with them.");
  });

  it("a 60 s wearer monologue fires at ~45 s, while the turn is still running", () => {
    const tr = new FloorTracker();
    const fires = run(tr, 2, 61, () => true); // first window [0,2] → floor starts at 0
    expect(fires).toHaveLength(1);
    expect(fires[0].t).toBe(45);
    expect(fires[0].floorStart).toBe(0);
    expect(fires[0].floorSeconds).toBeGreaterThanOrEqual(45);
    expect(fires[0].text).toBe(FLOOR_CUE_TEXT);
  });

  it("a 30 s wearer turn never fires", () => {
    const tr = new FloorTracker();
    const fires = run(tr, 2, 120, (t) => (t <= 30 ? true : t <= 40 ? "silent" : t <= 80 ? false : "silent"));
    expect(fires).toEqual([]);
  });

  it("another voice holding the floor for >= resetOtherSeconds resets the clock", () => {
    const tr = new FloorTracker();
    // wearer 0-30, other 31-34 (4 s), wearer again from 35: the new run
    // starts at the first wearer window after the interruption (33), so it
    // fires 45 s after that, not at 45 s of session time.
    const fires = run(tr, 2, 100, (t) => (t >= 31 && t <= 34 ? false : true));
    expect(fires).toHaveLength(1);
    expect(fires[0].t).toBe(33 + 45);
  });

  it("a backchannel shorter than resetOtherSeconds does not reset", () => {
    const tr = new FloorTracker();
    // one 1 s "mhm" window from the other person at 20 s
    const fires = run(tr, 2, 50, (t) => (t === 20 ? false : true));
    expect(fires.map((f) => f.t)).toEqual([45]);
  });

  it("a wearer pause longer than pauseSeconds ends the run; a short breath does not", () => {
    const short = run(new FloorTracker(), 2, 50, (t) => (t >= 20 && t <= 21 ? "silent" : true));
    expect(short.map((f) => f.t)).toEqual([45]);
    const long = run(new FloorTracker(), 2, 70, (t) => (t >= 20 && t <= 25 ? "silent" : true));
    // run restarts at the first window after the 6 s silence (26 → start 24)
    expect(long.map((f) => f.t)).toEqual([24 + 45]);
  });

  it("fires ONCE per floor run by default (quiet coach): a 120 s monologue gets one cue", () => {
    expect(FLOOR_TICK_DEFAULTS.oncePerRun).toBe(true);
    expect(run(new FloorTracker(), 2, 120, () => true).map((f) => f.t)).toEqual([45]);
  });

  it("a new run (after another voice took the floor) may cue again", () => {
    // wearer 0-50 fires at 45; other 51-54 ends the run; wearer from 55 reaches 45 s at ~100
    const fires = run(new FloorTracker(), 2, 110, (t) => (t >= 51 && t <= 54 ? false : true));
    expect(fires).toHaveLength(2);
    expect(fires[0].t).toBe(45);
    expect(fires[1].floorStart).toBeGreaterThanOrEqual(53);
  });

  it("oncePerRun: false restores the min-gap repeats on one long monologue", () => {
    const fires = run(new FloorTracker({ oncePerRun: false }), 2, 120, () => true);
    expect(fires.map((f) => f.t)).toEqual([45, 75, 105]);
    const slow = run(new FloorTracker({ oncePerRun: false, minGapSeconds: 60 }), 2, 120, () => true);
    expect(slow.map((f) => f.t)).toEqual([45, 105]);
  });

  it("the gap holds across runs too (a reset does not re-arm it)", () => {
    // wearer 0-50 fires at 45; other 51-54 resets; a new wearer run from 55
    // would reach 45 s at ~98, which is past the gap — but with a 120 s gap
    // it must stay silent.
    const fires = run(new FloorTracker({ minGapSeconds: 120 }), 2, 110, (t) => (t >= 51 && t <= 54 ? false : true));
    expect(fires.map((f) => f.t)).toEqual([45]);
  });

  it("an unknown wearer never fires", () => {
    expect(run(new FloorTracker(), 2, 200, () => null)).toEqual([]);
  });

  it("unknown windows inside a confirmed run do not extend it past pauseSeconds", () => {
    // confirmed at the start, then nothing but unknown windows: the run
    // breaks after pauseSeconds without a confirmed wearer window.
    expect(run(new FloorTracker(), 2, 200, (t) => (t < 10 ? true : null))).toEqual([]);
  });

  it("a configurable threshold", () => {
    const fires = run(new FloorTracker({ thresholdSeconds: 20 }), 2, 30, () => true);
    expect(fires.map((f) => f.t)).toEqual([20]);
  });

  it("reports the live floor clock between fires and resets cleanly", () => {
    const tr = new FloorTracker();
    run(tr, 2, 11, () => true);
    expect(tr.floorSeconds(10)).toBe(10);
    tr.reset();
    expect(tr.floorSeconds(10)).toBe(0);
  });
});
