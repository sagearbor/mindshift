/**
 * Mid-turn wearer floor tracker — "you've had the floor a while", decided
 * while the wearer is STILL TALKING, with no LLM.
 *
 * Why: the one coaching signal with real evidence behind it is the wearer
 * holding the floor (exp_PCA's self_share60 / PC1; exp_FT's best no-LLM rule:
 * floor >= 45 s → the fixed line below, min gap 30 s, 6/7 judged helpful on
 * DEV). But every phone decision waits for the turn to END, and a >= 45 s
 * floor run ends a median 30.8 s after the 45 s mark (fable_latency REPORT §3)
 * — an end-of-turn trigger lands after the monologue it is about.
 *
 * So the loop (fastLoop.ts, `FastLoopDeps.floorTick`, default OFF) feeds this
 * tracker one observation a second while audio flows: did the trailing
 * window contain speech, and who does the live wearer estimate (an ECAPA pass
 * on that window) say it was. The tracker keeps the floor clock and says when
 * to fire. Pure — no audio, no clock — so the timelines are unit tests.
 *
 * Rules (exp_FT's `floor_state`, translated to a streaming clock):
 * - a run starts at the window start of the first wearer-confirmed window;
 * - wearer silence (or unconfirmed windows) longer than `pauseSeconds` ends it;
 * - another voice confirmed for `resetOtherSeconds` (accumulated, cleared by a
 *   wearer window) ends it — a 1 s "mhm" does not;
 * - fire when the run is >= `thresholdSeconds`, the wearer was confirmed within
 *   `speakingRecencySeconds`, and the last fire was >= `minGapSeconds` ago (the
 *   gap is fire-to-fire, so a reset never re-arms it early);
 * - an unknown wearer (`null`) never starts a run, so it never fires.
 */

export const FLOOR_CUE_TEXT = "You've had the floor a while. Check in with them.";

export interface FloorTickConfig {
  /** Floor seconds before the cue (exp_FT X). */
  thresholdSeconds: number;
  /** Seconds between two cues, fire to fire (exp_FT gap). */
  minGapSeconds: number;
  /** At most one cue per floor run (quiet coach, 2026-10-30): a monologue
   *  gets ONE reminder, and a run the tracker still believes is going on
   *  after the wearer actually stopped cannot re-fire. false = exp_FT's
   *  repeat-every-minGapSeconds behaviour. */
  oncePerRun: boolean;
  /** Seconds of another confirmed voice that take the floor away. */
  resetOtherSeconds: number;
  /** Longest wearer pause (or run of unconfirmed windows) inside one run. */
  pauseSeconds: number;
  /** The wearer must have been confirmed this recently to fire. */
  speakingRecencySeconds: number;
  /** Trailing window the loop embeds per tick, and the tick period. */
  windowSeconds: number;
  tickSeconds: number;
  /** Cosine vs the wearer's print at/above which a window is the wearer,
   *  and below which it is clearly someone else (between = unknown). */
  selfThreshold: number;
  otherCeiling: number;
  /** How long the spoken line may wait for a pause before it is dropped
   *  (the haptic is immediate). */
  speakHoldMs: number;
  /** Speak the line (earpiece/speaker) and/or buzz ("A" hogging, level 1). */
  speak: boolean;
  buzz: boolean;
  text: string;
}

export const FLOOR_TICK_DEFAULTS: FloorTickConfig = {
  thresholdSeconds: 45,
  minGapSeconds: 30,
  oncePerRun: true,
  resetOtherSeconds: 2,
  pauseSeconds: 3,
  speakingRecencySeconds: 2,
  windowSeconds: 2,
  tickSeconds: 1,
  selfThreshold: 0.5,
  otherCeiling: 0.3,
  speakHoldMs: 10_000,
  speak: true,
  buzz: true,
  text: FLOOR_CUE_TEXT,
};

/** true = the wearer, false = someone else, null = can't tell. */
export type WearerEstimate = boolean | null;

export interface FloorObservation {
  /** End of the window, audio-clock seconds. */
  t: number;
  windowStart: number;
  /** The window contained speech. */
  speaking: boolean;
  /** Live wearer estimate for the window (ignored when not speaking). */
  wearer: WearerEstimate;
}

export interface FloorFire {
  t: number;
  floorStart: number;
  floorSeconds: number;
  text: string;
}

export class FloorTracker {
  readonly config: FloorTickConfig;
  private runStart: number | null = null;
  private lastWearerAt = -Infinity;
  private otherAccum = 0;
  private lastFireAt = -Infinity;
  private firedThisRun = false;
  private prevT: number | null = null;

  constructor(config: Partial<FloorTickConfig> = {}) {
    this.config = { ...FLOOR_TICK_DEFAULTS, ...config };
  }

  reset(): void {
    this.runStart = null;
    this.lastWearerAt = -Infinity;
    this.otherAccum = 0;
    this.lastFireAt = -Infinity;
    this.firedThisRun = false;
    this.prevT = null;
  }

  /** The wearer's current floor clock at `t` (0 = no run). */
  floorSeconds(t: number): number {
    return this.runStart === null ? 0 : Math.max(0, t - this.runStart);
  }

  observe(o: FloorObservation): FloorFire | null {
    const c = this.config;
    const dt = this.prevT === null ? c.tickSeconds : Math.min(Math.max(0, o.t - this.prevT), c.windowSeconds);
    this.prevT = o.t;

    if (o.speaking && o.wearer === true) {
      if (this.runStart === null) this.runStart = o.windowStart;
      this.lastWearerAt = o.t;
      this.otherAccum = 0;
    } else if (o.speaking && o.wearer === false) {
      this.otherAccum += dt;
      if (this.otherAccum >= c.resetOtherSeconds) this.endRun();
    }
    if (this.runStart !== null && o.t - this.lastWearerAt > c.pauseSeconds) this.endRun();

    if (this.runStart === null) return null;
    const floor = o.t - this.runStart;
    if (floor < c.thresholdSeconds) return null;
    if (o.t - this.lastWearerAt > c.speakingRecencySeconds) return null;
    if (o.t - this.lastFireAt < c.minGapSeconds) return null;
    if (c.oncePerRun && this.firedThisRun) return null;
    this.lastFireAt = o.t;
    this.firedThisRun = true;
    return { t: o.t, floorStart: this.runStart, floorSeconds: floor, text: c.text };
  }

  private endRun(): void {
    this.runStart = null;
    this.firedThisRun = false;
    this.otherAccum = 0;
  }
}

/** A window's wearer estimate from its cosine scores (SpeakerLabeler
 *  .scoreWindow): the wearer above `selfThreshold`; someone else below
 *  `otherCeiling`, or when another known voice beats the wearer's print;
 *  otherwise unknown. `self === null` (no wearer print) is unknown. */
export function wearerFromScores(
  scores: { self: number | null; otherMax: number | null },
  config: Pick<FloorTickConfig, "selfThreshold" | "otherCeiling">,
): WearerEstimate {
  const { self, otherMax } = scores;
  if (self === null) return null;
  if (self >= config.selfThreshold && (otherMax === null || self >= otherMax)) return true;
  if (self < config.otherCeiling) return false;
  if (otherMax !== null && otherMax > self && otherMax >= config.selfThreshold) return false;
  return null;
}
