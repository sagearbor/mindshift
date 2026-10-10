/**
 * ROOM-HEAT BUZZ — an identity-free alert on the instant tier (DARK: off
 * unless FastLoopDeps.roomHeatBuzz).
 *
 * Why: the shipped instant buzz only fires on a turn the phone is confident
 * is the WEARER's (fastLoop coachedAsSelf + loudness over the wearer's own
 * baseline). Held-out wearer recall is ~0.3, so across the whole corpus it
 * never fired — and an owner switching the app on mid-argument has no
 * confirmed voice at all. This buzz needs no identity and no baseline: it
 * reads the per-second instant-tier heat windows (instantTier.ts, angry-vs-
 * happy from 2 s of raw PCM) that fastLoop.tickHeat already computes for
 * whoever is talking, and buzzes once when the room as a whole runs hot.
 *
 * Rule (agent Z, 2026-10-10, tuned on the DEV split only; landscape Z-002):
 * the mean of the last `meanWindows` heat windows >= `threshold`, where a
 * window with fewer than `minVoicedFrames` voiced pitch frames counts as 0;
 * then `cooldownS` of silence from this buzz. DEV: 3/11 rater/transcriber
 * heated moments caught (chance 1.3), 0 buzzes on calm audio, 13.9/h on
 * heated audio. A loudness-only version disagreed with human raters
 * (owner, 2026-09-20), which is why this rides the instant-tier score.
 *
 * What it is NOT: about the wearer. It says "the conversation is heating",
 * so it is delivered as a single level-1 buzz (the gentlest rung).
 */

export interface RoomHeatWindow {
  /** End of the 2 s window on the audio clock (s). */
  t: number;
  /** Instant-tier score 0..1. */
  score: number;
  /** Voiced pitch frames in the window (of ~100). */
  voicedFrames: number;
}

export interface RoomHeatBuzzOptions {
  meanWindows: number;
  threshold: number;
  cooldownS: number;
  minVoicedFrames: number;
}

export const ROOM_HEAT_BUZZ_DEFAULTS: RoomHeatBuzzOptions = {
  meanWindows: 5,
  threshold: 0.9,
  cooldownS: 60,
  minVoicedFrames: 10,
};

export class RoomHeatBuzz {
  readonly opts: RoomHeatBuzzOptions;
  private readonly recent: number[] = [];
  private lastBuzzT = -Infinity;

  constructor(opts: Partial<RoomHeatBuzzOptions> = {}) {
    this.opts = { ...ROOM_HEAT_BUZZ_DEFAULTS, ...opts };
  }

  /** Feed one heat window; true when this window should buzz. */
  observe(w: RoomHeatWindow): boolean {
    const v = w.voicedFrames >= this.opts.minVoicedFrames && Number.isFinite(w.score) ? w.score : 0;
    this.recent.push(v);
    if (this.recent.length > this.opts.meanWindows) this.recent.shift();
    const mean = this.recent.reduce((a, b) => a + b, 0) / this.recent.length;
    if (mean < this.opts.threshold || w.t - this.lastBuzzT < this.opts.cooldownS) return false;
    this.lastBuzzT = w.t;
    return true;
  }
}
