/**
 * Pure logic of the Emotion dev tester: probabilities, speech gate, rate cap,
 * nudge gate, disagreement trigger. No I/O, no clock of its own (every
 * function takes `nowMs`), so all of it is unit-tested directly
 * (__tests__/emotionLogic.test.ts).
 */
import {
  EMOTION_DEV_DEFAULTS,
  EMOTION_LABELS,
  LOGIT_OFFSETS,
  type EmotionDevConfig,
  type EmotionLabel,
  type Probs,
} from "./config";

// ---------------------------------------------------------------------------
// Probabilities
// ---------------------------------------------------------------------------

export function topClass(p: Probs): { label: EmotionLabel; index: number; prob: number } {
  let index = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[index]) index = i;
  return { label: EMOTION_LABELS[index], index, prob: p[index] };
}

/** Add per-class logit offsets (log p + offset) and re-normalise. With all-zero
 *  offsets (the shipped setting) the input is returned unchanged. */
export function applyOffsets(p: Probs, offsets: readonly number[] = LOGIT_OFFSETS): number[] {
  if (offsets.every((o) => o === 0)) return Array.from(p);
  const z = p.map((v, i) => Math.log(Math.max(v, 1e-12)) + (offsets[i] ?? 0));
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** A model output is only believable if it is a probability vector. */
export function isProbVector(p: ArrayLike<number>): boolean {
  if (p.length !== EMOTION_LABELS.length) return false;
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    if (!Number.isFinite(p[i]) || p[i] < 0) return false;
    s += p[i];
  }
  return Math.abs(s - 1) < 1e-3;
}

// ---------------------------------------------------------------------------
// Speech-energy gate (cheap: decides whether a window is worth classifying)
// ---------------------------------------------------------------------------

export function windowHasSpeechEnergy(
  pcm: Int16Array,
  cfg: Pick<EmotionDevConfig, "gateFrameMs" | "gateFrameDbfs" | "gateMinActiveFraction"> = EMOTION_DEV_DEFAULTS,
  sampleRate = 16000,
): boolean {
  const frame = Math.round((sampleRate * cfg.gateFrameMs) / 1000);
  if (pcm.length < frame) return false;
  const thresh = Math.pow(10, cfg.gateFrameDbfs / 20) * 32768;
  let active = 0;
  let frames = 0;
  for (let off = 0; off + frame <= pcm.length; off += frame) {
    let s = 0;
    for (let i = 0; i < frame; i++) s += pcm[off + i] * pcm[off + i];
    if (Math.sqrt(s / frame) >= thresh) active++;
    frames++;
  }
  return frames > 0 && active / frames >= cfg.gateMinActiveFraction;
}

// ---------------------------------------------------------------------------
// Rate caps
// ---------------------------------------------------------------------------

/** At most `max` events in any rolling `windowMs`, and `minGapMs` between them. */
export class RollingRateCap {
  private times: number[] = [];
  constructor(
    private max: number,
    private readonly windowMs = 3_600_000,
    private readonly minGapMs = 0,
  ) {}

  setMax(max: number): void {
    this.max = max;
  }

  private prune(nowMs: number): void {
    this.times = this.times.filter((t) => nowMs - t < this.windowMs);
  }

  canFire(nowMs: number): boolean {
    this.prune(nowMs);
    if (this.times.length >= this.max) return false;
    const last = this.times[this.times.length - 1];
    return last === undefined || nowMs - last >= this.minGapMs;
  }

  /** Record an event; false (and nothing recorded) when the cap forbids it. */
  tryFire(nowMs: number): boolean {
    if (!this.canFire(nowMs)) return false;
    this.times.push(nowMs);
    return true;
  }
}

/** At most `max` events per local calendar day. */
export class DailyCap {
  private day = "";
  private count = 0;
  constructor(private readonly max: number) {}

  private dayKey(nowMs: number): string {
    const d = new Date(nowMs);
    return `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`;
  }

  tryFire(nowMs: number): boolean {
    const k = this.dayKey(nowMs);
    if (k !== this.day) {
      this.day = k;
      this.count = 0;
    }
    if (this.count >= this.max) return false;
    this.count++;
    return true;
  }
}

// ---------------------------------------------------------------------------
// Nudge gate (TINY only)
// ---------------------------------------------------------------------------

export interface NudgeDecision {
  nudge: boolean;
  /** Why not (for the log / the tester), "" when nudge is true. */
  reason: string;
}

/**
 * Decides whether TINY's latest window earns a buzz: non-neutral top class in
 * the allowed set, probability >= nudgeMinProb, the same class on
 * `nudgeConfirmWindows` consecutive windows, within the rolling-hour budget
 * and the minimum gap. A paused tester never reaches this (the controller
 * short-circuits first).
 */
export class NudgeGate {
  private streakLabel: EmotionLabel | null = null;
  private streak = 0;
  private readonly cap: RollingRateCap;

  constructor(private cfg: EmotionDevConfig = EMOTION_DEV_DEFAULTS) {
    this.cap = new RollingRateCap(cfg.maxNudgesPerHour, 3_600_000, cfg.minNudgeGapS * 1000);
  }

  setMaxPerHour(n: number): void {
    this.cap.setMax(n);
  }

  /** Forget the running streak (pause, silence, a skipped window). */
  reset(): void {
    this.streakLabel = null;
    this.streak = 0;
  }

  decide(tiny: Probs, nowMs: number): NudgeDecision {
    const top = topClass(tiny);
    if (top.label === "neutral" || !this.cfg.nudgeLabels.includes(top.label) || top.prob < this.cfg.nudgeMinProb) {
      this.reset();
      return { nudge: false, reason: top.label === "neutral" ? "neutral" : "below-threshold" };
    }
    if (this.streakLabel === top.label) this.streak++;
    else {
      this.streakLabel = top.label;
      this.streak = 1;
    }
    if (this.streak < this.cfg.nudgeConfirmWindows) return { nudge: false, reason: "unconfirmed" };
    if (!this.cap.tryFire(nowMs)) return { nudge: false, reason: "rate-capped" };
    this.reset();
    return { nudge: true, reason: "" };
  }
}

// ---------------------------------------------------------------------------
// Disagreement trigger
// ---------------------------------------------------------------------------

/** TINY and SMALL name different top classes and BOTH are confident. */
export function modelsDisagree(
  tiny: Probs,
  small: Probs,
  minProb: number = EMOTION_DEV_DEFAULTS.disagreeMinProb,
): boolean {
  const a = topClass(tiny);
  const b = topClass(small);
  return a.index !== b.index && a.prob >= minProb && b.prob >= minProb;
}
