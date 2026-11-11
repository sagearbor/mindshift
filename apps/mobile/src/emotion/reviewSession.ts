/**
 * Review mode, pure of platform: a queue of saved clips played back to back
 * and labelled with as few taps as possible. The UI owns the audio player and
 * haptics; this owns "which clip, who is labelling, what a tap means" so it is
 * unit-tested (__tests__/emotionReview.test.ts).
 *
 * Taps per clip:
 *  - "Emotion only": ONE tap. "Right" (the shown label was right) or an
 *    emotion chip (= wrong, and this is what it really was).
 *  - "Emotion + nudge": TWO taps. The same first tap, then "Would a nudge help
 *    here?" Yes / No / Not sure, which saves and advances.
 *
 * rating = "good" when the TINY label shown was right (truth === TINY's top
 * class). When TINY and SMALL disagree the UI offers "TINY right" / "SMALL
 * right"; either way the stored truth is the emotion, so agreement with each
 * model is recomputed from it.
 *
 * One label per rater per clip (the owner is "me"; his legacy single flag/truth
 * reads as the "me" label). The queue is the clips this rater has not labelled.
 */
import { EMOTION_LABELS, type EmotionLabel } from "./config";
import { topClass } from "./logic";
import type { NudgeWarranted, ReviewItemFull, ReviewLabel } from "./devStore";

/** Gap between clips, ms (so back-to-back clips do not blur together). */
export const REVIEW_GAP_MS = 1200;
/** A clip plays this many times with no tap before auto-advancing. */
export const REVIEW_PLAYS_PER_CLIP = 2;

export type ReviewMode = "emotion" | "emotion+nudge";
export type RaterKind = "me" | "mom" | "other";
export type EndedAction = "replay" | "advance" | "done";
export type Day = string; // "YYYY-MM-DD" (local) or "all"
export const ALL_DAYS = "all";

export interface ReviewDeps {
  save(id: string, label: ReviewLabel): void;
  clear(id: string, rater: string): void;
  now?: () => number;
}

// -- who ----------------------------------------------------------------------

export function raterKey(kind: RaterKind, name = ""): string {
  if (kind !== "other") return kind;
  const n = name.trim().toLowerCase().replace(/\s+/g, " ");
  return n ? `other:${n}` : "other";
}
/** The model's guess is hidden by default for anyone but the owner (no bias). */
export function defaultHideGuess(rater: string): boolean {
  return rater !== "me";
}

// -- what the models said / context -------------------------------------------

const top = (p: number[] | null | undefined): EmotionLabel | null =>
  p && p.length === EMOTION_LABELS.length ? (topClass(p).label as EmotionLabel) : null;
export const tinyTop = (i: Pick<ReviewItemFull, "tiny">) => top(i.tiny);
export const smallTop = (i: Pick<ReviewItemFull, "small">) => top(i.small);

/** "TINY: anger 62% · SMALL: neutral 80%" — what the models said. */
export function describeItem(item: Pick<ReviewItemFull, "tiny" | "small" | "kind">): string {
  const part = (name: string, p: number[] | null) => {
    if (!p || p.length !== EMOTION_LABELS.length) return `${name}: —`;
    const t = topClass(p);
    return `${name}: ${t.label} ${Math.round(t.prob * 100)}%`;
  };
  const head = item.kind === "mark" ? "Marked moment. " : "";
  return `${head}${part("TINY", item.tiny)} · ${part("SMALL", item.small)}`;
}

const KIND_LABEL: Record<string, string> = {
  saved: "emotion",
  disagree: "disagreement",
  mark: "marked moment",
  "neutral-sample": "neutral sample",
  nudge: "nudge",
};
export function kindLabel(kind: string): string {
  return KIND_LABEL[kind] ?? kind;
}

function clock(t: number): string {
  const d = new Date(t);
  const h = d.getHours();
  const m = String(d.getMinutes()).padStart(2, "0");
  return `${((h + 11) % 12) + 1}:${m} ${h < 12 ? "am" : "pm"}`;
}

/** "7:42 pm · 12 s · emotion · nudge fired: take a breath" (only fields that exist). */
export function contextLine(item: Partial<ReviewItemFull> & { t: number; kind: string }): string {
  const parts = [clock(item.t)];
  if (item.durationS) parts.push(`${item.durationS} s`);
  parts.push(kindLabel(item.kind));
  if (item.nudged) parts.push(item.nudgeReason ? `nudge fired: ${item.nudgeReason}` : "nudge fired");
  return parts.join(" · ");
}

// -- queue ----------------------------------------------------------------------

/** This rater's label on a clip: their own entry, or (for "me") the legacy flag/truth. */
export function labelOf(item: ReviewItemFull, rater: string): ReviewLabel | null {
  const mine = (item.reviews ?? []).find((r) => r.rater === rater);
  if (mine) return mine;
  if (rater === "me" && (item.flag === "good" || item.flag === "bad")) {
    return { rater: "me", rating: item.flag, truth: item.truth || null, t: item.t };
  }
  return null;
}

export function dayKey(t: number): string {
  const d = new Date(t);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

/** Days that have clips, newest first, with how many this rater has not labelled yet. */
export function listDays(items: ReviewItemFull[], rater: string): { day: string; unrated: number; total: number }[] {
  const m = new Map<string, { day: string; unrated: number; total: number }>();
  for (const it of items) {
    const k = dayKey(it.t);
    const e = m.get(k) ?? { day: k, unrated: 0, total: 0 };
    e.total++;
    if (!labelOf(it, rater)) e.unrated++;
    m.set(k, e);
  }
  return [...m.values()].sort((a, b) => (a.day < b.day ? 1 : -1));
}

/** Most recent day with unrated clips; "all" when there is none. */
export function defaultDay(items: ReviewItemFull[], rater: string): Day {
  return listDays(items, rater).find((d) => d.unrated > 0)?.day ?? ALL_DAYS;
}

/** Clips this rater has not labelled (for the day), oldest first. */
export function buildQueue(items: ReviewItemFull[], rater: string, day: Day = ALL_DAYS): ReviewItemFull[] {
  return items
    .filter((it) => !labelOf(it, rater) && (day === ALL_DAYS || dayKey(it.t) === day))
    .sort((a, b) => a.t - b.t);
}

// -- session --------------------------------------------------------------------

export type FirstTap = { right: "tiny" | "small" | "both" } | { truth: EmotionLabel };
export interface Summary {
  rated: number;
  skipped: number;
  /** agree / of, over labelled clips where that model's guess is known. */
  tiny: { agree: number; of: number };
  small: { agree: number; of: number };
}

interface Done {
  index: number;
  id: string | null; // null = skipped (nothing to clear)
  label?: ReviewLabel;
}

export class ReviewSession {
  private i = 0;
  private plays = 0;
  private pending: { truth: EmotionLabel | null; rating: "good" | "bad" } | null = null;
  private history: Done[] = [];

  constructor(
    private readonly items: ReviewItemFull[],
    private readonly deps: ReviewDeps,
    readonly opts: { rater: string; mode: ReviewMode },
  ) {}

  get total(): number {
    return this.items.length;
  }
  /** 1-based "7 of 42". */
  get position(): number {
    return Math.min(this.i + 1, this.items.length);
  }
  get ratio(): number {
    return this.items.length ? Math.min(this.i, this.items.length) / this.items.length : 0;
  }
  current(): ReviewItemFull | null {
    return this.items[this.i] ?? null;
  }
  /** "emotion" = waiting for the first tap; "nudge" = waiting for the nudge answer. */
  get stage(): "emotion" | "nudge" {
    return this.pending ? "nudge" : "emotion";
  }
  get pendingTruth(): EmotionLabel | null {
    return this.pending?.truth ?? null;
  }
  get canUndo(): boolean {
    return this.history.length > 0 || this.pending !== null;
  }

  /** First tap. Returns "saved" (emotion-only), "await-nudge", or false when there is no clip. */
  answer(tap: FirstTap): "saved" | "await-nudge" | false {
    const cur = this.current();
    if (!cur) return false;
    let truth: EmotionLabel | null;
    if ("truth" in tap) truth = tap.truth;
    else truth = tap.right === "small" ? smallTop(cur) ?? tinyTop(cur) : tinyTop(cur);
    const rating = tinyTop(cur) !== null && truth === tinyTop(cur) ? "good" : "bad";
    // A "Right" on a clip whose labels are unreadable is still a thumbs up.
    const finalRating = "right" in tap && tinyTop(cur) === null ? "good" : rating;
    this.pending = { truth, rating: finalRating };
    if (this.opts.mode === "emotion+nudge") return "await-nudge";
    this.commit(undefined);
    return "saved";
  }

  /** Second tap (nudge mode). Saves and advances. */
  answerNudge(v: NudgeWarranted): boolean {
    if (!this.pending || !this.current()) return false;
    this.commit(v);
    return true;
  }

  /** From the nudge question back to the emotion buttons. */
  backToEmotion(): void {
    this.pending = null;
  }

  private commit(nudge: NudgeWarranted | undefined): void {
    const cur = this.current()!;
    const p = this.pending!;
    const label: ReviewLabel = {
      rater: this.opts.rater,
      rating: p.rating,
      ...(p.truth ? { truth: p.truth } : {}),
      ...(nudge ? { nudgeWarranted: nudge } : {}),
      t: this.deps.now ? this.deps.now() : Date.now(),
    };
    this.deps.save(cur.id, label);
    this.history.push({ index: this.i, id: cur.id, label });
    this.next();
  }

  skip(): void {
    if (!this.current()) return;
    this.history.push({ index: this.i, id: null });
    this.next();
  }

  /** Back one step: drops a half-finished answer, else returns to the previous clip and clears its label. */
  undo(): boolean {
    if (this.pending) {
      this.pending = null;
      return true;
    }
    const h = this.history.pop();
    if (!h) return false;
    if (h.id) this.deps.clear(h.id, this.opts.rater);
    this.i = h.index;
    this.plays = 0;
    return true;
  }

  /** Replay the current clip (resets the no-tap counter). */
  replay(): void {
    this.plays = 0;
  }

  /** The current clip finished playing. */
  onEnded(): EndedAction {
    if (!this.current()) return "done";
    if (this.pending) return "replay"; // mid-answer: never skip it
    this.plays++;
    if (this.plays < REVIEW_PLAYS_PER_CLIP) return "replay";
    this.skip();
    return this.current() ? "advance" : "done";
  }

  summary(): Summary {
    const s: Summary = { rated: 0, skipped: 0, tiny: { agree: 0, of: 0 }, small: { agree: 0, of: 0 } };
    for (const h of this.history) {
      if (!h.id || !h.label) {
        s.skipped++;
        continue;
      }
      s.rated++;
      const item = this.items[h.index];
      const truth = h.label.truth ?? null;
      const t = tinyTop(item);
      if (truth && t) {
        s.tiny.of++;
        if (truth === t) s.tiny.agree++;
      } else if (!truth) {
        s.tiny.of++;
        if (h.label.rating === "good") s.tiny.agree++;
      }
      const sm = smallTop(item);
      if (truth && sm) {
        s.small.of++;
        if (truth === sm) s.small.agree++;
      }
    }
    return s;
  }

  private next(): void {
    this.i++;
    this.plays = 0;
    this.pending = null;
  }
}
