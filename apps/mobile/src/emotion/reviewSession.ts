/**
 * Review mode, pure of platform: a queue of saved-but-unrated clips played
 * back to back. The UI owns the audio player and haptics; this owns "which
 * clip, what was said about it, and what a tap means" so it is unit-tested
 * (__tests__/emotionReview.test.ts).
 *
 * Tap meaning: THUMBS UP = the models' emotion label was right (flag "good");
 * THUMBS DOWN = wrong (flag "bad"), optionally with "what was it really?".
 * Either rating advances to the next clip. A clip that plays to the end
 * twice with no tap is skipped (stays unrated, comes back next session).
 */
import { EMOTION_LABELS, type EmotionLabel } from "./config";
import { topClass } from "./logic";
import type { ReviewItem } from "./devStore";

/** Gap between clips, ms (so back-to-back clips do not blur together). */
export const REVIEW_GAP_MS = 1200;
/** A clip plays this many times with no tap before auto-advancing. */
export const REVIEW_PLAYS_PER_CLIP = 2;

export interface ReviewDeps {
  rate(id: string, rating: "good" | "bad", truth?: EmotionLabel | null): void;
}

export type EndedAction = "replay" | "advance" | "done";

/** "TINY: anger 62% · SMALL: neutral 80%" — what the models said. */
export function describeItem(item: Pick<ReviewItem, "tiny" | "small" | "kind">): string {
  const part = (name: string, p: number[] | null) => {
    if (!p || p.length !== EMOTION_LABELS.length) return `${name}: —`;
    const t = topClass(p);
    return `${name}: ${t.label} ${Math.round(t.prob * 100)}%`;
  };
  const head = item.kind === "mark" ? "Marked moment. " : "";
  return `${head}${part("TINY", item.tiny)} · ${part("SMALL", item.small)}`;
}

export class ReviewSession {
  private i = 0;
  private plays = 0;
  private truth: EmotionLabel | null = null;
  rated = 0;
  skipped = 0;

  constructor(
    private readonly items: ReviewItem[],
    private readonly deps: ReviewDeps,
  ) {}

  get total(): number {
    return this.items.length;
  }
  get position(): number {
    return Math.min(this.i + 1, this.items.length);
  }
  current(): ReviewItem | null {
    return this.items[this.i] ?? null;
  }
  get pendingTruth(): EmotionLabel | null {
    return this.truth;
  }

  /** "What was it really?" — remembered for the next thumb on this clip; tap again to clear. */
  setTruth(label: EmotionLabel): void {
    this.truth = this.truth === label ? null : label;
  }

  /** Thumb tap. Returns false when there is no current clip. */
  thumb(up: boolean): boolean {
    const cur = this.current();
    if (!cur) return false;
    this.deps.rate(cur.id, up ? "good" : "bad", this.truth);
    this.rated++;
    this.next();
    return true;
  }

  skip(): void {
    if (!this.current()) return;
    this.skipped++;
    this.next();
  }

  /** The current clip finished playing. */
  onEnded(): EndedAction {
    if (!this.current()) return "done";
    this.plays++;
    if (this.plays < REVIEW_PLAYS_PER_CLIP) return "replay";
    this.skipped++;
    this.next();
    return this.current() ? "advance" : "done";
  }

  private next(): void {
    this.i++;
    this.plays = 0;
    this.truth = null;
  }
}
