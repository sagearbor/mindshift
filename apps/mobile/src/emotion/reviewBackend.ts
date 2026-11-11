/**
 * The Review screen's window onto the on-device store: all clips with their
 * labels, save one rater's label, clear it (undo). A thin seam so the screen
 * is testable without native modules and does not depend on EmotionDevHub.
 */
import type { EmotionDevStore, ReviewItemFull, ReviewLabel } from "./devStore";

export interface ReviewBackend {
  items(): ReviewItemFull[];
  save(id: string, label: ReviewLabel): void;
  clear(id: string, rater: string): void;
  /** Full log as JSON Lines text (explicit export); null when unavailable. */
  exportText(): string | null;
}

export function storeBackend(store: EmotionDevStore): ReviewBackend {
  return {
    items: () => store.reviewItems(),
    save: (id, label) => store.setReview(id, label),
    clear: (id, rater) => store.clearReview(id, rater),
    exportText: () => store.exportLogText(),
  };
}

/** The real on-device store, opened lazily (works with the tester off). */
export function nativeBackend(): ReviewBackend | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const st = (require("./nativeRuntime") as typeof import("./nativeRuntime")).openNativeStore();
    return st ? storeBackend(st) : null;
  } catch {
    return null;
  }
}

// Labeller choices kept for the app session (a module variable, not disk).
export const reviewPrefs: {
  kind: "me" | "mom" | "other";
  name: string;
  mode: "emotion" | "emotion+nudge";
  hideGuess: boolean | null; // null = default for the rater
} = { kind: "me", name: "", mode: "emotion+nudge", hideGuess: null };
