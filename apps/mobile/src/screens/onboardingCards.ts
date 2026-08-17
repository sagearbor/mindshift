/**
 * Pure content + index-navigation logic for the first-launch onboarding
 * walkthrough (Task P3-7). Kept dependency-free (no React, no storage) so it
 * unit-tests as plain data/functions; OnboardingScreen.tsx is the only
 * consumer that wires it to UI state.
 *
 * Card copy is deliberately honest and short (2-3 sentences) — no invented
 * stats, no promises the app doesn't keep yet.
 */

export interface OnboardingCard {
  /** Stable id for testIDs and keys — never re-ordered/renumbered. */
  id: string;
  title: string;
  body: string;
  /** A single emoji-scale glyph standing in for artwork (see the task's
   *  restraint directive: no heavy image assets in the bundle). */
  glyph: string;
}

export const ONBOARDING_CARDS: readonly OnboardingCard[] = [
  {
    id: "live-coach",
    title: "Live Coach",
    body:
      "Start Live Coach before a hard conversation and MindShift listens in " +
      "real time, whispering short cues if your tone starts to slip. " +
      "It coaches you quietly — it never speaks to the other person.",
    glyph: "🎧",
  },
  {
    id: "analyze",
    title: "Analyze a conversation",
    body:
      "Record or upload a conversation afterward and MindShift works out " +
      "who said what, then surfaces tone patterns and concrete suggestions. " +
      "Nothing is shared without you choosing to.",
    glyph: "🔍",
  },
  {
    id: "watch",
    title: "Your watch",
    body:
      "Pair a watch in Settings and it nudges your wrist the moment your " +
      "voice rises — a private, physical cue no one else notices. " +
      "Entirely optional; the phone app works fully without one.",
    glyph: "⌚",
  },
  {
    id: "growth",
    title: "Growth",
    body:
      "Every analyzed conversation adds a point to your trend line. " +
      "Over weeks and months you can see whether things are actually " +
      "getting better — not just guess.",
    glyph: "📈",
  },
] as const;

/** Clamp any integer to a valid index into `ONBOARDING_CARDS` (or an
 *  arbitrary `total`, for testing). Never throws on out-of-range input —
 *  swipe/tap handlers can pass raw deltas without pre-checking bounds. */
export function clampCardIndex(index: number, total: number = ONBOARDING_CARDS.length): number {
  if (total <= 0) return 0;
  if (index < 0) return 0;
  if (index > total - 1) return total - 1;
  return Math.trunc(index);
}

export function nextCardIndex(current: number, total: number = ONBOARDING_CARDS.length): number {
  return clampCardIndex(current + 1, total);
}

export function prevCardIndex(current: number, total: number = ONBOARDING_CARDS.length): number {
  return clampCardIndex(current - 1, total);
}

/** True on the last card — the point at which "Next" becomes "Get started". */
export function isLastCard(current: number, total: number = ONBOARDING_CARDS.length): boolean {
  return clampCardIndex(current, total) === total - 1;
}
