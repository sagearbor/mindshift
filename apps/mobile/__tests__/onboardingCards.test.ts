import {
  ONBOARDING_CARDS,
  clampCardIndex,
  nextCardIndex,
  prevCardIndex,
  isLastCard,
} from "../src/screens/onboardingCards";

describe("ONBOARDING_CARDS", () => {
  it("has exactly the four scoped cards, in order", () => {
    expect(ONBOARDING_CARDS.map((c) => c.id)).toEqual([
      "live-coach",
      "analyze",
      "watch",
      "growth",
    ]);
  });

  it("every card has a non-empty title, body, and glyph", () => {
    for (const card of ONBOARDING_CARDS) {
      expect(card.title.length).toBeGreaterThan(0);
      expect(card.body.length).toBeGreaterThan(0);
      expect(card.glyph.length).toBeGreaterThan(0);
    }
  });

  it("body copy stays to 2-3 honest sentences (no walls of text)", () => {
    for (const card of ONBOARDING_CARDS) {
      const sentenceCount = (card.body.match(/[.!?]+(\s|$)/g) || []).length;
      expect(sentenceCount).toBeGreaterThanOrEqual(2);
      expect(sentenceCount).toBeLessThanOrEqual(3);
    }
  });
});

describe("clampCardIndex", () => {
  it("clamps below zero up to 0", () => {
    expect(clampCardIndex(-5, 4)).toBe(0);
  });

  it("clamps above the last index down to total - 1", () => {
    expect(clampCardIndex(99, 4)).toBe(3);
  });

  it("passes through an in-range index unchanged", () => {
    expect(clampCardIndex(2, 4)).toBe(2);
  });

  it("truncates a fractional index", () => {
    expect(clampCardIndex(1.9, 4)).toBe(1);
  });

  it("defaults total to the real card count", () => {
    expect(clampCardIndex(999)).toBe(ONBOARDING_CARDS.length - 1);
  });
});

describe("nextCardIndex / prevCardIndex", () => {
  it("advances by one", () => {
    expect(nextCardIndex(0, 4)).toBe(1);
  });

  it("never advances past the last card", () => {
    expect(nextCardIndex(3, 4)).toBe(3);
  });

  it("retreats by one", () => {
    expect(prevCardIndex(2, 4)).toBe(1);
  });

  it("never retreats before the first card", () => {
    expect(prevCardIndex(0, 4)).toBe(0);
  });
});

describe("isLastCard", () => {
  it("is false on every card but the last", () => {
    expect(isLastCard(0, 4)).toBe(false);
    expect(isLastCard(1, 4)).toBe(false);
    expect(isLastCard(2, 4)).toBe(false);
  });

  it("is true on the last card", () => {
    expect(isLastCard(3, 4)).toBe(true);
  });

  it("defaults total to the real card count", () => {
    expect(isLastCard(ONBOARDING_CARDS.length - 1)).toBe(true);
  });
});
