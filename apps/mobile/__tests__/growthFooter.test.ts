/**
 * The honest growth footer (GrowthScreen.growthFooter).
 *
 * "N of M recordings identified your voice" was true and useless: a user
 * cannot tell "the app failed to find me" — which catch-up fixes — from "I am
 * not in that recording", which nothing fixes. These cases pin that each
 * bucket is named only when it exists, and that catch-up is described as what
 * it is (a re-match, not a re-analysis).
 */
import { growthFooter } from "../src/screens/GrowthScreen";
import type { GrowthResult } from "../src/api/client";

function result(
  identified: number,
  total: number,
  gaps: Partial<GrowthResult["gaps"]> = {},
): GrowthResult {
  return {
    points: [],
    total_recordings: total,
    identified_recordings: identified,
    gaps: { not_analyzed: 0, not_your_conversation: 0, could_not_find_you: 0, ...gaps },
    people: [],
  };
}

describe("growthFooter", () => {
  it("says nothing more than N of M when nothing is missing", () => {
    expect(growthFooter(result(4, 4))).toBe("4 of 4 recordings identified your voice");
    expect(growthFooter(result(1, 1))).toBe("1 of 1 recording identified your voice");
  });

  it("falls back to the plain line on a server that reports no buckets", () => {
    // An older server sends no `gaps`, which the client zeroes. All-zero is
    // "we can't say why", NOT "nothing is missing" — so the footer must not
    // invent an explanation for the 3 that are absent.
    expect(growthFooter(result(2, 5))).toBe("2 of 5 recordings identified your voice");
  });

  it("separates the fixable gap from the one that isn't, and says what catch-up costs", () => {
    const line = growthFooter(result(2, 7, { could_not_find_you: 3, not_your_conversation: 2 }));
    expect(line).toContain("2 of 7 recordings identified your voice");
    expect(line).toContain("3 we couldn’t match to you");
    expect(line).toContain("2 you’re not in (you named the speakers yourself)");
    expect(line).toContain("doesn’t re-analyse anything");
  });

  it("does not offer catch-up when nothing catch-up could fix is missing", () => {
    const line = growthFooter(result(1, 4, { not_your_conversation: 2, not_analyzed: 1 }));
    expect(line).toContain("2 you’re not in");
    expect(line).toContain("1 not analysed yet");
    expect(line).not.toContain("Catch-up");
  });

  it("names one bucket without list punctuation, three with it", () => {
    expect(growthFooter(result(1, 2, { not_analyzed: 1 }))).toContain("Of the rest: 1 not analysed yet.");
    const three = growthFooter(result(1, 7, { could_not_find_you: 2, not_your_conversation: 2, not_analyzed: 2 }));
    expect(three).toContain("2 we couldn’t match to you, 2 you’re not in (you named the speakers yourself) and 2 not analysed yet");
  });
});
