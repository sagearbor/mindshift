import { planTakes, summarize, tooQuiet } from "../src/emotion/scriptedTest";

const P = (i: number) => Array.from({ length: 6 }, (_, k) => (k === i ? 0.9 : 0.02));

describe("scripted emotion test", () => {
  it("plans each emotion once per round, in the fixed order", () => {
    const plan = planTakes(2);
    expect(plan).toHaveLength(12);
    expect(plan.slice(0, 6).map((t) => t.expect)).toEqual(["anger", "disgust", "fear", "happiness", "neutral", "sadness"]);
    expect(plan[6]).toEqual({ expect: "anger", round: 2 });
  });

  it("scores asked vs guessed per model and names the usual wrong answer", () => {
    const sum = summarize([
      { expect: "anger", tiny: P(0), small: P(4) },
      { expect: "anger", tiny: P(1), small: P(4) },
      { expect: "sadness", tiny: P(5), small: null },
    ]);
    expect(sum.total).toBe(3);
    expect(sum.tinyRight).toBe(2);
    expect(sum.smallRight).toBe(0);
    const anger = sum.rows.find((r) => r.emotion === "anger")!;
    expect(anger).toMatchObject({ takes: 2, tinyRight: 1, smallRight: 0, tinyUsually: "disgust", smallUsually: "neutral" });
  });

  it("flags silence as too quiet", () => {
    expect(tooQuiet(new Int16Array(16000))).toBe(true);
    expect(tooQuiet(new Int16Array(16000).fill(3000))).toBe(false);
  });
});
