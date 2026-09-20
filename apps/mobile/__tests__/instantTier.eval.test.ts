/**
 * THE REAL-DATA GATE for the acoustic instant tier.
 *
 * Everything else about this feature can be true on synthetic tones and still
 * useless in a kitchen. This test pins the only number that decides whether
 * the tier earns its place: given real corpus audio, run through the SHIPPED
 * TypeScript extractor, how well does it tell an angry voice from a
 * happy-loud one on a corpus the model has never seen?
 *
 * The numbers come from fixtures/instantTier.eval.json, written by
 * `npx tsx scripts/instant_tier_eval.ts`, which decodes every RAVDESS
 * angry+happy clip (384 of them) and 500 CREMA-D angry+happy clips from the
 * corpora in the main checkout's tmp/ and scores each one with
 * src/live/instantTier.ts. The corpus audio is not in git and never will be —
 * what is committed is the measurement, dated and regenerable.
 *
 * The bar: cross-corpus angry-vs-happy AUC >= 0.78. For scale, on the same
 * split the signal this replaces — loudness over the speaker's own baseline —
 * scores 0.72, and openSMILE's full 88-descriptor eGeMAPS set scores 0.87.
 */
import evalResult from "./fixtures/instantTier.eval.json";
import model from "../src/live/instantTier.model.json";

const CROSS_CORPUS_FLOOR = 0.78;
/** What the shipped instant signal (loudness over own baseline) manages on
 *  the identical split. Anything at or below this is not worth the CPU. */
const LOUDNESS_ONLY = 0.72;

describe("instant tier — real audio, cross corpus", () => {
  it("was measured on the whole RAVDESS angry+happy set", () => {
    expect(evalResult.crossCorpus.corpus).toBe("ravdess");
    expect(evalResult.crossCorpus.n).toBeGreaterThanOrEqual(380);
    expect(evalResult.crossCorpus.nAngry).toBeGreaterThanOrEqual(150);
    // every clip got a score: a tier that silently declines half its windows
    // would post a flattering AUC on the easy half
    expect(evalResult.crossCorpus.unscored).toBe(0);
  });

  it("tells angry from happy on a corpus it was never fitted on", () => {
    console.log(
      `instant tier cross-corpus angry-vs-happy AUC ${evalResult.crossCorpus.aucAngryVsHappy} ` +
        `over ${evalResult.crossCorpus.n} RAVDESS clips (floor ${CROSS_CORPUS_FLOOR})`,
    );
    expect(evalResult.crossCorpus.aucAngryVsHappy).toBeGreaterThanOrEqual(CROSS_CORPUS_FLOOR);
  });

  it("beats the loudness ladder it is meant to replace", () => {
    expect(evalResult.crossCorpus.aucAngryVsHappy).toBeGreaterThan(LOUDNESS_ONLY);
  });

  it("also works on the corpus it WAS fitted on (and says so)", () => {
    expect(evalResult.withinCorpus.corpus).toBe("cremad");
    expect(evalResult.withinCorpus.n).toBeGreaterThanOrEqual(400);
    expect(evalResult.withinCorpus.aucAngryVsHappy).toBeGreaterThan(
      evalResult.crossCorpus.aucAngryVsHappy,
    );
    expect(evalResult.withinCorpus.note).toMatch(/not a generalisation claim/);
  });

  it("stays inside the 40 ms per-window budget", () => {
    expect(evalResult.latency.windowsBenched).toBeGreaterThanOrEqual(100);
    expect(evalResult.latency.msPerWindow).toBeLessThan(40);
  });

  it("reports the same headline number the shipped model file claims", () => {
    // the model file's metric is the Python refit; the eval is the TypeScript
    // extractor scoring real audio. They are two measurements of one thing and
    // must not drift apart, or one of them is stale.
    expect(
      Math.abs(model.metrics.xcorp_auc_angry_vs_happy - evalResult.crossCorpus.aucAngryVsHappy),
    ).toBeLessThan(0.02);
  });
});
