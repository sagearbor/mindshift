/**
 * src/live/pleasantness.ts — replays every golden case in
 * server/tests/fixtures/policy_vectors/pleasantness.json (the spec the
 * Python twin server/pleasantness.py replays too), plus the tracker's
 * mid-call rename and the batch scorer.
 */
import {
  BALANCE_WINDOW,
  CURRENT_WINDOW,
  DIMENSIONS,
  LEAD_MIN_MARGIN,
  PleasantnessTracker,
  SERIES_LENGTH,
  WEIGHTS,
  engagementFromWindow,
  leadOf,
  roundHalfUp,
  scoreSession,
  scoreTurn,
  type Dimension,
} from "../src/live/pleasantness";
import { loadFixture } from "../src/live/testing/synth";

interface Case {
  name: string;
  turns: { speaker: string; text_tone: Record<string, unknown> | null; prosody: Record<string, unknown> | null }[];
  expected: {
    per_turn: { dims: Record<Dimension, number | null>; score: number | null }[];
    people: { speaker: string; current: number | null; series: number[] }[];
    lead: { speaker: string; margin: number } | null;
  };
}

const doc = loadFixture<{
  _schema: { version: number };
  constants: Record<string, unknown>;
  cases: Case[];
}>("pleasantness.json");

describe("pleasantness.json golden vectors", () => {
  it("is the schema this port implements", () => {
    expect(doc._schema.version).toBe(1);
    expect(doc.constants.weights).toEqual(WEIGHTS);
    expect(doc.constants.balance_window).toBe(BALANCE_WINDOW);
    expect(doc.constants.current_window).toBe(CURRENT_WINDOW);
    expect(doc.constants.series_length).toBe(SERIES_LENGTH);
    expect(doc.constants.lead_min_margin).toBe(LEAD_MIN_MARGIN);
    const names = doc.cases.map((c) => c.name);
    for (const n of [
      "warm_then_defensive_two_people",
      "no_tone_no_prosody_is_unscored",
      "single_speaker_engagement_unmeasured",
      "contempt_label_caps_respect",
      "even_when_margin_below_three",
      "loud_penalty_capped_and_series_window_ten",
      "fast_speech_penalty_with_neutral_prior",
      "clamps_and_missing_keys",
    ]) {
      expect(names).toContain(n);
    }
    expect(new Set(names).size).toBe(names.length);
  });

  it.each(doc.cases.map((c) => [c.name, c] as const))("%s", (_name, c) => {
    const tracker = new PleasantnessTracker();
    c.turns.forEach((t, i) => {
      const got = tracker.observe(t.speaker, t.text_tone, t.prosody);
      const want = c.expected.per_turn[i];
      expect({ i, dims: got.dims, score: got.score }).toEqual({ i, dims: want.dims, score: want.score });
    });
    const board = tracker.board();
    expect(board.people.map((p) => ({ speaker: p.speaker, current: p.current, series: p.series }))).toEqual(
      c.expected.people,
    );
    expect(board.lead).toEqual(c.expected.lead);
    // The batch scorer is the same arithmetic in one call.
    const batch = scoreSession(c.turns);
    expect(batch.perTurn.map((t) => t.score)).toEqual(c.expected.per_turn.map((t) => t.score));
    expect(batch.board.lead).toEqual(c.expected.lead);
  });
});

describe("pleasantness helpers", () => {
  it("rounds half-up", () => {
    expect(roundHalfUp(0.5)).toBe(1);
    expect(roundHalfUp(2.5)).toBe(3);
    expect(roundHalfUp(2.49)).toBe(2);
  });

  it("weights sum to one over the five PRD dimensions", () => {
    const total = DIMENSIONS.reduce((a, d) => a + WEIGHTS[d], 0);
    expect(Math.abs(total - 1)).toBeLessThan(1e-9);
  });

  it("engagement needs two voices in the window", () => {
    expect(engagementFromWindow(["A"], "A")).toBeNull();
    expect(engagementFromWindow(["A", "A", "A"], "A")).toBeNull();
    expect(engagementFromWindow(["A", "B"], "A")).toBe(100);
    expect(engagementFromWindow(["A", "B", "A", "A", "A", "A"], "A")).toBe(33);
  });

  it("never fabricates a score from nothing", () => {
    expect(scoreTurn(null, null, { baselineRms: null, engagement: 100 }).score).toBeNull();
    expect(scoreTurn({}, {}, { baselineRms: null, engagement: null }).score).toBeNull();
    expect(scoreTurn({ warmth: 200 }, null, { baselineRms: null, engagement: null }).dims.warmth).toBe(100);
  });

  it("declares a lead only when clearly ahead", () => {
    const person = (speaker: string, current: number | null) => ({ speaker, current, series: [], scoredTurns: 0 });
    expect(leadOf([person("A", 70), person("B", 68)])).toBeNull();
    expect(leadOf([person("A", 70), person("B", 67)])).toEqual({ speaker: "A", margin: 3 });
    expect(leadOf([person("A", 70), person("B", null)])).toBeNull();
    expect(leadOf([person("A", 60), person("B", 70), person("C", 65)])).toEqual({ speaker: "B", margin: 5 });
  });

  it("folds a renamed speaker's history into the person (mid-call 'that's Mom')", () => {
    const tracker = new PleasantnessTracker();
    tracker.observe("Speaker A", { warmth: 80 }, null);
    tracker.observe("Speaker B", { warmth: 40 }, { rms_dbfs: -20 });
    tracker.observe("Speaker A", { warmth: 80 }, null);
    tracker.rename("Speaker B", "Mom");
    const board = tracker.board();
    expect(board.people.map((p) => p.speaker)).toEqual(["Speaker A", "Mom"]);
    expect(board.people[1].series).toEqual([55]); // warmth 40 + balance 100 → 55
    expect(board.lead).toEqual({ speaker: "Speaker A", margin: board.people[0].current! - 55 });
    // Later turns under the new label keep accumulating on the same person,
    // with the loudness baseline carried over (-20 → a +8 dB turn is penalized).
    const next = tracker.observe("Mom", { warmth: 40, frustration: 0 }, { rms_dbfs: -12 });
    expect(next.dims.calmness).toBe(80);
    expect(tracker.board().people[1].series.length).toBe(2);
    // Renaming onto an existing label merges rather than duplicating.
    tracker.rename("Mom", "Speaker A");
    expect(tracker.board().people.map((p) => p.speaker)).toEqual(["Speaker A"]);
    expect(tracker.board().people[0].scoredTurns).toBe(4);
  });
});
