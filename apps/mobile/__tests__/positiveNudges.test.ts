/**
 * src/live/positiveNudges.ts — replays
 * server/tests/fixtures/policy_vectors/positive_nudges.json, the same file
 * server/tests/test_positive_nudges_vectors.py replays against the Python
 * mirror. The cases that say a code must NOT fire matter as much as the ones
 * that say it must: a false "nice repair!" after something that was not one is
 * worse than saying nothing.
 */
import {
  CALM_STREAK_MIN_S,
  coalesceTurns,
  DEESCALATION_WINDOW_TURNS,
  detectPositiveNudges,
  hasRepairLanguage,
  LISTEN_MIN_TURN_S,
  normalizeForRepair,
  REPAIR_PHRASES,
  REPAIR_SOFTEN_DROP,
  selfHeatLevel,
  type PositiveTurn,
} from "../src/live/positiveNudges";
import { POSITIVE_CAP_S } from "../src/live/nudgeVocabulary";
import { loadFixture } from "../src/live/testing/synth";

interface TurnSpec {
  index: number;
  start: number;
  end: number;
  speaker: string;
  is_self: boolean;
  text: string;
  loud_level: number | null;
  tone_heat: number | null;
  tone_negativity: number | null;
  cut_in: boolean;
}
interface CaseSpec {
  name: string;
  description: string;
  turns: TurnSpec[];
  /** true = `turns` are FRAGMENTS; coalesce them first. */
  coalesce?: boolean;
  alert_nudge_times: number[];
  session_end_s: number | null;
  cap_s?: number;
  expected: {
    nudges: { code: string; t: number; turn_index: number; delivered: boolean }[];
    calm: { longest_s: number; badge: boolean };
  };
}
interface Doc {
  _schema: { version: number };
  constants: Record<string, number>;
  repair_phrases: string[];
  cases: CaseSpec[];
}

const doc = loadFixture<Doc>("positive_nudges.json");

function toTurn(t: TurnSpec): PositiveTurn {
  return {
    index: t.index,
    start: t.start,
    end: t.end,
    speaker: t.speaker,
    isSelf: t.is_self,
    text: t.text,
    loudLevel: t.loud_level,
    toneHeat: t.tone_heat,
    toneNegativity: t.tone_negativity,
    cutIn: t.cut_in,
  };
}

describe("positive_nudges.json golden vectors", () => {
  it("schema version, constants and the repair lexicon match the module", () => {
    expect(doc._schema.version).toBe(1);
    expect(doc.constants.listen_min_turn_s).toBe(LISTEN_MIN_TURN_S);
    expect(doc.constants.deescalation_window_turns).toBe(DEESCALATION_WINDOW_TURNS);
    expect(doc.constants.repair_soften_drop).toBe(REPAIR_SOFTEN_DROP);
    expect(doc.constants.calm_streak_min_s).toBe(CALM_STREAK_MIN_S);
    expect(doc.constants.positive_cap_s).toBe(POSITIVE_CAP_S);
    expect(doc.repair_phrases).toEqual([...REPAIR_PHRASES]);
  });

  it("covers every code and both directions of each", () => {
    const names = new Set(doc.cases.map((c) => c.name));
    for (const required of [
      "de_escalation_after_a_spike",
      "de_escalation_must_land_within_two_of_your_turns",
      "listened_to_a_long_uninterrupted_turn",
      "cutting_in_forfeits_the_listened_badge",
      "repair_softens_the_other_person",
      "a_repair_lands_when_they_go_from_hurt_to_relieved",
      "aggression_falling_while_they_stay_hurt_is_not_a_repair",
      "repair_language_without_softening_is_not_repair",
      "softening_without_repair_language_is_not_repair",
      "positives_are_capped_at_one_per_two_minutes",
      "coalescing_restores_the_long_turn_the_vad_cut_up",
      "two_different_people_are_never_one_turn",
      "calm_streak_is_the_longest_quiet_run",
    ]) {
      expect(names.has(required)).toBe(true);
    }
    const fired = new Set(doc.cases.flatMap((c) => c.expected.nudges.map((n) => n.code)));
    expect([...fired].sort()).toEqual(["D", "E", "R"]);
  });

  it.each(doc.cases.map((c) => [c.name, c] as const))("replays identically: %s", (_name, c) => {
    const rows = c.turns.map(toTurn);
    const got = detectPositiveNudges(
      c.coalesce ? coalesceTurns(rows) : rows,
      c.alert_nudge_times,
      c.session_end_s,
      c.cap_s,
    );
    expect(
      got.nudges.map((n) => ({ code: n.code, t: n.t, turn_index: n.turnIndex, delivered: n.delivered })),
    ).toEqual(c.expected.nudges);
    expect(got.calm.longestS).toBeCloseTo(c.expected.calm.longest_s, 6);
    expect(got.calm.badge).toBe(c.expected.calm.badge);
    // Every emitted nudge carries a human line — the report prints it verbatim.
    for (const n of got.nudges) expect(n.detail.length).toBeGreaterThan(0);
  });
});

describe("repair language", () => {
  it("survives punctuation, capitals and the apostrophes on-device STT drops", () => {
    expect(normalizeForRepair("I'm sorry.")).toBe("im sorry");
    expect(hasRepairLanguage("I'm sorry.")).toBe(true);
    expect(hasRepairLanguage("im  SORRY!!")).toBe(true);
    expect(hasRepairLanguage("You're right — that was on me")).toBe(true);
  });

  it("does not fire on the absence of a repair", () => {
    expect(hasRepairLanguage("")).toBe(false);
    expect(hasRepairLanguage("whatever")).toBe(false);
    expect(hasRepairLanguage("you are wrong")).toBe(false);
    // The lexicon is first-person on purpose: "sorry not sorry" is sarcasm,
    // and "you are wrong" is the opposite of a repair.
    expect(hasRepairLanguage("sorry not sorry")).toBe(false);
  });
});

describe("selfHeatLevel", () => {
  const base: PositiveTurn = {
    index: 0,
    start: 0,
    end: 1,
    speaker: "You",
    isSelf: true,
    text: "",
    loudLevel: null,
    toneHeat: null,
    toneNegativity: null,
    cutIn: false,
  };

  it("is null when neither ladder could be measured — never 0 as a guess", () => {
    expect(selfHeatLevel(base)).toBeNull();
  });

  it("takes the higher of the loudness and text-tone rungs", () => {
    expect(selfHeatLevel({ ...base, loudLevel: 1, toneHeat: 88 })).toBe(3);
    expect(selfHeatLevel({ ...base, loudLevel: 3, toneHeat: 10 })).toBe(3);
    expect(selfHeatLevel({ ...base, loudLevel: 0, toneHeat: null })).toBe(0);
    expect(selfHeatLevel({ ...base, loudLevel: null, toneHeat: 40 })).toBe(0);
  });
});
