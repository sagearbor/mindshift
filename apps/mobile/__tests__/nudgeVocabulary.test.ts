/**
 * src/live/nudgeVocabulary.ts — replays
 * server/tests/fixtures/policy_vectors/nudge_vocabulary.json, the same file
 * server/tests/test_nudge_vocabulary_vectors.py and the watch's
 * NudgeVocabularyVectorsTest.kt replay. An emoji or a millisecond can only
 * change in one place.
 */
import {
  codeForVectors,
  hapticFor,
  HAPTIC_GAP_EXCEPTIONS,
  iconFor,
  MIN_GAP_MS,
  NUDGE_VOCABULARY,
  POSITIVE_CAP_S,
  PositiveNudgeGate,
  vocabularyFor,
  vocabularyForCode,
  type NudgeCode,
} from "../src/live/nudgeVocabulary";
import { loadFixture } from "../src/live/testing/synth";

interface WaveSpec {
  timings_ms: number[];
  amplitudes: number[];
}
interface EntrySpec {
  code: NudgeCode;
  vector: string;
  icon: string;
  color: string;
  name: string;
  meaning: string;
  polarity: string;
  sources: string[];
  flash_text: string | null;
  summary_only: boolean;
  watch_only: boolean;
  levels: number[];
  haptic: Record<string, WaveSpec> | null;
}
interface VocabDoc {
  _schema: { version: number; haptic_encoding: { haptic_gap_exceptions: string[] } };
  constants: { min_gap_ms: number; positive_cap_s: number };
  vocabulary: EntrySpec[];
  cases: Record<string, unknown>[];
}

const doc = loadFixture<VocabDoc>("nudge_vocabulary.json");
const CASES = Object.fromEntries(doc.cases.map((c) => [c.name as string, c]));

describe("nudge_vocabulary.json golden contract", () => {
  it("schema version and constants match the module", () => {
    expect(doc._schema.version).toBe(1);
    expect(doc.constants.min_gap_ms).toBe(MIN_GAP_MS);
    expect(doc.constants.positive_cap_s).toBe(POSITIVE_CAP_S);
    expect(doc._schema.haptic_encoding.haptic_gap_exceptions).toEqual([...HAPTIC_GAP_EXCEPTIONS]);
  });

  it("the module is field-for-field the fixture, in order", () => {
    expect(NUDGE_VOCABULARY.length).toBe(doc.vocabulary.length);
    doc.vocabulary.forEach((spec, i) => {
      const e = NUDGE_VOCABULARY[i];
      expect(e.code).toBe(spec.code);
      expect(e.vector).toBe(spec.vector);
      expect(e.icon).toBe(spec.icon);
      expect(e.color).toBe(spec.color);
      expect(e.name).toBe(spec.name);
      expect(e.meaning).toBe(spec.meaning);
      expect(e.polarity).toBe(spec.polarity);
      expect(e.sources).toEqual(spec.sources);
      expect(e.flashText).toBe(spec.flash_text);
      expect(e.summaryOnly).toBe(spec.summary_only);
      expect(e.watchOnly).toBe(spec.watch_only);
      expect(e.levels).toEqual(spec.levels);
      if (spec.haptic === null) {
        expect(e.haptic).toBeNull();
        return;
      }
      expect(e.haptic).not.toBeNull();
      const levels = Object.keys(spec.haptic).map(Number).sort();
      expect(Object.keys(e.haptic!).map(Number).sort()).toEqual(levels);
      for (const [k, wave] of Object.entries(spec.haptic)) {
        const got = e.haptic![Number(k)];
        expect({ code: e.code, level: k, t: got.timingsMs }).toEqual({ code: e.code, level: k, t: wave.timings_ms });
        expect(got.amplitudes).toEqual(wave.amplitudes);
      }
    });
  });

  it("every code is unique and stable", () => {
    const expected = CASES.every_code_is_unique_and_stable.expected_codes as string[];
    expect(NUDGE_VOCABULARY.map((e) => e.code)).toEqual(expected);
    expect(new Set(NUDGE_VOCABULARY.map((e) => e.vector)).size).toBe(expected.length);
  });

  it("alert codes have three levels, positives have one", () => {
    const c = CASES.alert_codes_have_three_levels_positives_have_one as Record<string, string[]>;
    expect(NUDGE_VOCABULARY.filter((e) => e.polarity === "alert").map((e) => e.code)).toEqual(c.expected_alert_codes);
    expect(NUDGE_VOCABULARY.filter((e) => e.polarity === "positive").map((e) => e.code)).toEqual(
      c.expected_positive_codes,
    );
    for (const e of NUDGE_VOCABULARY) expect(e.levels).toEqual(e.polarity === "alert" ? [1, 2, 3] : [1]);
  });

  it.each(NUDGE_VOCABULARY.filter((e) => e.haptic).map((e) => [e.code, e] as const))(
    "%s: every waveform is well formed",
    (_code, e) => {
      for (const [lvl, wave] of Object.entries(e.haptic!)) {
        const { timingsMs: t, amplitudes: a } = wave;
        expect(t.length).toBe(a.length);
        expect(t.length % 2).toBe(0);
        expect(t.length).toBeGreaterThanOrEqual(2);
        expect(t[0]).toBe(0);
        t.forEach((ms, i) => {
          if (i % 2 === 0) {
            expect(a[i]).toBe(0);
            if (i > 0) {
              const floor = HAPTIC_GAP_EXCEPTIONS.includes(e.vector) ? 0 : MIN_GAP_MS;
              expect({ code: e.code, lvl, gap: ms >= floor }).toEqual({ code: e.code, lvl, gap: true });
            }
          } else {
            expect(ms).toBeGreaterThan(0);
            expect(a[i]).toBeGreaterThanOrEqual(1);
            expect(a[i]).toBeLessThanOrEqual(255);
          }
        });
      }
    },
  );

  it("the level is carried by rhythm — ON time strictly grows, amplitude aside", () => {
    const expected = CASES.level_is_carried_by_rhythm.expected_on_ms as Record<string, number[]>;
    const onMs = (code: string, lvl: number) =>
      hapticFor(code, lvl)!.timingsMs.filter((_, i) => i % 2 === 1).reduce((s, x) => s + x, 0);
    for (const [code, want] of Object.entries(expected)) {
      const got = [1, 2, 3].map((l) => onMs(code, l));
      expect({ code, got }).toEqual({ code, got: want });
      expect(got[0]).toBeLessThan(got[1]);
      expect(got[1]).toBeLessThan(got[2]);
    }
  });

  it("the positive cap admits one per two minutes across D/E/R", () => {
    const c = CASES.positive_cap_is_one_per_two_minutes as {
      offers: { t: number; code: string }[];
      expected_admitted: { t: number; code: string }[];
    };
    const gate = new PositiveNudgeGate();
    expect(c.offers.filter((o) => gate.admit(o.t))).toEqual(c.expected_admitted);
  });

  it("lookups resolve detector names and vocabulary ids, and never invent an icon", () => {
    expect(iconFor("yelling")).toBe("📈");
    expect(iconFor("heated")).toBe("📈");
    expect(iconFor("nonesuch")).toBeNull();
    expect(vocabularyFor("aggressive_tone")?.code).toBe("H");
    expect(vocabularyForCode("Z")).toBeNull();
    expect(codeForVectors(["airtime", "yelling"])).toBe("H");
    expect(codeForVectors(["airtime", "interrupting"])).toBe("C");
    expect(codeForVectors(["nonesuch"])).toBeNull();
    expect(codeForVectors([])).toBeNull();
  });

  it("positives are unleveled, K never buzzes, and a bad level is silent", () => {
    for (const code of ["D", "E", "R"]) {
      expect(hapticFor(code, 3)).toEqual(hapticFor(code, 1));
    }
    expect(hapticFor("K", 1)).toBeNull();
    expect(vocabularyForCode("K")!.summaryOnly).toBe(true);
    expect(hapticFor("H", 0)).toBeNull();
    expect(hapticFor("H", 4)).toBeNull();
  });
});
