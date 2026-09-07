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
  MIN_AMPLITUDE,
  MIN_GAP_MS,
  MIN_ON_MS,
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
  constants: { min_gap_ms: number; positive_cap_s: number; min_on_ms: number; min_amplitude: number };
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
    expect(doc.constants.min_on_ms).toBe(MIN_ON_MS);
    expect(doc.constants.min_amplitude).toBe(MIN_AMPLITUDE);
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
            expect(a[i]).toBeLessThanOrEqual(255);
            // The measured perceptibility floor — it binds the soft positives
            // too: a cue nobody can feel is not a soft cue, it is a missing one.
            expect({ code: e.code, lvl, ms: ms >= MIN_ON_MS }).toEqual({ code: e.code, lvl, ms: true });
            expect({ code: e.code, lvl, amp: a[i] >= MIN_AMPLITUDE }).toEqual({ code: e.code, lvl, amp: true });
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

  // --- the bug the owner found by hand, now a gate ------------------------

  const taps = (t: number[]) => t.filter((_, i) => i % 2 === 1);
  const gaps = (t: number[]) => t.filter((_, i) => i % 2 === 0).slice(1);

  /** Would a PHONE be unable to tell these two cues apart? Amplitude is
   *  removed on purpose: React Native can only switch an Android motor on and
   *  off, so strength is not a channel there at all. */
  const confusable = (a: number[], b: number[], gapMs: number, ratio: number) => {
    const ta = taps(a);
    const tb = taps(b);
    if (ta.length !== tb.length) return false;
    const ga = gaps(a);
    const gb = gaps(b);
    if (ga.some((x, i) => Math.abs(x - gb[i]) >= gapMs)) return false;
    return ta.every((x, i) => Math.max(x, tb[i]) / Math.min(x, tb[i]) < ratio);
  };

  it("no two cues are confusable once amplitude is removed", () => {
    // Shipped 📈 level 3 was three 100 ms taps and 📉 was three 70 ms taps —
    // they differed ONLY in amplitude, so both reached the motor as "three taps
    // 170 ms apart" and the owner reported them as the same cue. 👂 and 🤝 were
    // byte-for-byte identical.
    const c = CASES.no_two_cues_are_confusable as {
      confusable_gap_ms: number;
      confusable_tap_ratio: number;
      expected_confusable_pairs: string[][];
    };
    const cues = NUDGE_VOCABULARY.filter((e) => e.haptic).flatMap((e) =>
      Object.entries(e.haptic!).map(([lvl, w]) => [`${e.code} L${lvl}`, w.timingsMs] as const),
    );
    const pairs: string[][] = [];
    for (let i = 0; i < cues.length; i++) {
      for (let j = i + 1; j < cues.length; j++) {
        if (confusable(cues[i][1], cues[j][1], c.confusable_gap_ms, c.confusable_tap_ratio)) {
          pairs.push([cues[i][0], cues[j][0]]);
        }
      }
    }
    expect(pairs).toEqual(c.expected_confusable_pairs);
  });

  it("the rising and falling ramps are opposites in tap LENGTH, not just strength", () => {
    const rising = taps(hapticFor("H", 3)!.timingsMs);
    const falling = taps(hapticFor("D", 1)!.timingsMs);
    expect(rising).toEqual([...rising].sort((a, b) => a - b));
    expect(falling).toEqual([...falling].sort((a, b) => b - a));
    expect(falling).toEqual([...rising].reverse());
    // ...and by enough to feel, not a few milliseconds.
    expect(Math.max(...rising) / Math.min(...rising)).toBeGreaterThanOrEqual(3);
  });

  it("the first Heated tap is long enough to notice on a phone", () => {
    // 75 ms was reported as "does nothing" on a Pixel (2026-09-06).
    expect(hapticFor("H", 1)!.timingsMs[1]).toBeGreaterThanOrEqual(100);
  });
});
