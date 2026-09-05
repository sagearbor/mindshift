/**
 * src/live/naturalTurn.ts — replays the SAME golden vectors
 * server/tests/fixtures/policy_vectors/natural_turn.json (see its "_schema")
 * that server/tests/test_natural_turn.py replays against server/natural_turn.py.
 * classify_utterance/label_turns/merge_primaries are meant to be bit-identical
 * across the two ports; this is the reference consumer for that guarantee. If
 * this ever disagrees with the Python driver, the fixture's "_schema" wins —
 * fix whichever port is wrong, never the fixture, unless the upstream
 * NaturalTurn algorithm itself is being re-derived.
 *
 * sentences_from_words/collapse_short_pauses (the server-only pre-stage that
 * fixes the port-vs-published recall regression, see server/natural_turn.py's
 * module docstring) have no TS counterpart and are NOT covered here — the live
 * phone path hands classifyUtterance whole VAD-finalized turns, so there is
 * nothing to stitch or collapse.
 */
import { classifyUtterance, labelTurns, liveTurnKind, mergePrimaries } from "../src/live/naturalTurn";
import { loadFixture } from "../src/live/testing/synth";

interface ClassificationCase {
  name: string;
  text: string;
  duration_s: number;
  expected: "backchannel" | "secondary" | "other" | null;
}

interface ContainmentUtterance {
  speaker: string;
  start: number;
  end: number;
  text: string;
}

interface ExpectedLabel {
  speaker: string;
  start: number;
  end: number;
  kind: string;
  interjects: number | null;
}

interface ExpectedMerged {
  speaker: string;
  start: number;
  end: number;
  text: string;
  parts: number;
  attached_kinds: string[];
}

interface ContainmentCase {
  name: string;
  description: string;
  utterances: ContainmentUtterance[];
  expected_labels: ExpectedLabel[];
  expected_merged: ExpectedMerged[];
}

interface FixtureDoc {
  _schema: { version: number };
  classification_cases: ClassificationCase[];
  containment_cases: ContainmentCase[];
}

const doc = loadFixture<FixtureDoc>("natural_turn.json");
expect(doc._schema.version).toBe(1);

describe("natural_turn.json golden vectors: classify_utterance", () => {
  it("fixture covers every decision branch and has at least 30 cases", () => {
    const outcomes = new Set(doc.classification_cases.map((c) => c.expected));
    expect(outcomes).toEqual(new Set(["backchannel", "secondary", "other", null]));
    expect(doc.classification_cases.length).toBeGreaterThanOrEqual(30);
    const names = new Set(doc.classification_cases.map((c) => c.name));
    expect(names.size).toBe(doc.classification_cases.length);
  });

  it.each(doc.classification_cases.map((c) => [c.name, c] as const))(
    "replays identically: %s",
    (_name, c) => {
      expect(classifyUtterance(c.text, c.duration_s)).toBe(c.expected);
      const wantLive = c.expected === "backchannel" ? "backchannel" : "primary";
      expect(liveTurnKind(c.text, c.duration_s)).toBe(wantLive);
    },
  );
});

describe("natural_turn.json golden vectors: labelTurns + mergePrimaries", () => {
  it("fixture has at least 3 containment/merge scenarios with unique names", () => {
    expect(doc.containment_cases.length).toBeGreaterThanOrEqual(3);
    const names = new Set(doc.containment_cases.map((c) => c.name));
    expect(names.size).toBe(doc.containment_cases.length);
  });

  it.each(doc.containment_cases.map((c) => [c.name, c] as const))(
    "replays identically: %s",
    (_name, c) => {
      const labeled = labelTurns(c.utterances);
      expect(labeled).toHaveLength(c.expected_labels.length);
      labeled.forEach((got, i) => {
        const want = c.expected_labels[i];
        expect(got.speaker).toBe(want.speaker);
        expect(got.start).toBe(want.start);
        expect(got.end).toBe(want.end);
        expect(got.kind).toBe(want.kind);
        expect(got.interjects).toBe(want.interjects);
      });

      const merged = mergePrimaries(labeled);
      expect(merged).toHaveLength(c.expected_merged.length);
      merged.forEach((got, i) => {
        const want = c.expected_merged[i];
        expect(got.speaker).toBe(want.speaker);
        expect(got.start).toBe(want.start);
        expect(got.end).toBe(want.end);
        expect(got.text).toBe(want.text);
        expect(got.parts).toBe(want.parts);
        expect(got.attached.map((a) => a.kind)).toEqual(want.attached_kinds);
      });
    },
  );
});
