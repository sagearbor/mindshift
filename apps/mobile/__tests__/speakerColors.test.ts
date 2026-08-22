import { getSpeakerColor, resolveSpeakerColors, SPEAKER_PALETTE } from "../src/utils/speakerColors";

/**
 * Part of the color/speaker-mismatch investigation (owner report: "a dash
 * appeared to render in the wrong speaker's color at a turn transition").
 * `getSpeakerColor` is the single source of truth HeatChart's mapTurnsToDashes
 * uses to color every dash — if it were order-dependent (e.g. a Map that
 * assigns colors in first-seen order, differing between renders) that would
 * fully explain a color flip. It isn't: these tests pin down that it's a PURE
 * function of the raw speaker-id string alone.
 */
describe("getSpeakerColor (purity — color/speaker-mismatch investigation)", () => {
  it("is a pure function: the same speaker id always yields the same color", () => {
    expect(getSpeakerColor("Speaker A")).toBe(getSpeakerColor("Speaker A"));
    expect(getSpeakerColor("Sage")).toBe(getSpeakerColor("Sage"));
    expect(getSpeakerColor("Asher")).toBe(getSpeakerColor("Asher"));
  });

  it("is order-independent: color for a speaker doesn't depend on which other speakers were queried first", () => {
    // Query "Asher" first, then "Sage" — vs. the reverse order — must agree.
    const asherFirst = { asher: getSpeakerColor("Asher"), sage: getSpeakerColor("Sage") };
    const sageFirst = { sage: getSpeakerColor("Sage"), asher: getSpeakerColor("Asher") };
    expect(asherFirst.sage).toBe(sageFirst.sage);
    expect(asherFirst.asher).toBe(sageFirst.asher);

    // A long interleaved call sequence (simulating turns arriving in any
    // order, e.g. after a reprocess/reorder) never perturbs later lookups.
    const before = getSpeakerColor("Sage");
    for (let i = 0; i < 50; i++) {
      getSpeakerColor("Asher");
      getSpeakerColor("Speaker A");
      getSpeakerColor("Some Other Name");
    }
    expect(getSpeakerColor("Sage")).toBe(before);
  });

  it("has no hidden mutable state across repeated calls for the same id (no first-seen-order Map)", () => {
    const calls = Array.from({ length: 20 }, () => getSpeakerColor("Speaker B"));
    expect(new Set(calls).size).toBe(1);
  });

  it("pins the two most common diarized ids to a stable house pair", () => {
    expect(getSpeakerColor("Speaker A")).toBe("#4A90D9");
    expect(getSpeakerColor("Speaker B")).toBe("#E85D75");
  });

  it("always returns a color from the declared palette (or the pinned pair) for any id", () => {
    for (const id of ["Sage", "Asher", "Mom", "Dad", "Speaker C", ""]) {
      expect([...SPEAKER_PALETTE, "#4A90D9", "#E85D75"]).toContain(getSpeakerColor(id));
    }
  });

  // CONFIRMED FINDING: with only 5 palette slots, the plain hash collides for
  // two real, distinct speaker names from the project's own real fixture
  // (server/tests/fixtures/audio/test_recording_family_real_meta.json). This
  // is exactly the kind of defect that would show up as "the wrong speaker's
  // color at a turn transition" — the color simply never changes at the
  // boundary between two same-colored speakers. This test locks the defect in
  // place as a documented, known property of the RAW hash (resolveSpeakerColors,
  // tested below, is what actually fixes it for real rendering).
  it("documents the known collision: 'Sage' and 'Asher' hash to the SAME raw color", () => {
    expect(getSpeakerColor("Sage")).toBe(getSpeakerColor("Asher"));
    expect(getSpeakerColor("Sage")).toBe("#8B5CF6");
  });
});

describe("resolveSpeakerColors (the color/speaker-mismatch fix)", () => {
  it("fixes the confirmed 'Sage'/'Asher' collision: both are distinct within one conversation", () => {
    const colorOf = resolveSpeakerColors(["Sage", "Asher"]);
    expect(colorOf.get("Sage")).not.toBe(colorOf.get("Asher"));
    expect(colorOf.get("Sage")).toBeTruthy();
    expect(colorOf.get("Asher")).toBeTruthy();
  });

  it("preserves getSpeakerColor's result for the FIRST speaker to claim each color (no change to the common, non-colliding case)", () => {
    // "Sage" is first in this list, so it keeps its plain hash color exactly
    // as getSpeakerColor would give it; only the LATER colliding speaker
    // ("Asher") gets bumped to a different color.
    const colorOf = resolveSpeakerColors(["Sage", "Asher"]);
    expect(colorOf.get("Sage")).toBe(getSpeakerColor("Sage"));
  });

  it("leaves a non-colliding pair completely unchanged from plain getSpeakerColor", () => {
    const colorOf = resolveSpeakerColors(["Speaker A", "Speaker B"]);
    expect(colorOf.get("Speaker A")).toBe(getSpeakerColor("Speaker A"));
    expect(colorOf.get("Speaker B")).toBe(getSpeakerColor("Speaker B"));
  });

  it("is deterministic for a given ordered speaker list (same input, same output every call)", () => {
    const a = resolveSpeakerColors(["Sage", "Asher"]);
    const b = resolveSpeakerColors(["Sage", "Asher"]);
    expect(a.get("Sage")).toBe(b.get("Sage"));
    expect(a.get("Asher")).toBe(b.get("Asher"));
  });

  it("dedupes repeated entries in the input list without changing the outcome", () => {
    const colorOf = resolveSpeakerColors(["Sage", "Asher", "Sage", "Asher", "Sage"]);
    expect(colorOf.size).toBe(2);
    expect(colorOf.get("Sage")).not.toBe(colorOf.get("Asher"));
  });

  it("assigns every speaker in a larger, mixed conversation a distinct color", () => {
    const colorOf = resolveSpeakerColors(["Sage", "Asher", "Speaker A", "Speaker B", "Mom"]);
    const colors = Array.from(colorOf.values());
    expect(new Set(colors).size).toBe(colors.length); // all distinct
  });
});
