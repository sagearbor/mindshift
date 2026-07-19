import { summarizeReanalyze } from "../src/screens/reanalyzeDelta";
import type { AnalyzeResult } from "../src/api/client";

/** Minimal AnalyzeResult with the fields the summary reads. */
function analysis(
  perTurnHeats: number[],
  reportCards?: Record<string, number>,
  labels?: Record<string, { display_label: string; label_source: string }>,
): AnalyzeResult {
  return {
    per_turn: perTurnHeats.map((heat, index) => ({
      index,
      speaker: "S",
      heat,
      markers: [],
      is_spike: false,
      trigger_phrase: null,
    })),
    per_speaker: {},
    report_cards: reportCards
      ? Object.fromEntries(
          Object.entries(reportCards).map(([id, score]) => [
            id,
            { score, headline: "", did_well: "", work_on: "" },
          ]),
        )
      : undefined,
    dynamics: {
      coupling: { strength: null, leader: null, description: "" },
      deescalation: { who_first: null, follow_rate: null, description: "" },
      triggers: [],
      requests: [],
    },
    narrative: "",
    speaker_labels: labels,
  };
}

describe("summarizeReanalyze", () => {
  it("reports per-speaker score deltas for speakers scored in both runs", () => {
    const before = analysis([20, 88], { A: 60, B: 55 });
    const after = analysis([20, 80], { A: 68, B: 55 });
    const s = summarizeReanalyze(before, after);
    expect(s.changed).toBe(true);
    const a = s.scoreDeltas.find((d) => d.id === "A")!;
    expect(a.before).toBe(60);
    expect(a.after).toBe(68);
    expect(a.delta).toBe(8);
    const b = s.scoreDeltas.find((d) => d.id === "B")!;
    expect(b.delta).toBe(0);
  });

  it("uses display labels when provided", () => {
    const labels = {
      SPEAKER_00: { display_label: "Maria", label_source: "name" },
    };
    const before = analysis([30], { SPEAKER_00: 50 }, labels);
    const after = analysis([30], { SPEAKER_00: 55 }, labels);
    const s = summarizeReanalyze(before, after, labels);
    expect(s.scoreDeltas[0].label).toBe("Maria");
  });

  it("tracks peak-heat change even without report cards (old server)", () => {
    const before = analysis([20, 88]);
    const after = analysis([20, 74]);
    const s = summarizeReanalyze(before, after);
    expect(s.scoreDeltas).toHaveLength(0);
    expect(s.peakBefore).toBe(88);
    expect(s.peakAfter).toBe(74);
    expect(s.peakDelta).toBe(-14);
    expect(s.changed).toBe(true);
  });

  it("is honest about no change when nothing moved", () => {
    const before = analysis([20, 88], { A: 60 });
    const after = analysis([20, 88], { A: 60 });
    const s = summarizeReanalyze(before, after);
    expect(s.changed).toBe(false);
    expect(s.scoreDeltas.every((d) => d.delta === 0)).toBe(true);
    expect(s.peakDelta).toBe(0);
  });

  it("never invents a delta for a speaker missing from one run", () => {
    const before = analysis([40], { A: 60 });
    const after = analysis([40], { A: 60, B: 70 }); // B is new
    const s = summarizeReanalyze(before, after);
    // Only A is comparable; B is skipped (not scored in `before`).
    expect(s.scoreDeltas.map((d) => d.id)).toEqual(["A"]);
  });

  it("returns null peak when there are no turns", () => {
    const s = summarizeReanalyze(analysis([]), analysis([]));
    expect(s.peakBefore).toBeNull();
    expect(s.peakAfter).toBeNull();
    expect(s.peakDelta).toBeNull();
    expect(s.changed).toBe(false);
  });
});
