import {
  heatBand,
  horsemenTotal,
  buildSpeakerBars,
  barPct,
  deriveVerdict,
  verdictColors,
} from "../src/screens/glanceSummary";
import type { AnalyzePerSpeaker, AnalyzePerTurn } from "../src/api/client";

function speaker(over: Partial<AnalyzePerSpeaker> = {}): AnalyzePerSpeaker {
  return {
    turns: 2,
    talk_share: 0.5,
    avg_heat: 30,
    peak_heat: 40,
    peak_turn_index: 1,
    heat_variance: 100,
    interruptions: null,
    horsemen: { criticism: 0, contempt: 0, defensiveness: 0, stonewalling: 0 },
    repair_attempts: 0,
    repairs_accepted: 0,
    ...over,
  };
}

function turn(over: Partial<AnalyzePerTurn> & { index: number; heat: number }): AnalyzePerTurn {
  return {
    speaker: "Alice",
    markers: [],
    is_spike: false,
    trigger_phrase: null,
    ...over,
  };
}

describe("heatBand", () => {
  it("splits the 0–100 scale into calm / strained / rough thirds", () => {
    expect(heatBand(0)).toBe("calm");
    expect(heatBand(33)).toBe("calm");
    expect(heatBand(34)).toBe("strained");
    expect(heatBand(66)).toBe("strained");
    expect(heatBand(67)).toBe("rough");
    expect(heatBand(100)).toBe("rough");
  });
});

describe("horsemenTotal", () => {
  it("sums the four horsemen markers", () => {
    expect(
      horsemenTotal(
        speaker({
          horsemen: { criticism: 1, contempt: 2, defensiveness: 0, stonewalling: 3 },
        }),
      ),
    ).toBe(6);
  });
});

describe("barPct", () => {
  it("clamps to 0–100 and handles the edges honestly", () => {
    expect(barPct(0, 100)).toBe(0);
    expect(barPct(50, 100)).toBe(50);
    expect(barPct(100, 100)).toBe(100);
    // Over the max fills the bar rather than overflowing.
    expect(barPct(150, 100)).toBe(100);
    // Talk-share scale (max 1).
    expect(barPct(0.25, 1)).toBe(25);
    // A non-positive max yields an empty bar, never a divide-by-zero.
    expect(barPct(5, 0)).toBe(0);
    // Negative value clamps to 0.
    expect(barPct(-10, 100)).toBe(0);
    // Non-finite input is treated as empty.
    expect(barPct(NaN, 100)).toBe(0);
  });
});

describe("buildSpeakerBars", () => {
  it("maps per_speaker to rows with labels, colors, and the heat-ramp color", () => {
    const bars = buildSpeakerBars(
      {
        Alice: speaker({ avg_heat: 20, talk_share: 0.7, repair_attempts: 2 }),
        Bob: speaker({
          avg_heat: 80,
          talk_share: 0.3,
          horsemen: { criticism: 1, contempt: 1, defensiveness: 0, stonewalling: 0 },
        }),
      },
      { Alice: { display_label: "You", label_source: "enrolled" } },
    );
    expect(bars).toHaveLength(2);
    expect(bars[0].id).toBe("Alice");
    expect(bars[0].label).toBe("You"); // display label applied
    expect(bars[0].avgHeat).toBe(20);
    expect(bars[0].talkShare).toBe(0.7);
    expect(bars[0].repairAttempts).toBe(2);
    expect(bars[0].horsemenTotal).toBe(0);
    expect(typeof bars[0].heatBarColor).toBe("string");
    expect(bars[0].heatBarColor).toMatch(/^#/);
    // Bob: raw id fallback (no label entry) + summed horsemen.
    expect(bars[1].label).toBe("Bob");
    expect(bars[1].horsemenTotal).toBe(2);
  });

  it("returns an empty array for no speakers (caller then omits the summary)", () => {
    expect(buildSpeakerBars({})).toEqual([]);
  });
});

describe("deriveVerdict", () => {
  it("returns null for an empty conversation", () => {
    expect(deriveVerdict([])).toBeNull();
  });

  it("reads a calm conversation as calm (no false drama)", () => {
    const v = deriveVerdict([
      turn({ index: 0, heat: 12 }),
      turn({ index: 1, heat: 18 }),
      turn({ index: 2, heat: 15 }),
    ]);
    expect(v).not.toBeNull();
    expect(v!.tone).toBe("calm");
    expect(v!.text).toContain("calm");
  });

  it("anchors a single heated moment to its clock time when timing is present", () => {
    const perTurn = [
      turn({ index: 0, heat: 15 }),
      turn({ index: 1, heat: 20 }),
      turn({ index: 2, heat: 88, is_spike: true }),
    ];
    const timing = [
      { start_time: 0, end_time: 10 },
      { start_time: 10, end_time: 20 },
      { start_time: 161, end_time: 170 }, // 2:41
    ];
    const v = deriveVerdict(perTurn, timing);
    expect(v!.tone).toBe("mixed");
    expect(v!.text).toContain("one heated moment");
    expect(v!.text).toContain("2:41");
  });

  it("omits the timestamp when there is no timing", () => {
    const v = deriveVerdict([
      turn({ index: 0, heat: 15 }),
      turn({ index: 1, heat: 20 }),
      turn({ index: 2, heat: 88, is_spike: true }),
    ]);
    expect(v!.text).toContain("one heated moment");
    expect(v!.text).not.toContain(":");
  });

  it("counts multiple spikes as a few heated moments", () => {
    const v = deriveVerdict([
      turn({ index: 0, heat: 80, is_spike: true }),
      turn({ index: 1, heat: 30 }),
      turn({ index: 2, heat: 85, is_spike: true }),
    ]);
    expect(v!.tone).toBe("heated");
    expect(v!.text).toContain("2");
  });

  it("detects a rising trend toward the end", () => {
    const v = deriveVerdict([
      turn({ index: 0, heat: 10 }),
      turn({ index: 1, heat: 12 }),
      turn({ index: 2, heat: 15 }),
      turn({ index: 3, heat: 40 }),
      turn({ index: 4, heat: 45 }),
      turn({ index: 5, heat: 50 }),
    ]);
    expect(v!.text).toContain("rose toward the end");
  });

  it("detects a cooling trend by the end", () => {
    const v = deriveVerdict([
      turn({ index: 0, heat: 55 }),
      turn({ index: 1, heat: 50 }),
      turn({ index: 2, heat: 45 }),
      turn({ index: 3, heat: 20 }),
      turn({ index: 4, heat: 15 }),
      turn({ index: 5, heat: 10 }),
    ]);
    expect(v!.text).toContain("cooled");
  });

  it("handles a single-turn conversation without a trend clause", () => {
    const v = deriveVerdict([turn({ index: 0, heat: 20 })]);
    expect(v!.tone).toBe("calm");
    expect(v!.text).not.toContain("end");
  });
});

describe("verdictColors", () => {
  it("gives each tone a distinct background/foreground pair", () => {
    const calm = verdictColors("calm");
    const mixed = verdictColors("mixed");
    const heated = verdictColors("heated");
    expect(calm.bg).not.toBe(mixed.bg);
    expect(mixed.bg).not.toBe(heated.bg);
    expect(calm.fg).toMatch(/^#/);
  });
});
