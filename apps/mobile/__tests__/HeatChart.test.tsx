import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import HeatChart, {
  mapTurnsToLines,
  mapSimulatedToLines,
  playheadIndexForTime,
  playheadXForTime,
  mapTurnsToDashes,
  mapSimulatedToDashes,
  playheadXForSeconds,
  computeTalkShares,
  timingIsUsable,
  durationForTiming,
  buildTimeScrubCells,
  formatClock,
  type TurnTiming,
} from "../src/components/HeatChart";
import type { AnalyzePerTurn, SimulatedTurn } from "../src/api/client";
import { getSpeakerColor, resolveSpeakerColors } from "../src/utils/speakerColors";

// Real utterance timing, index-aligned with `perTurn`. Alice dominates the
// conversation: she talks 2s + 6s = 8s of the 10s, Bob 1s + 1s = 2s — an 80/20
// split, so the talk-share + dash-length assertions have a concrete answer.
const timedTiming = [
  { start_time: 0, end_time: 2 }, // turn 0, Alice
  { start_time: 2, end_time: 3 }, // turn 1, Bob
  { start_time: 3, end_time: 9 }, // turn 2, Alice (long)
  { start_time: 9, end_time: 10 }, // turn 3, Bob (spike)
];

/** A 2-speaker conversation with a spike on Bob's turn 3. */
const perTurn: AnalyzePerTurn[] = [
  { index: 0, speaker: "Alice", heat: 20, markers: [], is_spike: false, trigger_phrase: null },
  { index: 1, speaker: "Bob", heat: 30, markers: ["defensiveness"], is_spike: false, trigger_phrase: null },
  { index: 2, speaker: "Alice", heat: 45, markers: [], is_spike: false, trigger_phrase: null },
  { index: 3, speaker: "Bob", heat: 90, markers: ["contempt"], is_spike: true, trigger_phrase: "you always do this" },
];

// The react-native-svg mock renders each element as both a composite and a host
// node carrying the same testID, so we count DISTINCT testID values, not nodes.
function countIds(comp: renderer.ReactTestRenderer, prefix: string): number {
  const ids = new Set<string>();
  comp.root
    .findAll(
      (n) => typeof n.props?.testID === "string" && n.props.testID.startsWith(prefix),
    )
    .forEach((n) => ids.add(n.props.testID as string));
  return ids.size;
}

/** Drive the onLayout the SVG needs to know its width (react-test-renderer
 *  never fires layout on its own). */
function layout(comp: renderer.ReactTestRenderer, width = 300) {
  const node = comp.root.findAll((n) => typeof n.props?.onLayout === "function")[0];
  act(() => {
    node.props.onLayout({ nativeEvent: { layout: { width, height: 180 } } });
  });
}

describe("mapTurnsToLines (pure geometry)", () => {
  it("produces one line per distinct speaker, each carrying only its own turns", () => {
    const lines = mapTurnsToLines(perTurn, { width: 300, height: 180, padding: 16 });
    expect(lines.map((l) => l.speaker)).toEqual(["Alice", "Bob"]);
    expect(lines[0].points.map((p) => p.index)).toEqual([0, 2]);
    expect(lines[1].points.map((p) => p.index)).toEqual([1, 3]);
  });

  it("maps x by conversation-wide index and flags spike points", () => {
    const lines = mapTurnsToLines(perTurn, { width: 300, height: 180, padding: 16 });
    // Alice's first turn (index 0) sits at the left padding; higher heat -> higher (smaller y).
    expect(lines[0].points[0].x).toBeCloseTo(16);
    expect(lines[0].points[0].y).toBeGreaterThan(lines[1].points[1].y); // Alice heat 20 lower than Bob heat 90
    // Spike is carried through onto the point.
    expect(lines[1].points[1].isSpike).toBe(true);
    expect(lines[1].points[0].isSpike).toBe(false);
  });
});

describe("mapSimulatedToLines (overlay geometry)", () => {
  // Simulated projection spanning the pivot (index 2) to the last turn (3).
  const sim: SimulatedTurn[] = [
    { index: 2, speaker: "Alice", heat: 35 },
    { index: 3, speaker: "Bob", heat: 40 },
  ];
  const opts = { width: 300, height: 180, padding: 16, totalTurns: 4 };

  it("groups by speaker and only carries the pivot→end turns", () => {
    const lines = mapSimulatedToLines(sim, opts);
    expect(lines.map((l) => l.speaker)).toEqual(["Alice", "Bob"]);
    expect(lines[0].points.map((p) => p.index)).toEqual([2]);
    expect(lines[1].points.map((p) => p.index)).toEqual([3]);
  });

  it("aligns x with the real lines at the same conversation index", () => {
    // Real geometry over the full conversation (maxIndex derives to 3), and the
    // overlay pinned to the SAME totalTurns — a simulated point at index i must
    // land at the exact x of the real point at index i.
    const real = mapTurnsToLines(perTurn, { width: 300, height: 180, padding: 16 });
    const overlay = mapSimulatedToLines(sim, opts);

    const realAliceAt2 = real[0].points.find((p) => p.index === 2)!;
    const simAliceAt2 = overlay[0].points.find((p) => p.index === 2)!;
    expect(simAliceAt2.x).toBeCloseTo(realAliceAt2.x);

    const realBobAt3 = real[1].points.find((p) => p.index === 3)!;
    const simBobAt3 = overlay[1].points.find((p) => p.index === 3)!;
    expect(simBobAt3.x).toBeCloseTo(realBobAt3.x);
  });

  it("uses each speaker's own color and reflects heat in y (higher heat = smaller y)", () => {
    const lines = mapSimulatedToLines(
      [
        { index: 2, speaker: "Alice", heat: 10 },
        { index: 3, speaker: "Alice", heat: 90 },
      ],
      opts,
    );
    // One Alice line, two points; the hotter point sits higher on the chart.
    expect(lines).toHaveLength(1);
    expect(lines[0].points[1].y).toBeLessThan(lines[0].points[0].y);
  });
});

describe("playheadIndexForTime (pure)", () => {
  // Four contiguous turns: [0,3) [3,6) [6,9) [9,12), with a gap [12,14) after
  // the last one is impossible (it's last), but a gap between turns exists if
  // end < next start. Add a deliberate gap: turn 1 ends at 6, turn 2 starts at 7.
  const timing = [
    { start_time: 0, end_time: 3 },
    { start_time: 3, end_time: 6 },
    { start_time: 7, end_time: 9 }, // gap 6→7
    { start_time: 9, end_time: 12 },
  ];

  it("returns null when there is no timing", () => {
    expect(playheadIndexForTime(5, [])).toBeNull();
  });

  it("returns the turn whose [start,end) contains the time (exact containment)", () => {
    expect(playheadIndexForTime(0, timing)).toBe(0);
    expect(playheadIndexForTime(2.9, timing)).toBe(0);
    expect(playheadIndexForTime(3, timing)).toBe(1); // start is inclusive
    expect(playheadIndexForTime(8, timing)).toBe(2);
  });

  it("falls back to the previous turn in a between-turns gap", () => {
    // 6.5 is in the silent gap between turn 1 (ends 6) and turn 2 (starts 7).
    expect(playheadIndexForTime(6.5, timing)).toBe(1);
  });

  it("clamps to the first turn before the conversation starts", () => {
    expect(playheadIndexForTime(-5, timing)).toBe(0);
  });

  it("clamps to the last turn after the conversation ends", () => {
    expect(playheadIndexForTime(100, timing)).toBe(3);
    expect(playheadIndexForTime(12, timing)).toBe(3); // exactly at last end
  });
});

describe("playheadXForTime (pure)", () => {
  const timing = [
    { start_time: 0, end_time: 3 },
    { start_time: 3, end_time: 6 },
    { start_time: 6, end_time: 9 },
    { start_time: 9, end_time: 12 },
  ];
  const opts = { width: 300, padding: 16, totalTurns: 4 };

  it("returns null when there is no timing", () => {
    expect(playheadXForTime(5, [], opts)).toBeNull();
  });

  it("lands the playhead exactly on the current turn's line x", () => {
    // Real geometry over the same 4-turn conversation shares the x-scale.
    const real = mapTurnsToLines(perTurn, {
      width: 300,
      height: 180,
      padding: 16,
      totalTurns: 4,
    });
    // A time inside turn 3 → x of the real point at index 3 (Bob's spike).
    const bobAt3 = real[1].points.find((p) => p.index === 3)!;
    expect(playheadXForTime(10, timing, opts)).toBeCloseTo(bobAt3.x);
    // A time inside turn 0 → left padding.
    expect(playheadXForTime(1, timing, opts)).toBeCloseTo(16);
  });
});

describe("HeatChart", () => {
  it("renders one polyline per speaker and larger spike dots", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<HeatChart perTurn={perTurn} />);
    });
    layout(comp);

    // Two speakers -> two polylines.
    expect(countIds(comp, "heat-line-")).toBe(2);
    // The single spike turn gets a spike dot; the other three are normal points.
    expect(countIds(comp, "heat-spike-")).toBe(1);
    expect(countIds(comp, "heat-point-")).toBe(3);
    act(() => comp.unmount());
  });

  it("shows the turn inspector with the selected turn's content", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<HeatChart perTurn={perTurn} />);
    });
    layout(comp);

    // Nothing selected initially.
    expect(comp.root.findAll((n) => n.props?.testID === "turn-inspector")).toHaveLength(0);

    // Tap the scrubber cell for Bob's spike turn (index 3).
    const cell = comp.root.find((n) => n.props?.testID === "scrub-3");
    act(() => cell.props.onPress());

    const inspector = comp.root.find((n) => n.props?.testID === "turn-inspector");
    expect(inspector).toBeTruthy();
    // The trigger phrase for that turn is surfaced.
    const trigger = comp.root.find((n) => n.props?.testID === "turn-inspector-trigger");
    expect(trigger).toBeTruthy();
    act(() => comp.unmount());
  });

  const sim: SimulatedTurn[] = [
    { index: 2, speaker: "Alice", heat: 35 },
    { index: 3, speaker: "Bob", heat: 40 },
  ];

  it("draws a dashed overlay line per simulated speaker when the overlay is shown", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart perTurn={perTurn} simulated={sim} showSimulation />,
      );
    });
    layout(comp);

    // One dashed line per simulated speaker, plus a "simulated" legend entry.
    expect(countIds(comp, "sim-line-")).toBe(2);
    expect(comp.root.findAll((n) => n.props?.testID === "legend-simulated").length)
      .toBeGreaterThan(0);
    // Real solid lines still present alongside.
    expect(countIds(comp, "heat-line-")).toBe(2);
    act(() => comp.unmount());
  });

  it("hides the overlay (no refetch) when showSimulation is false, keeping the toggle", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart perTurn={perTurn} simulated={sim} showSimulation={false} />,
      );
    });
    layout(comp);

    expect(countIds(comp, "sim-line-")).toBe(0);
    expect(comp.root.findAll((n) => n.props?.testID === "legend-simulated")).toHaveLength(0);
    // Toggle chip persists so the overlay can be re-shown.
    expect(comp.root.find((n) => n.props?.testID === "simulation-toggle")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("renders outline voice chips (non-baseline only) when the selected turn has voice", () => {
    // Turn 0 carries voice: loud + fast are notable; pitch "mid" is baseline and
    // must be omitted (up-to-three chips, not always three).
    const withVoice: AnalyzePerTurn[] = [
      {
        index: 0,
        speaker: "Alice",
        heat: 20,
        markers: [],
        is_spike: false,
        trigger_phrase: null,
        voice: { energy_label: "loud", pitch_label: "mid", rate_label: "fast" },
      },
      { index: 1, speaker: "Bob", heat: 30, markers: [], is_spike: false, trigger_phrase: null },
    ];
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<HeatChart perTurn={withVoice} />);
    });
    layout(comp);

    act(() => comp.root.find((n) => n.props?.testID === "scrub-0").props.onPress());

    expect(comp.root.find((n) => n.props?.testID === "voice-chip-energy")).toBeTruthy();
    expect(comp.root.find((n) => n.props?.testID === "voice-chip-rate")).toBeTruthy();
    // Baseline pitch ("mid") produces no chip.
    expect(comp.root.findAll((n) => n.props?.testID === "voice-chip-pitch")).toHaveLength(0);
    const text = JSON.stringify(comp.toJSON());
    expect(text).toContain("loud");
    expect(text).toContain("fast");
    act(() => comp.unmount());
  });

  it("renders no pitch chip when pitch is null (unvoiced turn) — never an empty chip", () => {
    // Regression for a review CRITICAL: the server sends pitch_label null for
    // turns with too little voiced speech; that must not render a blank chip.
    const nullPitch: AnalyzePerTurn[] = [
      {
        index: 0,
        speaker: "Alice",
        heat: 20,
        markers: [],
        is_spike: false,
        trigger_phrase: null,
        voice: { energy_label: "loud", pitch_label: null, rate_label: "normal" },
      },
      { index: 1, speaker: "Bob", heat: 30, markers: [], is_spike: false, trigger_phrase: null },
    ];
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<HeatChart perTurn={nullPitch} />);
    });
    layout(comp);
    act(() => comp.root.find((n) => n.props?.testID === "scrub-0").props.onPress());
    expect(comp.root.find((n) => n.props?.testID === "voice-chip-energy")).toBeTruthy();
    expect(comp.root.findAll((n) => n.props?.testID === "voice-chip-pitch")).toHaveLength(0);
    act(() => comp.unmount());
  });

  it("renders no voice chips when the selected turn has no voice data", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<HeatChart perTurn={perTurn} />);
    });
    layout(comp);
    // perTurn (top of file) carries no voice field at all.
    act(() => comp.root.find((n) => n.props?.testID === "scrub-3").props.onPress());
    expect(comp.root.findAll((n) => typeof n.props?.testID === "string" && n.props.testID.startsWith("voice-chip-"))).toHaveLength(0);
    act(() => comp.unmount());
  });

  it("fires onWhatIf with the selected turn index from the inspector button", () => {
    const onWhatIf = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart perTurn={perTurn} onWhatIf={onWhatIf} />,
      );
    });
    layout(comp);

    // Select turn 3, then tap the what-if button.
    act(() => comp.root.find((n) => n.props?.testID === "scrub-3").props.onPress());
    act(() =>
      comp.root.find((n) => n.props?.testID === "what-if-button").props.onPress(),
    );
    expect(onWhatIf).toHaveBeenCalledWith(3);
    act(() => comp.unmount());
  });

  const timing = [
    { start_time: 0, end_time: 3 },
    { start_time: 3, end_time: 6 },
    { start_time: 6, end_time: 9 },
    { start_time: 9, end_time: 12 },
  ];

  it("draws the replay playhead line when a position + timing are given", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart perTurn={perTurn} turnsTiming={timing} playheadSeconds={7} />,
      );
    });
    layout(comp);
    expect(
      comp.root.findAll((n) => n.props?.testID === "playhead-line").length,
    ).toBeGreaterThan(0);
    act(() => comp.unmount());
  });

  it("omits the playhead when there is no position", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart perTurn={perTurn} turnsTiming={timing} playheadSeconds={null} />,
      );
    });
    layout(comp);
    expect(comp.root.findAll((n) => n.props?.testID === "playhead-line")).toHaveLength(0);
    act(() => comp.unmount());
  });

  it("calls onSeekToTurn with the tapped turn's start_time", () => {
    const onSeekToTurn = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart
          perTurn={perTurn}
          turnsTiming={timing}
          onSeekToTurn={onSeekToTurn}
        />,
      );
    });
    layout(comp);
    // Tap the scrubber cell for turn 3 → seeks to its start_time (9).
    act(() => comp.root.find((n) => n.props?.testID === "scrub-3").props.onPress());
    expect(onSeekToTurn).toHaveBeenCalledWith(9);
    act(() => comp.unmount());
  });
});

// ------------------------------- Time axis --------------------------------

describe("timingIsUsable (mode gate)", () => {
  it("accepts finite, index-aligned, non-decreasing timing with a positive span", () => {
    expect(timingIsUsable(timedTiming, 4)).toBe(true);
  });
  it("rejects missing / empty / count-mismatched timing", () => {
    expect(timingIsUsable(undefined, 4)).toBe(false);
    expect(timingIsUsable([], 0)).toBe(false);
    expect(timingIsUsable(timedTiming, 3)).toBe(false); // fewer turns than timings
  });
  it("rejects end-before-start and an all-zero span (nothing to place)", () => {
    expect(timingIsUsable([{ start_time: 5, end_time: 4 }], 1)).toBe(false);
    expect(timingIsUsable([{ start_time: 0, end_time: 0 }], 1)).toBe(false);
  });
});

describe("durationForTiming", () => {
  it("prefers an explicit recording duration (trailing silence is real)", () => {
    expect(durationForTiming(timedTiming, 12)).toBe(12);
  });
  it("falls back to the last utterance end when no explicit duration", () => {
    expect(durationForTiming(timedTiming)).toBe(10);
    expect(durationForTiming(timedTiming, null)).toBe(10);
    expect(durationForTiming(timedTiming, 0)).toBe(10); // 0 isn't a real span
  });
});

describe("mapTurnsToDashes (time-axis geometry)", () => {
  const opts = { width: 300, height: 180, padding: 16, duration: 10 };

  it("groups by speaker, each carrying its own turns as dashes", () => {
    const lines = mapTurnsToDashes(perTurn, timedTiming, opts);
    expect(lines.map((l) => l.speaker)).toEqual(["Alice", "Bob"]);
    expect(lines[0].dashes.map((d) => d.index)).toEqual([0, 2]);
    expect(lines[1].dashes.map((d) => d.index)).toEqual([1, 3]);
  });

  it("spans each dash across its real [start,end] on the seconds→x scale", () => {
    const lines = mapTurnsToDashes(perTurn, timedTiming, opts);
    // xFor(sec) = 16 + (sec/10)*268.
    const aliceT0 = lines[0].dashes[0];
    expect(aliceT0.x1).toBeCloseTo(16); // starts at 0s → left padding
    expect(aliceT0.x2).toBeCloseTo(16 + (2 / 10) * 268);
    // Alice's long turn (3→9s) is visibly the widest dash — she spoke longest.
    const aliceT2 = lines[0].dashes[1];
    const bobT3 = lines[1].dashes[1];
    expect(aliceT2.x2 - aliceT2.x1).toBeGreaterThan(bobT3.x2 - bobT3.x1);
  });

  it("carries the spike flag and reflects heat in y (higher heat = smaller y)", () => {
    const lines = mapTurnsToDashes(perTurn, timedTiming, opts);
    expect(lines[1].dashes[1].isSpike).toBe(true); // Bob turn 3
    expect(lines[0].dashes[0].isSpike).toBe(false);
    // Alice heat 20 (turn 0) sits lower than Bob heat 90 (turn 3).
    expect(lines[0].dashes[0].y).toBeGreaterThan(lines[1].dashes[1].y);
  });

  it("grows a zero-length turn to a minimum visible dash width", () => {
    const lines = mapTurnsToDashes(
      [perTurn[0]],
      [{ start_time: 5, end_time: 5 }],
      { ...opts, minDashPx: 4 },
    );
    expect(lines[0].dashes[0].x2 - lines[0].dashes[0].x1).toBeCloseTo(4);
  });
});

describe("mapSimulatedToDashes (overlay geometry)", () => {
  const opts = { width: 300, height: 180, padding: 16, duration: 10 };
  const sim: SimulatedTurn[] = [
    { index: 2, speaker: "Alice", heat: 35 },
    { index: 3, speaker: "Bob", heat: 40 },
  ];

  it("reuses the real turn's time span so the dashed dash lands over the solid one", () => {
    const real = mapTurnsToDashes(perTurn, timedTiming, opts);
    const overlay = mapSimulatedToDashes(sim, timedTiming, opts);
    const realAliceT2 = real[0].dashes.find((d) => d.index === 2)!;
    const simAliceT2 = overlay[0].dashes.find((d) => d.index === 2)!;
    expect(simAliceT2.x1).toBeCloseTo(realAliceT2.x1);
    expect(simAliceT2.x2).toBeCloseTo(realAliceT2.x2);
  });

  it("skips a simulated turn that has no matching real span", () => {
    const overlay = mapSimulatedToDashes(
      [{ index: 99, speaker: "Alice", heat: 10 }],
      timedTiming,
      opts,
    );
    expect(overlay).toHaveLength(0);
  });
});

describe("playheadXForSeconds (time-axis alignment)", () => {
  const opts = { width: 300, padding: 16, duration: 10 };

  it("maps a playback position by absolute seconds", () => {
    expect(playheadXForSeconds(5, opts)).toBeCloseTo(16 + (5 / 10) * 268);
    expect(playheadXForSeconds(0, opts)).toBeCloseTo(16);
  });

  it("clamps out-of-range positions and rejects a non-positive duration", () => {
    expect(playheadXForSeconds(-3, opts)).toBeCloseTo(16); // before start
    expect(playheadXForSeconds(100, opts)).toBeCloseTo(16 + 268); // past end
    expect(playheadXForSeconds(5, { ...opts, duration: 0 })).toBeNull();
  });

  it("lands inside the dash of whoever is speaking at that instant", () => {
    // 6s falls inside Alice's turn 2 ([3,9]); the playhead must sit within that
    // dash's x-span — this is the whole point: pause and see who's talking.
    const real = mapTurnsToDashes(perTurn, timedTiming, {
      width: 300,
      height: 180,
      padding: 16,
      duration: 10,
    });
    const aliceT2 = real[0].dashes.find((d) => d.index === 2)!;
    const x = playheadXForSeconds(6, opts)!;
    expect(x).toBeGreaterThanOrEqual(aliceT2.x1);
    expect(x).toBeLessThanOrEqual(aliceT2.x2);
  });
});

describe("computeTalkShares (talk-time math)", () => {
  it("splits total talking time by speaker (silence attributed to no one)", () => {
    const shares = computeTalkShares(perTurn, timedTiming);
    expect(shares.Alice).toBeCloseTo(0.8);
    expect(shares.Bob).toBeCloseTo(0.2);
  });

  it("ignores gaps: the denominator is summed utterance length, not wall-clock", () => {
    // 1s + 1s of talk inside a 10s window → 50/50, NOT weighted by the silence.
    const shares = computeTalkShares(
      [perTurn[0], perTurn[1]],
      [
        { start_time: 0, end_time: 1 },
        { start_time: 9, end_time: 10 },
      ],
    );
    expect(shares.Alice).toBeCloseTo(0.5);
    expect(shares.Bob).toBeCloseTo(0.5);
  });

  it("returns a 0 share (never NaN) when there is no talking at all", () => {
    const shares = computeTalkShares(
      [perTurn[0]],
      [{ start_time: 4, end_time: 4 }],
    );
    expect(shares.Alice).toBe(0);
  });
});

describe("buildTimeScrubCells (tap strip layout)", () => {
  it("emits a turn cell per turn, weighted by real seconds, no gaps when contiguous", () => {
    const cells = buildTimeScrubCells(perTurn, timedTiming, 10);
    expect(cells.map((c) => c.kind)).toEqual(["turn", "turn", "turn", "turn"]);
    expect(cells.map((c) => c.seconds)).toEqual([2, 1, 6, 1]);
    expect(cells.map((c) => c.index)).toEqual([0, 1, 2, 3]);
  });

  it("inserts silent-gap spacers (leading, mid, trailing) so cells track the dashes", () => {
    const cells = buildTimeScrubCells(
      [perTurn[0], perTurn[1]],
      [
        { start_time: 1, end_time: 2 },
        { start_time: 4, end_time: 5 },
      ],
      6,
    );
    expect(cells.map((c) => c.kind)).toEqual([
      "gap", // 0→1 leading silence
      "turn", // turn 0
      "gap", // 2→4 mid silence
      "turn", // turn 1
      "gap", // 5→6 trailing silence
    ]);
    expect(cells.filter((c) => c.kind === "gap").map((c) => c.seconds)).toEqual([
      1, 2, 1,
    ]);
  });

  it("never produces a negative gap when turns overlap (diarized crosstalk)", () => {
    const cells = buildTimeScrubCells(
      [perTurn[0], perTurn[1]],
      [
        { start_time: 0, end_time: 5 },
        { start_time: 3, end_time: 6 }, // starts before the previous ended
      ],
      6,
    );
    expect(cells.filter((c) => c.kind === "gap")).toHaveLength(0);
    expect(cells.map((c) => c.seconds)).toEqual([5, 3]);
  });
});

describe("formatClock", () => {
  it("formats seconds as m:ss", () => {
    expect(formatClock(0)).toBe("0:00");
    expect(formatClock(5)).toBe("0:05");
    expect(formatClock(65)).toBe("1:05");
    expect(formatClock(600)).toBe("10:00");
  });
});

describe("HeatChart — time axis rendering", () => {
  it("draws one dash per turn plus a talk-time-weighted tap strip and time axis", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart
          perTurn={perTurn}
          turnsTiming={timedTiming}
          durationSeconds={10}
        />,
      );
    });
    layout(comp);
    // A dash per turn (the primary mark), no legacy points.
    expect(countIds(comp, "heat-dash-")).toBe(4);
    expect(countIds(comp, "heat-point-")).toBe(0);
    // Bob's turn 3 spike still gets an amber marker.
    expect(countIds(comp, "heat-spike-")).toBe(1);
    // Tap strip + tiny time axis are present.
    expect(comp.root.findAll((n) => n.props?.testID === "heat-scrubber").length)
      .toBeGreaterThan(0);
    expect(comp.root.findAll((n) => n.props?.testID === "heat-time-axis").length)
      .toBeGreaterThan(0);
    act(() => comp.unmount());
  });

  it("shows each speaker's talk-time share in the legend", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart perTurn={perTurn} turnsTiming={timedTiming} durationSeconds={10} />,
      );
    });
    layout(comp);
    const text = JSON.stringify(comp.toJSON());
    expect(text).toContain("80% of talking");
    expect(text).toContain("20% of talking");
    act(() => comp.unmount());
  });

  it("seeks to the tapped dash's real start_time (aligned tap strip)", () => {
    const onSeekToTurn = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart
          perTurn={perTurn}
          turnsTiming={timedTiming}
          durationSeconds={10}
          onSeekToTurn={onSeekToTurn}
        />,
      );
    });
    layout(comp);
    // Tap Alice's long turn 2 → seeks to its start (3s).
    act(() => comp.root.find((n) => n.props?.testID === "scrub-2").props.onPress());
    expect(onSeekToTurn).toHaveBeenCalledWith(3);
    act(() => comp.unmount());
  });

  it("falls back to the legacy evenly-spaced view with a note when timing is unusable", () => {
    // Timing is PROVIDED but invalid (end before start) → honest fallback, never
    // a fabricated timeline.
    const badTiming = [
      { start_time: 2, end_time: 1 },
      { start_time: 3, end_time: 2 },
      { start_time: 4, end_time: 3 },
      { start_time: 5, end_time: 4 },
    ];
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart perTurn={perTurn} turnsTiming={badTiming} />,
      );
    });
    layout(comp);
    // Legacy polylines + points, no dashes, and the subtle note is shown.
    expect(countIds(comp, "heat-line-")).toBe(2);
    expect(countIds(comp, "heat-dash-")).toBe(0);
    expect(comp.root.findAll((n) => n.props?.testID === "heat-legacy-note").length)
      .toBeGreaterThan(0);
    act(() => comp.unmount());
  });

  it("matches snapshot in time-axis mode", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart perTurn={perTurn} turnsTiming={timedTiming} durationSeconds={10} />,
      );
    });
    layout(comp);
    expect(comp.toJSON()).toMatchSnapshot();
    act(() => comp.unmount());
  });

  it("matches snapshot in legacy (no-timing) mode", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<HeatChart perTurn={perTurn} />);
    });
    layout(comp);
    expect(comp.toJSON()).toMatchSnapshot();
    act(() => comp.unmount());
  });
});

describe("HeatChart speaker display labels (§3)", () => {
  const speakerLabels = {
    Alice: { display_label: "Joe", label_source: "name" },
    Bob: { display_label: "Higher voice", label_source: "voice" },
  };

  it("renders display labels in the legend, not the raw ids", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart perTurn={perTurn} speakerLabels={speakerLabels} />,
      );
    });
    layout(comp);
    const text = JSON.stringify(comp.toJSON());
    expect(text).toContain("Joe");
    expect(text).toContain("Higher voice");
    act(() => comp.unmount());
  });

  it("falls back to raw speaker ids when no labels are provided (old recording)", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<HeatChart perTurn={perTurn} />);
    });
    layout(comp);
    const text = JSON.stringify(comp.toJSON());
    // The raw ids still render exactly as before — nothing regresses.
    expect(text).toContain("Alice");
    expect(text).toContain("Bob");
    act(() => comp.unmount());
  });

  it("shows the display label in the turn inspector when a turn is tapped", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart
          perTurn={perTurn}
          turns={[
            { speaker: "Alice", text: "hi" },
            { speaker: "Bob", text: "hey" },
            { speaker: "Alice", text: "ok" },
            { speaker: "Bob", text: "fine" },
          ]}
          turnsTiming={timedTiming}
          durationSeconds={10}
          speakerLabels={speakerLabels}
        />,
      );
    });
    layout(comp);
    // Tap turn 0 (Alice) via its scrub cell, then read the inspector.
    act(() =>
      comp.root.find((n) => n.props?.testID === "scrub-0").props.onPress(),
    );
    const inspector = comp.root.find(
      (n) => n.props?.testID === "turn-inspector",
    );
    // The inspector's speaker line shows the display label, not the raw id.
    const named = inspector.findAll(
      (n) => n.props?.children === "Joe",
    );
    expect(named.length).toBeGreaterThan(0);
    const rawIds = inspector.findAll((n) => n.props?.children === "Alice");
    expect(rawIds.length).toBe(0);
    act(() => comp.unmount());
  });

  it("uses display labels in the time-axis scrub cells' accessibility labels", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart
          perTurn={perTurn}
          turnsTiming={timedTiming}
          durationSeconds={10}
          speakerLabels={speakerLabels}
        />,
      );
    });
    layout(comp);
    const cell = comp.root.find((n) => n.props?.testID === "scrub-0");
    expect(cell.props.accessibilityLabel).toContain("Joe");
    expect(cell.props.accessibilityLabel).not.toContain("Alice");
    act(() => comp.unmount());
  });
});

// --------------------------- Y-axis label (§ owner bug 1) ---------------------------

describe("HeatChart y-axis label (owner report: unlabeled y-axis)", () => {
  it("renders a plain-language 'Heat' axis label near the chart", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<HeatChart perTurn={perTurn} />);
    });
    layout(comp);
    const label = comp.root.find((n) => n.props?.testID === "heat-axis-label");
    expect(label).toBeTruthy();
    // Same word the section title above this chart uses elsewhere
    // (ReplayScreen/DynamicsScreen: "Heat over the conversation") — not
    // invented terminology.
    const text = JSON.stringify(comp.toJSON());
    expect(text).toContain("Heat");
    act(() => comp.unmount());
  });

  it("shows the axis label on the time-axis chart too", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart perTurn={perTurn} turnsTiming={timedTiming} durationSeconds={10} />,
      );
    });
    layout(comp);
    expect(comp.root.find((n) => n.props?.testID === "heat-axis-label")).toBeTruthy();
    act(() => comp.unmount());
  });
});

// ------------------- Wide invisible tap targets (§ owner bug 2, hypothesis A) -------------------

describe("HeatChart wide invisible tap targets (owner report: tap-to-seek works ~1/20)", () => {
  it("renders one wide, near-transparent hit-target line per dash, at the same y and wider than any visible stroke", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart perTurn={perTurn} turnsTiming={timedTiming} durationSeconds={10} />,
      );
    });
    layout(comp);
    expect(countIds(comp, "heat-hit-")).toBe(4);

    const hit = comp.root.find((n) => n.props?.testID === "heat-hit-0");
    const visible = comp.root.find(
      (n) => n.props?.testID === "heat-dash-0" && typeof n.props.onPress === "function",
    );
    // Wider than even the "selected" visible stroke (7px) — a real touch target.
    expect(hit.props.strokeWidth).toBeGreaterThan(7);
    expect(hit.props.strokeWidth).toBeGreaterThan(visible.props.strokeWidth);
    // Effectively invisible (not exactly 0 — some SVG renderers skip hit-testing
    // a fully-transparent stroke).
    expect(hit.props.strokeOpacity).toBeGreaterThan(0);
    expect(hit.props.strokeOpacity).toBeLessThan(0.05);
    // Same geometry as the dash it backs, so it sits exactly over it.
    expect(hit.props.x1).toBeCloseTo(visible.props.x1);
    expect(hit.props.x2).toBeCloseTo(visible.props.x2);
    expect(hit.props.y1).toBeCloseTo(visible.props.y1);
    act(() => comp.unmount());
  });

  it("tapping the hit-target line selects the turn and seeks, exactly like the visible dash", () => {
    const onSeekToTurn = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart
          perTurn={perTurn}
          turnsTiming={timedTiming}
          durationSeconds={10}
          onSeekToTurn={onSeekToTurn}
        />,
      );
    });
    layout(comp);
    act(() =>
      comp.root.find((n) => n.props?.testID === "heat-hit-3").props.onPress(),
    );
    expect(onSeekToTurn).toHaveBeenCalledWith(9); // turn 3's start_time
    const inspector = comp.root.find((n) => n.props?.testID === "turn-inspector");
    expect(inspector).toBeTruthy();
    act(() => comp.unmount());
  });
});

// ----------------- Color/speaker-mismatch investigation (§ owner bug 3) -----------------

describe("mapTurnsToDashes — real fixture (color/speaker-mismatch investigation)", () => {
  // Exact turns from server/tests/fixtures/audio/test_recording_family_real_meta.json
  // — the owner's actual real recording ("Sage & asher 5 sec turns"). Includes
  // the BACK-TO-BACK zero-gap transition at index 6→7 (both at 25.7549995s)
  // that the owner's screenshot circled as a suspicious color flip. heat/
  // markers/is_spike aren't in the fixture (it's a diarization fixture, not an
  // analysis one) so those are filled with plausible placeholders — only
  // `speaker` + timing (the fixture's actual ground truth) drive this test.
  const realTurns: AnalyzePerTurn[] = [
    { index: 0, speaker: "Sage", heat: 20, markers: [], is_spike: false, trigger_phrase: null },
    { index: 1, speaker: "Asher", heat: 25, markers: [], is_spike: false, trigger_phrase: null },
    { index: 2, speaker: "Asher", heat: 30, markers: [], is_spike: false, trigger_phrase: null },
    { index: 3, speaker: "Sage", heat: 22, markers: [], is_spike: false, trigger_phrase: null },
    { index: 4, speaker: "Asher", heat: 35, markers: [], is_spike: false, trigger_phrase: null },
    { index: 5, speaker: "Asher", heat: 60, markers: [], is_spike: false, trigger_phrase: null },
    { index: 6, speaker: "Sage", heat: 40, markers: [], is_spike: false, trigger_phrase: null },
    { index: 7, speaker: "Asher", heat: 70, markers: [], is_spike: true, trigger_phrase: null },
  ];
  const realTiming: TurnTiming[] = [
    { start_time: 0.96, end_time: 6.08 },
    { start_time: 6.08, end_time: 8.82 },
    { start_time: 9.28, end_time: 10.0199995 },
    { start_time: 10.48, end_time: 15.125 },
    { start_time: 15.425, end_time: 15.925 },
    { start_time: 16.385, end_time: 19.045 },
    { start_time: 20.465, end_time: 25.7549995 },
    { start_time: 25.7549995, end_time: 29.125 },
  ];
  const opts = { width: 400, height: 180, padding: 16, duration: 29.125 };

  it("BUG CONFIRMED at the pure getSpeakerColor level: 'Sage' and 'Asher' hash to the identical color", () => {
    // Direct evidence: mapTurnsToDashes' DEFAULT color resolver is plain
    // getSpeakerColor (used whenever a caller doesn't pass colorOf, exactly
    // as every unit test above this one in the file does). Fed the owner's
    // own real fixture speakers, it assigns Sage and Asher THE SAME color —
    // indistinguishable from "the wrong speaker's color at a turn
    // transition", since the color never actually changes at the boundary.
    const lines = mapTurnsToDashes(realTurns, realTiming, opts);
    expect(lines.map((l) => l.speaker)).toEqual(["Sage", "Asher"]);
    expect(lines[0].color).toBe(lines[1].color); // <- the confirmed bug
    expect(lines[0].color).toBe("#8B5CF6");
  });

  it("FIXED: passing the component's colorOf (resolveSpeakerColors) gives every dash its own turn's speaker's color, distinctly, including the zero-gap 6→7 boundary", () => {
    // This mirrors exactly what HeatChart itself now does: derive the
    // conversation's speaker order from perTurn, resolve collision-free
    // colors once, and thread that resolver into mapTurnsToDashes via the new
    // optional `colorOf` option.
    const speakerOrder = ["Sage", "Asher"]; // first-appearance order in realTurns
    const colorOf = (s: string) => resolveSpeakerColors(speakerOrder).get(s)!;
    const lines = mapTurnsToDashes(realTurns, realTiming, { ...opts, colorOf });

    const sageColor = colorOf("Sage");
    const asherColor = colorOf("Asher");
    expect(sageColor).not.toBe(asherColor);

    const bySpeaker = new Map(lines.map((l) => [l.speaker, l]));
    const expectedSpeaker = ["Sage", "Asher", "Asher", "Sage", "Asher", "Asher", "Sage", "Asher"];
    const byIndex = new Map<number, string>(); // index -> resolved color
    for (const line of lines) {
      for (const d of line.dashes) byIndex.set(d.index, line.color);
    }
    expectedSpeaker.forEach((sp, i) => {
      expect(byIndex.get(i)).toBe(sp === "Sage" ? sageColor : asherColor);
    });
    expect(bySpeaker.get("Sage")!.dashes.map((d) => d.index)).toEqual([0, 3, 6]);
    expect(bySpeaker.get("Asher")!.dashes.map((d) => d.index)).toEqual([1, 2, 4, 5, 7]);

    // The specifically-circled transition: index 6 (Sage) ends exactly where
    // index 7 (Asher) begins — zero gap — yet the two dashes are correctly
    // and distinctly colored.
    expect(byIndex.get(6)).toBe(sageColor);
    expect(byIndex.get(7)).toBe(asherColor);
    expect(byIndex.get(6)).not.toBe(byIndex.get(7));
  });

  it("is unaffected by manual speaker naming — SpeakerNaming.tsx relabels display text only, never the color-keying raw id", () => {
    // mapTurnsToDashes/getSpeakerColor/resolveSpeakerColors only ever see the
    // RAW speaker id ("Sage"/"Asher"); a speakerLabels display map (produced
    // by the "Name the speakers" flow) is never passed into any of them —
    // it's consumed separately, only for the text shown in the
    // legend/inspector. Naming one speaker cannot recolor another's
    // already-rendered dashes.
    const colorOf = (s: string) => resolveSpeakerColors(["Sage", "Asher"]).get(s)!;
    const lines = mapTurnsToDashes(realTurns, realTiming, { ...opts, colorOf });
    expect(lines.map((l) => l.speaker)).toEqual(["Sage", "Asher"]);
    expect(lines[0].color).toBe(colorOf("Sage"));
    expect(lines[1].color).toBe(colorOf("Asher"));
  });

  it("INTEGRATION: the full HeatChart component renders Sage's and Asher's dashes in distinct colors (the actual owner-facing fix)", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <HeatChart
          perTurn={realTurns}
          turnsTiming={realTiming}
          durationSeconds={29.125}
        />,
      );
    });
    layout(comp, 400);

    const swatches = comp.root.findAll(
      (n) => typeof n.props?.testID === "string" && n.props.testID.startsWith("legend-swatch-"),
    );
    // style is [styles.swatch, { backgroundColor }] — an array, not a flat object.
    const swatchBg = (n: ReactTestInstance): string =>
      (n.props.style as { backgroundColor?: string }[]).find((s) => s?.backgroundColor)!
        .backgroundColor!;
    const colorBySwatchId = new Map(
      swatches.map((n) => [n.props.testID as string, swatchBg(n)]),
    );
    expect(colorBySwatchId.get("legend-swatch-Sage")).toBeTruthy();
    expect(colorBySwatchId.get("legend-swatch-Asher")).toBeTruthy();
    // Before the fix these were both "#8B5CF6" — the confirmed collision.
    expect(colorBySwatchId.get("legend-swatch-Sage")).not.toBe(
      colorBySwatchId.get("legend-swatch-Asher"),
    );

    // And the rendered dashes at the circled 6→7 boundary agree with the legend.
    const dash6 = comp.root.find((n) => n.props?.testID === "heat-dash-6");
    const dash7 = comp.root.find((n) => n.props?.testID === "heat-dash-7");
    expect(dash6.props.stroke).toBe(colorBySwatchId.get("legend-swatch-Sage"));
    expect(dash7.props.stroke).toBe(colorBySwatchId.get("legend-swatch-Asher"));
    expect(dash6.props.stroke).not.toBe(dash7.props.stroke);
    act(() => comp.unmount());
  });
});
