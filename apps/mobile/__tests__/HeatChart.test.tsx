import React from "react";
import renderer, { act } from "react-test-renderer";
import HeatChart, {
  mapTurnsToLines,
  mapSimulatedToLines,
  playheadIndexForTime,
  playheadXForTime,
} from "../src/components/HeatChart";
import type { AnalyzePerTurn, SimulatedTurn } from "../src/api/client";

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
