import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import HeatChart, { heatRangeBand } from "../src/components/HeatChart";
import type { AnalyzePerTurn } from "../src/api/client";

function turn(index: number, heat: number): AnalyzePerTurn {
  return { index, speaker: index % 2 ? "Bob" : "Alice", heat, markers: [], is_spike: false, trigger_phrase: null };
}

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

describe("heatRangeBand", () => {
  it("annotates a narrow calm conversation (the flat-line problem)", () => {
    const band = heatRangeBand([turn(0, 12), turn(1, 18), turn(2, 15)]);
    expect(band).not.toBeNull();
    expect(band!.minHeat).toBe(12);
    expect(band!.maxHeat).toBe(18);
    expect(band!.label).toContain("calm range");
  });

  it("names the mid range for a narrow strained band", () => {
    const band = heatRangeBand([turn(0, 45), turn(1, 50), turn(2, 48)]);
    expect(band!.label).toContain("mid range");
  });

  it("names the high range for a narrow hot band", () => {
    const band = heatRangeBand([turn(0, 80), turn(1, 88), turn(2, 84)]);
    expect(band!.label).toContain("high range");
  });

  it("returns null for a wide spread (the line already tells the story)", () => {
    expect(heatRangeBand([turn(0, 10), turn(1, 90)])).toBeNull();
  });

  it("returns null with fewer than two turns", () => {
    expect(heatRangeBand([turn(0, 20)])).toBeNull();
    expect(heatRangeBand([])).toBeNull();
  });
});

describe("HeatChart narrow-range annotation", () => {
  it("shows the band caption for a narrow, calm conversation", () => {
    const perTurn = [turn(0, 12), turn(1, 16), turn(2, 14)];
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<HeatChart perTurn={perTurn} />);
    });
    // Give the chart a measured width so the SVG band renders too.
    const layoutNode = comp.root.findAll(
      (n) => typeof n.props?.onLayout === "function",
    )[0];
    act(() => {
      layoutNode.props.onLayout({ nativeEvent: { layout: { width: 300, height: 180 } } });
    });
    expect(queryId(comp, "heat-range-band-note")).toBeTruthy();
    expect(queryId(comp, "heat-range-band")).toBeTruthy();
    expect(JSON.stringify(comp.toJSON())).toContain("calm range");
    act(() => comp.unmount());
  });

  it("omits the band for a wide-spread conversation", () => {
    const perTurn = [turn(0, 10), turn(1, 40), turn(2, 92)];
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<HeatChart perTurn={perTurn} />);
    });
    expect(queryId(comp, "heat-range-band-note")).toBeNull();
    act(() => comp.unmount());
  });
});
