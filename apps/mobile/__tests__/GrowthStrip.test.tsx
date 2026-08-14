import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import GrowthStrip from "../src/components/GrowthStrip";
import { getGrowth } from "../src/api/client";
import type { GrowthPoint, GrowthResult } from "../src/api/client";

jest.mock("../src/api/client", () => ({
  getGrowth: jest.fn(),
}));
const mockGetGrowth = getGrowth as jest.Mock;

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

function textOf(node: ReactTestInstance): string {
  return node
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");
}

function pt(overrides: Partial<GrowthPoint> = {}): GrowthPoint {
  return {
    recording_id: "r1",
    timestamp: "2026-07-01T12:00:00+00:00",
    title: "A talk",
    my_score: 70,
    partner_names: [],
    ...overrides,
  };
}

function result(overrides: Partial<GrowthResult> = {}): GrowthResult {
  return {
    points: [],
    total_recordings: 0,
    identified_recordings: 0,
    ...overrides,
  };
}

async function render(onPress = jest.fn()) {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<GrowthStrip onPress={onPress} />);
  });
  return comp;
}

beforeEach(() => {
  mockGetGrowth.mockReset();
});

describe("GrowthStrip", () => {
  it("renders nothing when the fetch fails (no broken chart on home)", async () => {
    mockGetGrowth.mockRejectedValueOnce(new Error("API error: 503"));
    const comp = await render();
    expect(queryId(comp, "growth-strip")).toBeNull();
    act(() => comp.unmount());
  });

  it("shows an honest 'not tracked yet' row when nothing is identified", async () => {
    mockGetGrowth.mockResolvedValueOnce(result({ total_recordings: 3 }));
    const onPress = jest.fn();
    const comp = await render(onPress);
    const strip = queryId(comp, "growth-strip");
    expect(strip).toBeTruthy();
    expect(textOf(queryId(comp, "growth-strip-sub")!)).toContain(
      "not tracked yet",
    );
    act(() => strip!.props.onPress());
    expect(onPress).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });

  it("renders the identified count and score dots once measured", async () => {
    mockGetGrowth.mockResolvedValueOnce(
      result({
        points: [
          pt({ recording_id: "a", timestamp: "2026-07-01T00:00:00Z" }),
          pt({ recording_id: "b", timestamp: "2026-07-04T00:00:00Z" }),
        ],
        total_recordings: 5,
        identified_recordings: 2,
      }),
    );
    const comp = await render();
    expect(textOf(queryId(comp, "growth-strip-sub")!)).toContain(
      "2 of 5 recordings",
    );
    // The chart area measures itself before the SVG draws — simulate layout.
    const layoutNode = comp.root.findAll(
      (n) => typeof n.props?.onLayout === "function",
    )[0];
    act(() =>
      layoutNode.props.onLayout({ nativeEvent: { layout: { width: 180 } } }),
    );
    expect(queryId(comp, "growth-dot-a")).toBeTruthy();
    expect(queryId(comp, "growth-dot-b")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("null scores never become dots — gaps, not zeros", async () => {
    mockGetGrowth.mockResolvedValueOnce(
      result({
        points: [
          pt({ recording_id: "a", timestamp: "2026-07-01T00:00:00Z" }),
          pt({
            recording_id: "gap",
            timestamp: "2026-07-02T00:00:00Z",
            my_score: null,
          }),
          pt({ recording_id: "b", timestamp: "2026-07-04T00:00:00Z" }),
        ],
        total_recordings: 3,
        identified_recordings: 3,
      }),
    );
    const comp = await render();
    const layoutNode = comp.root.findAll(
      (n) => typeof n.props?.onLayout === "function",
    )[0];
    act(() =>
      layoutNode.props.onLayout({ nativeEvent: { layout: { width: 180 } } }),
    );
    expect(queryId(comp, "growth-dot-a")).toBeTruthy();
    expect(queryId(comp, "growth-dot-gap")).toBeNull();
    act(() => comp.unmount());
  });
});
