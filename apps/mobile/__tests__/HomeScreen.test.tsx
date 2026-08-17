import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import HomeScreen from "../src/screens/HomeScreen";
import { getGrowth } from "../src/api/client";

// The embedded GrowthStrip self-fetches; keep the home tests deterministic.
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

function makeHandlers() {
  return {
    onLiveCoach: jest.fn(),
    onAnalyze: jest.fn(),
    onOpenRecordings: jest.fn(),
    onOpenYourDay: jest.fn(),
    onOpenGrowth: jest.fn(),
  };
}

beforeEach(() => {
  // Default: growth unavailable → the strip renders nothing and home is
  // exactly the two-mode surface it always was.
  mockGetGrowth.mockReset();
  mockGetGrowth.mockRejectedValue(new Error("API error: 503"));
});

describe("HomeScreen", () => {
  it("renders the two primary modes and the history entry (no corner affordance — that's AppChrome's job now)", async () => {
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<HomeScreen {...makeHandlers()} />);
    });
    expect(queryId(comp, "home-live-coach")).toBeTruthy();
    expect(queryId(comp, "home-analyze")).toBeTruthy();
    expect(queryId(comp, "home-recordings-link")).toBeTruthy();
    expect(queryId(comp, "home-your-day-link")).toBeTruthy();
    // Task N3: the wordmark + Settings "⋯" corner moved into AppChrome — no
    // longer HomeScreen's own affordance.
    expect(queryId(comp, "home-advanced-button")).toBeNull();
    // Growth unavailable → no strip, no broken chart.
    expect(queryId(comp, "growth-strip")).toBeNull();
    expect(comp.toJSON()).toMatchSnapshot();
    act(() => comp.unmount());
  });

  it("each tap target calls exactly its own handler", async () => {
    const handlers = makeHandlers();
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<HomeScreen {...handlers} />);
    });

    act(() => queryId(comp, "home-live-coach")!.props.onPress());
    expect(handlers.onLiveCoach).toHaveBeenCalledTimes(1);

    act(() => queryId(comp, "home-analyze")!.props.onPress());
    expect(handlers.onAnalyze).toHaveBeenCalledTimes(1);

    act(() => queryId(comp, "home-recordings-link")!.props.onPress());
    expect(handlers.onOpenRecordings).toHaveBeenCalledTimes(1);

    act(() => queryId(comp, "home-your-day-link")!.props.onPress());
    expect(handlers.onOpenYourDay).toHaveBeenCalledTimes(1);

    // No cross-talk.
    expect(handlers.onLiveCoach).toHaveBeenCalledTimes(1);
    expect(handlers.onAnalyze).toHaveBeenCalledTimes(1);
    expect(handlers.onOpenRecordings).toHaveBeenCalledTimes(1);
    expect(handlers.onOpenYourDay).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });

  it("shows the growth strip between the mode cards and the history row, wired to onOpenGrowth", async () => {
    mockGetGrowth.mockResolvedValue({
      points: [
        {
          recording_id: "r1",
          timestamp: "2026-07-01T12:00:00Z",
          title: "A talk",
          my_score: 70,
          partner_names: [],
        },
      ],
      total_recordings: 2,
      identified_recordings: 1,
    });
    const handlers = makeHandlers();
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<HomeScreen {...handlers} />);
    });
    const strip = queryId(comp, "growth-strip");
    expect(strip).toBeTruthy();
    act(() => strip!.props.onPress());
    expect(handlers.onOpenGrowth).toHaveBeenCalledTimes(1);
    // The strip never hijacks the primary modes.
    expect(handlers.onAnalyze).not.toHaveBeenCalled();
    act(() => comp.unmount());
  });
});
