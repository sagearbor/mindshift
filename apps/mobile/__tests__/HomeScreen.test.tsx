import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import HomeScreen from "../src/screens/HomeScreen";
import { useLayoutStore, DEFAULT_HOME_BOXES } from "../src/store/layoutStore";
import { getGrowth } from "../src/api/client";

// The growth box's mini preview self-fetches (Task N4, useGrowthPreview) —
// keep the home tests deterministic.
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
    onNavigate: jest.fn(),
    onOpenYourDay: jest.fn(),
  };
}

async function render(overrides: Partial<React.ComponentProps<typeof HomeScreen>> = {}) {
  const handlers = { ...makeHandlers(), ...overrides };
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<HomeScreen {...handlers} />);
  });
  return { comp, handlers };
}

beforeEach(() => {
  mockGetGrowth.mockReset();
  // Default: growth unavailable → the growth box's mini preview renders
  // nothing extra, just the icon + title (no broken chart).
  mockGetGrowth.mockRejectedValue(new Error("API error: 503"));
  useLayoutStore.setState({ tabSlots: [], homeBoxes: [], hydrated: true });
});

describe("HomeScreen — box grid (Task N4)", () => {
  it("0 boxes: an honest hint, never a blank/broken-looking screen", async () => {
    useLayoutStore.setState({ homeBoxes: [] });
    const { comp } = await render();
    expect(queryId(comp, "home-boxes-empty")).toBeTruthy();
    expect(queryId(comp, "home-boxes-grid")).toBeNull();
    act(() => comp.unmount());
  });

  it("1 box: a single full-width banner card", async () => {
    useLayoutStore.setState({ homeBoxes: ["coach"] });
    const { comp } = await render();
    expect(queryId(comp, "home-box-coach")).toBeTruthy();
    expect(queryId(comp, "home-box-coach")!.props.style).toContainEqual(
      expect.objectContaining({ width: "100%" }),
    );
    act(() => comp.unmount());
  });

  it("2 boxes: two half-width cards", async () => {
    useLayoutStore.setState({ homeBoxes: ["coach", "analyze"] });
    const { comp } = await render();
    expect(queryId(comp, "home-box-coach")).toBeTruthy();
    expect(queryId(comp, "home-box-analyze")).toBeTruthy();
    expect(queryId(comp, "home-box-coach")!.props.style).toContainEqual(
      expect.objectContaining({ width: "48%" }),
    );
    act(() => comp.unmount());
  });

  it("3 boxes: a wrapping 2-column grid (2 + 1)", async () => {
    useLayoutStore.setState({ homeBoxes: ["coach", "analyze", "recordings"] });
    const { comp } = await render();
    expect(queryId(comp, "home-box-coach")).toBeTruthy();
    expect(queryId(comp, "home-box-analyze")).toBeTruthy();
    expect(queryId(comp, "home-box-recordings")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("4 boxes (the cap): a full 2x2 grid", async () => {
    useLayoutStore.setState({
      homeBoxes: ["coach", "analyze", "recordings", "growth"],
    });
    const { comp } = await render();
    for (const id of ["coach", "analyze", "recordings", "growth"]) {
      expect(queryId(comp, `home-box-${id}`)).toBeTruthy();
    }
    act(() => comp.unmount());
  });

  it("defaults (recordings + growth): both render as boxes", async () => {
    useLayoutStore.setState({ homeBoxes: [...DEFAULT_HOME_BOXES] });
    const { comp } = await render();
    expect(queryId(comp, "home-box-recordings")).toBeTruthy();
    expect(queryId(comp, "home-box-growth")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("tapping a box hands its destination's Screen straight to onNavigate — the same mechanism the chrome tabs use", async () => {
    useLayoutStore.setState({ homeBoxes: ["coach", "recordings"] });
    const { comp, handlers } = await render();

    act(() => queryId(comp, "home-box-coach")!.props.onPress());
    expect(handlers.onNavigate).toHaveBeenCalledWith({ name: "live-coach" });

    act(() => queryId(comp, "home-box-recordings")!.props.onPress());
    expect(handlers.onNavigate).toHaveBeenCalledWith({
      name: "recordings",
      returnTo: "home",
    });
    act(() => comp.unmount());
  });

  it("a stale/unknown persisted box id is dropped silently, not crashed on", async () => {
    // sanitizeSlots already drops these before they reach the store in real
    // usage; this proves HomeBoxGrid itself is defensive too.
    useLayoutStore.setState({
      homeBoxes: ["coach", "not-a-real-destination" as never],
    });
    const { comp } = await render();
    expect(queryId(comp, "home-box-coach")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("the growth box shows a mini trend preview when tracked data exists", async () => {
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
    useLayoutStore.setState({ homeBoxes: ["growth"] });
    const { comp } = await render();
    // Growth box still has its icon + title header regardless (owner rule:
    // every box is icon + label, never a bare block) — proven implicitly by
    // home-box-growth existing — plus the preview once data resolves.
    await act(async () => {
      await Promise.resolve();
    });
    expect(queryId(comp, "home-box-growth")).toBeTruthy();
    expect(queryId(comp, "home-box-growth-preview")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("the growth box has no preview (just icon + label) when the fetch fails", async () => {
    useLayoutStore.setState({ homeBoxes: ["growth"] });
    const { comp } = await render();
    expect(queryId(comp, "home-box-growth")).toBeTruthy();
    expect(queryId(comp, "home-box-growth-preview")).toBeNull();
    act(() => comp.unmount());
  });
});

describe("HomeScreen — content preserved across the N4 rework", () => {
  it("preserves the hero header and the 'Your day' link (no registry destination of its own)", async () => {
    useLayoutStore.setState({ homeBoxes: [] });
    const { comp, handlers } = await render();
    const link = queryId(comp, "home-your-day-link");
    expect(link).toBeTruthy();
    act(() => link!.props.onPress());
    expect(handlers.onOpenYourDay).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });
});
