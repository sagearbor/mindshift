import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";

import HomeDesignScreen from "../src/screens/HomeDesignScreen";
import {
  useLayoutStore,
  DEFAULT_TAB_SLOTS,
  DEFAULT_HOME_BOXES,
  TAB_SLOT_CAP,
  HOME_BOX_CAP,
} from "../src/store/layoutStore";
import { PRIMARY_ELIGIBLE_DESTINATIONS } from "../src/nav/destinations";

/** First node carrying the given testID, or null. */
function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

/** Every node whose testID starts with `prefix`, deduped (a TouchableOpacity's
 *  testID shows up on more than one node in the render tree). */
function queryAllIds(
  comp: renderer.ReactTestRenderer,
  prefix: string,
): ReactTestInstance[] {
  const all = comp.root.findAll(
    (n) =>
      typeof n.props?.testID === "string" && n.props.testID.startsWith(prefix),
  );
  const seen = new Set<string>();
  const out: ReactTestInstance[] = [];
  for (const n of all) {
    const id = n.props.testID as string;
    if (seen.has(id)) continue;
    seen.add(id);
    out.push(n);
  }
  return out;
}

/** Concatenated string content rendered under a node (its Text leaves). */
function textOf(node: ReactTestInstance): string {
  return node
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");
}

function resetStoreState() {
  useLayoutStore.setState({
    tabSlots: [...DEFAULT_TAB_SLOTS],
    homeBoxes: [...DEFAULT_HOME_BOXES],
    hydrated: true,
  });
}

function render(onBack = jest.fn()) {
  let comp!: renderer.ReactTestRenderer;
  act(() => {
    comp = renderer.create(<HomeDesignScreen onBack={onBack} />);
  });
  return { comp, onBack };
}

beforeEach(() => {
  resetStoreState();
});

describe("HomeDesignScreen — rendering per state", () => {
  it("renders the default tab slots and home boxes with icon + title", () => {
    const { comp } = render();
    for (const id of DEFAULT_TAB_SLOTS) {
      const row = queryId(comp, `home-design-tab-item-${id}`);
      expect(row).toBeTruthy();
    }
    for (const id of DEFAULT_HOME_BOXES) {
      const row = queryId(comp, `home-design-box-item-${id}`);
      expect(row).toBeTruthy();
    }
    act(() => comp.unmount());
  });

  it("shows an honest empty hint when a section has zero slots", () => {
    act(() => {
      useLayoutStore.getState().setTabSlots([]);
      useLayoutStore.getState().setHomeBoxes([]);
    });
    const { comp } = render();
    expect(queryId(comp, "home-design-tab-empty")).toBeTruthy();
    expect(queryId(comp, "home-design-box-empty")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("back button calls onBack", () => {
    const { comp, onBack } = render();
    act(() => queryId(comp, "home-design-back")!.props.onPress());
    expect(onBack).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });
});

describe("HomeDesignScreen — remove", () => {
  it("removing a tab slot updates the real store", () => {
    const { comp } = render();
    act(() => queryId(comp, "home-design-tab-remove-coach")!.props.onPress());
    expect(useLayoutStore.getState().tabSlots).toEqual(["analyze", "growth"]);
    expect(queryId(comp, "home-design-tab-item-coach")).toBeNull();
    act(() => comp.unmount());
  });

  it("removing a home box updates the real store", () => {
    const { comp } = render();
    act(() =>
      queryId(comp, "home-design-box-remove-recordings")!.props.onPress(),
    );
    expect(useLayoutStore.getState().homeBoxes).toEqual(["growth"]);
    act(() => comp.unmount());
  });
});

describe("HomeDesignScreen — add", () => {
  it("lists remaining primary-eligible destinations not already in the tab bar", () => {
    const { comp } = render();
    // Defaults are coach/analyze/growth — recordings and (2026-08-19
    // primary-eligible-expand) therapistDashboard are the remaining
    // primary-eligible destinations.
    expect(queryId(comp, "home-design-tab-add-recordings")).toBeTruthy();
    expect(queryId(comp, "home-design-tab-add-therapistDashboard")).toBeTruthy();
    expect(queryId(comp, "home-design-tab-add-coach")).toBeNull();
    act(() => comp.unmount());
  });

  // 2026-08-19 primary-eligible-expand: this is the exact editor behavior
  // the owner asked about — "shouldn't something like the Therapist
  // Dashboard also be an option there?" — now it is.
  it("offers Dashboard as an addable tab/box option now that it's primary-eligible", () => {
    const { comp } = render();
    expect(queryId(comp, "home-design-tab-add-therapistDashboard")).toBeTruthy();
    expect(queryId(comp, "home-design-box-add-therapistDashboard")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("tapping an add row appends that destination to the store", () => {
    const { comp } = render();
    act(() =>
      queryId(comp, "home-design-tab-add-recordings")!.props.onPress(),
    );
    expect(useLayoutStore.getState().tabSlots).toEqual([
      "coach",
      "analyze",
      "growth",
      "recordings",
    ]);
    act(() => comp.unmount());
  });

  it("home boxes add list offers destinations not already in homeBoxes", () => {
    const { comp } = render();
    // Defaults are recordings/growth — coach, analyze, and (2026-08-19
    // primary-eligible-expand) therapistDashboard remain offerable.
    expect(queryId(comp, "home-design-box-add-coach")).toBeTruthy();
    expect(queryId(comp, "home-design-box-add-analyze")).toBeTruthy();
    expect(queryId(comp, "home-design-box-add-therapistDashboard")).toBeTruthy();
    expect(queryId(comp, "home-design-box-add-recordings")).toBeNull();
    act(() => comp.unmount());
  });
});

describe("HomeDesignScreen — reorder", () => {
  it("moving the second tab up swaps it with the first", () => {
    const { comp } = render();
    act(() => queryId(comp, "home-design-tab-up-analyze")!.props.onPress());
    expect(useLayoutStore.getState().tabSlots).toEqual([
      "analyze",
      "coach",
      "growth",
    ]);
    act(() => comp.unmount());
  });

  it("moving the second tab down swaps it with the third", () => {
    const { comp } = render();
    act(() => queryId(comp, "home-design-tab-down-analyze")!.props.onPress());
    expect(useLayoutStore.getState().tabSlots).toEqual([
      "coach",
      "growth",
      "analyze",
    ]);
    act(() => comp.unmount());
  });

  it("moving the first item up is a no-op (already at the top)", () => {
    const { comp } = render();
    act(() => queryId(comp, "home-design-tab-up-coach")!.props.onPress());
    expect(useLayoutStore.getState().tabSlots).toEqual([
      "coach",
      "analyze",
      "growth",
    ]);
    act(() => comp.unmount());
  });

  it("moving the last item down is a no-op (already at the bottom)", () => {
    const { comp } = render();
    act(() => queryId(comp, "home-design-tab-down-growth")!.props.onPress());
    expect(useLayoutStore.getState().tabSlots).toEqual([
      "coach",
      "analyze",
      "growth",
    ]);
    act(() => comp.unmount());
  });
});

describe("HomeDesignScreen — caps", () => {
  // 2026-08-19 primary-eligible-expand: the registry now has exactly 5
  // primary-eligible destinations (coach, analyze, recordings, growth,
  // therapistDashboard — see nav/destinations.ts) against the 5-slot tab
  // cap, so placing every eligible destination now ALSO hits the literal
  // numeric cap — the two "nothing left to add" reasons coincide instead of
  // being independently reachable the way they were pre-2026-08-19 (4
  // eligible destinations against a 5-slot cap, so this test's "every
  // destination placed" case was strictly BELOW the cap). Both reasons hide
  // the add list behind the same hint either way (see the box cap test
  // below for a case where the cap is hit before all destinations are
  // placed, since the box section's cap is only 4).
  it("hides the add list once every primary-eligible destination is already on the tab bar", () => {
    act(() => {
      useLayoutStore
        .getState()
        .setTabSlots(PRIMARY_ELIGIBLE_DESTINATIONS.map((d) => d.id));
    });
    const { comp } = render();
    expect(useLayoutStore.getState().tabSlots.length).toBe(
      PRIMARY_ELIGIBLE_DESTINATIONS.length,
    );
    expect(useLayoutStore.getState().tabSlots.length).toBe(TAB_SLOT_CAP);
    expect(queryAllIds(comp, "home-design-tab-add-").length).toBe(0);
    expect(queryId(comp, "home-design-tab-cap-hint")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("hides the add list and shows a full hint once home boxes hit their cap", () => {
    act(() => {
      useLayoutStore
        .getState()
        .setHomeBoxes(
          PRIMARY_ELIGIBLE_DESTINATIONS.slice(0, HOME_BOX_CAP).map((d) => d.id),
        );
    });
    const { comp } = render();
    expect(useLayoutStore.getState().homeBoxes.length).toBe(HOME_BOX_CAP);
    expect(queryAllIds(comp, "home-design-box-add-").length).toBe(0);
    expect(queryId(comp, "home-design-box-cap-hint")).toBeTruthy();
    act(() => comp.unmount());
  });
});

describe("HomeDesignScreen — live preview", () => {
  it("mirrors the current tab slots, in order", () => {
    const { comp } = render();
    const preview = queryId(comp, "home-design-preview")!;
    expect(textOf(preview)).toBe("Live CoachAnalyze a ConversationYour Growth");
    act(() => comp.unmount());
  });

  it("updates live when a tab is removed", () => {
    const { comp } = render();
    act(() => queryId(comp, "home-design-tab-remove-coach")!.props.onPress());
    expect(queryId(comp, "home-design-preview-tab-coach")).toBeNull();
    expect(queryId(comp, "home-design-preview-tab-analyze")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("shows an honest empty state when the tab bar has zero slots", () => {
    act(() => useLayoutStore.getState().setTabSlots([]));
    const { comp } = render();
    expect(queryId(comp, "home-design-preview-empty")).toBeTruthy();
    act(() => comp.unmount());
  });
});

describe("HomeDesignScreen — reset to defaults", () => {
  it("restores both lists immediately, no confirmation dialog", () => {
    act(() => {
      useLayoutStore.getState().setTabSlots(["growth"]);
      useLayoutStore.getState().setHomeBoxes([]);
    });
    const { comp } = render();
    act(() => queryId(comp, "home-design-reset")!.props.onPress());
    expect(useLayoutStore.getState().tabSlots).toEqual([
      "coach",
      "analyze",
      "growth",
    ]);
    expect(useLayoutStore.getState().homeBoxes).toEqual([
      "recordings",
      "growth",
    ]);
    // Reflected in the UI immediately, same render pass.
    expect(queryId(comp, "home-design-tab-item-coach")).toBeTruthy();
    act(() => comp.unmount());
  });
});
