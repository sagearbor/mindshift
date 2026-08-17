import React from "react";
import { Text } from "react-native";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import AppChrome, { type AppChromeHandle } from "../src/components/AppChrome";
import { useLayoutStore, DEFAULT_TAB_SLOTS } from "../src/store/layoutStore";
import { DESTINATIONS } from "../src/nav/destinations";

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

/** Nodes whose testID starts with `prefix`, deduped to one entry per testID
 *  (a TouchableOpacity's testID prop shows up on more than one node in the
 *  render tree — the composite instance and the host node it forwards to —
 *  so a naive findAll over-counts each logical button several times over). */
function queryAllIds(
  comp: renderer.ReactTestRenderer,
  prefix: string,
): ReactTestInstance[] {
  const all = comp.root.findAll(
    (n) => typeof n.props?.testID === "string" && n.props.testID.startsWith(prefix),
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

function makeProps(
  overrides: Partial<React.ComponentProps<typeof AppChrome>> = {},
): React.ComponentProps<typeof AppChrome> {
  return {
    screenName: "home",
    onNavigate: jest.fn(),
    onGoHome: jest.fn(),
    onSignOut: jest.fn(),
    user: { email: "sophie@example.com", displayName: "Sophie" },
    children: <Text testID="chrome-child">content</Text>,
    ...overrides,
  };
}

beforeEach(() => {
  // Fresh, hydrated-looking default layout before every test — mocking the
  // store the same way the rest of this suite mocks other zustand stores
  // (App.test.tsx sets useSessionStore/useAnalyzeStore state directly).
  useLayoutStore.setState({
    tabSlots: [...DEFAULT_TAB_SLOTS],
    homeBoxes: [],
    hydrated: true,
  });
});

describe("AppChrome — top bar", () => {
  it("renders the hamburger, wordmark, avatar, and the wrapped children", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...makeProps()} />);
    });
    expect(queryId(comp, "chrome-hamburger-button")).toBeTruthy();
    expect(queryId(comp, "chrome-wordmark")).toBeTruthy();
    expect(queryId(comp, "chrome-avatar-button")).toBeTruthy();
    expect(queryId(comp, "chrome-child")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("wordmark tap calls onGoHome — Home has no registry entry of its own", () => {
    const props = makeProps();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...props} />);
    });
    act(() => queryId(comp, "chrome-wordmark")!.props.onPress());
    expect(props.onGoHome).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });

  it("shows the account's initial in the avatar slot (no photo set)", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...makeProps()} />);
    });
    expect(queryId(comp, "chrome-avatar-initial")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("shows the photo in the avatar slot once avatarUri is set (Task N6)", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AppChrome
          {...makeProps({ avatarUri: "file:///doc/avatar/profile.jpg" })}
        />,
      );
    });
    expect(queryId(comp, "chrome-avatar-photo")).toBeTruthy();
    expect(queryId(comp, "chrome-avatar-initial")).toBeNull();
    act(() => comp.unmount());
  });
});

describe("AppChrome — hamburger catalog", () => {
  it("opens the full catalog listing EVERY registry destination", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...makeProps()} />);
    });
    expect(queryId(comp, "chrome-catalog")).toBeNull();

    act(() => queryId(comp, "chrome-hamburger-button")!.props.onPress());

    expect(queryId(comp, "chrome-catalog")).toBeTruthy();
    for (const dest of DESTINATIONS) {
      const row = queryId(comp, `chrome-catalog-item-${dest.id}`);
      expect(row).toBeTruthy();
    }
    // Always complete regardless of tab customization: the catalog isn't
    // limited to the (3-slot, by default) tab bar.
    expect(DESTINATIONS.length).toBeGreaterThan(DEFAULT_TAB_SLOTS.length);
    act(() => comp.unmount());
  });

  it("tapping a catalog item navigates to its screen and closes the catalog", () => {
    const props = makeProps();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...props} />);
    });
    act(() => queryId(comp, "chrome-hamburger-button")!.props.onPress());
    act(() => queryId(comp, "chrome-catalog-item-tutorial")!.props.onPress());

    expect(props.onNavigate).toHaveBeenCalledWith({ name: "onboarding" });
    expect(queryId(comp, "chrome-catalog")).toBeNull();
    act(() => comp.unmount());
  });

  it("the close button dismisses the catalog without navigating", () => {
    const props = makeProps();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...props} />);
    });
    act(() => queryId(comp, "chrome-hamburger-button")!.props.onPress());
    act(() => queryId(comp, "chrome-catalog-close")!.props.onPress());

    expect(queryId(comp, "chrome-catalog")).toBeNull();
    expect(props.onNavigate).not.toHaveBeenCalled();
    act(() => comp.unmount());
  });
});

describe("AppChrome — avatar account menu", () => {
  it("opens showing the signed-in account, Settings, and Log out (profile-photo changes live in Settings only)", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...makeProps()} />);
    });
    act(() => queryId(comp, "chrome-avatar-button")!.props.onPress());

    expect(queryId(comp, "chrome-account-menu")).toBeTruthy();
    const email = queryId(comp, "chrome-account-email")!;
    expect(JSON.stringify(email.props.children)).toContain("sophie@example.com");
    expect(queryId(comp, "chrome-account-photo")).toBeNull();
    expect(queryId(comp, "chrome-account-settings")).toBeTruthy();
    expect(queryId(comp, "chrome-account-sign-out")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("Settings navigates to the advanced screen and closes the menu", () => {
    const props = makeProps();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...props} />);
    });
    act(() => queryId(comp, "chrome-avatar-button")!.props.onPress());
    act(() => queryId(comp, "chrome-account-settings")!.props.onPress());

    expect(props.onNavigate).toHaveBeenCalledWith({ name: "advanced" });
    expect(queryId(comp, "chrome-account-menu")).toBeNull();
    act(() => comp.unmount());
  });

  it("Log out calls the real signOut wiring and closes the menu", () => {
    const props = makeProps();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...props} />);
    });
    act(() => queryId(comp, "chrome-avatar-button")!.props.onPress());
    act(() => queryId(comp, "chrome-account-sign-out")!.props.onPress());

    expect(props.onSignOut).toHaveBeenCalledTimes(1);
    expect(props.onNavigate).not.toHaveBeenCalled();
    expect(queryId(comp, "chrome-account-menu")).toBeNull();
    act(() => comp.unmount());
  });

  it("tapping the backdrop closes the menu without side effects", () => {
    const props = makeProps();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...props} />);
    });
    act(() => queryId(comp, "chrome-avatar-button")!.props.onPress());
    act(() => queryId(comp, "chrome-account-backdrop")!.props.onPress());

    expect(queryId(comp, "chrome-account-menu")).toBeNull();
    expect(props.onNavigate).not.toHaveBeenCalled();
    expect(props.onSignOut).not.toHaveBeenCalled();
    act(() => comp.unmount());
  });
});

describe("AppChrome — bottom tab bar", () => {
  it("renders exactly layoutStore.tabSlots, in order", () => {
    act(() => {
      useLayoutStore.setState({ tabSlots: ["coach", "growth"], homeBoxes: [] });
    });
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...makeProps()} />);
    });
    const tabs = queryAllIds(comp, "chrome-tab-");
    // Excludes the tab-bar container itself ("chrome-tab-bar" also matches
    // the prefix "chrome-tab-") — filter to the two real slot buttons.
    const slotTabs = tabs.filter((t) => t.props.testID !== "chrome-tab-bar");
    expect(slotTabs.map((t) => t.props.testID)).toEqual([
      "chrome-tab-coach",
      "chrome-tab-growth",
    ]);
    act(() => comp.unmount());
  });

  it("is hidden entirely when there are zero tab slots", () => {
    act(() => {
      useLayoutStore.setState({ tabSlots: [], homeBoxes: [] });
    });
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...makeProps()} />);
    });
    expect(queryId(comp, "chrome-tab-bar")).toBeNull();
    act(() => comp.unmount());
  });

  it("tapping a tab navigates to its destination's screen", () => {
    act(() => {
      useLayoutStore.setState({ tabSlots: ["recordings"], homeBoxes: [] });
    });
    const props = makeProps();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...props} />);
    });
    act(() => queryId(comp, "chrome-tab-recordings")!.props.onPress());
    expect(props.onNavigate).toHaveBeenCalledWith({
      name: "recordings",
      returnTo: "home",
    });
    act(() => comp.unmount());
  });

  it("marks the tab matching the current screen as selected", () => {
    act(() => {
      useLayoutStore.setState({ tabSlots: ["coach", "analyze"], homeBoxes: [] });
    });
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...makeProps({ screenName: "analyze" })} />);
    });
    const coachTab = queryId(comp, "chrome-tab-coach")!;
    const analyzeTab = queryId(comp, "chrome-tab-analyze")!;
    expect(coachTab.props.accessibilityState).toEqual({ selected: false });
    expect(analyzeTab.props.accessibilityState).toEqual({ selected: true });
    act(() => comp.unmount());
  });
});

describe("AppChrome — closeOverlays imperative handle (fix round 1, CRITICAL 1)", () => {
  it("closes the open catalog and reports true", () => {
    const ref = React.createRef<AppChromeHandle>();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome ref={ref} {...makeProps()} />);
    });
    act(() => queryId(comp, "chrome-hamburger-button")!.props.onPress());
    expect(queryId(comp, "chrome-catalog")).toBeTruthy();

    let closed = false;
    act(() => {
      closed = ref.current!.closeOverlays();
    });
    expect(closed).toBe(true);
    expect(queryId(comp, "chrome-catalog")).toBeNull();
    act(() => comp.unmount());
  });

  it("closes the open account menu and reports true", () => {
    const ref = React.createRef<AppChromeHandle>();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome ref={ref} {...makeProps()} />);
    });
    act(() => queryId(comp, "chrome-avatar-button")!.props.onPress());
    expect(queryId(comp, "chrome-account-menu")).toBeTruthy();

    let closed = false;
    act(() => {
      closed = ref.current!.closeOverlays();
    });
    expect(closed).toBe(true);
    expect(queryId(comp, "chrome-account-menu")).toBeNull();
    act(() => comp.unmount());
  });

  it("reports false and does nothing when no overlay is open", () => {
    const ref = React.createRef<AppChromeHandle>();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome ref={ref} {...makeProps()} />);
    });
    let closed = true;
    act(() => {
      closed = ref.current!.closeOverlays();
    });
    expect(closed).toBe(false);
    act(() => comp.unmount());
  });
});

describe("AppChrome — accessibility while an overlay is open (MINOR fix)", () => {
  it("marks the background content no-hide-descendants when the catalog is open, auto otherwise", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...makeProps()} />);
    });
    const content = queryId(comp, "chrome-content")!;
    expect(content.props.importantForAccessibility).toBe("auto");

    act(() => queryId(comp, "chrome-hamburger-button")!.props.onPress());
    expect(content.props.importantForAccessibility).toBe("no-hide-descendants");
    act(() => comp.unmount());
  });

  it("the catalog and account menu declare themselves modal to assistive tech", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<AppChrome {...makeProps()} />);
    });
    act(() => queryId(comp, "chrome-hamburger-button")!.props.onPress());
    expect(queryId(comp, "chrome-catalog")!.props.accessibilityViewIsModal).toBe(
      true,
    );
    act(() => queryId(comp, "chrome-catalog-close")!.props.onPress());

    act(() => queryId(comp, "chrome-avatar-button")!.props.onPress());
    expect(
      queryId(comp, "chrome-account-menu")!.props.accessibilityViewIsModal,
    ).toBe(true);
    act(() => comp.unmount());
  });
});
