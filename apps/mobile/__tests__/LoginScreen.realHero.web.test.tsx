import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import LoginScreen from "../src/screens/LoginScreen";
import { useAuthStore } from "../src/store/authStore";

/**
 * Task P3-4b review round 3. The lightweight-mock test in
 * LoginScreen.web.test.tsx proved sibling ORDER but never real rendered
 * geometry — a real visual verification (owner/coordinator screenshotting an
 * actual `expo start --web` export) caught a genuine overlap the mock-based
 * test could not: on a short/constrained viewport, the form's `container`
 * View (`flex:1, justifyContent:'center'`, `overflow: visible`) held content
 * taller than its own box, and centering bled the excess equally above AND
 * below that box — pushing the "MindShift" title up and over the hero
 * banner's lower edge. `react-test-renderer` never runs a real CSS layout
 * engine, so no jest assertion — however this test is written — can detect
 * that class of bug directly; it was found and the fix (LoginScreen.tsx:
 * the form container is now a `ScrollView`, whose content can grow past the
 * visible area and scroll instead of bleeding outside its own box) was
 * verified against a live `expo start --web` page at the exact window size
 * that reproduced it (see the fix-round-3 report for those measurements —
 * `window.innerHeight`, `getBoundingClientRect()` on both the hero slot and
 * the title, before and after the fix).
 *
 * What THIS file honestly can and does pin at the jest level, rendering the
 * REAL `HeroWipe.web` implementation (via the same `../components/HeroWipe`
 * bare import LoginScreen uses in production, not a lightweight stub):
 * 1. The hero root's declared style owns an explicit, self-contained height
 *    and `overflow: hidden` — necessary (not sufficient) for it to never
 *    leak content into a sibling, and something a stub can't prove since a
 *    stub doesn't carry HeroWipe.web's actual stylesheet.
 * 2. The title text is not merely LATER than the hero in render order (the
 *    weaker claim LoginScreen.web.test.tsx's mock-based test makes) but
 *    structurally OUTSIDE the hero's own subtree — ruling out the title
 *    being nested inside the hero and relying on some absolute-position/
 *    z-index trick to appear "below" it visually.
 */
jest.mock("../src/components/HeroWipe", () => require("../src/components/HeroWipe.web"));

function flatten(root: ReactTestInstance): ReactTestInstance[] {
  const out: ReactTestInstance[] = [root];
  for (const child of root.children) {
    if (child && typeof child !== "string") {
      out.push(...flatten(child as ReactTestInstance));
    }
  }
  return out;
}

function isDescendantOf(node: ReactTestInstance, ancestor: ReactTestInstance): boolean {
  let cur: ReactTestInstance | null = node.parent;
  while (cur) {
    if (cur === ancestor) return true;
    cur = cur.parent;
  }
  return false;
}

beforeEach(() => {
  useAuthStore.setState({
    user: null,
    initializing: false,
    error: null,
    notice: null,
    busy: false,
    pendingGoogleCredential: null,
    pendingGoogleEmail: null,
  });
});

describe("LoginScreen with the REAL HeroWipe.web implementation", () => {
  it("the hero root declares an explicit, self-contained height and overflow:hidden", async () => {
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<LoginScreen />);
    });

    const hero = comp.root.findAll((n) => n.props?.testID === "hero-wipe")[0];
    expect(hero).toBeTruthy();
    const style = hero.props.style;
    const flatStyle = Array.isArray(style) ? Object.assign({}, ...style) : style;
    expect(flatStyle.height).toBe(200);
    expect(flatStyle.overflow).toBe("hidden");

    act(() => comp.unmount());
  });

  it("renders the title text structurally OUTSIDE the hero's subtree, not just later in order", async () => {
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<LoginScreen />);
    });

    const hero = comp.root.findAll((n) => n.props?.testID === "hero-wipe")[0];
    const brand = flatten(comp.root).find((n) => n.props?.children === "MindShift");
    expect(hero).toBeTruthy();
    expect(brand).toBeTruthy();
    expect(isDescendantOf(brand!, hero)).toBe(false);

    act(() => comp.unmount());
  });
});
