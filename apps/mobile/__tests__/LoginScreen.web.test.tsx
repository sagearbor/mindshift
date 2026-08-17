import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import LoginScreen from "../src/screens/LoginScreen";
import { useAuthStore } from "../src/store/authStore";

/**
 * Task P3-4b, owner decision: hero goes on BOTH web placements. This file
 * proves LoginScreen's own composition — that it places the hero ABOVE the
 * title block, in normal document flow (never overlapping it) — independent
 * of which platform file `../components/HeroWipe` actually resolves to in
 * this jest run (default: native, see heroWipeHomeIntegration.test.tsx).
 * `../src/components/HeroWipe` is mocked here to stand in for its real
 * web implementation (HeroWipe.web.tsx, whose "hero-wipe" testID and
 * behavior are pinned directly in HeroWipe.web.test.tsx) purely so this
 * file can observe render ORDER without needing web platform resolution.
 */
jest.mock("../src/components/HeroWipe", () => {
  const ReactActual = require("react");
  const { View: RNView } = require("react-native");
  return {
    __esModule: true,
    default: () => ReactActual.createElement(RNView, { testID: "hero-wipe" }),
  };
});

function flatten(root: ReactTestInstance): ReactTestInstance[] {
  const out: ReactTestInstance[] = [root];
  for (const child of root.children) {
    if (child && typeof child !== "string") {
      out.push(...flatten(child as ReactTestInstance));
    }
  }
  return out;
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

describe("LoginScreen (web hero placement)", () => {
  it("renders the hero banner above the MindShift title block, never overlapping it", async () => {
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<LoginScreen />);
    });

    const flat = flatten(comp.root);
    const heroIndex = flat.findIndex((n) => n.props?.testID === "hero-wipe");
    const brandIndex = flat.findIndex((n) => n.props?.children === "MindShift");
    const subtitleIndex = flat.findIndex(
      (n) => n.props?.children === "Sign in to continue",
    );

    expect(heroIndex).toBeGreaterThan(-1);
    expect(brandIndex).toBeGreaterThan(-1);
    expect(subtitleIndex).toBeGreaterThan(-1);
    // Pre-order tree traversal order matches column-flex visual order here
    // (no reordering styles are in play) — hero comes first, then the title
    // block, in normal flow. Not absolutely positioned, so this ordering
    // guarantee is also a non-overlap guarantee: each takes its own space.
    expect(heroIndex).toBeLessThan(brandIndex);
    expect(brandIndex).toBeLessThan(subtitleIndex);

    act(() => comp.unmount());
  });

  it("bounds the hero banner slot with an explicit max height (sensible on large screens)", async () => {
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<LoginScreen />);
    });

    const slot = comp.root.findAll(
      (n) => n.props?.testID === "hero-banner-slot",
    )[0];
    expect(slot).toBeTruthy();
    const style = slot.props.style;
    const flatStyle = Array.isArray(style) ? Object.assign({}, ...style) : style;
    expect(flatStyle.maxHeight).toBeDefined();
    expect(typeof flatStyle.maxHeight).toBe("number");
    expect(flatStyle.maxHeight).toBeLessThanOrEqual(320);

    act(() => comp.unmount());
  });
});
