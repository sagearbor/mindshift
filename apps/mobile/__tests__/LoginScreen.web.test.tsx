import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import LoginScreen from "../src/screens/LoginScreen";
import { useAuthStore } from "../src/store/authStore";

/**
 * Task P3-4b, owner decision: hero goes on BOTH web placements. This file
 * proves LoginScreen's composition — that it places the hero ABOVE the
 * title block in render order — independent of which platform file
 * `../components/HeroWipe` actually resolves to in this jest run (default:
 * native, see heroWipeHomeIntegration.test.tsx). `../src/components/HeroWipe`
 * is mocked here to a lightweight stub purely so this file can observe
 * order without needing web platform resolution.
 *
 * Review round 3 correction: an earlier version of this file claimed render
 * order alone proved "never overlapping" — a real `expo start --web`
 * screenshot caught an actual overlap this file's mock-based check could
 * not (a flexbox overflow-bleed bug; react-test-renderer never runs real
 * CSS layout, so no assertion here can prove non-overlap). See
 * LoginScreen.realHero.web.test.tsx for what IS honestly checkable at the
 * jest level with the REAL HeroWipe.web implementation, and the fix-round-3
 * report for the live-browser geometry verification that actually confirms
 * the fix.
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
  it("renders the hero banner before the MindShift title block in render order", async () => {
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
    // Order only, NOT a non-overlap proof (see this file's header comment
    // and LoginScreen.realHero.web.test.tsx / the fix-round-3 report for
    // what actually verifies no-overlap).
    expect(heroIndex).toBeLessThan(brandIndex);
    expect(brandIndex).toBeLessThan(subtitleIndex);

    act(() => comp.unmount());
  });

  it("shows a disabled 'Continue with Apple' placeholder on web too, honestly labeled coming soon", async () => {
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<LoginScreen />);
    });

    const appleButton = comp.root.findAll(
      (n) => n.props?.testID === "apple-button",
    )[0];
    expect(appleButton).toBeTruthy();
    expect(appleButton.props.disabled).toBe(true);
    expect(appleButton.props.accessibilityState).toEqual({ disabled: true });
    expect(appleButton.props.onPress).toBeUndefined();

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
