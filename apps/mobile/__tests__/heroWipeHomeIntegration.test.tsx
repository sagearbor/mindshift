import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import HomeScreen from "../src/screens/HomeScreen";
import LoginScreen from "../src/screens/LoginScreen";
import { getGrowth } from "../src/api/client";
import { useAuthStore } from "../src/store/authStore";

/**
 * Screen integration test for Task P3-4b, default (native) jest environment.
 * Covers BOTH web placements the owner asked for: home (the two-mode
 * surface) and login (the sign-in landing).
 *
 * HeroWipe is now a genuine platform-file split (HeroWipe.web.tsx /
 * HeroWipe.native.tsx — see HeroWipe.native.tsx's doc comment for why a
 * runtime `Platform.OS` check inside one shared file wasn't enough: it kept
 * the web implementation's imports — and its ~2MB of hero JPEGs — bundled
 * for every platform regardless). That means flipping `Platform.OS` at
 * runtime (an earlier version of this test) no longer proves anything:
 * module resolution for a bare `../components/HeroWipe` import happens once,
 * at import time, driven by jest's haste platform config (default: native),
 * not by a mutable Platform.OS property read afterward. So this file proves
 * the NATIVE side of the contract two ways per screen: (1) the screen
 * renders no "hero-wipe" node under the default jest resolution, and (2) the
 * hero image manifest (and therefore its six JPEGs) was never even
 * require()'d. HeroWipe.web.test.tsx covers the actual web implementation
 * directly, LoginScreen.web.test.tsx covers login's web layout ordering, and
 * heroWipeSchedule.test.tsx / heroWipeEffects.test.ts cover the pure logic —
 * none of those need a platform-resolved import at all.
 */
jest.mock("../src/api/client", () => ({
  getGrowth: jest.fn(),
}));
const mockGetGrowth = getGrowth as jest.Mock;

function queryAll(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance[] {
  return comp.root.findAll((n) => n.props?.testID === id);
}

function makeHandlers() {
  return {
    onNavigate: jest.fn(),
    onOpenYourDay: jest.fn(),
  };
}

beforeEach(() => {
  mockGetGrowth.mockReset();
  mockGetGrowth.mockRejectedValue(new Error("API error: 503"));
});

describe("HeroWipe on the home screen (default/native jest resolution)", () => {
  it("renders nothing for HeroWipe", async () => {
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<HomeScreen {...makeHandlers()} />);
    });
    expect(queryAll(comp, "hero-wipe").length).toBe(0);
    act(() => comp.unmount());
  });

  it("never resolves/requires the hero image manifest — pins the bundle-exclusion contract", async () => {
    // If HeroWipe.web.tsx (and therefore heroImages.ts and its six JPEGs)
    // had been reached, `require.cache` would contain its resolved path.
    // Resolving the path alone doesn't execute or cache the module, so this
    // assertion is only satisfied by NEVER having imported it.
    const heroImagesPath = require.resolve("../src/assets/heroImages");
    const heroWipeWebPath = require.resolve("../src/components/HeroWipe.web");

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<HomeScreen {...makeHandlers()} />);
    });

    expect(require.cache[heroImagesPath]).toBeUndefined();
    expect(require.cache[heroWipeWebPath]).toBeUndefined();
    act(() => comp.unmount());
  });
});

describe("HeroWipe on the login screen (default/native jest resolution)", () => {
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

  it("renders nothing for HeroWipe", async () => {
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<LoginScreen />);
    });
    expect(queryAll(comp, "hero-wipe").length).toBe(0);
    // The rest of the sign-in form is unaffected by the hero-banner slot.
    expect(queryAll(comp, "login-screen").length).toBeGreaterThan(0);
    expect(queryAll(comp, "email-input").length).toBeGreaterThan(0);
    act(() => comp.unmount());
  });

  it("never resolves/requires the hero image manifest — pins the bundle-exclusion contract", async () => {
    const heroImagesPath = require.resolve("../src/assets/heroImages");
    const heroWipeWebPath = require.resolve("../src/components/HeroWipe.web");

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<LoginScreen />);
    });

    expect(require.cache[heroImagesPath]).toBeUndefined();
    expect(require.cache[heroWipeWebPath]).toBeUndefined();
    act(() => comp.unmount());
  });
});
