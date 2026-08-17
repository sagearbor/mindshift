import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import HomeScreen from "../src/screens/HomeScreen";
import { getGrowth } from "../src/api/client";

/**
 * Screen integration test for Task P3-4b, default (native) jest environment.
 *
 * HeroWipe is now a genuine platform-file split (HeroWipe.web.tsx /
 * HeroWipe.native.tsx — see HeroWipe.native.tsx's doc comment for why a
 * runtime `Platform.OS` check inside one shared file wasn't enough: it kept
 * the web implementation's imports — and its ~2MB of hero JPEGs — bundled
 * for every platform regardless). That means flipping `Platform.OS` at
 * runtime (the previous version of this test) no longer proves anything:
 * module resolution for a bare `../components/HeroWipe` import happens once,
 * at import time, driven by jest's haste platform config (default: native),
 * not by a mutable Platform.OS property read afterward. So this file proves
 * the NATIVE side of the contract two ways: (1) HomeScreen renders no
 * "hero-wipe" node under the default jest resolution, and (2) the hero image
 * manifest (and therefore its six JPEGs) was never even require()'d.
 * HeroWipe.web.test.tsx covers the actual web implementation directly, and
 * heroWipeSchedule.test.tsx / heroWipeEffects.test.ts cover the pure logic —
 * neither of those needs a platform-resolved import at all.
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
    onLiveCoach: jest.fn(),
    onAnalyze: jest.fn(),
    onOpenRecordings: jest.fn(),
    onOpenYourDay: jest.fn(),
    onOpenAdvanced: jest.fn(),
    onOpenGrowth: jest.fn(),
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
