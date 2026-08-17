import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import { Platform } from "react-native";
import HomeScreen from "../src/screens/HomeScreen";
import { getGrowth } from "../src/api/client";

/**
 * Screen integration test for Task P3-4b: the web home screen renders
 * HeroWipe; native renders nothing for it (mobile is deferred per the plan).
 * HeroWipe itself branches on Platform.OS at the top of the component, so
 * this drives it the same way useAudioStreamWeb.test.tsx drives the web
 * audio-capture branch — overriding the (otherwise read-only) Platform.OS
 * data property.
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

const originalOS = Platform.OS;

function setPlatform(os: string) {
  Object.defineProperty(Platform, "OS", { value: os, configurable: true });
}

beforeEach(() => {
  mockGetGrowth.mockReset();
  mockGetGrowth.mockRejectedValue(new Error("API error: 503"));
});

afterEach(() => {
  setPlatform(originalOS);
});

describe("HeroWipe on the home screen", () => {
  it("renders on web", async () => {
    setPlatform("web");
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<HomeScreen {...makeHandlers()} />);
    });
    // react-test-renderer matches testID on both the composite View and its
    // host node, so "present" is ">0", not "exactly 1" (see queryId's
    // single-match convention in HomeScreen.test.tsx, which sidesteps this
    // by only ever reading [0]).
    expect(queryAll(comp, "hero-wipe").length).toBeGreaterThan(0);
    act(() => comp.unmount());
  });

  it("renders nothing on iOS", async () => {
    setPlatform("ios");
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<HomeScreen {...makeHandlers()} />);
    });
    expect(queryAll(comp, "hero-wipe").length).toBe(0);
    act(() => comp.unmount());
  });

  it("renders nothing on Android", async () => {
    setPlatform("android");
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<HomeScreen {...makeHandlers()} />);
    });
    expect(queryAll(comp, "hero-wipe").length).toBe(0);
    act(() => comp.unmount());
  });

  it("on web, never intercepts the primary mode taps (pointerEvents none)", async () => {
    setPlatform("web");
    const handlers = makeHandlers();
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<HomeScreen {...handlers} />);
    });
    const hero = queryAll(comp, "hero-wipe")[0];
    expect(hero.props.pointerEvents).toBe("none");
    act(() => queryAll(comp, "home-live-coach")[0].props.onPress());
    expect(handlers.onLiveCoach).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });
});
