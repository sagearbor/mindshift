import React from "react";
import renderer, { act } from "react-test-renderer";
import { Platform } from "react-native";
import { useAndroidBackHandler } from "../src/nav/useAndroidBackHandler";
import type { Screen } from "../App";

// The wiring itself (Platform gate, listener registration, ToastAndroid) —
// the underlying decisions (backTarget/shouldExitOnBack) are unit-tested on
// their own, framework-free, in backHandler.test.ts. Mocking the whole
// "react-native" package (rather than BackHandler/ToastAndroid's individual
// internal paths) sidesteps react-native/index.js's `.default` getter
// plumbing entirely — the hook imports BackHandler/ToastAndroid directly
// from "react-native", so this mock is a straight substitution.
const mockAddEventListener = jest.fn();
const mockRemove = jest.fn();
const mockToastShow = jest.fn();

jest.mock("react-native", () => {
  // Only pick out the one real export the test needs (Platform, so
  // `Platform.OS = "android"` below is honored by the hook) rather than
  // spreading the whole `jest.requireActual("react-native")` module: that
  // module's exports are lazy getters (BackHandler, FlatList, DevMenu, ...),
  // and a spread reads every one of them eagerly, which pulls in native
  // modules (DevMenu) that don't exist outside a real device/simulator and
  // crash the whole test file.
  const actual = jest.requireActual("react-native");
  return {
    __esModule: true,
    Platform: actual.Platform,
    BackHandler: {
      addEventListener: (...args: unknown[]) => {
        mockAddEventListener(...args);
        return { remove: mockRemove };
      },
    },
    ToastAndroid: { show: (...args: unknown[]) => mockToastShow(...args), SHORT: 0 },
  };
});

function Harness({
  screen,
  setScreen,
  closeOverlays,
}: {
  screen: Screen;
  setScreen: (s: Screen) => void;
  closeOverlays?: () => boolean;
}) {
  useAndroidBackHandler(screen, setScreen, closeOverlays);
  return null;
}

function pressedHandler(): () => boolean {
  const call = mockAddEventListener.mock.calls.find(
    (c) => c[0] === "hardwareBackPress",
  );
  if (!call) throw new Error("hardwareBackPress was never registered");
  return call[1];
}

describe("useAndroidBackHandler", () => {
  const originalOS = Platform.OS;

  beforeEach(() => {
    mockAddEventListener.mockClear();
    mockRemove.mockClear();
    mockToastShow.mockClear();
  });

  afterEach(() => {
    Platform.OS = originalOS;
  });

  it("registers nothing off Android (iOS/web no-op)", () => {
    Platform.OS = "ios";
    const setScreen = jest.fn();
    act(() => {
      renderer.create(<Harness screen={{ name: "home" }} setScreen={setScreen} />);
    });
    expect(mockAddEventListener).not.toHaveBeenCalled();
  });

  it("on Android, a pushed screen pops to its back target and swallows the press", () => {
    Platform.OS = "android";
    const setScreen = jest.fn();
    act(() => {
      renderer.create(
        <Harness screen={{ name: "live-coach" }} setScreen={setScreen} />,
      );
    });
    const handled = pressedHandler()();
    expect(setScreen).toHaveBeenCalledWith({ name: "home" });
    expect(handled).toBe(true);
  });

  it("on Android, the first home press shows the exit hint and doesn't exit", () => {
    Platform.OS = "android";
    const setScreen = jest.fn();
    act(() => {
      renderer.create(<Harness screen={{ name: "home" }} setScreen={setScreen} />);
    });
    const handled = pressedHandler()();
    expect(mockToastShow).toHaveBeenCalledWith("Press back again to exit", 0);
    expect(handled).toBe(true);
    expect(setScreen).not.toHaveBeenCalled();
  });

  it("on Android, a second home press within the window lets the system exit", () => {
    Platform.OS = "android";
    const setScreen = jest.fn();
    act(() => {
      renderer.create(<Harness screen={{ name: "home" }} setScreen={setScreen} />);
    });
    const handler = pressedHandler();
    expect(handler()).toBe(true); // first press: hint only
    expect(handler()).toBe(false); // second press: let the OS exit
  });

  it("cleans up its listener on unmount", () => {
    Platform.OS = "android";
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <Harness screen={{ name: "home" }} setScreen={jest.fn()} />,
      );
    });
    act(() => comp.unmount());
    expect(mockRemove).toHaveBeenCalledTimes(1);
  });

  // Task N3 fix round 1, CRITICAL 1: an open overlay (hamburger catalog /
  // account menu) must be dismissed by back BEFORE any navigation or
  // double-back-to-exit decision — otherwise back could act right
  // underneath it (including exiting the app with a menu still open).
  describe("closeOverlays (CRITICAL 1 fix)", () => {
    it("back closes an open overlay instead of navigating, on a pushed screen", () => {
      Platform.OS = "android";
      const setScreen = jest.fn();
      const closeOverlays = jest.fn(() => true); // an overlay WAS open and got closed
      act(() => {
        renderer.create(
          <Harness
            screen={{ name: "live-coach" }}
            setScreen={setScreen}
            closeOverlays={closeOverlays}
          />,
        );
      });
      const handled = pressedHandler()();
      expect(closeOverlays).toHaveBeenCalledTimes(1);
      expect(setScreen).not.toHaveBeenCalled(); // no navigation underneath it
      expect(handled).toBe(true);
    });

    it("back closes an open overlay instead of exiting, on Home", () => {
      Platform.OS = "android";
      const setScreen = jest.fn();
      const closeOverlays = jest.fn(() => true);
      act(() => {
        renderer.create(
          <Harness
            screen={{ name: "home" }}
            setScreen={setScreen}
            closeOverlays={closeOverlays}
          />,
        );
      });
      const handled = pressedHandler()();
      expect(closeOverlays).toHaveBeenCalledTimes(1);
      expect(mockToastShow).not.toHaveBeenCalled(); // no exit-hint either
      expect(handled).toBe(true); // never lets the app exit underneath a menu
    });

    it("with overlays closed, back behaves exactly as before (falls through to navigation)", () => {
      Platform.OS = "android";
      const setScreen = jest.fn();
      const closeOverlays = jest.fn(() => false); // nothing was open
      act(() => {
        renderer.create(
          <Harness
            screen={{ name: "analyze" }}
            setScreen={setScreen}
            closeOverlays={closeOverlays}
          />,
        );
      });
      const handled = pressedHandler()();
      expect(closeOverlays).toHaveBeenCalledTimes(1);
      expect(setScreen).toHaveBeenCalledWith({ name: "home" });
      expect(handled).toBe(true);
    });

    it("omitting closeOverlays entirely behaves exactly as before (back-compat)", () => {
      Platform.OS = "android";
      const setScreen = jest.fn();
      act(() => {
        renderer.create(
          <Harness screen={{ name: "growth" }} setScreen={setScreen} />,
        );
      });
      const handled = pressedHandler()();
      expect(setScreen).toHaveBeenCalledWith({ name: "home" });
      expect(handled).toBe(true);
    });
  });

  // MINOR fix: a first press on Home shouldn't count toward a later,
  // unrelated press after navigating away and back — only two CONSECUTIVE
  // presses while staying on Home should exit.
  it("resets the exit-window timer when navigating away from Home", () => {
    Platform.OS = "android";
    const setScreen = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<Harness screen={{ name: "home" }} setScreen={setScreen} />);
    });
    // First press on Home: hint only, sets the pending-exit timer.
    expect(pressedHandler()()).toBe(true);
    expect(mockToastShow).toHaveBeenCalledTimes(1);

    // Navigate away from Home (e.g. the user tapped Live Coach) — the
    // pending timer should clear, not silently carry over.
    act(() => {
      comp.update(<Harness screen={{ name: "live-coach" }} setScreen={setScreen} />);
    });
    // ...and back to Home.
    act(() => {
      comp.update(<Harness screen={{ name: "home" }} setScreen={setScreen} />);
    });

    mockToastShow.mockClear();
    // This press must be treated as a FIRST press again (hint, not exit) —
    // if the old timer had leaked through, this would incorrectly exit.
    const handled = pressedHandler()();
    expect(mockToastShow).toHaveBeenCalledTimes(1);
    expect(handled).toBe(true);
  });
});
