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
}: {
  screen: Screen;
  setScreen: (s: Screen) => void;
}) {
  useAndroidBackHandler(screen, setScreen);
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
});
