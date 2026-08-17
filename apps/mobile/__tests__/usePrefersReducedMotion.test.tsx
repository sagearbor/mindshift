import React from "react";
import renderer, { act } from "react-test-renderer";
import { usePrefersReducedMotion } from "../src/hooks/usePrefersReducedMotion";

/**
 * Task P3-4b review round 1: usePrefersReducedMotion had zero coverage.
 * Covers matchMedia-absent (default RN/jest env — jest-expo's setup
 * polyfills `window` as an alias to `global`, but that bare Node global has
 * no `matchMedia`), a matchMedia present/absent read, and both the modern
 * addEventListener and legacy Safari addListener subscription branches.
 *
 * Uses a plain react-test-renderer harness (not
 * @testing-library/react-native's async `renderHook`) — see
 * heroWipeSchedule.test.tsx for why: no timers/async settling needed here,
 * so the simpler harness keeps this file dependency-light.
 */
function renderHook(): { current: () => boolean; unmount: () => void } {
  let latest = false;
  function Harness() {
    latest = usePrefersReducedMotion();
    return null;
  }
  let root!: renderer.ReactTestRenderer;
  act(() => {
    root = renderer.create(<Harness />);
  });
  return {
    current: () => latest,
    unmount: () => act(() => root.unmount()),
  };
}

type MockMediaQueryList = {
  matches: boolean;
  addEventListener?: jest.Mock;
  removeEventListener?: jest.Mock;
  addListener?: jest.Mock;
  removeListener?: jest.Mock;
};

describe("usePrefersReducedMotion", () => {
  const originalWindow = (global as Record<string, unknown>).window;

  afterEach(() => {
    if (originalWindow === undefined) {
      delete (global as Record<string, unknown>).window;
    } else {
      (global as Record<string, unknown>).window = originalWindow;
    }
  });

  it("defaults to false when matchMedia doesn't exist (native, this test env)", () => {
    // Sanity check on the premise itself: jest-expo's setup polyfills
    // `window` as an alias to `global` (RN >=0.45 convention), but that bare
    // Node global has no `matchMedia` — the actual condition the hook guards.
    expect(
      typeof (global as Record<string, unknown>).window,
    ).toBe("object");
    expect(
      typeof (window as unknown as Record<string, unknown>).matchMedia,
    ).not.toBe("function");

    const hook = renderHook();
    expect(hook.current()).toBe(false);
    hook.unmount();
  });

  it("reads true from matchMedia when the browser reports reduced motion", () => {
    const mql: MockMediaQueryList = {
      matches: true,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
    };
    const matchMedia = jest.fn(() => mql);
    (global as Record<string, unknown>).window = { matchMedia };

    const hook = renderHook();
    expect(hook.current()).toBe(true);
    expect(matchMedia).toHaveBeenCalledWith("(prefers-reduced-motion: reduce)");
    hook.unmount();
  });

  it("reads false from matchMedia when the browser reports no preference", () => {
    const mql: MockMediaQueryList = {
      matches: false,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
    };
    (global as Record<string, unknown>).window = { matchMedia: jest.fn(() => mql) };

    const hook = renderHook();
    expect(hook.current()).toBe(false);
    hook.unmount();
  });

  it("subscribes via the modern addEventListener API and updates live on change", () => {
    let onChange: (() => void) | undefined;
    const mql: MockMediaQueryList = {
      matches: false,
      addEventListener: jest.fn((event: string, cb: () => void) => {
        if (event === "change") onChange = cb;
      }),
      removeEventListener: jest.fn(),
    };
    (global as Record<string, unknown>).window = { matchMedia: jest.fn(() => mql) };

    const hook = renderHook();
    expect(hook.current()).toBe(false);
    expect(mql.addEventListener).toHaveBeenCalledWith("change", expect.any(Function));

    mql.matches = true;
    act(() => onChange?.());
    expect(hook.current()).toBe(true);

    hook.unmount();
    expect(mql.removeEventListener).toHaveBeenCalledWith("change", expect.any(Function));
  });

  it("falls back to the legacy addListener/removeListener API when addEventListener is absent", () => {
    let onChange: (() => void) | undefined;
    const mql: MockMediaQueryList = {
      matches: false,
      addListener: jest.fn((cb: () => void) => {
        onChange = cb;
      }),
      removeListener: jest.fn(),
    };
    (global as Record<string, unknown>).window = { matchMedia: jest.fn(() => mql) };

    const hook = renderHook();
    expect(hook.current()).toBe(false);
    expect(mql.addListener).toHaveBeenCalledWith(expect.any(Function));

    mql.matches = true;
    act(() => onChange?.());
    expect(hook.current()).toBe(true);

    hook.unmount();
    expect(mql.removeListener).toHaveBeenCalledWith(expect.any(Function));
  });

  it("never throws if matchMedia itself throws", () => {
    (global as Record<string, unknown>).window = {
      matchMedia: () => {
        throw new Error("not supported in this context");
      },
    };
    const hook = renderHook();
    expect(hook.current()).toBe(false);
    hook.unmount();
  });
});
