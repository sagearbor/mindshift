import {
  backTarget,
  shouldExitOnBack,
  EXIT_WINDOW_MS,
} from "../src/nav/backHandler";
import type { Screen } from "../App";

describe("backTarget", () => {
  it("returns null for home — no pop target, handled as double-back-to-exit", () => {
    expect(backTarget({ name: "home" })).toBeNull();
  });

  it.each<[Screen, Screen]>([
    [{ name: "live-coach" }, { name: "home" }],
    [{ name: "analyze" }, { name: "home" }],
    [{ name: "growth" }, { name: "home" }],
    [{ name: "advanced" }, { name: "home" }],
    [{ name: "your-day" }, { name: "home" }],
  ])("%o pops to home", (screen, expected) => {
    expect(backTarget(screen)).toEqual(expected);
  });

  it.each<[Screen, Screen]>([
    [{ name: "watch-setup" }, { name: "advanced" }],
    [{ name: "onboarding" }, { name: "advanced" }],
    [{ name: "dashboard" }, { name: "advanced" }],
  ])("%o pops to Settings", (screen, expected) => {
    expect(backTarget(screen)).toEqual(expected);
  });

  it("record pops to analyze", () => {
    expect(backTarget({ name: "record" })).toEqual({ name: "analyze" });
  });

  it("detail pops to dashboard", () => {
    expect(backTarget({ name: "detail", sessionId: "s1" })).toEqual({
      name: "dashboard",
    });
  });

  describe("session", () => {
    it("pops to analyze when it was pushed from analyze", () => {
      expect(
        backTarget({ name: "session", returnTo: "analyze" }),
      ).toEqual({ name: "analyze" });
    });

    it("pops to home when it was pushed from home", () => {
      expect(backTarget({ name: "session", returnTo: "home" })).toEqual({
        name: "home",
      });
    });
  });

  describe("recordings", () => {
    it("pops to analyze when it carries returnTo analyze", () => {
      expect(
        backTarget({ name: "recordings", returnTo: "analyze" }),
      ).toEqual({ name: "analyze" });
    });

    it("pops to home when it carries returnTo home", () => {
      expect(
        backTarget({ name: "recordings", returnTo: "home" }),
      ).toEqual({ name: "home" });
    });
  });

  it("dynamics pops to whatever returnTo it carries", () => {
    const returnTo: Screen = { name: "session", returnTo: "home" };
    expect(
      backTarget({ name: "dynamics", returnTo }),
    ).toEqual(returnTo);
  });

  it("replay pops to whatever returnTo it carries", () => {
    const returnTo: Screen = { name: "growth" };
    expect(
      backTarget({
        name: "replay",
        recordingId: "r1",
        returnTo,
      }),
    ).toEqual(returnTo);
  });
});

describe("shouldExitOnBack", () => {
  it("does not exit on the very first press (no prior timestamp)", () => {
    expect(shouldExitOnBack(null, 1_000)).toBe(false);
  });

  it("exits when the second press lands within the window", () => {
    expect(shouldExitOnBack(1_000, 1_000 + EXIT_WINDOW_MS, EXIT_WINDOW_MS)).toBe(
      true,
    );
  });

  it("exits for a press strictly inside the window", () => {
    expect(shouldExitOnBack(1_000, 1_500, EXIT_WINDOW_MS)).toBe(true);
  });

  it("does not exit once the window has elapsed", () => {
    expect(
      shouldExitOnBack(1_000, 1_000 + EXIT_WINDOW_MS + 1, EXIT_WINDOW_MS),
    ).toBe(false);
  });

  it("honors a custom window", () => {
    expect(shouldExitOnBack(1_000, 1_400, 500)).toBe(true);
    expect(shouldExitOnBack(1_000, 1_600, 500)).toBe(false);
  });
});
