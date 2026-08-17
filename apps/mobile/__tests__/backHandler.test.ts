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

  // Task N3 fix round 1 (IMPORTANT 4): watch-setup/onboarding/dashboard now
  // carry a dynamic `returnTo` (wherever they were actually launched from —
  // Settings, or the hamburger catalog from any primary screen) instead of
  // a hardcoded "advanced". backTarget just hands that value straight back.
  it.each<[Screen, Screen]>([
    [{ name: "watch-setup", returnTo: { name: "advanced" } }, { name: "advanced" }],
    [{ name: "onboarding", returnTo: { name: "advanced" } }, { name: "advanced" }],
    [{ name: "dashboard", returnTo: { name: "advanced" } }, { name: "advanced" }],
  ])("%o pops to its returnTo (Settings here)", (screen, expected) => {
    expect(backTarget(screen)).toEqual(expected);
  });

  it.each<[Screen, Screen]>([
    [{ name: "watch-setup", returnTo: { name: "home" } }, { name: "home" }],
    [{ name: "onboarding", returnTo: { name: "live-coach" } }, { name: "live-coach" }],
    [{ name: "dashboard", returnTo: { name: "analyze" } }, { name: "analyze" }],
  ])(
    "%o pops to its returnTo even when that ISN'T Settings — catalog-opened from a primary screen",
    (screen, expected) => {
      expect(backTarget(screen)).toEqual(expected);
    },
  );

  it("record pops to analyze", () => {
    expect(backTarget({ name: "record" })).toEqual({ name: "analyze" });
  });

  // Task N5 of P3-10: only reachable from Settings' own row, so — unlike
  // watch-setup/onboarding/dashboard above — no dynamic returnTo is needed.
  it("home-design pops to advanced (Settings)", () => {
    expect(backTarget({ name: "home-design" })).toEqual({ name: "advanced" });
  });

  it("detail pops to whatever dashboard screen (with its own returnTo) it carries", () => {
    const returnTo: Screen = { name: "dashboard", returnTo: { name: "advanced" } };
    expect(backTarget({ name: "detail", sessionId: "s1", returnTo })).toEqual(
      returnTo,
    );
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
