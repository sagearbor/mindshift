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

  // 2026-08-19 primary-eligible-expand: dashboard is now unconditionally
  // PRIMARY (see App.tsx's PRIMARY_SCREEN_NAMES/isPrimary), so hardware back
  // goes Home like the other primary screens, regardless of its `returnTo` —
  // it no longer pops through that chain (see the dedicated describe block
  // below for the "regardless of returnTo" coverage).
  it.each<[Screen, Screen]>([
    [{ name: "live-coach" }, { name: "home" }],
    [{ name: "analyze" }, { name: "home" }],
    [{ name: "growth" }, { name: "home" }],
    [{ name: "advanced" }, { name: "home" }],
    [{ name: "your-day" }, { name: "home" }],
    [{ name: "dashboard", returnTo: { name: "advanced" } }, { name: "home" }],
  ])("%o pops to home", (screen, expected) => {
    expect(backTarget(screen)).toEqual(expected);
  });

  // Task N3 fix round 1 (IMPORTANT 4): watch-setup/onboarding still carry a
  // dynamic `returnTo` (wherever they were actually launched from —
  // Settings, or the hamburger catalog from any primary screen) instead of
  // a hardcoded "advanced". backTarget just hands that value straight back.
  // (dashboard used to be in this group too — see the "pops to home"
  // describe above and the dedicated describe block below for why it moved
  // once it became unconditionally PRIMARY.)
  it.each<[Screen, Screen]>([
    [{ name: "watch-setup", returnTo: { name: "advanced" } }, { name: "advanced" }],
    [{ name: "onboarding", returnTo: { name: "advanced" } }, { name: "advanced" }],
  ])("%o pops to its returnTo (Settings here)", (screen, expected) => {
    expect(backTarget(screen)).toEqual(expected);
  });

  it.each<[Screen, Screen]>([
    [{ name: "watch-setup", returnTo: { name: "home" } }, { name: "home" }],
    [{ name: "onboarding", returnTo: { name: "live-coach" } }, { name: "live-coach" }],
  ])(
    "%o pops to its returnTo even when that ISN'T Settings — catalog-opened from a primary screen",
    (screen, expected) => {
      expect(backTarget(screen)).toEqual(expected);
    },
  );

  // 2026-08-19 primary-eligible-expand: dashboard became unconditionally
  // PRIMARY, so — unlike watch-setup/onboarding above, and unlike its own
  // pre-flip behavior — hardware back always goes Home, regardless of what
  // `returnTo` it's carrying (dashboard's `returnTo` still exists, just no
  // longer drives back-navigation; it's only read again if `detail` gets
  // pushed from here, restoring the whole chain).
  it.each<[Screen, Screen]>([
    [{ name: "dashboard", returnTo: { name: "advanced" } }, { name: "home" }],
    [{ name: "dashboard", returnTo: { name: "analyze" } }, { name: "home" }],
    [{ name: "dashboard", returnTo: { name: "live-coach" } }, { name: "home" }],
  ])(
    "%o always pops to home, ignoring returnTo, now that it's PRIMARY",
    (screen, expected) => {
      expect(backTarget(screen)).toEqual(expected);
    },
  );

  // N7 fix round 1 (IMPORTANT 2): "advanced" (Settings/Voice profile) now
  // carries the same dynamic `returnTo` pattern as watch-setup/onboarding/
  // dashboard above — it's reachable from every primary screen's avatar
  // menu, not just Settings' own rows, so a hardcoded "home" target was the
  // most-hit instance of the exact bug class those three already fixed.
  it.each<[Screen, Screen]>([
    [{ name: "advanced", returnTo: { name: "live-coach" } }, { name: "live-coach" }],
    [{ name: "advanced", returnTo: { name: "analyze" } }, { name: "analyze" }],
    [{ name: "advanced", returnTo: { name: "home" } }, { name: "home" }],
  ])(
    "%o pops to its returnTo when present — catalog-opened from a primary screen",
    (screen, expected) => {
      expect(backTarget(screen)).toEqual(expected);
    },
  );

  it("record pops to analyze", () => {
    expect(backTarget({ name: "record" })).toEqual({ name: "analyze" });
  });

  // Task N5 of P3-10: only reachable from Settings' own row, so the
  // DESTINATION is always "advanced" — but (N7 fix round 1 re-review)
  // `returnTo` now carries the WHOLE `advanced` screen it was pushed from
  // (with its own `returnTo` intact), same as `detail` carries `dashboard`
  // below, so popping back restores that chain rather than resetting it.
  it("home-design pops to the whole advanced screen (with its own returnTo) it carries", () => {
    const returnTo: Screen = { name: "advanced", returnTo: { name: "live-coach" } };
    expect(backTarget({ name: "home-design", returnTo })).toEqual(returnTo);
  });

  it("home-design pops to a bare advanced screen when Settings itself had no returnTo (Home-originated)", () => {
    const returnTo: Screen = { name: "advanced" };
    expect(backTarget({ name: "home-design", returnTo })).toEqual(returnTo);
  });

  // Task N6: avatar-capture carries the same dynamic returnTo pattern as
  // watch-setup/onboarding/dashboard — launchable from any primary screen's
  // avatar menu or from Settings.
  it.each<[Screen, Screen]>([
    [
      { name: "avatar-capture", returnTo: { name: "advanced" } },
      { name: "advanced" },
    ],
    [
      { name: "avatar-capture", returnTo: { name: "home" } },
      { name: "home" },
    ],
  ])("%o pops to its returnTo", (screen, expected) => {
    expect(backTarget(screen)).toEqual(expected);
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
