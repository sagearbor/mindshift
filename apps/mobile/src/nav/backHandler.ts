/**
 * Pure Android hardware-back logic (Task N3 of P3-10). Two independent
 * pieces, both framework-free so they're unit-testable without mounting the
 * app or touching React Native's BackHandler:
 *
 *  - `backTarget`: mirrors, screen-for-screen, the exact same "go back to"
 *    destination each pushed screen's own `onBack` prop already uses in
 *    App.tsx's `renderScreen()`. Returns null for "home" — home has no pop
 *    target; App.tsx's wiring (useAndroidBackHandler.ts) handles that
 *    specially via `shouldExitOnBack` below. The switch is exhaustive over
 *    the `Screen` union (TypeScript enforces it via the `never` check at the
 *    bottom) — adding a new Screen variant without a case here is a compile
 *    error, so this can't silently drift from App.tsx's real navigation.
 *
 *  - `shouldExitOnBack`: the "press back again to exit" double-tap window,
 *    as a pure function of two timestamps + a threshold — no real timers,
 *    no BackHandler, fully deterministic in tests.
 */
import type { Screen } from "../../App";

/** How long a second home back-press must land within to actually exit
 *  (Android's standard "press back again to exit" pattern). */
export const EXIT_WINDOW_MS = 2000;

/**
 * Where hardware back should land for a given (pushed) screen. Null means
 * "no target" — currently only true for "home", which the caller handles as
 * the double-back-to-exit case instead of a pop.
 */
export function backTarget(screen: Screen): Screen | null {
  switch (screen.name) {
    case "home":
      return null;

    // Primary screens (Task N3: wrapped in AppChrome, no back button of
    // their own) — hardware back from any of them goes to Home, same target
    // as AppChrome's wordmark tap.
    case "live-coach":
    case "analyze":
    case "growth":
      return { name: "home" };

    // Pushed screens whose existing onBack targets Home.
    case "advanced":
    case "your-day":
      return { name: "home" };

    // Task N5 of P3-10: only reachable from Settings' own row, so back
    // always lands there — same as its onBack prop in App.tsx.
    case "home-design":
      return { name: "advanced" };

    // Task N3 fix round 1: these carry a dynamic `returnTo` now (wherever
    // they were actually launched from — Settings, or the hamburger catalog
    // from any primary screen) instead of a hardcoded "advanced".
    case "watch-setup":
    case "onboarding":
    case "dashboard":
    // Task N6: same dynamic-returnTo treatment — the avatar-capture flow can
    // be launched from any primary screen's avatar menu, or from Settings.
    case "avatar-capture":
      return screen.returnTo;

    case "record":
      return { name: "analyze" };

    case "session":
      return screen.returnTo === "analyze"
        ? { name: "analyze" }
        : { name: "home" };

    case "recordings":
      return screen.returnTo === "analyze"
        ? { name: "analyze" }
        : { name: "home" };

    case "dynamics":
      return screen.returnTo;

    case "replay":
      return screen.returnTo;

    case "detail":
      // returnTo is the whole dashboard screen (with ITS OWN returnTo) it
      // was pushed from — popping to it exactly restores that chain.
      return screen.returnTo;

    default: {
      // Exhaustiveness guard: a new Screen variant with no case above fails
      // the build here, not silently at runtime.
      const exhaustive: never = screen;
      return exhaustive;
    }
  }
}

/**
 * Pure double-back-to-exit decision: given when the last home back-press
 * happened (null = never, or the caller has already reset it) and the
 * current time, decide whether THIS press should be allowed to exit the app.
 */
export function shouldExitOnBack(
  lastPressAt: number | null,
  now: number,
  windowMs: number = EXIT_WINDOW_MS,
): boolean {
  return lastPressAt !== null && now - lastPressAt <= windowMs;
}
