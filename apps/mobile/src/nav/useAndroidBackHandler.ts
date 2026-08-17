/**
 * Android hardware-back wiring (Task N3 of P3-10). Thin glue over the pure
 * logic in backHandler.ts: an open AppChrome overlay (hamburger catalog or
 * account menu) gets dismissed FIRST — fix round 1, CRITICAL 1: back used to
 * navigate/exit right underneath an open overlay, which could exit the app
 * entirely with the account menu still open. Only once there's no overlay to
 * close does back fall through to pushed/primary screens popping to their
 * sensible target, or, on Home, a double-back-to-exit gesture — one press
 * shows a brief "Press back again to exit" hint, a second press within
 * EXIT_WINDOW_MS lets the OS actually exit the app.
 *
 * No-op on iOS/web — there's no hardware back event there (web URL routing
 * stays a later project, per the plan), so this hook simply never registers
 * a listener off Android.
 */
import { useEffect, useRef } from "react";
import { BackHandler, Platform, ToastAndroid } from "react-native";
import type { Screen } from "../../App";
import { EXIT_WINDOW_MS, backTarget, shouldExitOnBack } from "./backHandler";

export function useAndroidBackHandler(
  screen: Screen,
  setScreen: (next: Screen) => void,
  /** Called first, before any navigation decision. Should close an open
   *  overlay and return true if it did, or return false (or be omitted) when
   *  there's nothing to close — App.tsx wires this to AppChrome's
   *  `closeOverlays()` via a ref. */
  closeOverlays?: () => boolean,
) {
  const lastHomeBackAt = useRef<number | null>(null);

  // Fix round 1, MINOR: a stray first-press hint on Home shouldn't count
  // toward a later, unrelated back-press after the user has navigated away
  // and come back — only two CONSECUTIVE presses while staying on Home
  // should exit. Clearing this whenever we leave Home (rather than only
  // resetting it once a press is consumed) makes that unambiguous.
  useEffect(() => {
    if (screen.name !== "home") {
      lastHomeBackAt.current = null;
    }
  }, [screen.name]);

  useEffect(() => {
    if (Platform.OS !== "android") return;

    const subscription = BackHandler.addEventListener(
      "hardwareBackPress",
      () => {
        if (closeOverlays?.()) {
          return true; // an overlay was open and just closed — swallow this
          // press entirely; don't also navigate or exit underneath it.
        }

        const target = backTarget(screen);
        if (target) {
          setScreen(target);
          return true; // handled — don't let the OS also exit/pop
        }

        const now = Date.now();
        if (shouldExitOnBack(lastHomeBackAt.current, now, EXIT_WINDOW_MS)) {
          return false; // let the system handle it — exits the app
        }
        lastHomeBackAt.current = now;
        ToastAndroid.show("Press back again to exit", ToastAndroid.SHORT);
        return true; // handled — swallow this press, don't exit yet
      },
    );

    return () => subscription.remove();
  }, [screen, setScreen, closeOverlays]);
}
