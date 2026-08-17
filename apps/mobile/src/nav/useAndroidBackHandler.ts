/**
 * Android hardware-back wiring (Task N3 of P3-10). Thin glue over the pure
 * logic in backHandler.ts: pushed/primary screens pop to their sensible
 * back target; on Home, a double-back-to-exit gesture — one press shows a
 * brief "Press back again to exit" hint, a second press within
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
) {
  const lastHomeBackAt = useRef<number | null>(null);

  useEffect(() => {
    if (Platform.OS !== "android") return;

    const subscription = BackHandler.addEventListener(
      "hardwareBackPress",
      () => {
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
  }, [screen, setScreen]);
}
