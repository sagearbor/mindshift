/**
 * One honest, native-safe read of the running JS bundle's OTA (EAS Update)
 * state, plus the "apply now" action.
 *
 * WHY a wrapper (and not `useUpdates()` inline everywhere): both the About row
 * and the restart banner need the same facts, and we want a single place that
 * (a) collapses expo-updates' rich shape to just what the UI shows, and (b) is
 * trivially mockable in tests. expo-updates ships a web build, so importing it
 * is safe for `expo export -p web`; we still fold `Platform.OS === "web"` into
 * `supported` so the UI never claims OTA on web (web ships via normal deploys).
 *
 * IMPORTANT: OTA does nothing until a NEW STORE BUILD that contains the
 * expo-updates native module is installed. On the current store build (no
 * expo-updates yet) `isEnabled` is false and `supported` is false — the About
 * row says so plainly rather than pretending an update channel exists.
 */
import { Platform } from "react-native";
import { useUpdates, reloadAsync, isEnabled } from "expo-updates";

export interface OtaStatus {
  /** True only when expo-updates is actually active for this build: a real
   *  standalone/store build carrying the native module, on native. False on
   *  web, in Expo Go, and in dev — where OTA is not in play. */
  supported: boolean;
  /** True when the running JS is the bundle embedded in the store build — i.e.
   *  no OTA update has been applied (yet). */
  isEmbeddedLaunch: boolean;
  /** EAS Update channel of the current build (e.g. "production"), or null. */
  channel: string | null;
  /** Publish time of the update that's running, or null when unknown. */
  createdAt: Date | null;
  /** Runtime version the build is pinned to (appVersion policy → the app
   *  version, e.g. "1.14.0"). */
  runtimeVersion: string | null;
  /** UUID of the running update, or null (embedded/disabled). */
  updateId: string | null;
  /** A newer update finished downloading and will apply on the next launch —
   *  the trigger for the "restart to apply" affordance. */
  isUpdatePending: boolean;
  /** A check or download failed; surfaced so the UI never silently implies the
   *  app is up to date when it couldn't confirm that. */
  errored: boolean;
}

/**
 * Reactive OTA status for the running build. Safe on every platform: on web /
 * Expo Go / dev, `useUpdates()` reports a disabled build and `supported` is
 * false.
 */
export function useOtaStatus(): OtaStatus {
  const { currentlyRunning, isUpdatePending, checkError, downloadError } =
    useUpdates();
  return {
    supported: isEnabled && Platform.OS !== "web",
    isEmbeddedLaunch: currentlyRunning.isEmbeddedLaunch,
    channel: currentlyRunning.channel ?? null,
    createdAt: currentlyRunning.createdAt ?? null,
    runtimeVersion: currentlyRunning.runtimeVersion ?? null,
    updateId: currentlyRunning.updateId ?? null,
    isUpdatePending,
    errored: Boolean(checkError || downloadError),
  };
}

/**
 * Apply a downloaded-and-pending update by relaunching into it. Deliberately a
 * user-initiated action (from the banner) so we never hot-swap mid-session.
 */
export async function restartToApplyUpdate(): Promise<void> {
  await reloadAsync();
}
