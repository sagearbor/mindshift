/**
 * Persisted "has the first-launch onboarding walkthrough been shown" flag
 * (Task P3-7). Not sensitive, but the app has exactly one cross-platform
 * persistence pattern already wired up — src/auth/firebase.ts's
 * createAuth() — so this mirrors it rather than adding a new dependency:
 *  - native (iOS/Android): expo-secure-store (already a dependency, already
 *    mocked in jest-setup.ts)
 *  - web: window.localStorage (browserLocalPersistence's backing store)
 *
 * Kept as two tiny async functions (not a class/store) so it's trivial to
 * unit-test and trivial to call from both the App-level auto-show gate and
 * the Settings "Show tutorial" row.
 */
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";

const KEY = "mindshift.onboarding.seen.v1";

/** `globalThis.localStorage`, not `window.localStorage` — the latter throws a
 *  ReferenceError to `window` itself in non-DOM test environments (this
 *  repo's default jest preset has no `window`), where `globalThis` is always
 *  defined. Real web runtimes expose the same Storage object both ways. */
function webLocalStorage(): Storage | null {
  const g = globalThis as unknown as { localStorage?: Storage };
  return g.localStorage ?? null;
}

/** True once the walkthrough has been shown to completion or skipped.
 *  Fails open to `false` (show the tutorial) on any read error — a repeat
 *  showing is a mild annoyance, silently skipping it forever is not. */
export async function getOnboardingSeen(): Promise<boolean> {
  try {
    const raw =
      Platform.OS === "web"
        ? (webLocalStorage()?.getItem(KEY) ?? null)
        : await SecureStore.getItemAsync(KEY);
    return raw === "true";
  } catch {
    return false;
  }
}

/** Mark the walkthrough seen — called on both Skip and Get started, since
 *  either way the user has made an informed choice not to see it again on
 *  next launch. Best-effort: a write failure just means it may show again
 *  next launch, which is safe. */
export async function setOnboardingSeen(seen: boolean): Promise<void> {
  try {
    if (Platform.OS === "web") {
      webLocalStorage()?.setItem(KEY, seen ? "true" : "false");
      return;
    }
    await SecureStore.setItemAsync(KEY, seen ? "true" : "false");
  } catch {
    // Best-effort — see doc comment above.
  }
}
