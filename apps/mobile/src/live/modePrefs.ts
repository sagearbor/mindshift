/**
 * The live session mode the user last chose (earpiece / in person /
 * therapist / call), remembered PER ACCOUNT so Sage's phone opens on the
 * mode he used last and Mom's on hers without either re-picking every
 * session. ("In person" is stored under its original wire value `speaker`
 * so prefs saved before the rename still load.)
 *
 * Same cross-platform persistence as layoutStore.ts / onboardingStorage.ts
 * (expo-secure-store on native, localStorage on web), fail-open: a storage
 * error or an unknown stored value falls back to the default. Kept free of
 * React so the hook and the tests call it directly.
 */
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import type { LiveMode } from "./localLlm";

export const MODE_KEY_PREFIX = "mindshift.liveMode.v1";
export const DEFAULT_LIVE_MODE: LiveMode = "earpiece";

const MODES: readonly LiveMode[] = ["earpiece", "speaker", "therapist", "call"];

export function isLiveMode(value: unknown): value is LiveMode {
  return typeof value === "string" && (MODES as readonly string[]).includes(value);
}

/** SecureStore keys must be alphanumeric/./-/_ — a Firebase uid already is;
 *  anything else (an email, "anon") is sanitized the same way. */
export function modeKey(userId: string | null | undefined): string {
  const id = (userId || "anon").replace(/[^A-Za-z0-9._-]/g, "_");
  return `${MODE_KEY_PREFIX}.${id}`;
}

function webStorage(): Storage | null {
  try {
    const g = globalThis as { localStorage?: Storage };
    return g.localStorage ?? null;
  } catch {
    return null;
  }
}

export async function loadLiveMode(userId: string | null | undefined): Promise<LiveMode> {
  const key = modeKey(userId);
  try {
    const raw =
      Platform.OS === "web"
        ? webStorage()?.getItem(key) ?? null
        : await SecureStore.getItemAsync(key);
    return isLiveMode(raw) ? raw : DEFAULT_LIVE_MODE;
  } catch {
    return DEFAULT_LIVE_MODE;
  }
}

export async function saveLiveMode(
  userId: string | null | undefined,
  mode: LiveMode,
): Promise<void> {
  const key = modeKey(userId);
  try {
    if (Platform.OS === "web") {
      webStorage()?.setItem(key, mode);
    } else {
      await SecureStore.setItemAsync(key, mode);
    }
  } catch {
    // Fail-open: the choice still applies to this session.
  }
}
