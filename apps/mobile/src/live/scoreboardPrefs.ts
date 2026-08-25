/**
 * Whether the Live Coach screen shows the pleasantness scoreboard —
 * remembered PER ACCOUNT like the session mode (modePrefs.ts), OFF by
 * default: it is an opt-in game ("who's being nicer"), never something a
 * first session springs on someone.
 *
 * Same cross-platform persistence as modePrefs.ts (expo-secure-store on
 * native, localStorage on web), fail-open to the default.
 */
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";

export const SCOREBOARD_KEY_PREFIX = "mindshift.scoreboard.v1";
export const DEFAULT_SCOREBOARD_VISIBLE = false;

export function scoreboardKey(userId: string | null | undefined): string {
  const id = (userId || "anon").replace(/[^A-Za-z0-9._-]/g, "_");
  return `${SCOREBOARD_KEY_PREFIX}.${id}`;
}

function webStorage(): Storage | null {
  try {
    const g = globalThis as { localStorage?: Storage };
    return g.localStorage ?? null;
  } catch {
    return null;
  }
}

export async function loadScoreboardVisible(userId: string | null | undefined): Promise<boolean> {
  const key = scoreboardKey(userId);
  try {
    const raw =
      Platform.OS === "web"
        ? webStorage()?.getItem(key) ?? null
        : await SecureStore.getItemAsync(key);
    if (raw === "on") return true;
    if (raw === "off") return false;
    return DEFAULT_SCOREBOARD_VISIBLE;
  } catch {
    return DEFAULT_SCOREBOARD_VISIBLE;
  }
}

export async function saveScoreboardVisible(
  userId: string | null | undefined,
  visible: boolean,
): Promise<void> {
  const key = scoreboardKey(userId);
  const value = visible ? "on" : "off";
  try {
    if (Platform.OS === "web") {
      webStorage()?.setItem(key, value);
    } else {
      await SecureStore.setItemAsync(key, value);
    }
  } catch {
    // Fail-open: the toggle still applies to this session.
  }
}
