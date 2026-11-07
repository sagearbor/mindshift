/**
 * Emotion dev tester switches, remembered per account with the same
 * cross-platform persistence as experimentalPrefs.ts (expo-secure-store on
 * native, localStorage on web), fail-open to the defaults (mode OFF).
 */
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import { DEFAULT_USER_SETTINGS, type EmotionDevUserSettings } from "./config";

export const EMOTION_DEV_KEY_PREFIX = "mindshift.emotionDev.v1";

export function emotionDevKey(userId: string | null | undefined): string {
  const id = (userId || "anon").replace(/[^A-Za-z0-9._-]/g, "_");
  return `${EMOTION_DEV_KEY_PREFIX}.${id}`;
}

/** Defensive parse: anything unreadable or out of range falls back to the default. */
export function parseSettings(raw: string | null): EmotionDevUserSettings {
  if (!raw) return DEFAULT_USER_SETTINGS;
  try {
    const o = JSON.parse(raw) as Partial<EmotionDevUserSettings>;
    const perHour = Number(o.maxNudgesPerHour);
    return {
      enabled: o.enabled === true,
      maxNudgesPerHour: Number.isFinite(perHour) && perHour >= 0 && perHour <= 60 ? Math.floor(perHour) : DEFAULT_USER_SETTINGS.maxNudgesPerHour,
      nudgeSuggestion: o.nudgeSuggestion === true,
    };
  } catch {
    return DEFAULT_USER_SETTINGS;
  }
}

function webStorage(): Storage | null {
  try {
    return (globalThis as { localStorage?: Storage }).localStorage ?? null;
  } catch {
    return null;
  }
}

export async function loadEmotionDevSettings(userId: string | null | undefined): Promise<EmotionDevUserSettings> {
  const key = emotionDevKey(userId);
  try {
    const raw = Platform.OS === "web" ? webStorage()?.getItem(key) ?? null : await SecureStore.getItemAsync(key);
    return parseSettings(raw);
  } catch {
    return DEFAULT_USER_SETTINGS;
  }
}

export async function saveEmotionDevSettings(userId: string | null | undefined, s: EmotionDevUserSettings): Promise<void> {
  const key = emotionDevKey(userId);
  try {
    const value = JSON.stringify(s);
    if (Platform.OS === "web") webStorage()?.setItem(key, value);
    else await SecureStore.setItemAsync(key, value);
  } catch {
    // Fail-open: the switch still applies to this launch.
  }
}
