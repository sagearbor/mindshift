/**
 * "Experimental voice engine" — the Advanced-screen switch that reveals the
 * on-phone voice-separation action on a stored recording's replay
 * (components/DeviceDiarizationRow.tsx, engine: live/diarizeWindows.ts,
 * approach B of the 2026-08-29 bake-off run post-hoc on the phone). OFF by
 * default: it is a research instrument for scoring the phone's result
 * against a per-second rubric, not a product feature yet.
 *
 * Remembered PER ACCOUNT with the same cross-platform persistence as
 * keepAudioPrefs.ts (expo-secure-store on native, localStorage on web),
 * fail-open to the default.
 */
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";

export const EXPERIMENTAL_VOICE_ENGINE_KEY_PREFIX = "mindshift.experimentalVoiceEngine.v1";
export const DEFAULT_EXPERIMENTAL_VOICE_ENGINE = false;

export function experimentalVoiceEngineKey(userId: string | null | undefined): string {
  const id = (userId || "anon").replace(/[^A-Za-z0-9._-]/g, "_");
  return `${EXPERIMENTAL_VOICE_ENGINE_KEY_PREFIX}.${id}`;
}

function webStorage(): Storage | null {
  try {
    const g = globalThis as { localStorage?: Storage };
    return g.localStorage ?? null;
  } catch {
    return null;
  }
}

export async function loadExperimentalVoiceEngine(userId: string | null | undefined): Promise<boolean> {
  const key = experimentalVoiceEngineKey(userId);
  try {
    const raw = Platform.OS === "web" ? webStorage()?.getItem(key) ?? null : await SecureStore.getItemAsync(key);
    if (raw === "on") return true;
    if (raw === "off") return false;
    return DEFAULT_EXPERIMENTAL_VOICE_ENGINE;
  } catch {
    return DEFAULT_EXPERIMENTAL_VOICE_ENGINE;
  }
}

export async function saveExperimentalVoiceEngine(userId: string | null | undefined, on: boolean): Promise<void> {
  const key = experimentalVoiceEngineKey(userId);
  const value = on ? "on" : "off";
  try {
    if (Platform.OS === "web") webStorage()?.setItem(key, value);
    else await SecureStore.setItemAsync(key, value);
  } catch {
    // Fail-open: the switch still applies to this launch.
  }
}
