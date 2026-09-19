/**
 * Whether a live coaching session KEEPS its audio — remembered PER ACCOUNT
 * like the session mode (modePrefs.ts). ON by default (owner decision,
 * 2026-08-29): if you're recording yourself you'll likely want to go back
 * to it, and the audio is what makes replay, re-analysis and voice
 * learning possible for a live session. Turning it off means the session
 * is transcribed as it streams and nothing is written down — the app's
 * original live-session behaviour.
 *
 * The privacy policy and the Play Data Safety answers describe this
 * default (docs/privacy.html, docs/play/play-answers-mindshift.yaml);
 * changing it here without changing them makes them false.
 *
 * Same cross-platform persistence as modePrefs.ts (expo-secure-store on
 * native, localStorage on web), fail-open to the default.
 */
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";

export const KEEP_AUDIO_KEY_PREFIX = "mindshift.keepAudio.v1";
export const DEFAULT_KEEP_AUDIO = true;

export function keepAudioKey(userId: string | null | undefined): string {
  const id = (userId || "anon").replace(/[^A-Za-z0-9._-]/g, "_");
  return `${KEEP_AUDIO_KEY_PREFIX}.${id}`;
}

function webStorage(): Storage | null {
  try {
    const g = globalThis as { localStorage?: Storage };
    return g.localStorage ?? null;
  } catch {
    return null;
  }
}

export async function loadKeepAudio(userId: string | null | undefined): Promise<boolean> {
  const key = keepAudioKey(userId);
  try {
    const raw =
      Platform.OS === "web"
        ? webStorage()?.getItem(key) ?? null
        : await SecureStore.getItemAsync(key);
    if (raw === "on") return true;
    if (raw === "off") return false;
    return DEFAULT_KEEP_AUDIO;
  } catch {
    return DEFAULT_KEEP_AUDIO;
  }
}

export async function saveKeepAudio(
  userId: string | null | undefined,
  on: boolean,
): Promise<void> {
  const key = keepAudioKey(userId);
  const value = on ? "on" : "off";
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
