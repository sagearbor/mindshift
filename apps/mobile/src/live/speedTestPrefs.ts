/**
 * Speed test (2026-10-11, owner-approved preview build): "Send turn to
 * cloud immediately". When ON the fast loop sends each turn's turn_local
 * BEFORE its own on-device LLM answers (FastLoopDeps.earlyTurnLocal), so the
 * cloud suggestion's clock starts at speech end instead of after the local
 * model (up to its 8 s budget on a Pixel 10).
 *
 * Persisted PER DEVICE (not per account — it is a property of this phone's
 * build under test) with the same cross-platform pattern as
 * keepAudioPrefs.ts: expo-secure-store on native, localStorage on web,
 * fail-open to the default. DEFAULT ON in this build; the switch lives in
 * Settings → Advanced → Experimental (Developer mode on), so it can be
 * turned off without another OTA.
 *
 * SPEED_TEST_AUTO_SEND: while this build is under test every live session's
 * diagnostics record (with per-turn timing marks and the flag state) is
 * sent to /telemetry at session end, not only sessions with errors — so
 * ON vs OFF sessions can be compared from files afterwards.
 */
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import { create } from "zustand";

export const EARLY_TURN_LOCAL_KEY = "mindshift.earlyTurnLocal.device.v1";
export const DEFAULT_EARLY_TURN_LOCAL = true;
export const SPEED_TEST_AUTO_SEND = true;

function webStorage(): Storage | null {
  try {
    const g = globalThis as { localStorage?: Storage };
    return g.localStorage ?? null;
  } catch {
    return null;
  }
}

export async function loadEarlyTurnLocal(): Promise<boolean> {
  try {
    const raw =
      Platform.OS === "web"
        ? (webStorage()?.getItem(EARLY_TURN_LOCAL_KEY) ?? null)
        : await SecureStore.getItemAsync(EARLY_TURN_LOCAL_KEY);
    if (raw === "on") return true;
    if (raw === "off") return false;
    return DEFAULT_EARLY_TURN_LOCAL;
  } catch {
    return DEFAULT_EARLY_TURN_LOCAL;
  }
}

export async function saveEarlyTurnLocal(on: boolean): Promise<void> {
  const value = on ? "on" : "off";
  try {
    if (Platform.OS === "web") webStorage()?.setItem(EARLY_TURN_LOCAL_KEY, value);
    else await SecureStore.setItemAsync(EARLY_TURN_LOCAL_KEY, value);
  } catch {
    // Fail-open: the switch still applies to this launch.
  }
}

interface SpeedTestState {
  /** True = send turn_local before the on-device LLM (earlyTurnLocal). */
  earlyTurnLocal: boolean;
  hydrated: boolean;
  hydrate: () => Promise<void>;
  setEarlyTurnLocal: (on: boolean) => void;
}

export const useSpeedTestStore = create<SpeedTestState>((set, get) => ({
  earlyTurnLocal: DEFAULT_EARLY_TURN_LOCAL,
  hydrated: false,
  hydrate: async () => {
    if (get().hydrated) return;
    const on = await loadEarlyTurnLocal();
    // A flip made while the load was in flight wins.
    if (!get().hydrated) set({ earlyTurnLocal: on, hydrated: true });
  },
  setEarlyTurnLocal: (on) => {
    set({ earlyTurnLocal: on, hydrated: true });
    void saveEarlyTurnLocal(on);
  },
}));
