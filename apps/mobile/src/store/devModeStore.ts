/**
 * "Developer mode" — the Settings switch that reveals every diagnostic
 * detail the owner debugs with (capability strings, latency lines, engine
 * and model names, backend/update ids, raw connection states). OFF by
 * default: invited testers get the clean product surface; nothing is
 * deleted, only hidden — flipping the switch shows exactly what shipped
 * before the mode existed.
 *
 * A zustand store (so any component can read the flag without prop
 * drilling) persisted PER ACCOUNT with the same cross-platform pattern as
 * experimentalPrefs.ts / layoutStore.ts: expo-secure-store on native,
 * localStorage on web, fail-open to the default. App.tsx hydrates it on
 * auth changes, same as the other launch hydrations.
 */
import { create } from "zustand";
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";

export const DEV_MODE_KEY_PREFIX = "mindshift.developerMode.v1";
export const DEFAULT_DEV_MODE = false;

export function devModeKey(userId: string | null | undefined): string {
  const id = (userId || "anon").replace(/[^A-Za-z0-9._-]/g, "_");
  return `${DEV_MODE_KEY_PREFIX}.${id}`;
}

function webStorage(): Storage | null {
  try {
    const g = globalThis as { localStorage?: Storage };
    return g.localStorage ?? null;
  } catch {
    return null;
  }
}

export async function loadDevMode(userId: string | null | undefined): Promise<boolean> {
  const key = devModeKey(userId);
  try {
    const raw = Platform.OS === "web" ? (webStorage()?.getItem(key) ?? null) : await SecureStore.getItemAsync(key);
    if (raw === "on") return true;
    if (raw === "off") return false;
    return DEFAULT_DEV_MODE;
  } catch {
    return DEFAULT_DEV_MODE;
  }
}

export async function saveDevMode(userId: string | null | undefined, on: boolean): Promise<void> {
  const key = devModeKey(userId);
  try {
    if (Platform.OS === "web") webStorage()?.setItem(key, on ? "on" : "off");
    else await SecureStore.setItemAsync(key, on ? "on" : "off");
  } catch {
    // Fail-open: the switch still applies to this launch.
  }
}

interface DevModeState {
  /** True = show every diagnostic detail (the pre-dev-mode UI). */
  devMode: boolean;
  /** Load the persisted flag for this account (fail-open to OFF). */
  hydrate: (userId: string | null | undefined) => Promise<void>;
  /** Flip + persist for this account. */
  setDevMode: (userId: string | null | undefined, on: boolean) => void;
}

export const useDevModeStore = create<DevModeState>((set) => ({
  devMode: DEFAULT_DEV_MODE,

  hydrate: async (userId) => {
    const on = await loadDevMode(userId);
    set({ devMode: on });
  },

  setDevMode: (userId, on) => {
    set({ devMode: on });
    void saveDevMode(userId, on);
  },
}));
