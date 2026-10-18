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
 *
 * Also holds the owner-only "Debug session capture" switch shown under
 * Developer mode (see below).
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

// --- "Debug session capture" (2026-10-18, before 1.19.0 went public) -------
// The owner-only switch shown under Developer mode (Settings → Diagnostics).
// ON = keep a local copy of every finished live session's payload (the
// transcript) in the app's document dir (live/sessionPayloadKeep.ts) AND ask
// the server, per session, to keep its own transcript in memory so a
// zero-turn POST /sessions/live is stored instead of refused (config frame
// + POST body `debug_capture: true`). OFF by default. Only ACTIVE while
// Developer mode is on too, so turning Developer mode off can never leave a
// hidden transcript capture running. Same per-account persistence.
export const DEBUG_CAPTURE_KEY_PREFIX = "mindshift.debugSessionCapture.v1";
export const DEFAULT_DEBUG_CAPTURE = false;

export function debugCaptureKey(userId: string | null | undefined): string {
  const id = (userId || "anon").replace(/[^A-Za-z0-9._-]/g, "_");
  return `${DEBUG_CAPTURE_KEY_PREFIX}.${id}`;
}

export async function loadDebugCapture(userId: string | null | undefined): Promise<boolean> {
  const key = debugCaptureKey(userId);
  try {
    const raw = Platform.OS === "web" ? (webStorage()?.getItem(key) ?? null) : await SecureStore.getItemAsync(key);
    if (raw === "on") return true;
    if (raw === "off") return false;
    return DEFAULT_DEBUG_CAPTURE;
  } catch {
    return DEFAULT_DEBUG_CAPTURE;
  }
}

export async function saveDebugCapture(userId: string | null | undefined, on: boolean): Promise<void> {
  const key = debugCaptureKey(userId);
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
  /** The stored "Debug session capture" switch (see isDebugCaptureActive
   *  for whether it is in force). */
  debugCapture: boolean;
  /** Load the persisted flags for this account (fail-open to OFF). */
  hydrate: (userId: string | null | undefined) => Promise<void>;
  /** Flip + persist for this account. */
  setDevMode: (userId: string | null | undefined, on: boolean) => void;
  /** Flip + persist "Debug session capture" for this account. */
  setDebugCapture: (userId: string | null | undefined, on: boolean) => void;
}

export const useDevModeStore = create<DevModeState>((set) => ({
  devMode: DEFAULT_DEV_MODE,
  debugCapture: DEFAULT_DEBUG_CAPTURE,

  hydrate: async (userId) => {
    const [on, capture] = await Promise.all([loadDevMode(userId), loadDebugCapture(userId)]);
    set({ devMode: on, debugCapture: capture });
  },

  setDevMode: (userId, on) => {
    set({ devMode: on });
    void saveDevMode(userId, on);
  },

  setDebugCapture: (userId, on) => {
    set({ debugCapture: on });
    void saveDebugCapture(userId, on);
  },
}));

/** Debug session capture is in force: the switch is on AND Developer mode
 *  is on. Read at session start by hooks/useAudioStream.ts. */
export function isDebugCaptureActive(): boolean {
  const s = useDevModeStore.getState();
  return s.devMode && s.debugCapture;
}
