/**
 * Remembers which Settings accordion sections are open, per account. Same
 * cross-platform persistence as live/experimentalPrefs.ts (expo-secure-store
 * on native, localStorage on web — no extra native dependency, so OTA-safe),
 * every access wrapped in try/catch and fail-open to "all collapsed".
 */
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";

export type OpenSections = Record<string, boolean>;

export const ACCORDION_KEY_PREFIX = "mindshift.settingsSections.v1";

export function accordionKey(userId: string | null | undefined): string {
  const id = (userId || "anon").replace(/[^A-Za-z0-9._-]/g, "_");
  return `${ACCORDION_KEY_PREFIX}.${id}`;
}

function webStorage(): Storage | null {
  try {
    return (globalThis as { localStorage?: Storage }).localStorage ?? null;
  } catch {
    return null;
  }
}

export async function loadOpenSections(userId: string | null | undefined): Promise<OpenSections> {
  try {
    const key = accordionKey(userId);
    const raw = Platform.OS === "web" ? webStorage()?.getItem(key) ?? null : await SecureStore.getItemAsync(key);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    const out: OpenSections = {};
    for (const [k, v] of Object.entries(parsed)) if (typeof v === "boolean") out[k] = v;
    return out;
  } catch {
    return {};
  }
}

export async function saveOpenSections(
  userId: string | null | undefined,
  open: OpenSections,
): Promise<void> {
  try {
    const key = accordionKey(userId);
    const raw = JSON.stringify(open);
    if (Platform.OS === "web") webStorage()?.setItem(key, raw);
    else await SecureStore.setItemAsync(key, raw);
  } catch {
    // best-effort: a failed write just means the section re-collapses next launch
  }
}
