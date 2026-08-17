/**
 * Persisted home/nav layout (Task N1 of P3-10 — the foundation the chrome
 * (N3), home boxes (N4), and the "Home screen design" editor (N5) build on).
 *
 * Two slot lists, both storing only `DestId`s from the destination registry:
 *  - tabSlots: the configurable bottom bar, 0–5 entries.
 *  - homeBoxes: the home screen's icon+label boxes, 0–4 entries.
 *
 * App.tsx calls `hydrate()` once on mount (Task N3 fix round 1 — it was
 * defined here but never actually invoked anywhere, so the tab bar ran on
 * hardcoded defaults forever). Otherwise this file is pure store +
 * persistence, fully unit-tested on its own.
 *
 * Persistence mirrors onboardingStorage.ts's cross-platform pattern exactly
 * (this app's one established non-Firebase persistence approach) rather than
 * adding a new dependency (e.g. zustand's `persist` middleware, unused
 * anywhere else in this repo):
 *  - native (iOS/Android): expo-secure-store (already mocked in jest-setup.ts)
 *  - web: window.localStorage (via globalThis, not window — see
 *    onboardingStorage.ts's comment on why)
 *
 * Fail-open, always: a storage error or corrupt/unparsable payload never
 * throws — it falls back to the defaults below. A partially-valid payload
 * (e.g. one array with some unknown ids mixed in) keeps its still-valid
 * entries rather than discarding the whole thing, per "unknown ids dropped
 * on load".
 */
import { create } from "zustand";
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import { type DestId, isPrimaryEligible } from "../nav/destinations";

// Exported so tests can assert against it directly instead of duplicating
// the literal string (Task N3 fix round 1's hydrate-on-mount test).
export const KEY = "mindshift.layout.v1";

/** Bottom bar cap — 0–5 slots per the owner's locked design (P3-9 RESOLVED). */
export const TAB_SLOT_CAP = 5;
/** Home boxes cap — 0–4 boxes per the owner's locked design. */
export const HOME_BOX_CAP = 4;

/** Default layout: the two daily modes + growth on the tab bar, and a home
 *  area that leans on recordings + the growth trend — the owner's directive
 *  that "home could be trend/history" once the tab bar covers the two daily
 *  actions (P3-9 RESOLVED). */
export const DEFAULT_TAB_SLOTS: readonly DestId[] = [
  "coach",
  "analyze",
  "growth",
];
export const DEFAULT_HOME_BOXES: readonly DestId[] = ["recordings", "growth"];

interface PersistedLayout {
  tabSlots: DestId[];
  homeBoxes: DestId[];
}

function defaultLayout(): PersistedLayout {
  return {
    tabSlots: [...DEFAULT_TAB_SLOTS],
    homeBoxes: [...DEFAULT_HOME_BOXES],
  };
}

/** `globalThis.localStorage`, not `window.localStorage` — see
 *  onboardingStorage.ts for why (no `window` in this repo's default jest
 *  environment). */
function webLocalStorage(): Storage | null {
  const g = globalThis as unknown as { localStorage?: Storage };
  return g.localStorage ?? null;
}

async function readRaw(): Promise<string | null> {
  try {
    return Platform.OS === "web"
      ? (webLocalStorage()?.getItem(KEY) ?? null)
      : await SecureStore.getItemAsync(KEY);
  } catch {
    return null;
  }
}

async function writeRaw(value: string): Promise<void> {
  try {
    if (Platform.OS === "web") {
      webLocalStorage()?.setItem(KEY, value);
      return;
    }
    await SecureStore.setItemAsync(KEY, value);
  } catch {
    // Best-effort — a write failure just means the layout may revert to
    // defaults next launch, which is safe, never a crash.
  }
}

/**
 * Validate a candidate slot list: drop anything that isn't a known,
 * primary-eligible destination id, drop duplicates (first occurrence wins),
 * and truncate to `cap`. Used identically for setter input and for values
 * loaded from storage, so the same rules hold whether the list came from the
 * editor UI or from a prior app version's persisted (possibly stale) data.
 */
export function sanitizeSlots(ids: unknown, cap: number): DestId[] {
  if (!Array.isArray(ids)) return [];
  const seen = new Set<DestId>();
  const out: DestId[] = [];
  for (const raw of ids) {
    if (typeof raw !== "string" || !isPrimaryEligible(raw)) continue;
    if (seen.has(raw)) continue;
    seen.add(raw);
    out.push(raw);
    if (out.length >= cap) break;
  }
  return out;
}

async function loadLayout(): Promise<PersistedLayout> {
  const raw = await readRaw();
  if (raw == null) return defaultLayout();
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return defaultLayout(); // corrupt payload — fail open to defaults
  }
  if (typeof parsed !== "object" || parsed === null) return defaultLayout();
  const record = parsed as Record<string, unknown>;
  return {
    tabSlots: sanitizeSlots(record.tabSlots, TAB_SLOT_CAP),
    homeBoxes: sanitizeSlots(record.homeBoxes, HOME_BOX_CAP),
  };
}

async function persistLayout(layout: PersistedLayout): Promise<void> {
  await writeRaw(JSON.stringify(layout));
}

interface LayoutState {
  tabSlots: DestId[];
  homeBoxes: DestId[];
  /** True once `hydrate()` has completed at least once — lets a future
   *  consumer avoid flashing the hardcoded defaults over a real saved
   *  layout, the same gate pattern authStore uses for `initializing`. */
  hydrated: boolean;

  /** Load the persisted layout (or defaults, fail-open) and apply it.
   *  Idempotent to call repeatedly; safe to call from App-level init. */
  hydrate: () => Promise<void>;
  /** Replace the tab slots, validated (dedup/eligibility/cap), then persist. */
  setTabSlots: (ids: DestId[]) => void;
  /** Replace the home boxes, validated (dedup/eligibility/cap), then persist. */
  setHomeBoxes: (ids: DestId[]) => void;
  /** Restore both lists to the shipped defaults and persist that. */
  resetToDefaults: () => void;
}

export const useLayoutStore = create<LayoutState>((set, get) => ({
  ...defaultLayout(),
  hydrated: false,

  hydrate: async () => {
    const loaded = await loadLayout();
    set({ ...loaded, hydrated: true });
  },

  setTabSlots: (ids) => {
    const tabSlots = sanitizeSlots(ids, TAB_SLOT_CAP);
    set({ tabSlots });
    void persistLayout({ tabSlots, homeBoxes: get().homeBoxes });
  },

  setHomeBoxes: (ids) => {
    const homeBoxes = sanitizeSlots(ids, HOME_BOX_CAP);
    set({ homeBoxes });
    void persistLayout({ tabSlots: get().tabSlots, homeBoxes });
  },

  resetToDefaults: () => {
    const next = defaultLayout();
    set({ ...next, hydrated: true });
    void persistLayout(next);
  },
}));
