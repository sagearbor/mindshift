/**
 * The outcome-engine mood check (Workstream 4 —
 * docs/plans/2026-09-04-naturalturn-conversation-quality.md): CANDOR's
 * single outcome item ("positive vs negative feelings right now", 1–9)
 * taken once BEFORE and once AFTER a live session. CANDOR found mood
 * improved for 66% of people post-conversation (median +1) — this is the
 * app's therapy-evidence primitive, and the values ride to the server on
 * the stored episode (mood_before on POST /sessions/live, mood_after via a
 * follow-up PATCH — see api/liveSessions.ts).
 *
 * A zustand store (LiveCoachScreen reads/writes it directly, no prop
 * drilling) holding only the CURRENT session's answers — `reset()` clears
 * both for a new session. The last COMPLETED pair also persists locally per
 * session id with the same cross-platform pattern as devModeStore.ts:
 * expo-secure-store on native, localStorage on web, fail-open (a storage
 * failure never blocks the check-in itself — it just isn't remembered).
 */
import { create } from "zustand";
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";

export const MOOD_KEY_PREFIX = "mindshift.moodCheck.v1";

/** One session's answers (either half may be null — a skip, or not yet
 *  answered). */
export interface MoodPair {
  before: number | null;
  after: number | null;
}

export function moodKey(sessionId: string): string {
  const id = (sessionId || "unknown").replace(/[^A-Za-z0-9._-]/g, "_");
  return `${MOOD_KEY_PREFIX}.${id}`;
}

function webStorage(): Storage | null {
  try {
    const g = globalThis as { localStorage?: Storage };
    return g.localStorage ?? null;
  } catch {
    return null;
  }
}

function isMoodValue(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v) && v >= 1 && v <= 9;
}

/** The last completed pair persisted for `sessionId`, or null on garbage /
 *  nothing stored / a storage failure (fail-open — never throws). */
export async function loadMoodPair(sessionId: string): Promise<MoodPair | null> {
  const key = moodKey(sessionId);
  try {
    const raw = Platform.OS === "web" ? (webStorage()?.getItem(key) ?? null) : await SecureStore.getItemAsync(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { before?: unknown; after?: unknown };
    const before = isMoodValue(parsed?.before) ? parsed.before : null;
    const after = isMoodValue(parsed?.after) ? parsed.after : null;
    if (before === null && after === null) return null;
    return { before, after };
  } catch {
    return null;
  }
}

/** Persist `pair` for `sessionId` (fail-open — a write failure is silently
 *  swallowed; the in-memory store already holds the current answer). */
export async function saveMoodPair(sessionId: string, pair: MoodPair): Promise<void> {
  const key = moodKey(sessionId);
  try {
    const raw = JSON.stringify(pair);
    if (Platform.OS === "web") webStorage()?.setItem(key, raw);
    else await SecureStore.setItemAsync(key, raw);
  } catch {
    // Fail-open: the mood check still works for the rest of this session.
  }
}

interface MoodState {
  /** The BEFORE answer for the session in progress (1–9, or null = not
   *  answered / skipped). */
  before: number | null;
  /** The AFTER answer for the session that just ended. */
  after: number | null;
  /** Record the BEFORE answer. `sessionId` is null before a session id
   *  exists (nothing to persist against yet — the value still updates). */
  setBefore: (sessionId: string | null, value: number | null) => void;
  /** Record the AFTER answer (and persist the completed pair, when there is
   *  a session id to key it by). */
  setAfter: (sessionId: string | null, value: number | null) => void;
  /** Clear both — called when a new session starts, so a stale answer from
   *  the previous session never rides along on the next one's POST. */
  reset: () => void;
}

export const useMoodStore = create<MoodState>((set, get) => ({
  before: null,
  after: null,

  setBefore: (sessionId, value) => {
    set({ before: value });
    if (sessionId) void saveMoodPair(sessionId, { before: value, after: get().after });
  },

  setAfter: (sessionId, value) => {
    set({ after: value });
    if (sessionId) void saveMoodPair(sessionId, { before: get().before, after: value });
  },

  reset: () => set({ before: null, after: null }),
}));
