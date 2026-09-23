/**
 * The guest quota's two numbers, as the SERVER states them.
 *
 * Guest mode is bounded by `server/guest_quota.py`: N live sessions per UTC
 * day and M minutes per session, both read from the deploy's environment
 * (`MINDSHIFT_GUEST_MAX_SESSIONS_PER_DAY` / `MINDSHIFT_GUEST_MAX_SESSION_MIN`).
 * The app used to say nothing about either until one of them bit mid-session.
 *
 * Why this store instead of two constants in the UI: a hardcoded "3" in the
 * app is only true until someone changes that env var, and a number the app
 * states but does not enforce is exactly the kind of claim that quietly turns
 * into a lie. So the numbers are LEARNED from the wire — the server's
 * `guest_limit` frame carries `max_sessions_per_day` and `max_session_minutes`
 * (see `_close_ws_guest_limit` in server/audio_pipeline.py) — cached on the
 * device, and shown only once known. Until then the UI says that the limits
 * exist without inventing what they are.
 *
 * Honest limitation, written down rather than glossed: that frame is sent when
 * a limit is REFUSING a session, so a phone that has never hit a limit has
 * never been told the numbers and will show the number-free line. Fixing that
 * properly needs the server to state the quota up front (e.g. on `GET /me` or
 * in `config_ack`), which is a server change this deliberately does not make.
 *
 * Stored device-globally, not per account: the quota belongs to the server,
 * not to a uid, and a guest uid is thrown away on every sign-out anyway — a
 * per-uid key would forget the numbers exactly when a new guest needs them.
 */
import { create } from "zustand";
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";

export const GUEST_LIMITS_KEY = "mindshift.guestLimits.v1";

export interface GuestLimits {
  /** Live sessions one guest account may start per UTC day, or null = the
   *  server has not told this device yet (never a guess). */
  maxSessionsPerDay: number | null;
  /** Minutes one guest session may run, or null = not yet known. */
  maxSessionMinutes: number | null;
}

export const UNKNOWN_GUEST_LIMITS: GuestLimits = {
  maxSessionsPerDay: null,
  maxSessionMinutes: null,
};

/** A whole positive number, or null for anything else (missing, 0, negative,
 *  fractional, NaN, a string, an object). The server sends ints; anything
 *  else on the wire is a bug we refuse to render rather than repeat. */
function positiveInt(value: unknown): number | null {
  return typeof value === "number" &&
    Number.isInteger(value) &&
    value > 0
    ? value
    : null;
}

/**
 * Pull the quota out of an arbitrary server frame (or a parsed cache entry).
 * Each field is independent: a frame carrying only one of them teaches us
 * only that one.
 */
export function readGuestLimits(frame: unknown): GuestLimits {
  if (typeof frame !== "object" || frame === null) return UNKNOWN_GUEST_LIMITS;
  const f = frame as Record<string, unknown>;
  return {
    maxSessionsPerDay: positiveInt(f.max_sessions_per_day),
    maxSessionMinutes: positiveInt(f.max_session_minutes),
  };
}

function webStorage(): Storage | null {
  try {
    const g = globalThis as { localStorage?: Storage };
    return g.localStorage ?? null;
  } catch {
    return null;
  }
}

/** Read the cached numbers. Anything unreadable/unparseable reads as unknown
 *  — the UI's honest default, never a fabricated limit. */
export async function loadGuestLimits(): Promise<GuestLimits> {
  try {
    const raw =
      Platform.OS === "web"
        ? (webStorage()?.getItem(GUEST_LIMITS_KEY) ?? null)
        : await SecureStore.getItemAsync(GUEST_LIMITS_KEY);
    if (!raw) return UNKNOWN_GUEST_LIMITS;
    return readGuestLimits(JSON.parse(raw));
  } catch {
    return UNKNOWN_GUEST_LIMITS;
  }
}

export async function saveGuestLimits(limits: GuestLimits): Promise<void> {
  const body = JSON.stringify({
    max_sessions_per_day: limits.maxSessionsPerDay,
    max_session_minutes: limits.maxSessionMinutes,
  });
  try {
    if (Platform.OS === "web") webStorage()?.setItem(GUEST_LIMITS_KEY, body);
    else await SecureStore.setItemAsync(GUEST_LIMITS_KEY, body);
  } catch {
    // Fail-open: the numbers still apply to this launch.
  }
}

interface GuestLimitsState extends GuestLimits {
  /** Load the cached numbers (fail-open to unknown). */
  hydrate: () => Promise<void>;
  /** Learn from a server frame. Fields the frame does not carry are left
   *  alone, so a partial frame can never erase a number we already know. */
  learnFromServer: (frame: unknown) => void;
}

export const useGuestLimitsStore = create<GuestLimitsState>((set, get) => ({
  ...UNKNOWN_GUEST_LIMITS,

  hydrate: async () => {
    const stored = await loadGuestLimits();
    // Never downgrade something already learned this launch (a frame can
    // arrive before a slow storage read resolves).
    const { maxSessionsPerDay, maxSessionMinutes } = get();
    set({
      maxSessionsPerDay: maxSessionsPerDay ?? stored.maxSessionsPerDay,
      maxSessionMinutes: maxSessionMinutes ?? stored.maxSessionMinutes,
    });
  },

  learnFromServer: (frame) => {
    const heard = readGuestLimits(frame);
    if (heard.maxSessionsPerDay === null && heard.maxSessionMinutes === null) {
      return; // nothing usable on the wire — leave what we have
    }
    const current = get();
    const next: GuestLimits = {
      maxSessionsPerDay: heard.maxSessionsPerDay ?? current.maxSessionsPerDay,
      maxSessionMinutes: heard.maxSessionMinutes ?? current.maxSessionMinutes,
    };
    if (
      next.maxSessionsPerDay === current.maxSessionsPerDay &&
      next.maxSessionMinutes === current.maxSessionMinutes
    ) {
      return; // unchanged — no re-render, no needless write
    }
    set(next);
    void saveGuestLimits(next);
  },
}));
