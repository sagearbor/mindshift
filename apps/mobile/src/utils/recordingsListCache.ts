import { Platform } from "react-native";
import type {
  RecordingSummary,
  RecordingsListResult,
  SharedRecordingSummary,
} from "../api/client";

/**
 * Persisted cache of the LAST successful `GET /recordings` response, keyed
 * PER ACCOUNT, so the Recordings list opens instantly from what's already on
 * the phone and only *refreshes* from the network in the background
 * (stale-while-revalidate). Before this the list fetched on every open and
 * sat on a spinner for the whole cold start of the scale-to-zero server.
 *
 * Storage: NOT SecureStore (a multi-KB JSON blob doesn't belong in the
 * keychain, and its values are size-capped). Native keeps one small JSON
 * file per account under `Paths.cache` (the OS-reclaimable cache dir, same
 * home as mediaCache.ts — if the OS clears it, the next open just shows the
 * spinner like today) using expo-file-system's SDK-57 SYNCHRONOUS
 * `textSync()`/`write()` so the cached list can be read during the screen's
 * FIRST render — no async gap, no flash. Web uses localStorage. A per-process
 * memory copy sits in front of both so re-opens within one app run never
 * touch disk at all.
 *
 * Fail-open everywhere: a storage error, a missing file, or a corrupt /
 * unexpectedly-shaped blob reads as "no cache" (and the screen behaves
 * exactly as it did before: spinner, then the network result). A failed
 * write is ignored — the list still renders from the network. expo-file-
 * system is lazily required (mediaCache.ts / avatarStore.ts pattern) so the
 * module never touches native code at import time.
 */

export const RECORDINGS_CACHE_KEY_PREFIX = "mindshift.recordingsList.v1";
const CACHE_SUBDIR = "recordings-list";

/** What we persist: the two list sections plus WHEN they were fetched
 *  (epoch ms), so the screen can decide whether a foreground return warrants
 *  a refresh. */
export interface CachedRecordingsList {
  recordings: RecordingSummary[];
  sharedWithMe: SharedRecordingSummary[];
  fetched_at: number;
}

/** Per-account key: a Firebase uid is already key-safe; anything else
 *  ("anon", an email) is sanitized the same way modePrefs.ts's modeKey does. */
export function recordingsCacheKey(userId: string | null | undefined): string {
  const id = (userId || "anon").replace(/[^A-Za-z0-9._-]/g, "_");
  return `${RECORDINGS_CACHE_KEY_PREFIX}.${id}`;
}

// Per-process memory copy (key → blob). Populated on first read/write; the
// screen's first render within an app run after that is zero-I/O.
const memory = new Map<string, CachedRecordingsList>();

// Bumped whenever a client-side mutation changes what the list would show
// (delete / rename / share / a new recording landing). A cached blob fetched
// BEFORE this instant is considered stale regardless of its age, so the next
// foreground return refreshes it even inside the normal max-age window.
let staleSince = 0;

/** Mark every cached list as needing a refresh (see `staleSince`). Called by
 *  the recordings mutation helpers in api/client.ts on success. */
export function markRecordingsListStale(): void {
  staleSince = Date.now();
}

/** True when a list fetched at `fetchedAt` (epoch ms) predates the last
 *  known list mutation. */
export function isRecordingsListStale(fetchedAt: number): boolean {
  return fetchedAt < staleSince;
}

/** Test-only: forget the in-memory copy and the stale mark (disk/localStorage
 *  are left alone; tests own their own mocks for those). */
export function resetRecordingsCacheMemory(): void {
  memory.clear();
  staleSince = 0;
}

function webStorage(): Storage | null {
  try {
    const g = globalThis as { localStorage?: Storage };
    return g.localStorage ?? null;
  } catch {
    return null;
  }
}

/** The (not-necessarily-existing) cache File for `key` on native. */
function nativeFile(key: string) {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { Directory, File, Paths } = require("expo-file-system");
  const dir = new Directory(Paths.cache, CACHE_SUBDIR);
  return { dir, file: new File(dir, `${key}.json`) };
}

function readRaw(key: string): string | null {
  if (Platform.OS === "web") return webStorage()?.getItem(key) ?? null;
  const { file } = nativeFile(key);
  return file.exists ? file.textSync() : null;
}

function writeRaw(key: string, text: string): void {
  if (Platform.OS === "web") {
    webStorage()?.setItem(key, text);
    return;
  }
  const { dir, file } = nativeFile(key);
  if (!dir.exists) dir.create({ intermediates: true, idempotent: true });
  file.write(text);
}

function deleteRaw(key: string): void {
  if (Platform.OS === "web") {
    webStorage()?.removeItem(key);
    return;
  }
  const { file } = nativeFile(key);
  if (file.exists) file.delete();
}

/** Validate a parsed blob's SHAPE (not every row field — a row the server
 *  wrote is trusted as it was). Anything else is treated as corrupt. */
function isCachedList(value: unknown): value is CachedRecordingsList {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    Array.isArray(v.recordings) &&
    Array.isArray(v.sharedWithMe) &&
    typeof v.fetched_at === "number" &&
    Number.isFinite(v.fetched_at)
  );
}

/**
 * SYNCHRONOUS cache read for `userId` — memory first, then disk /
 * localStorage. Null on a miss, a storage error, or a corrupt blob (which is
 * also best-effort deleted so it isn't re-parsed on every open).
 */
export function readRecordingsCache(
  userId: string | null | undefined,
): CachedRecordingsList | null {
  const key = recordingsCacheKey(userId);
  const hit = memory.get(key);
  if (hit) return hit;
  let raw: string | null;
  try {
    raw = readRaw(key);
  } catch {
    return null;
  }
  if (raw === null) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isCachedList(parsed)) throw new Error("unexpected shape");
    memory.set(key, parsed);
    return parsed;
  } catch {
    try {
      deleteRaw(key);
    } catch {
      // Best-effort — a corrupt blob we can't delete still just reads as null.
    }
    return null;
  }
}

/**
 * Persist a fresh list result for `userId` (memory + disk). Never throws —
 * a write failure only means the next open fetches like today. Returns the
 * blob that was stored (handy for callers that keep `fetched_at`).
 */
export function writeRecordingsCache(
  userId: string | null | undefined,
  result: RecordingsListResult,
  fetchedAt: number = Date.now(),
): CachedRecordingsList {
  const key = recordingsCacheKey(userId);
  const blob: CachedRecordingsList = {
    recordings: result.recordings,
    sharedWithMe: result.sharedWithMe,
    fetched_at: fetchedAt,
  };
  memory.set(key, blob);
  try {
    writeRaw(key, JSON.stringify(blob));
  } catch {
    // Fail-open — see module doc comment.
  }
  return blob;
}

/** Drop the cache for `userId` (memory + disk). Never throws. */
export function clearRecordingsCache(userId: string | null | undefined): void {
  const key = recordingsCacheKey(userId);
  memory.delete(key);
  try {
    deleteRaw(key);
  } catch {
    // Fail-open.
  }
}

/** Stable-identity merge for the screen: when the fresh list is
 *  row-for-row identical to what's already rendered, hand back the SAME
 *  array so React skips a pointless re-render (and the ScrollView doesn't
 *  so much as blink). Otherwise the fresh list wins as-is — the server
 *  orders newest-first, so a new recording appears at the top and a deleted
 *  one simply drops out; React keys (`rec.id`) keep the surviving rows and
 *  the scroll position where they were. */
export function mergeRecordingsList<T extends RecordingSummary>(
  prev: T[],
  next: T[],
): T[] {
  if (prev === next) return prev;
  if (prev.length !== next.length) return next;
  for (let i = 0; i < prev.length; i++) {
    if (prev[i].id !== next[i].id) return next;
    if (JSON.stringify(prev[i]) !== JSON.stringify(next[i])) return next;
  }
  return prev;
}
