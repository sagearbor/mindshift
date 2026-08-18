import { Platform } from "react-native";
import type { MediaType } from "../api/client";

/**
 * Local disk cache for replayed recording media (Task: "cache replayed media
 * locally instead of re-downloading every play", 2026-08-18 brief).
 *
 * The problem: `getRecordingMediaUrl`/`/media` mints a fresh signed URL on
 * EVERY call and the server re-reads the blob from GCS every request — so
 * replaying (or even re-opening) the same recording re-fetches the whole
 * file from the network every single time, with real GCS egress + Cloud Run
 * bandwidth cost. The stored derivative for a given `recording_id` is
 * immutable for the life of that recording (recordings_store.py's own
 * docstring: re-analysis reuses the SAME stored audio/video, never
 * replacing it) — so caching by `recording_id` alone, forever, is safe.
 *
 * Design (deliberately does NOT change first-play behavior): ReplayScreen
 * still streams the remote signed URL immediately on first play — that's
 * what makes progressive HTTP Range playback work for large videos. This
 * module only adds two things around that unchanged path:
 *   1. `getCachedMediaUri` — a synchronous, zero-network check the caller
 *      makes BEFORE fetching a signed URL at all. A hit means the caller can
 *      skip the network entirely and hand the player a local `file://` uri.
 *   2. `cacheMediaInBackground` — fire-and-forget: after the FIRST play
 *      starts streaming from the network, kick off a low-priority local
 *      download of the SAME url so the NEXT play/replay gets a cache hit.
 *      Never blocks, never throws — a failed background cache just means
 *      the next replay behaves exactly as it does today (a fresh fetch).
 *
 * Native-only: `Platform.OS === "web"` short-circuits every function to a
 * no-op/null. react-native-web has no expo-file-system disk-cache
 * equivalent worth building — browsers already have their own HTTP cache —
 * so web keeps exactly today's remote-streaming-only behavior.
 *
 * Cache key: `recording_id` (+ media-type extension), matched EXACTLY — a
 * different id is always a miss, even if some other file happens to already
 * exist locally under a different name (there is no fuzzy/prefix matching).
 * expo-file-system's modern `Directory`/`File`/`Paths` API is used (the
 * SAME pattern as `avatarStore.ts`/`expoFs.ts` in this codebase), lazily
 * required so this module never touches native code at import time — same
 * reasoning as `avatarStore.ts`'s `deleteAvatarFile` and
 * `VoiceTrainingFlow.ts`'s `saveWav`.
 *
 * Cache location: `Paths.cache` (the OS-reclaimable cache dir, NOT the
 * document dir) under a dedicated "media" subdirectory — this is disposable,
 * re-fetchable data, not user data that must survive an OS-triggered cache
 * clear. If the OS clears it, the next play just re-fetches, same as today.
 *
 * Eviction: NONE yet (deliberately out of scope for this task — see the
 * brief/report). A cache-size cap / LRU eviction across all cached
 * recordings is a ledgered follow-up.
 */

const CACHE_SUBDIR = "media";

/** `.mp4` for video, `.m4a` for audio — matches this codebase's stored
 *  derivative types (server/main.py's media_type). */
function extensionFor(mediaType: MediaType): string {
  return mediaType === "video" ? ".mp4" : ".m4a";
}

/** The (not-necessarily-existing) local File for this recording's cached
 *  media. Lazy-required — see module doc comment. */
function cacheFile(recordingId: string, mediaType: MediaType) {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { Directory, File, Paths } = require("expo-file-system");
  const dir = new Directory(Paths.cache, CACHE_SUBDIR);
  return { File, dir, file: new File(dir, `${recordingId}${extensionFor(mediaType)}`) };
}

/**
 * Zero-network cache lookup. Returns the local `file://` uri if this
 * recording's media is already cached, or null on a miss (nothing cached
 * yet, wrong platform, or any file-system error — fail-open to "not
 * cached" so the caller always falls back to its existing network path).
 */
export function getCachedMediaUri(
  recordingId: string,
  mediaType: MediaType,
): string | null {
  if (Platform.OS === "web") return null;
  try {
    const { file } = cacheFile(recordingId, mediaType);
    return file.exists ? file.uri : null;
  } catch {
    return null;
  }
}

/**
 * Fire-and-forget background download of `remoteUrl` into the local cache
 * for `recordingId`, so the NEXT `getCachedMediaUri` call for the same
 * recording is a hit. Returns immediately (does not return a promise the
 * caller is expected to await) — never interferes with playback that's
 * already streaming from `remoteUrl`. Best-effort: any failure (network,
 * disk, permissions) is swallowed, same fail-open pattern as
 * `avatarStore.ts`'s `deleteAvatarFile`.
 */
export function cacheMediaInBackground(
  recordingId: string,
  mediaType: MediaType,
  remoteUrl: string,
): void {
  if (Platform.OS === "web") return;
  void (async () => {
    try {
      const { File, dir, file } = cacheFile(recordingId, mediaType);
      if (!dir.exists) dir.create({ intermediates: true, idempotent: true });
      await File.downloadFileAsync(remoteUrl, file, { idempotent: true });
    } catch {
      // Best-effort — a failed background cache just means the next replay
      // re-fetches from the network, exactly like today (no regression).
    }
  })();
}

/**
 * Best-effort delete of any cached media for `recordingId` — hooked into
 * the delete-recording flow (RecordingsScreen's `confirmDelete`) so a
 * removed recording doesn't leave an orphaned local copy behind. Tries both
 * known extensions since the caller doesn't always have the media type
 * handy at delete time; a missing file for either extension is silently
 * skipped. Never throws (mirrors `avatarStore.ts`'s `deleteAvatarFile`).
 */
export async function deleteCachedMedia(recordingId: string): Promise<void> {
  if (Platform.OS === "web") return;
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { Directory, File, Paths } = require("expo-file-system");
    const dir = new Directory(Paths.cache, CACHE_SUBDIR);
    for (const mediaType of ["video", "audio"] as const) {
      const file = new File(dir, `${recordingId}${extensionFor(mediaType)}`);
      if (file.exists) file.delete();
    }
  } catch {
    // Best-effort — see doc comment above.
  }
}
