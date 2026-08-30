/**
 * On-disk journal files for the Journal mode (journalLoop.ts writes into
 * them; journalRecorder.ts rotates and uploads them).
 *
 * One journal file is ONE growing WAV (16 kHz mono s16le, reusing the
 * crash-safe keeper from liveAudioKeeper.ts — the header is patched after
 * every flush, so a file left behind by a process death is still a valid
 * WAV of everything flushed) plus a JSON sidecar next to it:
 *
 *   <cache>/journal/journal-<epoch ms>.wav
 *   <cache>/journal/journal-<epoch ms>.json   { version, started_at, ended_at,
 *                                               sample_rate, segments: [...] }
 *
 * `segments` is one entry per kept stretch of the owner's voice —
 * `{ start_wall_iso, offset_s, duration_s, lead_s, speech_s, score, basis }`
 * — `offset_s` being where the stretch (context included) starts inside the
 * WAV, so the file's audio can be put back on the day's clock later.
 *
 * Files stay in the directory until an upload succeeds (`remove`); a
 * relaunch finds them with `listClosed` and the recorder retries them on
 * its next boundary. Nothing is ever deleted before the server has it.
 * Everything platform-specific sits behind the RecorderFs seam so the whole
 * thing is unit-tested over MemoryFs.
 */
import { Platform } from "react-native";
import type { MatchBasis } from "./speakerId";
import { Paths } from "expo-file-system";
import type { RecorderFs } from "../recorder/types";
import { ExpoRecorderFs } from "../recorder/expoFs";
import { WAV_HEADER_BYTES } from "../recorder/wav";
import {
  createLiveAudioKeeper,
  keepAudioStorageProblem,
  LIVE_AUDIO_SAMPLE_RATE,
  type LiveAudioKeeper,
} from "./liveAudioKeeper";
import type { KeptSegmentMeta } from "./journalLoop";

export const JOURNAL_DIR_NAME = "journal";
export const JOURNAL_FILE_PREFIX = "journal-";
export const JOURNAL_SIDECAR_VERSION = 1;

/** One kept stretch, as the sidecar records it (snake_case: written as-is). */
export interface JournalSegment {
  /** Wall clock (ISO) at which the owner's speech began. */
  start_wall_iso: string;
  /** Seconds into the WAV where the kept stretch (lead context included) starts. */
  offset_s: number;
  /** Seconds of kept audio (context included). */
  duration_s: number;
  /** Seconds of leading context inside the stretch. */
  lead_s: number;
  /** Seconds of the speech itself. */
  speech_s: number;
  score: number | null;
  basis: MatchBasis | null;
}

export interface JournalSidecar {
  version: number;
  started_at: string;
  ended_at: string | null;
  sample_rate: number;
  segments: JournalSegment[];
}

/** A journal file that is finished on disk and waiting to be uploaded. */
export interface ClosedJournalFile {
  uri: string;
  sidecarUri: string;
  startedAt: string;
  endedAt: string;
  /** Whole-file size (header + PCM). */
  bytes: number;
  /** Seconds of audio in the file. */
  seconds: number;
  /** Kept stretches recorded in the sidecar (null when the sidecar is
   *  missing — a file left behind by a crash before its first flush). */
  segments: JournalSegment[] | null;
}

export interface OpenJournalFile {
  readonly uri: string;
  readonly sidecarUri: string;
  readonly startedAtMs: number;
  /** Seconds of audio so far (including unflushed). */
  readonly seconds: number;
  /** Whole-file size so far (including unflushed). */
  readonly bytes: number;
  readonly segmentCount: number;
  /** Append one kept stretch (16 kHz mono int16) and record it in the sidecar. */
  append(pcm: Int16Array, meta: KeptSegmentMeta): void;
  /** Flush pending audio to disk (crash-safe point). */
  flush(): void;
  /** Finish the file. Null when nothing was ever kept (the empty file and
   *  sidecar are removed). `endedAtMs` is the wall time the caller decided
   *  to close (default: now). Idempotent. */
  close(endedAtMs?: number): Promise<ClosedJournalFile | null>;
}

export interface JournalStore {
  readonly dir: string;
  open(startWallMs: number): OpenJournalFile;
  /** Every finished journal file in the directory (oldest first) that is
   *  not currently open — this run's closed files and any earlier run's
   *  leftovers. */
  listClosed(): ClosedJournalFile[];
  /** Delete a file (and its sidecar) — only after the server has it. */
  remove(file: ClosedJournalFile): void;
}

export interface JournalStoreOptions {
  fs: RecorderFs;
  /** Directory holding the journal files (no trailing slash). */
  dir: string;
  now?: () => number;
  /** Disk flush cadence for the WAV; 0 flushes on every append. */
  flushMs?: number;
  sampleRate?: number;
}

function trimSlash(uri: string): string {
  return uri.endsWith("/") ? uri.slice(0, -1) : uri;
}

export function journalFileId(startWallMs: number): string {
  return `${JOURNAL_FILE_PREFIX}${Math.round(startWallMs)}`;
}

/** The start time encoded in a journal file name, or null. */
export function journalStartFromName(name: string): number | null {
  const m = /^journal-(\d+)\.(wav|json)$/.exec(name);
  return m ? Number(m[1]) : null;
}

function round3(x: number): number {
  return Math.round(x * 1000) / 1000;
}

function parseSidecar(text: string): JournalSidecar | null {
  try {
    const data = JSON.parse(text) as Partial<JournalSidecar> | null;
    if (!data || typeof data !== "object" || !Array.isArray(data.segments)) return null;
    return {
      version: typeof data.version === "number" ? data.version : JOURNAL_SIDECAR_VERSION,
      started_at: typeof data.started_at === "string" ? data.started_at : "",
      ended_at: typeof data.ended_at === "string" ? data.ended_at : null,
      sample_rate: typeof data.sample_rate === "number" ? data.sample_rate : LIVE_AUDIO_SAMPLE_RATE,
      segments: data.segments as JournalSegment[],
    };
  } catch {
    return null;
  }
}

export function createJournalStore(opts: JournalStoreOptions): JournalStore {
  const fs = opts.fs;
  const dir = trimSlash(opts.dir);
  const now = opts.now ?? Date.now;
  const sampleRate = opts.sampleRate ?? LIVE_AUDIO_SAMPLE_RATE;
  const openUris = new Set<string>();

  const writeSidecar = (uri: string, sidecar: JournalSidecar) => {
    fs.ensureDir(dir);
    fs.writeText(uri, JSON.stringify(sidecar));
  };

  const open = (startWallMs: number): OpenJournalFile => {
    const id = journalFileId(startWallMs);
    const keeper: LiveAudioKeeper = createLiveAudioKeeper({
      fs,
      dir,
      sessionId: id,
      sampleRate,
      flushMs: opts.flushMs,
      now,
    });
    const sidecarUri = `${dir}/${id}.json`;
    const sidecar: JournalSidecar = {
      version: JOURNAL_SIDECAR_VERSION,
      started_at: new Date(startWallMs).toISOString(),
      ended_at: null,
      sample_rate: sampleRate,
      segments: [],
    };
    openUris.add(keeper.uri);
    let closed: Promise<ClosedJournalFile | null> | null = null;
    let sidecarError: string | null = null;

    return {
      uri: keeper.uri,
      sidecarUri,
      startedAtMs: startWallMs,
      get seconds() {
        return keeper.seconds;
      },
      get bytes() {
        return keeper.bytes;
      },
      get segmentCount() {
        return sidecar.segments.length;
      },
      append(pcm, meta) {
        if (closed || pcm.length === 0) return;
        // The offset is the audio already in the file BEFORE this stretch.
        const offset = keeper.seconds;
        keeper.append(pcm);
        // Flush at once so the audio the sidecar describes is on disk with it.
        keeper.flush();
        sidecar.segments.push({
          start_wall_iso: new Date(meta.startWallMs).toISOString(),
          offset_s: round3(offset),
          duration_s: round3(pcm.length / sampleRate),
          lead_s: round3(meta.leadSeconds),
          speech_s: round3(meta.speechSeconds),
          score: meta.score === null ? null : Math.round(meta.score * 1e4) / 1e4,
          basis: meta.basis,
        });
        try {
          writeSidecar(sidecarUri, sidecar);
        } catch (err) {
          // The audio is the record; a sidecar that won't write is reported
          // at close and the file still uploads.
          sidecarError = err instanceof Error ? err.message : String(err);
        }
      },
      flush() {
        keeper.flush();
      },
      close(endedAtMs?: number) {
        if (closed) return closed;
        const endedAt = endedAtMs ?? now();
        closed = (async () => {
          openUris.delete(keeper.uri);
          let kept: Awaited<ReturnType<LiveAudioKeeper["finish"]>>;
          try {
            kept = await keeper.finish();
          } catch (err) {
            keeper.discard();
            try {
              if (fs.exists(sidecarUri)) fs.deleteRecursive(sidecarUri);
            } catch {
              // Best effort.
            }
            throw err;
          }
          if (!kept) {
            try {
              if (fs.exists(sidecarUri)) fs.deleteRecursive(sidecarUri);
            } catch {
              // Best effort — an empty sidecar in the cache is harmless.
            }
            return null;
          }
          sidecar.ended_at = new Date(endedAt).toISOString();
          try {
            writeSidecar(sidecarUri, sidecar);
          } catch (err) {
            sidecarError = err instanceof Error ? err.message : String(err);
          }
          if (sidecarError) console.warn("[journal] sidecar write failed:", sidecarError);
          return {
            uri: kept.uri,
            sidecarUri,
            startedAt: sidecar.started_at,
            endedAt: sidecar.ended_at,
            bytes: kept.bytes,
            seconds: kept.seconds,
            segments: [...sidecar.segments],
          };
        })();
        return closed;
      },
    };
  };

  const listClosed = (): ClosedJournalFile[] => {
    let names: string[];
    try {
      if (!fs.exists(dir)) return [];
      names = fs.listFileNames(dir);
    } catch {
      return [];
    }
    const out: ClosedJournalFile[] = [];
    for (const name of names) {
      if (!name.endsWith(".wav")) continue;
      const startMs = journalStartFromName(name);
      if (startMs === null) continue;
      const uri = `${dir}/${name}`;
      if (openUris.has(uri)) continue;
      const bytes = fs.sizeOf(uri);
      if (bytes === null || bytes <= WAV_HEADER_BYTES) {
        // Nothing but a header (or unreadable): not worth a server round trip.
        try {
          fs.deleteRecursive(uri);
          const stray = `${dir}/${name.slice(0, -4)}.json`;
          if (fs.exists(stray)) fs.deleteRecursive(stray);
        } catch {
          // Best effort.
        }
        continue;
      }
      const sidecarUri = `${dir}/${name.slice(0, -4)}.json`;
      let sidecar: JournalSidecar | null = null;
      try {
        if (fs.exists(sidecarUri)) sidecar = parseSidecar(fs.readText(sidecarUri));
      } catch {
        sidecar = null;
      }
      const seconds = Math.round(((bytes - WAV_HEADER_BYTES) / 2 / sampleRate) * 10) / 10;
      const startedAt = sidecar?.started_at || new Date(startMs).toISOString();
      const endedAt = sidecar?.ended_at || new Date(startMs + seconds * 1000).toISOString();
      out.push({ uri, sidecarUri, startedAt, endedAt, bytes, seconds, segments: sidecar?.segments ?? null });
    }
    out.sort((a, b) => (a.startedAt < b.startedAt ? -1 : a.startedAt > b.startedAt ? 1 : 0));
    return out;
  };

  const remove = (file: ClosedJournalFile) => {
    for (const uri of [file.uri, file.sidecarUri]) {
      try {
        if (fs.exists(uri)) fs.deleteRecursive(uri);
      } catch {
        // A cache file that won't delete is not worth failing on; it is
        // re-listed (and re-uploaded — the server de-duplicates nothing,
        // so this is logged) next boundary.
        console.warn("[journal] could not delete", uri);
      }
    }
  };

  return { dir, open, listClosed, remove };
}

/** Result of opening the production store: either a store, or the honest
 *  reason none could be opened (web, disk full, file system unavailable). */
export type JournalStoreOpen =
  | { store: JournalStore; reason: null }
  | { store: null; reason: string };

/** Production store: `Paths.cache/journal/`. Never throws. */
export function openDefaultJournalStore(): JournalStoreOpen {
  if (Platform.OS === "web") {
    return { store: null, reason: "the journal isn't available in the browser" };
  }
  try {
    const fs = new ExpoRecorderFs();
    const dir = `${trimSlash(Paths.cache.uri)}/${JOURNAL_DIR_NAME}`;
    const problem = keepAudioStorageProblem(fs);
    if (problem) return { store: null, reason: problem };
    return { store: createJournalStore({ fs, dir }), reason: null };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { store: null, reason: `could not open the journal folder (${msg})` };
  }
}
