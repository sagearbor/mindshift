/**
 * Keep a live session's microphone audio on the phone as ONE growing WAV.
 *
 * The Live Coach mic PCM is already in JS (useAudioStream's handleAudioBuffer
 * produces 16 kHz mono int16 frames for the fast loop and the WebSocket); the
 * keeper is a THIRD consumer of those same frames — it never opens a mic of
 * its own (a second native recorder would collide with the Android
 * SpeechRecognizer). Frames accumulate in memory and are appended to
 * `<dir>/<session_id>.wav` on a flush cadence, with the RIFF/data sizes
 * patched after every append, so at any instant the file on disk is a valid
 * WAV of everything flushed so far (the v2 recorder's crash-loss bound).
 *
 * After the session is saved (`POST /sessions/live` → episode id) the hook
 * uploads the file to `POST /sessions/{id}/audio` (api/liveSessions.ts) and
 * deletes it; a session the server never stored is discarded, never
 * uploaded. Everything platform-specific sits behind the RecorderFs seam so
 * the whole thing is unit-tested over MemoryFs.
 */
import { Platform } from "react-native";
import { Paths } from "expo-file-system";
import type { RecorderFs } from "../recorder/types";
import { ExpoRecorderFs } from "../recorder/expoFs";
import { DEFAULT_FLUSH_MS } from "../recorder/streamSession";
import { runPreflight } from "../recorder/preflight";
import {
  DATA_SIZE_OFFSET,
  RIFF_SIZE_OFFSET,
  WAV_HEADER_BYTES,
  buildWavHeader,
  dataSizeBytes,
  int16ToBytes,
  riffSizeBytes,
} from "../recorder/wav";

/** The hook hands us 16 kHz mono int16 — the server's native format. */
export const LIVE_AUDIO_SAMPLE_RATE = 16000;
/** Cache subdirectory holding one WAV per session. */
export const LIVE_AUDIO_DIR_NAME = "live-audio";
/** On-disk cost of one second at 16 kHz mono s16le, for the storage check. */
export const LIVE_AUDIO_BYTES_PER_SECOND = LIVE_AUDIO_SAMPLE_RATE * 2;
/** Plan for an hour-long session when checking free space. */
export const LIVE_AUDIO_PLANNED_SECONDS = 60 * 60;

/** What `finish()` hands back: the file the hook uploads. */
export interface LiveAudioKept {
  uri: string;
  /** Whole-file size (header + PCM). */
  bytes: number;
  /** Seconds of audio in the file, to one decimal. */
  seconds: number;
}

export interface LiveAudioKeeper {
  readonly uri: string;
  /** Whole-file size so far (44 + 2 × samples), including unflushed audio. */
  readonly bytes: number;
  /** Seconds of audio so far, including unflushed audio. */
  readonly seconds: number;
  /** Take one frame of 16 kHz mono int16. Never throws — a disk failure
   *  closes the keeper and is reported by `finish()`. */
  append(int16: Int16Array): void;
  /** Write everything pending and patch the header (crash-safe point). */
  flush(): void;
  /** Flush the tail and hand back the finished file; null when nothing was
   *  ever appended (and no file was left behind). Rejects when a disk write
   *  failed during the session. Idempotent. */
  finish(): Promise<LiveAudioKept | null>;
  /** Drop everything and delete the file (if any). Safe at any time,
   *  including after `finish()` — the hook's "uploaded, clean up" path. */
  discard(): void;
}

export interface LiveAudioKeeperOptions {
  fs: RecorderFs;
  /** Directory holding the per-session WAVs (no trailing slash). */
  dir: string;
  sessionId: string;
  sampleRate?: number;
  /** Disk flush cadence; 0 flushes on every append. */
  flushMs?: number;
  now?: () => number;
}

/** `<dir>/<session_id>.wav`, with anything path-hostile in the id replaced. */
export function liveAudioFileUri(dir: string, sessionId: string): string {
  const safe = sessionId.replace(/[^A-Za-z0-9._-]/g, "_") || "session";
  return `${dir.endsWith("/") ? dir.slice(0, -1) : dir}/${safe}.wav`;
}

export function createLiveAudioKeeper(opts: LiveAudioKeeperOptions): LiveAudioKeeper {
  const fs = opts.fs;
  const sampleRate = opts.sampleRate ?? LIVE_AUDIO_SAMPLE_RATE;
  const flushMs = opts.flushMs ?? DEFAULT_FLUSH_MS;
  const now = opts.now ?? Date.now;
  const uri = liveAudioFileUri(opts.dir, opts.sessionId);

  let pending: Int16Array[] = [];
  let pendingSamples = 0;
  let samplesOnDisk = 0;
  let fileOpen = false;
  let closed = false;
  let error: string | null = null;
  let lastFlushAt = now();
  let finished: Promise<LiveAudioKept | null> | null = null;

  const totalSamples = () => samplesOnDisk + pendingSamples;

  const mergePending = (): Int16Array => {
    const merged = new Int16Array(pendingSamples);
    let cursor = 0;
    for (const chunk of pending) {
      merged.set(chunk, cursor);
      cursor += chunk.length;
    }
    pending = [];
    pendingSamples = 0;
    return merged;
  };

  const fail = (err: unknown) => {
    error = err instanceof Error ? err.message : String(err);
    closed = true;
    pending = [];
    pendingSamples = 0;
  };

  const flush = () => {
    lastFlushAt = now();
    if (error || pendingSamples === 0) return;
    const merged = mergePending();
    try {
      if (!fileOpen) {
        fs.ensureDir(opts.dir);
        fs.writeBytes(uri, buildWavHeader(sampleRate, 0));
        fileOpen = true;
      }
      fs.appendBytes(uri, int16ToBytes(merged));
      samplesOnDisk += merged.length;
      // Patch the sizes so the file is a valid WAV of everything flushed so
      // far (a crash between the append and this patch loses one batch).
      const dataBytes = samplesOnDisk * 2;
      fs.writeBytesAt(uri, RIFF_SIZE_OFFSET, riffSizeBytes(dataBytes));
      fs.writeBytesAt(uri, DATA_SIZE_OFFSET, dataSizeBytes(dataBytes));
    } catch (err) {
      fail(err);
    }
  };

  const discard = () => {
    closed = true;
    pending = [];
    pendingSamples = 0;
    samplesOnDisk = 0; // a later finish() must not describe a deleted file
    if (fileOpen || fs.exists(uri)) {
      try {
        fs.deleteRecursive(uri);
      } catch {
        // Best effort — a cache file that won't delete is not worth failing on.
      }
    }
    fileOpen = false;
  };

  return {
    uri,
    get bytes() {
      return WAV_HEADER_BYTES + totalSamples() * 2;
    },
    get seconds() {
      return Math.round((totalSamples() / sampleRate) * 10) / 10;
    },
    append(int16) {
      if (closed || int16.length === 0) return;
      // Copy: the hook reuses/transfers frame buffers to other consumers.
      pending.push(int16.slice());
      pendingSamples += int16.length;
      if (now() - lastFlushAt >= flushMs) flush();
    },
    flush,
    finish() {
      if (finished) return finished;
      finished = (async () => {
        flush();
        closed = true;
        if (error) throw new Error(`keeping the session audio failed: ${error}`);
        if (samplesOnDisk === 0) {
          discard();
          return null;
        }
        return {
          uri,
          bytes: WAV_HEADER_BYTES + samplesOnDisk * 2,
          seconds: Math.round((samplesOnDisk / sampleRate) * 10) / 10,
        };
      })();
      return finished;
    },
    discard,
  };
}

/** Why keeping audio should NOT start on this device right now (disk nearly
 *  full), or null when there is room. Reuses the recorder's preflight math:
 *  an unknown free-space figure is a warning there, never a block. */
export function keepAudioStorageProblem(fs: RecorderFs): string | null {
  const report = runPreflight({
    freeBytes: fs.freeBytes(),
    batteryLevel: null,
    bytesPerSecond: LIVE_AUDIO_BYTES_PER_SECOND,
    plannedSeconds: LIVE_AUDIO_PLANNED_SECONDS,
  });
  return report.canStart ? null : report.storage.message;
}

/** Result of opening a keeper for a session: either a keeper, or the honest
 *  reason none could be opened (web, disk full, file system unavailable). */
export type LiveAudioKeeperOpen =
  | { keeper: LiveAudioKeeper; reason: null }
  | { keeper: null; reason: string };

export type LiveAudioKeeperFactory = (sessionId: string) => LiveAudioKeeperOpen;

/** Production factory: one WAV per session under `Paths.cache/live-audio/`.
 *  Never throws — the session must start whether or not audio can be kept. */
export function openDefaultLiveAudioKeeper(sessionId: string): LiveAudioKeeperOpen {
  if (Platform.OS === "web") {
    return { keeper: null, reason: "keeping audio isn't available in the browser" };
  }
  try {
    const fs = new ExpoRecorderFs();
    const base = Paths.cache.uri;
    const dir = `${base.endsWith("/") ? base.slice(0, -1) : base}/${LIVE_AUDIO_DIR_NAME}`;
    const problem = keepAudioStorageProblem(fs);
    if (problem) return { keeper: null, reason: problem };
    return { keeper: createLiveAudioKeeper({ fs, dir, sessionId }), reason: null };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { keeper: null, reason: `could not open a local audio file (${msg})` };
  }
}
