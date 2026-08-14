/**
 * Shared contracts for the crash-hardened audio-only recorder.
 *
 * The design principle throughout: every finished segment is an on-disk fact
 * that survives a process death. The engine (segmentedSession), the persistence
 * layer (sessionStore) and the platform adapters (expoFs / expoRecorderPort)
 * talk to each other only through these interfaces so every piece is testable
 * without native modules.
 */

/** The two segment container formats we record.
 *
 *  - `adts` (Android): raw AAC in an ADTS stream — segments are valid when
 *    byte-concatenated, so stitching is a plain join.
 *  - `wav` (iOS): linear PCM in a WAV container — AVAudioRecorder cannot write
 *    ADTS, but WAV segments stitch trivially (strip headers, patch sizes).
 */
export type SegmentContainer = "adts" | "wav";

/** Per-platform recording format facts, including the honest storage-cost rate
 *  the preflight check uses. */
export interface SegmentFormat {
  format: SegmentContainer;
  /** File extension including the dot (".aac" / ".wav"). */
  extension: string;
  mimeType: string;
  /** Approximate on-disk cost of one second of audio, for preflight math. */
  bytesPerSecond: number;
}

/** One finished, crash-safe segment inside a session directory. */
export interface SegmentRecord {
  index: number;
  /** File name within the session directory (e.g. "seg-000.wav"). */
  file: string;
  durationMs: number;
  bytes: number;
}

/** The on-disk session manifest (manifest.json). Rewritten after every
 *  finalized segment — its presence at app launch means the session never
 *  finished cleanly and is a recovery candidate. */
export interface SessionManifest {
  version: 1;
  sessionId: string;
  startedAt: string;
  updatedAt: string;
  format: SegmentContainer;
  extension: string;
  mimeType: string;
  segmentSeconds: number;
  segments: SegmentRecord[];
}

/** A crashed/interrupted session found on launch. */
export interface RecoverableSession {
  manifest: SessionManifest;
  segmentCount: number;
  totalDurationMs: number;
}

/** The stitched, uploadable result — shape-compatible with the video path's
 *  RecordedFile so it drops straight into the existing upload flow. */
export interface RecordedAudioFile {
  uri: string;
  name: string;
  mimeType: string;
  size?: number;
}

/** Minimal synchronous filesystem the recorder needs. Implemented for real by
 *  expoFs (expo-file-system) and in-memory by MemoryFs for tests. */
export interface RecorderFs {
  /** Root for durable app storage, no trailing slash (e.g. "file:///doc"). */
  documentDirUri(): string;
  ensureDir(dirUri: string): void;
  exists(uri: string): boolean;
  /** Names of the immediate subdirectories of a directory. */
  listDirNames(dirUri: string): string[];
  /** Names of the FILES (not subdirectories) directly inside a directory. */
  listFileNames(dirUri: string): string[];
  readText(fileUri: string): string;
  writeText(fileUri: string, text: string): void;
  readBytes(fileUri: string): Uint8Array;
  writeBytes(fileUri: string, bytes: Uint8Array): void;
  move(srcUri: string, destUri: string): void;
  /** Delete a file, or a directory and everything under it. */
  deleteRecursive(uri: string): void;
  sizeOf(fileUri: string): number | null;
  /** Free disk space in bytes, or null when genuinely unknown. */
  freeBytes(): number | null;
}

/** One native recorder instance recording ONE segment. A new port is created
 *  per segment (mirroring expo-audio's AudioRecorder lifecycle). */
export interface RecorderPort {
  prepare(): Promise<void>;
  start(): void;
  /** Stop and finalize the segment file; resolves with its uri (null if the
   *  recorder never produced one). */
  stop(): Promise<{ uri: string | null }>;
  /** Live status — polled to detect the OS silently killing the recording
   *  (phone call, audio-focus loss). */
  isRecording(): boolean;
  release(): void;
}

export type RecorderFactory = () => RecorderPort;
