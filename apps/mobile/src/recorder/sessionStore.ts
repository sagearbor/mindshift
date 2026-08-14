import { stitchSegments } from "./stitch";
import type {
  RecorderFs,
  RecordedAudioFile,
  RecoverableSession,
  SegmentContainer,
  SessionManifest,
} from "./types";

/** Options describing the session being created (per-platform format facts). */
export interface CreateSessionOptions {
  format: SegmentContainer;
  extension: string;
  mimeType: string;
  segmentSeconds: number;
}

let sessionCounter = 0;

/**
 * Durable persistence for segmented recording sessions.
 *
 * Layout (under the app's document directory — storage the OS won't purge):
 *
 *   recorder-sessions/<sessionId>/manifest.json
 *   recorder-sessions/<sessionId>/seg-000.wav …
 *   recorder-out/mindshift-audio-<ts>.wav      (stitched results)
 *
 * The manifest is rewritten after EVERY finalized segment, so at any instant
 * the disk tells the whole truth: a manifest with segments and no clean
 * finish IS a recoverable session — no flags, no heuristics. finishToFile()
 * removes the session directory, which is what "finished cleanly" means.
 */
export class RecorderSessionStore {
  constructor(private fs: RecorderFs) {}

  private rootDir(): string {
    return `${this.fs.documentDirUri()}/recorder-sessions`;
  }

  private outDir(): string {
    return `${this.fs.documentDirUri()}/recorder-out`;
  }

  private sessionDir(sessionId: string): string {
    return `${this.rootDir()}/${sessionId}`;
  }

  private manifestUri(sessionId: string): string {
    return `${this.sessionDir(sessionId)}/manifest.json`;
  }

  private writeManifest(manifest: SessionManifest): void {
    this.fs.writeText(
      this.manifestUri(manifest.sessionId),
      JSON.stringify(manifest),
    );
  }

  createSession(opts: CreateSessionOptions): SessionManifest {
    sessionCounter += 1;
    const now = new Date();
    const sessionId = `${now.getTime()}-${sessionCounter}`;
    const manifest: SessionManifest = {
      version: 1,
      sessionId,
      startedAt: now.toISOString(),
      updatedAt: now.toISOString(),
      format: opts.format,
      extension: opts.extension,
      mimeType: opts.mimeType,
      segmentSeconds: opts.segmentSeconds,
      segments: [],
    };
    this.fs.ensureDir(this.sessionDir(sessionId));
    this.writeManifest(manifest);
    return manifest;
  }

  /**
   * Move a finished recorder file into the session directory and record it in
   * the manifest. After this returns, that audio survives any crash.
   */
  finalizeSegment(
    manifest: SessionManifest,
    srcUri: string,
    durationMs: number,
  ): SessionManifest {
    const index = manifest.segments.length;
    const file = `seg-${String(index).padStart(3, "0")}${manifest.extension}`;
    const destUri = `${this.sessionDir(manifest.sessionId)}/${file}`;
    this.fs.move(srcUri, destUri);
    const bytes = this.fs.sizeOf(destUri) ?? 0;
    const updated: SessionManifest = {
      ...manifest,
      updatedAt: new Date().toISOString(),
      segments: [...manifest.segments, { index, file, durationMs, bytes }],
    };
    this.writeManifest(updated);
    return updated;
  }

  /**
   * Scan for sessions that never finished. Sessions with zero finished
   * segments (crashed before the first rotation — nothing recoverable) are
   * cleaned up silently. Corrupt manifests are skipped without blocking the
   * healthy ones.
   */
  listRecoverable(): RecoverableSession[] {
    const root = this.rootDir();
    if (!this.fs.exists(root)) return [];
    const found: RecoverableSession[] = [];
    for (const name of this.fs.listDirNames(root)) {
      const dir = `${root}/${name}`;
      let manifest: SessionManifest;
      try {
        manifest = JSON.parse(
          this.fs.readText(`${dir}/manifest.json`),
        ) as SessionManifest;
        if (manifest.version !== 1 || !Array.isArray(manifest.segments)) {
          continue;
        }
      } catch {
        // Unreadable manifest — leave the directory alone (the audio bytes may
        // still matter to someone debugging) but don't offer recovery on it.
        continue;
      }
      const segments = manifest.segments.filter((s) =>
        this.fs.exists(`${dir}/${s.file}`),
      );
      if (segments.length === 0) {
        this.fs.deleteRecursive(dir);
        continue;
      }
      found.push({
        manifest: { ...manifest, segments },
        segmentCount: segments.length,
        totalDurationMs: segments.reduce((sum, s) => sum + s.durationMs, 0),
      });
    }
    return found;
  }

  /**
   * Stitch a session's segments (in index order) into one uploadable audio
   * file, then remove the session directory. Used identically by the live
   * stop path and the crash-recovery path — recovery is not a special case.
   *
   * Order of operations is crash-safe: the stitched file is fully written
   * BEFORE the segments are deleted, so a death mid-finish leaves either a
   * recoverable session or a finished file — never neither.
   */
  finishToFile(manifest: SessionManifest): RecordedAudioFile {
    const dir = this.sessionDir(manifest.sessionId);
    const ordered = [...manifest.segments].sort((a, b) => a.index - b.index);
    const buffers = ordered.map((s) => this.fs.readBytes(`${dir}/${s.file}`));
    const stitched = stitchSegments(manifest.format, buffers);

    const out = this.outDir();
    // Previous stitched outputs were consumed by their upload flow — clear
    // them so recovered files don't accumulate forever.
    if (this.fs.exists(out)) this.fs.deleteRecursive(out);
    this.fs.ensureDir(out);
    const name = `mindshift-audio-${Date.now()}${manifest.extension}`;
    const uri = `${out}/${name}`;
    this.fs.writeBytes(uri, stitched);

    this.fs.deleteRecursive(dir);
    return {
      uri,
      name,
      mimeType: manifest.mimeType,
      size: stitched.byteLength,
    };
  }

  /** Throw a session away (user chose not to recover). */
  discard(sessionId: string): void {
    const dir = this.sessionDir(sessionId);
    if (this.fs.exists(dir)) this.fs.deleteRecursive(dir);
  }

  freeBytes(): number | null {
    return this.fs.freeBytes();
  }
}
