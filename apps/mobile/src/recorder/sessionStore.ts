import { stitchSegments } from "./stitch";
import type {
  RecorderFs,
  RecordedAudioFile,
  RecoverableSession,
  SegmentContainer,
  SegmentRecord,
  SessionManifest,
} from "./types";

/** Options describing the session being created (per-platform format facts). */
export interface CreateSessionOptions {
  format: SegmentContainer;
  extension: string;
  mimeType: string;
  segmentSeconds: number;
  /** Diagnostic tag for which engine produced the session (see manifest). */
  engine?: "stream" | "recorder";
}

/**
 * Who may claim recordings on this device right now: a uid, or a thunk that
 * reads the live one (production passes the auth store, so a sign-out between
 * constructing the store and scanning the disk is seen). Null = nobody known,
 * which claims nothing.
 */
export type OwnerSource = string | null | (() => string | null);

/** Sidecar carrying the owning uid of a stitched output, written next to it
 *  as "<name>.owner". A stitched file has no manifest to stamp, and the uid
 *  is stored verbatim as file CONTENT (not in the file name or a directory
 *  name) so no uid ever has to survive a charset or path-safety rewrite. */
const OWNER_SUFFIX = ".owner";

/** Name of a stitched output (excluding its owner sidecar). */
function isStitchedName(name: string): boolean {
  return name.startsWith("mindshift-audio-") && !name.endsWith(OWNER_SUFFIX);
}

/** Canonical segment file name ("seg-000.wav") — shared by the v1 move-in
 *  path (finalizeSegment) and the v2 write-in-place path (StreamAudioSession)
 *  so both produce identical on-disk sessions. */
export function segmentFileName(index: number, extension: string): string {
  return `seg-${String(index).padStart(3, "0")}${extension}`;
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
 *
 * Everything on that disk is OWNED. The document directory outlives a Firebase
 * sign-out, so without an owner the next account to open the app — the next
 * guest on a shared or handed-over phone — was offered the previous person's
 * conversation and could upload it as their own (found on an emulator
 * 2026-09-23). Every session manifest and every stitched output now carries
 * the uid that made it, and the scans below offer back only what the CURRENT
 * uid owns. Nothing is deleted to achieve that: logging out and back into the
 * SAME account still finds the recording waiting.
 */
export class RecorderSessionStore {
  /** Public so the v2 stream engine (which appends to segment files in
   *  place rather than moving finished recorder files in) can share the one
   *  filesystem seam instead of being handed a second, possibly-different fs.
   *
   *  `owner` defaults to null — fail CLOSED: a store wired without an owner
   *  records no owner and offers nothing back, rather than silently handing
   *  the disk to whoever asks. */
  constructor(
    readonly fs: RecorderFs,
    private readonly owner: OwnerSource = null,
  ) {}

  /** The uid that may claim recordings right now; null when unknown. */
  private currentOwnerUid(): string | null {
    const uid = typeof this.owner === "function" ? this.owner() : this.owner;
    return typeof uid === "string" && uid.length > 0 ? uid : null;
  }

  /**
   * May the signed-in account be offered something stamped `ownerUid`?
   *
   * Only on an exact uid match. An UNOWNED artifact (written before ownership
   * existed, or by a store with no owner) matches nobody: we cannot know who
   * recorded it, and adopting it into the first account that happens to open
   * the app is precisely the bug. Such files are left on disk, not deleted.
   */
  private claimableBy(ownerUid: string | undefined): boolean {
    const me = this.currentOwnerUid();
    return me !== null && ownerUid === me;
  }

  /** Uid stamped on a stitched output, or null when it carries none. */
  private stitchedOwnerUid(uri: string): string | null {
    const sidecar = `${uri}${OWNER_SUFFIX}`;
    try {
      if (!this.fs.exists(sidecar)) return null;
      const uid = this.fs.readText(sidecar).trim();
      return uid.length > 0 ? uid : null;
    } catch {
      // An unreadable sidecar is an unknown owner, never a free-for-all.
      return null;
    }
  }

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
    // Stamped once, at creation, and carried by every later rewrite (both
    // upsertSegment and finalizeSegment spread the manifest forward) — so the
    // "rewritten after EVERY finalized segment" invariant above keeps the
    // owner as durable as the segment list itself.
    const ownerUid = this.currentOwnerUid();
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
      ...(ownerUid ? { ownerUid } : {}),
      ...(opts.engine ? { engine: opts.engine } : {}),
    };
    this.fs.ensureDir(this.sessionDir(sessionId));
    this.writeManifest(manifest);
    return manifest;
  }

  /** Absolute uri for a segment file inside a session directory — the v2
   *  engine writes its WAVs there directly (no cache-file move). */
  segmentFileUri(sessionId: string, file: string): string {
    return `${this.sessionDir(sessionId)}/${file}`;
  }

  /**
   * Record (or refresh) a segment's manifest entry. The v2 flush path: called
   * every few seconds for the OPEN segment, so at any instant the manifest
   * describes all audio that is durable on disk — including the segment still
   * being written. Idempotent per index: an existing entry is replaced.
   */
  upsertSegment(
    manifest: SessionManifest,
    record: SegmentRecord,
  ): SessionManifest {
    const others = manifest.segments.filter((s) => s.index !== record.index);
    const updated: SessionManifest = {
      ...manifest,
      updatedAt: new Date().toISOString(),
      segments: [...others, record].sort((a, b) => a.index - b.index),
    };
    this.writeManifest(updated);
    return updated;
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
    const file = segmentFileName(index, manifest.extension);
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
   * Scan for sessions that never finished — ONLY those owned by the signed-in
   * uid. Sessions with zero finished segments (crashed before the first
   * rotation — nothing recoverable) are cleaned up silently. Corrupt manifests
   * are skipped without blocking the healthy ones.
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
      // Someone else's (or an unowned legacy) session: not ours to offer, and
      // not ours to clean up either — leave the directory exactly as found.
      if (!this.claimableBy(manifest.ownerUid)) continue;
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
    // Keep the newest few stitched outputs instead of clearing the dir: an
    // output is NOT necessarily consumed — the 2026-08-14 incident destroyed
    // two recovered recordings because each new stitch wiped the previous
    // (possibly never-uploaded) one. Names embed Date.now(), so lexicographic
    // order is chronological.
    this.fs.ensureDir(out);
    const KEEP = 3;
    const existing = this.fs.listFileNames(out).filter(isStitchedName).sort();
    for (const old of existing.slice(0, Math.max(0, existing.length - (KEEP - 1)))) {
      this.fs.deleteRecursive(`${out}/${old}`);
      // The sidecar follows its file — never left behind to describe nothing.
      const sidecar = `${out}/${old}${OWNER_SUFFIX}`;
      if (this.fs.exists(sidecar)) this.fs.deleteRecursive(sidecar);
    }
    const name = `mindshift-audio-${Date.now()}-${Math.random()
      .toString(36)
      .slice(2, 6)}${manifest.extension}`;
    const uri = `${out}/${name}`;
    // Owner sidecar BEFORE the audio: a death between the two leaves a sidecar
    // describing a file that doesn't exist (invisible to every scan), never an
    // unowned — and therefore unclaimable — recording. The session's own owner
    // is what carries over, so a recovery run by the right account still
    // produces a file that account can claim.
    const ownerUid = manifest.ownerUid ?? this.currentOwnerUid();
    if (ownerUid) this.fs.writeText(`${uri}${OWNER_SUFFIX}`, ownerUid);
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

  /**
   * Finished (stitched) outputs that no upload ever consumed. This happens
   * when recovery stitches a file and the subsequent upload fails or the app
   * dies before it runs (the 2026-08-14 incident: a broken OTA made the
   * upload unreachable, the app restarted, and the audio was stranded with
   * no UI path back to it). The recovery prompt offers these too — but only
   * the ones the signed-in uid owns (see claimableBy).
   */
  listOrphanStitched(): RecordedAudioFile[] {
    const out = this.outDir();
    if (!this.fs.exists(out)) return [];
    return this.fs
      .listFileNames(out)
      .filter(isStitchedName)
      .filter((name) =>
        this.claimableBy(this.stitchedOwnerUid(`${out}/${name}`) ?? undefined),
      )
      .map((name) => {
        const uri = `${out}/${name}`;
        const mimeType = name.endsWith(".aac") ? "audio/aac" : "audio/wav";
        return {
          uri,
          name,
          mimeType,
          size: this.fs.sizeOf(uri) ?? 0,
        };
      });
  }

  /** Delete one orphaned stitched output, and its owner sidecar (user chose
   *  not to use it, or already analyzed it). */
  discardOrphan(uri: string): void {
    if (this.fs.exists(uri)) this.fs.deleteRecursive(uri);
    const sidecar = `${uri}${OWNER_SUFFIX}`;
    if (this.fs.exists(sidecar)) this.fs.deleteRecursive(sidecar);
  }

  freeBytes(): number | null {
    return this.fs.freeBytes();
  }
}
