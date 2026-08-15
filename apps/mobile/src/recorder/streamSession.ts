import type { RecorderSessionStore } from "./sessionStore";
import { segmentFileName } from "./sessionStore";
import type { PcmFrame, PcmSource, PcmSourceFactory } from "./pcmSource";
import type {
  RecordedAudioFile,
  RecorderFs,
  SegmentFormat,
  SessionManifest,
} from "./types";
import {
  DATA_SIZE_OFFSET,
  RIFF_SIZE_OFFSET,
  WAV_HEADER_BYTES,
  buildWavHeader,
  dataSizeBytes,
  int16ToBytes,
  riffSizeBytes,
} from "./wav";
import type { SessionPhase } from "./segmentedSession";

/** Flush buffered audio to disk (and update the manifest) this often — the
 *  crash-loss bound of the v2 engine: a crash loses at most this much. */
export const DEFAULT_FLUSH_MS = 5000;

/** Rotate to a new segment file roughly this often. Rotation is pure file
 *  bookkeeping between two buffer writes — the stream never pauses, so the
 *  interval costs nothing in audio; it exists to keep any single file's
 *  header-patch blast radius (and per-file corruption risk) bounded. */
export const DEFAULT_STREAM_SEGMENT_MS = 5 * 60 * 1000;

/** How long to wait between auto-resume attempts after an interruption. */
export const DEFAULT_RESUME_RETRY_MS = 3000;

/** If the source claims to be capturing but no frame has arrived for this
 *  long, treat it as a silent interruption. Frames normally arrive ~10x per
 *  second, so 10 s of silence from the native layer means the mic is gone
 *  whatever the status flag says. Deliberately conservative — a false
 *  interruption costs a segment rotation, a missed one costs the session. */
export const DEFAULT_STALL_MS = 10_000;

/** Format facts for the stream engine: raw PCM in WAV at 16 kHz mono s16le —
 *  the server's native format. bytesPerSecond is the 16 kHz preflight
 *  estimate; a device that honestly delivers a higher rate costs
 *  proportionally more disk (see StreamAudioSession's honest-rate handling). */
export const STREAM_WAV_FORMAT: SegmentFormat = {
  format: "wav",
  extension: ".wav",
  mimeType: "audio/wav",
  bytesPerSecond: 32100, // 16 kHz * 2 bytes mono + headers
};

export interface StreamSessionDeps {
  makeSource: PcmSourceFactory;
  store: RecorderSessionStore;
  segmentMs?: number;
  flushMs?: number;
  resumeRetryMs?: number;
  stallMs?: number;
  now?: () => number;
}

/**
 * One growing WAV segment file. Every append is immediately followed by a
 * header size-patch, so at ANY instant the file on disk is a valid, playable
 * WAV describing exactly the audio flushed so far:
 *
 * - crash between append and patch → the header understates by one flush
 *   batch; the extra tail bytes are ignored by any parser (bounded loss).
 * - crash mid-append (torn write) → declared size exceeds the file;
 *   stitch.ts clamps the data chunk to what exists (bounded loss).
 */
class WavSegmentWriter {
  private samplesWritten = 0;

  constructor(
    private fs: RecorderFs,
    readonly uri: string,
    readonly sampleRate: number,
  ) {
    this.fs.writeBytes(uri, buildWavHeader(sampleRate, 0));
  }

  append(samples: Int16Array): void {
    if (samples.length === 0) return;
    this.fs.appendBytes(this.uri, int16ToBytes(samples));
    this.samplesWritten += samples.length;
    const dataBytes = this.samplesWritten * 2;
    this.fs.writeBytesAt(this.uri, RIFF_SIZE_OFFSET, riffSizeBytes(dataBytes));
    this.fs.writeBytesAt(this.uri, DATA_SIZE_OFFSET, dataSizeBytes(dataBytes));
  }

  get samples(): number {
    return this.samplesWritten;
  }

  get fileBytes(): number {
    return WAV_HEADER_BYTES + this.samplesWritten * 2;
  }

  get durationMs(): number {
    return Math.round((this.samplesWritten / this.sampleRate) * 1000);
  }
}

/**
 * The v2 GAPLESS recording engine. The microphone stream is opened once and
 * never stopped while recording — segmentation happens at the file-writer
 * level, so rotating to a new segment file loses ZERO samples (v1 lost ~0.2 s
 * per rotation restarting the platform recorder).
 *
 * Durability model:
 * - incoming frames buffer in memory (`pending`);
 * - every ~5 s (flushMs) the buffer is appended to the current segment WAV,
 *   the WAV header is size-patched, and the manifest is rewritten — so a
 *   crash loses at most ~5 s, not up to a 5-minute segment;
 * - every ~5 min (segmentMs) the writer rotates to a new file BETWEEN two
 *   appends; every closed segment is independently playable;
 * - on-disk sessions are byte-compatible with v1 wav sessions, so recovery
 *   (manifest scan → stitch → one WAV) is the existing, already-tested path.
 *
 * Honesty: frames carry the TRUE hardware sample rate. If it isn't the 16 kHz
 * we asked for, the audio is recorded at the reported rate and the header
 * says so; a mid-session rate change forces a rotation so no file ever
 * mislabels its contents.
 *
 * Driven by the same external 1-second `tick()` contract as v1's engine —
 * no internal timers, fully deterministic under test.
 */
export class StreamAudioSession {
  phase: SessionPhase = "idle";

  private makeSource: PcmSourceFactory;
  private store: RecorderSessionStore;
  private fs: RecorderFs;
  private segmentMs: number;
  private flushMs: number;
  private resumeRetryMs: number;
  private stallMs: number;
  private now: () => number;

  private manifest: SessionManifest | null = null;
  private source: PcmSource | null = null;
  private writer: WavSegmentWriter | null = null;
  private nextSegmentIndex = 0;

  /** Frames captured but not yet flushed to disk, all at `pendingRate`. */
  private pending: Int16Array[] = [];
  private pendingSamples = 0;
  private pendingRate: number | null = null;

  /** Audio actually captured, in ms — the UI clock. Advances per frame. */
  private capturedMs = 0;
  private lastFlushAt = 0;
  private lastFrameAt = 0;
  private lastResumeAttemptAt = 0;

  constructor(deps: StreamSessionDeps) {
    this.makeSource = deps.makeSource;
    this.store = deps.store;
    this.fs = deps.store.fs;
    this.segmentMs = deps.segmentMs ?? DEFAULT_STREAM_SEGMENT_MS;
    this.flushMs = deps.flushMs ?? DEFAULT_FLUSH_MS;
    this.resumeRetryMs = deps.resumeRetryMs ?? DEFAULT_RESUME_RETRY_MS;
    this.stallMs = deps.stallMs ?? DEFAULT_STALL_MS;
    this.now = deps.now ?? Date.now;
  }

  get segmentsSaved(): number {
    return this.manifest?.segments.length ?? 0;
  }

  /** Milliseconds of audio already crash-safe on disk (flushed + manifested). */
  get savedDurationMs(): number {
    return (
      this.manifest?.segments.reduce((sum, s) => sum + s.durationMs, 0) ?? 0
    );
  }

  /** Audio captured so far (including the not-yet-flushed tail). Unlike v1
   *  this is audio time, not wall time — it does not advance while the mic
   *  is interrupted, which is exactly what the UI clock should show. */
  elapsedMs(): number {
    return Math.round(this.capturedMs);
  }

  async start(): Promise<void> {
    if (this.phase !== "idle") return;
    this.manifest = this.store.createSession({
      format: STREAM_WAV_FORMAT.format,
      extension: STREAM_WAV_FORMAT.extension,
      mimeType: STREAM_WAV_FORMAT.mimeType,
      segmentSeconds: Math.round(this.segmentMs / 1000),
      engine: "stream",
    });
    await this.openSource();
    const t = this.now();
    this.lastFlushAt = t;
    this.lastFrameAt = t;
    this.phase = "recording";
  }

  private async openSource(): Promise<void> {
    const source = this.makeSource();
    await source.start((frame) => this.onFrame(frame));
    this.source = source;
  }

  /** Never throws — releasing a dead native stream must not take the
   *  session down. */
  private stopSource(): void {
    const source = this.source;
    this.source = null;
    if (!source) return;
    try {
      source.stop();
    } catch {
      // Already dead; nothing further to release.
    }
  }

  /**
   * Synchronous frame intake. Deliberately does NO filesystem work — frames
   * only accumulate in memory; disk writes happen on the flush cadence. The
   * one exception is a sample-rate change, which must rotate segments so no
   * WAV file ever contains audio at a rate its header doesn't declare.
   */
  private onFrame(frame: PcmFrame): void {
    if (this.phase !== "recording") return; // trailing native buffer after stop
    if (frame.samples.length === 0) return;
    if (this.pendingRate !== null && frame.sampleRate !== this.pendingRate) {
      // The hardware switched rates mid-stream (e.g. a headset (dis)connect).
      // Flush what we hold at the old rate and close that segment; the new
      // rate starts a fresh file with an honest header.
      this.flush();
      this.closeSegment();
    }
    this.pendingRate = frame.sampleRate;
    this.pending.push(frame.samples);
    this.pendingSamples += frame.samples.length;
    this.capturedMs += (frame.samples.length / frame.sampleRate) * 1000;
    this.lastFrameAt = this.now();
  }

  private mergePending(): Int16Array {
    const merged = new Int16Array(this.pendingSamples);
    let cursor = 0;
    for (const chunk of this.pending) {
      merged.set(chunk, cursor);
      cursor += chunk.length;
    }
    this.pending = [];
    this.pendingSamples = 0;
    return merged;
  }

  /**
   * Append everything pending to the current segment file, patch its header,
   * and rewrite the manifest — after this returns, all audio received so far
   * survives a process death.
   */
  private flush(): void {
    this.lastFlushAt = this.now();
    if (this.pendingSamples === 0 || this.pendingRate === null) return;
    if (!this.manifest) return;
    if (!this.writer) {
      const file = segmentFileName(
        this.nextSegmentIndex,
        this.manifest.extension,
      );
      this.writer = new WavSegmentWriter(
        this.fs,
        this.store.segmentFileUri(this.manifest.sessionId, file),
        this.pendingRate,
      );
    }
    this.writer.append(this.mergePending());
    this.manifest = this.store.upsertSegment(this.manifest, {
      index: this.nextSegmentIndex,
      file: segmentFileName(this.nextSegmentIndex, this.manifest.extension),
      durationMs: this.writer.durationMs,
      bytes: this.writer.fileBytes,
    });
  }

  /** Close the current segment. Its file needs no finalization pass — the
   *  header was patched at the last append — so this is pure bookkeeping,
   *  which is exactly why rotation can never drop a sample. */
  private closeSegment(): void {
    if (!this.writer) return;
    this.writer = null;
    this.nextSegmentIndex += 1;
  }

  /**
   * Drive the engine (~once per second, same contract as v1):
   * - recording: detect a dead/stalled stream (→ salvage everything and go
   *   interrupted), else flush on the flush cadence and rotate when the
   *   current segment is full;
   * - interrupted: attempt an auto-resume every resumeRetryMs.
   */
  async tick(): Promise<void> {
    if (this.phase === "recording") {
      const dead = this.source ? !this.source.isCapturing() : true;
      const stalled = this.now() - this.lastFrameAt >= this.stallMs;
      if (dead || stalled) {
        // The OS took the microphone (call, focus loss) — or the native
        // layer went silent without saying so. Salvage EVERYTHING held in
        // memory (unlike v1, interruption loses zero already-captured
        // audio), then try to come back.
        this.flush();
        this.closeSegment();
        this.stopSource();
        this.phase = "interrupted";
        this.lastResumeAttemptAt = this.now();
        return;
      }
      if (this.now() - this.lastFlushAt >= this.flushMs) {
        this.flush();
        // Rotate BETWEEN two writes: the segment target is expressed in
        // samples at the segment's own rate, so a full segment holds
        // segmentMs of real audio whatever the hardware rate is.
        if (
          this.writer &&
          this.writer.samples >=
            (this.segmentMs / 1000) * this.writer.sampleRate
        ) {
          this.closeSegment();
        }
      }
      return;
    }
    if (this.phase === "interrupted") {
      if (this.now() - this.lastResumeAttemptAt < this.resumeRetryMs) return;
      this.lastResumeAttemptAt = this.now();
      try {
        await this.openSource();
        const t = this.now();
        this.lastFrameAt = t;
        this.lastFlushAt = t;
        this.phase = "recording";
      } catch {
        // Mic still unavailable (call ongoing). Stay interrupted; everything
        // flushed so far remains recoverable even if the app dies now.
      }
    }
  }

  /**
   * The user is leaving mid-recording (back button, screen unmount). Salvage
   * every captured sample to disk and leave the session in its recoverable
   * state — the recovery prompt will offer it back rather than silently
   * losing it.
   */
  async abandon(): Promise<void> {
    if (this.phase !== "recording") return;
    this.stopSource();
    this.flush();
    this.closeSegment();
    this.phase = "interrupted";
  }

  /**
   * Finish the session: stop the stream, flush the tail, and stitch all
   * segments into one uploadable WAV (headers stripped, PCM concatenated —
   * pure byte math through the same store path v1 and crash recovery use).
   * On success the session leaves no recovery residue.
   */
  async stopAndFinish(): Promise<RecordedAudioFile> {
    if (!this.manifest) throw new Error("Session was never started");
    if (this.phase === "recording" || this.phase === "interrupted") {
      this.stopSource();
      this.flush();
      this.closeSegment();
    }
    const file = this.store.finishToFile(this.manifest);
    this.phase = "done";
    return file;
  }
}
