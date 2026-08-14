import type { RecorderSessionStore } from "./sessionStore";
import type {
  RecordedAudioFile,
  RecorderFactory,
  RecorderPort,
  SegmentFormat,
  SessionManifest,
} from "./types";

/** Finalize a completed segment this often. A crash loses at most this much. */
export const DEFAULT_SEGMENT_MS = 5 * 60 * 1000;

/** How long to wait between auto-resume attempts after an interruption. */
export const DEFAULT_RESUME_RETRY_MS = 3000;

export type SessionPhase = "idle" | "recording" | "interrupted" | "done";

export interface SegmentedSessionDeps {
  makeRecorder: RecorderFactory;
  store: RecorderSessionStore;
  format: SegmentFormat;
  segmentMs?: number;
  resumeRetryMs?: number;
  now?: () => number;
}

/**
 * The crash-hardened recording engine. Records audio in ~5-minute segments,
 * finalizing each one to durable storage before starting the next — so a
 * crash, kill, or battery death loses at most the current segment.
 *
 * The engine is driven by an external `tick()` (the UI's 1-second clock):
 * each tick rotates a due segment, detects the OS having silently killed the
 * recorder (phone call / audio-focus loss → finalize immediately, then
 * auto-retry), with no internal timers — which also makes it deterministic
 * under test.
 */
export class SegmentedAudioSession {
  phase: SessionPhase = "idle";

  private makeRecorder: RecorderFactory;
  private store: RecorderSessionStore;
  private format: SegmentFormat;
  private segmentMs: number;
  private resumeRetryMs: number;
  private now: () => number;

  private manifest: SessionManifest | null = null;
  private recorder: RecorderPort | null = null;
  private segmentStartedAt = 0;
  private startedAt = 0;
  /** Recorded time already safely finalized to disk. */
  private finalizedMs = 0;
  /** Elapsed at the moment of interruption (the clock stops while dead). */
  private frozenElapsedMs = 0;
  private lastResumeAttemptAt = 0;

  constructor(deps: SegmentedSessionDeps) {
    this.makeRecorder = deps.makeRecorder;
    this.store = deps.store;
    this.format = deps.format;
    this.segmentMs = deps.segmentMs ?? DEFAULT_SEGMENT_MS;
    this.resumeRetryMs = deps.resumeRetryMs ?? DEFAULT_RESUME_RETRY_MS;
    this.now = deps.now ?? Date.now;
  }

  get segmentsSaved(): number {
    return this.manifest?.segments.length ?? 0;
  }

  /** Milliseconds of audio already crash-safe on disk. */
  get savedDurationMs(): number {
    return this.finalizedMs;
  }

  elapsedMs(): number {
    if (this.phase === "recording") {
      return this.frozenElapsedMs + (this.now() - this.segmentStartedAt);
    }
    return this.frozenElapsedMs;
  }

  async start(): Promise<void> {
    if (this.phase !== "idle") return;
    this.manifest = this.store.createSession({
      format: this.format.format,
      extension: this.format.extension,
      mimeType: this.format.mimeType,
      segmentSeconds: Math.round(this.segmentMs / 1000),
    });
    this.startedAt = this.now();
    await this.beginSegment();
    this.phase = "recording";
  }

  private async beginSegment(): Promise<void> {
    const recorder = this.makeRecorder();
    await recorder.prepare();
    recorder.start();
    this.recorder = recorder;
    this.segmentStartedAt = this.now();
  }

  /** Stop the live recorder and (best-effort) finalize its file as the next
   *  segment. Never throws — a dead recorder must not take the session down. */
  private async finalizeCurrent(): Promise<void> {
    const recorder = this.recorder;
    this.recorder = null;
    if (!recorder || !this.manifest) return;
    const durationMs = this.now() - this.segmentStartedAt;
    try {
      const { uri } = await recorder.stop();
      if (uri) {
        this.manifest = this.store.finalizeSegment(
          this.manifest,
          uri,
          durationMs,
        );
        this.finalizedMs += durationMs;
      }
    } catch {
      // The recorder died without a usable file — the audio since the last
      // rotation is lost (this is exactly the bounded loss the segmenting
      // exists to cap). Everything already finalized is safe.
    } finally {
      this.frozenElapsedMs += durationMs;
      try {
        recorder.release();
      } catch {
        // Releasing a dead native object must never crash the session.
      }
    }
  }

  /**
   * Drive the engine. Call ~once per second while the screen is up:
   * - recording: detect an OS-killed recorder (→ interruption), else rotate
   *   when the segment interval has elapsed.
   * - interrupted: attempt an auto-resume every resumeRetryMs.
   */
  async tick(): Promise<void> {
    if (this.phase === "recording") {
      const recorder = this.recorder;
      if (recorder && !recorder.isRecording()) {
        // The OS took the microphone (phone call, focus loss, media-services
        // reset). Salvage what the recorder wrote, then try to come back.
        await this.finalizeCurrent();
        this.phase = "interrupted";
        this.lastResumeAttemptAt = this.now();
        return;
      }
      if (this.now() - this.segmentStartedAt >= this.segmentMs) {
        await this.finalizeCurrent();
        await this.beginSegment();
      }
      return;
    }
    if (this.phase === "interrupted") {
      if (this.now() - this.lastResumeAttemptAt < this.resumeRetryMs) return;
      this.lastResumeAttemptAt = this.now();
      try {
        await this.beginSegment();
        this.phase = "recording";
      } catch {
        // Mic still unavailable (call ongoing). Stay interrupted; everything
        // finalized so far remains recoverable even if the app dies now.
      }
    }
  }

  /**
   * The user is leaving mid-recording (back button, screen unmount). Salvage
   * the live segment to disk and leave the session in its recoverable state —
   * the recovery prompt will offer it back rather than silently losing it.
   */
  async abandon(): Promise<void> {
    if (this.phase !== "recording") return;
    await this.finalizeCurrent();
    this.phase = "interrupted";
  }

  /**
   * Finish the session: finalize the live segment (if any) and stitch all
   * segments into one uploadable audio file. On success the session leaves
   * no recovery residue.
   */
  async stopAndFinish(): Promise<RecordedAudioFile> {
    if (!this.manifest) throw new Error("Session was never started");
    if (this.phase === "recording") {
      await this.finalizeCurrent();
    }
    const file = this.store.finishToFile(this.manifest);
    this.phase = "done";
    return file;
  }
}
