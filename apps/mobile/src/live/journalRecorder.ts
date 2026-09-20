/**
 * The Journal recorder: the loop (journalLoop.ts) + the files
 * (journalStore.ts) + the 30-minute rotation and the upload/retry rules,
 * with no React and no native imports so it is unit-tested end to end.
 *
 *   pushSamples ─► JournalLoop ─► keep(pcm) ─► current OpenJournalFile
 *        │
 *        └─ every `rotateSeconds` of LISTENING (audio clock, not a timer —
 *           the audio clock keeps ticking exactly when the mic does):
 *           close the file, open a fresh one, upload everything closed.
 *
 * Uploads: `upload(file)` is the seam (production: journalDeps.ts →
 * the chunked upload JOB, title "Journal — <date> <start–end>", consent +
 * store). A file is deleted only after `upload` resolves; a rejection keeps
 * it on disk and it is retried at the next boundary (the next rotation, the
 * next Stop, the next Start) — nothing is lost, the failure is shown.
 * Uploads run detached from the audio path (never block a frame) and one
 * at a time.
 *
 * `hasSelfPrint` is the honest gate: without an enrolled owner voiceprint
 * the labeler can never say `isSelf: true`, so a journal would keep nothing
 * — the hook refuses to start and the screen says "enroll your voice first".
 */
import type { FrameVad } from "./vad";
import type { Embedder, SpeakerLabeler } from "./speakerId";
import { JournalLoop, type DiscardedSegmentMeta, type JournalLoopStats } from "./journalLoop";
import type { ClosedJournalFile, JournalStore, OpenJournalFile } from "./journalStore";

/** A journal file is closed and uploaded after this much listening. */
export const JOURNAL_ROTATE_SECONDS = 30 * 60;
/** The screen shows "haven't heard you for a while" past this. */
export const JOURNAL_QUIET_NOTE_SECONDS = 10 * 60;

export interface JournalUploadCounts {
  /** Files on disk waiting to be uploaded (this run's and earlier runs'). */
  pending: number;
  /** Files the server accepted this run. */
  sent: number;
  /** Upload attempts that failed this run (the files are kept and retried). */
  failed: number;
  lastError: string | null;
  inFlight: boolean;
}

export interface JournalState {
  status: "idle" | "starting" | "listening" | "stopping" | "stopped";
  /** Seconds the mic has been open (audio clock). */
  listeningSeconds: number;
  /** Kept stretches of the owner's voice this run (all files). */
  selfCount: number;
  /** Seconds of the owner's speech kept this run (context excluded). */
  selfSeconds: number;
  /** Wall-clock ms the owner was last heard; null until once. */
  lastSelfAt: number | null;
  /** Wall-clock ms the run started; null when idle. */
  startedAt: number | null;
  /** The open file: bytes on disk (+ unflushed) and its start. */
  fileBytes: number;
  fileStartedAt: number | null;
  /** Files closed this run (rotations + the final one). */
  filesClosed: number;
  uploads: JournalUploadCounts;
  /** The VAD fell back to the energy rule (Silero failed). */
  vadDegraded: boolean;
  /** A problem that ended (or prevented) the run; null otherwise. */
  error: string | null;
}

export const IDLE_JOURNAL_STATE: JournalState = Object.freeze({
  status: "idle",
  listeningSeconds: 0,
  selfCount: 0,
  selfSeconds: 0,
  lastSelfAt: null,
  startedAt: null,
  fileBytes: 0,
  fileStartedAt: null,
  filesClosed: 0,
  uploads: Object.freeze({ pending: 0, sent: 0, failed: 0, lastError: null, inFlight: false }),
  vadDegraded: false,
  error: null,
}) as JournalState;

export type JournalUploader = (file: ClosedJournalFile) => Promise<void>;

export interface JournalRecorderOptions {
  vad: FrameVad;
  embedder: Embedder | null;
  labeler: SpeakerLabeler;
  store: JournalStore;
  upload: JournalUploader;
  onState?: (state: JournalState) => void;
  onDiscard?: (meta: DiscardedSegmentMeta) => void;
  now?: () => number;
  rotateSeconds?: number;
  /** Loop tuning passthrough (tests). */
  loop?: {
    minSegmentSeconds?: number;
    contextSeconds?: number;
    mergeGapSeconds?: number;
    maxEmbedSeconds?: number;
  };
}

export class JournalRecorder {
  private readonly loop: JournalLoop;
  private readonly store: JournalStore;
  private readonly upload: JournalUploader;
  private readonly now: () => number;
  private readonly rotateSeconds: number;
  private readonly onState: ((state: JournalState) => void) | null;

  private file: OpenJournalFile | null = null;
  /** Audio-clock second at which the current file was opened. */
  private fileOpenedAtAudio = 0;
  private rotating: Promise<void> | null = null;
  private uploading: Promise<void> | null = null;
  private lastEmittedSecond = -1;
  private state: JournalState = { ...IDLE_JOURNAL_STATE, uploads: { ...IDLE_JOURNAL_STATE.uploads } };

  constructor(private readonly opts: JournalRecorderOptions) {
    this.store = opts.store;
    this.upload = opts.upload;
    this.now = opts.now ?? Date.now;
    this.rotateSeconds = opts.rotateSeconds ?? JOURNAL_ROTATE_SECONDS;
    this.onState = opts.onState ?? null;
    this.loop = new JournalLoop({
      vad: opts.vad,
      embedder: opts.embedder,
      labeler: opts.labeler,
      now: this.now,
      keep: (pcm, meta) => {
        const file = this.file;
        if (!file) return;
        file.append(pcm, meta);
        this.emit(true);
      },
      onDiscard: (meta) => {
        opts.onDiscard?.(meta);
        this.emit(true);
      },
      onDegrade: () => this.emit(true),
      ...(opts.loop ?? {}),
    });
  }

  /** Whether the labeler has an enrolled owner print — without one nothing
   *  could ever be kept. */
  get hasSelfPrint(): boolean {
    return this.opts.labeler.hasSelfPrint;
  }

  get stateSnapshot(): JournalState {
    return this.state;
  }

  get isRunning(): boolean {
    return this.loop.isRunning;
  }

  /** Files on disk waiting for an upload right now. */
  get pendingUploads(): number {
    return this.store.listClosed().length;
  }

  async start(): Promise<void> {
    if (this.loop.isRunning) return;
    const startedAt = this.now();
    this.state = {
      ...IDLE_JOURNAL_STATE,
      status: "listening",
      startedAt,
      uploads: { ...IDLE_JOURNAL_STATE.uploads, pending: this.pendingUploads },
    };
    this.file = this.store.open(startedAt);
    this.fileOpenedAtAudio = 0;
    this.lastEmittedSecond = -1;
    this.loop.start();
    this.emit(true);
    // Leftovers from an earlier run (a crash, a failed upload): send now.
    void this.uploadClosed();
  }

  /** Feed 16 kHz mono int16 samples. Synchronous, never throws. */
  pushSamples(samples: Int16Array): void {
    if (!this.loop.isRunning) return;
    this.loop.pushSamples(samples);
    if (this.loop.audioClock - this.fileOpenedAtAudio >= this.rotateSeconds && !this.rotating) {
      void this.rotate();
    }
    this.emit(false);
  }

  /** Wait for every queued frame/segment and any rotation in flight. */
  async settle(): Promise<void> {
    await this.loop.settle();
    if (this.rotating) await this.rotating;
    await this.loop.settle();
  }

  /** Close the current file, open a fresh one, upload what is closed. */
  async rotate(): Promise<void> {
    if (this.rotating) return this.rotating;
    this.rotating = (async () => {
      const closing = this.file;
      if (!closing || !this.loop.isRunning) return;
      // The new file takes over first so a stretch landing mid-rotation has
      // somewhere to go; the closing file only ever shrinks from here.
      const at = this.now();
      this.file = this.store.open(at);
      this.fileOpenedAtAudio = this.loop.audioClock;
      await this.closeFile(closing, at);
      this.emit(true);
      void this.uploadClosed();
    })().finally(() => {
      this.rotating = null;
    });
    return this.rotating;
  }

  /** Stop listening: flush the loop, close the file, start the uploads
   *  (they continue after this resolves — `uploadsSettled` waits on them). */
  async stop(): Promise<JournalState> {
    if (!this.loop.isRunning) return this.state;
    this.state = { ...this.state, status: "stopping" };
    this.emit(true);
    if (this.rotating) await this.rotating;
    let loopStats: JournalLoopStats | null = null;
    try {
      loopStats = await this.loop.stop();
    } catch (err) {
      this.state = { ...this.state, error: `journal stopped with an error: ${err instanceof Error ? err.message : String(err)}` };
    }
    const closing = this.file;
    this.file = null;
    if (closing) await this.closeFile(closing, this.now());
    this.state = {
      ...this.state,
      ...(loopStats ? this.fromLoop(loopStats) : {}),
      status: "stopped",
      fileBytes: 0,
      fileStartedAt: null,
    };
    this.emit(true);
    void this.uploadClosed();
    return this.state;
  }

  /** Resolves once no upload is in flight (tests, and a Stop that wants to
   *  report the final count). */
  async uploadsSettled(): Promise<void> {
    while (this.uploading) await this.uploading;
  }

  /** Retry every closed file now (the screen's "retry" affordance). */
  retryUploads(): Promise<void> {
    return this.uploadClosed();
  }

  // -------------------------------------------------------------------------

  private fromLoop(s: JournalLoopStats): Partial<JournalState> {
    return {
      listeningSeconds: Math.floor(s.listeningSeconds),
      selfCount: s.selfCount,
      selfSeconds: Math.round(s.selfSeconds * 10) / 10,
      lastSelfAt: s.lastSelfWallMs,
      vadDegraded: this.loop.isVadDegraded,
    };
  }

  /** Publish state; unforced calls are throttled to once per audio second. */
  private emit(force: boolean): void {
    const second = Math.floor(this.loop.audioClock);
    if (!force && second === this.lastEmittedSecond) return;
    this.lastEmittedSecond = second;
    if (this.loop.isRunning) {
      this.state = {
        ...this.state,
        ...this.fromLoop(this.loop.statsSnapshot),
        fileBytes: this.file?.bytes ?? 0,
        fileStartedAt: this.file?.startedAtMs ?? null,
      };
    }
    this.onState?.(this.state);
  }

  private async closeFile(file: OpenJournalFile, endedAtMs: number): Promise<void> {
    try {
      const closed = await file.close(endedAtMs);
      this.state = {
        ...this.state,
        filesClosed: this.state.filesClosed + 1,
        uploads: { ...this.state.uploads, pending: this.state.uploads.pending + (closed ? 1 : 0) },
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.warn("[journal] closing the journal file failed:", msg);
      this.state = { ...this.state, error: `writing the journal failed: ${msg}` };
    }
  }

  /** Upload every closed file, oldest first, one at a time; a failure keeps
   *  the file for the next boundary and moves on to the next file. */
  private uploadClosed(): Promise<void> {
    if (this.uploading) return this.uploading;
    this.uploading = (async () => {
      const files = this.store.listClosed();
      this.state = { ...this.state, uploads: { ...this.state.uploads, pending: files.length, inFlight: files.length > 0 } };
      this.emit(true);
      for (const file of files) {
        try {
          await this.upload(file);
          this.store.remove(file);
          this.state = {
            ...this.state,
            uploads: {
              ...this.state.uploads,
              sent: this.state.uploads.sent + 1,
              pending: Math.max(0, this.state.uploads.pending - 1),
            },
          };
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          console.warn("[journal] upload failed (kept for retry):", file.uri, msg);
          this.state = {
            ...this.state,
            uploads: { ...this.state.uploads, failed: this.state.uploads.failed + 1, lastError: msg },
          };
        }
        this.emit(true);
      }
      this.state = { ...this.state, uploads: { ...this.state.uploads, inFlight: false, pending: this.store.listClosed().length } };
      this.emit(true);
    })().finally(() => {
      this.uploading = null;
    });
    return this.uploading;
  }
}

/** "Journal — 2026-08-30 09:00–09:30" (local time, zero-padded). */
export function journalTitle(file: Pick<ClosedJournalFile, "startedAt" | "endedAt">): string {
  const start = new Date(file.startedAt);
  const end = new Date(file.endedAt);
  const pad = (n: number) => String(n).padStart(2, "0");
  const date = `${start.getFullYear()}-${pad(start.getMonth() + 1)}-${pad(start.getDate())}`;
  const hm = (d: Date) => `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  return `Journal — ${date} ${hm(start)}–${hm(end)}`;
}

/** The analysis `context` line (≤ 500 chars): what this file is and when
 *  the owner spoke, so the transcript can be put back on the clock. */
export function journalContext(file: ClosedJournalFile, maxChars = 500): string {
  const n = file.segments?.length ?? 0;
  let text =
    `Voice journal (source: journal): only stretches of my own voice, kept by the phone; ` +
    `${n} stretch${n === 1 ? "" : "es"} between ${file.startedAt} and ${file.endedAt}.`;
  if (file.segments && file.segments.length > 0) {
    const parts: string[] = [];
    for (const s of file.segments) {
      const t = s.start_wall_iso.slice(11, 19);
      const piece = `${Math.round(s.offset_s)}s@${t}`;
      const candidate = `${text} Offsets: ${[...parts, piece].join(", ")}`;
      if (candidate.length > maxChars - 1) break;
      parts.push(piece);
    }
    if (parts.length > 0) text = `${text} Offsets: ${parts.join(", ")}`;
  }
  return text.length > maxChars ? text.slice(0, maxChars) : text;
}
