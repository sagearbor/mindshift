/**
 * The Journal loop — "listen for my voice all day" (Live Coach mode
 * `journal`). The battery-conscious cousin of the fast loop (fastLoop.ts):
 *
 *   PCM frames ─► VAD (Silero, one 32 ms frame at a time) ─► segmenter
 *                                                              │ turn end (≥ 1.0 s)
 *                                                              ▼
 *                                    ONE ECAPA embedding ─► labeler.label()
 *                                                              │
 *                                 isSelf === true ──► keep(segment ± context)
 *                                 otherwise      ──► discarded (never written)
 *
 * What runs PER FRAME: an int16 copy into the rolling history, the
 * int16 → float32 conversion of one 512-sample frame, one VAD call, one
 * segmenter push. Nothing else — no STT, no LLM, no TTS, no prosody, no
 * WebSocket. What runs PER COMPLETED SEGMENT (≥ `minSegmentSeconds`): one
 * embedding over at most the last `maxEmbedSeconds` of the segment, one
 * labeler match (cosine against the enrolled prints + the session's unknown
 * clusters), and — for the owner's own voice only — one slice of the history
 * handed to `keep`.
 *
 * Self identity uses the labeler's own rules (speakerId.ts): the absolute
 * bar (cosine ≥ 0.65) always; the contrast rule only once a second cluster
 * exists to contrast against (and the print pools ≥ 2 recordings). Anything
 * that is not `isSelf === true` — another voice, an unidentified short
 * fragment, an unembeddable segment — is dropped from memory as the history
 * rolls on and is never written anywhere.
 *
 * Context: a kept segment carries up to `contextSeconds` of audio on each
 * side, bounded by silence — the lead never reaches back into the previous
 * speech run (or the previous kept chunk), and the trail is cut short the
 * moment a new speech run begins. The trail needs audio that does not exist
 * yet when the segment finalizes, so kept segments wait in `kept` until the
 * audio clock has passed their trail end (or the loop stops).
 *
 * Everything is behind injected seams (`JournalLoopDeps`) so Jest drives the
 * whole loop with synthetic PCM, the energy VAD and a fake embedder.
 */
import type { FrameVad } from "./vad";
import { EnergyVad, SILERO_SAMPLE_RATE } from "./vad";
import { StreamingSegmenter, type Span } from "./segmenter";
import type { Embedder, MatchBasis, SpeakerLabeler, SpeakerVerdict } from "./speakerId";

/** A stretch shorter than this is never embedded (ECAPA is unstable below
 *  ~1 s; the labeler would not cluster it either). */
/** Journal-only "solo" self rule — see the discard branch below. Same
 *  numbers as the server's contrast floor (speaker_id.CROSS_MATCH_*). */
export const SOLO_SELF_THRESHOLD = 0.4;
export const SOLO_SELF_MIN_SETTINGS = 2;

export const JOURNAL_MIN_SEGMENT_SECONDS = 1.0;
/** Audio kept on each side of the owner's speech, bounded by silence. */
export const JOURNAL_CONTEXT_SECONDS = 1.0;
/** Pauses up to this long stay inside one segment (segmenter merge gap). */
export const JOURNAL_MERGE_GAP_SECONDS = 0.3;
/** Seconds of a segment handed to the embedder, taken from its END. */
export const JOURNAL_MAX_EMBED_SECONDS = 10;
/** Rolling history always retained (covers context + queue latency). */
export const JOURNAL_BASE_HISTORY_SECONDS = 30;
/** Hard cap on retained history, however long an open speech run is
 *  (180 s of 16 kHz int16 is 5.8 MB). A longer monologue loses its head. */
export const JOURNAL_MAX_HISTORY_SECONDS = 180;

/** What `keep` receives alongside the PCM of one kept stretch. */
export interface KeptSegmentMeta {
  /** Wall-clock ms at which the owner's speech (not the lead context) began. */
  startWallMs: number;
  /** Seconds of context before the speech, inside the PCM. */
  leadSeconds: number;
  /** Seconds of the speech itself. */
  speechSeconds: number;
  /** Seconds of context after the speech, inside the PCM. */
  trailSeconds: number;
  /** The labeler's cosine (absolute or contrast) — null never happens for a
   *  kept segment today, but the wire shape allows it. */
  score: number | null;
  basis: MatchBasis | null;
}

export interface DiscardedSegmentMeta {
  speechSeconds: number;
  isSelf: boolean | null;
  score: number | null;
  basis: MatchBasis | null;
  /** Why it was dropped: not the owner, unidentifiable, or unembeddable. */
  reason: "other" | "unidentified" | "unembeddable" | "no-embedder";
}

export interface JournalLoopStats {
  /** Seconds of audio pushed so far (the audio clock). */
  listeningSeconds: number;
  /** Kept stretches of the owner's voice. */
  selfCount: number;
  /** Seconds of the owner's speech kept (context excluded). */
  selfSeconds: number;
  /** Wall-clock ms of the most recent kept stretch's end; null until one. */
  lastSelfWallMs: number | null;
  discardedCount: number;
  discardedSeconds: number;
}

export interface JournalLoopDeps {
  vad: FrameVad;
  /** Null => nothing can be identified: every segment is discarded (the
   *  recorder never starts the loop in that state; kept honest here too). */
  embedder: Embedder | null;
  labeler: SpeakerLabeler;
  /** Take one kept stretch (16 kHz mono int16, context included). */
  keep: (pcm: Int16Array, meta: KeptSegmentMeta) => void;
  onDiscard?: (meta: DiscardedSegmentMeta) => void;
  /** The VAD fell back to the energy rule mid-session. */
  onDegrade?: (stage: "vad", reason: string) => void;
  /** Wall clock (ms). */
  now?: () => number;
  minSegmentSeconds?: number;
  contextSeconds?: number;
  mergeGapSeconds?: number;
  maxEmbedSeconds?: number;
  baseHistorySeconds?: number;
  maxHistorySeconds?: number;
}

interface KeptPending {
  span: Span;
  /** Audio seconds where the kept chunk starts (lead context applied). */
  leadStart: number;
  /** Audio seconds where the kept chunk should end (trail context; may be
   *  cut shorter by a new speech onset). */
  trailEnd: number;
  score: number | null;
  basis: MatchBasis | null;
}

export class JournalLoop {
  private readonly segmenter: StreamingSegmenter;
  private readonly now: () => number;
  private readonly contextSeconds: number;
  private readonly maxEmbedSamples: number;
  private readonly baseHistorySamples: number;
  private readonly maxHistorySamples: number;

  private vad: FrameVad;
  private vadDegraded = false;
  private running = false;
  private samplesSeen = 0;
  private pending = new Int16Array(0);
  private history: { startSample: number; data: Int16Array }[] = [];
  private vadQueue: Promise<void> = Promise.resolve();
  private turnQueue: Promise<void> = Promise.resolve();
  /** Spans finalized but whose finalize task has not sliced its PCM yet
   *  (their audio must survive history trimming). */
  private awaiting: Span[] = [];
  /** Self segments waiting for their trailing context to exist. */
  private kept: KeptPending[] = [];
  private lastFrameEnd = 0;
  /** Wall clock ↔ audio clock anchor, taken when samples ARRIVE (the mic
   *  callback is real time; frame processing may lag it by a queue). */
  private wallAtPush = 0;
  private samplesAtPush = 0;
  private wasInSpeech = false;
  /** End of the most recent finalized speech run (lead-context floor). */
  private prevSpanEnd = 0;
  /** End of the most recent written chunk (never duplicate audio). */
  private prevWrittenEnd = 0;

  private stats: JournalLoopStats = {
    listeningSeconds: 0,
    selfCount: 0,
    selfSeconds: 0,
    lastSelfWallMs: null,
    discardedCount: 0,
    discardedSeconds: 0,
  };

  constructor(private readonly deps: JournalLoopDeps) {
    this.segmenter = new StreamingSegmenter({
      mergeGapSeconds: deps.mergeGapSeconds ?? JOURNAL_MERGE_GAP_SECONDS,
      minSeconds: deps.minSegmentSeconds ?? JOURNAL_MIN_SEGMENT_SECONDS,
    });
    this.now = deps.now ?? Date.now;
    this.contextSeconds = deps.contextSeconds ?? JOURNAL_CONTEXT_SECONDS;
    this.maxEmbedSamples = Math.round((deps.maxEmbedSeconds ?? JOURNAL_MAX_EMBED_SECONDS) * SILERO_SAMPLE_RATE);
    this.baseHistorySamples = Math.round(
      (deps.baseHistorySeconds ?? JOURNAL_BASE_HISTORY_SECONDS) * SILERO_SAMPLE_RATE,
    );
    this.maxHistorySamples = Math.round(
      (deps.maxHistorySeconds ?? JOURNAL_MAX_HISTORY_SECONDS) * SILERO_SAMPLE_RATE,
    );
    this.vad = deps.vad;
  }

  get isRunning(): boolean {
    return this.running;
  }

  get isVadDegraded(): boolean {
    return this.vadDegraded;
  }

  /** Session seconds by the audio clock (samples pushed so far). */
  get audioClock(): number {
    return this.samplesSeen / SILERO_SAMPLE_RATE;
  }

  /** A snapshot of the counters (listeningSeconds tracks the audio clock). */
  get statsSnapshot(): JournalLoopStats {
    return { ...this.stats, listeningSeconds: this.audioClock };
  }

  start(): void {
    this.running = true;
    this.samplesSeen = 0;
    this.pending = new Int16Array(0);
    this.history = [];
    this.awaiting = [];
    this.kept = [];
    this.lastFrameEnd = 0;
    this.wallAtPush = this.now();
    this.samplesAtPush = 0;
    this.wasInSpeech = false;
    this.prevSpanEnd = 0;
    this.prevWrittenEnd = 0;
    this.stats = {
      listeningSeconds: 0,
      selfCount: 0,
      selfSeconds: 0,
      lastSelfWallMs: null,
      discardedCount: 0,
      discardedSeconds: 0,
    };
    this.segmenter.reset();
    this.vad = this.deps.vad;
    this.vadDegraded = false;
    this.vad.reset();
    this.deps.labeler.reset();
  }

  /** Feed 16 kHz mono int16 samples (any length). Synchronous; VAD work is
   *  queued so frames are never dropped or reordered. */
  pushSamples(samples: Int16Array): void {
    if (!this.running || samples.length === 0) return;
    // Copy: the hook reuses/transfers frame buffers to other consumers.
    this.history.push({ startSample: this.samplesSeen, data: samples.slice() });
    this.samplesSeen += samples.length;
    this.wallAtPush = this.now();
    this.samplesAtPush = this.samplesSeen;
    this.trimHistory();

    const joined = new Int16Array(this.pending.length + samples.length);
    joined.set(this.pending, 0);
    joined.set(samples, this.pending.length);
    const frameN = this.vad.frameSamples;
    let offset = 0;
    const base = this.samplesSeen - joined.length;
    while (joined.length - offset >= frameN) {
      const frame = new Float32Array(frameN);
      for (let i = 0; i < frameN; i++) frame[i] = joined[offset + i] / 32768;
      const startSample = base + offset;
      offset += frameN;
      this.vadQueue = this.vadQueue.then(() => this.runFrame(frame, startSample)).catch(() => {});
    }
    this.pending = joined.slice(offset);
  }

  /** Wait for every queued frame and segment to finish (tests, stop()). */
  async settle(): Promise<void> {
    await this.vadQueue;
    await this.turnQueue;
    // A segment finalized by the last frame may have queued further work.
    await this.vadQueue;
    await this.turnQueue;
  }

  /** Flush the open run, wait for every queued stage, write what is kept. */
  async stop(): Promise<JournalLoopStats> {
    this.running = false;
    await this.vadQueue;
    const tail = this.segmenter.flush();
    if (tail) this.enqueueSegment(tail);
    await this.turnQueue;
    // Whatever trailing context exists is all there will be.
    this.drainKept(true);
    return this.statsSnapshot;
  }

  // -------------------------------------------------------------------------

  private trimHistory(): void {
    // Keep everything an open run, a span awaiting finalization, or a kept
    // segment awaiting its trail still needs — bounded by the hard cap.
    const ctxSamples = Math.round(this.contextSeconds * SILERO_SAMPLE_RATE);
    let keepFrom = this.samplesSeen - this.baseHistorySamples;
    const active = this.segmenter.active;
    if (active) keepFrom = Math.min(keepFrom, Math.round(active.start * SILERO_SAMPLE_RATE) - ctxSamples);
    for (const s of this.awaiting) {
      keepFrom = Math.min(keepFrom, Math.round(s.start * SILERO_SAMPLE_RATE) - ctxSamples);
    }
    if (this.kept.length > 0) {
      keepFrom = Math.min(keepFrom, Math.round(this.kept[0].leadStart * SILERO_SAMPLE_RATE));
    }
    keepFrom = Math.max(keepFrom, this.samplesSeen - this.maxHistorySamples);
    while (this.history.length > 0) {
      const h = this.history[0];
      if (h.startSample + h.data.length <= keepFrom) this.history.shift();
      else break;
    }
  }

  private earliestHistorySeconds(): number {
    return this.history.length > 0 ? this.history[0].startSample / SILERO_SAMPLE_RATE : this.audioClock;
  }

  private async runFrame(frame: Float32Array, startSample: number): Promise<void> {
    let isSpeech: boolean;
    try {
      isSpeech = await this.vad.isSpeech(frame);
    } catch (err) {
      // Same rule as the fast loop: a detector that throws is gone; the
      // energy rule takes over for the rest of the session.
      if (!this.vadDegraded) {
        this.vadDegraded = true;
        this.vad = new EnergyVad(undefined, frame.length / SILERO_SAMPLE_RATE);
        this.deps.onDegrade?.("vad", err instanceof Error ? err.message : String(err));
      }
      isSpeech = await this.vad.isSpeech(frame);
    }
    const tStart = startSample / SILERO_SAMPLE_RATE;
    const tEnd = (startSample + frame.length) / SILERO_SAMPLE_RATE;
    this.lastFrameEnd = tEnd;
    const span = this.segmenter.push(isSpeech, tStart, tEnd);
    const inSpeech = this.segmenter.inSpeech;
    if (inSpeech && !this.wasInSpeech) {
      // A new speech run begins: pending trails end here (bounded by silence).
      for (const k of this.kept) {
        if (k.trailEnd > tStart && k.span.end <= tStart) k.trailEnd = tStart;
      }
    }
    this.wasInSpeech = inSpeech;
    if (span) this.enqueueSegment(span);
    this.drainKept(false);
  }

  private enqueueSegment(span: Span): void {
    this.awaiting.push(span);
    this.turnQueue = this.turnQueue.then(() => this.finalizeSegment(span)).catch(() => {});
  }

  /** Wall-clock ms for an audio-clock second, anchored on the latest push. */
  private wallOf(audioSeconds: number): number {
    return Math.round(this.wallAtPush - (this.samplesAtPush / SILERO_SAMPLE_RATE - audioSeconds) * 1000);
  }

  private sliceInt16(fromSeconds: number, toSeconds: number): Int16Array {
    const a = Math.round(fromSeconds * SILERO_SAMPLE_RATE);
    const b = Math.round(toSeconds * SILERO_SAMPLE_RATE);
    const out = new Int16Array(Math.max(0, b - a));
    for (const h of this.history) {
      const hEnd = h.startSample + h.data.length;
      if (hEnd <= a || h.startSample >= b) continue;
      const from = Math.max(a, h.startSample);
      const to = Math.min(b, hEnd);
      out.set(h.data.subarray(from - h.startSample, to - h.startSample), from - a);
    }
    return out;
  }

  private async finalizeSegment(span: Span): Promise<void> {
    const duration = span.end - span.start;
    // The lead-context floor for the NEXT segment, whatever this one is.
    const leadFloor = Math.max(this.prevSpanEnd, this.prevWrittenEnd, this.earliestHistorySeconds());
    this.prevSpanEnd = span.end;
    // Embed the tail of the span (identity saturates in a few seconds).
    const embedFrom = Math.max(span.start, span.end - this.maxEmbedSamples / SILERO_SAMPLE_RATE);
    const int16 = this.sliceInt16(embedFrom, span.end);
    // This span's PCM is in hand: the history may roll past it now.
    this.awaiting = this.awaiting.filter((s) => s !== span);
    const discard = (reason: DiscardedSegmentMeta["reason"], isSelf: boolean | null, score: number | null, basis: MatchBasis | null) => {
      this.stats.discardedCount += 1;
      this.stats.discardedSeconds += duration;
      this.deps.onDiscard?.({ speechSeconds: duration, isSelf, score, basis, reason });
    };
    if (!this.deps.embedder) {
      discard("no-embedder", null, null, null);
      return;
    }
    const pcm = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) pcm[i] = int16[i] / 32768;
    let verdict: SpeakerVerdict;
    try {
      const emb = await this.deps.embedder.embed(pcm, SILERO_SAMPLE_RATE);
      verdict = this.deps.labeler.label(emb, duration);
    } catch {
      discard("unembeddable", null, null, null);
      return;
    }
    if (verdict.isSelf !== true) {
      // SOLO rule (journal only): with no second voice in the session the
      // contrast rule cannot run, and the 0.65 absolute bar is rarely met
      // across rooms (owner measured 0.24-0.45; strangers <= 0.28). A lone
      // voice at >= SOLO_SELF_THRESHOLD against a print pooled from
      // >= SOLO_SELF_MIN_SETTINGS recordings is kept as the owner.
      const solo =
        this.deps.labeler.clusterCount <= 1 &&
        this.deps.labeler.selfSettings >= SOLO_SELF_MIN_SETTINGS &&
        typeof verdict.selfScore === "number" &&
        verdict.selfScore >= SOLO_SELF_THRESHOLD;
      if (!solo) {
        discard(verdict.isSelf === false ? "other" : "unidentified", verdict.isSelf, verdict.score, verdict.basis);
        return;
      }
      verdict = { ...verdict, isSelf: true, score: verdict.selfScore ?? null, basis: "solo" };
    }
    this.stats.selfCount += 1;
    this.stats.selfSeconds += duration;
    this.stats.lastSelfWallMs = this.wallOf(span.end);
    this.kept.push({
      span,
      leadStart: Math.max(span.start - this.contextSeconds, leadFloor),
      trailEnd: span.end + this.contextSeconds,
      score: verdict.score,
      basis: verdict.basis,
    });
    // If a new run already began after this span (the embedding took a
    // moment), its trail is bounded by that onset.
    const active = this.segmenter.active;
    if (active && active.start > span.end) {
      const k = this.kept[this.kept.length - 1];
      k.trailEnd = Math.min(k.trailEnd, active.start);
    }
    this.drainKept(false);
  }

  /** Write kept segments whose trailing context now exists, in order. */
  private drainKept(force: boolean): void {
    while (this.kept.length > 0) {
      const k = this.kept[0];
      if (!force && this.lastFrameEnd < k.trailEnd) return;
      this.kept.shift();
      const from = Math.max(k.leadStart, this.earliestHistorySeconds());
      const to = Math.min(k.trailEnd, this.lastFrameEnd);
      if (to <= from) continue;
      const pcm = this.sliceInt16(from, to);
      this.prevWrittenEnd = to;
      try {
        this.deps.keep(pcm, {
          startWallMs: this.wallOf(k.span.start),
          leadSeconds: Math.max(0, k.span.start - from),
          speechSeconds: k.span.end - k.span.start,
          trailSeconds: Math.max(0, to - k.span.end),
          score: k.score,
          basis: k.basis,
        });
      } catch {
        // A sink failure is the sink's to report (the store closes itself);
        // the loop keeps listening.
      }
    }
  }
}
