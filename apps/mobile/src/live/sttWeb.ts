/**
 * `SpeechRecognizer` over the browser's Web Speech API — the web build's
 * stand-in for expo-speech-recognition. The target is iOS Safari
 * (`webkitSpeechRecognition`, Apple's recognizer); Chrome works the same way
 * (Google's). Honesty note: unlike the native path, the browser sends the
 * audio to the vendor's speech service — "on-device" in the protocol means
 * "the client produced the words, not the server's Deepgram", which is what
 * the server keys on; the status line on screen says "browser speech".
 *
 * Quirks this class absorbs, all observed on iOS Safari:
 *
 * - `start()` must happen inside a user gesture (the permission prompt is
 *   gated on it). `primeWebRecognizer()` in webDeps.ts is called
 *   synchronously from the Start button's handler; the fast loop's own
 *   `start()` later is idempotent.
 * - Recognition ENDS on its own: after ~60 s, after a pause, when the
 *   service hiccups. `onend` restarts it (with a short back-off) until
 *   `stop()`; the gap costs a few hundred ms of speech — a known limit.
 * - Finals are unreliable: Safari often reports one ever-growing interim
 *   result and only flags it final at `end` (sometimes never). So an
 *   interim that is still pending when the session ends is emitted AS the
 *   final, and every final carries synthesized word timing spread over the
 *   window in which its interims were seen — that is what keeps the
 *   TranscriptAligner from smearing a 60-second-late final over a minute.
 * - `no-speech` / `aborted` errors are the recognizer being idle, not
 *   broken (they precede an `end` that restarts it); `not-allowed`,
 *   `service-not-allowed`, `audio-capture` and `network` are fatal and
 *   reported once, so the loop can hand the transcript back to the server.
 */
import type { SpeechRecognizer, SttResultEvent } from "./stt";

/** The slice of the Web Speech API we drive (structural: tests fake it). */
export interface SpeechRecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onresult: ((e: SpeechRecognitionEventLike) => void) | null;
  onerror: ((e: { error: string; message?: string }) => void) | null;
  onend: (() => void) | null;
  onstart?: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

export interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: ArrayLike<{
    isFinal: boolean;
    length: number;
    0?: { transcript: string; confidence?: number };
    item?(i: number): { transcript: string; confidence?: number };
  }>;
}

export type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

/** The constructor this browser exposes, if any. */
export function webSpeechRecognitionCtor(
  g: Record<string, unknown> = globalThis as Record<string, unknown>,
): SpeechRecognitionCtor | null {
  const ctor = (g.SpeechRecognition ?? g.webkitSpeechRecognition) as SpeechRecognitionCtor | undefined;
  return typeof ctor === "function" ? ctor : null;
}

export function webSttAvailable(g?: Record<string, unknown>): boolean {
  return webSpeechRecognitionCtor(g) !== null;
}

export const FATAL_SPEECH_ERRORS = new Set([
  "not-allowed",
  "service-not-allowed",
  "audio-capture",
  "network",
  "language-not-supported",
]);

export interface WebSttOptions {
  lang?: string;
  ctor?: SpeechRecognitionCtor | null;
  now?: () => number;
  /** Delay before an automatic restart after `end` (ms). */
  restartDelayMs?: number;
  /** Give up restarting after this many consecutive restarts with no result. */
  maxSilentRestarts?: number;
  setTimeoutImpl?: (fn: () => void, ms: number) => unknown;
}

interface PendingResult {
  text: string;
  firstSeenMs: number;
  lastSeenMs: number;
}

/** Words a late final adds beyond its last interim are assumed to have been
 *  spoken within this long after that interim. */
const FINAL_GROWTH_MS = 1500;

/**
 * When a final's text is exactly the last interim, it was fully spoken by
 * the time that interim was seen (Safari flags finals late, sometimes by a
 * minute). If the final grew, the extra words came shortly after.
 */
function finalEndMs(seen: PendingResult | undefined, text: string, nowMs: number): number {
  if (!seen) return nowMs;
  if (seen.text === text) return seen.lastSeenMs;
  return Math.min(nowMs, seen.lastSeenMs + FINAL_GROWTH_MS);
}

export class WebSpeechRecognizer implements SpeechRecognizer {
  private rec: SpeechRecognitionLike | null = null;
  private started = false;
  private stopped = false;
  private fatal = false;
  private startPromise: Promise<void> | null = null;
  private resultCbs: ((e: SttResultEvent) => void)[] = [];
  private errorCbs: ((code: string, message: string) => void)[] = [];
  /** An error raised before any listener was attached (priming). */
  private deferredError: { code: string; message: string } | null = null;
  /** performance.now() at the FIRST start — segment times are relative to it
   *  across restarts, matching the aligner's single recognizer epoch. */
  private epochMs = 0;
  private pending = new Map<number, PendingResult>();
  private silentRestarts = 0;
  private sawResultThisRun = false;
  private readonly now: () => number;
  private readonly lang: string;
  private readonly ctor: SpeechRecognitionCtor | null;
  private readonly restartDelayMs: number;
  private readonly maxSilentRestarts: number;
  private readonly setTimeoutImpl: (fn: () => void, ms: number) => unknown;
  /** Restarts performed so far (diagnostics / tests). */
  restarts = 0;

  constructor(options: WebSttOptions = {}) {
    this.lang = options.lang ?? "en-US";
    this.ctor = options.ctor === undefined ? webSpeechRecognitionCtor() : options.ctor;
    this.now = options.now ?? (() => (typeof performance !== "undefined" ? performance.now() : Date.now()));
    this.restartDelayMs = options.restartDelayMs ?? 250;
    this.maxSilentRestarts = options.maxSilentRestarts ?? 40;
    this.setTimeoutImpl = options.setTimeoutImpl ?? ((fn, ms) => setTimeout(fn, ms));
  }

  get isStarted() {
    return this.started && !this.stopped;
  }

  /** Idempotent: a second call (the loop's, after priming) resolves at once. */
  start(): Promise<void> {
    if (this.startPromise) return this.startPromise;
    this.startPromise = (async () => {
      if (!this.ctor) throw new Error("this browser has no speech recognition (SpeechRecognition API)");
      if (this.deferredError) throw new Error(`${this.deferredError.code}: ${this.deferredError.message}`);
      this.stopped = false;
      this.fatal = false;
      this.epochMs = this.now();
      this.started = true;
      this.launch();
    })();
    return this.startPromise;
  }

  private launch() {
    if (!this.ctor || this.stopped || this.fatal) return;
    let rec: SpeechRecognitionLike;
    try {
      rec = new this.ctor();
    } catch (err) {
      this.raise("start-failed", err instanceof Error ? err.message : String(err));
      return;
    }
    rec.lang = this.lang;
    rec.continuous = true;
    rec.interimResults = true;
    rec.maxAlternatives = 1;
    rec.onresult = (e) => this.handleResult(e);
    rec.onerror = (e) => this.handleError(e.error, e.message ?? "");
    rec.onend = () => this.handleEnd(rec);
    this.rec = rec;
    this.sawResultThisRun = false;
    try {
      rec.start();
    } catch (err) {
      // InvalidStateError: a previous instance is still winding down. The
      // restart path retries after the delay; a first start reports.
      const msg = err instanceof Error ? err.message : String(err);
      if (this.restarts > 0) this.scheduleRestart();
      else this.raise("start-failed", msg);
    }
  }

  private handleResult(e: SpeechRecognitionEventLike) {
    if (this.stopped) return;
    this.sawResultThisRun = true;
    this.silentRestarts = 0;
    const t = this.now() - this.epochMs;
    const interim: string[] = [];
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      if (!r) continue;
      const alt = r[0] ?? r.item?.(0);
      const text = (alt?.transcript ?? "").trim();
      const key = i;
      if (r.isFinal) {
        const seen = this.pending.get(key);
        this.pending.delete(key);
        if (text) this.emitFinal(text, seen?.firstSeenMs ?? t, finalEndMs(seen, text, t));
        continue;
      }
      if (!text) continue;
      const seen = this.pending.get(key);
      if (seen) {
        seen.text = text;
        seen.lastSeenMs = t;
      } else {
        this.pending.set(key, { text, firstSeenMs: t, lastSeenMs: t });
      }
      interim.push(text);
    }
    if (interim.length > 0) {
      this.emit({ text: interim.join(" "), isFinal: false });
    }
  }

  /** A final with word timing synthesized over [firstSeen, lastSeen]. */
  private emitFinal(text: string, startMs: number, endMs: number) {
    const words = text.split(/\s+/).filter(Boolean);
    // A final with no interim history (a snap result) still gets a width:
    // ~120 ms per word, ending now.
    const span = endMs > startMs ? endMs - startMs : words.length * 120;
    const start = endMs - span;
    const per = span / words.length;
    this.emit({
      text,
      isFinal: true,
      segments: words.map((w, i) => ({
        startTimeMillis: Math.max(0, Math.round(start + i * per)),
        endTimeMillis: Math.max(0, Math.round(start + (i + 1) * per)),
        segment: w,
      })),
    });
  }

  /** Whatever was still interim becomes final (Safari never flags it) —
   *  timed over the window its interims were seen in, not up to now. */
  private flushPending() {
    const keys = Array.from(this.pending.keys()).sort((a, b) => a - b);
    for (const k of keys) {
      const p = this.pending.get(k);
      if (p && p.text) this.emitFinal(p.text, p.firstSeenMs, p.lastSeenMs);
    }
    this.pending.clear();
  }

  private handleError(code: string, message: string) {
    if (this.stopped) return;
    if (FATAL_SPEECH_ERRORS.has(code)) {
      this.fatal = true;
      this.flushPending();
      this.raise(code, message);
      return;
    }
    // no-speech / aborted / a transient blip: `end` follows and restarts.
  }

  private handleEnd(rec: SpeechRecognitionLike) {
    if (this.rec !== rec) return; // a superseded instance
    this.flushPending();
    if (this.stopped || this.fatal) return;
    if (!this.sawResultThisRun) {
      this.silentRestarts += 1;
      if (this.silentRestarts > this.maxSilentRestarts) {
        this.fatal = true;
        this.raise("restart-limit", `recognition kept ending with no results (${this.silentRestarts} restarts)`);
        return;
      }
    }
    this.scheduleRestart();
  }

  private scheduleRestart() {
    this.restarts += 1;
    this.setTimeoutImpl(() => {
      if (this.stopped || this.fatal) return;
      this.launch();
    }, this.restartDelayMs);
  }

  private raise(code: string, message: string) {
    if (this.errorCbs.length === 0) {
      this.deferredError = { code, message };
      return;
    }
    for (const cb of this.errorCbs) cb(code, message);
  }

  private emit(e: SttResultEvent) {
    for (const cb of this.resultCbs) cb(e);
  }

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    this.flushPending();
    const rec = this.rec;
    this.rec = null;
    if (rec) {
      rec.onresult = null;
      rec.onerror = null;
      rec.onend = null;
      try {
        rec.stop();
      } catch {
        // Already stopped.
      }
      try {
        rec.abort();
      } catch {
        // Already gone.
      }
    }
  }

  onResult(cb: (e: SttResultEvent) => void) {
    this.resultCbs.push(cb);
    return () => {
      this.resultCbs = this.resultCbs.filter((c) => c !== cb);
    };
  }

  onError(cb: (code: string, message: string) => void) {
    this.errorCbs.push(cb);
    // A failure that happened while primed (before the loop listened) is
    // delivered now rather than lost.
    const deferred = this.deferredError;
    if (deferred) {
      this.deferredError = null;
      cb(deferred.code, deferred.message);
    }
    return () => {
      this.errorCbs = this.errorCbs.filter((c) => c !== cb);
    };
  }
}
