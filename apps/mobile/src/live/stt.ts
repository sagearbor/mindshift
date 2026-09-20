/**
 * On-device speech-to-text for the fast loop.
 *
 * The OS recognizer (expo-speech-recognition: Apple Speech / Android's
 * on-device `com.google.android.as` service) runs CONTINUOUSLY over the
 * whole session and emits interim + final results on its own cadence; the
 * VAD/segmenter decides where turns start and end. `TranscriptAligner` is the
 * pure piece that joins the two: it keeps a timeline of recognized text and
 * answers "what was said during [start, end)".
 *
 * Time bases: everything is in SESSION seconds (the same sample-count clock
 * the segmenter uses). Android final results carry per-word
 * `startTimeMillis`/`endTimeMillis` relative to recognition start, which the
 * aligner maps onto the session clock; results without word timing (iOS,
 * interims) are attributed to the window since the previous final.
 *
 * `SpeechRecognizer` is the native seam; `FakeSpeechRecognizer` drives tests.
 */

export interface SttWord {
  text: string;
  /** Session seconds. */
  start: number;
  end: number;
}

export interface SttResultEvent {
  text: string;
  isFinal: boolean;
  /** Word timing relative to recognizer start (ms), when the platform gives it. */
  segments?: { startTimeMillis: number; endTimeMillis: number; segment: string }[];
}

export interface SpeechRecognizer {
  /** Begin continuous on-device recognition. Rejects when unavailable. */
  start(): Promise<void>;
  /** Stop and release. Must never throw. */
  stop(): void;
  onResult(cb: (e: SttResultEvent) => void): () => void;
  /** A FATAL failure: the recognizer is gone for the rest of the session.
   *  Transient ends (Android tears the session down on every error, even
   *  "no-speech" after a pause) are handled inside the recognizer by
   *  restarting — see `onRestart`. */
  onError(cb: (code: string, message: string) => void): () => void;
  /** Fires after the recognizer brought itself back (a new native session).
   *  Platform word timings restart from zero at that moment, so the aligner
   *  must re-mark its clock. Optional for implementations that never
   *  restart. */
  onRestart?(cb: () => void): () => void;
}

/** Test double: tests push results directly. */
export class FakeSpeechRecognizer implements SpeechRecognizer {
  started = false;
  stopped = false;
  private resultCbs: ((e: SttResultEvent) => void)[] = [];
  private errorCbs: ((code: string, message: string) => void)[] = [];
  private restartCbs: (() => void)[] = [];
  async start() {
    this.started = true;
  }
  stop() {
    this.stopped = true;
  }
  onResult(cb: (e: SttResultEvent) => void) {
    this.resultCbs.push(cb);
    return () => {
      this.resultCbs = this.resultCbs.filter((c) => c !== cb);
    };
  }
  onError(cb: (code: string, message: string) => void) {
    this.errorCbs.push(cb);
    return () => {
      this.errorCbs = this.errorCbs.filter((c) => c !== cb);
    };
  }
  onRestart(cb: () => void) {
    this.restartCbs.push(cb);
    return () => {
      this.restartCbs = this.restartCbs.filter((c) => c !== cb);
    };
  }
  emit(e: SttResultEvent) {
    for (const cb of this.resultCbs) cb(e);
  }
  emitError(code: string, message = "") {
    for (const cb of this.errorCbs) cb(code, message);
  }
  /** Simulate the recognizer restarting itself (Android after "no-speech"). */
  emitRestart() {
    for (const cb of this.restartCbs) cb();
  }
}

export interface AlignedText {
  text: string;
  /** false when only an interim (not yet final) result covered the span. */
  final: boolean;
}

/** How far outside a VAD span a word may sit and still belong to it: STT and
 *  VAD disagree by a few hundred ms at turn edges. */
export const ALIGN_SLACK_SECONDS = 0.35;

/** The words in `curr` beyond its longest common word-prefix with `prev`,
 *  comparing on a punctuation-stripped, lower-cased form (the recognizer
 *  re-punctuates as an utterance grows). Empty when `curr` only re-states
 *  `prev`. Turns a cumulative recognizer's repeated finals into the new suffix;
 *  when `curr` shares no prefix (a genuinely fresh utterance) it returns all of
 *  `curr`. English-only app, so ASCII stripping is enough. */
export function suffixWords(prev: string, curr: string): string[] {
  const norm = (w: string) => w.replace(/[^a-zA-Z0-9']/g, "").toLowerCase();
  const currWords = curr.split(/\s+/).filter(Boolean);
  const prevWords = prev.split(/\s+/).filter(Boolean);
  let i = 0;
  while (
    i < currWords.length &&
    i < prevWords.length &&
    norm(currWords[i]) === norm(prevWords[i])
  ) {
    i++;
  }
  // Only strip the prefix when `curr` STRICTLY GROWS from `prev`: prev is
  // entirely a word-prefix of curr AND curr adds more. That is the cumulative
  // re-emission we're fixing ("A B" → "A B C"). An identical re-emit (no growth)
  // or a divergent utterance that merely shares an opening ("I feel sad" → "I
  // feel angry") is a genuine new turn and keeps all its words — otherwise a
  // person repeating a phrase would be silently dropped.
  const strictGrowth =
    prevWords.length > 0 && i === prevWords.length && currWords.length > prevWords.length;
  return strictGrowth ? currWords.slice(i) : currWords;
}

export class TranscriptAligner {
  private words: (SttWord & { consumed: boolean })[] = [];
  private interim: { text: string; start: number; end: number } | null = null;
  /** Session second at which the recognizer's clock started. */
  private recognizerStart = 0;
  /** End of the last attributed text — the start of the next untimed window. */
  private cursor = 0;
  /** The previous CUMULATIVE final text, for the no-word-timing path. Android's
   *  SpeechRecognizer re-emits the whole utterance-so-far on each final result,
   *  so without this every final re-pushed already-transcribed words and the
   *  coach fired again on the same growing sentence (real Pixel 10, 2026-08-26:
   *  "Um, the house is dirty" → "Um, the house is dirty and I do all the work" →
   *  … each coached separately). We push only the new suffix. */
  private lastFinalText = "";

  constructor(private readonly slack = ALIGN_SLACK_SECONDS) {}

  /** Call when the recognizer starts, with the session clock at that moment. */
  markRecognizerStart(sessionSeconds: number) {
    this.recognizerStart = sessionSeconds;
    this.cursor = sessionSeconds;
    // A fresh recognizer session restarts its cumulative text from empty.
    this.lastFinalText = "";
  }

  /** Feed a recognizer event; `now` is the session clock when it arrived. */
  push(e: SttResultEvent, now: number) {
    const text = e.text.trim();
    if (!e.isFinal) {
      this.interim = text ? { text, start: this.cursor, end: now } : null;
      return;
    }
    this.interim = null;
    if (!text) return;
    if (e.segments && e.segments.length > 0) {
      for (const s of e.segments) {
        const w = s.segment.trim();
        if (!w) continue;
        this.words.push({
          text: w,
          start: this.recognizerStart + s.startTimeMillis / 1000,
          end: this.recognizerStart + s.endTimeMillis / 1000,
          consumed: false,
        });
      }
      this.cursor = Math.max(this.cursor, this.words[this.words.length - 1].end);
      return;
    }
    // No word timing: the recognizer re-sends the whole utterance-so-far on
    // each final, so contribute only the NEW words beyond the previous
    // cumulative final. The common prefix is compared word-by-word on a
    // punctuation-stripped, lower-cased form because the recognizer re-punctuates
    // as it grows ("dirty and." → "dirty, and I…"), which a raw startsWith would
    // miss and re-push everything.
    const parts = suffixWords(this.lastFinalText, text);
    this.lastFinalText = text;
    if (parts.length === 0) return;
    const start = this.cursor;
    const width = Math.max(now - start, 0.001);
    parts.forEach((p, i) => {
      this.words.push({
        text: p,
        start: start + (i / parts.length) * width,
        end: start + ((i + 1) / parts.length) * width,
        consumed: false,
      });
    });
    this.cursor = now;
  }

  /** Unconsumed final words whose midpoint falls inside the (slack-padded)
   *  span, marked consumed; else the overlapping interim text, if any. */
  textForSpan(start: number, end: number): AlignedText {
    const lo = start - this.slack;
    const hi = end + this.slack;
    const picked: string[] = [];
    for (const w of this.words) {
      if (w.consumed) continue;
      const mid = (w.start + w.end) / 2;
      if (mid >= lo && mid <= hi) {
        picked.push(w.text);
        w.consumed = true;
      }
    }
    if (picked.length > 0) return { text: picked.join(" "), final: true };
    if (this.interim && this.interim.end >= lo && this.interim.start <= hi) {
      return { text: this.interim.text, final: false };
    }
    return { text: "", final: true };
  }

  /** True when a final word already covers the span's tail (no need to wait). */
  hasFinalCovering(end: number): boolean {
    return this.words.some((w) => !w.consumed && w.end >= end - this.slack);
  }

  reset() {
    this.words = [];
    this.interim = null;
    this.cursor = this.recognizerStart;
  }
}
