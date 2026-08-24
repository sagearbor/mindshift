/**
 * The replay harness's stand-ins for the phone's OS services. Everything the
 * real loop would get from Android/iOS (continuous STT, the on-device LLM,
 * expo-speech) is replaced by a fake that is driven by the SCRIPT and the
 * VIRTUAL CLOCK — so what is exercised for real is the loop itself, Silero,
 * the segmenter, ECAPA, the aligner, the provider chain and the nudge policy.
 *
 * - `ScriptedRecognizer` behaves like expo-speech-recognition in continuous
 *   mode: interim results while a scripted turn is in progress, one final
 *   result (with per-word timings, Android-style, or without, iOS-style)
 *   `finalLatencyMs` after the turn's scripted end.
 * - `scriptedProvider` is a `SuggestionProvider` that answers after a virtual
 *   latency with a canned suggestion and a `text_tone` derived from the
 *   matched turn's `emotion_coarse`; optionally it "refuses" every k-th call
 *   the way a guard-railed on-device model would, so the chain's fall-through
 *   runs.
 * - `SpokenLog` is expo-speech: it records what would have been said, when.
 * - `TrackedVad` / `TrackedEmbedder` wrap the real models: they register the
 *   native call with the `InflightTracker` (so the clock waits for it) and
 *   record every VAD verdict for the "never speak over speech" check.
 */
import type { FrameVad } from "../vad";
import { SILERO_SAMPLE_RATE } from "../vad";
import type { Embedder } from "../speakerId";
import type { SpeechRecognizer, SttResultEvent } from "../stt";
import type { SuggestInput, SuggestOutput, SuggestionProvider, TextTone } from "../localLlm";
import { EMPTY_TONE } from "../localLlm";
import type { NudgeEvent, NudgePolicy, VectorEvent } from "../nudgePolicy";
import type { EmotionCoarse, ReplayScript, ScriptTurn } from "./meta";
import type { InflightTracker, VirtualClock } from "./virtualClock";

// ---------------------------------------------------------------------------
// STT
// ---------------------------------------------------------------------------

export interface ScriptedRecognizerOptions {
  /** Final result arrives this long after the scripted turn ends. */
  finalLatencyMs: number;
  /** Interim cadence while a turn is in progress (0 = no interims). */
  interimEveryMs: number;
  /** Android-style per-word timings on finals; false = iOS-style untimed. */
  wordTimings: boolean;
}

export const DEFAULT_STT_OPTIONS: ScriptedRecognizerOptions = {
  finalLatencyMs: 500,
  interimEveryMs: 300,
  wordTimings: true,
};

export interface EmittedStt {
  /** Audio second at which the event was emitted. */
  at: number;
  turnIndex: number;
  isFinal: boolean;
  text: string;
}

export class ScriptedRecognizer implements SpeechRecognizer {
  started = false;
  stopped = false;
  /** Audio second at which start() was called (recognizer clock origin). */
  startedAt = 0;
  readonly emitted: EmittedStt[] = [];
  private resultCbs: ((e: SttResultEvent) => void)[] = [];
  private errorCbs: ((code: string, message: string) => void)[] = [];
  private finalsDone = new Set<number>();
  private lastInterimAt = new Map<number, number>();
  private readonly turns: ScriptTurn[];

  constructor(
    script: ReplayScript,
    private readonly opts: ScriptedRecognizerOptions = DEFAULT_STT_OPTIONS,
    private readonly audioNow: () => number = () => 0,
  ) {
    this.turns = script.turns.filter((t) => t.text.trim().length > 0);
  }

  async start() {
    this.started = true;
    this.startedAt = this.audioNow();
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

  /** Drive from the replay: called once per delivered frame with the audio
   *  clock (seconds of PCM pushed so far). Emits every result now due. */
  tick(audioSeconds: number) {
    if (!this.started || this.stopped) return;
    const latency = this.opts.finalLatencyMs / 1000;
    for (const t of this.turns) {
      if (this.finalsDone.has(t.index)) continue;
      if (audioSeconds >= t.end + latency) {
        this.finalsDone.add(t.index);
        this.emit(this.finalFor(t), t.index, audioSeconds);
        continue;
      }
      // Interim while the words are being spoken (and until the final
      // lands): a growing prefix proportional to how far into the turn we are.
      if (this.opts.interimEveryMs > 0 && audioSeconds >= t.start + 0.3 && audioSeconds < t.end + latency) {
        const last = this.lastInterimAt.get(t.index) ?? -Infinity;
        if (audioSeconds - last >= this.opts.interimEveryMs / 1000 - 1e-9) {
          const words = t.text.split(/\s+/);
          const frac = Math.min(1, Math.max(0, (audioSeconds - t.start) / Math.max(0.001, t.end - t.start)));
          const n = Math.max(1, Math.floor(words.length * frac));
          this.lastInterimAt.set(t.index, audioSeconds);
          this.emit({ text: words.slice(0, n).join(" "), isFinal: false }, t.index, audioSeconds);
        }
      }
    }
  }

  private finalFor(t: ScriptTurn): SttResultEvent {
    if (!this.opts.wordTimings) return { text: t.text, isFinal: true };
    const words = t.text.split(/\s+/);
    const width = (t.end - t.start) / words.length;
    return {
      text: t.text,
      isFinal: true,
      segments: words.map((w, i) => ({
        startTimeMillis: Math.round((t.start - this.startedAt + i * width) * 1000),
        endTimeMillis: Math.round((t.start - this.startedAt + (i + 1) * width) * 1000),
        segment: w,
      })),
    };
  }

  private emit(e: SttResultEvent, turnIndex: number, at: number) {
    this.emitted.push({ at, turnIndex, isFinal: e.isFinal, text: e.text });
    for (const cb of this.resultCbs) cb(e);
  }
}

// ---------------------------------------------------------------------------
// LLM
// ---------------------------------------------------------------------------

/** `emotion_coarse` -> the text tone a well-behaved on-device model would
 *  score. Integers (the wire contract is `int | None`). "angry" lands at
 *  aggressive_tone level 2 (nudgePolicy.aggressiveToneLevel: >= 70) so the
 *  policy can escalate on tone alone; loudness pushes a shout to 3. */
export const TONE_BY_EMOTION: Record<EmotionCoarse, TextTone> = {
  neutral: { warmth: 40, defensiveness: 15, sarcasm: 5, sadness: 10, frustration: 15, label: "neutral" },
  angry: { warmth: 10, defensiveness: 60, sarcasm: 20, sadness: 10, frustration: 72, label: "angry" },
  sad: { warmth: 35, defensiveness: 20, sarcasm: 5, sadness: 75, frustration: 20, label: "sad" },
  happy: { warmth: 85, defensiveness: 5, sarcasm: 5, sadness: 5, frustration: 5, label: "happy" },
};

const EMOTION_PRIORITY: EmotionCoarse[] = ["angry", "sad", "happy", "neutral"];

/** The scripted turns whose text appears in what STT handed the LLM. */
export function turnsInText(script: ReplayScript, text: string): ScriptTurn[] {
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]+/g, " ").replace(/\s+/g, " ").trim();
  const hay = norm(text);
  if (!hay) return [];
  return script.turns.filter((t) => {
    const needle = norm(t.text);
    // A VAD fragment / interim carries a substring of the turn; a merged
    // span carries several whole turns. Accept containment either way.
    return needle.length > 0 && (hay.includes(needle) || (hay.length >= 12 && needle.includes(hay)));
  });
}

export function toneForTurns(turns: ScriptTurn[]): TextTone {
  let best: EmotionCoarse | null = null;
  for (const t of turns) {
    if (!t.emotionCoarse) continue;
    if (best === null || EMOTION_PRIORITY.indexOf(t.emotionCoarse) < EMOTION_PRIORITY.indexOf(best)) best = t.emotionCoarse;
  }
  return best ? TONE_BY_EMOTION[best] : EMPTY_TONE;
}

export interface ScriptedProviderOptions {
  name: string;
  latencyMs: number;
  /** Refuse every k-th call (0/undefined = never). */
  refuseEveryK?: number;
  available?: boolean;
}

export interface ProviderCall {
  n: number;
  atMs: number;
  input: SuggestInput;
  outcome: "ok" | "refused";
}

export const REFUSAL_TEXT = "I can't help with that request.";

export function scriptedProvider(
  script: ReplayScript,
  clock: VirtualClock,
  opts: ScriptedProviderOptions,
): SuggestionProvider & { calls: ProviderCall[] } {
  let n = 0;
  const calls: ProviderCall[] = [];
  return {
    name: opts.name,
    calls,
    async isAvailable() {
      return opts.available ?? true;
    },
    async suggest(input: SuggestInput): Promise<SuggestOutput | null> {
      n += 1;
      await clock.sleep(opts.latencyMs);
      const k = opts.refuseEveryK ?? 0;
      if (k > 0 && n % k === 0) {
        calls.push({ n, atMs: clock.now(), input, outcome: "refused" });
        return { suggestion: REFUSAL_TEXT, textTone: EMPTY_TONE };
      }
      calls.push({ n, atMs: clock.now(), input, outcome: "ok" });
      const matched = turnsInText(script, input.text);
      const tone = toneForTurns(matched);
      const suggestion = input.isSelf
        ? tone.label === "angry"
          ? "ease up"
          : "keep going"
        : `Try: "I hear you — ${firstWords(input.text, 4)}…"`;
      return { suggestion, textTone: tone };
    },
  };
}

function firstWords(text: string, n: number): string {
  return text.split(/\s+/).slice(0, n).join(" ");
}

// ---------------------------------------------------------------------------
// TTS
// ---------------------------------------------------------------------------

export interface SpokenLine {
  text: string;
  /** Virtual ms at which speak() was called. */
  atMs: number;
  /** The same instant on the audio timeline (seconds). */
  atSec: number;
  /** The VAD verdict the loop had processed last when it spoke. */
  vadSpeechKnown: boolean | null;
}

export class SpokenLog {
  readonly lines: SpokenLine[] = [];
  constructor(
    private readonly clock: VirtualClock,
    /** The loop's most recently PROCESSED VAD verdict — what it knew when
     *  it chose to speak (null = no verdict yet / not tracked). */
    private readonly vadKnown: () => boolean | null = () => null,
  ) {}
  speak = (text: string) => {
    this.lines.push({ text, atMs: this.clock.now(), atSec: this.clock.now() / 1000, vadSpeechKnown: this.vadKnown() });
  };
}

// ---------------------------------------------------------------------------
// Model wrappers
// ---------------------------------------------------------------------------

export interface VadVerdict {
  start: number;
  end: number;
  speech: boolean;
}

export class TrackedVad implements FrameVad {
  readonly frameSamples: number;
  readonly verdicts: VadVerdict[] = [];
  private chunk = 0;
  constructor(
    private readonly inner: FrameVad,
    private readonly tracker: InflightTracker,
  ) {
    this.frameSamples = inner.frameSamples;
  }
  async isSpeech(frame: Float32Array): Promise<boolean> {
    const start = (this.chunk * this.frameSamples) / SILERO_SAMPLE_RATE;
    this.chunk += 1;
    const end = (this.chunk * this.frameSamples) / SILERO_SAMPLE_RATE;
    const speech = await this.tracker.track("vad", this.inner.isSpeech(frame));
    this.verdicts.push({ start, end, speech });
    return speech;
  }
  reset() {
    this.inner.reset();
    this.chunk = 0;
    this.verdicts.length = 0;
  }
  /** The verdict most recently handed to the loop. */
  get lastVerdict(): boolean | null {
    return this.verdicts.length ? this.verdicts[this.verdicts.length - 1].speech : null;
  }
  /** Was the detector reporting speech at audio second `t`? (Last verdict
   *  ending at or before t — what the loop could know at that instant.) */
  speechAt(t: number): boolean | null {
    let last: VadVerdict | null = null;
    for (const v of this.verdicts) {
      if (v.end <= t + 1e-9) last = v;
      else break;
    }
    return last ? last.speech : null;
  }
}

export class TrackedEmbedder implements Embedder {
  constructor(
    private readonly inner: Embedder,
    private readonly tracker: InflightTracker,
    private readonly clock: VirtualClock,
    /** Virtual cost charged per embed (a phone-side estimate; the real wall
     *  time is recorded separately and never pinned). */
    private readonly costMs: number,
  ) {}
  async embed(pcm: Float32Array, sampleRate: number): Promise<Float32Array> {
    const out = await this.tracker.track("ecapa", this.inner.embed(pcm, sampleRate));
    if (this.costMs > 0) await this.clock.sleep(this.costMs);
    return out;
  }
}

// ---------------------------------------------------------------------------
// Nudge policy spy
// ---------------------------------------------------------------------------

export interface PolicyCall {
  t: number;
  events: VectorEvent[];
  /** Highest raw event level handed to the policy this call (0 when the
   *  turn wasn't the coached user's). */
  rawLevel: number;
  emitted: NudgeEvent[];
  levelAfter: number;
}

/** A NudgePolicy that records every call — the loop only reports the
 *  hysteresis-filtered events, but scoring against `expected_nudges` needs
 *  the per-turn escalation level too. */
export function recordingPolicy(inner: NudgePolicy): NudgePolicy & { log: PolicyCall[] } {
  const log: PolicyCall[] = [];
  const spy = Object.create(inner) as NudgePolicy & { log: PolicyCall[] };
  spy.log = log;
  spy.onEvents = (events: VectorEvent[], t: number) => {
    const emitted = inner.onEvents(events, t);
    log.push({
      t,
      events,
      rawLevel: events.reduce((m, e) => Math.max(m, e.level), 0),
      emitted,
      levelAfter: inner.current().A ?? 0,
    });
    return emitted;
  };
  return spy;
}
