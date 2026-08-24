/**
 * The on-device realtime coaching fast loop (Track 3-mobile).
 *
 *   PCM frames ─► VAD (Silero) ─► segmenter ─► turn end
 *                                                 │
 *                    ┌────────────────────────────┴───────────────┐
 *                    ▼                                            ▼
 *              prosody (sync)              speaker-ID (ECAPA) ‖ STT text for the span
 *                    └────────────────────────────┬───────────────┘
 *                                                 ▼
 *                                   local LLM chain (os → bundled → cloud)
 *                                                 ▼
 *                    speak (expo-speech) ─ send turn_local ─ show on screen
 *
 * No server is in this loop: the WebSocket keeps receiving PCM exactly as
 * before (the cloud augments later with `suggestion` / `tone_flag` /
 * `speaker_identity` events), but the words the user hears come from the
 * phone. Every stage is behind an injected seam (`FastLoopDeps`), so Jest
 * drives the whole loop with synthetic PCM, a fake recognizer and fake
 * providers — and the real Silero model via onnxruntime-node.
 *
 * Latency: each finalized turn logs per-stage `performance.now()` timings
 * (`latencyLog`); the hook prints them at session end. The demo target is
 * < 1.5 s from the end of the other person's turn to the first spoken word.
 *
 * Modes (PRD Tier 1 "push-to-suggest only fires when the coached person is
 * silent"): `earpiece` and `speaker` both speak, but never over live speech —
 * a suggestion that lands mid-utterance is HELD until the VAD goes quiet
 * (and dropped if that takes longer than `speakHoldMaxMs`, stale advice
 * being worse than none). `therapist` never speaks: on-screen only, both
 * partners enrolled, posted with mode "therapist". The cloud's suggestions
 * go through the same gate (`offerSpeech`), so nothing the phone voices —
 * local or cloud — talks over a live utterance.
 *
 * Degradation inside a running session: a VAD that throws (a lost ORT
 * session) is swapped for the energy VAD on the spot (`onDegrade`), and a
 * recognizer that has died stops the loop from claiming turns it never
 * heard — no `turn_local` goes out while STT is unavailable, so the
 * server's own transcript for those spans is not suppressed.
 */
import type { FrameVad } from "./vad";
import { EnergyVad, SILERO_SAMPLE_RATE } from "./vad";
import type { SegmenterConfig, Span } from "./segmenter";
import { DEFAULT_SEGMENTER_CONFIG, StreamingSegmenter } from "./segmenter";
import type { TurnProsody } from "./prosody";
import { LIVE_MAX_PITCH_SECONDS, turnProsodyAsync } from "./prosody";
import type { Embedder, SpeakerLabeler, SpeakerVerdict } from "./speakerId";
import type { SpeechRecognizer } from "./stt";
import { TranscriptAligner } from "./stt";
import type { LiveMode, ProviderChain, TextTone } from "./localLlm";
import type { HapticSink, NudgeEvent, NudgePolicy } from "./nudgePolicy";
import { LoudnessBaseline, phoneNudgePolicy, selfTurnVectorEvents } from "./nudgePolicy";
import type { TurnLocalEvent } from "./types";

export type SuggestionKind = "response" | "nudge";

/** Seconds of a turn handed to the voiceprint model, taken from the END of
 *  the turn. Identity saturates in a few seconds (the server pools up to
 *  60 s only for enrollment); an unbounded monologue would ship megabytes
 *  of samples across the native bridge per turn. */
export const MAX_EMBED_SECONDS = 10;

export interface TurnLatency {
  turn: number;
  /** Session ms at which the segmenter finalized the turn. */
  segmentEndMs: number;
  prosodyMs: number;
  speakerMs: number;
  sttWaitMs: number;
  llmMs: number;
  /** Segment end -> speak() called; null when nothing was spoken. */
  toSpeakMs: number | null;
  provider: string;
  held: boolean;
}

export interface LocalTurn {
  index: number;
  speaker: string;
  text: string;
  /** false when only an interim STT result covered the span. */
  transcriptFinal: boolean;
  startTime: number;
  endTime: number;
  isSelf: boolean | null;
  personId: string | null;
  matchScore: number | null;
  prosody: TurnProsody;
  textTone: TextTone | null;
  suggestion: string | null;
  suggestionKind: SuggestionKind | null;
  /** Which provider produced the suggestion ("cloud" = waiting on server). */
  provider: string;
  spoken: boolean;
  latency: TurnLatency;
}

export interface FastLoopDeps {
  vad: FrameVad;
  segmenter?: SegmenterConfig;
  /** Null => speaker-ID disabled (no ECAPA model): every turn is "Unknown". */
  embedder: Embedder | null;
  labeler: SpeakerLabeler | null;
  /** Null => no on-device STT: turns carry empty text and no suggestion. */
  recognizer: SpeechRecognizer | null;
  aligner?: TranscriptAligner;
  llm: ProviderChain;
  speak: (text: string) => void;
  send: (event: TurnLocalEvent) => void;
  onTurn: (turn: LocalTurn) => void;
  onNudge?: (nudge: NudgeEvent) => void;
  /** Called when STT fails after start (so the UI can say so honestly). */
  onSttError?: (code: string, message: string) => void;
  /** A stage fell back mid-session (today: the VAD to the energy rule). */
  onDegrade?: (stage: "vad", reason: string) => void;
  haptics?: HapticSink | null;
  policy?: NudgePolicy;
  now?: () => number;
  sleep?: (ms: number) => Promise<void>;
  /** How long to wait for the recognizer to deliver a span's words. */
  sttGraceMs?: number;
  pollMs?: number;
  /** Prior turns handed to the LLM as context. */
  contextTurns?: number;
  speakHoldMaxMs?: number;
  /** Minimum VAD-quiet (ms since the last speech frame) before a suggestion
   *  is voiced. 0 = speak as soon as the segmenter closes the turn (a 300 ms
   *  pause) — which the replay harness showed lands most suggestions inside
   *  the other person's sentence pauses; see the PR for the measurements. */
  speakQuietMs?: number;
  /** With no voiceprint verdict, which unknown-cluster label is treated as
   *  the coached user for coaching purposes (the app's "you speak first"
   *  convention). Null disables the convention. Changeable mid-session via
   *  `setSelfSpeakerFallback` (the screen's "You: Speaker A ⇄" chip). */
  selfSpeakerFallback?: string | null;
  /** Seconds of PCM kept for segment extraction. */
  historySeconds?: number;
  /** Cap on the PCM handed to the embedder (see MAX_EMBED_SECONDS). */
  maxEmbedSeconds?: number;
  /** Cap on the pitch-analysis window (see prosody.LIVE_MAX_PITCH_SECONDS). */
  maxPitchSeconds?: number;
}

export interface FastLoopSession {
  sessionId: string;
  mode: LiveMode;
  empathy: number;
}

export interface FastLoopSummary {
  turns: LocalTurn[];
  latencyLog: TurnLatency[];
  sttAvailable: boolean;
}

const defaultNow = () =>
  typeof performance !== "undefined" ? performance.now() : Date.now();
const defaultSleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

interface HeldSpeech {
  text: string;
  expiresAtMs: number;
  /** The local turn that produced it; null for a line offered from outside
   *  (the cloud's suggestion). */
  latency: TurnLatency | null;
  turn: LocalTurn | null;
}

export class FastLoop {
  private readonly segmenter: StreamingSegmenter;
  private readonly aligner: TranscriptAligner;
  private readonly policy: NudgePolicy;
  private readonly baseline = new LoudnessBaseline();
  private readonly now: () => number;
  private readonly sleep: (ms: number) => Promise<void>;
  private readonly sttGraceMs: number;
  private readonly pollMs: number;
  private readonly contextTurns: number;
  private readonly speakHoldMaxMs: number;
  private readonly speakQuietMs: number;
  private readonly historySamples: number;
  private readonly maxEmbedSamples: number;
  private readonly maxPitchSeconds: number;

  /** The detector in use — starts as deps.vad, swapped for the energy rule
   *  if it ever throws. */
  private vad: FrameVad;
  private vadDegraded = false;
  private selfSpeakerFallback: string | null;

  private session: FastLoopSession | null = null;
  private running = false;
  private startWallMs = 0;
  private samplesSeen = 0;
  private pending = new Float32Array(0);
  private history: { startSample: number; data: Float32Array }[] = [];
  private vadQueue: Promise<void> = Promise.resolve();
  private turnQueue: Promise<void> = Promise.resolve();
  private turns: LocalTurn[] = [];
  private held: HeldSpeech | null = null;
  /** Audio seconds: end of the most recent speech frame / most recent
   *  frame — quiet is measured on the frame clock, like the segmenter. */
  private lastSpeechEnd = -Infinity;
  private lastFrameEnd = 0;
  private sttAvailable = false;
  private sttStartSeconds = 0;
  private unsubscribe: (() => void)[] = [];

  readonly latencyLog: TurnLatency[] = [];

  constructor(private readonly deps: FastLoopDeps) {
    this.segmenter = new StreamingSegmenter(deps.segmenter ?? DEFAULT_SEGMENTER_CONFIG);
    this.aligner = deps.aligner ?? new TranscriptAligner();
    this.policy = deps.policy ?? phoneNudgePolicy();
    this.now = deps.now ?? defaultNow;
    this.sleep = deps.sleep ?? defaultSleep;
    this.sttGraceMs = deps.sttGraceMs ?? 700;
    this.pollMs = deps.pollMs ?? 50;
    this.contextTurns = deps.contextTurns ?? 6;
    this.speakHoldMaxMs = deps.speakHoldMaxMs ?? 3000;
    this.speakQuietMs = deps.speakQuietMs ?? 0;
    this.historySamples = Math.round((deps.historySeconds ?? 30) * SILERO_SAMPLE_RATE);
    this.maxEmbedSamples = Math.round((deps.maxEmbedSeconds ?? MAX_EMBED_SECONDS) * SILERO_SAMPLE_RATE);
    this.maxPitchSeconds = deps.maxPitchSeconds ?? LIVE_MAX_PITCH_SECONDS;
    this.vad = deps.vad;
    this.selfSpeakerFallback =
      deps.selfSpeakerFallback === undefined ? "Speaker A" : deps.selfSpeakerFallback;
  }

  get isRunning() {
    return this.running;
  }

  /** Session seconds by the audio clock (samples pushed so far). */
  get audioClock(): number {
    return this.samplesSeen / SILERO_SAMPLE_RATE;
  }

  /** Whether on-device STT is currently delivering words. */
  get sttIsAvailable(): boolean {
    return this.sttAvailable;
  }

  /** True once the VAD fell back to the energy rule this session. */
  get isVadDegraded(): boolean {
    return this.vadDegraded;
  }

  async start(session: FastLoopSession): Promise<void> {
    this.session = session;
    this.running = true;
    this.startWallMs = this.now();
    this.samplesSeen = 0;
    this.pending = new Float32Array(0);
    this.history = [];
    this.turns = [];
    this.held = null;
    this.lastSpeechEnd = -Infinity;
    this.lastFrameEnd = 0;
    this.latencyLog.length = 0;
    this.segmenter.reset();
    this.aligner.reset();
    this.vad = this.deps.vad;
    this.vadDegraded = false;
    this.vad.reset();
    this.deps.labeler?.reset();
    for (const u of this.unsubscribe) u();
    this.unsubscribe = [];

    const rec = this.deps.recognizer;
    if (rec) {
      this.unsubscribe.push(
        // STT events are stamped with the AUDIO clock (samples pushed so
        // far): in production it tracks wall time to within a buffer, and
        // it keeps the segmenter and the aligner on one time base.
        rec.onResult((e) => this.aligner.push(e, this.audioClock)),
        // The recognizer restarts itself after a transient end (Android
        // tears the native session down on every error, "no-speech" after
        // a pause included); only what it reports here is fatal.
        rec.onError((code, message) => {
          this.sttAvailable = false;
          this.deps.onSttError?.(code, message);
        }),
      );
      if (rec.onRestart) {
        // A fresh native session: platform word timings restart at zero.
        this.unsubscribe.push(rec.onRestart(() => this.aligner.markRecognizerStart(this.audioClock)));
      }
      try {
        this.aligner.markRecognizerStart(this.audioClock);
        await rec.start();
        this.sttStartSeconds = this.audioClock;
        this.aligner.markRecognizerStart(this.sttStartSeconds);
        this.sttAvailable = true;
      } catch (err) {
        this.sttAvailable = false;
        this.deps.onSttError?.("start-failed", err instanceof Error ? err.message : String(err));
      }
    }
  }

  setEmpathy(level: number) {
    if (this.session) this.session.empathy = level;
  }

  /** Which unknown-cluster label counts as the coached user when there is
   *  no voiceprint verdict (null = no convention). Takes effect from the
   *  next finalized turn. */
  setSelfSpeakerFallback(label: string | null) {
    this.selfSpeakerFallback = label;
  }

  /** Feed 16 kHz mono int16 samples (any length). Synchronous; VAD work is
   *  queued so frames are never dropped or reordered. */
  pushSamples(samples: Int16Array): void {
    if (!this.running || samples.length === 0) return;
    const f32 = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) f32[i] = samples[i] / 32768;
    this.history.push({ startSample: this.samplesSeen, data: f32 });
    this.samplesSeen += samples.length;
    this.trimHistory();

    const joined = new Float32Array(this.pending.length + f32.length);
    joined.set(this.pending, 0);
    joined.set(f32, this.pending.length);
    const frameN = this.vad.frameSamples;
    let offset = 0;
    // Session sample index of joined[0].
    const base = this.samplesSeen - joined.length;
    while (joined.length - offset >= frameN) {
      const frame = joined.slice(offset, offset + frameN);
      const startSample = base + offset;
      offset += frameN;
      this.vadQueue = this.vadQueue
        .then(() => this.runFrame(frame, startSample))
        .catch(() => {});
    }
    this.pending = joined.slice(offset);
  }

  /** Wait for every queued VAD frame and turn to finish (tests, stop()). */
  async settle(): Promise<void> {
    await this.vadQueue;
    await this.turnQueue;
  }

  /**
   * A line from outside the loop (the cloud's suggestion for a turn the
   * local providers passed on) that wants to be spoken: same rule as a local
   * one — never over live speech, never in therapist mode. Returns whether
   * the loop took it (false = this mode never speaks; the caller shows it
   * on screen only).
   */
  offerSpeech(text: string): boolean {
    if (!this.session || this.session.mode === "therapist") return false;
    if (!this.running) return false;
    if (!this.quietEnoughToSpeak()) {
      this.held = { text, expiresAtMs: this.now() + this.speakHoldMaxMs, latency: null, turn: null };
    } else {
      this.speakNow(text, null, null);
    }
    return true;
  }

  /** Flush the open turn, wait for every queued stage, stop STT. */
  async stop(): Promise<FastLoopSummary> {
    this.running = false;
    await this.vadQueue;
    const tail = this.segmenter.flush();
    if (tail) this.enqueueTurn(tail);
    await this.turnQueue;
    // The user ended the session: a line still waiting for a quiet moment
    // is dropped, never spoken after Stop.
    this.held = null;
    for (const u of this.unsubscribe) u();
    this.unsubscribe = [];
    try {
      this.deps.recognizer?.stop();
    } catch {
      // Recognizer teardown must never fail the session.
    }
    return { turns: [...this.turns], latencyLog: [...this.latencyLog], sttAvailable: this.sttAvailable };
  }

  // -------------------------------------------------------------------------

  private trimHistory() {
    const minStart = this.samplesSeen - this.historySamples;
    while (this.history.length > 0) {
      const h = this.history[0];
      if (h.startSample + h.data.length <= minStart) this.history.shift();
      else break;
    }
  }

  private async runFrame(frame: Float32Array, startSample: number) {
    let isSpeech: boolean;
    try {
      isSpeech = await this.vad.isSpeech(frame);
    } catch (err) {
      // A detector that throws is a detector that's gone (a lost native
      // session, a shape the model rejects): without a verdict no turn
      // would ever finalize and the screen would stay blank. The energy
      // rule takes over for the rest of the session; the UI is told once.
      if (!this.vadDegraded) {
        this.vadDegraded = true;
        this.vad = new EnergyVad();
        this.deps.onDegrade?.("vad", err instanceof Error ? err.message : String(err));
      }
      isSpeech = await this.vad.isSpeech(frame);
    }
    const tStart = startSample / SILERO_SAMPLE_RATE;
    const tEnd = (startSample + frame.length) / SILERO_SAMPLE_RATE;
    if (isSpeech) this.lastSpeechEnd = tEnd;
    this.lastFrameEnd = tEnd;
    const span = this.segmenter.push(isSpeech, tStart, tEnd);
    if (span) this.enqueueTurn(span);
    if (this.quietEnoughToSpeak()) this.releaseHeld();
  }

  /** Nobody is talking, and hasn't been for at least speakQuietMs. */
  private quietEnoughToSpeak(): boolean {
    if (this.segmenter.inSpeech) return false;
    return (this.lastFrameEnd - this.lastSpeechEnd) * 1000 >= this.speakQuietMs;
  }

  private enqueueTurn(span: Span) {
    const segmentEndMs = this.now() - this.startWallMs;
    this.turnQueue = this.turnQueue
      .then(() => this.finalizeTurn(span, segmentEndMs))
      .catch(() => {});
  }

  private sliceHistory(span: Span): Float32Array {
    const a = Math.round(span.start * SILERO_SAMPLE_RATE);
    const b = Math.round(span.end * SILERO_SAMPLE_RATE);
    const out = new Float32Array(Math.max(0, b - a));
    for (const h of this.history) {
      const hEnd = h.startSample + h.data.length;
      if (hEnd <= a || h.startSample >= b) continue;
      const from = Math.max(a, h.startSample);
      const to = Math.min(b, hEnd);
      out.set(h.data.subarray(from - h.startSample, to - h.startSample), from - a);
    }
    return out;
  }

  private async waitForText(span: Span): Promise<{ text: string; final: boolean; waitedMs: number }> {
    const t0 = this.now();
    if (!this.deps.recognizer || !this.sttAvailable) {
      return { text: "", final: true, waitedMs: 0 };
    }
    // Poll until final words cover the span or the grace window closes;
    // interim text is accepted at the deadline rather than nothing.
    for (;;) {
      const got = this.aligner.textForSpan(span.start, span.end);
      if (got.text && got.final) return { ...got, waitedMs: this.now() - t0 };
      if (this.now() - t0 >= this.sttGraceMs) {
        return { ...got, waitedMs: this.now() - t0 };
      }
      await this.sleep(this.pollMs);
    }
  }

  private async finalizeTurn(span: Span, segmentEndMs: number) {
    const session = this.session;
    if (!session) return;
    const index = this.turns.length;
    const pcm = this.sliceHistory(span);
    const duration = span.end - span.start;
    // Whether the words of this span were the phone's to report: a dead
    // recognizer means the server's transcript owns them (see send below).
    const sttOwned = this.deps.recognizer !== null && this.sttAvailable;

    // Speaker-ID and STT are independent — run them together.
    const speakerPromise = (async (): Promise<{ verdict: SpeakerVerdict; ms: number }> => {
      const t0 = this.now();
      let verdict: SpeakerVerdict = { speaker: "Unknown", personId: null, displayName: null, isSelf: null, score: null };
      if (this.deps.embedder && this.deps.labeler) {
        try {
          const embedPcm =
            pcm.length > this.maxEmbedSamples ? pcm.subarray(pcm.length - this.maxEmbedSamples) : pcm;
          const emb = await this.deps.embedder.embed(embedPcm, SILERO_SAMPLE_RATE);
          verdict = this.deps.labeler.label(emb, duration);
        } catch {
          // Unembeddable segment: no identity, never a guess.
        }
      }
      return { verdict, ms: this.now() - t0 };
    })();
    const textPromise = this.waitForText(span);
    const [{ verdict, ms: speakerMs }, aligned] = await Promise.all([speakerPromise, textPromise]);

    const tp0 = this.now();
    const prosody = await turnProsodyAsync(pcm, SILERO_SAMPLE_RATE, aligned.text, duration, {
      maxPitchSeconds: this.maxPitchSeconds,
      sleep: this.sleep,
    });
    const prosodyMs = this.now() - tp0;

    // Coaching identity: the voiceprint verdict when there is one, else the
    // "you speak first" convention (Speaker A) — never sent as is_self.
    const fallback = this.selfSpeakerFallback;
    const coachedAsSelf =
      verdict.isSelf === true ||
      (verdict.isSelf === null && fallback !== null && verdict.speaker === fallback);

    // Local LLM: only when there are words to coach on.
    let suggestion: string | null = null;
    let textTone: TextTone | null = null;
    let provider = "none";
    let llmMs = 0;
    if (aligned.text) {
      const tl0 = this.now();
      const context = this.turns
        .slice(-this.contextTurns)
        .map((t) => ({ speaker: t.speaker, text: t.text }))
        .filter((t) => t.text);
      const result = await this.deps.llm.suggest({
        text: aligned.text,
        speaker: verdict.speaker,
        isSelf: coachedAsSelf,
        empathy: session.empathy,
        context,
        prosodyHint: prosodyHint(prosody),
        mode: session.mode,
      });
      llmMs = this.now() - tl0;
      provider = result.provider;
      if (result.output) {
        suggestion = result.output.suggestion;
        textTone = result.output.textTone;
      }
    }

    const latency: TurnLatency = {
      turn: index,
      segmentEndMs,
      prosodyMs,
      speakerMs,
      sttWaitMs: aligned.waitedMs,
      llmMs,
      toSpeakMs: null,
      provider,
      held: false,
    };
    const turn: LocalTurn = {
      index,
      speaker: verdict.speaker,
      text: aligned.text,
      transcriptFinal: aligned.final,
      startTime: span.start,
      endTime: span.end,
      isSelf: verdict.isSelf,
      personId: verdict.personId,
      matchScore: verdict.score,
      prosody,
      textTone,
      suggestion,
      suggestionKind: suggestion ? (coachedAsSelf ? "nudge" : "response") : null,
      provider,
      spoken: false,
      latency,
    };
    this.turns.push(turn);
    this.latencyLog.push(latency);

    // Nudge policy over the user's own delivery; other turns tick the clock.
    const events = coachedAsSelf
      ? selfTurnVectorEvents(span.end, this.baseline.observe(prosody.rms_dbfs), textTone ?? { frustration: null, defensiveness: null })
      : [];
    for (const n of this.policy.onEvents(events, span.end)) {
      this.deps.onNudge?.(n);
      // Haptic on ESCALATION only (`vectors` names what raised the level).
      // A cooldown decay (level 2 -> 1, no vectors) updates the screen but
      // never buzzes — before this it buzzed if and only if the decay
      // happened to land on the user's own turn (replay-harness finding).
      if (n.level > 0 && n.vectors.length > 0 && this.deps.haptics) {
        void this.deps.haptics.nudge(n.level).catch(() => {});
      }
    }

    if (suggestion && session.mode !== "therapist") {
      if (!this.quietEnoughToSpeak()) {
        // Someone is talking (or just stopped): hold until the VAD has been
        // quiet long enough (most recent wins).
        this.held = { text: suggestion, expiresAtMs: this.now() + this.speakHoldMaxMs, latency, turn };
        latency.held = true;
      } else {
        this.speakNow(suggestion, latency, turn);
      }
    }

    // A turn_local tells the server "the phone handled these words": it
    // suppresses the server's own transcript/suggestion for the span. Only
    // claim that when on-device STT actually heard the span — with STT
    // dead (or absent) the server's transcript is the only one there is.
    if (sttOwned) {
      this.deps.send({
        type: "turn_local",
        session_id: session.sessionId,
        speaker: verdict.speaker,
        speaker_person_id: verdict.personId,
        speaker_match_score: verdict.score,
        is_self: verdict.isSelf,
        text: aligned.text,
        start_time: span.start,
        end_time: span.end,
        transcript_source: "on-device",
        prosody,
        text_tone: textTone,
        suggestion,
        suggestion_source: suggestion ? "on-device" : null,
        tts_source: "on-device",
      });
    }
    this.deps.onTurn(turn);
  }

  private speakNow(text: string, latency: TurnLatency | null, turn: LocalTurn | null) {
    if (latency) latency.toSpeakMs = this.now() - this.startWallMs - latency.segmentEndMs;
    if (turn) turn.spoken = true;
    try {
      this.deps.speak(text);
    } catch {
      // TTS failure is the speaker's problem to report; the turn is logged.
    }
  }

  private releaseHeld() {
    const h = this.held;
    if (!h) return;
    this.held = null;
    if (this.now() > h.expiresAtMs) return; // stale: dropped, never spoken
    this.speakNow(h.text, h.latency, h.turn);
  }
}

/** One-line delivery cue for the prompt from raw prosody — absolute-ish
 *  thresholds since the realtime path has no recording-wide distribution. */
export function prosodyHint(p: TurnProsody): string | undefined {
  const parts: string[] = [];
  if (p.rms_dbfs !== null) {
    if (p.rms_dbfs > -15) parts.push("loud");
    else if (p.rms_dbfs < -35) parts.push("quiet");
  }
  if (p.speech_rate !== null) {
    if (p.speech_rate > 3.5) parts.push("fast");
    else if (p.speech_rate < 1.5) parts.push("slow");
  }
  return parts.length ? parts.join(", ") : undefined;
}

/** Human-readable latency table for the console at session end. */
export function formatLatencyLog(log: TurnLatency[]): string {
  if (log.length === 0) return "[fastLoop] no turns";
  const lines = log.map(
    (l) =>
      `#${l.turn} end=${l.segmentEndMs.toFixed(0)}ms ` +
      `spk=${l.speakerMs.toFixed(0)} stt=${l.sttWaitMs.toFixed(0)} ` +
      `pros=${l.prosodyMs.toFixed(1)} llm=${l.llmMs.toFixed(0)} ` +
      `speak=${l.toSpeakMs === null ? "-" : l.toSpeakMs.toFixed(0)}${l.held ? "(held)" : ""} ` +
      `via=${l.provider}`,
  );
  const spoken = log.filter((l) => l.toSpeakMs !== null).map((l) => l.toSpeakMs as number);
  const median = spoken.length
    ? [...spoken].sort((a, b) => a - b)[spoken.length >> 1].toFixed(0)
    : "-";
  return `[fastLoop] ${log.length} turns, median segment-end→speak ${median} ms\n` + lines.join("\n");
}
