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
import type { Embedder, EnrolledPerson, MatchBasis, SpeakerLabeler, SpeakerVerdict } from "./speakerId";
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
/**
 * Mid-call naming (cluster → person binding): "that Speaker B is Mom". Set
 * through `FastLoop.bindSpeaker`; from then on every turn the labeler puts
 * on that raw label carries the person (id / display name / is_self) while
 * the WIRE label stays the raw cluster label, so the session record keeps
 * one stable key per voice.
 */
export interface SpeakerBinding {
  personId: string;
  displayName: string;
  isSelf: boolean;
}

/** Seconds of each speaker's VAD-cut speech kept for a mid-call enrollment. */
export const DEFAULT_SPEAKER_AUDIO_SECONDS = 20;

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
  /** Per-provider outcome for THIS turn's suggestion attempt (os/bundled/cloud
   *  × ok/refused/timeout/unavailable/error/cloud), with the error/refusal
   *  detail so diagnostics can show WHY a local provider didn't answer — e.g.
   *  the exact Gemini Nano error. Absent when no suggestion was attempted. */
  attempts?: { provider: string; outcome: string; detail?: string }[];
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
  /** The person's name when identified (a voiceprint match or a mid-call
   *  binding); null for an unknown cluster. `speaker` stays the raw label. */
  displayName: string | null;
  matchScore: number | null;
  /** How the voiceprint match was reached ("absolute" | "contrast"); null
   *  for an unidentified cluster or a mid-call binding. A contrast identity
   *  is REVISABLE: if a later cluster beats this one for the same person by
   *  the margin, the person moves and this turn's identity is cleared. */
  matchBasis: MatchBasis | null;
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
  /** Seconds of each speaker's finalized-turn PCM pooled for a mid-call
   *  "remember this voice" (most recent kept). 0 disables pooling. */
  speakerAudioSeconds?: number;
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
  /** How often the recognizer had to be restarted this session (both
   *  recognizers count their own restarts); 0 when there was none. */
  sttRestarts: number;
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
  private readonly speakerAudioSamples: number;
  /** Mid-call naming: raw cluster label → person, and the reverse (so a
   *  later voiceprint match on that person maps back to the same raw
   *  label the session has been using). */
  private bindings = new Map<string, SpeakerBinding>();
  private boundLabelOfPerson = new Map<string, string>();
  /** Per raw label: the most recent `speakerAudioSamples` of turn PCM. */
  private speakerPools = new Map<string, { chunks: Float32Array[]; samples: number }>();

  private session: FastLoopSession | null = null;
  private running = false;
  private startWallMs = 0;
  private samplesSeen = 0;
  private pending = new Float32Array(0);
  private history: { startSample: number; data: Float32Array }[] = [];
  private vadQueue: Promise<void> = Promise.resolve();
  private turnQueue: Promise<void> = Promise.resolve();
  private turns: LocalTurn[] = [];
  /** The labeler identity revision the past turns were last aligned to. */
  private seenIdentityRevision = 0;
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
    this.speakerAudioSamples = Math.round(
      (deps.speakerAudioSeconds ?? DEFAULT_SPEAKER_AUDIO_SECONDS) * SILERO_SAMPLE_RATE,
    );
  }

  // --- Mid-call naming hooks (cluster → person binding) ---------------------

  /**
   * Bind a raw speaker label ("Speaker B") to a person for the rest of the
   * session. Past turns are re-attributed in the session summary; future
   * turns on that label carry the person while keeping the raw wire label.
   * `print` (the person's voiceprint — e.g. embedded from this session's
   * pooled audio via `embedSpeaker`) is added to the labeler so a later
   * turn that matches by VOICE also lands on this binding.
   */
  bindSpeaker(label: string, binding: SpeakerBinding, print?: EnrolledPerson | null): void {
    // One raw label per person: re-binding a person to a new label frees
    // the old one (the user corrected themselves).
    const previous = this.boundLabelOfPerson.get(binding.personId);
    if (previous !== undefined && previous !== label) this.bindings.delete(previous);
    this.bindings.set(label, { ...binding });
    this.boundLabelOfPerson.set(binding.personId, label);
    if (print && this.deps.labeler) {
      this.deps.labeler.addPerson({ ...print, personId: binding.personId, displayName: binding.displayName, isSelf: binding.isSelf });
    }
    for (const turn of this.turns) {
      if (turn.speaker !== label) continue;
      turn.personId = binding.personId;
      turn.displayName = binding.displayName;
      turn.isSelf = binding.isSelf;
    }
  }

  /** The person bound to a raw label, if any. */
  bindingOf(label: string): SpeakerBinding | null {
    return this.bindings.get(label) ?? null;
  }

  /** What to call a raw label in prompts: the bound/matched name, else the label. */
  displayNameOf(label: string): string {
    return this.bindings.get(label)?.displayName ?? label;
  }

  /** Seconds of pooled speech held for a raw label. */
  speakerAudioSeconds(label: string): number {
    return (this.speakerPools.get(label)?.samples ?? 0) / SILERO_SAMPLE_RATE;
  }

  /** The pooled PCM (16 kHz float32) of a raw label's finalized turns —
   *  the input for a mid-call enrollment. Empty when nothing was pooled. */
  speakerAudio(label: string): Float32Array {
    const pool = this.speakerPools.get(label);
    if (!pool || pool.samples === 0) return new Float32Array(0);
    const out = new Float32Array(pool.samples);
    let offset = 0;
    for (const chunk of pool.chunks) {
      out.set(chunk, offset);
      offset += chunk.length;
    }
    return out;
  }

  /** Embed a raw label's pooled audio with the session's ECAPA embedder;
   *  null when speaker-ID is off, the pool is empty, or embedding fails. */
  async embedSpeaker(label: string): Promise<Float32Array | null> {
    const pcm = this.speakerAudio(label);
    if (!this.deps.embedder || pcm.length === 0) return null;
    try {
      return await this.deps.embedder.embed(pcm, SILERO_SAMPLE_RATE);
    } catch {
      return null;
    }
  }

  private poolSpeakerAudio(label: string, pcm: Float32Array) {
    if (this.speakerAudioSamples <= 0 || pcm.length === 0) return;
    const pool = this.speakerPools.get(label) ?? { chunks: [], samples: 0 };
    pool.chunks.push(pcm);
    pool.samples += pcm.length;
    while (pool.samples > this.speakerAudioSamples && pool.chunks.length > 1) {
      const dropped = pool.chunks.shift() as Float32Array;
      pool.samples -= dropped.length;
    }
    if (pool.samples > this.speakerAudioSamples) {
      // A single oversized turn: keep its most recent tail.
      const only = pool.chunks[0];
      pool.chunks[0] = only.subarray(only.length - this.speakerAudioSamples);
      pool.samples = this.speakerAudioSamples;
    }
    this.speakerPools.set(label, pool);
  }

  /** Apply the mid-call bindings to a labeler verdict (pure w.r.t. state). */
  private applyBindings(verdict: SpeakerVerdict): SpeakerVerdict {
    // The user's own naming of a raw cluster label outranks an INFERRED
    // (contrast) identity on that label: "Speaker B is Mom" was said out
    // loud; the contrast rule only concluded it.
    if (verdict.basis === "contrast") {
      const said = this.bindings.get(verdict.speaker);
      if (said && said.personId !== verdict.personId) {
        return { ...verdict, personId: said.personId, displayName: said.displayName, isSelf: said.isSelf, score: null, basis: null };
      }
    }
    // A voiceprint match on a person bound to a raw label → keep the raw
    // label on the wire (one key per voice for the whole session).
    if (verdict.personId !== null) {
      const raw = this.boundLabelOfPerson.get(verdict.personId);
      if (raw !== undefined) {
        const b = this.bindings.get(raw);
        return {
          ...verdict,
          speaker: raw,
          displayName: b?.displayName ?? verdict.displayName,
          isSelf: b ? b.isSelf : verdict.isSelf,
        };
      }
      return verdict;
    }
    const bound = this.bindings.get(verdict.speaker);
    if (bound) {
      return {
        ...verdict,
        personId: bound.personId,
        displayName: bound.displayName,
        isSelf: bound.isSelf,
      };
    }
    // Someone bound as "me" makes every other unidentified voice honestly
    // not-me (the same rule an enrolled self print gives the labeler).
    if (verdict.isSelf === null) {
      for (const b of this.bindings.values()) {
        if (b.isSelf) return { ...verdict, isSelf: false };
      }
    }
    return verdict;
  }

  /**
   * Carry a revised cluster identity back over the session's past turns
   * (same in-place update `bindSpeaker` does). The labeler re-resolves who
   * is who after every cluster update; a cluster can gain a person once a
   * second cluster exists to contrast against, or lose it to a later
   * cluster that beats it by the margin — a person is one voice. Turns on
   * a label the USER bound, and turns matched outright (absolute), are
   * never touched. Already-sent turn_local events are not re-sent: the raw
   * label is the stable wire key, and the record shows the move.
   */
  private reattributeTurns(): void {
    const labeler = this.deps.labeler;
    if (!labeler || labeler.identityRevision === this.seenIdentityRevision) return;
    this.seenIdentityRevision = labeler.identityRevision;
    const assignments = labeler.clusterAssignments();
    const someoneBoundAsSelf = Array.from(this.bindings.values()).some((b) => b.isSelf);
    for (const turn of this.turns) {
      if (turn.matchBasis === "absolute" || turn.speaker === "Unknown") continue;
      if (this.bindings.has(turn.speaker)) continue;
      const now = assignments.get(turn.speaker) ?? null;
      if (now) {
        if (turn.personId === now.personId && turn.matchBasis === now.basis) continue;
        turn.personId = now.personId;
        turn.displayName = now.displayName;
        turn.isSelf = now.isSelf;
        turn.matchScore = now.score;
        turn.matchBasis = now.basis;
      } else if (turn.matchBasis === "contrast") {
        // Lost its person to a stronger cluster: back to an unidentified
        // voice, with the same honesty rule the labeler applies.
        turn.personId = null;
        turn.displayName = null;
        turn.isSelf = labeler.hasSelfPrint || someoneBoundAsSelf ? false : null;
        turn.matchScore = null;
        turn.matchBasis = null;
      }
    }
  }

  get isRunning() {
    return this.running;
  }

  /** The turns finalized so far this session (live view; `stop()` returns
   *  the same list in its summary). Identities on past turns may be revised
   *  in place — see `reattributeTurns`. */
  get turnsSoFar(): readonly LocalTurn[] {
    return this.turns;
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
    this.bindings = new Map();
    this.boundLabelOfPerson = new Map();
    this.speakerPools = new Map();
    this.seenIdentityRevision = 0;
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
    const restarts = (this.deps.recognizer as { restarts?: unknown } | null)?.restarts;
    return {
      turns: [...this.turns],
      latencyLog: [...this.latencyLog],
      sttAvailable: this.sttAvailable,
      sttRestarts: typeof restarts === "number" ? restarts : 0,
    };
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
    // Whether the words of this span were the phone's to report. A
    // recognizer that has DIED (failed to start, fatal error) means the
    // server's transcript owns them (see send below). A loop deliberately
    // built without one (the replay harness on a transcript-less scene; the
    // production capability gate never starts the loop without STT) still
    // reports its turns — there are no words to claim, and identity,
    // prosody and the turn ranges are the point of the record.
    const sttOwned = this.deps.recognizer === null || this.sttAvailable;

    // Speaker-ID and STT are independent — run them together.
    const speakerPromise = (async (): Promise<{ verdict: SpeakerVerdict; ms: number }> => {
      const t0 = this.now();
      let verdict: SpeakerVerdict = { speaker: "Unknown", personId: null, displayName: null, isSelf: null, score: null, basis: null };
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
    const [{ verdict: rawVerdict, ms: speakerMs }, aligned] = await Promise.all([speakerPromise, textPromise]);
    // Mid-call naming: a bound cluster carries its person from here on.
    const verdict = this.applyBindings(rawVerdict);
    // This turn may have moved a person between clusters: past turns follow.
    this.reattributeTurns();
    this.poolSpeakerAudio(verdict.speaker, pcm);

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
    let attempts: { provider: string; outcome: string }[] | undefined;
    if (aligned.text) {
      const tl0 = this.now();
      // The prompt names people the way the user does ("Mom", not
      // "Speaker B") once a binding or a voiceprint match says who they are.
      const context = this.turns
        .slice(-this.contextTurns)
        .map((t) => ({ speaker: t.displayName ?? this.displayNameOf(t.speaker), text: t.text }))
        .filter((t) => t.text);
      const result = await this.deps.llm.suggest({
        text: aligned.text,
        speaker: verdict.displayName ?? this.displayNameOf(verdict.speaker),
        isSelf: coachedAsSelf,
        empathy: session.empathy,
        context,
        prosodyHint: prosodyHint(prosody),
        mode: session.mode,
      });
      llmMs = this.now() - tl0;
      provider = result.provider;
      attempts = result.attempts.map((a) => ({
        provider: a.provider,
        outcome: a.outcome,
        ...(a.detail ? { detail: a.detail } : {}),
      }));
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
      attempts,
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
      displayName: verdict.displayName,
      matchScore: verdict.score,
      matchBasis: verdict.basis,
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
        speaker_match_basis: verdict.basis,
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
