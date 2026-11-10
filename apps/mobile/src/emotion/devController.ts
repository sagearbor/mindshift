/**
 * The Emotion dev tester's orchestrator. Pure of platform: engines, store,
 * watch link, clock and ids are injected, so the whole flow (windowing, both
 * models, nudge gate, disagreement clip, flags, pause) is unit-tested with
 * fakes (__tests__/emotionDevController.test.ts).
 *
 * Data flow: the live-session mic frames (16 kHz mono int16) arrive in
 * {@link pushPcm}; every `hopS` of audio the last `windowS` is gated for
 * speech energy, turned into ONE log-mel, and run through TINY (which may
 * nudge) and SMALL (silent). Raw probabilities of both are logged.
 */
import { EMOTION_DEV_DEFAULTS, type EmotionDevConfig, type EmotionDevUserSettings, type EmotionLabel, type Probs } from "./config";
import type { EmotionEngine } from "./engine";
import { computeLogMel, int16ToFloat32, waveformProblem } from "./logMel";
import { DailyCap, NudgeGate, applyOffsets, isProbVector, modelsDisagree, topClass, windowHasSpeechEnergy } from "./logic";
import { ClipScheduler, PcmRing, RING_SAMPLE_RATE } from "./pcmRing";
import type { EmotionDevStore, FlagValue } from "./devStore";
import type { WatchControl, WatchEmotionState, WatchLink } from "./watchLink";

export type DevFlag = Exclude<FlagValue, "unrated">;

/** Optional one-liners shown on the watch with the label. Descriptive of the
 *  voice, not of the person's inner state, and only when the tester opts in. */
export const SUGGESTIONS: Partial<Record<EmotionLabel, string>> = {
  anger: "Voice sounds tense. Slow down?",
  disgust: "Voice sounds put off.",
  fear: "Voice sounds anxious. Breathe?",
  happiness: "Voice sounds upbeat.",
  sadness: "Voice sounds low.",
};

export interface DevSnapshot {
  paused: boolean;
  windows: number;
  skippedSilent: number;
  errors: number;
  lastLabel: EmotionLabel | null;
  lastProb: number | null;
  lastSmallLabel: EmotionLabel | null;
  nudges: number;
  marks: number;
  lastNudgeId: string | null;
  lastError: string | null;
}

export interface DevControllerDeps {
  tiny: EmotionEngine | null;
  small: EmotionEngine | null;
  store: EmotionDevStore;
  link: WatchLink | null;
  cfg?: EmotionDevConfig;
  now?: () => number;
  newId?: () => string;
  showSuggestion?: boolean;
  /** Per-class logit offsets for TINY's nudge decision (default: the config constant, all zero). */
  offsets?: readonly number[];
  onSnapshot?: (s: DevSnapshot) => void;
}

interface RecentWindow {
  id: string;
  t: number;
  tiny: Probs;
  small: Probs | null;
}

export class EmotionDevController {
  private readonly cfg: EmotionDevConfig;
  private readonly now: () => number;
  private ring: PcmRing;
  private clips: ClipScheduler;
  private readonly gate: NudgeGate;
  private readonly disagreeCap: DailyCap;
  private sinceTick = 0;
  private busy = false;
  private running: Promise<void> = Promise.resolve();
  private paused = false;
  private counter = 0;
  private lastSent: { label: string; atMs: number } | null = null;
  private recent: RecentWindow[] = [];
  private flaggedIds = new Set<string>();
  private saveAudio = false;
  private saveAudioMinProb = EMOTION_DEV_DEFAULTS.saveAudioDefaultMinProb;
  private nudgeIds = new Set<string>();
  private lastNudge: { id: string; t: number } | null = null;
  private snap: DevSnapshot = {
    paused: false,
    windows: 0,
    skippedSilent: 0,
    errors: 0,
    lastLabel: null,
    lastProb: null,
    lastSmallLabel: null,
    nudges: 0,
    marks: 0,
    lastNudgeId: null,
    lastError: null,
  };

  constructor(private readonly d: DevControllerDeps) {
    this.cfg = d.cfg ?? EMOTION_DEV_DEFAULTS;
    this.now = d.now ?? Date.now;
    this.ring = new PcmRing(this.cfg.ringS * RING_SAMPLE_RATE);
    this.clips = new ClipScheduler(this.ring);
    this.gate = new NudgeGate(this.cfg);
    this.disagreeCap = new DailyCap(this.cfg.maxDisagreeClipsPerDay);
  }

  snapshot(): DevSnapshot {
    return { ...this.snap, paused: this.paused };
  }

  private publish(): void {
    this.d.onSnapshot?.(this.snapshot());
  }

  private id(): string {
    return this.d.newId ? this.d.newId() : `${this.now().toString(36)}-${(this.counter++).toString(36)}`;
  }

  setMaxNudgesPerHour(n: number): void {
    this.gate.setMaxPerHour(n);
  }

  /** Apply the tester's switches: nudge rule, buzz budget, save-audio mode and its storage cap. */
  applySettings(s: EmotionDevUserSettings): void {
    this.gate.setMaxPerHour(s.maxNudgesPerHour);
    this.gate.setRule(s.nudgeRule);
    this.saveAudio = s.saveAudio;
    this.saveAudioMinProb = s.saveAudioMinProb;
    this.d.store.setClipCapBytes(s.saveAudio ? s.storageCapMb * 1024 * 1024 : this.cfg.maxClipBytes);
  }

  // -- pause ----------------------------------------------------------------

  setPaused(paused: boolean): void {
    if (paused === this.paused) return;
    this.paused = paused;
    if (paused) {
      // Paused = not listening, not buffering: drop the ring so a later flag
      // can never reach back across the pause.
      this.resetAudio();
      this.gate.reset();
    }
    this.publish();
    void this.pushState({ label: this.snap.lastLabel ?? "neutral", prob: this.snap.lastProb ?? 0, nudge: false, suggestion: null, force: true });
  }

  private resetAudio(): void {
    this.ring = new PcmRing(this.cfg.ringS * RING_SAMPLE_RATE);
    this.clips = new ClipScheduler(this.ring);
    this.sinceTick = 0;
  }

  /** Resolves when the in-flight window (if any) has finished. */
  idle(): Promise<void> {
    return this.running;
  }

  isPaused(): boolean {
    return this.paused;
  }

  // -- audio in -------------------------------------------------------------

  pushPcm(pcm: Int16Array): void {
    if (this.paused || pcm.length === 0) return;
    this.ring.push(pcm);
    for (const clip of this.clips.onPcm(pcm)) this.persistClip(clip.id, clip.reason, clip.atMs, clip.pcm);
    this.sinceTick += pcm.length;
    if (this.sinceTick >= this.cfg.hopS * RING_SAMPLE_RATE && !this.busy) {
      this.sinceTick = 0;
      this.running = this.tick();
    }
  }

  private persistClip(id: string, reason: string, atMs: number, pcm: Int16Array): void {
    try {
      this.d.store.saveClip({ id, reason, atMs, pcm });
    } catch (e) {
      this.fail(e);
    }
  }

  private fail(e: unknown): void {
    this.snap.errors++;
    this.snap.lastError = e instanceof Error ? e.message : String(e);
    this.publish();
  }

  /** Scripted test: classify exactly one captured take (no ring, gate or nudge)
   *  and log it next to the emotion that was asked for. Returns the raw
   *  probabilities, or null when TINY is not loaded / the clip is unusable. */
  async classifyClip(pcm: Int16Array, expect: string, round: number): Promise<{ tiny: number[]; small: number[] | null } | null> {
    if (!this.d.tiny || pcm.length < RING_SAMPLE_RATE) return null;
    const audio = int16ToFloat32(pcm.length > this.cfg.windowS * RING_SAMPLE_RATE ? pcm.subarray(pcm.length - this.cfg.windowS * RING_SAMPLE_RATE) : pcm);
    if (waveformProblem(audio)) return null;
    const mel = computeLogMel(audio);
    const tiny = await this.d.tiny.classify(mel);
    let small: number[] | null = null;
    if (this.d.small) {
      try {
        small = await this.d.small.classify(mel);
      } catch (e) {
        this.fail(e);
      }
    }
    if (!isProbVector(tiny)) return null;
    this.d.store.addRecord({
      id: this.id(), t: this.now(), kind: "scripted", tiny, small: small && isProbVector(small) ? small : null,
      nudged: false, expect, round,
    });
    return { tiny: Array.from(tiny), small: small && isProbVector(small) ? Array.from(small) : null };
  }

  // -- one window -----------------------------------------------------------

  /** Classify the latest window now (exposed so tests can drive it deterministically). */
  async tick(): Promise<void> {
    if (this.paused || this.busy || !this.d.tiny) return;
    this.busy = true;
    try {
      const pcm = this.ring.tail(this.cfg.windowS * RING_SAMPLE_RATE);
      if (pcm.length < RING_SAMPLE_RATE * 1.5 || !windowHasSpeechEnergy(pcm, this.cfg)) {
        this.snap.skippedSilent++;
        this.gate.reset();
        this.publish();
        return;
      }
      const audio = int16ToFloat32(pcm);
      const problem = waveformProblem(audio);
      if (problem) {
        this.snap.skippedSilent++;
        this.publish();
        return;
      }
      const mel = computeLogMel(audio);
      const t = this.now();
      const tinyRaw = await this.d.tiny.classify(mel);
      let smallRaw: number[] | null = null;
      if (this.d.small) {
        try {
          smallRaw = await this.d.small.classify(mel);
        } catch (e) {
          this.fail(e); // SMALL is silent: its failure must not stop TINY
        }
      }
      if (this.paused) return; // paused while the models were running: discard
      this.afterInference(t, tinyRaw, smallRaw, pcm);
    } catch (e) {
      this.fail(e);
    } finally {
      this.busy = false;
    }
  }

  private afterInference(t: number, tinyRaw: number[], smallRaw: number[] | null, windowPcm: Int16Array): void {
    if (!isProbVector(tinyRaw)) {
      this.fail(new Error("tiny output is not a probability vector"));
      return;
    }
    const id = this.id();
    const adjusted = applyOffsets(tinyRaw, this.d.offsets);
    const adjustedDiffers = adjusted.some((v, i) => v !== tinyRaw[i]);
    const top = topClass(adjusted);
    const small = smallRaw && isProbVector(smallRaw) ? smallRaw : null;
    const decision = this.gate.decide(adjusted, t, small);

    const disagree = small !== null && modelsDisagree(tinyRaw, small, this.cfg.disagreeMinProb);

    this.snap.windows++;
    this.snap.lastLabel = top.label;
    this.snap.lastProb = top.prob;
    this.snap.lastSmallLabel = small ? topClass(small).label : null;
    this.d.store.addWindow({ id, t, tiny: tinyRaw, small });
    this.remember({ id, t, tiny: tinyRaw, small });

    // Save-audio mode lifts the 20/day disagreement cap (the storage cap bounds it instead).
    const clipDisagreement = disagree && (this.saveAudio || this.disagreeCap.tryFire(t));
    const heard = this.saveAudio ? this.heardEmotion(tinyRaw, small) : null;
    if (decision.nudge) {
      this.snap.nudges++;
      this.snap.lastNudgeId = id;
      this.lastNudge = { id, t };
      this.nudgeIds.add(id);
      this.d.store.addRecord({
        id, t, kind: "nudge", tiny: tinyRaw, small,
        ...(adjustedDiffers ? { tinyAdjusted: adjusted } : {}),
        nudged: true,
      });
    } else if (clipDisagreement) {
      this.d.store.addRecord({ id, t, kind: "disagree", tiny: tinyRaw, small, nudged: false, nudgeReason: decision.reason });
    }
    if (heard && !decision.nudge && !clipDisagreement) {
      this.d.store.addRecord({ id, t, kind: "saved", tiny: tinyRaw, small, nudged: false, nudgeReason: decision.reason });
    }
    if (clipDisagreement || heard) {
      // Independent of whether TINY also buzzed: the disagreement itself is the finding.
      // One clip per window even when several reasons apply.
      const secs = heard && !clipDisagreement ? this.cfg.saveAudioClipS : this.cfg.disagreeClipS;
      this.persistClip(id, clipDisagreement ? "disagree" : "heard", t, windowPcm.subarray(Math.max(0, windowPcm.length - secs * RING_SAMPLE_RATE)));
    }
    this.publish();
    void this.pushState({
      label: top.label,
      prob: top.prob,
      nudge: decision.nudge,
      suggestion: decision.nudge && this.d.showSuggestion ? SUGGESTIONS[top.label] ?? null : null,
      id,
    });
  }

  /** Save-audio trigger: EITHER model's top class is non-neutral with prob >= threshold. */
  private heardEmotion(tiny: Probs, small: Probs | null): { model: "tiny" | "small"; label: EmotionLabel; prob: number } | null {
    for (const [model, p] of [["tiny", tiny], ["small", small]] as const) {
      if (!p) continue;
      const top = topClass(p);
      if (top.label !== "neutral" && top.prob >= this.saveAudioMinProb) return { model, label: top.label, prob: top.prob };
    }
    return null;
  }

  private remember(w: RecentWindow): void {
    this.recent.push(w);
    if (this.recent.length > 20) this.recent.shift();
  }

  // -- watch ---------------------------------------------------------------

  private async pushState(s: { label: string; prob: number; nudge: boolean; suggestion: string | null; id?: string; force?: boolean }): Promise<void> {
    const link = this.d.link;
    if (!link) return;
    const t = this.now();
    // Quiet windows only refresh the watch when the label changed or 15 s
    // passed — a nudge, and a pause toggle, always go out at once.
    const stale = !this.lastSent || this.lastSent.label !== s.label || t - this.lastSent.atMs >= 15_000;
    if (!s.nudge && !s.force && !stale && this.lastSent) return;
    this.lastSent = { label: s.label, atMs: t };
    await link.sendState({
      v: 1,
      id: s.id ?? this.snap.lastNudgeId ?? "",
      t,
      label: s.label,
      prob: Math.round(s.prob * 1000) / 1000,
      nudge: s.nudge,
      paused: this.paused,
      suggestion: s.suggestion,
    } satisfies WatchEmotionState);
  }

  // -- flags ---------------------------------------------------------------

  /**
   * Flag the target (an explicit nudge id from the watch, else the latest
   * nudge if it is still within `unratedAfterS`, else the latest window).
   * Saves 45 s before + 15 s after to the clip store. Returns the flagged id,
   * or null when there is nothing to flag (paused / no window yet).
   */
  flag(flag: DevFlag, nudgeId?: string | null): string | null {
    if (this.paused) return null;
    const t = this.now();
    let targetId = nudgeId ?? null;
    if (!targetId && this.lastNudge && t - this.lastNudge.t <= this.cfg.unratedAfterS * 1000) targetId = this.lastNudge.id;
    const recent = this.recent[this.recent.length - 1];
    if (!targetId) targetId = recent?.id ?? null;
    if (!targetId) return null;

    // A flag on a plain window needs a main-log record to hang the flag on.
    const isNudge = this.nudgeIds.has(targetId);
    const w = this.recent.find((r) => r.id === targetId);
    if (!isNudge && w && !this.flaggedIds.has(targetId)) {
      this.d.store.addRecord({ id: w.id, t: w.t, kind: "window", tiny: w.tiny, small: w.small, nudged: false });
    }
    this.d.store.setFlag(targetId, flag);
    if (!this.flaggedIds.has(targetId)) {
      this.flaggedIds.add(targetId);
      this.clips.request({ id: targetId, reason: `flag-${flag}`, atMs: t }, this.cfg.flagBeforeS, this.cfg.flagAfterS);
    }
    this.publish();
    return targetId;
  }

  /**
   * "Mark this moment": keep the previous 45 s and next 15 s with a user mark,
   * independent of any model output. The record carries the latest window's
   * probabilities (empty when nothing was classified yet). Null when paused.
   */
  mark(): string | null {
    if (this.paused) return null;
    const t = this.now();
    const id = `mark-${this.id()}`;
    const w = this.recent[this.recent.length - 1];
    this.d.store.addRecord({ id, t, kind: "mark", tiny: w?.tiny ?? [], small: w?.small ?? null, nudged: false });
    this.clips.request({ id, reason: "mark", atMs: t }, this.cfg.flagBeforeS, this.cfg.flagAfterS);
    this.snap.marks++;
    this.publish();
    return id;
  }

  /** Apply a control that arrived from the watch. */
  handleControl(c: WatchControl): void {
    if (c.action === "pause") this.setPaused(true);
    else if (c.action === "resume") this.setPaused(false);
    else if (c.action === "flag" && c.flag) this.flag(c.flag, c.nudgeId ?? null);
    else if (c.action === "mark") this.mark();
  }

  /** Session ended / mode turned off: finish pending flag clips with the audio that exists. */
  stop(): void {
    for (const clip of this.clips.flushAll()) this.persistClip(clip.id, clip.reason, clip.atMs, clip.pcm);
  }
}
