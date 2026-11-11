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
import type { CapState, EmotionDevStore, EpisodeInfo, EpisodeWindow, FlagValue } from "./devStore";
import type { RecordContext } from "./context";
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
  /** Save-audio is on (clips are being kept). */
  saveAudio: boolean;
  /** Clips written to disk this run (episodes, neutral samples, marks, flags). */
  clipsSaved: number;
  /** Saves REFUSED this run (storage full / write error): each is also a `skip` line in the log. */
  skippedSaves: number;
  episodes: number;
  neutralSamples: number;
  storage: CapState;
  lastSkipReason: string | null;
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
  /** What the Live Coach session knows right now (talk share, pace, mode...); null/absent = nothing. */
  context?: () => RecordContext | null;
  /** Randomness for neutral sampling (tests inject a fixed one). */
  rand?: () => number;
}

interface Episode {
  id: string;
  startT: number;
  clipStartAbs: number;
  lastTriggerEndAbs: number;
  windows: EpisodeWindow[];
  triggers: Set<string>;
  nudgeIds: string[];
  context: RecordContext | null;
  peak: { win: RecentWindow; score: number };
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
  private neutralEveryS = EMOTION_DEV_DEFAULTS.neutralSampleDefaultEveryMin * 60;
  private episode: Episode | null = null;
  private speechSinceNeutralS = 0;
  private nextNeutralAtS: number | null = null;
  private contextFn: (() => RecordContext | null) | null = null;
  private snapListener: (() => void) | null = null;
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
    saveAudio: false,
    clipsSaved: 0,
    skippedSaves: 0,
    episodes: 0,
    neutralSamples: 0,
    storage: "ok",
    lastSkipReason: null,
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
    this.snapListener?.();
  }

  /** The hub subscribes here so the status line / panel refresh as windows and clips happen. */
  setSnapshotListener(fn: (() => void) | null): void {
    this.snapListener = fn;
  }

  setContextProvider(fn: (() => RecordContext | null) | null): void {
    this.contextFn = fn;
  }

  private context(): RecordContext | null {
    try {
      return (this.d.context ?? this.contextFn)?.() ?? null;
    } catch {
      return null; // context is a nice-to-have; never let it break a record
    }
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
    this.snap.saveAudio = s.saveAudio;
    this.saveAudioMinProb = s.saveAudioMinProb;
    this.neutralEveryS = Math.max(0, s.neutralSampleEveryMin) * 60;
    this.nextNeutralAtS = null;
    // One cap for everything (episodes, samples, marks, flags). Reaching it stops NEW saves; nothing unrated is deleted.
    this.d.store.setClipCapBytes(s.storageCapMb * 1024 * 1024);
    this.refreshStorage();
  }

  // -- pause ----------------------------------------------------------------

  setPaused(paused: boolean): void {
    if (paused === this.paused) return;
    this.paused = paused;
    if (paused) {
      // Paused = not listening, not buffering: drop the ring so a later flag
      // can never reach back across the pause. Close an open episode FIRST,
      // while its audio is still in the ring.
      this.closeEpisode();
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
    this.episode = null;
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
    for (const clip of this.clips.onPcm(pcm)) this.persistClip(clip.id, clip.reason, clip.atMs, clip.pcm, true);
    this.sinceTick += pcm.length;
    if (this.sinceTick >= this.cfg.hopS * RING_SAMPLE_RATE && !this.busy) {
      this.sinceTick = 0;
      this.running = this.tick();
    }
  }

  /** Save a clip and COUNT the outcome: a refusal (storage full) or a write error is never silent. */
  private persistClip(id: string, reason: string, atMs: number, pcm: Int16Array, userInitiated = false): string | null {
    try {
      const name = this.d.store.saveClip({ id, reason, atMs, pcm, userInitiated });
      if (name) this.snap.clipsSaved++;
      else {
        this.snap.skippedSaves++;
        this.snap.lastSkipReason = this.d.store.lastSkipReason;
      }
      this.refreshStorage();
      return name;
    } catch (e) {
      this.snap.skippedSaves++;
      this.snap.lastSkipReason = "write-error";
      this.fail(e);
      return null;
    }
  }

  private refreshStorage(): void {
    try {
      this.snap.storage = this.d.store.capState();
    } catch {
      // keep the previous value
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
      const endAbs = this.ring.total;
      if (pcm.length < RING_SAMPLE_RATE * 1.5 || !windowHasSpeechEnergy(pcm, this.cfg)) {
        this.snap.skippedSilent++;
        this.gate.reset();
        this.maybeCloseEpisode();
        this.publish();
        return;
      }
      const audio = int16ToFloat32(pcm);
      const problem = waveformProblem(audio);
      if (problem) {
        this.snap.skippedSilent++;
        this.maybeCloseEpisode();
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
      this.afterInference(t, tinyRaw, smallRaw, pcm, endAbs);
    } catch (e) {
      this.fail(e);
    } finally {
      this.busy = false;
    }
  }

  private afterInference(t: number, tinyRaw: number[], smallRaw: number[] | null, windowPcm: Int16Array, endAbs: number): void {
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
        context: this.context(),
      });
    }
    // Episode clips: consecutive/overlapping triggered windows become ONE clip.
    const triggers: string[] = [];
    if (clipDisagreement) triggers.push("disagree");
    if (heard) triggers.push("heard");
    if (this.saveAudio && decision.nudge) triggers.push("nudge");
    const win: EpisodeWindow = { t, tiny: Array.from(tinyRaw), small: small ? Array.from(small) : null, triggers, nudged: decision.nudge };
    if (triggers.length) {
      this.addTriggeredWindow(id, win, tinyRaw, small, endAbs, windowPcm.length);
    } else {
      if (this.episode) {
        this.episode.windows.push(win); // lead/tail window: keep its probabilities with the episode
        this.maybeCloseEpisode();
      }
      this.maybeNeutralSample(id, t, tinyRaw, small, endAbs);
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

  // -- episodes ---------------------------------------------------------------

  private addTriggeredWindow(id: string, win: EpisodeWindow, tiny: Probs, small: Probs | null, endAbs: number, winSamples: number): void {
    const sr = RING_SAMPLE_RATE;
    const tailAbs = Math.round(this.cfg.episodeTailS * sr);
    const maxAbs = Math.round(this.cfg.episodeMaxS * sr);
    const ep0 = this.episode;
    // Beyond ~45 s the episode is closed and a new one starts (split).
    if (ep0 && endAbs + tailAbs - ep0.clipStartAbs > maxAbs) this.closeEpisode();
    if (!this.episode) {
      const winStart = Math.max(0, endAbs - winSamples);
      this.episode = {
        id: `ep-${id}`,
        startT: win.t,
        clipStartAbs: Math.max(0, winStart - Math.round(this.cfg.episodeLeadS * sr)),
        lastTriggerEndAbs: endAbs,
        windows: [],
        triggers: new Set(),
        nudgeIds: [],
        context: this.context(),
        peak: { win: { id, t: win.t, tiny, small }, score: -1 },
      };
    }
    const ep = this.episode;
    ep.windows.push(win);
    ep.lastTriggerEndAbs = endAbs;
    for (const tr of win.triggers) ep.triggers.add(tr);
    if (win.nudged) ep.nudgeIds.push(id);
    const score = Math.max(nonNeutralTop(tiny), small ? nonNeutralTop(small) : 0);
    if (score > ep.peak.score) ep.peak = { win: { id, t: win.t, tiny, small }, score };
  }

  /** Close the open episode once the tail has elapsed (the next window did not trigger). */
  private maybeCloseEpisode(): void {
    const ep = this.episode;
    if (!ep) return;
    if (this.ring.total >= ep.lastTriggerEndAbs + Math.round(this.cfg.episodeTailS * RING_SAMPLE_RATE)) this.closeEpisode();
  }

  /** Write the episode record and its clip with whatever audio exists. */
  private closeEpisode(): void {
    const ep = this.episode;
    this.episode = null;
    if (!ep) return;
    const sr = RING_SAMPLE_RATE;
    const endAbs = Math.min(ep.lastTriggerEndAbs + Math.round(this.cfg.episodeTailS * sr), this.ring.total);
    const pcm = this.ring.read(ep.clipStartAbs, endAbs);
    const pk = ep.peak.win;
    const peakOf = (idx: 0 | 1) => {
      let best: { label: string; prob: number } | null = null;
      for (const w of ep.windows) {
        const p = idx === 0 ? w.tiny : w.small;
        if (!p) continue;
        const top = topClass(p);
        if (top.label !== "neutral" && (!best || top.prob > best.prob)) best = { label: top.label, prob: top.prob };
      }
      return best ?? (idx === 0 ? topClass(pk.tiny) : pk.small ? topClass(pk.small) : null);
    };
    const info: EpisodeInfo = {
      startT: ep.startT,
      endT: ep.windows[ep.windows.length - 1]?.t ?? ep.startT,
      clipS: Math.round((pcm.length / sr) * 10) / 10,
      windows: ep.windows,
      peakTiny: peakOf(0)!,
      peakSmall: peakOf(1),
      nudge: ep.nudgeIds.length > 0,
      nudgeIds: ep.nudgeIds,
      triggers: [...ep.triggers],
    };
    const kind = ep.triggers.has("disagree") && !ep.triggers.has("heard") && !ep.triggers.has("nudge") ? "disagree" : "saved";
    this.snap.episodes++;
    try {
      this.d.store.addRecord({
        id: ep.id, t: ep.startT, kind, tiny: pk.tiny, small: pk.small, nudged: info.nudge, episode: info, context: ep.context,
      });
    } catch (e) {
      this.fail(e);
      return;
    }
    if (pcm.length > 0) this.persistClip(ep.id, kind === "disagree" ? "disagree" : "heard", ep.startT, pcm);
    this.publish();
  }

  /** Neutral sampling: about one `neutralSampleClipS` clip per `neutralEveryS` of speech, when BOTH models say neutral. */
  private maybeNeutralSample(id: string, t: number, tiny: Probs, small: Probs | null, endAbs: number): void {
    this.speechSinceNeutralS += this.cfg.hopS;
    if (!this.saveAudio || this.neutralEveryS <= 0 || this.episode) return;
    const rand = this.d.rand ?? Math.random;
    if (this.nextNeutralAtS === null) this.nextNeutralAtS = this.neutralEveryS * (0.5 + rand());
    if (this.speechSinceNeutralS < this.nextNeutralAtS) return;
    if (topClass(tiny).label !== "neutral" || (small && topClass(small).label !== "neutral")) return;
    this.speechSinceNeutralS = 0;
    this.nextNeutralAtS = this.neutralEveryS * (0.5 + rand());
    const sr = RING_SAMPLE_RATE;
    const pcm = this.ring.read(endAbs - Math.round(this.cfg.neutralSampleClipS * sr), endAbs);
    if (pcm.length < sr * 2) return;
    const nid = `ns-${id}`;
    try {
      this.d.store.addRecord({ id: nid, t, kind: "neutral-sample", tiny, small, nudged: false, context: this.context() });
    } catch (e) {
      this.fail(e);
      return;
    }
    if (this.persistClip(nid, "neutral-sample", t, pcm)) this.snap.neutralSamples++;
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
      this.d.store.addRecord({ id: w.id, t: w.t, kind: "window", tiny: w.tiny, small: w.small, nudged: false, context: this.context() });
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
    this.d.store.addRecord({ id, t, kind: "mark", tiny: w?.tiny ?? [], small: w?.small ?? null, nudged: false, context: this.context() });
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
    else if (c.action === "windows" && c.rows) {
      // On-watch TINY (Phase 2): the watch classified these itself. Log them beside the phone's own
      // rows (kind "watch", small null) so one export holds both; they never drive a buzz here.
      for (const r of c.rows) {
        this.d.store.addRecord({ id: `watch-${r.t}`, t: r.t, kind: "watch", tiny: r.p, small: null, nudged: r.nudged });
      }
    }
    else if (c.action === "mark") this.mark();
  }

  /** Session ended / mode turned off: finish pending flag clips with the audio that exists. */
  stop(): void {
    this.closeEpisode();
    for (const clip of this.clips.flushAll()) this.persistClip(clip.id, clip.reason, clip.atMs, clip.pcm, true);
  }
}

/** Highest probability among the non-neutral classes (how "emotional" a reading is). */
function nonNeutralTop(p: Probs): number {
  const top = topClass(p);
  return top.label === "neutral" ? 0 : top.prob;
}
