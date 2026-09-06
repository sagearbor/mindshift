/**
 * File-driven nudge verification (owner, 2026-09-06: "I cannot be the
 * limiting factor to test manually").
 *
 * Given a `ReplayResult` — a recording replayed through the REAL fast loop
 * (sceneReplay.ts) — this module explains every nudge decision the loop
 * made, turn by turn, and renders ONE mobile-first HTML page a human can
 * read in a minute:
 *
 *   per scene:  timeline (one lane per speaker, dashes for turns, ⚡ instant
 *               haptic / 🧠 LLM-tier nudge / ⌚ call-mode steamroll markers,
 *               ✓ ✗ FP verdicts, dark-probe tags) + scorecard + turn table
 *   per turn:   who, self?, when, first words, loudness over the user's own
 *               baseline, instant level, text-tone level, vocal-activation
 *               probability (dark), single-mic overlap probe (dark), every
 *               NudgeEvent the policy emitted with its scoring verdict, and
 *               the haptic times split into the instant tier (~40 ms after
 *               the turn closes) and the policy tier (after STT + LLM).
 *
 * The "call-mode equivalent" column runs `interruptingEvents` over the
 * scene's GROUND-TRUTH turn timings: where a steamroll nudge would fire if
 * the same conversation were a call (each phone its own stream). On a
 * single mic the segmenter never produces overlapping turns, so this is the
 * only place the vector can be shown for these fixtures.
 *
 * Two lanes answer "what would I have FELT?" (owner addendum, 2026-09-06):
 *
 *   🎧 EARPIECE — what the phone did in earpiece mode, per fragment: the
 *      instant haptic (time + level), the spoken line's time (LLM tier) and
 *      the on-screen nudge — all straight from the replay's logs.
 *   ⌚ WATCH — what the wrist would buzz for the same audio. The watch is
 *      acoustic-only (no text tone): the SAME NudgePolicy (shared three-
 *      runtime contract) is run over loudness-only events (yellingLevel over
 *      the user's baseline, the watch's YELLING_LEVELS ladder) plus
 *      `interrupting` from the ground-truth timings and `airtime` (the
 *      120 s share rule of server/watch/vectors.py, ported below).
 *
 * `gateFailures` is the test gate (__tests__/replay.nudgeReport.test.ts):
 * expected nudges hit >= the number replay.scenes.test.ts pins, no false
 * positives, the instant haptic ahead of the LLM tier on every loud self
 * turn, and the watch lane never buzzing on someone else's turn. Pure
 * functions; Node-only like the rest of this directory.
 */
import type { LocalTurn } from "../fastLoop";
import {
  aggressiveToneLevel,
  interruptingEvents,
  LoudnessBaseline,
  NudgePolicy,
  yellingLevel,
  type NudgeEvent,
  type VectorEvent,
} from "../nudgePolicy";
import { vocabularyForCode } from "../nudgeVocabulary";
import { POSITIVE_CAP_S } from "../nudgeVocabulary";
import type { CalmStreak } from "../positiveNudges";
import type { ReplayScript } from "./meta";
import { matchLoopTurns, median, type NudgeOutcome } from "./score";
import type { ReplayResult } from "./sceneReplay";

// ---------------------------------------------------------------------------
// Report shape
// ---------------------------------------------------------------------------

export type Verdict = NudgeOutcome["verdict"];
export type HapticTier = "instant" | "policy";

export interface NudgeRow {
  /** Audio second the turn closed (the policy's clock). */
  t: number;
  /** Virtual clock at emission — after STT + LLM. */
  atMs: number;
  /** atMs − t·1000: how far behind the turn's end the LLM tier landed. */
  lagMs: number;
  level: number;
  vectors: string[];
  /** A cooldown decay (level down, no vectors) reaches the screen only. */
  decay: boolean;
  loopTurn: number | null;
  scriptTurn: number | null;
  verdict: Verdict | null;
}

export interface HapticRow {
  level: number;
  atMs: number;
  atSec: number;
  tier: HapticTier;
  loopTurn: number | null;
  scriptTurn: number | null;
  /** Instant tier: how long before the same turn's policy tick it buzzed. */
  leadOverPolicyMs: number | null;
}

/** 🎧 What the phone did for this fragment in earpiece mode. */
export interface EarpieceLane {
  /** The ~1 s loudness tier (identity + RMS only). */
  instantHaptic: { atSec: number; level: number } | null;
  /** Escalations the policy put on screen (and buzzed, unless the instant
   *  tier already had) after STT + LLM. */
  screenNudges: { atSec: number; level: number; vectors: string[] }[];
  /** When the coach's line was voiced (null: held past the limit, therapist
   *  mode, or no suggestion). */
  spokenAtSec: number | null;
  spokenText: string | null;
  suggestionKind: LocalTurn["suggestionKind"];
}

export interface FragmentRow {
  /** Loop turn index. */
  index: number;
  start: number;
  end: number;
  /** What the loop called it: an enrolled name, or "?Speaker B" for its own cluster. */
  label: string;
  isSelf: boolean | null;
  /** The loop treated it as the coached user (voiceprint or the "you speak first" fallback). */
  coachedAsSelf: boolean;
  kind: LocalTurn["kind"];
  text: string;
  transcriptFinal: boolean;
  rmsDbfs: number | null;
  /** dB over the user's own running baseline (null: not a self turn / no baseline yet). */
  dbOverBaseline: number | null;
  instantLevel: number;
  toneLevel: number;
  frustration: number | null;
  defensiveness: number | null;
  activation: { probability: number; level: number } | null;
  overlap: { mixedSeconds: number; longestMixedRunSeconds: number; windows: number } | null;
  policy: { atMs: number; rawLevel: number; levelAfter: number } | null;
  scriptTurn: number | null;
  nudges: NudgeRow[];
  haptics: HapticRow[];
  /** The positives this turn earned (delivered or cap-dropped). */
  positives: PositiveRow[];
  earpiece: EarpieceLane;
}

/** 💚 One thing the user did WELL (positiveNudges.ts), as the replay saw it. */
export interface PositiveRow {
  code: string;
  icon: string;
  name: string;
  /** Audio second the detection landed on. */
  t: number;
  /** Virtual clock at delivery. */
  atMs: number;
  loopTurn: number;
  scriptTurn: number | null;
  /** False when the two-minute cap dropped it — detected, never felt. */
  delivered: boolean;
  detail: string;
}

/** ⌚ One watch escalation (buzz) or decay (face only). */
export interface WatchBuzz {
  t: number;
  level: number;
  vectors: string[];
  decay: boolean;
  loopTurn: number | null;
  scriptTurn: number | null;
  /** The buzz landed on the wearer's own turn (ground truth). */
  onSelfTurn: boolean;
}

export interface WatchLane {
  /** The acoustic-only inputs, in the order the policy saw them. */
  events: VectorEvent[];
  buzzes: WatchBuzz[];
  decays: WatchBuzz[];
  byVector: Record<string, number>;
  /** Script turns where the wrist buzzes but the earpiece lane has nothing
   *  (no instant haptic, no screen nudge) — and the reverse. */
  watchOnlyTurns: number[];
  earpieceOnlyTurns: number[];
  bothTurns: number[];
  allOnSelfTurns: boolean;
}

export interface TurnRow {
  index: number;
  speaker: string;
  isSelf: boolean;
  start: number;
  end: number;
  text: string;
  emotion: string | null;
  expected: "mild" | "strong" | null;
  verdict: Verdict;
  /** Highest raw policy level over the turn's fragments. */
  level: number;
  emitted: number[];
  predicted: string | null;
  attributionOk: boolean;
  fragments: FragmentRow[];
  dbOverBaselineMax: number | null;
  instantLevelMax: number;
  toneLevelMax: number;
  activationMax: { probability: number; level: number } | null;
  overlapMax: { mixedSeconds: number; longestMixedRunSeconds: number } | null;
  /** Steamroll events a call would have raised on this (ground-truth) turn. */
  callMode: VectorEvent[];
  /** Highest earpiece-lane level on this turn (instant haptic or screen nudge). */
  earpieceLevel: number;
  /** Highest watch-lane buzz level on this turn. */
  watchLevel: number;
  watchBuzzes: WatchBuzz[];
}

export interface Stat {
  min: number;
  median: number;
  max: number;
  n: number;
}

export interface SceneScorecard {
  scene: string;
  mode: string;
  durationSec: number;
  selfSpeaker: string | null;
  enrolled: string;
  scriptTurns: number;
  loopTurns: number;
  coachedFragments: number;
  expected: number;
  hits: number;
  misses: number;
  falsePositives: number;
  hitsSilent: number;
  instantHaptics: number;
  policyHaptics: number;
  /** Instant haptic → the same turn's policy tick (virtual ms). */
  instantLeadMs: Stat | null;
  /** True when every instant haptic preceded its turn's policy tick and
   *  every LLM-tier nudge on that turn; null when no loud self turn exists. */
  instantBeforePolicy: boolean | null;
  /** Turn end → LLM-tier nudge emission (virtual ms). */
  policyLagMs: Stat | null;
  attribution: { correct: number; total: number; selfCorrect: number; selfTotal: number };
  callModeInterrupting: number;
  activationProbed: number;
  activationMaxProbability: number | null;
  overlapProbed: number;
  overlapMaxMixedSeconds: number | null;
  spokenOverSpeech: number;
  earpiece: {
    instantHaptics: number;
    screenNudges: number;
    /** Coach lines voiced on the user's own turns (nudge lines) / on others'. */
    spokenNudges: number;
    spokenResponses: number;
    /** Segment end → first spoken word, over the coached user's turns. */
    nudgeToSpeakMs: Stat | null;
  };
  /** ⚡ How many fragments the DARK vocal-activation classifier would have
   *  flagged (level >= 1) on turns nobody expects a nudge on — i.e. its false
   *  positives, out of its own training corpus. The other half of the gate in
   *  __tests__/activationGate.test.ts, and the reason `activationNudges` is
   *  still off. `activationFlaggedTurns` names them. */
  activationFalseFlags: number;
  activationFlaggedTurns: number[];
  /** 💚 What the user did well: per-code counts of the positives that were
   *  actually DELIVERED, how many the two-minute cap withheld, and 🧘's
   *  longest quiet run. */
  positives: {
    delivered: Record<string, number>;
    suppressed: number;
    calmStreakS: number;
    calmBadge: boolean;
  };
  watch: {
    buzzes: number;
    byVector: Record<string, number>;
    watchOnlyTurns: number[];
    earpieceOnlyTurns: number[];
    bothTurns: number[];
    allOnSelfTurns: boolean;
  };
}

export interface NudgeReport {
  scene: string;
  mode: string;
  generatedAt: string;
  durationSec: number;
  speakers: string[];
  selfSpeaker: string | null;
  approxBoundaries: boolean;
  turns: TurnRow[];
  /** Loop turns that overlap no scripted turn (VAD fired on non-speech). */
  extraFragments: FragmentRow[];
  nudges: NudgeRow[];
  haptics: HapticRow[];
  /** Every positive DETECTION, cap-dropped ones included. */
  positives: PositiveRow[];
  /** 🧘 the longest run of the session with no alert escalation. */
  calm: CalmStreak;
  callMode: VectorEvent[];
  watch: WatchLane;
  scorecard: SceneScorecard;
}

// ---------------------------------------------------------------------------
// Watch lane: the shared policy over acoustic-only vectors
// ---------------------------------------------------------------------------

/** server/watch/vectors.py AIRTIME_WINDOW_S / AIRTIME_LEVELS. */
export const AIRTIME_WINDOW_S = 120;
/** server/watch/vectors.py AIRTIME_MIN_SPEECH_S: no airtime verdict until
 *  this much speech (self + others) sits in the window — the wearer's first
 *  sentence must not read as dominating. */
export const AIRTIME_MIN_SPEECH_S = 30;
export const AIRTIME_LEVELS: [number, number][] = [
  [0.9, 3],
  [0.75, 2],
  [0.6, 1],
];

export function airtimeLevel(share: number): number {
  for (const [threshold, level] of AIRTIME_LEVELS) if (share >= threshold) return level;
  return 0;
}

/**
 * Port of `VectorEngine._airtime_events`: after each turn is pushed, the
 * wearer's share of SPEECH (not wall clock) among the turns touching the
 * trailing 120 s window; a level when it clears 60/75/90 %. Evaluated as
 * the turns arrive in time order; an event is raised only when the turn
 * just pushed is the wearer's own (the share is measured about self, and a
 * buzz on someone else's turn would be the wrong moment on the wrist).
 */
export function airtimeEvents(turns: { speaker: string; start: number; end: number }[], selfSpeaker: string): VectorEvent[] {
  const out: VectorEvent[] = [];
  const seen: { speaker: string; start: number; end: number }[] = [];
  for (const turn of [...turns].sort((a, b) => a.start - b.start || a.end - b.end)) {
    seen.push(turn);
    if (turn.speaker !== selfSpeaker) continue;
    const now = Math.max(...seen.map((t) => t.end));
    const windowStart = Math.max(0, now - AIRTIME_WINDOW_S);
    let self = 0;
    let other = 0;
    for (const t of seen) {
      const ov = Math.min(t.end, now) - Math.max(t.start, windowStart);
      if (ov <= 0) continue;
      if (t.speaker === selfSpeaker) self += ov;
      else other += ov;
    }
    const total = self + other;
    if (total < AIRTIME_MIN_SPEECH_S) continue; // too little conversation to call anyone a hog
    const share = self / total;
    const level = airtimeLevel(share);
    if (level > 0) out.push({ vector: "airtime", level, t: now, value: Math.round(share * 1000) / 1000 });
  }
  return out;
}

/** The wrist's lane: the same hysteresis/cooldown machine as the phone
 *  (nudgePolicy.ts mirrors watch NudgeStateMachine.kt), subscribed to the
 *  vectors a watch can produce without words. Channel A only (no HR here). */
export function watchNudgePolicy(cooldownS = 20): NudgePolicy {
  return new NudgePolicy(
    [
      { vector: "yelling", sensitivity: 1.0, haptics: true, channel: "A" },
      { vector: "interrupting", sensitivity: 1.0, haptics: true, channel: "A" },
      { vector: "airtime", sensitivity: 1.0, haptics: true, channel: "A" },
    ],
    cooldownS,
    ["A"],
  );
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

const EPS = 1e-6;
const r1 = (x: number) => Math.round(x * 10) / 10;
const r3 = (x: number) => Math.round(x * 1000) / 1000;

export function firstWords(text: string, n = 8): string {
  const words = text.trim().split(/\s+/).filter(Boolean);
  return words.length <= n ? words.join(" ") : `${words.slice(0, n).join(" ")}…`;
}

/** Tally a list of strings — {"E": 2, "D": 1}, in first-seen order. */
function countBy(codes: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const c of codes) out[c] = (out[c] ?? 0) + 1;
  return out;
}

function stat(xs: number[]): Stat | null {
  if (xs.length === 0) return null;
  return { min: Math.min(...xs), median: median(xs), max: Math.max(...xs), n: xs.length };
}

export function fragmentLabel(t: LocalTurn): string {
  if (t.displayName) return t.displayName;
  if (t.personId) return t.speaker;
  return t.speaker === "Unknown" ? "Unknown" : `?${t.speaker}`;
}

/** The steamroll vector over the script's own timings: self turns that
 *  start inside another speaker's turn and keep going ≥ 2 s. */
export function callModeInterrupting(script: ReplayScript): VectorEvent[] {
  if (!script.selfSpeaker) return [];
  const self = script.turns.filter((t) => t.speaker === script.selfSpeaker);
  const others = script.turns.filter((t) => t.speaker !== script.selfSpeaker);
  return interruptingEvents(self, others);
}

export function buildNudgeReport(r: ReplayResult, generatedAt = new Date().toISOString()): NudgeReport {
  const script = r.script;
  const loopToScript = matchLoopTurns(script, r.turns);
  const loopByEnd = (t: number) => {
    const i = r.turns.findIndex((lt) => Math.abs(lt.endTime - t) < EPS);
    return i >= 0 ? i : null;
  };
  const callFor = (lt: LocalTurn) => r.policyLog.find((c) => Math.abs(c.t - lt.endTime) < EPS) ?? null;

  // --- nudge emissions ------------------------------------------------------
  const nudges: NudgeRow[] = r.nudgeLog.map((n) => {
    const loopTurn = loopByEnd(n.t);
    const scriptTurn = loopTurn === null ? null : loopToScript[loopTurn];
    return {
      t: n.t,
      atMs: n.atMs,
      lagMs: Math.round(n.atMs - n.t * 1000),
      level: n.level,
      vectors: n.vectors,
      decay: n.vectors.length === 0,
      loopTurn,
      scriptTurn,
      verdict: scriptTurn === null ? null : r.nudgeScore.perTurn[scriptTurn].verdict,
    };
  });

  // --- haptics: instant tier vs policy tier ---------------------------------
  // A policy-tier haptic is emitted inside the same emitNudges call as an
  // escalation NudgeEvent (same virtual instant, same level). Anything else
  // is the instant loudness tier, which fires between the previous turn's
  // policy tick and its own — so its turn is the first policy call at or
  // after the buzz.
  // Positives ride the same haptic sink on the device but are a different
  // lane entirely: they are soft, unleveled and capped, and pinning them
  // against the alert policy's escalations would be nonsense.
  const alertLog = r.hapticLog.filter((h) => {
    const entry = h.code ? vocabularyForCode(h.code) : null;
    return entry?.polarity !== "positive";
  });
  // --- 💚 positives ----------------------------------------------------------
  // What the user did WELL. Every detection is listed, cap-dropped ones
  // included, so a reader can see the praise the two-minute cap withheld.
  const positives: PositiveRow[] = r.positives.map((pos) => {
    const entry = vocabularyForCode(pos.code);
    return {
      code: pos.code,
      icon: entry?.icon ?? "?",
      name: entry?.name ?? pos.code,
      t: r3(pos.t),
      atMs: pos.atMs,
      loopTurn: pos.turnIndex,
      scriptTurn: loopToScript[pos.turnIndex] ?? null,
      delivered: pos.delivered,
      detail: pos.detail,
    };
  });

  const haptics: HapticRow[] = alertLog.map((h) => {
    const policyMatch = r.nudgeLog.find((n) => n.atMs === h.atMs && n.level === h.level && n.vectors.length > 0);
    if (policyMatch) {
      const loopTurn = loopByEnd(policyMatch.t);
      return {
        level: h.level,
        atMs: h.atMs,
        atSec: h.atSec,
        tier: "policy",
        loopTurn,
        scriptTurn: loopTurn === null ? null : loopToScript[loopTurn],
        leadOverPolicyMs: null,
      };
    }
    const call = r.policyLog.find((c) => Number.isFinite(c.atMs) && c.atMs >= h.atMs) ?? null;
    const loopTurn = call ? loopByEnd(call.t) : null;
    return {
      level: h.level,
      atMs: h.atMs,
      atSec: h.atSec,
      tier: "instant",
      loopTurn,
      scriptTurn: loopTurn === null ? null : loopToScript[loopTurn],
      leadOverPolicyMs: call ? Math.round(call.atMs - h.atMs) : null,
    };
  });

  // --- fragments (loop turns) --------------------------------------------
  // The loop folds only the coached user's turns into its loudness baseline
  // (fastLoop.finalizeTurn); replay that walk over the same turns, in order.
  const baseline = new LoudnessBaseline();
  const fragments: FragmentRow[] = r.turns.map((lt, i) => {
    const call = callFor(lt);
    const coached = call !== null && call.events.length > 0;
    let dbOver: number | null = null;
    let instantLevel = 0;
    if (coached) {
      const over = baseline.observe(lt.prosody.rms_dbfs);
      dbOver = baseline.value === null ? null : r1(over);
      instantLevel = call.events.find((e) => e.vector === "yelling")?.level ?? yellingLevel(over);
    }
    const toneFromPolicy = coached ? (call.events.find((e) => e.vector === "aggressive_tone")?.level ?? null) : null;
    const frustration = lt.textTone?.frustration ?? null;
    const defensiveness = lt.textTone?.defensiveness ?? null;
    return {
      index: lt.index,
      start: lt.startTime,
      end: lt.endTime,
      label: fragmentLabel(lt),
      isSelf: lt.isSelf,
      coachedAsSelf: coached,
      kind: lt.kind,
      text: firstWords(lt.text),
      transcriptFinal: lt.transcriptFinal,
      rmsDbfs: lt.prosody.rms_dbfs,
      dbOverBaseline: dbOver,
      instantLevel,
      toneLevel: toneFromPolicy ?? aggressiveToneLevel(frustration, defensiveness),
      frustration,
      defensiveness,
      activation: lt.activation ? { probability: r3(lt.activation.probability), level: lt.activation.level } : null,
      overlap: lt.overlap
        ? { mixedSeconds: lt.overlap.mixedSeconds, longestMixedRunSeconds: lt.overlap.longestMixedRunSeconds, windows: lt.overlap.windows }
        : null,
      policy: call ? { atMs: call.atMs, rawLevel: call.rawLevel, levelAfter: call.levelAfter } : null,
      scriptTurn: loopToScript[i],
      nudges: nudges.filter((n) => n.loopTurn === i),
      haptics: haptics.filter((h) => h.loopTurn === i),
      positives: positives.filter((pos) => pos.loopTurn === i),
      earpiece: earpieceLane(lt, i, nudges, haptics),
    };
  });

  // --- ⌚ watch lane -----------------------------------------------------------
  const watch = buildWatchLane(script, r.turns, fragments, loopToScript);

  // --- script turns ---------------------------------------------------------
  const callMode = callModeInterrupting(script);
  const turns: TurnRow[] = script.turns.map((st) => {
    const frags = fragments.filter((f) => f.scriptTurn === st.index);
    const outcome = r.nudgeScore.perTurn[st.index];
    const att = r.attribution.perTurn[st.index];
    const dbs = frags.map((f) => f.dbOverBaseline).filter((x): x is number => x !== null);
    const acts = frags.map((f) => f.activation).filter((a): a is NonNullable<FragmentRow["activation"]> => a !== null);
    const ovs = frags.map((f) => f.overlap).filter((o): o is NonNullable<FragmentRow["overlap"]> => o !== null);
    return {
      index: st.index,
      speaker: st.speaker,
      isSelf: st.speaker === script.selfSpeaker,
      start: st.start,
      end: st.end,
      text: firstWords(st.text),
      emotion: st.scriptedEmotion,
      expected: outcome.expected,
      verdict: outcome.verdict,
      level: outcome.level,
      emitted: outcome.emitted,
      predicted: att.predicted,
      attributionOk: att.ok,
      fragments: frags,
      dbOverBaselineMax: dbs.length ? Math.max(...dbs) : null,
      instantLevelMax: Math.max(0, ...frags.map((f) => f.instantLevel)),
      toneLevelMax: Math.max(0, ...frags.filter((f) => f.coachedAsSelf).map((f) => f.toneLevel)),
      activationMax: acts.length ? acts.reduce((a, b) => (b.probability > a.probability ? b : a)) : null,
      overlapMax: ovs.length
        ? {
            mixedSeconds: Math.max(...ovs.map((o) => o.mixedSeconds)),
            longestMixedRunSeconds: Math.max(...ovs.map((o) => o.longestMixedRunSeconds)),
          }
        : null,
      callMode: callMode.filter((e) => Math.abs(e.t - st.start) < EPS),
      earpieceLevel: Math.max(
        0,
        ...frags.map((f) => f.earpiece.instantHaptic?.level ?? 0),
        ...frags.flatMap((f) => f.earpiece.screenNudges.map((n) => n.level)),
      ),
      watchLevel: Math.max(0, ...watch.buzzes.filter((b) => b.scriptTurn === st.index).map((b) => b.level)),
      watchBuzzes: watch.buzzes.filter((b) => b.scriptTurn === st.index),
    };
  });
  for (const t of turns) {
    if (t.watchLevel > 0 && t.earpieceLevel > 0) watch.bothTurns.push(t.index);
    else if (t.watchLevel > 0) watch.watchOnlyTurns.push(t.index);
    else if (t.earpieceLevel > 0) watch.earpieceOnlyTurns.push(t.index);
  }

  // --- scorecard ------------------------------------------------------------
  const instant = haptics.filter((h) => h.tier === "instant");
  const leads = instant.map((h) => h.leadOverPolicyMs).filter((x): x is number => x !== null);
  let instantBeforePolicy: boolean | null = null;
  if (instant.length) {
    instantBeforePolicy = instant.every((h) => {
      if (h.leadOverPolicyMs === null || h.leadOverPolicyMs <= 0) return false;
      return nudges.filter((n) => n.loopTurn === h.loopTurn && !n.decay).every((n) => n.atMs > h.atMs);
    });
  }
  const enrolled = r.capability.enrolled.length
    ? r.capability.enrolled
        .map((e) => `${e.displayName}${e.isSelf ? " (self)" : ""} ← ${e.crossScene ? "cross-scene" : "same-scene"} ${e.fromScene}/${e.fromSpeaker}`)
        .join("; ")
    : r.capability.speakerId
      ? "nobody (unknown clusters only)"
      : "speaker-ID off";
  const activations = fragments.map((f) => f.activation).filter((a): a is NonNullable<FragmentRow["activation"]> => a !== null);
  // ⚡ Fragments the DARK classifier would have nudged on, sitting inside a
  // turn the script expects NOTHING for. Out of RAVDESS this is where the
  // classifier's real error rate shows up.
  const expectedTurns = new Set(turns.filter((t) => t.expected).map((t) => t.index));
  const activationFalse = fragments.filter(
    (f) => (f.activation?.level ?? 0) > 0 && !(f.scriptTurn !== null && expectedTurns.has(f.scriptTurn)),
  );
  const overlaps = fragments.map((f) => f.overlap).filter((o): o is NonNullable<FragmentRow["overlap"]> => o !== null);
  const a = r.attribution;
  const scorecard: SceneScorecard = {
    scene: r.scene,
    mode: r.mode,
    durationSec: r1(r.durationSec),
    selfSpeaker: script.selfSpeaker,
    enrolled,
    scriptTurns: script.turns.length,
    loopTurns: r.turns.length,
    coachedFragments: fragments.filter((f) => f.coachedAsSelf).length,
    expected: script.expectedNudges.length,
    hits: r.nudgeScore.hits,
    misses: r.nudgeScore.misses,
    falsePositives: r.nudgeScore.falsePositives,
    hitsSilent: r.nudgeScore.hitsSilent,
    instantHaptics: instant.length,
    policyHaptics: haptics.length - instant.length,
    instantLeadMs: stat(leads),
    instantBeforePolicy,
    policyLagMs: stat(nudges.filter((n) => !n.decay).map((n) => n.lagMs)),
    attribution: { correct: a.correct, total: a.total, selfCorrect: a.selfCorrect, selfTotal: a.selfTotal },
    positives: {
      delivered: countBy(positives.filter((pos) => pos.delivered).map((pos) => pos.code)),
      suppressed: positives.filter((pos) => !pos.delivered).length,
      calmStreakS: r1(r.calm.longestS),
      calmBadge: r.calm.badge,
    },
    callModeInterrupting: callMode.length,
    activationProbed: activations.length,
    activationFalseFlags: activationFalse.length,
    activationFlaggedTurns: [...new Set(activationFalse.map((f) => f.index))].sort((a, b) => a - b),
    activationMaxProbability: activations.length ? Math.max(...activations.map((x) => x.probability)) : null,
    overlapProbed: overlaps.length,
    overlapMaxMixedSeconds: overlaps.length ? Math.max(...overlaps.map((o) => o.mixedSeconds)) : null,
    spokenOverSpeech: r.speaking.overVadSpeech,
    earpiece: {
      instantHaptics: fragments.filter((f) => f.earpiece.instantHaptic).length,
      screenNudges: fragments.reduce((n, f) => n + f.earpiece.screenNudges.length, 0),
      spokenNudges: fragments.filter((f) => f.earpiece.spokenAtSec !== null && f.earpiece.suggestionKind === "nudge").length,
      spokenResponses: fragments.filter((f) => f.earpiece.spokenAtSec !== null && f.earpiece.suggestionKind === "response").length,
      nudgeToSpeakMs: stat(
        fragments
          .filter((f) => f.earpiece.spokenAtSec !== null && f.earpiece.suggestionKind === "nudge")
          .map((f) => Math.round(((f.earpiece.spokenAtSec as number) - f.end) * 1000)),
      ),
    },
    watch: {
      buzzes: watch.buzzes.length,
      byVector: watch.byVector,
      watchOnlyTurns: watch.watchOnlyTurns,
      earpieceOnlyTurns: watch.earpieceOnlyTurns,
      bothTurns: watch.bothTurns,
      allOnSelfTurns: watch.allOnSelfTurns,
    },
  };

  return {
    scene: r.scene,
    mode: r.mode,
    generatedAt,
    durationSec: r1(r.durationSec),
    speakers: script.speakers,
    selfSpeaker: script.selfSpeaker,
    approxBoundaries: script.approxBoundaries,
    turns,
    extraFragments: fragments.filter((f) => f.scriptTurn === null),
    nudges,
    haptics,
    positives,
    calm: r.calm,
    callMode,
    watch,
    scorecard,
  };
}

/** 🎧 The earpiece lane of one loop turn, from the replay's own logs. The
 *  spoken instant is segmentEnd + toSpeakMs on the virtual clock (the loop
 *  starts at virtual 0). */
function earpieceLane(lt: LocalTurn, i: number, nudges: NudgeRow[], haptics: HapticRow[]): EarpieceLane {
  const instant = haptics.find((h) => h.tier === "instant" && h.loopTurn === i) ?? null;
  const spokenAtSec = lt.latency.toSpeakMs === null ? null : r3((lt.latency.segmentEndMs + lt.latency.toSpeakMs) / 1000);
  return {
    instantHaptic: instant ? { atSec: r3(instant.atSec), level: instant.level } : null,
    screenNudges: nudges.filter((n) => n.loopTurn === i && !n.decay).map((n) => ({ atSec: r3(n.atMs / 1000), level: n.level, vectors: n.vectors })),
    spokenAtSec,
    spokenText: spokenAtSec === null ? null : lt.suggestion,
    suggestionKind: lt.suggestionKind,
  };
}

/**
 * ⌚ Run the shared policy over what a wrist could measure for the same
 * audio: per coached-self fragment a `yelling` event at its end (the
 * loop's own dB-over-baseline level — what watch/relay.py forwards from the
 * phone's per-turn loudness), a bare tick on every other fragment (the
 * cooldown clock, exactly as the phone ticks), plus `interrupting` at the
 * start of each ground-truth self turn that talks over someone for >= 2 s
 * and `airtime` at the end of self turns whose share clears 60 %.
 */
function buildWatchLane(script: ReplayScript, loopTurns: LocalTurn[], fragments: FragmentRow[], loopToScript: (number | null)[]): WatchLane {
  const self = script.selfSpeaker;
  const ticks = new Map<number, VectorEvent[]>();
  const add = (t: number, e: VectorEvent | null) => {
    const list = ticks.get(t) ?? [];
    if (e) list.push(e);
    ticks.set(t, list);
  };
  for (const f of fragments) {
    add(f.end, f.coachedAsSelf ? { vector: "yelling", level: f.instantLevel, t: f.end, value: f.dbOverBaseline ?? 0 } : null);
  }
  const ground: VectorEvent[] = [];
  if (self) {
    ground.push(...callModeInterrupting(script));
    ground.push(...airtimeEvents(script.turns, self));
  }
  for (const e of ground) add(e.t, e);
  const policy = watchNudgePolicy();
  const events: VectorEvent[] = [];
  const emitted: (NudgeEvent & { t: number })[] = [];
  for (const t of [...ticks.keys()].sort((a, b) => a - b)) {
    const evs = ticks.get(t) ?? [];
    events.push(...evs);
    emitted.push(...policy.onEvents(evs, t));
  }
  const selfTurnAt = (t: number) => script.turns.find((st) => st.speaker === self && st.start - EPS <= t && t <= st.end + EPS) ?? null;
  const rows: WatchBuzz[] = emitted.map((n) => {
    const li = loopTurns.findIndex((lt) => Math.abs(lt.endTime - n.t) < EPS);
    const loopTurn = li >= 0 ? li : null;
    const frag = loopTurn === null ? null : fragments[loopTurn];
    const scriptTurn = loopTurn !== null ? loopToScript[loopTurn] : (selfTurnAt(n.t)?.index ?? null);
    const groundSelf = selfTurnAt(n.t) !== null;
    const onSelfTurn = frag ? frag.coachedAsSelf && scriptTurn !== null && script.turns[scriptTurn].speaker === self : groundSelf;
    return { t: n.t, level: n.level, vectors: n.vectors, decay: n.vectors.length === 0, loopTurn, scriptTurn, onSelfTurn };
  });
  const buzzes = rows.filter((b) => !b.decay);
  const byVector: Record<string, number> = {};
  for (const b of buzzes) for (const v of b.vectors) byVector[v] = (byVector[v] ?? 0) + 1;
  return {
    events,
    buzzes,
    decays: rows.filter((b) => b.decay),
    byVector,
    watchOnlyTurns: [],
    earpieceOnlyTurns: [],
    bothTurns: [],
    allOnSelfTurns: buzzes.every((b) => b.onSelfTurn),
  };
}

// ---------------------------------------------------------------------------
// Gate
// ---------------------------------------------------------------------------

export interface NudgeGate {
  /** Expected nudges that must be hit — the number replay.scenes.test.ts pins. */
  minHits: number;
}

/** Empty when the scene passes; otherwise one line per broken rule. */
export function gateFailures(report: NudgeReport, gate: NudgeGate): string[] {
  const s = report.scorecard;
  const out: string[] = [];
  if (s.hits < gate.minHits) out.push(`${s.scene}: expected nudges hit ${s.hits} < pinned ${gate.minHits}`);
  if (s.falsePositives !== 0) {
    const fps = report.turns.filter((t) => t.verdict === "fp").map((t) => `#${t.index} ${t.speaker} "${t.text}"`);
    out.push(`${s.scene}: ${s.falsePositives} false positive(s): ${fps.join("; ")}`);
  }
  if (s.instantBeforePolicy === false) {
    const late = report.haptics
      .filter((h) => h.tier === "instant" && (h.leadOverPolicyMs === null || h.leadOverPolicyMs <= 0))
      .map((h) => `L${h.level}@${h.atSec.toFixed(2)}s`);
    out.push(`${s.scene}: instant haptic did not precede the LLM tier: ${late.join(", ") || "(ordering)"}`);
  }
  // 💚 positives: soft, unleveled, and never more often than the cap allows —
  // praise on a loop is its own nag, and that is the one way a positive can
  // do harm.
  const delivered = report.positives.filter((pos) => pos.delivered);
  for (let i = 1; i < delivered.length; i++) {
    const gapS = delivered[i].t - delivered[i - 1].t;
    if (gapS < POSITIVE_CAP_S) {
      out.push(
        `${s.scene}: positives ${delivered[i - 1].code}@${delivered[i - 1].t.toFixed(1)}s and ${delivered[i].code}@${delivered[i].t.toFixed(1)}s are ${gapS.toFixed(1)}s apart, under the ${POSITIVE_CAP_S}s cap`,
      );
    }
  }
  const unknown = report.positives.filter((pos) => vocabularyForCode(pos.code)?.polarity !== "positive");
  if (unknown.length) {
    out.push(`${s.scene}: ${unknown.length} positive row(s) carry a non-positive code: ${unknown.map((pos) => pos.code).join(", ")}`);
  }

  const offSelf = report.watch.buzzes.filter((b) => !b.onSelfTurn);
  if (offSelf.length) {
    out.push(`${s.scene}: watch lane buzzed on a non-self turn: ${offSelf.map((b) => `L${b.level}[${b.vectors.join(",")}]@${b.t.toFixed(2)}s`).join(", ")}`);
  }
  return out;
}

// ---------------------------------------------------------------------------
// AMI section (server/watch/vectors.py over real overlapping speech; JSON
// produced by tmp/nudge-report/ami_vectors.py)
// ---------------------------------------------------------------------------

export interface AmiVectorEvent {
  vector: string;
  level: number;
  t: number;
  value: number;
  detail: string;
}

export interface AmiSpeakerVectors {
  interrupting: AmiVectorEvent[];
  airtime: AmiVectorEvent[];
}

export interface AmiClip {
  name: string;
  path: string;
  duration_sec: number;
  diarization: {
    num_speakers: number;
    k_eigengap: number;
    windows: number;
    hop: number;
    cluster_seconds: Record<string, number>;
    segments: { start: number; end: number; label: string }[];
  };
  /** Segments as turns, each speaker in turn as "self" — non-overlapping by
   *  construction, so `interrupting` is empty here and `airtime` is the
   *  streaming share series. */
  timeline: Record<string, AmiSpeakerVectors>;
  /** The single-mic overlap probe (overlapProbe.ts rules) over the window
   *  embeddings: a window with two voices within the margin belongs to both,
   *  so spans CAN overlap and the steamroll vector has something to see. */
  probe: {
    window_sec: number;
    hop_sec: number;
    margin: number;
    min_score: number;
    windows: number;
    mixed_windows: number;
    mixed_seconds: number;
    spans: Record<string, [number, number][]>;
    vectors: Record<string, AmiSpeakerVectors>;
  };
}

export interface AmiVectorsJson {
  generated_at: string;
  engine: string;
  model: string;
  python: string;
  notes: string[];
  clips: AmiClip[];
}

// ---------------------------------------------------------------------------
// HTML
// ---------------------------------------------------------------------------

export function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

const fmt = {
  s: (x: number) => `${x.toFixed(1)}s`,
  ms: (x: number | null | undefined) => (x === null || x === undefined || !Number.isFinite(x) ? "–" : `${Math.round(x)} ms`),
  db: (x: number | null) => (x === null ? "–" : `${x >= 0 ? "+" : ""}${x.toFixed(1)} dB`),
  pct: (x: number | null) => (x === null ? "–" : `${Math.round(x * 100)}%`),
  stat: (st: Stat | null) => (st === null ? "–" : st.n === 1 ? fmt.ms(st.median) : `${fmt.ms(st.median)} (min ${fmt.ms(st.min)}, max ${fmt.ms(st.max)}, n=${st.n})`),
};

const LEVEL_ICON = ["·", "L1", "L2", "L3"];
const VERDICT_MARK: Record<Verdict, string> = { hit: "✓", miss: "✗", fp: "FP", quiet: "" };

// Legend + card markup follow tmp/hlist-20260905-abcde.html (docs/ICON-LEGEND.md).
const LEGEND_HTML = `<details class="legend"><summary><b>Tags:</b> displayed | method | latency | computed &nbsp;·&nbsp; <span title="~1 second — on-device acoustics (the buzz)">⚡</span><span title="3–7 seconds — the LLM's words">🧠</span><span title="minutes — after the session ends (summary, replay, re-analyze)">⏳</span><span title="hours / overnight — batch runs, corpus mining">🌙</span> latency &nbsp;·&nbsp; <span style="opacity:.8">tap for legend</span></summary>
<div class="legendbody">
<b>1 · Where displayed:</b> 🎧 earpiece · ⌚ watch · 📱 phone · 💻 computer · – nowhere<br>
<b>2 · Method:</b> 🗣️ voice · 📳 vibrate · ✨ flash · 📝 text · 📄 report · 👓 Developer-mode only · 🔕 not shown (numbers only) · 🚧 planned<br>
<b>3 · Latency:</b> ⚡ instant ≈ <b>1 s</b> · 🧠 LLM ≈ <b>3–7 s</b> · ⏳ post-session ≈ <b>minutes</b> · 🌙 batch ≈ <b>hours/overnight</b><br>
<b>4 · Where computed:</b> 📱 phone · ⌚ watch · 🌐 browser · ☁️ cloud · 🔀 hybrid &nbsp;<span style="opacity:.8">(docs/ICON-LEGEND.md)</span>
</div></details>`;

const STYLE = `
:root{--bg:#fff;--fg:#111827;--muted:#6b7280;--line:#e5e7eb;--card:#f9fafb;--done:#16a34a;--prog:#d97706;--todo:#9ca3af;--bad:#dc2626;--self:#2563eb;--other:#9ca3af;--truth:#d1d5db;--accent:#7c3aed;--tag:#eef2ff}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0b0f17;--fg:#e5e7eb;--muted:#9ca3af;--line:#1f2937;--card:#111827;--done:#4ade80;--prog:#fbbf24;--todo:#6b7280;--bad:#f87171;--self:#60a5fa;--other:#6b7280;--truth:#374151;--accent:#c4b5fd;--tag:#1e1b4b}}
:root[data-theme="dark"]{--bg:#0b0f17;--fg:#e5e7eb;--muted:#9ca3af;--line:#1f2937;--card:#111827;--done:#4ade80;--prog:#fbbf24;--todo:#6b7280;--bad:#f87171;--self:#60a5fa;--other:#6b7280;--truth:#374151;--accent:#c4b5fd;--tag:#1e1b4b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,Segoe UI,Roboto,sans-serif;padding:14px;max-width:760px;margin-inline:auto;overflow-x:hidden}
h1{font-size:17px;margin:0 0 10px}h2{font-size:15px;margin:14px 0 6px}
details{border:1px solid var(--line);border-radius:10px;background:var(--card);margin:8px 0}
summary{display:flex;align-items:center;gap:10px;padding:11px 12px;cursor:pointer;list-style:none}
summary::-webkit-details-marker{display:none}
.dot{width:10px;height:10px;border-radius:50%;flex:none}.t{flex:1;font-weight:600;min-width:0}.d{color:var(--muted);font-size:12.5px;flex:none;text-align:right}
.body{padding:2px 12px 12px 12px;color:var(--muted);font-size:13.5px}.body b{color:var(--fg)}
code{background:var(--line);border-radius:4px;padding:1px 5px;font-size:12px}
details.legend{border:1px solid var(--line);border-radius:10px;background:var(--card);margin:6px 0 12px;font-size:12.5px;color:var(--muted)}
details.legend summary{padding:8px 12px;cursor:pointer;list-style:none}details.legend summary::-webkit-details-marker{display:none}
details.legend .legendbody{padding:0 12px 10px;line-height:1.7}details.legend .legendbody b{color:var(--fg)}
[title]{cursor:help}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:6px 0}
table{border-collapse:collapse;font-size:12.5px;min-width:100%}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top;white-space:nowrap}th{color:var(--muted);font-weight:600}
td.txt{white-space:normal;min-width:160px;max-width:280px}
tr.self td{background:color-mix(in srgb,var(--self) 8%,transparent)}
.ok{color:var(--done);font-weight:700}.bad{color:var(--bad);font-weight:700}.warn{color:var(--prog);font-weight:700}
.tag{display:inline-block;background:var(--tag);border-radius:5px;padding:0 5px;font-size:11.5px;margin-right:4px;white-space:nowrap}
svg.tl{width:100%;min-width:720px;height:auto;display:block;margin:6px 0 2px;font:11px -apple-system,system-ui,sans-serif}
svg.tl text{fill:var(--fg)}svg.tl .ax{stroke:var(--line)}svg.tl .truth{fill:var(--truth)}svg.tl .self{fill:var(--self)}svg.tl .other{fill:var(--other)}svg.tl .lane{fill:var(--muted);font-size:10.5px}
svg.tl .mark{font-size:12px}svg.tl .tagt{font-size:9.5px;fill:var(--accent)}svg.tl .v{font-size:11px;font-weight:700}svg.tl .hit{fill:var(--done)}svg.tl .miss,svg.tl .fp{fill:var(--bad)}svg.tl .call{fill:var(--prog)}
svg.tl .probe{fill:var(--accent);opacity:.55}svg.tl .tick{fill:var(--muted);font-size:9.5px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:13px}.kv div:nth-child(odd){color:var(--muted)}
ul.feat{list-style:none;padding:0;margin:6px 0}ul.feat li{padding:4px 0;border-bottom:1px solid var(--line);font-size:13px}ul.feat li:last-child{border:0}
`;

function tag(where: string, method: string, latency: string, computed: string): string {
  return `<span class="tag" title="displayed | method | latency | computed">${where} | ${method} | ${latency} | ${computed}</span>`;
}

// --- timeline SVG ------------------------------------------------------------

const W = 1000;
const LANE_H = 26;
const TOP = 46;

function ticks(duration: number): number[] {
  const step = duration <= 45 ? 5 : duration <= 120 ? 10 : duration <= 400 ? 30 : 60;
  const out: number[] = [];
  for (let t = 0; t <= duration + EPS; t += step) out.push(t);
  return out;
}

function sceneTimelineSvg(rep: NudgeReport): string {
  const dur = Math.max(1, rep.durationSec + 1);
  const x = (t: number) => (Math.min(dur, Math.max(0, t)) / dur) * W;
  const lanes = [...rep.speakers];
  if (rep.extraFragments.length) lanes.push("(no script turn)");
  const H = TOP + lanes.length * LANE_H + 18;
  const parts: string[] = [];
  parts.push(`<div class="scroll"><svg class="tl" viewBox="0 0 ${W} ${H}" role="img" aria-label="timeline of ${escapeHtml(rep.scene)}">`);
  // grid + axis
  for (const t of ticks(dur)) {
    parts.push(`<line class="ax" x1="${x(t).toFixed(1)}" y1="${TOP - 4}" x2="${x(t).toFixed(1)}" y2="${H - 16}" stroke-width="1"/>`);
    parts.push(`<text class="tick" x="${(x(t) + 2).toFixed(1)}" y="${H - 5}">${t}s</text>`);
  }
  // lanes: ground-truth turns (light) + loop fragments (coloured)
  lanes.forEach((lane, li) => {
    const y = TOP + li * LANE_H;
    parts.push(`<text class="lane" x="0" y="${y + 9}">${escapeHtml(lane === rep.selfSpeaker ? `${lane} (you)` : lane)}</text>`);
    const truth = rep.turns.filter((t) => t.speaker === lane);
    for (const t of truth) {
      parts.push(`<rect class="truth" x="${x(t.start).toFixed(1)}" y="${y + 12}" width="${Math.max(1.5, x(t.end) - x(t.start)).toFixed(1)}" height="6" rx="1"/>`);
    }
    const frags = lane === "(no script turn)" ? rep.extraFragments : truth.flatMap((t) => t.fragments);
    for (const f of frags) {
      const cls = f.coachedAsSelf ? "self" : "other";
      parts.push(`<rect class="${cls}" x="${x(f.start).toFixed(1)}" y="${y + 19}" width="${Math.max(1.5, x(f.end) - x(f.start)).toFixed(1)}" height="4" rx="1"/>`);
      // dark-probe tags on the coached user's longer fragments
      if (f.coachedAsSelf && f.end - f.start >= 1.5) {
        const bits: string[] = [];
        if (f.activation) bits.push(`⚡${Math.round(f.activation.probability * 100)}%`);
        if (f.overlap) bits.push(`⟂${f.overlap.mixedSeconds.toFixed(1)}s`);
        if (bits.length) parts.push(`<text class="tagt" x="${x(f.start).toFixed(1)}" y="${y + 9}">${bits.join(" ")}</text>`);
      }
    }
    // verdicts on expected / fp turns
    for (const t of truth) {
      if (t.verdict === "quiet") continue;
      parts.push(`<text class="v ${t.verdict}" x="${(x(t.end) - 8).toFixed(1)}" y="${y + 9}">${VERDICT_MARK[t.verdict]}</text>`);
    }
    // call-mode steamroll events on the ground-truth self turns
    for (const t of truth) {
      for (const e of t.callMode) parts.push(`<text class="mark call" x="${(x(e.t) - 5).toFixed(1)}" y="${y + 10}"><title>call mode: interrupting L${e.level}, ${e.value}s of mutual speech</title>⟂</text>`);
    }
  });
  // 🎧 earpiece row: ⚡ instant haptics, 🗣️ spoken nudge lines, 🧠 screen nudges
  const yEar = TOP - 22;
  const yWatch = TOP - 8;
  parts.push(`<text class="lane" x="0" y="${yEar}">🎧</text><text class="lane" x="0" y="${yWatch}">⌚</text>`);
  for (const h of rep.haptics.filter((h) => h.tier === "instant")) {
    parts.push(`<text class="mark" x="${(x(h.atSec) - 6).toFixed(1)}" y="${yEar}"><title>instant haptic L${h.level} at ${h.atSec.toFixed(2)}s (${fmt.ms(h.leadOverPolicyMs)} before the LLM tier)</title>⚡</text>`);
  }
  const frags = [...rep.turns.flatMap((t) => t.fragments), ...rep.extraFragments];
  for (const f of frags) {
    const e = f.earpiece;
    if (e.spokenAtSec !== null && e.suggestionKind === "nudge") {
      parts.push(`<text class="mark" x="${(x(e.spokenAtSec) - 6).toFixed(1)}" y="${yEar}"><title>spoken at ${e.spokenAtSec.toFixed(2)}s: "${escapeHtml(e.spokenText ?? "")}" (${Math.round((e.spokenAtSec - f.end) * 1000)} ms after your turn closed)</title>🗣️</text>`);
    }
  }
  for (const n of rep.nudges.filter((n) => !n.decay)) {
    parts.push(`<text class="mark" x="${(x(n.atMs / 1000) - 6).toFixed(1)}" y="${yEar}"><title>on-screen nudge L${n.level} [${n.vectors.join(",")}] at ${(n.atMs / 1000).toFixed(2)}s (turn closed ${n.t.toFixed(2)}s)</title>🧠</text>`);
  }
  for (const n of rep.nudges.filter((n) => n.decay)) {
    parts.push(`<text class="mark" x="${(x(n.atMs / 1000) - 6).toFixed(1)}" y="${yEar}" opacity=".5"><title>cooldown decay to L${n.level} (screen only)</title>↓</text>`);
  }
  // ⌚ watch row: buzzes from the acoustic-only policy run
  for (const b of rep.watch.buzzes) {
    parts.push(`<text class="mark ${b.onSelfTurn ? "" : "fp"}" x="${(x(b.t) - 6).toFixed(1)}" y="${yWatch}"><title>watch buzz L${b.level} [${b.vectors.join(",")}] at ${b.t.toFixed(2)}s${b.onSelfTurn ? "" : " — NOT on your turn"}</title>⌚</text>`);
  }
  for (const b of rep.watch.decays) {
    parts.push(`<text class="mark" x="${(x(b.t) - 6).toFixed(1)}" y="${yWatch}" opacity=".5"><title>watch face decay to L${b.level}</title>↓</text>`);
  }
  // 💚 positives ride the 🎧 row: they are felt in the same place, and seeing
  // them next to the complaints is the whole point of the lane.
  for (const pos of rep.positives) {
    const suppressed = pos.delivered ? "" : ' opacity=".35"';
    const why = pos.delivered ? "" : " — DETECTED but withheld by the 2-minute cap";
    parts.push(`<text class="mark" x="${(x(pos.t) - 6).toFixed(1)}" y="${yEar}"${suppressed}><title>${escapeHtml(pos.name)} (${pos.code}) at ${pos.t.toFixed(2)}s: ${escapeHtml(pos.detail)}${why}</title>${pos.icon}</text>`);
  }
  parts.push(`<text class="tick" x="0" y="10">🎧 row: ⚡ instant haptic · 🗣️ spoken nudge line · 🧠 on-screen nudge · ↓ decay · 📉👂🤝 positives (faded = withheld by the cap) &nbsp; ⌚ row: wrist buzz &nbsp; lanes: ⟂ call-mode steamroll · ✓/✗/FP verdicts · light dash = script, dark = loop (blue = coached as you)</text>`);
  parts.push("</svg></div>");
  return parts.join("");
}

// --- scene block -------------------------------------------------------------

/** The time of an airtime buzz inside the first 30 s (the engine rule's
 *  "you own 100 % of the first sentence" artefact), or null. */
function earlyAirtime(rep: NudgeReport): string | null {
  const b = rep.watch.buzzes.find((b) => b.vectors.includes("airtime") && b.t < 30);
  return b ? b.t.toFixed(1) : null;
}

function verdictCell(v: Verdict, expected: "mild" | "strong" | null): string {
  if (v === "hit") return `<span class="ok">✓ hit</span> <span style="opacity:.7">(${expected})</span>`;
  if (v === "miss") return `<span class="bad">✗ miss</span> <span style="opacity:.7">(${expected})</span>`;
  if (v === "fp") return `<span class="bad">FP</span>`;
  return "";
}

function levelCell(level: number): string {
  return level > 0 ? `<b>${LEVEL_ICON[Math.min(3, level)]}</b>` : "·";
}

/** 🎧 cell: instant haptic · spoken line · screen nudge, with times. */
function earpieceCell(f: FragmentRow): string {
  const e = f.earpiece;
  const bits: string[] = [];
  if (e.instantHaptic) bits.push(`<b>⚡L${e.instantHaptic.level}</b>@${e.instantHaptic.atSec.toFixed(2)}s`);
  for (const n of e.screenNudges) bits.push(`<b>🧠L${n.level}</b>[${n.vectors.join(",")}]@${n.atSec.toFixed(2)}s`);
  if (e.spokenAtSec !== null) bits.push(`${e.suggestionKind === "nudge" ? "🗣️" : "💬"}@${e.spokenAtSec.toFixed(2)}s <span title="${escapeHtml(e.spokenText ?? "")}" style="opacity:.7">“${escapeHtml(firstWords(e.spokenText ?? "", 4))}”</span>`);
  return bits.join("<br>");
}

/** 💚 cell: the positives this fragment earned, faded when the cap withheld them. */
function positiveCell(rows: PositiveRow[]): string {
  return rows
    .map(
      (pos) =>
        `<span title="${escapeHtml(pos.detail)}${pos.delivered ? "" : " — withheld by the 2-minute cap"}"${pos.delivered ? "" : ' style="opacity:.45"'}>${pos.icon} <b>${escapeHtml(pos.name)}</b>@${pos.t.toFixed(1)}s</span>`,
    )
    .join("<br>");
}

function watchCell(buzzes: WatchBuzz[]): string {
  return buzzes.map((b) => `<b>⌚L${b.level}</b>[${b.vectors.join(",")}]@${b.t.toFixed(2)}s${b.onSelfTurn ? "" : ' <span class="bad">not you!</span>'}`).join("<br>");
}

function fragmentRows(t: TurnRow, watch: WatchLane): string {
  return t.fragments
    .map((f) => {
      const wb = watch.buzzes.filter((b) => b.loopTurn === f.index);
      return `<tr class="frag"><td>↳${f.index}</td><td>${escapeHtml(f.label)}${f.coachedAsSelf ? " ★" : ""}${f.kind === "backchannel" ? " (backchannel)" : ""}</td><td>${f.start.toFixed(2)}–${f.end.toFixed(2)}</td><td class="txt">${escapeHtml(f.text)}${f.transcriptFinal ? "" : " <i>(interim)</i>"}</td><td>${fmt.db(f.dbOverBaseline)}</td><td>${levelCell(f.instantLevel)}</td><td>${f.coachedAsSelf ? levelCell(f.toneLevel) : "·"}</td><td>${f.activation ? `${fmt.pct(f.activation.probability)} ${f.activation.level ? `L${f.activation.level}` : ""}` : "–"}</td><td>${f.overlap ? `${f.overlap.mixedSeconds.toFixed(1)}s / run ${f.overlap.longestMixedRunSeconds.toFixed(1)}s` : "–"}</td><td>${f.policy ? `${f.policy.rawLevel}→${f.policy.levelAfter}` : "–"}</td><td>${earpieceCell(f)}</td><td>${watchCell(wb)}</td><td>${positiveCell(f.positives)}</td><td></td></tr>`;
    })
    .join("");
}

function turnTable(rep: NudgeReport): string {
  const head = `<tr><th>#</th><th>who</th><th>when</th><th>words</th><th title="dB over your own running baseline">dB over</th><th title="instant loudness tier level">⚡lvl</th><th title="text-tone level from the LLM's frustration/defensiveness">🧠tone</th><th title="vocal activation probability (dark)">⚡% dark</th><th title="single-mic overlap probe: mixed-voice seconds (dark)">⟂ dark</th><th title="policy raw level → level held after">policy</th><th title="what you would feel in the earpiece / on the phone: instant haptic, screen nudge, spoken line">🎧 earpiece</th><th title="what the wrist would buzz (acoustic-only policy run)">⌚ watch</th><th title="what you did well: D de-escalated, E listened, R repair (faded = withheld by the 2-minute cap)">💚 well</th><th title="call-mode equivalent: interruptingEvents over the ground-truth timings">⟂ call</th></tr>`;
  const rows = rep.turns
    .map((t) => {
      const call = t.callMode.map((e) => `L${e.level} ${e.value}s`).join(" ");
      const who = `${escapeHtml(t.speaker)}${t.isSelf ? " (you)" : ""}${t.attributionOk ? "" : ` <span class="warn" title="loop heard ${escapeHtml(t.predicted ?? "nobody")}">≠${escapeHtml(t.predicted ?? "?")}</span>`}`;
      const lane = t.watchLevel > 0 && t.earpieceLevel === 0 ? '<br><span class="warn">⌚ only</span>' : t.earpieceLevel > 0 && t.watchLevel === 0 ? '<br><span class="warn">🎧 only</span>' : "";
      const main = `<tr class="${t.isSelf ? "self" : ""}"><td><b>${t.index}</b></td><td>${who}</td><td>${t.start.toFixed(1)}–${t.end.toFixed(1)}</td><td class="txt">${escapeHtml(t.text)}${t.emotion ? ` <span style="opacity:.6">[${escapeHtml(t.emotion)}]</span>` : ""}</td><td>${fmt.db(t.dbOverBaselineMax)}</td><td>${levelCell(t.instantLevelMax)}</td><td>${levelCell(t.toneLevelMax)}</td><td>${t.activationMax ? fmt.pct(t.activationMax.probability) : "–"}</td><td>${t.overlapMax ? `${t.overlapMax.mixedSeconds.toFixed(1)}s` : "–"}</td><td>${t.level}${t.verdict !== "quiet" ? `<br>${verdictCell(t.verdict, t.expected)}` : ""}</td><td>${t.earpieceLevel ? `<b>L${t.earpieceLevel}</b>` : "·"}${lane}</td><td>${t.watchLevel ? `<b>L${t.watchLevel}</b>` : "·"}</td><td>${positiveCell(rep.positives.filter((pos) => pos.scriptTurn === t.index))}</td><td>${call || (t.isSelf ? "·" : "")}</td></tr>`;
      return main + fragmentRows(t, rep.watch);
    })
    .join("");
  const extra = rep.extraFragments.length
    ? `<tr><td colspan="14" style="color:var(--muted)">${rep.extraFragments.length} loop turn(s) outside every scripted turn: ${rep.extraFragments.map((f) => `#${f.index} ${f.start.toFixed(1)}–${f.end.toFixed(1)} ${escapeHtml(f.label)}`).join("; ")}</td></tr>`
    : "";
  return `<div class="scroll"><table>${head}${rows}${extra}</table></div><div style="font-size:12px;color:var(--muted)">★ = the loop coached this fragment as you · ↳ rows are the loop's own turns (the segmenter cuts scripted turns at 300 ms pauses) · dB over = loudness over your own running median · ⚡% and ⟂ are dark probes (measured, never buzzing) · 🗣️ = nudge line spoken to you, 💬 = a response suggested on someone else's turn</div>`;
}

function scorecardHtml(rep: NudgeReport, gate: NudgeGate | null): string {
  const s = rep.scorecard;
  const fails = gate ? gateFailures(rep, gate) : [];
  const pass = gate ? fails.length === 0 : null;
  const lead = s.instantLeadMs;
  const w = s.watch;
  const e = s.earpiece;
  const turnList = (xs: number[]) => (xs.length ? xs.map((i) => `#${i}`).join(" ") : "none");
  const items: string[] = [
    `${tag("📱", "📳", "🧠", "📱")} <b>Expected nudges hit ${s.hits}/${s.expected}</b>, misses ${s.misses}, false positives <span class="${s.falsePositives ? "bad" : "ok"}">${s.falsePositives}</span>${s.hitsSilent ? ` (${s.hitsSilent} silent: level already held)` : ""}${gate ? ` · gate: hits ≥ ${gate.minHits}, fp = 0 → <span class="${pass ? "ok" : "bad"}">${pass ? "PASS" : "FAIL"}</span>` : ""}`,
    `${tag("🎧", "🗣️📳", "⚡🧠", "📱")} <b>Earpiece lane — what you would have felt</b>: ${e.instantHaptics} instant buzz${e.instantHaptics === 1 ? "" : "es"}${lead ? ` (${fmt.stat(lead)} ahead of the LLM tier)` : ""}, ${e.screenNudges} on-screen nudge${e.screenNudges === 1 ? "" : "s"} (${fmt.stat(s.policyLagMs)} after your turn closed), ${e.spokenNudges} nudge line${e.spokenNudges === 1 ? "" : "s"} spoken to you${e.nudgeToSpeakMs ? ` (first word ${fmt.stat(e.nudgeToSpeakMs)} after your turn)` : ""} + ${e.spokenResponses} response line${e.spokenResponses === 1 ? "" : "s"} on others' turns`,
    `${tag("⌚", "📳", "⚡", "⌚")} <b>Watch lane — what the wrist would buzz</b> (same policy, loudness + interrupting + airtime only): ${w.buzzes} buzz${w.buzzes === 1 ? "" : "es"}${w.buzzes ? ` ${Object.entries(w.byVector).map(([k, v]) => `${k}×${v}`).join(", ")}` : ""} · ${w.allOnSelfTurns ? '<span class="ok">never on someone else’s turn</span>' : '<span class="bad">buzzed on a non-self turn</span>'} · watch-only turns: ${turnList(w.watchOnlyTurns)} · earpiece-only turns: ${turnList(w.earpieceOnlyTurns)} · both: ${turnList(w.bothTurns)}${w.earpieceOnlyTurns.length ? " — the earpiece hears the words (text tone); the wrist cannot" : ""}${earlyAirtime(rep) ? ` · <span class="warn">airtime fires at ${earlyAirtime(rep)}s — the end of your FIRST turn, when you own 100 % of a few seconds of speech. Faithful to server/watch/vectors.py, which has no minimum-evidence floor; it needs one (e.g. ≥ 30 s of speech in the window) before it reaches a wrist.</span>` : ""}`,
    `${tag("📱", "📳", "⚡", "📱")} <b>Instant loudness haptic</b>: ${s.instantHaptics} fired${lead ? `, ${fmt.stat(lead)} ahead of the LLM tier` : ""} · ${s.instantBeforePolicy === null ? "no self turn crossed +6 dB over baseline (nothing for the instant tier to do)" : s.instantBeforePolicy ? '<span class="ok">always before the LLM-tier nudge</span>' : '<span class="bad">NOT before the LLM tier</span>'}`,
    `${tag("🎧", "🗣️", "🧠", "📱")} <b>LLM-tier nudge lag</b> (turn end → emission): ${fmt.stat(s.policyLagMs)} · policy-tier haptics ${s.policyHaptics}`,
    `${tag("📱", "📳", "🧠", "☁️")} <b>Call-mode equivalent (steamroll)</b>: ${s.callModeInterrupting} interrupting event(s) from the ground-truth timings${s.callModeInterrupting === 0 ? " — no self turn starts inside another turn and lasts ≥ 2 s" : ""}`,
    `${tag("📱", "👓", "⚡", "📱")} <b>Dark probes</b>: activation measured on ${s.activationProbed} fragment(s)${s.activationMaxProbability !== null ? `, max ${fmt.pct(s.activationMaxProbability)}` : ""}${
      s.activationFalseFlags
        ? ` · <span class="bad">would have nudged on ${s.activationFalseFlags} turn(s) nobody expects one on (#${s.activationFlaggedTurns.join(", #")})</span> — this is why ⚡ is still dark`
        : ' · <span class="ok">no false flag</span>'
    } · overlap probe on ${s.overlapProbed} long self turn(s)${s.overlapMaxMixedSeconds !== null ? `, max mixed ${s.overlapMaxMixedSeconds.toFixed(1)} s` : ""}`,
    positivesLine(rep),
    ...(missLine(rep) ? [missLine(rep) as string] : []),
    `${tag("📱", "📝", "⚡", "📱")} <b>Who is who</b>: ${s.attribution.correct}/${s.attribution.total} turns, self ${s.attribution.selfCorrect}/${s.attribution.selfTotal} · enrolled: ${escapeHtml(s.enrolled)} · loop turns ${s.loopTurns} (${s.coachedFragments} coached as you) for ${s.scriptTurns} scripted`,
    `${tag("🎧", "🗣️", "🧠", "📱")} <b>Never talks over you</b>: spoken over live speech ${s.spokenOverSpeech === 0 ? '<span class="ok">0</span>' : `<span class="bad">${s.spokenOverSpeech}</span>`}`,
  ];
  return `<ul class="feat">${items.map((i) => `<li>${i}</li>`).join("")}</ul>${fails.length ? `<div class="bad">Gate failures:<br>${fails.map(escapeHtml).join("<br>")}</div>` : ""}`;
}

/**
 * A line for every expected nudge that did NOT fire, with the reason visible
 * from the report's own numbers. A miss is the failure a user actually
 * notices — "I shouted and it said nothing" — so it gets its own line rather
 * than a number in a row of counts.
 */
function missLine(rep: NudgeReport): string | null {
  const missed = rep.turns.filter((t) => t.verdict === "miss");
  if (!missed.length) return null;
  const parts = missed.map((t) => {
    const why = t.isSelf && t.fragments.length && t.fragments.every((f) => !f.coachedAsSelf)
      ? "the loop did not recognise this as YOUR voice"
      : t.instantLevelMax === 0 && t.toneLevelMax === 0
        ? "neither loudness nor text tone crossed a rung"
        : "the policy held the level it was already at";
    return `#${t.index} ${escapeHtml(t.speaker)} “${escapeHtml(firstWords(t.text, 6))}” (${escapeHtml(t.expected ?? "?")}) — ${why}`;
  });
  return `${tag("📱", "📳", "🧠", "📱")} <b class="bad">Expected but never fired</b>: ${parts.join(" · ")}`;
}

/** 💚 The other half of the coach: what the user did WELL. Reads the same way
 *  whether nothing was earned (which is itself a finding) or several were. */
function positivesLine(rep: NudgeReport): string {
  const p = rep.scorecard.positives;
  const delivered = rep.positives.filter((pos) => pos.delivered);
  const detail = delivered.length
    ? delivered
        .map((pos) => `${pos.icon} <b>${escapeHtml(pos.name)}</b>@${pos.t.toFixed(1)}s <span style="opacity:.7">(${escapeHtml(pos.detail)})</span>`)
        .join(" · ")
    : "nothing earned — no recovery, no long turn let run, no repair that landed";
  const withheld = p.suppressed
    ? ` · <span class="warn">${p.suppressed} more detected but withheld</span> (one positive per ${POSITIVE_CAP_S / 60} min, so praise never becomes its own nag)`
    : "";
  const calm = `🧘 longest calm streak ${fmt.s(p.calmStreakS)}${p.calmBadge ? ' <span class="ok">badge</span>' : " (no badge — under 5 min)"}`;
  return `${tag("📱", "📳", "🧠", "📱")} <b>Positives — what you did well</b>: ${detail}${withheld} · ${calm}`;
}

export function renderSceneSection(rep: NudgeReport, gate: NudgeGate | null, open = true): string {
  const s = rep.scorecard;
  const fails = gate ? gateFailures(rep, gate) : [];
  const color = gate ? (fails.length ? "var(--bad)" : "var(--done)") : "var(--todo)";
  const earned = rep.positives.filter((pos) => pos.delivered).length;
  const right = `hit ${s.hits}/${s.expected} · fp ${s.falsePositives} · 🎧 ${s.earpiece.instantHaptics + s.earpiece.screenNudges} · ⌚ ${s.watch.buzzes} · 💚 ${earned}`;
  return `<details${open ? " open" : ""}><summary><span class="dot" style="background:${color}"></span><span class="t">${tag("📱", "📳", "🧠", "📱")} ${escapeHtml(rep.scene)} <span style="font-weight:400;color:var(--muted)">${rep.mode} · ${fmt.s(rep.durationSec)} · you = ${escapeHtml(rep.selfSpeaker ?? "–")}</span></span><span class="d">${right}</span></summary>
<div class="body">
${sceneTimelineSvg(rep)}
${scorecardHtml(rep, gate)}
<details><summary><span class="t">Turn by turn (${rep.turns.length} scripted, ${s.loopTurns} loop)</span><span class="d">tap</span></summary><div class="body">${turnTable(rep)}</div></details>
</div></details>`;
}

// --- AMI section --------------------------------------------------------------

function amiTimelineSvg(clip: AmiClip): string {
  const dur = Math.max(1, clip.duration_sec);
  const x = (t: number) => (Math.min(dur, Math.max(0, t)) / dur) * W;
  const labels = Object.keys(clip.diarization.cluster_seconds).sort();
  const H = TOP + labels.length * LANE_H + 18;
  const parts: string[] = [`<div class="scroll"><svg class="tl" viewBox="0 0 ${W} ${H}" role="img" aria-label="timeline of ${escapeHtml(clip.name)}">`];
  for (const t of ticks(dur)) {
    parts.push(`<line class="ax" x1="${x(t).toFixed(1)}" y1="${TOP - 4}" x2="${x(t).toFixed(1)}" y2="${H - 16}" stroke-width="1"/>`);
    parts.push(`<text class="tick" x="${(x(t) + 2).toFixed(1)}" y="${H - 5}">${t}s</text>`);
  }
  labels.forEach((lab, li) => {
    const y = TOP + li * LANE_H;
    parts.push(`<text class="lane" x="0" y="${y + 9}">${escapeHtml(lab)} · ${clip.diarization.cluster_seconds[lab]}s</text>`);
    for (const sg of clip.diarization.segments.filter((s) => s.label === lab)) {
      parts.push(`<rect class="other" x="${x(sg.start).toFixed(1)}" y="${y + 12}" width="${Math.max(1.5, x(sg.end) - x(sg.start)).toFixed(1)}" height="5" rx="1"/>`);
    }
    for (const [a, b] of clip.probe.spans[lab] ?? []) {
      parts.push(`<rect class="probe" x="${x(a).toFixed(1)}" y="${y + 19}" width="${Math.max(1.5, x(b) - x(a)).toFixed(1)}" height="4" rx="1"/>`);
    }
    for (const e of clip.probe.vectors[lab]?.interrupting ?? []) {
      parts.push(`<text class="mark call" x="${(x(e.t) - 6).toFixed(1)}" y="${y + 10}"><title>${escapeHtml(lab)} as self: interrupting L${e.level}, ${e.value}s of mutual speech at ${e.t.toFixed(1)}s</title>⌚</text>`);
    }
  });
  parts.push(`<text class="tick" x="0" y="10">grey = windows-engine segments (one voice at a time) · purple = overlap-probe spans (a window with two voices belongs to both) · ⌚ = steamroll vector fires with that lane as "you"</text>`);
  parts.push("</svg></div>");
  return parts.join("");
}

function amiClipSection(clip: AmiClip): string {
  const labels = Object.keys(clip.diarization.cluster_seconds).sort();
  const rows = labels
    .map((lab) => {
      const tl = clip.timeline[lab];
      const pr = clip.probe.vectors[lab];
      const air = tl?.airtime ?? [];
      const airMax = air.length ? air.reduce((a, b) => (b.value > a.value ? b : a)) : null;
      const ints = pr?.interrupting ?? [];
      const worst = ints.length ? ints.reduce((a, b) => (b.value > a.value ? b : a)) : null;
      return `<tr><td><b>${escapeHtml(lab)}</b></td><td>${clip.diarization.cluster_seconds[lab]}s</td><td>${tl?.interrupting.length ?? 0}</td><td>${airMax ? `${fmt.pct(airMax.value)} (L${airMax.level}) at ${airMax.t.toFixed(0)}s` : "below 60%"}</td><td>${ints.length}${worst ? ` · worst L${worst.level} ${worst.value}s at ${worst.t.toFixed(1)}s` : ""}</td><td class="txt">${ints
        .slice(0, 6)
        .map((e) => `${e.t.toFixed(1)}s L${e.level} ${e.value}s`)
        .join(" · ")}${ints.length > 6 ? " · …" : ""}</td></tr>`;
    })
    .join("");
  const d = clip.diarization;
  return `<details open><summary><span class="dot" style="background:var(--done)"></span><span class="t">${tag("⌚", "📳", "🧠", "☁️")} ${escapeHtml(clip.name)} <span style="font-weight:400;color:var(--muted)">${fmt.s(clip.duration_sec)} · ${d.num_speakers} voices (eigengap k=${d.k_eigengap})</span></span><span class="d">mixed ${clip.probe.mixed_seconds.toFixed(0)}s</span></summary>
<div class="body">
${amiTimelineSvg(clip)}
<div class="scroll"><table><tr><th>as "you"</th><th>talk</th><th title="interrupting_events over the engine's own non-overlapping segments">⌚ segments</th><th title="VectorEngine.push_diarization airtime (share of speech in the trailing 120 s)">airtime max</th><th title="interrupting_events over the overlap-probe spans">⌚ probe</th><th>where</th></tr>${rows}</table></div>
<div style="font-size:12.5px">${d.windows} windows at ${d.hop}s hop · probe: ${clip.probe.window_sec}s windows, ${clip.probe.mixed_windows} of ${clip.probe.windows} mixed (${clip.probe.mixed_seconds.toFixed(1)}s; two voices within ${clip.probe.margin} cosine, both ≥ ${clip.probe.min_score}) · <code>${escapeHtml(clip.path)}</code></div>
</div></details>`;
}

export function renderAmiSection(ami: AmiVectorsJson): string {
  return `<h2>Real overlapping speech — AMI 4-person meetings through the server's steamroll vector</h2>
<div style="font-size:13px;color:var(--muted);margin-bottom:6px">${tag("⌚", "📳", "🧠", "☁️")} <code>server/watch/vectors.py</code> <code>interrupting_events</code> + <code>VectorEngine.push_diarization</code> (airtime) over the <b>${escapeHtml(ami.engine)}</b> diarization (${escapeHtml(ami.model)}), each voice in turn as "you". A single-mic timeline has no overlap by construction, so the second pass re-scores the engine's own 1.5 s windows with the phone's overlap-probe rule (<code>overlapProbe.ts</code>) and lets a two-voice window belong to both speakers — that is where sustained talking-over becomes visible. ${ami.notes.map(escapeHtml).join(" ")} <span style="opacity:.7">generated ${escapeHtml(ami.generated_at)} · ${escapeHtml(ami.python)}</span></div>
${ami.clips.map(amiClipSection).join("\n")}`;
}

// --- page --------------------------------------------------------------------

export interface RenderOptions {
  title?: string;
  generatedAt?: string;
  gates?: Record<string, NudgeGate>;
  ami?: AmiVectorsJson | null;
  /** Shown in the "How to test" card. */
  rerun?: string[];
}

export function renderNudgeReportHtml(reports: NudgeReport[], opts: RenderOptions = {}): string {
  const generatedAt = opts.generatedAt ?? new Date().toISOString();
  const title = opts.title ?? "Nudge Verification";
  const gateOf = (rep: NudgeReport) => opts.gates?.[rep.scene] ?? null;
  const overall = reports.map((rep) => {
    const g = gateOf(rep);
    const fails = g ? gateFailures(rep, g) : [];
    return { rep, g, fails };
  });
  const allPass = overall.every((o) => o.fails.length === 0);
  const rerun = opts.rerun ?? [
    "cd apps/mobile && npx jest __tests__/replay.nudgeReport.test.ts --runInBand --forceExit",
    "MINDSHIFT_NUDGE_REPORT_DIR=/path/to/tmp  (where the HTML + JSON land; MINDSHIFT_NUDGE_REPORT=0 skips writing)",
    "tmp/nudge-report/ami_vectors.py  (server side: AMI clips → tmp/nudge-report/ami_vectors.json, picked up by the next jest run)",
  ];
  const summaryRows = overall
    .map(({ rep, g, fails }) => {
      const s = rep.scorecard;
      const color = g ? (fails.length ? "var(--bad)" : "var(--done)") : "var(--todo)";
      return `<li><span class="dot" style="background:${color};display:inline-block;vertical-align:middle;margin-right:6px"></span><b>${escapeHtml(rep.scene)}</b> — hit ${s.hits}/${s.expected}${g ? ` (pinned ≥ ${g.minHits})` : ""}, miss ${s.misses}, fp ${s.falsePositives} · 🎧 ${s.earpiece.instantHaptics} instant${s.instantLeadMs ? ` (${fmt.ms(s.instantLeadMs.median)} ahead)` : ""} + ${s.earpiece.screenNudges} screen (LLM lag ${s.policyLagMs ? fmt.ms(s.policyLagMs.median) : "–"}) + ${s.earpiece.spokenNudges} spoken · ⌚ ${s.watch.buzzes} buzz${s.watch.buzzes === 1 ? "" : "es"}${s.watch.watchOnlyTurns.length ? ` (watch-only: ${s.watch.watchOnlyTurns.map((i) => `#${i}`).join(" ")})` : ""}${s.watch.earpieceOnlyTurns.length ? ` (earpiece-only: ${s.watch.earpieceOnlyTurns.map((i) => `#${i}`).join(" ")})` : ""} · call-mode steamroll ${s.callModeInterrupting}</li>`;
    })
    .join("");
  return `<title>${escapeHtml(title)}</title>
<style>${STYLE}</style>
<h1>Nudge verification — from recorded files, no phone in the loop <span style="font-weight:400;color:var(--muted)">(${escapeHtml(generatedAt.slice(0, 16).replace("T", " "))})</span></h1>
${LEGEND_HTML}
<details open><summary><span class="dot" style="background:var(--prog)"></span><span class="t">How to test (≈4 min, no phone)</span><span class="d">do first</span></summary>
<div class="body">Every number below comes from replaying the recorded fixtures through the <b>real</b> on-device loop (Silero VAD + ECAPA voiceprints + the nudge policy) on a virtual clock — the same code the phone runs, deterministic, so the jest gate keeps it true. Re-run:<br>${rerun.map((c) => `<code>${escapeHtml(c)}</code>`).join("<br>")}<br><br>
<b>Gate</b> (per scene): expected nudges hit ≥ what <code>replay.scenes.test.ts</code> pins · false positives = 0 · the instant loudness haptic fires before the LLM-tier nudge on every loud self turn · the watch lane never buzzes on someone else's turn.<br>
<b>Two lanes per recording</b> — 🎧 <b>earpiece</b>: what the phone did (⚡ instant buzz, 🧠 on-screen nudge, 🗣️ the line spoken to you, each with its time); ⌚ <b>watch</b>: what the wrist would buzz for the same audio — the same policy run over loudness + interrupting + airtime only (no words on the wrist).<br>
<b>Read a scene</b>: timeline first (the 🎧 and ⌚ rows show the buzzes; ✓/✗ sit at the turn a nudge was expected), then the scorecard, then tap "Turn by turn" for the evidence.</div></details>
<details open><summary><span class="dot" style="background:${allPass ? "var(--done)" : "var(--bad)"}"></span><span class="t">Gate: ${allPass ? "all scenes pass" : "FAILING"}</span><span class="d">${reports.length} scenes</span></summary><div class="body"><ul style="margin:4px 0;padding-left:0;list-style:none">${summaryRows}</ul></div></details>
${overall.map(({ rep, g }, i) => renderSceneSection(rep, g, i === 0)).join("\n")}
${opts.ami ? renderAmiSection(opts.ami) : `<h2>Real overlapping speech (AMI)</h2><div class="body">No <code>tmp/nudge-report/ami_vectors.json</code> found — run <code>tmp/nudge-report/ami_vectors.py</code> first and re-run the jest test to include the server-side steamroll section.</div>`}
<div style="font-size:12px;color:var(--muted);margin-top:14px">apps/mobile/src/live/replay/nudgeReport.ts · fixtures server/tests/fixtures/audio/ · ${escapeHtml(generatedAt)}</div>`;
}
