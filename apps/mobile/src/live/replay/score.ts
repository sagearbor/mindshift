/**
 * Scoring a replay against its script. Pure functions over the loop's
 * output (`LocalTurn[]`, spoken lines, policy log, VAD verdicts) and the
 * normalized `ReplayScript`, so every rule is unit-testable without audio.
 *
 * Attribution follows server/tests/test_diarize_scenes.py: exact per-turn
 * accuracy, where an ENROLLED verdict must name the right person and the
 * loop's unknown clusters ("Speaker A/B/…" with no personId) are mapped to
 * the un-enrolled script speakers under the best bijection.
 */
import type { LocalTurn } from "../fastLoop";
import type { EnrollmentRecord } from "./enroll";
import type { PolicyCall, SpokenLine, TrackedVad } from "./fakes";
import type { ReplayScript, ScriptTurn } from "./meta";

// ---------------------------------------------------------------------------
// Turn matching
// ---------------------------------------------------------------------------

export function overlap(a: { start: number; end: number }, b: { start: number; end: number }): number {
  return Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start));
}

const asSpan = (t: LocalTurn) => ({ start: t.startTime, end: t.endTime });

/** For each script turn, the loop turn overlapping it most (null when none). */
export function matchScriptTurns(script: ReplayScript, turns: LocalTurn[]): (number | null)[] {
  return script.turns.map((st) => {
    let best: number | null = null;
    let bestOv = 0;
    turns.forEach((lt, i) => {
      const ov = overlap(st, asSpan(lt));
      if (ov > bestOv) {
        bestOv = ov;
        best = i;
      }
    });
    return best;
  });
}

/** For each loop turn, the script turn overlapping it most (null when none). */
export function matchLoopTurns(script: ReplayScript, turns: LocalTurn[]): (number | null)[] {
  return turns.map((lt) => {
    let best: number | null = null;
    let bestOv = 0;
    for (const st of script.turns) {
      const ov = overlap(st, asSpan(lt));
      if (ov > bestOv) {
        bestOv = ov;
        best = st.index;
      }
    }
    return best;
  });
}

// ---------------------------------------------------------------------------
// Speaker attribution
// ---------------------------------------------------------------------------

/** The label attribution compares: an enrolled name, or "?<cluster>" for
 *  the loop's own unknown clusters (which reuse the "Speaker A" wording). */
export function predictedLabel(t: LocalTurn): string {
  // An identified turn predicts the PERSON. For a per-turn absolute match
  // `speaker` already is the display name; for a cluster identified by its
  // centroid (absolute or contrast) `speaker` stays the raw "Speaker X" wire
  // key and the person rides on `displayName`.
  return t.personId ? (t.displayName ?? t.speaker) : `?${t.speaker}`;
}

export interface AttributionTurn {
  index: number;
  truth: string;
  predicted: string | null;
  mapped: string | null;
  ok: boolean;
  loopTurn: number | null;
}

export interface AttributionScore {
  correct: number;
  total: number;
  accuracy: number;
  selfCorrect: number;
  selfTotal: number;
  /** Enrolled-name turns (exact), unknown turns (best-permutation). */
  enrolledCorrect: number;
  enrolledTotal: number;
  unknownClusters: number;
  /** Enrolled names actually matched + unknown clusters seen. */
  speakersDetected: number;
  mapping: Record<string, string>;
  perTurn: AttributionTurn[];
}

function permutations<T>(items: T[], width: number): T[][] {
  if (width === 0) return [[]];
  const out: T[][] = [];
  items.forEach((x, i) => {
    const rest = [...items.slice(0, i), ...items.slice(i + 1)];
    for (const p of permutations(rest, width - 1)) out.push([x, ...p]);
  });
  return out;
}

export function scoreAttribution(
  script: ReplayScript,
  turns: LocalTurn[],
  enrolled: EnrollmentRecord[],
): AttributionScore {
  const match = matchScriptTurns(script, turns);
  const enrolledNames = new Set(enrolled.map((e) => e.displayName));
  const truthUnenrolled = script.speakers.filter((s) => !enrolledNames.has(s));
  const preds = script.turns.map((st) => {
    const li = match[st.index];
    return li === null ? null : predictedLabel(turns[li]);
  });
  const unknownLabels = [...new Set(preds.filter((p): p is string => p !== null && p.startsWith("?")))].sort();

  // Best bijection unknown-cluster -> un-enrolled truth speaker.
  const width = Math.min(unknownLabels.length, truthUnenrolled.length);
  let bestMap: Record<string, string> = {};
  let bestCorrect = -1;
  for (const perm of permutations(truthUnenrolled, width)) {
    const map: Record<string, string> = {};
    unknownLabels.slice(0, width).forEach((u, i) => (map[u] = perm[i]));
    let correct = 0;
    script.turns.forEach((st, i) => {
      const p = preds[i];
      if (p === null) return;
      const mapped = p.startsWith("?") ? map[p] : p;
      if (mapped === st.speaker) correct += 1;
    });
    if (correct > bestCorrect) {
      bestCorrect = correct;
      bestMap = map;
    }
  }
  if (bestCorrect < 0) bestCorrect = 0;

  const perTurn: AttributionTurn[] = script.turns.map((st, i) => {
    const p = preds[i];
    const mapped = p === null ? null : p.startsWith("?") ? (bestMap[p] ?? null) : p;
    return { index: st.index, truth: st.speaker, predicted: p, mapped, ok: mapped === st.speaker, loopTurn: match[i] };
  });
  const self = script.selfSpeaker;
  const selfTurns = perTurn.filter((t) => t.truth === self);
  const enrolledTurns = perTurn.filter((t) => enrolledNames.has(t.truth));
  const matchedNames = new Set(turns.filter((t) => t.personId).map((t) => t.personId as string));
  const clusters = new Set(turns.filter((t) => !t.personId && t.speaker !== "Unknown").map((t) => t.speaker));
  return {
    correct: bestCorrect,
    total: script.turns.length,
    accuracy: script.turns.length ? bestCorrect / script.turns.length : 0,
    selfCorrect: selfTurns.filter((t) => t.ok).length,
    selfTotal: selfTurns.length,
    enrolledCorrect: enrolledTurns.filter((t) => t.ok).length,
    enrolledTotal: enrolledTurns.length,
    unknownClusters: clusters.size,
    speakersDetected: matchedNames.size + clusters.size,
    mapping: bestMap,
    perTurn,
  };
}

// ---------------------------------------------------------------------------
// Turn boundaries
// ---------------------------------------------------------------------------

export interface BoundaryScore {
  /** ms, over script turns that matched a loop turn. */
  startErrMs: number[];
  endErrMs: number[];
  medianStartMs: number;
  medianEndMs: number;
  p90StartMs: number;
  p90EndMs: number;
  maxMs: number;
  /** Script turns covered by >= 2 loop turns (each overlapping >= minOverlap). */
  split: number;
  /** Loop turns covering >= 2 script turns. */
  merged: number;
  /** Script turns no loop turn overlapped at all. */
  unmatched: number;
  /** Loop turns overlapping no script turn (VAD fired on non-speech). */
  extra: number;
}

export function median(xs: number[]): number {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export function percentile(xs: number[], p: number): number {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.max(0, Math.ceil((p / 100) * s.length) - 1));
  return s[idx];
}

export function scoreBoundaries(script: ReplayScript, turns: LocalTurn[], minOverlapSec = 0.5): BoundaryScore {
  const match = matchScriptTurns(script, turns);
  const startErr: number[] = [];
  const endErr: number[] = [];
  let split = 0;
  let unmatched = 0;
  script.turns.forEach((st, i) => {
    const li = match[i];
    if (li === null) {
      unmatched += 1;
      return;
    }
    const lt = turns[li];
    startErr.push(Math.abs(lt.startTime - st.start) * 1000);
    endErr.push(Math.abs(lt.endTime - st.end) * 1000);
    const covering = turns.filter((t) => overlap(st, asSpan(t)) >= minOverlapSec).length;
    if (covering >= 2) split += 1;
  });
  let merged = 0;
  let extra = 0;
  for (const lt of turns) {
    const covered = script.turns.filter((st) => overlap(st, asSpan(lt)) >= minOverlapSec).length;
    if (covered >= 2) merged += 1;
    if (!script.turns.some((st) => overlap(st, asSpan(lt)) > 0)) extra += 1;
  }
  const all = [...startErr, ...endErr];
  return {
    startErrMs: startErr,
    endErrMs: endErr,
    medianStartMs: median(startErr),
    medianEndMs: median(endErr),
    p90StartMs: percentile(startErr, 90),
    p90EndMs: percentile(endErr, 90),
    maxMs: all.length ? Math.max(...all) : NaN,
    split,
    merged,
    unmatched,
    extra,
  };
}

// ---------------------------------------------------------------------------
// Nudges
// ---------------------------------------------------------------------------

export interface NudgeOutcome {
  scriptTurn: number;
  expected: "mild" | "strong" | null;
  /** Escalation level the policy computed on that turn's events (0 when the
   *  loop did not treat it as the coached user's turn). */
  level: number;
  /** Levels of the escalation events actually emitted (hysteresis; a
   *  cooldown decay is not counted). */
  emitted: number[];
  verdict: "hit" | "miss" | "fp" | "quiet";
}

export interface NudgeScore {
  hits: number;
  misses: number;
  falsePositives: number;
  /** Expected nudges the policy reached but only after hysteresis
   *  swallowed the event (level already that high) — hits, flagged. */
  hitsSilent: number;
  hapticsFired: { t: number; level: number; scriptTurn: number | null }[];
  perTurn: NudgeOutcome[];
}

/** mild = level >= 1, strong = level >= 2 (the policy's 0..3 ladder). */
export function levelSatisfies(level: number, expected: "mild" | "strong"): boolean {
  return expected === "mild" ? level >= 1 : level >= 2;
}

export function scoreNudges(script: ReplayScript, turns: LocalTurn[], policyLog: PolicyCall[]): NudgeScore {
  const loopToScript = matchLoopTurns(script, turns);
  // Policy calls are keyed by the loop turn's end time.
  const callFor = (lt: LocalTurn) => policyLog.find((c) => Math.abs(c.t - lt.endTime) < 1e-6) ?? null;
  const levelByScript = new Map<number, { level: number; emitted: number[] }>();
  turns.forEach((lt, i) => {
    const si = loopToScript[i];
    if (si === null) return;
    const call = callFor(lt);
    if (!call) return;
    const cur = levelByScript.get(si) ?? { level: 0, emitted: [] };
    cur.level = Math.max(cur.level, call.rawLevel);
    cur.emitted.push(...call.emitted.filter((e) => e.level > 0 && e.vectors.length > 0).map((e) => e.level));
    levelByScript.set(si, cur);
  });
  const expectedByTurn = new Map(script.expectedNudges.map((n) => [n.afterTurnIndex, n.level]));
  let hits = 0;
  let misses = 0;
  let fps = 0;
  let hitsSilent = 0;
  const perTurn: NudgeOutcome[] = script.turns.map((st) => {
    const got = levelByScript.get(st.index) ?? { level: 0, emitted: [] };
    const expected = expectedByTurn.get(st.index) ?? null;
    let verdict: NudgeOutcome["verdict"];
    if (expected) {
      if (levelSatisfies(got.level, expected)) {
        verdict = "hit";
        hits += 1;
        if (got.emitted.length === 0) hitsSilent += 1;
      } else {
        verdict = "miss";
        misses += 1;
      }
    } else if (got.level >= 1) {
      verdict = "fp";
      fps += 1;
    } else {
      verdict = "quiet";
    }
    return { scriptTurn: st.index, expected, level: got.level, emitted: got.emitted, verdict };
  });
  // What buzzes: escalation events (the loop never buzzes on a decay).
  const hapticsFired = policyLog.flatMap((c) =>
    c.emitted
      .filter((e) => e.level > 0 && e.vectors.length > 0)
      .map((e) => {
        const li = turns.findIndex((lt) => Math.abs(lt.endTime - c.t) < 1e-6);
        return { t: c.t, level: e.level, scriptTurn: li >= 0 ? loopToScript[li] : null };
      }),
  );
  return { hits, misses, falsePositives: fps, hitsSilent, hapticsFired, perTurn };
}

// ---------------------------------------------------------------------------
// Speaking
// ---------------------------------------------------------------------------

export interface SpeakScore {
  spoken: number;
  /** Spoken while the loop's latest processed VAD verdict was "speech" —
   *  the invariant (the loop can only act on what it has seen). */
  overVadSpeech: number;
  /** Spoken while the VAD timeline (including chunks of the same 100 ms
   *  frame not yet processed) shows speech at that instant. */
  overVadTimeline: number;
  /** Spoken inside a scripted turn's span — i.e. into a pause WITHIN
   *  someone's turn rather than after it (informational). */
  overScriptedSpeech: number;
  held: number;
  /** Suggestions produced but never spoken (held past speakHoldMaxMs, or
   *  therapist mode). */
  dropped: number;
  lines: (SpokenLine & { overVad: boolean; overTimeline: boolean; overScript: boolean })[];
}

export function scoreSpeaking(
  script: ReplayScript,
  turns: LocalTurn[],
  spoken: SpokenLine[],
  vad: Pick<TrackedVad, "speechAt">,
): SpeakScore {
  const lines = spoken.map((l) => ({
    ...l,
    overVad: l.vadSpeechKnown === true,
    overTimeline: vad.speechAt(l.atSec) === true,
    overScript: script.turns.some((st) => l.atSec > st.start && l.atSec < st.end),
  }));
  const withSuggestion = turns.filter((t) => t.suggestion);
  return {
    spoken: spoken.length,
    overVadSpeech: lines.filter((l) => l.overVad).length,
    overVadTimeline: lines.filter((l) => l.overTimeline).length,
    overScriptedSpeech: lines.filter((l) => l.overScript).length,
    held: turns.filter((t) => t.latency.held).length,
    dropped: withSuggestion.filter((t) => !t.spoken).length,
    lines,
  };
}

// ---------------------------------------------------------------------------
// Latency
// ---------------------------------------------------------------------------

export interface LatencyScore {
  turns: number;
  spokenTurns: number;
  /** Segment end -> speak() (virtual ms) over spoken turns. */
  toSpeakMedianMs: number;
  toSpeakMaxMs: number;
  sttWaitMedianMs: number;
  llmMedianMs: number;
  speakerMedianMs: number;
  /** Segment close lag: how long after the scripted end the loop finalized
   *  the matching turn (VAD hysteresis + merge gap), median ms. */
  segmentCloseMedianMs: number;
  providers: Record<string, number>;
  textless: number;
  interimOnly: number;
}

export function scoreLatency(script: ReplayScript, turns: LocalTurn[]): LatencyScore {
  const spoken = turns.filter((t) => t.latency.toSpeakMs !== null);
  const providers: Record<string, number> = {};
  for (const t of turns) providers[t.provider] = (providers[t.provider] ?? 0) + 1;
  const loopToScript = matchLoopTurns(script, turns);
  const closeLag: number[] = [];
  turns.forEach((lt, i) => {
    const si = loopToScript[i];
    if (si === null) return;
    closeLag.push(lt.latency.segmentEndMs - script.turns[si].end * 1000);
  });
  return {
    turns: turns.length,
    spokenTurns: spoken.length,
    toSpeakMedianMs: median(spoken.map((t) => t.latency.toSpeakMs as number)),
    toSpeakMaxMs: spoken.length ? Math.max(...spoken.map((t) => t.latency.toSpeakMs as number)) : NaN,
    sttWaitMedianMs: median(turns.map((t) => t.latency.sttWaitMs)),
    llmMedianMs: median(turns.filter((t) => t.latency.llmMs > 0).map((t) => t.latency.llmMs)),
    speakerMedianMs: median(turns.map((t) => t.latency.speakerMs)),
    segmentCloseMedianMs: median(closeLag),
    providers,
    textless: turns.filter((t) => !t.text).length,
    interimOnly: turns.filter((t) => t.text && !t.transcriptFinal).length,
  };
}
