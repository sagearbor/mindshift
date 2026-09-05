/**
 * Post-hoc conversation-dynamics metrics computed from a session's finished
 * turn list — response gaps and overlap — surfaced dark (Developer mode
 * only, sessionSummary.ts → SessionSummaryCard) with NO nudge wired to any
 * of these numbers. Workstream 2 of
 * docs/plans/2026-09-04-naturalturn-conversation-quality.md.
 *
 * Why gaps, not overlap, are the headline: CANDOR analysis (see below) found
 * brief overlap is a NORMAL feature of fluent conversation, uncorrelated
 * with harm, while long response gaps are the real anti-signal. An earlier
 * draft of this plan assumed sustained overlap = steamrolling; the mined
 * distributions reversed that (see the plan's "Overnight results" section),
 * so this module reports gap stats as the primary insight and overlap
 * purely as a neutral, informational count — never as a "you interrupted"
 * signal.
 *
 * Research constants (CANDOR corpus, 850 h of dyadic conversation; Reece et
 * al., Sci Adv 2023, doi:10.1126/sciadv.adf3197; NaturalTurn follow-up,
 * Cooney & Reece, Sci Reports 2025, doi:10.1038/s41598-025-24381-1; mined
 * locally in tmp/candor/analysis/ on 2026-09-05 — BY-NC source data, never
 * shipped, only these summary constants are):
 *  - Median between-turn response gap ≈ 0.33 s.
 *  - 34.7% of turn transitions carry brief overlap — the norm, not a
 *    steamroll signal.
 *  - Long response gaps correlate NEGATIVELY with partner enjoyment
 *    (r = -0.11).
 *  - Sustained overlap (>2 s) is rare (~13 episodes/hour) and itself
 *    correlates POSITIVELY with partner enjoyment (r = +0.09) — the
 *    opposite of the naive "overlap = steamrolling" hypothesis. No
 *    overlap-based nudge is planned until/unless real distributions from
 *    owner sessions justify one.
 *
 * Turns are second-based intervals carrying an explicit `isSelf`
 * (true = the phone's owner, false = a partner, null = unknown/unattributed
 * — excluded from the self/partner gap split since it can't be classified
 * either way). Backchannels ("yeah", "mhm" — the live/naturalTurn.ts `kind`
 * tag) are ignored everywhere in this module; only turns whose `kind` is
 * not `"backchannel"` ("primary", or untagged legacy lines) count.
 *
 * Single-mic sessions (one phone mic hears everyone) produce sequential,
 * non-overlapping turns by construction — `overlapSecondsTotal` and both
 * sustained-overlap counts are legitimately 0 there, not a measurement
 * failure. Call mode (WebRTC, one stream per participant) is where real
 * overlap shows up; only call-mode dynamics should be read as "did overlap
 * happen".
 *
 * Pure module: no React, no I/O, no imports from the rest of the app.
 */

/** Gaps longer than this count as "slow responses" (dev-mode diagnostic). */
export const SLOW_RESPONSE_THRESHOLD_S = 2;

/** Overlap longer than this is a "sustained overlap" episode (lower band). */
export const SUSTAINED_OVERLAP_SHORT_S = 1;

/** Overlap longer than this is a "sustained overlap" episode (upper band,
 *  matches the CANDOR ~13/hour rarity figure above). */
export const SUSTAINED_OVERLAP_LONG_S = 2;

/** CANDOR (850 h): median between-turn response gap, seconds. Shown next to
 *  the session's own median gap as a norm, not a target. */
export const CANDOR_MEDIAN_GAP_S = 0.33;

/** CANDOR (850 h): fraction of turn transitions that carry brief overlap. */
export const CANDOR_OVERLAP_TRANSITION_RATE = 0.347;

/** CANDOR (850 h): correlation of mean response gap with partner enjoyment
 *  (negative — longer gaps, less enjoyment). */
export const CANDOR_GAP_ENJOYMENT_CORR = -0.11;

/** CANDOR (850 h): sustained overlap (>2 s) episodes per hour of
 *  conversation — rare. */
export const CANDOR_SUSTAINED_OVERLAP_PER_HOUR = 13;

export type DynamicsTurnKind = "primary" | "backchannel";

export interface DynamicsTurn {
  speaker: string;
  /** true = the phone's owner, false = a partner, null = unknown. */
  isSelf: boolean | null;
  /** Utterance start, seconds (any consistent session-relative or epoch
   *  clock — only differences between turns are used). */
  startTime: number;
  /** Utterance end, seconds; must be >= startTime. */
  endTime: number;
  /** Absent/"primary" counts; "backchannel" turns are ignored entirely. */
  kind?: DynamicsTurnKind;
}

export interface ResponseGapStats {
  /** Number of qualifying partner→self (or self→partner) transitions. */
  count: number;
  /** Median gap, seconds; negative values are overlaps. Null when count is 0. */
  medianS: number | null;
  /** 90th-percentile gap, seconds. Null when count is 0. */
  p90S: number | null;
  /** How many of those gaps exceeded SLOW_RESPONSE_THRESHOLD_S. */
  slowCount: number;
}

/** One overlap between two different speakers' turns, sustained enough
 *  (> SUSTAINED_OVERLAP_SHORT_S) to report individually. */
export interface OverlapEpisode {
  speakerA: string;
  speakerB: string;
  /** Wall/session-clock bounds of the overlapping interval itself. */
  startTime: number;
  endTime: number;
  durationS: number;
}

export interface ConversationDynamics {
  /** Gaps before the self's turns, i.e. how quickly the user responded to
   *  a partner. */
  selfResponseGaps: ResponseGapStats;
  /** Gaps before partner turns, i.e. how quickly the partner responded to
   *  the user — symmetric to selfResponseGaps. */
  partnerResponseGaps: ResponseGapStats;
  /** Sum of every pairwise overlap duration between different-speaker
   *  turns, seconds. Pairwise, so a rare 3-way overlap double-counts the
   *  shared instant — documented, not corrected, since sessions are
   *  overwhelmingly 2-party. Always 0 on single-mic sessions. */
  overlapSecondsTotal: number;
  /** Overlaps longer than SUSTAINED_OVERLAP_SHORT_S, most recent last. */
  overlapEpisodes: OverlapEpisode[];
  /** Count of overlapEpisodes (i.e. > SUSTAINED_OVERLAP_SHORT_S). */
  sustainedOverlapCountOver1s: number;
  /** Count of overlapEpisodes whose duration also exceeds
   *  SUSTAINED_OVERLAP_LONG_S. */
  sustainedOverlapCountOver2s: number;
}

/** Linear-interpolated percentile (numpy's default), 0-100. Null on empty input. */
function percentile(sortedAsc: number[], p: number): number | null {
  if (sortedAsc.length === 0) return null;
  if (sortedAsc.length === 1) return sortedAsc[0];
  const idx = (p / 100) * (sortedAsc.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedAsc[lo];
  const frac = idx - lo;
  return sortedAsc[lo] + (sortedAsc[hi] - sortedAsc[lo]) * frac;
}

function gapStats(gaps: number[]): ResponseGapStats {
  const sorted = [...gaps].sort((a, b) => a - b);
  return {
    count: gaps.length,
    medianS: percentile(sorted, 50),
    p90S: percentile(sorted, 90),
    slowCount: gaps.filter((g) => g > SLOW_RESPONSE_THRESHOLD_S).length,
  };
}

/**
 * Compute the session's response-gap and overlap metrics. Input order does
 * not need to be chronological — turns are sorted by startTime here.
 */
export function computeConversationDynamics(turns: DynamicsTurn[]): ConversationDynamics {
  const primary = turns
    .filter((t) => t.kind !== "backchannel")
    .slice()
    .sort((a, b) => a.startTime - b.startTime);

  // (a) + (b): response gaps at self<->partner transitions only. Two
  // consecutive primary turns from the "same side" (self-self, or two
  // different partners back to back) aren't a response and are skipped, as
  // is any transition touching an unattributed (isSelf === null) turn.
  const selfGaps: number[] = [];
  const partnerGaps: number[] = [];
  for (let i = 1; i < primary.length; i++) {
    const prev = primary[i - 1];
    const curr = primary[i];
    if (prev.isSelf === null || curr.isSelf === null) continue;
    if (prev.isSelf === curr.isSelf) continue;
    const gap = curr.startTime - prev.endTime; // negative = overlap
    if (curr.isSelf) selfGaps.push(gap);
    else partnerGaps.push(gap);
  }

  // (c): overlap from ANY two different-speaker turns whose intervals
  // intersect — speaker label, not isSelf, so this also sees multi-partner
  // overlap (e.g. two partners talking over each other) if it's ever
  // present. O(n^2); session turn counts are small enough that this never
  // matters in practice.
  let overlapSecondsTotal = 0;
  const overlapEpisodes: OverlapEpisode[] = [];
  let sustainedOverlapCountOver1s = 0;
  let sustainedOverlapCountOver2s = 0;
  for (let i = 0; i < primary.length; i++) {
    for (let j = i + 1; j < primary.length; j++) {
      const a = primary[i];
      const b = primary[j];
      if (a.speaker === b.speaker) continue;
      const start = Math.max(a.startTime, b.startTime);
      const end = Math.min(a.endTime, b.endTime);
      const duration = end - start;
      if (duration <= 0) continue;
      overlapSecondsTotal += duration;
      if (duration > SUSTAINED_OVERLAP_SHORT_S) {
        overlapEpisodes.push({
          speakerA: a.speaker,
          speakerB: b.speaker,
          startTime: start,
          endTime: end,
          durationS: duration,
        });
        sustainedOverlapCountOver1s += 1;
        if (duration > SUSTAINED_OVERLAP_LONG_S) sustainedOverlapCountOver2s += 1;
      }
    }
  }

  return {
    selfResponseGaps: gapStats(selfGaps),
    partnerResponseGaps: gapStats(partnerGaps),
    overlapSecondsTotal,
    overlapEpisodes,
    sustainedOverlapCountOver1s,
    sustainedOverlapCountOver2s,
  };
}
