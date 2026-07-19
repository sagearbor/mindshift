/**
 * Pure, React-free helpers for the ReplayScreen "re-analyze" delta summary — the
 * honest "here's what changed" read shown after re-running a stored recording
 * through the latest engine. Kept out of the component (like glanceSummary.ts /
 * dayTimeline.ts) so the before/after diff is unit-testable directly.
 *
 * Honesty rules enforced here:
 *   - Nothing is fabricated: every number is a straight before/after from the
 *     two analyses. A metric absent in EITHER run is simply not compared.
 *   - `changed` is true only when a comparable number actually moved — so the UI
 *     can say "no change" plainly rather than implying the re-run did something
 *     it didn't.
 */
import type { AnalyzeResult } from "../api/client";
import { speakerLabel, type SpeakerLabels } from "../utils/speakerLabels";

/** One speaker's report-card score change across the re-analysis. */
export interface ScoreDelta {
  id: string;
  label: string;
  before: number;
  after: number;
  delta: number; // after - before (positive = improved conduct grade)
}

export interface ReanalyzeSummary {
  /** Per-speaker report-card score changes, for speakers scored in BOTH runs. */
  scoreDeltas: ScoreDelta[];
  /** Peak per-turn heat before/after (always available when there are turns). */
  peakBefore: number | null;
  peakAfter: number | null;
  peakDelta: number | null; // after - before, or null when either is unknown
  /** True when any comparable number actually moved. */
  changed: boolean;
}

/** Highest per-turn heat in an analysis, or null when there are no turns. */
function peakHeat(a: AnalyzeResult | null): number | null {
  if (!a || a.per_turn.length === 0) return null;
  return a.per_turn.reduce((m, t) => Math.max(m, t.heat), 0);
}

/**
 * Diff two analyses of the same recording (the one shown before re-analyzing vs
 * the fresh one) into an honest change summary. Only compares numbers present in
 * both runs; never invents a delta.
 */
export function summarizeReanalyze(
  before: AnalyzeResult | null,
  after: AnalyzeResult | null,
  labels?: SpeakerLabels,
): ReanalyzeSummary {
  const scoreDeltas: ScoreDelta[] = [];
  const beforeCards = before?.report_cards ?? {};
  const afterCards = after?.report_cards ?? {};
  for (const [id, card] of Object.entries(afterCards)) {
    const prev = beforeCards[id];
    // Only speakers scored in BOTH runs are comparable.
    if (!prev) continue;
    scoreDeltas.push({
      id,
      label: speakerLabel(id, labels),
      before: prev.score,
      after: card.score,
      delta: card.score - prev.score,
    });
  }

  const peakBefore = peakHeat(before);
  const peakAfter = peakHeat(after);
  const peakDelta =
    peakBefore != null && peakAfter != null ? peakAfter - peakBefore : null;

  const changed =
    scoreDeltas.some((d) => d.delta !== 0) ||
    (peakDelta != null && peakDelta !== 0);

  return { scoreDeltas, peakBefore, peakAfter, peakDelta, changed };
}
