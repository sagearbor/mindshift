/**
 * Pure, React-free helpers for the DynamicsScreen "glanceable summary" — the
 * headline act that lets a fresh-eyes user read the result at a glance without
 * decoding the time-axis chart. Kept out of the component (like dayTimeline.ts)
 * so the bar math and verdict derivation are unit-testable directly.
 *
 * Honesty rules enforced here:
 *   - Nothing is fabricated: every number comes straight from the analysis
 *     (per_speaker / per_turn). Absent data yields an omitted row or a null
 *     verdict, never a made-up value.
 *   - The verdict is DERIVED deterministically from the per-turn heats (and the
 *     server's own is_spike flags), so the same conversation always reads the
 *     same way. No drama that the data doesn't support.
 */
import type { AnalyzePerSpeaker, AnalyzePerTurn } from "../api/client";
import { getSpeakerColor } from "../utils/speakerColors";
import { heatColor } from "./dayTimeline";
import { speakerLabel, type SpeakerLabels } from "../utils/speakerLabels";

/** Heat bands on the house 0–100 scale, matching the green→amber→red ramp:
 *  calm (green third), strained (amber middle), rough (red top). */
export type HeatBand = "calm" | "strained" | "rough";

export function heatBand(heat: number): HeatBand {
  if (heat <= 33) return "calm";
  if (heat <= 66) return "strained";
  return "rough";
}

/** One speaker's row in the glance summary: their display label, line color, and
 *  the three at-a-glance measures (average heat, talk share, and the tally of
 *  four-horsemen markers vs repair attempts). All values are straight from the
 *  analysis; `heatBarColor` is the ramp color for the average-heat bar. */
export interface SpeakerBar {
  id: string;
  label: string;
  color: string;
  avgHeat: number; // 0–100
  talkShare: number; // 0–1
  horsemenTotal: number; // criticism+contempt+defensiveness+stonewalling
  repairAttempts: number;
  heatBarColor: string;
}

/** Total four-horsemen markers for a speaker (the sum the summary tallies against
 *  repair attempts). */
export function horsemenTotal(stats: AnalyzePerSpeaker): number {
  const h = stats.horsemen;
  return h.criticism + h.contempt + h.defensiveness + h.stonewalling;
}

/**
 * Build the per-speaker summary rows from the analysis's per_speaker map, in the
 * map's own key order (which the server emits in first-appearance order). Each
 * row carries the speaker's display label and the heat-ramp color for its bar.
 */
export function buildSpeakerBars(
  perSpeaker: Record<string, AnalyzePerSpeaker>,
  labels?: SpeakerLabels,
): SpeakerBar[] {
  return Object.entries(perSpeaker).map(([id, stats]) => ({
    id,
    label: speakerLabel(id, labels),
    color: getSpeakerColor(id),
    avgHeat: stats.avg_heat,
    talkShare: stats.talk_share,
    horsemenTotal: horsemenTotal(stats),
    repairAttempts: stats.repair_attempts,
    heatBarColor: heatColor(stats.avg_heat),
  }));
}

/**
 * A bar fill percentage (0–100) for `value` against `max`, clamped so a value at
 * or above the max fills the bar and a non-positive max yields an empty bar. Used
 * for both the heat bars (max 100) and the talk-share bars (max 1).
 */
export function barPct(value: number, max: number): number {
  if (!(max > 0) || !Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, (value / max) * 100));
}

export type VerdictTone = "calm" | "mixed" | "heated";

/** The one-line glance verdict: a short, warm sentence plus a tone that drives
 *  the chip's color. Derived, never invented. */
export interface Verdict {
  text: string;
  tone: VerdictTone;
}

/** m:ss for a moment label inside the verdict ("at 2:41"). Local to keep this
 *  module free of the chart component. */
function formatMoment(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${rem.toString().padStart(2, "0")}`;
}

/** Mean of a numeric slice, or 0 for an empty slice. */
function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

/**
 * Derive the glance verdict from the per-turn heats. Deterministic and honest:
 *
 *   - The headline reflects the PEAK band (how hot it ever got) and how many
 *     turns the server flagged as spikes — one heated moment vs a few.
 *   - A significant trend (first third vs last third of the conversation) is
 *     appended when present ("heat rose toward the end" / "cooled by the end").
 *   - When timing is available, a single heated moment is anchored to its clock
 *     position ("at 2:41") — the exact glance cue the owner asked for.
 *
 * Returns null for an empty conversation (nothing to summarize — the caller then
 * omits the chip rather than showing a hollow verdict).
 */
export function deriveVerdict(
  perTurn: Pick<AnalyzePerTurn, "index" | "heat" | "is_spike">[],
  timing?: { start_time: number; end_time: number }[] | null,
): Verdict | null {
  if (perTurn.length === 0) return null;

  const heats = perTurn.map((t) => t.heat);
  const peak = Math.max(...heats);
  const peakBand = heatBand(peak);
  const spikeTurns = perTurn.filter((t) => t.is_spike);

  // Trend: compare the average of the first third to the last third. Guard tiny
  // conversations (thirds overlap) by requiring at least 3 turns.
  const n = perTurn.length;
  let trend: "rising" | "falling" | null = null;
  if (n >= 3) {
    const k = Math.max(1, Math.floor(n / 3));
    const firstAvg = mean(heats.slice(0, k));
    const lastAvg = mean(heats.slice(n - k));
    const delta = lastAvg - firstAvg;
    if (delta >= 15) trend = "rising";
    else if (delta <= -15) trend = "falling";
  }

  let text: string;
  let tone: VerdictTone;

  if (peakBand === "rough") {
    if (spikeTurns.length >= 2) {
      text = `A few heated moments (${spikeTurns.length})`;
      tone = "heated";
    } else {
      // One (or zero flagged, but a rough peak) heated moment — anchor it to the
      // clock when we have honest timing for that turn.
      const hot = spikeTurns[0] ?? perTurn[heats.indexOf(peak)];
      const at = timing?.[hot.index];
      const stamp =
        at && Number.isFinite(at.start_time)
          ? ` at ${formatMoment(at.start_time)}`
          : "";
      text = `Mostly calm — one heated moment${stamp}`;
      tone = "mixed";
    }
  } else if (peakBand === "strained") {
    text = "Steady, with a little tension";
    tone = "mixed";
  } else {
    text = "A calm conversation";
    tone = "calm";
  }

  if (trend === "rising" && tone !== "heated") {
    text += " — heat rose toward the end";
  } else if (trend === "falling") {
    text += " — and it cooled by the end";
  }

  return { text, tone };
}

/** Chip background/text colors for a verdict tone, from the house heat ramp
 *  (calm green, mixed amber, heated red). */
export function verdictColors(tone: VerdictTone): { bg: string; fg: string } {
  switch (tone) {
    case "heated":
      return { bg: "#FEECEC", fg: "#B42318" };
    case "mixed":
      return { bg: "#FEF4E6", fg: "#B25E09" };
    case "calm":
    default:
      return { bg: "#E7F6EE", fg: "#1B7A4B" };
  }
}
