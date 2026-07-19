import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
} from "react-native";
import Svg, { Polyline, Circle, Line, Rect } from "react-native-svg";
import type { AnalyzePerTurn, SimulatedTurn, Voice } from "../api/client";
import { getSpeakerColor } from "../utils/speakerColors";
import { speakerLabel, type SpeakerLabels } from "../utils/speakerLabels";

// The baseline prosody label per dimension — a turn at baseline on a dimension
// isn't noteworthy, so we don't render a chip for it. This keeps the inspector
// to "up to three" chips that actually say something (e.g. loud + fast), rather
// than three always-on chips two of which read "normal".
const VOICE_BASELINE: Record<keyof Voice, string> = {
  energy_label: "normal",
  pitch_label: "mid",
  rate_label: "normal",
};

/** Non-baseline prosody labels for a turn, as {kind,label} chips in a stable
 *  order (energy, pitch, rate). Empty when voice is absent or all-baseline. */
export function voiceChipsFor(voice: Voice | null | undefined): {
  kind: "energy" | "pitch" | "rate";
  label: string;
}[] {
  if (!voice) return [];
  const chips: { kind: "energy" | "pitch" | "rate"; label: string }[] = [];
  if (voice.energy_label !== VOICE_BASELINE.energy_label)
    chips.push({ kind: "energy", label: voice.energy_label });
  // pitch_label is null when the turn had too little voiced speech to measure
  // — no reading means no chip, never an empty one.
  if (voice.pitch_label !== null && voice.pitch_label !== VOICE_BASELINE.pitch_label)
    chips.push({ kind: "pitch", label: voice.pitch_label });
  if (voice.rate_label !== VOICE_BASELINE.rate_label)
    chips.push({ kind: "rate", label: voice.rate_label });
  return chips;
}

// House colors.
const AMBER = "#F59E0B"; // spikes / triggers
const INK = "#1F2937";
const MUTED = "#6B7280";
const PRIMARY = "#4A90D9";
const DANGER = "#DC2626";

// Simulated overlay: each speaker's own color, drawn dashed at reduced opacity
// so it reads as a hypothetical laid over the real (solid, full-opacity) lines.
const SIM_OPACITY = 0.55;
const SIM_DASH = "6,4";

const HEAT_MIN = 0;
const HEAT_MAX = 100;

// Minimum width (px) of a time-axis tap target, so a very short utterance's dash
// is still comfortably hittable on a phone.
const MIN_TAP_PX = 28;

export interface ChartPoint {
  index: number; // turn index across the WHOLE conversation
  heat: number; // 0–100
  isSpike: boolean;
  x: number; // pixel x
  y: number; // pixel y
}

export interface SpeakerLine {
  speaker: string;
  color: string;
  points: ChartPoint[]; // that speaker's turns only, in conversation order
}

export interface MapOptions {
  width: number;
  height: number;
  padding: number;
  /** Total turns in the conversation, used to normalize x. When omitted, it's
   *  derived from the max turn index in `perTurn`. */
  totalTurns?: number;
}

/**
 * Pure point-mapping: turns the backend's flat per_turn array into one polyline
 * per speaker, in pixel space. Exported so the geometry can be unit-tested
 * directly without rendering.
 *
 * Key rule (from the spec): x is the turn index across the ENTIRE conversation,
 * not each speaker's own running count — so a speaker's line "carries across
 * gaps", i.e. its points sit at their true conversational positions and we only
 * connect *consecutive* points belonging to that speaker. This is what makes
 * two speakers' lines visibly interleave and cross as the conversation heats up.
 */
export function mapTurnsToLines(
  perTurn: AnalyzePerTurn[],
  opts: MapOptions,
): SpeakerLine[] {
  const { width, height, padding } = opts;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;

  // Normalize x against the last turn index so the line spans the full width.
  const maxIndex =
    opts.totalTurns !== undefined
      ? opts.totalTurns - 1
      : perTurn.reduce((m, t) => Math.max(m, t.index), 0);

  const xFor = (index: number) =>
    padding + (maxIndex <= 0 ? chartWidth / 2 : (index / maxIndex) * chartWidth);
  const yFor = (heat: number) => {
    const clamped = Math.max(HEAT_MIN, Math.min(HEAT_MAX, heat));
    return (
      padding +
      (chartHeight -
        ((clamped - HEAT_MIN) / (HEAT_MAX - HEAT_MIN)) * chartHeight)
    );
  };

  // Group by speaker, preserving first-seen order for a stable legend/z-order.
  const order: string[] = [];
  const bySpeaker = new Map<string, ChartPoint[]>();
  for (const t of perTurn) {
    if (!bySpeaker.has(t.speaker)) {
      bySpeaker.set(t.speaker, []);
      order.push(t.speaker);
    }
    bySpeaker.get(t.speaker)!.push({
      index: t.index,
      heat: t.heat,
      isSpike: t.is_spike,
      x: xFor(t.index),
      y: yFor(t.heat),
    });
  }

  return order.map((speaker) => ({
    speaker,
    color: getSpeakerColor(speaker),
    points: bySpeaker.get(speaker)!,
  }));
}

/**
 * Pure geometry for the "what-if" simulated overlay. Mirrors mapTurnsToLines
 * exactly for x/y so a simulated point at conversation index `i` lands at the
 * SAME x as the real point at index `i` — the caller passes `totalTurns` (the
 * real conversation's length) so both share one x-scale, and the dashed overlay
 * (which only spans pivot_index → last turn) aligns perfectly with the solid
 * lines beneath it. Grouped per speaker so each gets a dashed line in its own
 * color. Simulated points carry no spike/marker data, so isSpike is always
 * false. Exported for direct unit testing of the alignment + grouping.
 */
export function mapSimulatedToLines(
  simulated: SimulatedTurn[],
  opts: MapOptions,
): SpeakerLine[] {
  const { width, height, padding } = opts;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;

  const maxIndex =
    opts.totalTurns !== undefined
      ? opts.totalTurns - 1
      : simulated.reduce((m, t) => Math.max(m, t.index), 0);

  const xFor = (index: number) =>
    padding + (maxIndex <= 0 ? chartWidth / 2 : (index / maxIndex) * chartWidth);
  const yFor = (heat: number) => {
    const clamped = Math.max(HEAT_MIN, Math.min(HEAT_MAX, heat));
    return (
      padding +
      (chartHeight -
        ((clamped - HEAT_MIN) / (HEAT_MAX - HEAT_MIN)) * chartHeight)
    );
  };

  const order: string[] = [];
  const bySpeaker = new Map<string, ChartPoint[]>();
  for (const t of simulated) {
    if (!bySpeaker.has(t.speaker)) {
      bySpeaker.set(t.speaker, []);
      order.push(t.speaker);
    }
    bySpeaker.get(t.speaker)!.push({
      index: t.index,
      heat: t.heat,
      isSpike: false,
      x: xFor(t.index),
      y: yFor(t.heat),
    });
  }

  return order.map((speaker) => ({
    speaker,
    color: getSpeakerColor(speaker),
    points: bySpeaker.get(speaker)!,
  }));
}

/** Per-turn timing, index-aligned with `perTurn`, used to place the replay
 *  playhead. Only the boundaries matter here. */
export interface TurnTiming {
  start_time: number;
  end_time: number;
}

export interface PlayheadOptions {
  width: number;
  padding: number;
  /** Total turns in the conversation — the same value passed to mapTurnsToLines
   *  so the playhead shares the real lines' x-scale exactly. */
  totalTurns: number;
}

/**
 * Pure: map a playback position (seconds) to the conversation turn index it
 * falls in. The turn whose [start_time, end_time) contains `seconds` wins;
 * failing that we fall back to the nearest EARLIER turn (so a position in the
 * silent gap between two turns stays anchored to the turn that just spoke).
 * Before the first turn → first turn; after the last → last turn. Returns null
 * only when there is no timing at all. Exported for direct unit testing.
 */
export function playheadIndexForTime(
  seconds: number,
  turnsTiming: TurnTiming[],
): number | null {
  if (turnsTiming.length === 0) return null;
  // Exact containment: [start, end).
  for (let i = 0; i < turnsTiming.length; i++) {
    const t = turnsTiming[i];
    if (seconds >= t.start_time && seconds < t.end_time) return i;
  }
  // Before the first turn starts → clamp to the first turn.
  if (seconds < turnsTiming[0].start_time) return 0;
  // Otherwise the nearest earlier turn: the last one that has already started.
  // This covers both between-turns gaps (→ previous turn) and after-last (→ last).
  let idx = 0;
  for (let i = 0; i < turnsTiming.length; i++) {
    if (turnsTiming[i].start_time <= seconds) idx = i;
    else break;
  }
  return idx;
}

/**
 * Pure: the pixel x of the replay playhead for a given playback position. Maps
 * `seconds` → turn index (playheadIndexForTime) → x using the SAME formula as
 * mapTurnsToLines' xFor (maxIndex = totalTurns - 1), so the playhead lands
 * exactly on the current turn's point. Returns null when there's no timing.
 * Exported for direct unit testing of the alignment.
 */
export function playheadXForTime(
  seconds: number,
  turnsTiming: TurnTiming[],
  opts: PlayheadOptions,
): number | null {
  const idx = playheadIndexForTime(seconds, turnsTiming);
  if (idx === null) return null;
  const { width, padding, totalTurns } = opts;
  const chartWidth = width - padding * 2;
  const maxIndex = totalTurns - 1;
  return (
    padding + (maxIndex <= 0 ? chartWidth / 2 : (idx / maxIndex) * chartWidth)
  );
}

// --- Time-axis geometry (the primary "dashes over real recording time" view) ---
//
// Instead of one evenly-spaced dot per turn, each turn is a HORIZONTAL DASH
// spanning its real [start_time, end_time] on an x-axis that IS the recording's
// clock. A speaker who talks two-thirds of the time visibly covers two-thirds of
// the axis; silence is simply empty. The playhead (mapped by the same seconds→x
// scale) therefore sits on the dash of whoever is actually speaking.

/** One turn drawn as a horizontal dash. x1..x2 is its real span in pixels. */
export interface DashSegment {
  index: number;
  heat: number;
  isSpike: boolean;
  x1: number; // px at start_time
  x2: number; // px at end_time (grown to a minimum so short turns stay visible)
  xMid: number;
  y: number;
}

export interface SpeakerDashes {
  speaker: string;
  color: string;
  dashes: DashSegment[];
}

export interface TimeMapOptions {
  width: number;
  height: number;
  padding: number;
  /** Total recording length in seconds — the x-axis span. Must be > 0. */
  duration: number;
  /** Floor on a dash's pixel width so a very short utterance is still visible
   *  and tappable; the dash is grown symmetrically around its center. */
  minDashPx?: number;
}

/**
 * True when timing is present, index-aligned with `count` turns, finite, non-
 * decreasing (end >= start), and spans a positive duration — the precondition
 * for the honest time axis. Anything else (missing timing on a pre-timestamp
 * recording, a pasted transcript) falls back to index spacing.
 */
export function timingIsUsable(
  timing: TurnTiming[] | undefined | null,
  count: number,
): boolean {
  if (!timing || timing.length === 0 || timing.length !== count) return false;
  let maxEnd = 0;
  for (const t of timing) {
    if (!Number.isFinite(t.start_time) || !Number.isFinite(t.end_time))
      return false;
    if (t.end_time < t.start_time) return false;
    if (t.end_time > maxEnd) maxEnd = t.end_time;
  }
  return maxEnd > 0;
}

/** Total x-axis duration: an explicit recording duration wins (it can exceed the
 *  last utterance's end — trailing silence is real), else the last end_time. */
export function durationForTiming(
  timing: TurnTiming[],
  explicit?: number | null,
): number {
  if (explicit != null && explicit > 0) return explicit;
  return timing.reduce((m, t) => Math.max(m, t.end_time), 0);
}

/**
 * Pure: map per-turn heat + real timing into one set of horizontal dashes per
 * speaker. x spans [start_time, end_time] in recording seconds; y = heat 0–100.
 * `timing` is index-aligned with `perTurn`. Exported for direct geometry tests.
 */
export function mapTurnsToDashes(
  perTurn: AnalyzePerTurn[],
  timing: TurnTiming[],
  opts: TimeMapOptions,
): SpeakerDashes[] {
  const { width, height, padding, duration } = opts;
  const minDashPx = opts.minDashPx ?? 3;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;

  const xFor = (sec: number) => {
    const frac = duration <= 0 ? 0 : Math.max(0, Math.min(1, sec / duration));
    return padding + frac * chartWidth;
  };
  const yFor = (heat: number) => {
    const clamped = Math.max(HEAT_MIN, Math.min(HEAT_MAX, heat));
    return (
      padding +
      (chartHeight - ((clamped - HEAT_MIN) / (HEAT_MAX - HEAT_MIN)) * chartHeight)
    );
  };

  const order: string[] = [];
  const bySpeaker = new Map<string, DashSegment[]>();
  perTurn.forEach((t, i) => {
    const tm = timing[i];
    let x1 = xFor(tm.start_time);
    let x2 = xFor(tm.end_time);
    if (x2 - x1 < minDashPx) {
      const mid = (x1 + x2) / 2;
      x1 = Math.max(padding, mid - minDashPx / 2);
      x2 = Math.min(padding + chartWidth, x1 + minDashPx);
    }
    if (!bySpeaker.has(t.speaker)) {
      bySpeaker.set(t.speaker, []);
      order.push(t.speaker);
    }
    bySpeaker.get(t.speaker)!.push({
      index: t.index,
      heat: t.heat,
      isSpike: t.is_spike,
      x1,
      x2,
      xMid: (x1 + x2) / 2,
      y: yFor(t.heat),
    });
  });

  return order.map((speaker) => ({
    speaker,
    color: getSpeakerColor(speaker),
    dashes: bySpeaker.get(speaker)!,
  }));
}

/**
 * Pure: simulated ("what-if") turns as dashes, reusing the REAL turn's time span
 * at the same conversation index (sim turns carry no timing of their own) so the
 * dashed hypothetical lands exactly over the solid dash it replaces. Grouped per
 * speaker, in its own color. Skips any sim turn without a matching real span.
 */
export function mapSimulatedToDashes(
  simulated: SimulatedTurn[],
  timing: TurnTiming[],
  opts: TimeMapOptions,
): SpeakerDashes[] {
  const asPerTurn: AnalyzePerTurn[] = [];
  const alignedTiming: TurnTiming[] = [];
  for (const s of simulated) {
    const tm = timing[s.index];
    if (!tm) continue;
    asPerTurn.push({
      index: s.index,
      speaker: s.speaker,
      heat: s.heat,
      markers: [],
      is_spike: false,
      trigger_phrase: null,
    });
    alignedTiming.push(tm);
  }
  return mapTurnsToDashes(asPerTurn, alignedTiming, opts);
}

/** Pure: playhead x for a playback position on the time axis — the SAME seconds→x
 *  scale as mapTurnsToDashes, so the playhead sits on the current speaker's dash.
 *  Returns null when there's no positive duration. Exported for alignment tests. */
export function playheadXForSeconds(
  seconds: number,
  opts: { width: number; padding: number; duration: number },
): number | null {
  const { width, padding, duration } = opts;
  if (!(duration > 0)) return null;
  const chartWidth = width - padding * 2;
  const frac = Math.max(0, Math.min(1, seconds / duration));
  return padding + frac * chartWidth;
}

/**
 * Pure: each speaker's share (0–1) of total TALKING time, from real durations.
 * Silence isn't attributed to anyone — the denominator is summed utterance
 * length — so shares answer "of the talking, who did how much" (the owner's
 * 2/3-vs-1/3 intuition). `entries` is index-aligned with `timing`.
 */
export function computeTalkShares(
  entries: { speaker: string }[],
  timing: TurnTiming[],
): Record<string, number> {
  const totals: Record<string, number> = {};
  let grand = 0;
  entries.forEach((e, i) => {
    const tm = timing[i];
    if (!tm) return;
    const d = Math.max(0, tm.end_time - tm.start_time);
    totals[e.speaker] = (totals[e.speaker] ?? 0) + d;
    grand += d;
  });
  const shares: Record<string, number> = {};
  for (const s of Object.keys(totals)) {
    shares[s] = grand > 0 ? totals[s] / grand : 0;
  }
  return shares;
}

/** A cell in the time-aligned tap strip: either a tappable turn or a silent gap
 *  spacer. `seconds` drives its flex weight, so the strip mirrors the dashes'
 *  time positions WITHOUT needing a measured pixel width. */
export interface ScrubCell {
  kind: "turn" | "gap";
  index?: number;
  speaker?: string;
  heat?: number;
  seconds: number;
}

/**
 * Pure: lay out the time axis as a sequence of turn cells and silent-gap spacers
 * proportional to real seconds. Overlapping turns (diarization can overlap) never
 * produce a negative gap. Exported for direct testing of the layout weights.
 */
export function buildTimeScrubCells(
  perTurn: AnalyzePerTurn[],
  timing: TurnTiming[],
  duration: number,
): ScrubCell[] {
  const cells: ScrubCell[] = [];
  let cursor = 0;
  perTurn.forEach((t, i) => {
    const tm = timing[i];
    const start = Math.max(0, tm.start_time);
    if (start - cursor > 1e-6) cells.push({ kind: "gap", seconds: start - cursor });
    cells.push({
      kind: "turn",
      index: t.index,
      speaker: t.speaker,
      heat: t.heat,
      seconds: Math.max(0, tm.end_time - tm.start_time),
    });
    cursor = Math.max(cursor, tm.end_time);
  });
  if (duration - cursor > 1e-6) cells.push({ kind: "gap", seconds: duration - cursor });
  return cells;
}

/** m:ss for the tiny time-axis end label. */
export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${rem.toString().padStart(2, "0")}`;
}

/** A narrow-range band annotation (§1 y-scale honesty). When the whole
 *  conversation's heat sits in a narrow window on the absolute 0–100 scale — a
 *  calm talk renders as a nearly flat line, which is ACCURATE but reads as
 *  "nothing here" — we keep the honest absolute scale and instead draw a subtle
 *  shaded band + plain caption ("all turns stayed in the calm range"), rather
 *  than auto-zooming the y-axis into meaningless noise. Returns null when the
 *  spread is wide enough that the line already tells the story on its own, or when
 *  there aren't enough turns to characterize a range. */
export interface HeatRangeBand {
  label: string;
  minHeat: number;
  maxHeat: number;
}

export function heatRangeBand(
  perTurn: Pick<AnalyzePerTurn, "heat">[],
): HeatRangeBand | null {
  if (perTurn.length < 2) return null;
  const heats = perTurn.map((t) => t.heat);
  const minHeat = Math.min(...heats);
  const maxHeat = Math.max(...heats);
  // Wide spread → the line's shape is already informative; no band needed.
  if (maxHeat - minHeat > 20) return null;
  const mid = (minHeat + maxHeat) / 2;
  const label =
    mid <= 33
      ? "All turns stayed in the calm range"
      : mid <= 66
        ? "All turns stayed in the mid range"
        : "All turns stayed in the high range";
  return { label, minHeat, maxHeat };
}

interface HeatChartProps {
  perTurn: AnalyzePerTurn[];
  // The original transcript, index-aligned with perTurn. The backend's per_turn
  // carries no text (only heat/markers), so the inspector resolves each turn's
  // words from here by index. Optional so the chart still renders without it.
  turns?: { speaker: string; text: string }[];
  // §2/§3 — per-speaker display labels (name → deeper/higher voice → generic).
  // Optional: when absent (old recording / pre-labels server) every speaker
  // falls back to its raw id, so the legend + inspector render exactly as before.
  speakerLabels?: SpeakerLabels;
  height?: number;

  // --- "What if" simulated overlay (all optional; the chart is fully usable
  // without any of these) ---
  /** Simulated per-turn heat (pivot → last turn) to overlay as dashed lines.
   *  Null/undefined = no simulation available. */
  simulated?: SimulatedTurn[] | null;
  /** Whether the overlay is currently shown (toggle state owned by the parent).
   *  When false the dashed lines + "simulated" legend entry hide, without any
   *  refetch. */
  showSimulation?: boolean;
  /** Toggle the overlay on/off (the chip near the legend). */
  onToggleSimulation?: () => void;
  /** Fire a counterfactual for the currently selected turn (the pivot). */
  onWhatIf?: (pivotIndex: number) => void;
  /** True while a counterfactual request is in flight (button spinner). */
  whatIfLoading?: boolean;
  /** The turn index the in-flight/errored request pertains to, so loading and
   *  error states only attach to the inspector for that turn. */
  whatIfPivotIndex?: number | null;
  /** Honest inline error from the last counterfactual attempt (null = none). */
  whatIfError?: string | null;
  /** Retry the last failed counterfactual. */
  onRetryWhatIf?: () => void;

  // --- Replay playhead sync (all optional; the chart is fully usable without
  // any of these) ---
  /** Current playback position in seconds. A thin vertical playhead is drawn at
   *  the x of the turn this position falls in. Null/undefined = no playhead. */
  playheadSeconds?: number | null;
  /** Per-turn timing, index-aligned with `perTurn`, that the playhead maps
   *  against. Without it the playhead can't be placed and is omitted. */
  turnsTiming?: TurnTiming[];
  /** Tapping a chart point (or scrubber cell) also seeks playback to that turn's
   *  start_time. Wired by ReplayScreen to the media player. */
  onSeekToTurn?: (startTime: number) => void;
  /** Total recording length in seconds, for the time axis span. Optional; when
   *  absent (or ≤ 0) the axis falls back to the last utterance's end_time. Only
   *  meaningful alongside usable `turnsTiming`. */
  durationSeconds?: number | null;
}

/**
 * Per-speaker heat over the course of the conversation, modeled on
 * ToneSparkline (Svg + Polyline + Circle). One line per speaker, x = turn index
 * across the whole conversation, y = heat 0–100. Spike turns get a larger amber
 * dot. Tapping any point selects that turn and reveals a "turn inspector" card
 * below the chart (speaker, text, heat, markers, trigger phrase).
 *
 * Width is measured via onLayout so the chart is full-width responsive.
 */
export default function HeatChart({
  perTurn,
  turns,
  speakerLabels,
  height = 180,
  simulated,
  showSimulation = true,
  onToggleSimulation,
  onWhatIf,
  whatIfLoading = false,
  whatIfPivotIndex = null,
  whatIfError = null,
  onRetryWhatIf,
  playheadSeconds = null,
  turnsTiming,
  onSeekToTurn,
  durationSeconds = null,
}: HeatChartProps) {
  const [width, setWidth] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);

  const padding = 16;

  // Selecting a turn (via a point or a scrubber cell) also seeks playback to
  // that turn's start_time when replay is wired up — so the chart doubles as a
  // tap-to-seek control. Guarded on timing being present for that index.
  const selectTurn = (index: number) => {
    setSelected(index);
    const timing = turnsTiming?.[index];
    if (onSeekToTurn && timing) {
      onSeekToTurn(timing.start_time);
    }
  };
  // Mode selection. When we have honest, index-aligned utterance timing we draw
  // the TIME AXIS (dashes over real recording seconds). Otherwise — a pre-
  // timestamp recording, or a pasted transcript that never had timing — we fall
  // back to the legacy evenly-spaced polyline. `showLegacyNote` distinguishes
  // "timing was expected but unusable" (worth a subtle note) from "this was
  // never a timed conversation" (no note).
  const useTimeAxis = timingIsUsable(turnsTiming, perTurn.length);
  const timingProvided = !!turnsTiming && turnsTiming.length > 0;
  const showLegacyNote = timingProvided && !useTimeAxis;
  const duration =
    useTimeAxis && turnsTiming ? durationForTiming(turnsTiming, durationSeconds) : 0;

  // Stable speaker order + color for the legend (independent of measured width,
  // so the legend is populated before the first layout pass), shared by both
  // modes.
  const speakerOrder: { speaker: string; color: string }[] = [];
  {
    const seen = new Set<string>();
    for (const t of perTurn) {
      if (!seen.has(t.speaker)) {
        seen.add(t.speaker);
        speakerOrder.push({ speaker: t.speaker, color: getSpeakerColor(t.speaker) });
      }
    }
  }

  const overlayActive = !!simulated && simulated.length > 0 && showSimulation;

  // §1 y-scale honesty: when every turn sits in a narrow window we keep the
  // absolute scale but draw a subtle shaded band + caption (below) instead of
  // zooming into noise. Band rect spans the [min,max] heat range across the
  // chart width, with a small floor so a dead-flat conversation still shows.
  const rangeBand = heatRangeBand(perTurn);
  const bandRect =
    rangeBand && width > 0
      ? (() => {
          const chartHeight = height - padding * 2;
          const yOf = (h: number) =>
            padding +
            (chartHeight - (Math.max(0, Math.min(100, h)) / 100) * chartHeight);
          const yTop = yOf(rangeBand.maxHeat);
          const yBottom = yOf(rangeBand.minHeat);
          return {
            x: padding,
            y: yTop,
            w: width - padding * 2,
            h: Math.max(6, yBottom - yTop),
          };
        })()
      : null;

  // --- Time-axis geometry (primary) ---
  const dashLines =
    useTimeAxis && width > 0 && turnsTiming
      ? mapTurnsToDashes(perTurn, turnsTiming, { width, height, padding, duration })
      : [];
  const simDashLines =
    overlayActive && useTimeAxis && width > 0 && turnsTiming
      ? mapSimulatedToDashes(simulated!, turnsTiming, {
          width,
          height,
          padding,
          duration,
        })
      : [];
  const talkShares =
    useTimeAxis && turnsTiming ? computeTalkShares(perTurn, turnsTiming) : {};
  // Time-aligned tap strip, rendered as proportional flex so it needs no measured
  // width (works before/without a layout pass) yet still mirrors the dashes.
  const scrubCells =
    useTimeAxis && turnsTiming
      ? buildTimeScrubCells(perTurn, turnsTiming, duration)
      : [];
  const timePlayheadX =
    useTimeAxis && playheadSeconds != null && width > 0
      ? playheadXForSeconds(playheadSeconds, { width, padding, duration })
      : null;

  // --- Legacy index-spaced geometry (fallback) ---
  // Only compute geometry once we've measured a width (first layout pass).
  // totalTurns is passed EXPLICITLY so the real and simulated lines share one
  // x-scale by construction, not by the (currently true) coincidence that
  // server turn indexes are contiguous 0..n-1.
  const lines =
    !useTimeAxis && width > 0
      ? mapTurnsToLines(perTurn, {
          width,
          height,
          padding,
          totalTurns: perTurn.length,
        })
      : [];
  const simLines =
    !useTimeAxis && overlayActive && width > 0
      ? mapSimulatedToLines(simulated!, {
          width,
          height,
          padding,
          totalTurns: perTurn.length,
        })
      : [];
  const legacyPlayheadX =
    !useTimeAxis &&
    playheadSeconds != null &&
    turnsTiming &&
    turnsTiming.length > 0 &&
    width > 0
      ? playheadXForTime(playheadSeconds, turnsTiming, {
          width,
          padding,
          totalTurns: perTurn.length,
        })
      : null;

  const activePlayheadX = useTimeAxis ? timePlayheadX : legacyPlayheadX;

  const selectedTurn =
    selected !== null
      ? perTurn.find((t) => t.index === selected) ?? null
      : null;

  // Non-baseline prosody chips for the selected turn (empty when no voice data).
  const voiceChips = voiceChipsFor(selectedTurn?.voice);

  // Loading/error only belong to the inspector when they pertain to the turn
  // currently selected (the pivot the parent is acting on).
  const isPivotSelected = selected !== null && selected === whatIfPivotIndex;
  const showWhatIfLoading = whatIfLoading && isPivotSelected;
  const showWhatIfError = !!whatIfError && isPivotSelected && !whatIfLoading;

  return (
    <View testID="heat-chart">
      {/* Legend: color swatch + speaker name, one row. On the time axis each
          speaker also carries their share of total talking time (from real
          utterance durations) — the direct answer to "who spoke more". */}
      <View style={styles.legend}>
        {speakerOrder.map((s) => {
          const share = talkShares[s.speaker];
          return (
            <View key={s.speaker} style={styles.legendItem}>
              <View
                style={[styles.swatch, { backgroundColor: s.color }]}
                testID={`legend-swatch-${s.speaker}`}
              />
              <Text style={styles.legendText}>
                {speakerLabel(s.speaker, speakerLabels)}
                {useTimeAxis && share !== undefined
                  ? ` — ${Math.round(share * 100)}% of talking`
                  : ""}
              </Text>
            </View>
          );
        })}

        {/* Dashed-line legend entry, only while the overlay is visible. */}
        {overlayActive && (
          <View style={styles.legendItem} testID="legend-simulated">
            <View style={styles.dashSwatch}>
              <View style={styles.dashSeg} />
              <View style={styles.dashSeg} />
            </View>
            <Text style={styles.legendText}>simulated</Text>
          </View>
        )}

        {/* Toggle chip: show/hide the overlay without refetching. Present
            whenever a simulation exists (even when currently hidden). */}
        {!!simulated && simulated.length > 0 && (
          <TouchableOpacity
            testID="simulation-toggle"
            style={styles.simToggle}
            onPress={onToggleSimulation}
          >
            <Text style={styles.simToggleText}>
              {showSimulation ? "Simulation ✕" : "Show simulation"}
            </Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Chart surface — onLayout gives us the responsive width. position:
          relative so the time-axis tap overlay can be absolutely placed over the
          dashes. */}
      <View
        style={{ height, position: "relative" }}
        onLayout={(e) => setWidth(e.nativeEvent.layout.width)}
      >
        {width > 0 && (
          <Svg width={width} height={height}>
            {/* §1 narrow-range band: a subtle shaded strip over the [min,max]
                heat window, drawn first so the marks sit on top. Honest — the
                y-axis is still the absolute 0–100 scale; this just says "the
                whole talk lived in this narrow band" instead of a flat line that
                reads as nothing. */}
            {bandRect && (
              <Rect
                testID="heat-range-band"
                x={bandRect.x}
                y={bandRect.y}
                width={bandRect.w}
                height={bandRect.h}
                fill={INK}
                fillOpacity={0.05}
                rx={4}
              />
            )}

            {/* Replay playhead: a thin vertical ink line, drawn first so the
                marks sit on top of it. On the time axis it maps by real seconds,
                so it lands on the dash of whoever is speaking right now. */}
            {activePlayheadX != null && (
              <Line
                testID="playhead-line"
                x1={activePlayheadX}
                y1={padding}
                x2={activePlayheadX}
                y2={height - padding}
                stroke={INK}
                strokeWidth={1.5}
                strokeOpacity={0.35}
              />
            )}

            {useTimeAxis ? (
              <>
                {/* Faint per-speaker connector through dash midpoints — an aid to
                    read a speaker's arc, deliberately subordinate to the dashes
                    (thin, low opacity). */}
                {dashLines.map((line) =>
                  line.dashes.length > 1 ? (
                    <Polyline
                      key={`connector-${line.speaker}`}
                      testID={`heat-line-${line.speaker}`}
                      points={line.dashes.map((d) => `${d.xMid},${d.y}`).join(" ")}
                      fill="none"
                      stroke={line.color}
                      strokeWidth={1}
                      strokeOpacity={0.25}
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                  ) : null,
                )}
                {/* Simulated dashes FIRST so the real dashes sit on top. */}
                {simDashLines.flatMap((line) =>
                  line.dashes.map((d) => (
                    <Line
                      key={`sim-dash-${line.speaker}-${d.index}`}
                      testID={`sim-dash-${d.index}`}
                      x1={d.x1}
                      y1={d.y}
                      x2={d.x2}
                      y2={d.y}
                      stroke={line.color}
                      strokeWidth={3}
                      strokeDasharray={SIM_DASH}
                      strokeOpacity={SIM_OPACITY}
                      strokeLinecap="round"
                    />
                  )),
                )}
                {/* The primary mark: one horizontal dash per turn, speaker-
                    colored, spanning its real talk time. The selected dash reads
                    thicker; others dim slightly so the selection stands out. */}
                {dashLines.flatMap((line) =>
                  line.dashes.map((d) => (
                    <Line
                      key={`dash-${line.speaker}-${d.index}`}
                      testID={`heat-dash-${d.index}`}
                      x1={d.x1}
                      y1={d.y}
                      x2={d.x2}
                      y2={d.y}
                      stroke={line.color}
                      strokeWidth={selected === d.index ? 7 : 4}
                      strokeOpacity={
                        selected === null || selected === d.index ? 1 : 0.8
                      }
                      strokeLinecap="round"
                      onPress={() => selectTurn(d.index)}
                    />
                  )),
                )}
                {/* Spike markers: a small amber dot centered on the dash keeps
                    spikes legible without discarding the speaker's color. */}
                {dashLines.flatMap((line) =>
                  line.dashes
                    .filter((d) => d.isSpike)
                    .map((d) => (
                      <Circle
                        key={`spikept-${d.index}`}
                        testID={`heat-spike-${d.index}`}
                        cx={d.xMid}
                        cy={d.y}
                        r={4}
                        fill={AMBER}
                        stroke={selected === d.index ? INK : "none"}
                        strokeWidth={selected === d.index ? 2 : 0}
                        onPress={() => selectTurn(d.index)}
                      />
                    )),
                )}
              </>
            ) : (
              <>
                {/* Legacy fallback: evenly-spaced polyline per speaker. */}
                {simLines.map((line) => (
                  <Polyline
                    key={`sim-line-${line.speaker}`}
                    testID={`sim-line-${line.speaker}`}
                    points={line.points.map((p) => `${p.x},${p.y}`).join(" ")}
                    fill="none"
                    stroke={line.color}
                    strokeWidth={2}
                    strokeDasharray={SIM_DASH}
                    strokeOpacity={SIM_OPACITY}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                ))}
                {lines.map((line) => (
                  <Polyline
                    key={`line-${line.speaker}`}
                    testID={`heat-line-${line.speaker}`}
                    points={line.points.map((p) => `${p.x},${p.y}`).join(" ")}
                    fill="none"
                    stroke={line.color}
                    strokeWidth={2}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                ))}
                {lines.flatMap((line) =>
                  line.points.map((p) => (
                    <Circle
                      key={`pt-${line.speaker}-${p.index}`}
                      testID={
                        p.isSpike
                          ? `heat-spike-${p.index}`
                          : `heat-point-${p.index}`
                      }
                      cx={p.x}
                      cy={p.y}
                      r={p.isSpike ? 6 : 4}
                      fill={p.isSpike ? AMBER : line.color}
                      stroke={selected === p.index ? INK : "none"}
                      strokeWidth={selected === p.index ? 2 : 0}
                      onPress={() => selectTurn(p.index)}
                    />
                  )),
                )}
              </>
            )}
          </Svg>
        )}
      </View>

      {/* §1 narrow-range caption: names the band in plain words so the shaded
          strip above is self-explanatory (and honest — nothing was zoomed). */}
      {rangeBand && (
        <Text style={styles.rangeBandNote} testID="heat-range-band-note">
          {rangeBand.label} ({rangeBand.minHeat}–{rangeBand.maxHeat} of 100).
        </Text>
      )}

      {/* Tiny time axis: 0:00 on the left, the recording length on the right, so
          the dashes read plainly as "position in the recording". */}
      {useTimeAxis && duration > 0 && (
        <View style={styles.timeAxis} testID="heat-time-axis">
          <Text style={styles.timeAxisLabel}>0:00</Text>
          <Text style={styles.timeAxisLabel}>{formatClock(duration)}</Text>
        </View>
      )}

      {/* Subtle honesty note: timing was expected for this recording but isn't
          usable, so we're showing turns evenly spaced rather than inventing a
          timeline. */}
      {showLegacyNote && (
        <Text style={styles.legacyNote} testID="heat-legacy-note">
          Timeline unavailable for this recording — turns shown evenly spaced.
        </Text>
      )}

      {/* Tap targets: SVG dashes/circles are small/unreliable to hit, so we back
          them with a full-height touch strip, one cell per turn.
          - Time axis: cells are flex-weighted by real seconds (with silent-gap
            spacers), so a cell sits under the dash it selects — no measured width
            needed. Short turns keep a minimum width so they stay hittable.
          - Legacy: evenly-spaced columns, one per turn. */}
      {useTimeAxis ? (
        <View
          style={[styles.scrubberRow, styles.scrubberRowTime]}
          testID="heat-scrubber"
        >
          {scrubCells.map((c, i) =>
            c.kind === "gap" ? (
              <View
                key={`gap-${i}`}
                style={{ flexGrow: c.seconds, flexShrink: 1, flexBasis: 0 }}
              />
            ) : (
              <TouchableOpacity
                key={`scrub-${c.index}`}
                testID={`scrub-${c.index}`}
                onPress={() => selectTurn(c.index!)}
                accessibilityLabel={`Turn ${c.index! + 1}, ${speakerLabel(c.speaker!, speakerLabels)}, heat ${c.heat}`}
                style={{
                  flexGrow: Math.max(c.seconds, 1e-3),
                  flexShrink: 0,
                  flexBasis: 0,
                  minWidth: MIN_TAP_PX,
                  height: 24,
                }}
              />
            ),
          )}
        </View>
      ) : (
        <View style={styles.scrubberRow} testID="heat-scrubber">
          {perTurn.map((t) => (
            <TouchableOpacity
              key={`scrub-${t.index}`}
              testID={`scrub-${t.index}`}
              style={styles.scrubCell}
              onPress={() => selectTurn(t.index)}
              accessibilityLabel={`Turn ${t.index + 1}, ${speakerLabel(t.speaker, speakerLabels)}, heat ${t.heat}`}
            />
          ))}
        </View>
      )}

      {/* Turn inspector — shows the selected turn's detail. */}
      {selectedTurn && (
        <View style={styles.inspector} testID="turn-inspector">
          <View style={styles.inspectorHeader}>
            <Text
              style={[
                styles.inspectorSpeaker,
                { color: getSpeakerColor(selectedTurn.speaker) },
              ]}
            >
              {speakerLabel(selectedTurn.speaker, speakerLabels)}
            </Text>
            <Text style={styles.inspectorHeat}>heat {selectedTurn.heat}</Text>
          </View>
          <Text style={styles.inspectorText}>
            {turns?.[selectedTurn.index]?.text ?? ""}
          </Text>
          {(selectedTurn.markers.length > 0 || voiceChips.length > 0) && (
            <View style={styles.chipRow}>
              {/* Behavioral markers: filled chips. */}
              {selectedTurn.markers.map((m) => (
                <View key={m} style={styles.chip}>
                  <Text style={styles.chipText}>{m.replace(/_/g, " ")}</Text>
                </View>
              ))}
              {/* Voice/prosody: outline chips, visually distinct from the filled
                  marker chips. Only the non-baseline dimensions appear; nothing
                  when voice is null/absent (old servers / degraded prosody). */}
              {voiceChips.map((c) => (
                <View
                  key={`voice-${c.kind}`}
                  testID={`voice-chip-${c.kind}`}
                  style={styles.voiceChip}
                >
                  <Text style={styles.voiceChipText}>{c.label}</Text>
                </View>
              ))}
            </View>
          )}
          {selectedTurn.trigger_phrase && (
            <Text style={styles.trigger} testID="turn-inspector-trigger">
              Trigger: “{selectedTurn.trigger_phrase}”
            </Text>
          )}

          {/* "What if this was said differently?" — the counterfactual entry
              point. Only shown when the parent wired up onWhatIf. */}
          {onWhatIf && (
            <View style={styles.whatIfBlock}>
              <TouchableOpacity
                testID="what-if-button"
                style={[
                  styles.whatIfButton,
                  showWhatIfLoading && styles.whatIfButtonLoading,
                ]}
                disabled={showWhatIfLoading}
                onPress={() => onWhatIf(selectedTurn.index)}
              >
                {showWhatIfLoading ? (
                  <View style={styles.whatIfLoadingRow}>
                    <ActivityIndicator size="small" color={PRIMARY} />
                    <Text style={styles.whatIfButtonText}>Imagining…</Text>
                  </View>
                ) : (
                  <Text style={styles.whatIfButtonText}>
                    What if this was said differently?
                  </Text>
                )}
              </TouchableOpacity>

              {/* Honest inline error — never a fabricated simulation. */}
              {showWhatIfError && (
                <View style={styles.whatIfError} testID="what-if-error">
                  <Text style={styles.whatIfErrorText}>{whatIfError}</Text>
                  <TouchableOpacity
                    testID="what-if-retry"
                    onPress={onRetryWhatIf}
                  >
                    <Text style={styles.whatIfRetryText}>Try again</Text>
                  </TouchableOpacity>
                </View>
              )}
            </View>
          )}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  legend: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 14,
    marginBottom: 8,
  },
  legendItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  swatch: {
    width: 12,
    height: 12,
    borderRadius: 3,
  },
  legendText: {
    fontSize: 13,
    color: MUTED,
    fontWeight: "600",
  },
  scrubberRow: {
    flexDirection: "row",
    marginTop: 6,
  },
  scrubCell: {
    flex: 1,
    height: 24,
  },
  // Time-axis strip: pad the sides by the chart's inner padding (16) so a cell's
  // horizontal position lines up with the dash it selects.
  scrubberRowTime: {
    paddingHorizontal: 16,
  },
  timeAxis: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 4,
    paddingHorizontal: 16,
  },
  timeAxisLabel: {
    fontSize: 11,
    color: MUTED,
    fontVariant: ["tabular-nums"],
  },
  legacyNote: {
    fontSize: 12,
    color: MUTED,
    fontStyle: "italic",
    marginTop: 6,
  },
  rangeBandNote: {
    fontSize: 12,
    color: MUTED,
    fontStyle: "italic",
    marginTop: 6,
  },
  inspector: {
    marginTop: 10,
    padding: 12,
    borderRadius: 10,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  inspectorHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 4,
  },
  inspectorSpeaker: {
    fontSize: 13,
    fontWeight: "700",
    textTransform: "uppercase",
  },
  inspectorHeat: {
    fontSize: 13,
    fontWeight: "600",
    color: AMBER,
  },
  inspectorText: {
    fontSize: 15,
    lineHeight: 21,
    color: INK,
  },
  chipRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
    marginTop: 8,
  },
  chip: {
    backgroundColor: "#F3F4F6",
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  chipText: {
    fontSize: 11,
    color: MUTED,
    fontWeight: "600",
  },
  // Outline (unfilled) chip so prosody reads as a different KIND of tag than the
  // filled marker chips sitting beside it.
  voiceChip: {
    backgroundColor: "transparent",
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  voiceChipText: {
    fontSize: 11,
    color: MUTED,
    fontWeight: "600",
  },
  trigger: {
    marginTop: 8,
    fontSize: 13,
    color: AMBER,
    fontStyle: "italic",
  },
  // Dashed legend swatch: two short segments with a gap, echoing the overlay.
  dashSwatch: {
    width: 16,
    height: 12,
    flexDirection: "row",
    alignItems: "center",
    gap: 3,
    opacity: SIM_OPACITY,
  },
  dashSeg: {
    width: 6,
    height: 2,
    borderRadius: 1,
    backgroundColor: MUTED,
  },
  simToggle: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#F9FAFB",
  },
  simToggleText: {
    fontSize: 12,
    fontWeight: "600",
    color: MUTED,
  },
  whatIfBlock: {
    marginTop: 12,
  },
  whatIfButton: {
    backgroundColor: "#EFF6FF",
    borderWidth: 1,
    borderColor: PRIMARY,
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 14,
    alignItems: "center",
  },
  whatIfButtonLoading: {
    opacity: 0.7,
  },
  whatIfLoadingRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  whatIfButtonText: {
    fontSize: 14,
    fontWeight: "700",
    color: PRIMARY,
  },
  whatIfError: {
    marginTop: 8,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  },
  whatIfErrorText: {
    flex: 1,
    fontSize: 13,
    color: DANGER,
  },
  whatIfRetryText: {
    fontSize: 13,
    fontWeight: "700",
    color: PRIMARY,
  },
});
