/**
 * The Replay chart's Y-AXIS, as data.
 *
 * Owner report (2026-09-06): the chart's vertical axis was unlabeled and
 * turned out to be the LLM's per-turn "heat" — so the same voice sat at
 * different heights and the axis read as if it were about the voice. What
 * he wants to see is HIMSELF (his color) at different heights over time:
 * angry, talking over people, calming down, getting emotional. This module
 * makes the y-axis a CHOICE with a name and a unit:
 *
 *   heat          — the LLM's 0–100 intensity score per turn (server/main.py)
 *   frustration / defensiveness / warmth / sadness
 *                 — 0–100 from the per-turn TEXT TONE the phone's on-device
 *                   model read off the words during a live session
 *                   (live/localLlm.ts TextTone, stored as `text_tone` on each
 *                   turn). Uploaded recordings have no per-turn tone today
 *                   (their analysis scores heat + markers per turn and tone
 *                   per transcript), so these chips are disabled for them —
 *                   honestly absent, never faked from markers.
 *   loudness      — RMS level in dBFS (0 = digital full scale; quieter = more
 *                   negative). Phone: live/prosody.ts rmsDbfs; server: prosody.py
 *   pitch         — median F0 in Hz over the turn's VOICED frames; null when
 *                   the turn was mostly unvoiced (never invented)
 *   rate          — words per second (words / turn duration)
 *   person        — not a measurement: one lane per speaker, so the dashes
 *                   read as a turn-taking chart (who spoke when)
 *
 * Everything here is pure TypeScript with NO React / React Native imports so
 * the SAME functions run under plain Node: the HeatChart component, its axis
 * ticks, and the offline proof renderer (tmp/chart-proof/render.ts) all map a
 * value to a pixel through `metricY` below — there is no second copy of the
 * arithmetic to drift.
 */

import type { AnalyzePerTurn, SimulatedTurn, Voice } from "../api/client";
import { getSpeakerColor } from "../utils/speakerColors";
import { secondsToX, type ZoomWindow } from "./chartZoom";

export type ChartMetric =
  | "heat"
  | "frustration"
  | "defensiveness"
  | "warmth"
  | "sadness"
  | "loudness"
  | "pitch"
  | "rate"
  | "person";

/** Selector order (left → right), as the owner asked for it. Heat first: it
 *  is the default and what the chart has always shown. */
export const CHART_METRICS: readonly ChartMetric[] = [
  "heat",
  "frustration",
  "defensiveness",
  "warmth",
  "sadness",
  "loudness",
  "pitch",
  "rate",
  "person",
] as const;

export const DEFAULT_CHART_METRIC: ChartMetric = "heat";

/** Where a metric's number comes from — drives the "which recordings carry
 *  which metrics" story: `llm` = the server's analysis of the transcript
 *  (every analyzed recording); `tone` = the phone's per-turn text tone (live
 *  sessions only); `prosody` = per-turn audio measurement (live sessions from
 *  the phone, uploads from the server); `structure` = no measurement. */
export type MetricKind = "llm" | "tone" | "prosody" | "structure";

export interface MetricSpec {
  id: ChartMetric;
  kind: MetricKind;
  /** Short chip text. */
  chip: string;
  /** Axis title including the unit, e.g. "Heat (0–100, LLM)". */
  axisTitle: string;
  /** The unit alone, appended to formatted values ("dBFS", "Hz", …). */
  unit: string;
  /** Default scale. The domain EXTENDS (in whole tick steps) to cover any data
   *  outside it — a value is never clipped off the chart — but never shrinks,
   *  so two recordings of the same metric share a scale unless one is out of
   *  range. (Person ignores these: its scale is the speaker count.) */
  min: number;
  max: number;
  tickStep: number;
  /** Decimal places for printed values. */
  decimals: number;
  /** Plain-words provenance, for the inspector / proof page. */
  source: string;
}

const TONE_SCALE = { min: 0, max: 100, tickStep: 50, decimals: 0, unit: "/100" };

export const METRIC_SPECS: Record<ChartMetric, MetricSpec> = {
  heat: {
    id: "heat",
    kind: "llm",
    chip: "Heat",
    axisTitle: "Heat (0–100, LLM)",
    unit: "/100",
    min: 0,
    max: 100,
    tickStep: 50,
    decimals: 0,
    source: "LLM-scored intensity per turn from the transcript (0 calm … 100 peak); every analyzed recording",
  },
  frustration: {
    id: "frustration",
    kind: "tone",
    chip: "Frustration",
    axisTitle: "Frustration (0–100, text tone)",
    ...TONE_SCALE,
    source: "on-device text-tone read of the turn's words (live sessions; text_tone.frustration)",
  },
  defensiveness: {
    id: "defensiveness",
    kind: "tone",
    chip: "Defensiveness",
    axisTitle: "Defensiveness (0–100, text tone)",
    ...TONE_SCALE,
    source: "on-device text-tone read of the turn's words (live sessions; text_tone.defensiveness)",
  },
  warmth: {
    id: "warmth",
    kind: "tone",
    chip: "Warmth",
    axisTitle: "Warmth (0–100, text tone)",
    ...TONE_SCALE,
    source: "on-device text-tone read of the turn's words (live sessions; text_tone.warmth)",
  },
  sadness: {
    id: "sadness",
    kind: "tone",
    chip: "Sadness",
    axisTitle: "Sadness (0–100, text tone)",
    ...TONE_SCALE,
    source: "on-device text-tone read of the turn's words (live sessions; text_tone.sadness)",
  },
  loudness: {
    id: "loudness",
    kind: "prosody",
    chip: "Loudness",
    axisTitle: "Loudness (dBFS)",
    unit: "dBFS",
    min: -60,
    max: 0,
    tickStep: 20,
    decimals: 1,
    source: "RMS level of the turn's audio; 0 dBFS = full scale, quieter is more negative (phone on live sessions, server on uploads)",
  },
  pitch: {
    id: "pitch",
    kind: "prosody",
    chip: "Pitch",
    axisTitle: "Pitch (Hz)",
    unit: "Hz",
    min: 0,
    max: 400,
    tickStep: 100,
    decimals: 0,
    source: "median F0 over voiced frames; blank when the turn was mostly unvoiced (phone on live sessions, server on uploads)",
  },
  rate: {
    id: "rate",
    kind: "prosody",
    chip: "Speech rate",
    axisTitle: "Speech rate (words/s)",
    unit: "words/s",
    min: 0,
    max: 6,
    tickStep: 2,
    decimals: 2,
    source: "words in the transcript ÷ turn duration (phone on live sessions, server on uploads)",
  },
  person: {
    id: "person",
    kind: "structure",
    chip: "Person",
    axisTitle: "Person (who is speaking)",
    unit: "",
    min: 0,
    max: 1,
    tickStep: 1,
    decimals: 0,
    source: "one lane per speaker — the turn-taking view, not a measurement",
  },
};

export function metricSpec(metric: ChartMetric): MetricSpec {
  return METRIC_SPECS[metric];
}

export function isChartMetric(value: unknown): value is ChartMetric {
  return typeof value === "string" && (CHART_METRICS as readonly string[]).includes(value);
}

/** The raw per-turn numbers a turn can plot, merged from every source into
 *  one shape. Prosody fields are the wire shape BOTH sources use: live
 *  sessions (`RecordingTurn.prosody`, measured on the phone) and uploads
 *  (`AnalyzePerTurn.voice.{rms_dbfs,pitch_hz,speech_rate}`, measured by the
 *  server). Tone fields are the phone's `RecordingTurn.text_tone` (0–100).
 *  Every field nullable — measurement is best-effort and an unvoiced turn
 *  honestly has no pitch. */
export interface TurnMetrics {
  rms_dbfs?: number | null;
  pitch_hz?: number | null;
  speech_rate?: number | null;
  warmth?: number | null;
  defensiveness?: number | null;
  sadness?: number | null;
  frustration?: number | null;
}

/** What a stored turn carries that the chart can plot (the `RecordingTurn`
 *  subset): the phone's prosody + text tone on a live session; both absent
 *  on an upload. */
export interface TurnFacts {
  prosody?: {
    rms_dbfs?: number | null;
    pitch_hz?: number | null;
    speech_rate?: number | null;
  } | null;
  text_tone?: {
    warmth?: number | null;
    defensiveness?: number | null;
    sadness?: number | null;
    frustration?: number | null;
  } | null;
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function isEmpty(m: TurnMetrics): boolean {
  return (
    m.rms_dbfs == null &&
    m.pitch_hz == null &&
    m.speech_rate == null &&
    m.warmth == null &&
    m.defensiveness == null &&
    m.sadness == null &&
    m.frustration == null
  );
}

/** An upload's `voice` block as TurnMetrics, or null when the server (an old
 *  analysis, or a pre-raw-numbers server) only stored the relative labels. */
export function voiceMetrics(voice: Voice | null | undefined): TurnMetrics | null {
  if (!voice) return null;
  const m: TurnMetrics = {
    rms_dbfs: num(voice.rms_dbfs),
    pitch_hz: num(voice.pitch_hz),
    speech_rate: num(voice.speech_rate),
  };
  return isEmpty(m) ? null : m;
}

/**
 * Index-aligned metrics for every turn, merged from the stored turn's facts
 * (live session: phone prosody + text tone) and the analysis' per-turn voice
 * block (upload: server prosody). For each prosody number the phone's own
 * live measurement wins when present (it is the number the coach acted on),
 * else the server's; tone only ever comes from the phone. Null when a turn
 * has nothing to plot beyond heat.
 */
export function turnMetricsFor(
  perTurn: AnalyzePerTurn[],
  facts?: (TurnFacts | null | undefined)[] | null,
): (TurnMetrics | null)[] {
  return perTurn.map((t, i) => {
    const f = facts?.[i];
    const voice = voiceMetrics(t.voice);
    const m: TurnMetrics = {
      rms_dbfs: num(f?.prosody?.rms_dbfs) ?? voice?.rms_dbfs ?? null,
      pitch_hz: num(f?.prosody?.pitch_hz) ?? voice?.pitch_hz ?? null,
      speech_rate: num(f?.prosody?.speech_rate) ?? voice?.speech_rate ?? null,
      warmth: num(f?.text_tone?.warmth),
      defensiveness: num(f?.text_tone?.defensiveness),
      sadness: num(f?.text_tone?.sadness),
      frustration: num(f?.text_tone?.frustration),
    };
    return isEmpty(m) ? null : m;
  });
}

/** Speakers in first-appearance order — the legend order, and the lane
 *  order of the Person axis. */
export function speakerOrderOf(perTurn: { speaker: string }[]): string[] {
  const order: string[] = [];
  const seen = new Set<string>();
  for (const t of perTurn) {
    if (!seen.has(t.speaker)) {
      seen.add(t.speaker);
      order.push(t.speaker);
    }
  }
  return order;
}

/** Person axis: lane value per speaker. The FIRST speaker gets the TOP lane
 *  (highest value), reading down in legend order. */
export function personLanes(speakerOrder: string[]): Record<string, number> {
  const lanes: Record<string, number> = {};
  const n = speakerOrder.length;
  speakerOrder.forEach((s, i) => {
    lanes[s] = n - 1 - i;
  });
  return lanes;
}

/** Per-conversation context a metric may need beyond the turn itself. */
export interface MetricContext {
  /** Person axis lanes (see personLanes). */
  lanes?: Record<string, number>;
}

/** The number a turn plots under `metric`, or null when it was not measured
 *  (drawn as NOTHING — never as 0). */
export function metricValue(
  metric: ChartMetric,
  turn: { heat: number; speaker: string },
  metrics: TurnMetrics | null | undefined,
  ctx?: MetricContext,
): number | null {
  switch (metric) {
    case "heat":
      return num(turn.heat);
    case "frustration":
      return num(metrics?.frustration);
    case "defensiveness":
      return num(metrics?.defensiveness);
    case "warmth":
      return num(metrics?.warmth);
    case "sadness":
      return num(metrics?.sadness);
    case "loudness":
      return num(metrics?.rms_dbfs);
    case "pitch":
      return num(metrics?.pitch_hz);
    case "rate":
      return num(metrics?.speech_rate);
    case "person":
      return num(ctx?.lanes?.[turn.speaker]);
  }
}

/** Index-aligned values for the whole conversation under one metric. The
 *  Person lanes are derived from `perTurn` when no context is given. */
export function metricValues(
  metric: ChartMetric,
  perTurn: AnalyzePerTurn[],
  metrics: (TurnMetrics | null)[],
  ctx?: MetricContext,
): (number | null)[] {
  const c = ctx ?? (metric === "person" ? { lanes: personLanes(speakerOrderOf(perTurn)) } : undefined);
  return perTurn.map((t, i) => metricValue(metric, t, metrics[i], c));
}

/** True when at least one turn has a value for `metric` (heat and person
 *  always do, given any turns). */
export function metricHasData(
  metric: ChartMetric,
  perTurn: AnalyzePerTurn[],
  metrics: (TurnMetrics | null)[],
): boolean {
  return metricValues(metric, perTurn, metrics).some((v) => v !== null);
}

/** "−23.4 dBFS", "182 Hz", "2.35 words/s", "45/100". A true minus sign so a
 *  negative dBFS reads as a number, not a dash. (Person has no number to
 *  print — the caller shows the speaker's name.) */
export function formatMetricValue(metric: ChartMetric, value: number): string {
  const spec = METRIC_SPECS[metric];
  const number = value.toFixed(spec.decimals).replace("-", "−");
  if (spec.unit === "") return number;
  return spec.unit.startsWith("/") ? `${number}${spec.unit}` : `${number} ${spec.unit}`;
}

export interface MetricDomain {
  metric: ChartMetric;
  min: number;
  max: number;
  /** Ascending, from min to max in whole tick steps. */
  ticks: number[];
  /** Person axis only: the speaker (display) name for each tick. */
  tickLabels?: string[];
}

/** Tick label: the number only ("−40", "200" — the unit lives in the axis
 *  title), or the speaker name on the Person axis. */
export function formatTick(tick: number, domain?: MetricDomain): string {
  if (domain?.tickLabels) {
    const i = domain.ticks.indexOf(tick);
    if (i >= 0 && domain.tickLabels[i] !== undefined) return domain.tickLabels[i];
  }
  // Ticks sit on whole steps for every numeric metric (rate ticks are
  // 0/2/4/6), so plain integer formatting is exact.
  return String(tick).replace("-", "−");
}

/**
 * The y-axis domain for a NUMERIC metric given the values actually present:
 * the spec's default scale, extended outward in whole tick steps until every
 * finite value fits. Never narrowed — a calm conversation keeps the honest
 * full scale rather than auto-zooming into noise (same rule the heat band
 * annotation follows). For "person" use personDomain (lanes, not numbers).
 */
export function metricDomain(
  metric: ChartMetric,
  values: (number | null | undefined)[],
): MetricDomain {
  const spec = METRIC_SPECS[metric];
  let min = spec.min;
  let max = spec.max;
  for (const v of values) {
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    if (v < min) min = Math.floor(v / spec.tickStep) * spec.tickStep;
    if (v > max) max = Math.ceil(v / spec.tickStep) * spec.tickStep;
  }
  const ticks: number[] = [];
  // Round each tick to kill float drift (e.g. 0.1 steps) — steps here are
  // integers, but the guard costs nothing.
  for (let t = min; t <= max + 1e-9; t += spec.tickStep) {
    ticks.push(Math.round(t * 1e6) / 1e6);
  }
  return { metric, min, max, ticks };
}

/** The Person axis domain: one tick per speaker lane, labeled by display
 *  name, first speaker on top. A single speaker sits centered (min = max). */
export function personDomain(
  speakerOrder: string[],
  labelOf: (speaker: string) => string = (s) => s,
): MetricDomain {
  const lanes = personLanes(speakerOrder);
  const n = speakerOrder.length;
  const ticks = speakerOrder.map((s) => lanes[s]).sort((a, b) => a - b);
  const bySpeakerLane = new Map(speakerOrder.map((s) => [lanes[s], labelOf(s)]));
  return {
    metric: "person",
    min: 0,
    max: Math.max(0, n - 1),
    ticks,
    tickLabels: ticks.map((t) => bySpeakerLane.get(t) ?? ""),
  };
}

/** Everything one chart layer needs for a metric: the per-turn values, the
 *  shared domain, and the context — computed ONCE so dashes, ticks, band and
 *  inspector all agree. */
export interface MetricScale {
  metric: ChartMetric;
  values: (number | null)[];
  domain: MetricDomain;
  ctx: MetricContext;
}

export function metricScale(
  metric: ChartMetric,
  perTurn: AnalyzePerTurn[],
  metrics: (TurnMetrics | null)[],
  labelOf?: (speaker: string) => string,
): MetricScale {
  if (metric === "person") {
    const order = speakerOrderOf(perTurn);
    const ctx = { lanes: personLanes(order) };
    return {
      metric,
      values: metricValues(metric, perTurn, metrics, ctx),
      domain: personDomain(order, labelOf),
      ctx,
    };
  }
  const values = metricValues(metric, perTurn, metrics);
  return { metric, values, domain: metricDomain(metric, values), ctx: {} };
}

export interface YGeometry {
  height: number;
  padding: number;
}

/**
 * THE y-mapping. A value at `domain.min` lands on the chart's bottom edge
 * (height − padding), `domain.max` on its top edge (padding); linear between.
 * A degenerate domain (min = max, e.g. one speaker on the Person axis) maps
 * to the vertical center. Clamped to the domain as a last defence (the
 * domain already covers every value, so the clamp is a no-op in practice).
 * Used by the dashes, the legacy polyline, the axis ticks, the band
 * annotation, and the offline proof.
 */
export function metricY(value: number, domain: MetricDomain, geom: YGeometry): number {
  const chartHeight = geom.height - geom.padding * 2;
  const span = domain.max - domain.min;
  const clamped = Math.max(domain.min, Math.min(domain.max, value));
  const frac = span > 0 ? (clamped - domain.min) / span : 0.5;
  return geom.padding + (chartHeight - frac * chartHeight);
}

/** The heat scale, exactly as before this module existed (0–100 fixed). */
export const HEAT_DOMAIN: MetricDomain = metricDomain("heat", []);

// ---------------------------------------------------------------------------
// Time-axis geometry — dashes over real recording seconds. Lives here (not in
// HeatChart.tsx) so the proof renderer draws through the identical code path.
// ---------------------------------------------------------------------------

/** Per-turn timing, index-aligned with `perTurn`. */
export interface TurnTiming {
  start_time: number;
  end_time: number;
}

/** One turn drawn as a horizontal dash. x1..x2 is its real span in pixels. */
export interface DashSegment {
  index: number;
  heat: number;
  /** The plotted number under the active metric (equals `heat` for heat). */
  value: number;
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
  /** Visible time window (zoom). When present, seconds map onto the full width
   *  through this `[start, end]` slice instead of the whole `[0, duration]`, and
   *  x is NOT clamped — off-window dashes fall outside the SVG viewport and are
   *  clipped. Absent = the full unzoomed view (identical to before). */
  window?: ZoomWindow;
  /** Per-speaker color resolver — see HeatChart's MapOptions.colorOf for why
   *  this exists (a confirmed real hash collision between two speakers' plain
   *  getSpeakerColor() results). Defaults to getSpeakerColor when omitted. */
  colorOf?: (speaker: string) => string;
  /** Which number is plotted on y. Default "heat" (prior behavior). */
  metric?: ChartMetric;
  /** Index-aligned raw numbers per turn (see turnMetricsFor). Only read for
   *  the non-heat metrics. */
  turnMetrics?: (TurnMetrics | null)[];
  /** A precomputed scale (metricScale) so several layers (ticks, overlay)
   *  share one domain. Defaults to metricScale over the inputs. */
  scale?: MetricScale;
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
 * Pure: map per-turn values + real timing into one set of horizontal dashes per
 * speaker. x spans [start_time, end_time] in recording seconds; y = the active
 * metric's value through `metricY`. `timing` is index-aligned with `perTurn`.
 * A turn with NO value under the metric (an unvoiced turn's pitch, a turn the
 * phone never measured, an upload's tone) produces NO dash — it is skipped,
 * never drawn at 0.
 */
export function mapTurnsToDashes(
  perTurn: AnalyzePerTurn[],
  timing: TurnTiming[],
  opts: TimeMapOptions,
): SpeakerDashes[] {
  const { width, height, padding, duration } = opts;
  const minDashPx = opts.minDashPx ?? 3;
  const chartWidth = width - padding * 2;
  const metric = opts.metric ?? "heat";
  const metrics = opts.turnMetrics ?? perTurn.map(() => null);
  const { values, domain } = opts.scale ?? metricScale(metric, perTurn, metrics);

  // Zoom: when a window is given, map seconds through it (no clamp — the SVG
  // viewport clips anything off-window). Without one, keep the exact prior
  // behavior: the full [0, duration] view, clamped into the chart.
  const win = opts.window;
  const xFor = (sec: number) => {
    if (win) return secondsToX(sec, win, { width, padding });
    const frac = duration <= 0 ? 0 : Math.max(0, Math.min(1, sec / duration));
    return padding + frac * chartWidth;
  };

  const order: string[] = [];
  const bySpeaker = new Map<string, DashSegment[]>();
  perTurn.forEach((t, i) => {
    // Speaker order is by first appearance in the conversation even when that
    // speaker's first turn has no value — legend/z-order must not depend on
    // which metric is showing.
    if (!bySpeaker.has(t.speaker)) {
      bySpeaker.set(t.speaker, []);
      order.push(t.speaker);
    }
    const value = values[i];
    if (value === null) return;
    const tm = timing[i];
    let x1 = xFor(tm.start_time);
    let x2 = xFor(tm.end_time);
    // Grow a sub-minimum dash to a hittable width. When zoomed we only apply the
    // floor to dashes actually within view, so off-window dashes aren't dragged
    // onto the chart edges (they stay clipped).
    const bothOffWindow =
      !!win && (x2 < padding || x1 > padding + chartWidth);
    if (!bothOffWindow && x2 - x1 < minDashPx) {
      const mid = (x1 + x2) / 2;
      x1 = Math.max(padding, mid - minDashPx / 2);
      x2 = Math.min(padding + chartWidth, x1 + minDashPx);
    }
    bySpeaker.get(t.speaker)!.push({
      index: t.index,
      heat: t.heat,
      value,
      isSpike: t.is_spike,
      x1,
      x2,
      xMid: (x1 + x2) / 2,
      y: metricY(value, domain, { height, padding }),
    });
  });

  const colorOf = opts.colorOf ?? getSpeakerColor;
  return order.map((speaker) => ({
    speaker,
    color: colorOf(speaker),
    dashes: bySpeaker.get(speaker)!,
  }));
}

/**
 * Pure: simulated ("what-if") turns as dashes, reusing the REAL turn's time span
 * at the same conversation index (sim turns carry no timing of their own) so the
 * dashed hypothetical lands exactly over the solid dash it replaces. Grouped per
 * speaker, in its own color. Skips any sim turn without a matching real span.
 * Simulated turns only carry heat, so this is always on the heat scale.
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
  const heatScale: MetricScale = {
    metric: "heat",
    values: asPerTurn.map((t) => t.heat),
    domain: opts.scale?.metric === "heat" ? opts.scale.domain : HEAT_DOMAIN,
    ctx: {},
  };
  return mapTurnsToDashes(asPerTurn, alignedTiming, {
    ...opts,
    metric: "heat",
    turnMetrics: undefined,
    scale: heatScale,
  });
}
