/**
 * Pure geometry + series math for the "Your growth" chart (strip + screen).
 *
 * The x axis is TIME (recording created_at), never an index — two recordings a
 * month apart sit a month apart. The mapping reuses chartZoom's window→pixel
 * math (`secondsToX` is unit-agnostic; here the "window" is the epoch-ms span
 * of the user's points). Null scores are GAPS: an identified recording whose
 * stored analysis carries no usable report card gets NO dot and never bends the
 * trend — a gap is honest, a zero would be a lie.
 *
 * Kept free of React / react-native so the series math (moving average, partner
 * filter, gap handling) is unit-testable in isolation — mirrors chartZoom.ts.
 */

import type { GrowthPoint } from "../api/client";
import { secondsToX, type ChartGeom, type ZoomWindow } from "./chartZoom";

/** The moving-average trend line is only drawn once there are at least this
 *  many SCORED points — below that a "trend" would be noise presented as
 *  signal. */
export const TREND_MIN_POINTS = 5;

/** Trailing moving-average window (in scored points). */
export const TREND_WINDOW = 3;

/** Points that can actually be drawn: identified AND scored. The rest of an
 *  identified-but-scoreless point's life is the gap it leaves. */
export function scoredPoints(points: GrowthPoint[]): GrowthPoint[] {
  return points.filter((p) => typeof p.my_score === "number");
}

/** The epoch-ms time window spanned by `points`, or null when none parse.
 *  Shaped as a chartZoom window so the same mapping math applies. */
export function timeWindow(points: GrowthPoint[]): ZoomWindow | null {
  const times = points
    .map((p) => Date.parse(p.timestamp))
    .filter((t) => Number.isFinite(t));
  if (times.length === 0) return null;
  return { start: Math.min(...times), end: Math.max(...times) };
}

/** Map a point's timestamp into chart x. A single-point (zero-span) window
 *  centers the dot, like ToneSparkline's single-score case. */
export function timeToX(
  timestamp: string,
  window: ZoomWindow,
  geom: ChartGeom,
): number {
  if (window.end - window.start <= 0) return geom.width / 2;
  return secondsToX(Date.parse(timestamp), window, geom);
}

/** Map a 0–100 score into chart y (top = 100). */
export function scoreToY(score: number, height: number, padding: number): number {
  const chartHeight = height - padding * 2;
  return padding + chartHeight - (score / 100) * chartHeight;
}

/**
 * Trailing moving average over the SCORED series: out[i] is the mean of the
 * last `window` scores up to and including i. Same length as the input — the
 * early entries average what exists so the trend starts at the first dot.
 */
export function movingAverage(
  scores: number[],
  window: number = TREND_WINDOW,
): number[] {
  return scores.map((_, i) => {
    const slice = scores.slice(Math.max(0, i - window + 1), i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

// --- Partner filter ---------------------------------------------------------
// Chips are built from the server's partner_names (real names only — a manual
// tag or a transcript-confirmed name). Points with NO named partner form the
// honest "Unidentified partner" bucket: their partners exist, we just can't
// name them across recordings.

export type PartnerFilter =
  | { kind: "all" }
  | { kind: "partner"; name: string }
  | { kind: "unidentified" };

/** Distinct partner names across all points, first-seen order. */
export function partnerNames(points: GrowthPoint[]): string[] {
  const seen: string[] = [];
  for (const p of points) {
    for (const name of p.partner_names) {
      if (!seen.includes(name)) seen.push(name);
    }
  }
  return seen;
}

/** True when at least one point has no nameable partner. */
export function hasUnidentifiedPartner(points: GrowthPoint[]): boolean {
  return points.some((p) => p.partner_names.length === 0);
}

export function filterPoints(
  points: GrowthPoint[],
  filter: PartnerFilter,
): GrowthPoint[] {
  switch (filter.kind) {
    case "all":
      return points;
    case "partner":
      return points.filter((p) => p.partner_names.includes(filter.name));
    case "unidentified":
      return points.filter((p) => p.partner_names.length === 0);
  }
}
