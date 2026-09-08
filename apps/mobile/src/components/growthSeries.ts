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
  return timeMsToX(Date.parse(timestamp), window, geom);
}

/** Same mapping for an already-parsed epoch-ms instant (axis ticks share the
 *  dots' geometry, so a tick and the dot recorded at that instant line up). */
export function timeMsToX(
  t: number,
  window: ZoomWindow,
  geom: ChartGeom,
): number {
  if (window.end - window.start <= 0) return geom.width / 2;
  return secondsToX(t, window, geom);
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

// --- Date axis ticks --------------------------------------------------------
// The x axis is real time, so its labels must be real calendar boundaries
// (midnights / first-of-months), not "older … newer". Ticks are chosen from
// the window's span: day labels ("Aug 20") for short windows, month labels
// ("Aug", with the year wherever it changes) once the window is long enough
// that individual days would just be noise.

export interface DateTick {
  /** Epoch ms — map with `timeMsToX` against the same window as the dots. */
  t: number;
  label: string;
}

export interface DateTickOptions {
  /** Place boundaries + name days in UTC instead of the device's local time
   *  zone. On device the default (local) is the honest choice — a recording
   *  made at 11pm belongs to *that* day; tests pass `utc: true` so the
   *  expected labels don't depend on the machine running them. */
  utc?: boolean;
}

const DAY_MS = 86_400_000;
/** Up to this span, ticks sit on day boundaries and carry day labels. */
export const DAY_TICKS_MAX_SPAN_MS = 60 * DAY_MS;
const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

interface Calendar {
  year(t: number): number;
  month(t: number): number;
  day(t: number): number;
  /** Midnight at the start of the given calendar date. */
  make(y: number, m: number, d: number): number;
}

const LOCAL: Calendar = {
  year: (t) => new Date(t).getFullYear(),
  month: (t) => new Date(t).getMonth(),
  day: (t) => new Date(t).getDate(),
  make: (y, m, d) => new Date(y, m, d).getTime(),
};

const UTC: Calendar = {
  year: (t) => new Date(t).getUTCFullYear(),
  month: (t) => new Date(t).getUTCMonth(),
  day: (t) => new Date(t).getUTCDate(),
  make: (y, m, d) => Date.UTC(y, m, d),
};

function dayLabel(t: number, cal: Calendar): string {
  return `${MONTHS[cal.month(t)]} ${cal.day(t)}`;
}

function dayKey(t: number, cal: Calendar): string {
  return `${cal.year(t)}-${cal.month(t)}-${cal.day(t)}`;
}

/** Keep at most `max` of `items`, spread evenly, always keeping the first. */
function thin<T>(items: T[], max: number): T[] {
  if (max <= 0) return [];
  if (items.length <= max) return items;
  const step = Math.ceil(items.length / max);
  return items.filter((_, i) => i % step === 0);
}

/**
 * Pick "nice" date ticks for an epoch-ms window, never more than `maxTicks`.
 *
 *  - zero span → a single day label at the point itself;
 *  - span ≤ DAY_TICKS_MAX_SPAN_MS → day labels: the window's first and last
 *    dates are always present (at the window's own ends), plus the midnights
 *    in between, thinned evenly to fit `maxTicks`;
 *  - longer → month labels on first-of-month boundaries inside the window,
 *    thinned evenly; the year is appended on the first tick and wherever it
 *    changes, but only when the window actually crosses a year boundary.
 */
export function dateTicks(
  window: ZoomWindow,
  maxTicks: number,
  opts: DateTickOptions = {},
): DateTick[] {
  const cal = opts.utc ? UTC : LOCAL;
  const { start, end } = window;
  if (!Number.isFinite(start) || !Number.isFinite(end)) return [];
  const max = Math.max(1, Math.floor(maxTicks));
  const span = end - start;

  if (span <= 0) return [{ t: start, label: dayLabel(start, cal) }];

  if (span <= DAY_TICKS_MAX_SPAN_MS) {
    const first: DateTick = { t: start, label: dayLabel(start, cal) };
    if (dayKey(start, cal) === dayKey(end, cal)) return [first];
    const last: DateTick = { t: end, label: dayLabel(end, cal) };
    if (max === 1) return [first];
    if (max === 2) return [first, last];
    // Interior midnights strictly inside the window, excluding the two dates
    // the ends already name — and anything hugging an end so closely its label
    // would collide with first/last.
    const minGap = span * 0.08;
    const interior: DateTick[] = [];
    let t = cal.make(cal.year(start), cal.month(start), cal.day(start) + 1);
    while (t < end) {
      const key = dayKey(t, cal);
      if (
        key !== dayKey(start, cal) &&
        key !== dayKey(end, cal) &&
        t - start >= minGap &&
        end - t >= minGap
      ) {
        interior.push({ t, label: dayLabel(t, cal) });
      }
      t = cal.make(cal.year(t), cal.month(t), cal.day(t) + 1);
    }
    return [first, ...thin(interior, max - 2), last];
  }

  // Month mode.
  const boundaries: number[] = [];
  let y = cal.year(start);
  let m = cal.month(start);
  let t = cal.make(y, m, 1);
  if (t < start) {
    m += 1;
    t = cal.make(y, m, 1);
  }
  while (t <= end) {
    boundaries.push(t);
    m += 1;
    t = cal.make(y, m, 1); // Date normalises month overflow into the next year
  }
  const kept = thin(boundaries, max);
  const crossesYear = cal.year(start) !== cal.year(end);
  let prevYear: number | null = null;
  return kept.map((b) => {
    const year = cal.year(b);
    const withYear = crossesYear && (prevYear === null || year !== prevYear);
    prevYear = year;
    const name = MONTHS[cal.month(b)];
    return { t: b, label: withYear ? `${name} ${year}` : name };
  });
}
