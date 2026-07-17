/**
 * Pure helpers for the "Your Day" timeline screen (Companion P1) — date
 * bucketing, clock formatting, and the heat→color ramp. Kept free of React so
 * the day/episode math is unit-testable exactly like recordTiming.ts.
 */
import type { Episode, RecordingSummary } from "../api/client";

/** A recording + its episodes, ready for the timeline. `episodes` is null for
 *  a stored-but-unanalyzed recording (rendered honestly as "not analyzed"). */
export interface DayEntry {
  recording: RecordingSummary;
  episodes: Episode[] | null;
}

/** Local-calendar-day key ("YYYY-MM-DD") for a Date. The timeline buckets by
 *  the USER'S local day — a 11:30 PM conversation belongs to that evening, not
 *  to the UTC morning after. */
export function dateKey(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/** Local-day key for an ISO timestamp (a recording's `created_at`). */
export function dateKeyOfIso(iso: string): string {
  return dateKey(new Date(iso));
}

/** A new Date `n` days after `d` (n may be negative), at the same wall time. */
export function addDays(d: Date, n: number): Date {
  const out = new Date(d);
  out.setDate(out.getDate() + n);
  return out;
}

/** Human title for the selected day: "Today" / "Yesterday" / a local date
 *  with weekday. `now` is injectable for deterministic tests. */
export function dayTitle(d: Date, now: Date = new Date()): string {
  const key = dateKey(d);
  if (key === dateKey(now)) return "Today";
  if (key === dateKey(addDays(now, -1))) return "Yesterday";
  return d.toLocaleDateString(undefined, {
    weekday: "long",
    month: "short",
    day: "numeric",
  });
}

/** The recordings that belong to one local day, oldest first (a timeline reads
 *  top-to-bottom through the day). */
export function recordingsForDay(
  recordings: RecordingSummary[],
  day: Date,
): RecordingSummary[] {
  const key = dateKey(day);
  return recordings
    .filter((r) => dateKeyOfIso(r.created_at) === key)
    .sort((a, b) => a.created_at.localeCompare(b.created_at));
}

/** Wall-clock label ("2:14 PM") for `offsetSeconds` into a recording that
 *  started at `createdAtIso`. Null offset (no transcript timing) → the
 *  recording's own start time — the closest honest anchor we have. */
export function clockLabel(
  createdAtIso: string,
  offsetSeconds: number | null,
): string {
  const t = new Date(createdAtIso).getTime() + (offsetSeconds ?? 0) * 1000;
  return new Date(t).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}

/** "2:14 PM – 2:31 PM" for an episode, or just the start label when the end
 *  is unknown. */
export function episodeTimeRange(createdAtIso: string, ep: Episode): string {
  const start = clockLabel(createdAtIso, ep.start_time);
  if (ep.end_time === null || ep.end_time === ep.start_time) return start;
  return `${start} – ${clockLabel(createdAtIso, ep.end_time)}`;
}

// Heat ribbon ramp: calm green → strained amber → rough red, matching the
// house heat vocabulary (0–100). Interpolated in two linear segments so the
// ribbon's intensity tracks the score continuously.
const CALM: [number, number, number] = [0x2f, 0x9e, 0x6e]; // green
const STRAINED: [number, number, number] = [0xe8, 0xa1, 0x3a]; // amber
const ROUGH: [number, number, number] = [0xd6, 0x45, 0x45]; // red
// No aligned heats stored (old/degraded analysis) → a neutral gray, never a
// fake "calm" green.
export const HEAT_UNKNOWN_COLOR = "#9CA3AF";

function lerp(
  a: [number, number, number],
  b: [number, number, number],
  t: number,
): string {
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return `#${c.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

/** Ribbon color for a heat score 0–100; gray for null (heat unknown). */
export function heatColor(heat: number | null): string {
  if (heat === null || Number.isNaN(heat)) return HEAT_UNKNOWN_COLOR;
  const h = Math.max(0, Math.min(100, heat));
  return h <= 50 ? lerp(CALM, STRAINED, h / 50) : lerp(STRAINED, ROUGH, (h - 50) / 50);
}

/** "You, Alex" / "You" — the episode's participant line. Empty string when the
 *  stored analysis carried no speakers (never invented names). */
export function participantsLine(ep: Episode): string {
  return ep.participants.join(", ");
}
