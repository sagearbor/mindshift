/**
 * One shared, honest way to render "when did this happen" across every screen
 * that opens a stored item (recordings list, replay header, Your Day rows).
 *
 * The house standard is absolute + unambiguous + friendly:
 *   "Sat, Jul 19 · 2:41 PM"     (this year — the year is implied, so omit it)
 *   "Sat, Jul 19, 2024 · 2:41 PM" (an earlier year — include it, no ambiguity)
 *
 * Timestamps always come from a recording's real `created_at` (plus, for
 * episodes, a real transcript offset). We NEVER fabricate: a missing or
 * unparseable timestamp returns `null` so the caller renders nothing rather than
 * a guessed date. Formatting uses the device locale/timezone (like the existing
 * dayTimeline helpers), so a user reads the time in their own wall clock.
 */

/** Parse an ISO timestamp to a Date, or null when it's missing/invalid. */
function parseIso(iso: string | null | undefined): Date | null {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

/** The absolute, friendly date+time label for an ISO timestamp — the app-wide
 *  standard. Returns null when there's no honest timestamp to show (missing or
 *  unparseable `created_at`), so callers omit the line entirely. `now` is
 *  injectable for deterministic tests. */
export function formatDateTime(
  iso: string | null | undefined,
  now: Date = new Date(),
): string | null {
  const d = parseIso(iso);
  if (!d) return null;
  return `${formatDatePart(d, now)} · ${formatTimePart(d)}`;
}

/** Just the date portion ("Sat, Jul 19", or with the year for earlier years).
 *  Null for a missing/invalid timestamp. */
export function formatDate(
  iso: string | null | undefined,
  now: Date = new Date(),
): string | null {
  const d = parseIso(iso);
  return d ? formatDatePart(d, now) : null;
}

/** Just the wall-clock time ("2:41 PM"). Null for a missing/invalid timestamp. */
export function formatTimeOfDay(iso: string | null | undefined): string | null {
  const d = parseIso(iso);
  return d ? formatTimePart(d) : null;
}

/** Weekday + month + day, adding the year only when it isn't the current one. */
function formatDatePart(d: Date, now: Date): string {
  const sameYear = d.getFullYear() === now.getFullYear();
  return d.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    ...(sameYear ? {} : { year: "numeric" }),
  });
}

function formatTimePart(d: Date): string {
  return d.toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}
