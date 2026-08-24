/**
 * Pure, React-free helpers for Track 2's "how you sound over time" views —
 * the per-day self-tone buckets on GrowthScreen, the per-person rows
 * ("with Mom" vs "with Asher"), the tone chips on YourDay episodes and the
 * Replay/Dynamics/therapist tone cards. Kept out of the components (like
 * dayTimeline.ts / glanceSummary.ts / growthSeries.ts) so the bucketing and
 * wording are unit-testable directly.
 *
 * Honesty rules:
 *   - Every count comes straight from the server's tone buckets (label
 *     counts the phone's classifier produced, escalations the server
 *     derived). Nothing here invents a tone for a turn that had none.
 *   - A day/person with NO scored turns is omitted, never rendered as
 *     "neutral"; a null calm share is an absence, never 100%.
 *   - Days are bucketed by the USER'S local calendar day (dayTimeline's
 *     dateKeyOfIso), same as the YourDay timeline.
 */
import type {
  GrowthPerson,
  GrowthPoint,
  GrowthSelfTone,
  ToneBucket,
} from "../api/client";
import { dateKeyOfIso } from "./dayTimeline";

/** Labels the server treats as the user ESCALATING (live_sessions.py's
 *  ESCALATION_LABELS, mirrored here only for chip coloring — the server's
 *  `escalation_count` is the source of truth for the numbers). */
export const ESCALATION_LABELS = new Set([
  "defensive",
  "sarcastic",
  "frustrated",
  "angry",
  "anger",
  "hostile",
  "contempt",
  "contemptuous",
  "irritated",
  "annoyed",
  "critical",
  "aggressive",
  "dismissive",
  "frustration",
  "defensiveness",
  "sarcasm",
]);

export function isEscalationLabel(label: string): boolean {
  return ESCALATION_LABELS.has(label.trim().toLowerCase());
}

/** Chip colors per tone family, from the house heat ramp: warm/happy =
 *  calm green, escalations = rough red, sad = a quiet blue, everything
 *  else (neutral, unknown) = gray. Background/foreground pairs so a chip
 *  stays legible on the white cards. */
export function toneChipColors(label: string): { bg: string; fg: string } {
  const l = label.trim().toLowerCase();
  if (isEscalationLabel(l)) return { bg: "#FEECEC", fg: "#B42318" };
  if (l === "warm" || l === "happy" || l === "warmth") return { bg: "#E7F6EE", fg: "#1B7A4B" };
  if (l === "sad" || l === "sadness") return { bg: "#EAF2FB", fg: "#2F5F9E" };
  return { bg: "#F3F4F6", fg: "#4B5563" };
}

export interface LabelCount {
  label: string;
  count: number;
}

/** The label counts as a list, most frequent first (ties: alphabetical, so
 *  the order is stable across renders). `limit` trims to the top N. */
export function topLabels(
  labels: Record<string, number> | null | undefined,
  limit = 3,
): LabelCount[] {
  if (!labels) return [];
  return Object.entries(labels)
    .filter(([, n]) => typeof n === "number" && n > 0)
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))
    .slice(0, limit);
}

/** "mostly warm · 2 escalations" — one line for a bucket. Null when the
 *  bucket has no scored turns (the caller then omits the line rather than
 *  saying anything). */
export function describeBucket(
  labels: Record<string, number> | null | undefined,
  escalationCount: number,
  scoredTurns: number,
): string | null {
  if (scoredTurns <= 0) return null;
  const top = topLabels(labels, 1)[0];
  const parts: string[] = [];
  if (top) parts.push(`mostly ${top.label}`);
  parts.push(
    escalationCount === 0
      ? "no escalations"
      : `${escalationCount} escalation${escalationCount === 1 ? "" : "s"}`,
  );
  return parts.join(" · ");
}

/** Share (0–100) of the user's scored turns that did NOT escalate — the
 *  "calm share" sparkline value. Null when nothing was scored. */
export function calmShare(
  scoredTurns: number,
  escalationCount: number,
): number | null {
  if (scoredTurns <= 0) return null;
  const calm = Math.max(0, scoredTurns - escalationCount);
  return Math.round((calm / scoredTurns) * 100);
}

/** One local calendar day's self-tone bucket, summed over every live
 *  session that day which carried tone. */
export interface DayTone {
  key: string; // "YYYY-MM-DD" local
  // The earliest session timestamp that day — for a display label.
  timestamp: string;
  sessions: number;
  scored_turns: number;
  labels: Record<string, number>;
  escalation_count: number;
}

function addLabels(into: Record<string, number>, from: Record<string, number>) {
  for (const [label, n] of Object.entries(from)) {
    if (typeof n === "number" && n > 0) into[label] = (into[label] ?? 0) + n;
  }
}

/**
 * Bucket growth points by the user's local day, keeping ONLY points that
 * carry `self_tone` with at least one scored turn (an upload, or a live
 * session whose phone sent no tone, contributes nothing — not a neutral
 * day). Ascending by day.
 */
export function bucketToneByDay(points: GrowthPoint[]): DayTone[] {
  const byKey = new Map<string, DayTone>();
  for (const p of points) {
    const tone: GrowthSelfTone | null | undefined = p.self_tone;
    if (!tone || !(tone.scored_turns > 0)) continue;
    const ts = Date.parse(p.timestamp);
    if (!Number.isFinite(ts)) continue;
    const key = dateKeyOfIso(p.timestamp);
    let day = byKey.get(key);
    if (!day) {
      day = {
        key,
        timestamp: p.timestamp,
        sessions: 0,
        scored_turns: 0,
        labels: {},
        escalation_count: 0,
      };
      byKey.set(key, day);
    }
    if (Date.parse(day.timestamp) > ts) day.timestamp = p.timestamp;
    day.sessions += 1;
    day.scored_turns += tone.scored_turns;
    day.escalation_count += tone.escalation_count;
    addLabels(day.labels, tone.labels ?? {});
  }
  return Array.from(byKey.values()).sort((a, b) => a.key.localeCompare(b.key));
}

/** Short local date label for a day bucket ("Aug 24"). */
export function dayLabel(day: DayTone): string {
  return new Date(day.timestamp).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

/** A person row's display name: the real identity when known, else the
 *  raw speaker label the server carried (never blank). */
export function personName(
  p: { display_name?: string | null; person_id?: string | null; speaker?: string | null },
): string {
  const name = (p.display_name ?? "").trim();
  if (name) return name;
  const raw = (p.speaker ?? "").trim();
  if (raw) return raw;
  return (p.person_id ?? "").trim() || "Someone";
}

/** Cross-session person rows worth showing: at least one scored turn, most
 *  sessions first (the server's order), name resolved. */
export function peopleRows(
  people: GrowthPerson[] | null | undefined,
): (GrowthPerson & { name: string; summary: string | null })[] {
  return (people ?? [])
    .filter((p) => p.scored_turns > 0)
    .map((p) => ({
      ...p,
      name: personName(p),
      summary: describeBucket(p.labels, p.escalation_count, p.scored_turns),
    }));
}

/** The one-line episode chip: "you: warm ×2, frustrated ×1 · 1 escalation".
 *  Null when the episode carries no self tone (the chip is omitted). */
export function episodeToneLine(
  labels: Record<string, number> | null | undefined,
  escalationCount: number | null | undefined,
): string | null {
  const top = topLabels(labels, 3);
  if (top.length === 0) return null;
  const chips = top.map((t) => `${t.label} ×${t.count}`).join(", ");
  const esc = escalationCount ?? 0;
  const tail =
    esc > 0 ? ` · ${esc} escalation${esc === 1 ? "" : "s"}` : "";
  return `you: ${chips}${tail}`;
}

/** The mean-score line for a self bucket ("warmth 50 · frustration 42"),
 *  listing only the dimensions that were actually scored. Null when none. */
export function meanLine(bucket: Pick<ToneBucket, "mean"> | null | undefined): string | null {
  if (!bucket?.mean) return null;
  const parts = Object.entries(bucket.mean)
    .filter(([, v]) => typeof v === "number")
    .map(([dim, v]) => `${dim} ${Math.round(v as number)}`);
  return parts.length > 0 ? parts.join(" · ") : null;
}

/** Human label for a live coaching mode. */
export function modeLabel(mode: string | null | undefined): string | null {
  switch (mode) {
    case "earpiece":
      return "Earpiece";
    case "speaker":
      return "Speaker";
    case "therapist":
      return "Therapist";
    default:
      return null;
  }
}
