/**
 * Listening-time countdown helpers (pure; the Live Coach pill and its
 * notices are built on these). The server states the BINDING allowance in
 * seconds on GET /plan (`listening`); the app ticks it down locally while a
 * session runs and re-syncs every few minutes.
 */

export type Tone = "ok" | "amber" | "red" | "out";

export const AMBER_SEC = 600; // 10 min
export const RED_SEC = 300; // 5 min
/** Gentle notices fire once each as the clock crosses these, in seconds. */
export const NOTICE_THRESHOLDS = [600, 300, 60] as const;
/** How often the pill re-syncs with the server while a session runs. */
export const RESYNC_MS = 3 * 60 * 1000;

export type Period = "lifetime" | "month" | "day";

export function toneFor(sec: number): Tone {
  if (sec <= 0) return "out";
  if (sec <= RED_SEC) return "red";
  if (sec <= AMBER_SEC) return "amber";
  return "ok";
}

/** "42 min", "1 h 05 min", "45 s". */
export function formatLeft(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  if (s < 60) return `${s} s`;
  const totalMin = Math.ceil(s / 60);
  if (totalMin < 120) return `${totalMin} min`;
  const h = Math.floor(totalMin / 60);
  const m = totalMin % 60;
  return `${h} h ${String(m).padStart(2, "0")} min`;
}

/**
 * The pill text. Free (lifetime) users see minutes; paid users on a monthly
 * allowance see hours left this month; a daily backstop reads "left today".
 */
export function pillText(sec: number, period: Period, isFree: boolean): string {
  if (sec <= 0) return isFree ? "Out of free time" : "Out of listening time";
  if (isFree || period === "lifetime") return `Free time left: ${formatLeft(sec)}`;
  if (period === "month") return `${(sec / 3600).toFixed(1)} h left this month`;
  return `${formatLeft(sec)} left today`;
}

/** The smallest threshold crossed going from `prev` down to `cur`, else null. */
export function noticeCrossed(prev: number, cur: number): number | null {
  let hit: number | null = null;
  for (const t of NOTICE_THRESHOLDS) {
    if (prev > t && cur <= t) hit = t;
  }
  return hit;
}

export function noticeCopy(threshold: number, isFree: boolean): string {
  const what = isFree ? "free time" : "listening time";
  if (threshold >= 120) return `About ${Math.round(threshold / 60)} minutes of ${what} left.`;
  return `About a minute of ${what} left. Coaching will pause when it runs out; your notes stay safe.`;
}

export const OUT_OF_TIME_MESSAGE =
  "You're out of free time. Cloud coaching has paused; everything you've saved is safe and on-device features keep working. Redeem a code or upgrade to keep going.";
