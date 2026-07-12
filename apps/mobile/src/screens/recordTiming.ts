// Pure timing helpers for the in-app recorder. Deliberately free of ANY
// native imports so both the platform gate (RecordScreen) and the native
// implementation can share them without dragging expo-media-library into the
// web bundle (a static re-export of these from the native file once blanked
// the whole web app at load).

/** Hard cap on an in-app recording. Owner decision: 10 minutes, auto-stopped, so
 *  files stay small and predictable (paired with the 480p quality preset). */
export const MAX_RECORDING_SECONDS = 600;

/**
 * Seconds remaining before the hard cap, floored and never negative. Pure and
 * exported so the cap logic is unit-testable without the camera. `elapsed` is a
 * float (the ticking clock); the cap defaults to MAX_RECORDING_SECONDS.
 */
export function remainingSeconds(
  elapsed: number,
  cap: number = MAX_RECORDING_SECONDS,
): number {
  return Math.max(0, cap - Math.floor(elapsed));
}

/**
 * Whether the elapsed time has reached the cap (so recording must auto-stop).
 * Pure and exported for unit testing.
 */
export function isAtCap(
  elapsed: number,
  cap: number = MAX_RECORDING_SECONDS,
): boolean {
  return Math.floor(elapsed) >= cap;
}

/** Format a whole-second count as m:ss (e.g. 65 → "1:05"). Pure/exported. */
export function formatClock(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${rem.toString().padStart(2, "0")}`;
}
