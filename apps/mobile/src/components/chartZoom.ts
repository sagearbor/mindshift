/**
 * Pure transform math for the heat chart's TIME-AXIS zoom (see HeatChart).
 *
 * The chart only ever zooms the TIME axis — never the heat (y) axis: stretching
 * the heat scale would reintroduce the "flat-line dishonesty" the family just
 * fixed. So zoom state is nothing more than a *visible time window* — a
 * `[start, end]` slice of the recording's seconds — that gets mapped onto the
 * full chart width. Everything time-based (dashes, playhead, connectors) is
 * placed with `secondsToX` against the current window, so it all re-renders
 * consistently as the window changes.
 *
 * Kept free of React / react-native so the window clamping and both-directions
 * coordinate mapping are unit-testable in isolation (mirrors HeatChart's own
 * pure-geometry exports).
 */

/** A visible slice of recording time, in seconds. `start < end`, both within
 *  `[0, duration]`. The full (unzoomed) view is `{ start: 0, end: duration }`. */
export interface ZoomWindow {
  start: number;
  end: number;
}

/** Pixel geometry shared with the chart: the measured SVG width and the inner
 *  padding the marks are inset by (so x maps into `[padding, width - padding]`). */
export interface ChartGeom {
  width: number;
  padding: number;
}

/** Max zoom-in: never show fewer than this many seconds across the full width,
 *  so a pinch can't collapse the window to a meaningless sliver. When the whole
 *  recording is shorter than this, the floor is the recording itself. */
export const MIN_ZOOM_SECONDS = 5;

/** The full, unzoomed window for a recording of `duration` seconds. */
export function fullWindow(duration: number): ZoomWindow {
  return { start: 0, end: Math.max(0, duration) };
}

/** Visible span in seconds. */
export function windowSpan(w: ZoomWindow): number {
  return w.end - w.start;
}

/** True when the window is a strict subset of the full recording — i.e. the
 *  user has zoomed and/or panned in. Drives the "Reset view" affordance. */
export function isZoomed(
  w: ZoomWindow,
  duration: number,
  eps = 1e-3,
): boolean {
  if (!(duration > 0)) return false;
  return w.start > eps || w.end < duration - eps;
}

/**
 * Constrain a proposed window so it's always a valid, on-recording slice:
 *  - span is at least `min(minSpan, duration)` (max zoom-in) and at most
 *    `duration` (max zoom-out is the whole recording);
 *  - the slice sits fully within `[0, duration]` (pans never run off the edge),
 *    preserving the requested span by sliding it back in-bounds.
 */
export function clampWindow(
  w: ZoomWindow,
  duration: number,
  minSpan = MIN_ZOOM_SECONDS,
): ZoomWindow {
  if (!(duration > 0)) return { start: 0, end: 0 };
  const floor = Math.min(minSpan, duration);
  // Clamp the span first (max zoom in / out), then slide the start in-bounds.
  const span = Math.max(floor, Math.min(duration, w.end - w.start));
  let start = w.start;
  if (start + span > duration) start = duration - span;
  if (start < 0) start = 0;
  return { start, end: start + span };
}

/** Map a recording position (seconds) to a pixel x within the given window.
 *  Positions outside the window map outside `[padding, width-padding]` on
 *  purpose — the caller lets the SVG viewport clip them (so off-window dashes
 *  simply aren't drawn). */
export function secondsToX(
  seconds: number,
  w: ZoomWindow,
  geom: ChartGeom,
): number {
  const chartWidth = geom.width - geom.padding * 2;
  const span = w.end - w.start;
  if (span <= 0) return geom.padding;
  const frac = (seconds - w.start) / span;
  return geom.padding + frac * chartWidth;
}

/** Inverse of `secondsToX`: map a pixel x back to a recording position. Used to
 *  anchor cursor-/pinch-centered zoom and to hit-test taps through the current
 *  transform. */
export function xToSeconds(x: number, w: ZoomWindow, geom: ChartGeom): number {
  const chartWidth = geom.width - geom.padding * 2;
  if (chartWidth <= 0) return w.start;
  const span = w.end - w.start;
  return w.start + ((x - geom.padding) / chartWidth) * span;
}

/**
 * Re-window to a new visible span while keeping `focusSec` pinned at the same
 * fractional position it occupied in `startWindow`. This is the heart of
 * cursor-centered wheel zoom and pinch zoom: the point under the cursor / pinch
 * midpoint stays put as the span grows or shrinks. Result is clamped to a valid
 * on-recording slice.
 */
export function windowForZoom(
  startWindow: ZoomWindow,
  focusSec: number,
  newSpan: number,
  duration: number,
  minSpan = MIN_ZOOM_SECONDS,
): ZoomWindow {
  const startSpan = windowSpan(startWindow);
  const frac = startSpan <= 0 ? 0.5 : (focusSec - startWindow.start) / startSpan;
  const start = focusSec - frac * newSpan;
  return clampWindow({ start, end: start + newSpan }, duration, minSpan);
}

/**
 * Zoom the window by a multiplicative `factor` about `focusSec` (the point that
 * stays fixed). `factor < 1` zooms in (smaller span), `factor > 1` zooms out.
 * A convenience wrapper over `windowForZoom` for discrete wheel notches.
 */
export function zoomAt(
  w: ZoomWindow,
  focusSec: number,
  factor: number,
  duration: number,
  minSpan = MIN_ZOOM_SECONDS,
): ZoomWindow {
  return windowForZoom(w, focusSec, windowSpan(w) * factor, duration, minSpan);
}

/** Slide the window by `deltaSec`, preserving its span and clamping at the
 *  recording edges (a pan can never run off the ends or change the zoom level). */
export function panBySeconds(
  w: ZoomWindow,
  deltaSec: number,
  duration: number,
): ZoomWindow {
  const span = windowSpan(w);
  // Pin the span exactly by passing it as the min: clampWindow's span floor then
  // equals the current span, so only the start is slid back in-bounds.
  return clampWindow(
    { start: w.start + deltaSec, end: w.end + deltaSec },
    duration,
    span,
  );
}

/** Where the playhead sits relative to the visible window — so the chart can
 *  show an honest "off-screen →/←" hint (with tap-to-recenter) instead of
 *  silently losing the playhead while zoomed. */
export type PlayheadVisibility = "visible" | "before" | "after";

export function playheadVisibility(
  seconds: number | null | undefined,
  w: ZoomWindow,
): PlayheadVisibility {
  if (seconds == null) return "visible";
  if (seconds < w.start) return "before";
  if (seconds > w.end) return "after";
  return "visible";
}

/** Recenter the window on `focusSec` (keeping the current span), clamped to the
 *  recording. Used when the user taps the "playhead off-screen" hint. */
export function centerWindowOn(
  w: ZoomWindow,
  focusSec: number,
  duration: number,
): ZoomWindow {
  const span = windowSpan(w);
  return clampWindow(
    { start: focusSec - span / 2, end: focusSec + span / 2 },
    duration,
    span,
  );
}
