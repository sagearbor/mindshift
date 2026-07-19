import {
  MIN_ZOOM_SECONDS,
  fullWindow,
  windowSpan,
  isZoomed,
  clampWindow,
  secondsToX,
  xToSeconds,
  windowForZoom,
  zoomAt,
  panBySeconds,
  playheadVisibility,
  centerWindowOn,
  type ZoomWindow,
} from "../src/components/chartZoom";

const GEOM = { width: 300, padding: 16 }; // chartWidth = 268

describe("fullWindow / windowSpan / isZoomed", () => {
  it("fullWindow spans the whole recording", () => {
    expect(fullWindow(100)).toEqual({ start: 0, end: 100 });
    expect(windowSpan(fullWindow(100))).toBe(100);
    // A non-positive duration collapses to an empty window (no crash downstream).
    expect(fullWindow(0)).toEqual({ start: 0, end: 0 });
    expect(fullWindow(-5)).toEqual({ start: 0, end: 0 });
  });

  it("isZoomed is false for the full view and true for any strict subset", () => {
    expect(isZoomed({ start: 0, end: 100 }, 100)).toBe(false);
    expect(isZoomed({ start: 10, end: 100 }, 100)).toBe(true); // panned/zoomed from the left
    expect(isZoomed({ start: 0, end: 50 }, 100)).toBe(true); // zoomed from the right
    expect(isZoomed({ start: 10, end: 50 }, 100)).toBe(true);
    // A recording with no duration can't be zoomed.
    expect(isZoomed({ start: 0, end: 0 }, 0)).toBe(false);
  });
});

describe("clampWindow", () => {
  it("keeps a valid in-bounds window unchanged", () => {
    expect(clampWindow({ start: 20, end: 60 }, 100)).toEqual({ start: 20, end: 60 });
  });

  it("caps the span at the full duration (max zoom-out)", () => {
    expect(clampWindow({ start: -10, end: 200 }, 100)).toEqual({ start: 0, end: 100 });
  });

  it("floors the span at MIN_ZOOM_SECONDS (max zoom-in)", () => {
    const w = clampWindow({ start: 10, end: 11 }, 100);
    expect(windowSpan(w)).toBe(MIN_ZOOM_SECONDS);
    expect(w.start).toBe(10);
  });

  it("slides an over-the-edge window back in-bounds, preserving its span", () => {
    const w = clampWindow({ start: 98, end: 103 }, 100);
    expect(w).toEqual({ start: 95, end: 100 });
  });

  it("never demands a span larger than the recording (short recording)", () => {
    // A 3s recording can't show 5s: the floor becomes the recording itself.
    expect(clampWindow({ start: 0, end: 10 }, 3)).toEqual({ start: 0, end: 3 });
  });

  it("returns an empty window for a non-positive duration", () => {
    expect(clampWindow({ start: 5, end: 9 }, 0)).toEqual({ start: 0, end: 0 });
  });
});

describe("secondsToX / xToSeconds", () => {
  it("maps the window edges to the chart edges", () => {
    const w = { start: 20, end: 40 };
    expect(secondsToX(20, w, GEOM)).toBeCloseTo(16); // left padding
    expect(secondsToX(40, w, GEOM)).toBeCloseTo(16 + 268); // right edge
    expect(secondsToX(30, w, GEOM)).toBeCloseTo(16 + 134); // midpoint
  });

  it("does NOT clamp positions outside the window (viewport clips them)", () => {
    const w = { start: 20, end: 40 };
    // 50s is past the window end → maps beyond the right edge, on purpose.
    expect(secondsToX(50, w, GEOM)).toBeGreaterThan(16 + 268);
    // 10s is before the start → maps left of the padding.
    expect(secondsToX(10, w, GEOM)).toBeLessThan(16);
  });

  it("round-trips seconds <-> x for any window", () => {
    const w = { start: 12.5, end: 47.5 };
    for (const s of [12.5, 20, 33, 47.5]) {
      expect(xToSeconds(secondsToX(s, w, GEOM), w, GEOM)).toBeCloseTo(s);
    }
  });
});

describe("windowForZoom (focus-anchored re-window)", () => {
  it("keeps the focus point at the same fractional position", () => {
    const start: ZoomWindow = { start: 0, end: 100 };
    const w = windowForZoom(start, 50, 20, 100);
    expect(w).toEqual({ start: 40, end: 60 }); // 50 was at frac 0.5 → stays centered
  });

  it("anchors an off-center focus", () => {
    const w = windowForZoom({ start: 0, end: 100 }, 10, 20, 100);
    // 10 was at frac 0.1 → start = 10 - 0.1*20 = 8
    expect(w).toEqual({ start: 8, end: 28 });
  });

  it("clamps the resulting span to the zoom floor", () => {
    const w = windowForZoom({ start: 0, end: 100 }, 50, 2, 100);
    expect(windowSpan(w)).toBe(MIN_ZOOM_SECONDS);
  });
});

describe("zoomAt", () => {
  it("zooms in with factor < 1, keeping the focus x fixed", () => {
    const before = { start: 0, end: 100 };
    const after = zoomAt(before, 50, 0.5, 100);
    expect(windowSpan(after)).toBe(50);
    // The focus second lands at the SAME pixel x before and after.
    expect(secondsToX(50, after, GEOM)).toBeCloseTo(secondsToX(50, before, GEOM));
  });

  it("zooms out with factor > 1", () => {
    const after = zoomAt({ start: 25, end: 75 }, 50, 1.2, 100);
    expect(windowSpan(after)).toBeCloseTo(60);
    expect(secondsToX(50, after, GEOM)).toBeCloseTo(150); // stays centered
  });

  it("can't zoom out past the full recording", () => {
    expect(zoomAt({ start: 0, end: 100 }, 50, 4, 100)).toEqual({ start: 0, end: 100 });
  });
});

describe("panBySeconds", () => {
  it("slides the window, preserving its span", () => {
    expect(panBySeconds({ start: 20, end: 40 }, 10, 100)).toEqual({ start: 30, end: 50 });
  });

  it("clamps at the recording start, keeping the span", () => {
    expect(panBySeconds({ start: 20, end: 40 }, -30, 100)).toEqual({ start: 0, end: 20 });
  });

  it("clamps at the recording end, keeping the span", () => {
    expect(panBySeconds({ start: 80, end: 100 }, 50, 100)).toEqual({ start: 80, end: 100 });
  });

  it("a full-view window has nowhere to pan", () => {
    expect(panBySeconds({ start: 0, end: 100 }, 25, 100)).toEqual({ start: 0, end: 100 });
  });
});

describe("playheadVisibility", () => {
  const w = { start: 20, end: 40 };
  it("classifies before / after / visible", () => {
    expect(playheadVisibility(10, w)).toBe("before");
    expect(playheadVisibility(50, w)).toBe("after");
    expect(playheadVisibility(30, w)).toBe("visible");
  });
  it("treats the edges as visible and a null position as visible", () => {
    expect(playheadVisibility(20, w)).toBe("visible");
    expect(playheadVisibility(40, w)).toBe("visible");
    expect(playheadVisibility(null, w)).toBe("visible");
    expect(playheadVisibility(undefined, w)).toBe("visible");
  });
});

describe("centerWindowOn", () => {
  it("recenters on the focus, keeping the span", () => {
    expect(centerWindowOn({ start: 20, end: 40 }, 60, 100)).toEqual({ start: 50, end: 70 });
  });
  it("clamps to the recording edge when the focus is near the end", () => {
    expect(centerWindowOn({ start: 20, end: 40 }, 95, 100)).toEqual({ start: 80, end: 100 });
  });
});
