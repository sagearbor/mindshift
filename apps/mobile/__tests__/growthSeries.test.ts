import type { GrowthPoint } from "../src/api/client";
import {
  DAY_TICKS_MAX_SPAN_MS,
  TREND_MIN_POINTS,
  TREND_WINDOW,
  dateTicks,
  filterPoints,
  hasUnidentifiedPartner,
  movingAverage,
  partnerNames,
  scoreToY,
  scoredPoints,
  timeMsToX,
  timeToX,
  timeWindow,
} from "../src/components/growthSeries";

function pt(overrides: Partial<GrowthPoint> = {}): GrowthPoint {
  return {
    recording_id: "r1",
    timestamp: "2026-07-01T12:00:00+00:00",
    title: "A talk",
    my_score: 70,
    partner_names: [],
    ...overrides,
  };
}

describe("scoredPoints — null scores are gaps, never zeros", () => {
  it("drops null-score points and keeps scored ones", () => {
    const points = [
      pt({ recording_id: "a", my_score: 60 }),
      pt({ recording_id: "b", my_score: null }),
      pt({ recording_id: "c", my_score: 0 }),
    ];
    expect(scoredPoints(points).map((p) => p.recording_id)).toEqual(["a", "c"]);
  });
});

describe("timeWindow / timeToX — a real time axis", () => {
  it("spans the earliest to latest timestamp", () => {
    const w = timeWindow([
      pt({ timestamp: "2026-07-10T00:00:00Z" }),
      pt({ timestamp: "2026-07-01T00:00:00Z" }),
      pt({ timestamp: "2026-07-05T00:00:00Z" }),
    ]);
    expect(w).not.toBeNull();
    expect(w!.start).toBe(Date.parse("2026-07-01T00:00:00Z"));
    expect(w!.end).toBe(Date.parse("2026-07-10T00:00:00Z"));
  });

  it("is null when nothing parses", () => {
    expect(timeWindow([])).toBeNull();
    expect(timeWindow([pt({ timestamp: "not a date" })])).toBeNull();
  });

  it("positions points proportionally to TIME, not index", () => {
    const w = {
      start: Date.parse("2026-07-01T00:00:00Z"),
      end: Date.parse("2026-07-11T00:00:00Z"),
    };
    const geom = { width: 110, padding: 5 };
    // 1 day into a 10-day span → 10% across the 100px chart area.
    expect(timeToX("2026-07-02T00:00:00Z", w, geom)).toBeCloseTo(15);
    expect(timeToX("2026-07-11T00:00:00Z", w, geom)).toBeCloseTo(105);
  });

  it("centers a single point (zero-span window)", () => {
    const t = Date.parse("2026-07-01T00:00:00Z");
    expect(
      timeToX("2026-07-01T00:00:00Z", { start: t, end: t }, {
        width: 100,
        padding: 4,
      }),
    ).toBe(50);
  });
});

describe("scoreToY", () => {
  it("puts 100 at the top pad and 0 at the bottom pad", () => {
    expect(scoreToY(100, 50, 5)).toBe(5);
    expect(scoreToY(0, 50, 5)).toBe(45);
    expect(scoreToY(50, 50, 5)).toBe(25);
  });
});

describe("movingAverage", () => {
  it("trails over the window and starts from the first value", () => {
    expect(movingAverage([10, 20, 30, 40], 3)).toEqual([10, 15, 20, 30]);
  });

  it("default window matches TREND_WINDOW", () => {
    expect(TREND_WINDOW).toBe(3);
    expect(TREND_MIN_POINTS).toBe(5);
    expect(movingAverage([6, 6, 6, 6, 6])).toEqual([6, 6, 6, 6, 6]);
  });
});

describe("partner filter", () => {
  const points = [
    pt({ recording_id: "a", partner_names: ["Linda"] }),
    pt({ recording_id: "b", partner_names: [] }),
    pt({ recording_id: "c", partner_names: ["Sam", "Linda"] }),
  ];

  it("collects distinct names in first-seen order", () => {
    expect(partnerNames(points)).toEqual(["Linda", "Sam"]);
  });

  it("flags the honest unidentified bucket", () => {
    expect(hasUnidentifiedPartner(points)).toBe(true);
    expect(hasUnidentifiedPartner([points[0]])).toBe(false);
  });

  it("filters all / by partner / unidentified", () => {
    expect(filterPoints(points, { kind: "all" })).toHaveLength(3);
    expect(
      filterPoints(points, { kind: "partner", name: "Linda" }).map(
        (p) => p.recording_id,
      ),
    ).toEqual(["a", "c"]);
    expect(
      filterPoints(points, { kind: "unidentified" }).map((p) => p.recording_id),
    ).toEqual(["b"]);
  });
});

describe("dateTicks — real calendar boundaries on the time axis", () => {
  const utc = { utc: true };
  const w = (a: string, b: string) => ({ start: Date.parse(a), end: Date.parse(b) });

  it("short span: day labels, always including the first and last dates", () => {
    const ticks = dateTicks(
      w("2026-08-20T12:00:00Z", "2026-08-24T12:00:00Z"),
      5,
      utc,
    );
    expect(ticks.map((t) => t.label)).toEqual([
      "Aug 20", "Aug 21", "Aug 22", "Aug 23", "Aug 24",
    ]);
    // First/last sit at the window's own ends; interior ticks at midnight.
    expect(ticks[0].t).toBe(Date.parse("2026-08-20T12:00:00Z"));
    expect(ticks[4].t).toBe(Date.parse("2026-08-24T12:00:00Z"));
    expect(ticks[1].t).toBe(Date.UTC(2026, 7, 21));
    // Ticks are ascending, so they map left→right like the dots.
    for (let i = 1; i < ticks.length; i++) expect(ticks[i].t).toBeGreaterThan(ticks[i - 1].t);
  });

  it("short span thinned to maxTicks keeps first and last", () => {
    const ticks = dateTicks(
      w("2026-08-01T00:00:00Z", "2026-08-10T00:00:00Z"),
      4,
      utc,
    );
    expect(ticks.length).toBeLessThanOrEqual(4);
    expect(ticks[0].label).toBe("Aug 1");
    expect(ticks[ticks.length - 1].label).toBe("Aug 10");
  });

  it("a window inside one calendar day gets that single day", () => {
    expect(
      dateTicks(w("2026-08-20T09:00:00Z", "2026-08-20T21:00:00Z"), 5, utc),
    ).toEqual([{ t: Date.parse("2026-08-20T09:00:00Z"), label: "Aug 20" }]);
  });

  it("multi-month span: month labels on first-of-month boundaries", () => {
    const ticks = dateTicks(
      w("2026-06-15T00:00:00Z", "2026-09-20T00:00:00Z"),
      6,
      utc,
    );
    expect(ticks.map((t) => t.label)).toEqual(["Jul", "Aug", "Sep"]);
    expect(ticks[0].t).toBe(Date.UTC(2026, 6, 1));
  });

  it("names the year on the first tick and where it changes when the window crosses a year", () => {
    const ticks = dateTicks(
      w("2026-11-15T00:00:00Z", "2027-02-10T00:00:00Z"),
      6,
      utc,
    );
    expect(ticks.map((t) => t.label)).toEqual(["Dec 2026", "Jan 2027", "Feb"]);
  });

  it("zero span: a single day label at the point itself", () => {
    const t = Date.parse("2026-08-20T15:30:00Z");
    expect(dateTicks({ start: t, end: t }, 5, utc)).toEqual([
      { t, label: "Aug 20" },
    ]);
  });

  it("never exceeds maxTicks, whatever the span", () => {
    const spans: Array<[string, string]> = [
      ["2026-08-20T00:00:00Z", "2026-08-21T00:00:00Z"],
      ["2026-08-01T00:00:00Z", "2026-09-15T00:00:00Z"],
      ["2024-01-01T00:00:00Z", "2026-08-20T00:00:00Z"],
      ["2016-01-01T00:00:00Z", "2026-08-20T00:00:00Z"],
    ];
    for (const [a, b] of spans) {
      for (const max of [1, 2, 3, 5, 8]) {
        const ticks = dateTicks(w(a, b), max, utc);
        expect(ticks.length).toBeLessThanOrEqual(max);
        expect(ticks.length).toBeGreaterThan(0);
      }
    }
  });

  it("switches from day to month labels past the day-span ceiling", () => {
    const start = Date.parse("2026-03-01T00:00:00Z");
    const days = dateTicks({ start, end: start + DAY_TICKS_MAX_SPAN_MS }, 4, utc);
    expect(days[0].label).toBe("Mar 1");
    const months = dateTicks(
      { start, end: start + DAY_TICKS_MAX_SPAN_MS + 1 },
      4,
      utc,
    );
    expect(months.map((t) => t.label)).toEqual(["Mar", "Apr"]);
  });

  it("ticks map with the same geometry as the dots (timeMsToX)", () => {
    const win = w("2026-07-01T00:00:00Z", "2026-07-11T00:00:00Z");
    const geom = { width: 110, padding: 5 };
    expect(timeMsToX(Date.parse("2026-07-02T00:00:00Z"), win, geom)).toBeCloseTo(
      timeToX("2026-07-02T00:00:00Z", win, geom),
    );
    // Zero-span window centers, like the single dot.
    const t = win.start;
    expect(timeMsToX(t, { start: t, end: t }, geom)).toBe(55);
  });
});
