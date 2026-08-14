import type { GrowthPoint } from "../src/api/client";
import {
  TREND_MIN_POINTS,
  TREND_WINDOW,
  filterPoints,
  hasUnidentifiedPartner,
  movingAverage,
  partnerNames,
  scoreToY,
  scoredPoints,
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
