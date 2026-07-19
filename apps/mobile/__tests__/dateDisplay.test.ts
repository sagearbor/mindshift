import {
  formatDateTime,
  formatDate,
  formatTimeOfDay,
} from "../src/utils/dateDisplay";

// Build ISO strings from LOCAL Dates so device-locale/timezone formatting is
// deterministic (the same trick the dayTimeline tests use). Assertions check
// year presence and the " · " / ":" structure rather than locale-specific
// weekday/month strings, so they hold under any locale.
const iso = (y: number, m0: number, d: number, h: number, min: number) =>
  new Date(y, m0, d, h, min).toISOString();

describe("formatDateTime", () => {
  const now = new Date(2026, 6, 19, 9, 0); // Sun Jul 19 2026, local

  it("renders an absolute date + wall-clock time joined by ' · '", () => {
    const out = formatDateTime(iso(2026, 6, 19, 14, 41), now)!;
    expect(out).toContain(" · ");
    expect(out).toContain(":"); // the time part
    // Date part comes first, time part after the separator.
    const [datePart, timePart] = out.split(" · ");
    expect(datePart.length).toBeGreaterThan(0);
    expect(timePart).toContain(":");
  });

  it("omits the year for a timestamp in the current year", () => {
    const out = formatDateTime(iso(2026, 6, 19, 14, 41), now)!;
    expect(out).not.toContain("2026");
  });

  it("includes the year for an earlier year (no ambiguity)", () => {
    const out = formatDateTime(iso(2024, 1, 3, 8, 5), now)!;
    expect(out).toContain("2024");
  });

  it("returns null for a missing or unparseable timestamp (never a guess)", () => {
    expect(formatDateTime(null, now)).toBeNull();
    expect(formatDateTime(undefined, now)).toBeNull();
    expect(formatDateTime("", now)).toBeNull();
    expect(formatDateTime("not-a-date", now)).toBeNull();
  });
});

describe("formatDate / formatTimeOfDay", () => {
  const now = new Date(2026, 6, 19, 9, 0);

  it("formatDate returns just the date part (no time separator/colon)", () => {
    const out = formatDate(iso(2026, 6, 19, 14, 41), now)!;
    expect(out).not.toContain(" · ");
    expect(out).not.toContain(":");
    expect(out.length).toBeGreaterThan(0);
  });

  it("formatTimeOfDay returns just the wall-clock time", () => {
    const out = formatTimeOfDay(iso(2026, 6, 19, 14, 41))!;
    expect(out).toContain(":");
    expect(out).not.toContain(" · ");
  });

  it("both return null for a bad timestamp", () => {
    expect(formatDate("nope", now)).toBeNull();
    expect(formatTimeOfDay("")).toBeNull();
  });
});
