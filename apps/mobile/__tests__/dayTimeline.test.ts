import {
  addDays,
  clockLabel,
  dateKey,
  dateKeyOfIso,
  dayTitle,
  episodeTimeRange,
  heatColor,
  HEAT_UNKNOWN_COLOR,
  participantsLine,
  recordingsForDay,
} from "../src/screens/dayTimeline";
import type { Episode, RecordingSummary } from "../src/api/client";

function ep(overrides: Partial<Episode> = {}): Episode {
  return {
    index: 0,
    start_time: 0,
    end_time: 60,
    duration_seconds: 60,
    first_turn_index: 0,
    last_turn_index: 3,
    turn_count: 4,
    speakers: ["Speaker A", "Speaker B"],
    participants: ["You", "Alex"],
    mean_heat: 40,
    peak_heat: 62,
    summary: "Morning planning",
    summary_source: "excerpt",
    ...overrides,
  };
}

function rec(id: string, createdAt: string): RecordingSummary {
  return {
    id,
    created_at: createdAt,
    filename: `${id}.m4a`,
    media_type: "audio",
    duration_seconds: 120,
    has_analysis: true,
  };
}

describe("dateKey / addDays / dayTitle", () => {
  it("buckets by the LOCAL calendar day", () => {
    const d = new Date(2026, 6, 14, 23, 30); // local July 14, 11:30 PM
    expect(dateKey(d)).toBe("2026-07-14");
    expect(dateKeyOfIso(d.toISOString())).toBe("2026-07-14");
  });

  it("addDays steps across month boundaries", () => {
    expect(dateKey(addDays(new Date(2026, 6, 1), -1))).toBe("2026-06-30");
    expect(dateKey(addDays(new Date(2026, 6, 31), 1))).toBe("2026-08-01");
  });

  it("dayTitle says Today / Yesterday / a dated label", () => {
    const now = new Date(2026, 6, 14, 9, 0);
    expect(dayTitle(new Date(2026, 6, 14, 22, 0), now)).toBe("Today");
    expect(dayTitle(new Date(2026, 6, 13, 1, 0), now)).toBe("Yesterday");
    // Further back: a real date string, not a relative word.
    const older = dayTitle(new Date(2026, 6, 4), now);
    expect(older).not.toBe("Today");
    expect(older).toMatch(/4/);
  });
});

describe("recordingsForDay", () => {
  it("filters to the local day and sorts oldest-first", () => {
    const morning = new Date(2026, 6, 14, 8, 0).toISOString();
    const evening = new Date(2026, 6, 14, 20, 0).toISOString();
    const otherDay = new Date(2026, 6, 13, 12, 0).toISOString();
    const out = recordingsForDay(
      [rec("b", evening), rec("c", otherDay), rec("a", morning)],
      new Date(2026, 6, 14),
    );
    expect(out.map((r) => r.id)).toEqual(["a", "b"]);
  });
});

describe("heatColor", () => {
  it("maps the calm→strained→rough ramp endpoints", () => {
    expect(heatColor(0)).toBe("#2f9e6e"); // calm green
    expect(heatColor(50)).toBe("#e8a13a"); // strained amber
    expect(heatColor(100)).toBe("#d64545"); // rough red
  });

  it("is neutral gray for unknown heat — never a fake calm green", () => {
    expect(heatColor(null)).toBe(HEAT_UNKNOWN_COLOR);
  });

  it("clamps out-of-range scores", () => {
    expect(heatColor(-10)).toBe(heatColor(0));
    expect(heatColor(250)).toBe(heatColor(100));
  });
});

describe("clock labels", () => {
  const createdAt = new Date(2026, 6, 14, 14, 0).toISOString(); // 2:00 PM local

  it("offsets the episode clock from the recording start", () => {
    // 14:00 + 900s = 14:15 — assert on the minutes so the test is
    // locale-independent (12h vs 24h clock).
    expect(clockLabel(createdAt, 900)).toContain(":15");
    expect(clockLabel(createdAt, null)).toContain(":00");
  });

  it("episodeTimeRange renders a range, or just the start when end unknown", () => {
    const range = episodeTimeRange(createdAt, ep({ start_time: 0, end_time: 900 }));
    expect(range).toContain(":00");
    expect(range).toContain(" – ");
    expect(range).toContain(":15");
    const startOnly = episodeTimeRange(
      createdAt,
      ep({ start_time: 0, end_time: null }),
    );
    expect(startOnly).not.toContain(" – ");
  });
});

describe("participantsLine", () => {
  it("joins display labels and is empty when none are stored", () => {
    expect(participantsLine(ep())).toBe("You, Alex");
    expect(participantsLine(ep({ participants: [] }))).toBe("");
  });
});
