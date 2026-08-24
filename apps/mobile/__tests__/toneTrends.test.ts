/** toneTrends.ts — the pure helpers behind "How you sound" (Track 2). */
import type { GrowthPoint } from "../src/api/client";
import { dateKeyOfIso } from "../src/screens/dayTimeline";
import {
  bucketToneByDay,
  calmShare,
  describeBucket,
  episodeToneLine,
  isEscalationLabel,
  meanLine,
  modeLabel,
  peopleRows,
  personName,
  toneChipColors,
  topLabels,
} from "../src/screens/toneTrends";

function pt(overrides: Partial<GrowthPoint> = {}): GrowthPoint {
  return {
    recording_id: "r",
    timestamp: "2026-08-24T18:05:00+00:00",
    title: "t",
    my_score: null,
    partner_names: [],
    ...overrides,
  };
}

describe("labels", () => {
  it("orders by count then name and trims to the limit", () => {
    expect(topLabels({ warm: 2, frustrated: 2, sad: 1, neutral: 0 })).toEqual([
      { label: "frustrated", count: 2 },
      { label: "warm", count: 2 },
      { label: "sad", count: 1 },
    ]);
    expect(topLabels({ a: 1, b: 2, c: 3, d: 4 }, 2)).toHaveLength(2);
    expect(topLabels(undefined)).toEqual([]);
  });

  it("colors escalations red, warmth green, sad blue, else gray", () => {
    expect(isEscalationLabel("Defensive ")).toBe(true);
    expect(isEscalationLabel("warm")).toBe(false);
    expect(toneChipColors("angry").fg).toBe("#B42318");
    expect(toneChipColors("warm").fg).toBe("#1B7A4B");
    expect(toneChipColors("sad").fg).toBe("#2F5F9E");
    expect(toneChipColors("neutral").fg).toBe("#4B5563");
    expect(toneChipColors("whatever").fg).toBe("#4B5563");
  });
});

describe("describeBucket / calmShare / meanLine", () => {
  it("writes the one-liner only when something was scored", () => {
    expect(describeBucket({ warm: 2, defensive: 1 }, 1, 3)).toBe(
      "mostly warm · 1 escalation",
    );
    expect(describeBucket({ warm: 2 }, 0, 2)).toBe("mostly warm · no escalations");
    expect(describeBucket({ frustrated: 3 }, 3, 3)).toBe(
      "mostly frustrated · 3 escalations",
    );
    // Nothing scored → nothing said (never "mostly neutral").
    expect(describeBucket({}, 0, 0)).toBeNull();
  });

  it("calm share is a percentage of scored turns, null when none", () => {
    expect(calmShare(4, 1)).toBe(75);
    expect(calmShare(3, 3)).toBe(0);
    expect(calmShare(0, 0)).toBeNull();
  });

  it("mean line lists only scored dimensions", () => {
    expect(meanLine({ mean: { warmth: 50.4, frustration: null, sarcasm: 12 } })).toBe(
      "warmth 50 · sarcasm 12",
    );
    expect(meanLine({ mean: { warmth: null } })).toBeNull();
    expect(meanLine(null)).toBeNull();
  });
});

describe("bucketToneByDay", () => {
  it("sums live sessions per LOCAL day and skips points without tone", () => {
    const sameDayA = "2026-08-24T09:00:00";
    const sameDayB = "2026-08-24T21:30:00";
    const nextDay = "2026-08-25T10:00:00";
    const days = bucketToneByDay([
      pt({
        recording_id: "a",
        timestamp: sameDayB,
        self_tone: { scored_turns: 2, labels: { warm: 2 }, mean: {}, escalation_count: 0, people: [] },
      }),
      pt({
        recording_id: "b",
        timestamp: sameDayA,
        self_tone: { scored_turns: 3, labels: { warm: 1, frustrated: 2 }, mean: {}, escalation_count: 2, people: [] },
      }),
      // An upload: no tone → contributes nothing, not a neutral day.
      pt({ recording_id: "c", timestamp: sameDayA, self_tone: null }),
      // A live session whose phone sent no tone → also nothing.
      pt({
        recording_id: "d",
        timestamp: nextDay,
        self_tone: { scored_turns: 0, labels: {}, mean: {}, escalation_count: 0, people: [] },
      }),
      pt({
        recording_id: "e",
        timestamp: nextDay,
        self_tone: { scored_turns: 1, labels: { sad: 1 }, mean: {}, escalation_count: 0, people: [] },
      }),
    ]);
    expect(days.map((d) => d.key)).toEqual([
      dateKeyOfIso(sameDayA),
      dateKeyOfIso(nextDay),
    ]);
    const [day1, day2] = days;
    expect(day1.sessions).toBe(2);
    expect(day1.scored_turns).toBe(5);
    expect(day1.labels).toEqual({ warm: 3, frustrated: 2 });
    expect(day1.escalation_count).toBe(2);
    // The day's timestamp is its EARLIEST session (for the label).
    expect(day1.timestamp).toBe(sameDayA);
    expect(day2.sessions).toBe(1);
    expect(day2.labels).toEqual({ sad: 1 });
  });

  it("is empty when no point carries tone", () => {
    expect(bucketToneByDay([pt(), pt({ self_tone: null })])).toEqual([]);
  });
});

describe("people + episodes + modes", () => {
  it("resolves a person's name honestly", () => {
    expect(personName({ display_name: " Mom " })).toBe("Mom");
    expect(personName({ display_name: null, speaker: "Speaker B" })).toBe("Speaker B");
    expect(personName({ person_id: "p1" })).toBe("p1");
    expect(personName({})).toBe("Someone");
  });

  it("peopleRows keeps scored people with a summary", () => {
    const rows = peopleRows([
      { person_id: "p-mom", display_name: "Mom", sessions: 2, scored_turns: 6,
        labels: { warm: 4, defensive: 2 }, escalation_count: 2 },
      { person_id: "p-x", display_name: "X", sessions: 1, scored_turns: 0,
        labels: {}, escalation_count: 0 },
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0].name).toBe("Mom");
    expect(rows[0].summary).toBe("mostly warm · 2 escalations");
    expect(peopleRows(undefined)).toEqual([]);
  });

  it("episode chip line", () => {
    expect(episodeToneLine({ warm: 2, frustrated: 1 }, 1)).toBe(
      "you: warm ×2, frustrated ×1 · 1 escalation",
    );
    expect(episodeToneLine({ warm: 1 }, 0)).toBe("you: warm ×1");
    expect(episodeToneLine({}, 0)).toBeNull();
    expect(episodeToneLine(undefined, undefined)).toBeNull();
  });

  it("mode labels", () => {
    expect(modeLabel("earpiece")).toBe("Earpiece");
    expect(modeLabel("speaker")).toBe("Speaker");
    expect(modeLabel("therapist")).toBe("Therapist");
    expect(modeLabel(null)).toBeNull();
    expect(modeLabel("loud")).toBeNull();
  });
});
