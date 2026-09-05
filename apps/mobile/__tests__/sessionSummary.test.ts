import {
  formatDuration,
  formatLatency,
  summarizeSession,
} from "../src/live/sessionSummary";
import type { TurnLatency } from "../src/live/fastLoop";

function lat(turn: number, toSpeakMs: number | null, provider = "os"): TurnLatency {
  return {
    turn,
    segmentEndMs: turn * 1000,
    prosodyMs: 1,
    speakerMs: 20,
    sttWaitMs: 100,
    llmMs: 300,
    toSpeakMs,
    provider,
    held: false,
  };
}

describe("summarizeSession", () => {
  it("counts turns per speaker (most first), duration from the start stamp, and escalations", () => {
    const s = summarizeSession({
      startedAt: "2026-08-24T18:05:00.000Z",
      endedAt: "2026-08-24T18:07:14.000Z",
      transcript: [
        { speaker: "You", text: "hi" },
        { speaker: "Mom", text: "hey" },
        { speaker: "You", text: "so" },
        { speaker: "Mom", text: "yeah", kind: "backchannel" }, // listener noise: shown, never counted
        { speaker: "You", text: "" }, // empty text never counts
      ],
      latencyLog: [],
      escalations: 2,
    });
    expect(s.durationMs).toBe(134000);
    expect(s.turnsBySpeaker).toEqual([
      { speaker: "You", turns: 2 },
      { speaker: "Mom", turns: 1 },
    ]);
    expect(s.totalTurns).toBe(3);
    expect(s.escalations).toBe(2);
    // Nothing spoken: latency is honestly unknown, not 0.
    expect(s.firstWordsMedianMs).toBeNull();
    expect(s.firstWordsBestMs).toBeNull();
    expect(s.spokenTurns).toBe(0);
    expect(s.topProvider).toBeNull();
  });

  it("first-words latency is the median/best of the SPOKEN turns; provider is the most frequent", () => {
    const s = summarizeSession({
      startedAt: null,
      transcript: [],
      latencyLog: [lat(0, 900), lat(1, null, "cloud"), lat(2, 400), lat(3, 700, "bundled")],
      escalations: -3,
    });
    expect(s.durationMs).toBeNull();
    expect(s.firstWordsMedianMs).toBe(700);
    expect(s.firstWordsBestMs).toBe(400);
    expect(s.spokenTurns).toBe(3);
    expect(s.topProvider).toBe("os");
    expect(s.escalations).toBe(0);
  });

  it("unparseable stamps give a null duration; an end before the start is not negative", () => {
    expect(summarizeSession({ startedAt: "nope", transcript: [], latencyLog: [], escalations: 0 }).durationMs).toBeNull();
    expect(
      summarizeSession({
        startedAt: "2026-08-24T18:07:00Z",
        endedAt: "2026-08-24T18:05:00Z",
        transcript: [],
        latencyLog: [],
        escalations: 0,
      }).durationMs,
    ).toBeNull();
  });
});

describe("formatters", () => {
  it("formats durations and latencies for the card", () => {
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(48000)).toBe("48s");
    expect(formatDuration(134000)).toBe("2m 14s");
    expect(formatDuration(600500)).toBe("10m 01s");
    expect(formatLatency(null)).toBe("—");
    expect(formatLatency(640.4)).toBe("640 ms");
    expect(formatLatency(1340)).toBe("1.3 s");
  });
});
