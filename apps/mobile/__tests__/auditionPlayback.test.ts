import {
  SEGMENT_END_EPS,
  speakerSegments,
  createSegmentAudition,
  type AuditionSegment,
  type SegmentPlayer,
} from "../src/components/auditionPlayback";

// A fake player that records the exact call sequence — the seam the audition
// engine drives. Order matters (seek → play → seek → … → pause), so calls are
// logged into one flat list rather than separate spies.
function fakePlayer(): { player: SegmentPlayer; calls: string[] } {
  const calls: string[] = [];
  return {
    calls,
    player: {
      seek: (s: number) => calls.push(`seek:${s}`),
      play: () => calls.push("play"),
      pause: () => calls.push("pause"),
    },
  };
}

const turns = [
  { speaker: "Alice", start_time: 0, end_time: 3 },
  { speaker: "Bob", start_time: 3, end_time: 6 },
  { speaker: "Alice", start_time: 6, end_time: 9 },
  { speaker: "Bob", start_time: 9, end_time: 12 },
  { speaker: "Alice", start_time: 14, end_time: 15 }, // after a silent gap
];

describe("speakerSegments", () => {
  it("keeps only the given speaker's turns, in chronological order", () => {
    expect(speakerSegments(turns, "Alice")).toEqual([
      { start: 0, end: 3 },
      { start: 6, end: 9 },
      { start: 14, end: 15 },
    ]);
    expect(speakerSegments(turns, "Bob")).toEqual([
      { start: 3, end: 6 },
      { start: 9, end: 12 },
    ]);
  });

  it("returns [] for an unknown speaker and drops zero/negative-length turns", () => {
    expect(speakerSegments(turns, "Carol")).toEqual([]);
    expect(
      speakerSegments(
        [
          { speaker: "Alice", start_time: 5, end_time: 5 },
          { speaker: "Alice", start_time: 8, end_time: 7 },
          { speaker: "Alice", start_time: 1, end_time: 2 },
        ],
        "Alice",
      ),
    ).toEqual([{ start: 1, end: 2 }]);
  });

  it("orders segments by start time even when turns arrive unsorted", () => {
    const shuffled = [
      { speaker: "Alice", start_time: 6, end_time: 9 },
      { speaker: "Alice", start_time: 0, end_time: 3 },
    ];
    expect(speakerSegments(shuffled, "Alice")).toEqual([
      { start: 0, end: 3 },
      { start: 6, end: 9 },
    ]);
  });
});

describe("createSegmentAudition", () => {
  // Alice's segments incl. NON-ADJACENT ones (0–3, 6–9, 14–15): the engine must
  // jump the gaps, never play Bob's 3–6 / 9–12 or the 12–14 silence.
  const segs: AuditionSegment[] = [
    { start: 0, end: 3 },
    { start: 6, end: 9 },
    { start: 14, end: 15 },
  ];

  it("start() seeks to the first segment and plays", () => {
    const { player, calls } = fakePlayer();
    const a = createSegmentAudition(player, segs);
    a.start();
    expect(calls).toEqual(["seek:0", "play"]);
    expect(a.isActive()).toBe(true);
  });

  it("chains segment → segment on position updates and pauses after the last", () => {
    const { player, calls } = fakePlayer();
    const ended = jest.fn();
    const a = createSegmentAudition(player, segs, ended);
    a.start();

    // Mid-segment positions do nothing.
    a.handlePosition(1.2);
    a.handlePosition(2.0);
    expect(calls).toEqual(["seek:0", "play"]);

    // Reaching (within EPS of) segment 1's end jumps to segment 2's start —
    // playback keeps running, no pause in between.
    a.handlePosition(3 - SEGMENT_END_EPS / 2);
    expect(calls).toEqual(["seek:0", "play", "seek:6"]);
    expect(ended).not.toHaveBeenCalled();

    // …and across the second gap.
    a.handlePosition(7.5);
    a.handlePosition(9.01);
    expect(calls).toEqual(["seek:0", "play", "seek:6", "seek:14"]);

    // End of the last segment → pause, done.
    a.handlePosition(15);
    expect(calls).toEqual(["seek:0", "play", "seek:6", "seek:14", "pause"]);
    expect(a.isActive()).toBe(false);
    expect(ended).toHaveBeenCalledTimes(1);

    // Late position ticks after the end are ignored.
    a.handlePosition(15.5);
    expect(calls).toEqual(["seek:0", "play", "seek:6", "seek:14", "pause"]);
  });

  it("a stale position from BEFORE the current segment never advances it", () => {
    const { player, calls } = fakePlayer();
    const a = createSegmentAudition(player, segs);
    a.start();
    a.handlePosition(2.99); // → segment 2 (seek:6)
    // The ~4Hz poll can still report a pre-seek position once; must not advance.
    a.handlePosition(2.99);
    a.handlePosition(0.4);
    expect(calls).toEqual(["seek:0", "play", "seek:6"]);
  });

  it("stop() pauses mid-run and goes inert", () => {
    const { player, calls } = fakePlayer();
    const ended = jest.fn();
    const a = createSegmentAudition(player, segs, ended);
    a.start();
    a.handlePosition(2.99); // now on segment 2
    a.stop();
    expect(calls).toEqual(["seek:0", "play", "seek:6", "pause"]);
    expect(a.isActive()).toBe(false);
    expect(ended).toHaveBeenCalledTimes(1);

    // Inert after stop: positions and repeat stops do nothing further.
    a.handlePosition(9.5);
    a.stop();
    expect(calls).toEqual(["seek:0", "play", "seek:6", "pause"]);
    expect(ended).toHaveBeenCalledTimes(1);
  });

  it("an empty segment list never touches the player and reports ended", () => {
    const { player, calls } = fakePlayer();
    const ended = jest.fn();
    const a = createSegmentAudition(player, [], ended);
    a.start();
    expect(calls).toEqual([]);
    expect(a.isActive()).toBe(false);
    expect(ended).toHaveBeenCalledTimes(1);
  });
});
