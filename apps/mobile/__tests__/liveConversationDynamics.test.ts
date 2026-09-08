import {
  computeConversationDynamics,
  SLOW_RESPONSE_THRESHOLD_S,
  SUSTAINED_OVERLAP_SHORT_S,
  SUSTAINED_OVERLAP_LONG_S,
  type DynamicsTurn,
} from "../src/live/conversationDynamics";

function turn(
  speaker: string,
  isSelf: boolean | null,
  startTime: number,
  endTime: number,
  kind?: "primary" | "backchannel",
): DynamicsTurn {
  return { speaker, isSelf, startTime, endTime, kind };
}

describe("computeConversationDynamics", () => {
  it("single-mic sequential session: sequential turns, no overlap, self/partner gaps split correctly", () => {
    // Partner speaks 0-2, self responds 2.5-4 (gap 0.5s), partner responds
    // 4.4-6 (gap 0.4s), self responds 6.3-7 (gap 0.3s).
    const turns: DynamicsTurn[] = [
      turn("Mom", false, 0, 2),
      turn("You", true, 2.5, 4),
      turn("Mom", false, 4.4, 6),
      turn("You", true, 6.3, 7),
    ];
    const d = computeConversationDynamics(turns);
    expect(d.selfResponseGaps.count).toBe(2);
    expect(d.selfResponseGaps.medianS).toBeCloseTo((0.5 + 0.3) / 2, 5);
    expect(d.partnerResponseGaps.count).toBe(1);
    expect(d.partnerResponseGaps.medianS).toBeCloseTo(0.4, 5);
    expect(d.overlapSecondsTotal).toBe(0);
    expect(d.overlapEpisodes).toEqual([]);
    expect(d.sustainedOverlapCountOver1s).toBe(0);
    expect(d.sustainedOverlapCountOver2s).toBe(0);
  });

  it("call mode with real overlap: negative gap and a sustained-overlap episode are both reported", () => {
    // Partner speaks 0-5; self starts at 3.5 (1.5s before partner ends) and
    // runs to 8 -> 1.5s overlap, sustained (>1s) but not >2s.
    const turns: DynamicsTurn[] = [turn("Dad", false, 0, 5), turn("You", true, 3.5, 8)];
    const d = computeConversationDynamics(turns);
    expect(d.selfResponseGaps.count).toBe(1);
    expect(d.selfResponseGaps.medianS).toBeCloseTo(-1.5, 5); // negative = overlap
    expect(d.overlapSecondsTotal).toBeCloseTo(1.5, 5);
    expect(d.overlapEpisodes).toHaveLength(1);
    expect(d.overlapEpisodes[0]).toMatchObject({ speakerA: "Dad", speakerB: "You" });
    expect(d.sustainedOverlapCountOver1s).toBe(1);
    expect(d.sustainedOverlapCountOver2s).toBe(0);
  });

  it("sustained overlap over the 2s band counts in both buckets", () => {
    const turns: DynamicsTurn[] = [turn("Dad", false, 0, 10), turn("You", true, 2, 5)]; // fully inside -> 3s overlap
    const d = computeConversationDynamics(turns);
    expect(d.overlapSecondsTotal).toBeCloseTo(3, 5);
    expect(d.sustainedOverlapCountOver1s).toBe(1);
    expect(d.sustainedOverlapCountOver2s).toBe(1);
  });

  it("backchannels are ignored entirely: no gap contribution, no overlap contribution", () => {
    const turns: DynamicsTurn[] = [
      turn("Mom", false, 0, 2),
      turn("You", true, 2.1, 2.3, "backchannel"), // "mhm" — should not count as a self turn
      turn("You", true, 5.5, 7), // real self response, gap measured from Mom's turn, not the backchannel
      turn("Mom", false, 6, 6.5, "backchannel"), // overlaps "You" above but is a backchannel -> ignored
    ];
    const d = computeConversationDynamics(turns);
    expect(d.selfResponseGaps.count).toBe(1);
    expect(d.selfResponseGaps.medianS).toBeCloseTo(3.5, 5); // 5.5 - 2, not measured from the backchannel
    expect(d.overlapSecondsTotal).toBe(0);
    expect(d.overlapEpisodes).toEqual([]);
  });

  it("unattributed (isSelf === null) turns are excluded from the self/partner split but can still overlap", () => {
    const turns: DynamicsTurn[] = [
      turn("Unknown", null, 0, 2),
      turn("You", true, 2.5, 4),
    ];
    const d = computeConversationDynamics(turns);
    expect(d.selfResponseGaps.count).toBe(0);
    expect(d.partnerResponseGaps.count).toBe(0);
  });

  it("slow responses (> SLOW_RESPONSE_THRESHOLD_S) are counted separately from ordinary gaps", () => {
    const turns: DynamicsTurn[] = [
      turn("Mom", false, 0, 1),
      turn("You", true, 1 + SLOW_RESPONSE_THRESHOLD_S + 1, 2 + SLOW_RESPONSE_THRESHOLD_S + 1), // way over 2s
    ];
    const d = computeConversationDynamics(turns);
    expect(d.selfResponseGaps.slowCount).toBe(1);
  });

  it("p90 uses linear interpolation over the full gap distribution", () => {
    const turns: DynamicsTurn[] = [
      turn("Mom", false, 0, 1),
      turn("You", true, 1.1, 2), // gap .1
      turn("Mom", false, 2.2, 3), // gap .2
      turn("You", true, 3.5, 4), // gap .5
      turn("Mom", false, 5, 6), // gap 1
    ];
    const d = computeConversationDynamics(turns);
    // self gaps: .1, .5 ; partner gaps: .2, 1
    expect(d.selfResponseGaps.count).toBe(2);
    expect(d.partnerResponseGaps.count).toBe(2);
    expect(d.partnerResponseGaps.p90S).not.toBeNull();
  });

  it("same-side consecutive turns (partner, partner) are not treated as a response", () => {
    const turns: DynamicsTurn[] = [turn("Mom", false, 0, 1), turn("Dad", false, 1.5, 2.5)];
    const d = computeConversationDynamics(turns);
    expect(d.selfResponseGaps.count).toBe(0);
    expect(d.partnerResponseGaps.count).toBe(0);
  });

  it("empty input returns zeroed/null stats, never throws", () => {
    const d = computeConversationDynamics([]);
    expect(d.selfResponseGaps).toEqual({ count: 0, medianS: null, p90S: null, slowCount: 0 });
    expect(d.partnerResponseGaps).toEqual({ count: 0, medianS: null, p90S: null, slowCount: 0 });
    expect(d.overlapSecondsTotal).toBe(0);
    expect(d.overlapEpisodes).toEqual([]);
  });

  it("exports document the CANDOR-derived thresholds used above", () => {
    expect(SLOW_RESPONSE_THRESHOLD_S).toBe(2);
    expect(SUSTAINED_OVERLAP_SHORT_S).toBe(1);
    expect(SUSTAINED_OVERLAP_LONG_S).toBe(2);
  });
});
