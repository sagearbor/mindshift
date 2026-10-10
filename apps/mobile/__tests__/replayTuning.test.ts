import { describeTuning, tuningFromEnv } from "../src/live/replay/tuning";
import { DEFAULT_SEGMENTER_CONFIG } from "../src/live/segmenter";
import { TURN_SPLIT_DEFAULTS } from "../src/live/turnSplit";
import { SILERO_SPEECH_OFF, SILERO_SPEECH_ON, VAD_AGC_ENABLED } from "../src/live/vad";

describe("replay listening tuning from env", () => {
  it("unset = what the phone ships", () => {
    const t = tuningFromEnv({});
    expect(t).toEqual({
      agc: VAD_AGC_ENABLED,
      vadOn: SILERO_SPEECH_ON,
      vadOff: SILERO_SPEECH_OFF,
      segmenter: DEFAULT_SEGMENTER_CONFIG,
      turnSplit: null,
    });
  });

  it("threshold on implies off = on - 0.15; explicit off wins", () => {
    expect(tuningFromEnv({ MINDSHIFT_VAD_ON: "0.4" }).vadOff).toBeCloseTo(0.25);
    expect(tuningFromEnv({ MINDSHIFT_VAD_ON: "0.4", MINDSHIFT_VAD_OFF: "0.3" }).vadOff).toBe(0.3);
  });

  it("agc, segmenter and split knobs", () => {
    const t = tuningFromEnv({
      MINDSHIFT_VAD_AGC: "0",
      MINDSHIFT_SEG_MERGE_GAP: "0.5",
      MINDSHIFT_SEG_MIN: "0.4",
      MINDSHIFT_TURN_SPLIT_THR: "0.2",
    });
    expect(t.agc).toBe(false);
    expect(t.segmenter).toEqual({ mergeGapSeconds: 0.5, minSeconds: 0.4 });
    expect(t.turnSplit).toEqual({ ...TURN_SPLIT_DEFAULTS, threshold: 0.2 });
    expect(describeTuning(t)).toContain("thr0.2");
    expect(tuningFromEnv({ MINDSHIFT_TURN_SPLIT: "1" }).turnSplit).toBe(true);
    expect(tuningFromEnv({ MINDSHIFT_TURN_SPLIT: "0", MINDSHIFT_TURN_SPLIT_THR: "0.2" }).turnSplit).toBe(false);
  });

  it("rejects garbage", () => {
    expect(() => tuningFromEnv({ MINDSHIFT_VAD_ON: "abc" })).toThrow(/not a number/);
  });
});
