/**
 * Quiet coach (src/live/coachTrigger.ts + src/live/liveCoachFlags.ts): the
 * trigger rules the phone applies before voicing a free-form line, the
 * production defaults, and the replay harness's env switches.
 */
import vectors from "./fixtures/coachTrigger.vectors.json";
import {
  isDirectQuestion,
  quietCoachFrom,
  QUIET_COACH_DEFAULTS,
  turnTriggers,
} from "../src/live/coachTrigger";
import { LIVE_COACH_FLAGS, liveCoachDeps } from "../src/live/liveCoachFlags";
import { liveFlagsFromEnv, floorTickFromEnv } from "../src/live/replay/recordingReplay";
import { phoneAnsweredTurn } from "../src/hooks/useAudioStream";

describe("isDirectQuestion", () => {
  it("agrees with the server on every shared vector (server/tests/test_coach_trigger.py)", () => {
    for (const v of vectors.question) {
      expect([v.text, isDirectQuestion(v.text)]).toEqual([v.text, v.q]);
    }
  });
});

describe("turnTriggers", () => {
  it("heat, floor and question each fire on their own", () => {
    expect(turnTriggers({ text: "shut up", isSelf: false, floorFired: false })).toEqual(["heat"]);
    expect(turnTriggers({ text: "and then we drove on", isSelf: true, floorFired: true })).toEqual(["floor"]);
    expect(turnTriggers({ text: "what do you think?", isSelf: false, floorFired: false })).toEqual(["question"]);
    expect(turnTriggers({ text: "what do you think?", isSelf: null, floorFired: false })).toEqual(["question"]);
  });

  it("calm talk and the wearer's own questions trigger nothing", () => {
    expect(turnTriggers({ text: "we should look at the budget for next quarter", isSelf: false, floorFired: false })).toEqual([]);
    expect(turnTriggers({ text: "what do you think?", isSelf: true, floorFired: false })).toEqual([]);
  });

  it("each trigger can be switched off", () => {
    const opts = { ...QUIET_COACH_DEFAULTS, heat: false, question: false, floor: false };
    expect(turnTriggers({ text: "shut up, what do you think?", isSelf: false, floorFired: true }, opts)).toEqual([]);
  });

  it("quietCoachFrom: off by default, true = defaults, object overrides", () => {
    expect(quietCoachFrom(undefined)).toBeNull();
    expect(quietCoachFrom(false)).toBeNull();
    expect(quietCoachFrom(true)).toEqual(QUIET_COACH_DEFAULTS);
    expect(quietCoachFrom({ heatMin: 3 })?.heatMin).toBe(3);
  });
});

describe("LIVE_COACH_FLAGS (production defaults, OTA-reversible)", () => {
  it("turns on the floor cue, the heated-words buzz, the quiet coach and earlyTurnLocal", () => {
    expect(LIVE_COACH_FLAGS).toEqual({ floorTick: true, heatLexiconBuzz: true, quietCoach: true, earlyTurnLocal: true });
    expect(liveCoachDeps()).toEqual(LIVE_COACH_FLAGS);
    expect(liveCoachDeps({ floorTick: false, heatLexiconBuzz: false, quietCoach: false, earlyTurnLocal: false })).toEqual({
      floorTick: false,
      heatLexiconBuzz: false,
      quietCoach: false,
      earlyTurnLocal: false,
    });
  });

  it("the replay harness stays dark unless asked; MINDSHIFT_LIVE_DEFAULTS=1 = production", () => {
    expect(liveFlagsFromEnv({})).toEqual({ heatLexiconBuzz: false, quietCoach: false, earlyTurnLocal: false });
    expect(floorTickFromEnv({})).toBeNull();
    expect(liveFlagsFromEnv({ MINDSHIFT_LIVE_DEFAULTS: "1" })).toEqual({
      heatLexiconBuzz: true,
      quietCoach: true,
      earlyTurnLocal: true,
    });
    expect(floorTickFromEnv({ MINDSHIFT_LIVE_DEFAULTS: "1" })).toEqual({});
    expect(liveFlagsFromEnv({ MINDSHIFT_LIVE_DEFAULTS: "1", MINDSHIFT_QUIET_COACH: "0" }).quietCoach).toBe(false);
    expect(floorTickFromEnv({ MINDSHIFT_LIVE_DEFAULTS: "1", MINDSHIFT_FLOOR_TICK: "0" })).toBeNull();
  });
});

describe("phoneAnsweredTurn (cloudAnswersOpenMoment input)", () => {
  it("a voiced local line answered; a quiet-muted one did not; a floor cue did", () => {
    expect(phoneAnsweredTurn({ suggestion: "x" })).toBe(true);
    expect(phoneAnsweredTurn({ suggestion: null })).toBe(false);
    expect(phoneAnsweredTurn({ suggestion: "x", quietMuted: true })).toBe(false);
    expect(phoneAnsweredTurn({ suggestion: null, triggers: ["floor"] })).toBe(true);
  });
});
