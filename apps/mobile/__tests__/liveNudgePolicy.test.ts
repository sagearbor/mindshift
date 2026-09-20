/**
 * src/live/nudgePolicy.ts — replays every golden case in
 * server/tests/fixtures/policy_vectors/nudge_policy.json tagged `phone` (or
 * `server`), plus the phone-side helpers that feed it.
 */
import {
  aggressiveToneLevel,
  CoachRepeatGate,
  coachingOverlap,
  LoudnessBaseline,
  NudgePolicy,
  phoneNudgePolicy,
  roundHalfUp,
  selfTurnVectorEvents,
  yellingLevel,
  type Channel,
  type VectorName,
} from "../src/live/nudgePolicy";
import { loadFixture } from "../src/live/testing/synth";

interface NudgeCase {
  name: string;
  applies_to: string[];
  config: {
    cooldown_s: number;
    /** Schema v2: the loudness ladder's hold-N hysteresis, in seconds. */
    hold_s: number;
    channels: Channel[];
    subscriptions: { vector: VectorName; sensitivity: number; haptics: boolean; channel: Channel }[];
  };
  inputs: { t: number; events: { vector: VectorName; level: number; db_over_baseline?: number }[] }[];
  expected: { nudges: { channel: Channel; level: number; vectors: string[] }[]; levels: Record<string, number> }[];
}

const doc = loadFixture<{ _schema: { version: number }; cases: NudgeCase[] }>("nudge_policy.json");
expect(doc._schema.version).toBe(2);
const CASES = doc.cases.filter((c) => c.applies_to.includes("phone") || c.applies_to.includes("server"));

describe("nudge_policy.json golden vectors", () => {
  it("every case applies to the phone and the required names exist", () => {
    expect(CASES.length).toBe(doc.cases.length);
    expect(CASES.length).toBeGreaterThanOrEqual(8);
    const names = new Set(CASES.map((c) => c.name));
    for (const n of [
      "below_threshold_no_nudge",
      "single_nudge_then_sustain_is_silent",
      "cooldown_is_strictly_greater_than",
      "sustained_observation_refreshes_clock",
      "stepwise_deescalation_3_to_0",
      "full_decay_then_fresh_escalation",
      // hold-3s (2026-09-20): the loudness ladder's hysteresis, pinned on all three runtimes.
      "hold_two_loud_windows_then_quiet_never_buzzes",
      "hold_three_loud_windows_escalates_on_the_third",
      "hold_of_one_is_the_pre_hysteresis_ladder",
    ]) {
      expect(names.has(n)).toBe(true);
    }
  });

  it.each(CASES.map((c) => [c.name, c] as const))("replays identically: %s", (_name, c) => {
    // `hold_s` is REQUIRED by schema v2 and never defaulted here, so a case
    // cannot silently be replayed under a different gate than it was written for.
    const policy = new NudgePolicy(c.config.subscriptions, c.config.cooldown_s, c.config.channels, c.config.hold_s);
    expect(c.inputs.length).toBe(c.expected.length);
    c.inputs.forEach((step, i) => {
      const got = policy.onEvents(
        step.events.map((e) => ({ vector: e.vector, level: e.level, t: step.t, value: e.db_over_baseline ?? 0 })),
        step.t,
      );
      expect(got.map((n) => ({ channel: n.channel, level: n.level, vectors: n.vectors }))).toEqual(c.expected[i].nudges);
      expect(got.every((n) => n.t === step.t)).toBe(true);
      expect(policy.current()).toEqual(c.expected[i].levels);
    });
  });

  it("watch-shaped cases' db_over_baseline agrees with the level via the shared ladder", () => {
    for (const c of doc.cases.filter((x) => x.applies_to.includes("watch"))) {
      for (const step of c.inputs) {
        for (const e of step.events) {
          expect(yellingLevel(e.db_over_baseline as number)).toBe(e.level);
        }
      }
    }
  });
});

describe("policy surface", () => {
  it("default channels are A then B; hr_spike defaults to B; empty channels rejected", () => {
    const p = new NudgePolicy([{ vector: "yelling" }, { vector: "hr_spike" }]);
    expect(p.channels).toEqual(["A", "B"]);
    expect(p.current()).toEqual({ A: 0, B: 0 });
    expect(p.onEvents([{ vector: "hr_spike", level: 2, t: 1 }], 1)).toEqual([{ channel: "B", level: 2, t: 1, vectors: ["hr_spike"] }]);
    expect(() => new NudgePolicy([{ vector: "yelling" }], 20, [])).toThrow();
  });

  it("roundHalfUp matches Kotlin Math.round", () => {
    expect([0.5, 1.5, 2.4, -0.5].map(roundHalfUp)).toEqual([1, 2, 2, -1]);
  });
});

describe("phone-side inputs", () => {
  it("yellingLevel uses the watch ladder, aggressiveToneLevel the text ladder", () => {
    expect([0, 5.9, 6, 10, 14, 30].map(yellingLevel)).toEqual([0, 0, 1, 2, 3, 3]);
    expect(aggressiveToneLevel(null, null)).toBe(0);
    expect(aggressiveToneLevel(49, 10)).toBe(0);
    expect(aggressiveToneLevel(50, null)).toBe(1);
    expect(aggressiveToneLevel(10, 70)).toBe(2);
    expect(aggressiveToneLevel(90, 0)).toBe(3);
  });

  it("LoudnessBaseline is the median of prior self turns, 0-over until it exists", () => {
    const b = new LoudnessBaseline(2);
    expect(b.observe(-30)).toBe(0);
    expect(b.value).toBeNull();
    expect(b.observe(-28)).toBe(0);
    expect(b.value).toBe(-29);
    expect(b.observe(-15)).toBe(14);
    expect(b.observe(null)).toBe(0);
    expect(b.observe(-Infinity)).toBe(0);
  });

  it("selfTurnVectorEvents + phoneNudgePolicy: a loud, frustrated turn fires level 3 on A only", () => {
    const policy = phoneNudgePolicy();
    const events = selfTurnVectorEvents(3.0, 15, { frustration: 90, defensiveness: null });
    expect(events).toEqual([
      { vector: "yelling", level: 3, t: 3.0, value: 15 },
      { vector: "aggressive_tone", level: 3, t: 3.0 },
    ]);
    // hold-3s: one 1 s observation is not a raised voice, so the loudness lane
    // is still serving its hold and only `aggressive_tone` is credited. The
    // level is unchanged — the words alone reach 3 here — which is the point:
    // the hold removes a REASON, never a nudge the tone lane would have made.
    const nudges = policy.onEvents(events, 3.0);
    expect(nudges).toEqual([{ channel: "A", level: 3, t: 3.0, vectors: ["aggressive_tone"] }]);
    expect(policy.current()).toEqual({ A: 3 });
  });

  it("hold-3s: the loudness lane joins once the rung has held, by seconds or by windows", () => {
    // A quiet-tongued but loud turn: nothing for the tone lane to carry, so
    // this isolates the hold. Three 1 s windows, and the third climbs.
    const byWindows = phoneNudgePolicy();
    const loud = (t: number) => selfTurnVectorEvents(t, 15, { frustration: 0, defensiveness: 0 });
    expect(byWindows.onEvents(loud(1.0), 1.0)).toEqual([]);
    expect(byWindows.onEvents(loud(2.0), 2.0)).toEqual([]);
    expect(byWindows.onEvents(loud(3.0), 3.0)).toEqual([
      { channel: "A", level: 3, t: 3.0, vectors: ["yelling"] },
    ]);

    // The phone observes once per TURN, so one 3 s loud turn is the same three
    // windows of hold and buzzes on the spot — that is why hold-3s costs the
    // phone almost no real escalations while halving the watch's window-rate dose.
    const byTurn = phoneNudgePolicy();
    expect(byTurn.onEvents(loud(4.0), 4.0, 3.0)).toEqual([
      { channel: "A", level: 3, t: 4.0, vectors: ["yelling"] },
    ]);

    // …and holdS 1 is the pre-2026-09-20 ladder: the first window climbs.
    const legacy = phoneNudgePolicy(20.0, 1.0);
    expect(legacy.onEvents(loud(1.0), 1.0)).toEqual([
      { channel: "A", level: 3, t: 1.0, vectors: ["yelling"] },
    ]);
  });

  it("hold-3s: a tick that heard no audio (observedS null) leaves the run alone", () => {
    // The mirror of the server's `hr` frame: one policy serves both lanes, and
    // a tick that did not hear the wearer is not evidence that they went quiet.
    // Counting it as a quiet window would make a raised voice go unbuzzed
    // because something else happened to tick in the middle of it.
    const p = phoneNudgePolicy();
    const loud = (t: number) => selfTurnVectorEvents(t, 15, { frustration: 0, defensiveness: 0 });
    expect(p.onEvents(loud(1.0), 1.0)).toEqual([]);
    expect(p.onEvents([], 1.5, null)).toEqual([]);
    expect(p.hold.run).toBe(1);
    expect(p.onEvents(loud(2.0), 2.0)).toEqual([]);
    expect(p.onEvents(loud(3.0), 3.0)).toEqual([
      { channel: "A", level: 3, t: 3.0, vectors: ["yelling"] },
    ]);
  });
});

// Don't-nag gate over the coach's lines (docs/research/2026-08-30-nudge-quality).
// Mirrors server/audio_pipeline.py's _coaching_overlap / _is_repeat_coaching.
describe("CoachRepeatGate", () => {
  it("coachingOverlap is bigram Jaccard: 1 for the same words, ~0 for unrelated, unigrams for one word", () => {
    expect(coachingOverlap("ease up", "Ease up!")).toBe(1);
    expect(coachingOverlap("ease up", "let them finish")).toBe(0);
    expect(coachingOverlap("", "ease up")).toBe(0);
    expect(coachingOverlap("breathe", "breathe")).toBe(1);
    // A reworded re-issue keeps most bigrams.
    expect(
      coachingOverlap(
        "Please be specific. I need clear directions to find you.",
        "Please be specific, I need clear directions to find you now.",
      ),
    ).toBeGreaterThanOrEqual(0.5);
    // Sharing one phrase is not a repeat.
    expect(coachingOverlap("I hear you, that sounds hard.", "I hear you — where do we meet?")).toBeLessThan(0.5);
  });

  it("the exact line the real session repeated on two fragments is silenced the second time", () => {
    // 4a7ec4ed (2026-08-26): turns 1 and 2 both got this line, 3 s apart.
    const gate = new CoachRepeatGate();
    const line = "Please be specific. I need clear directions to find you.";
    expect(gate.admit(line, 4.5, "response")).toBe(line);
    expect(gate.admit(line, 7.7, "response")).toBeNull();
    // A different line goes through and is remembered.
    expect(gate.admit("Where exactly are you standing?", 11.1, "response")).toBe("Where exactly are you standing?");
    expect(gate.isRepeat("Where exactly are you standing right now?", 12.0)).toBe(true);
  });

  it("a repeat outside the cooldown is allowed again; the cooldown is session time", () => {
    const gate = new CoachRepeatGate(45);
    expect(gate.admit("ease up", 10, "nudge")).toBe("ease up");
    expect(gate.admit("ease up", 55, "nudge")).toBeNull(); // exactly 45 s later: still inside
    expect(gate.admit("ease up", 55.1, "nudge")).toBe("ease up");
  });

  it("a nudge is only gated against earlier NUDGES; a response against every line", () => {
    const gate = new CoachRepeatGate();
    gate.remember("slow down", 1, "response");
    expect(gate.admit("slow down", 2, "nudge")).toBe("slow down");
    expect(gate.admit("slow down", 3, "response")).toBeNull();
    expect(gate.admit(null, 4, "nudge")).toBeNull();
    expect(gate.admit("", 4, "nudge")).toBeNull();
  });
});
