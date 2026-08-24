/**
 * src/live/nudgePolicy.ts — replays every golden case in
 * server/tests/fixtures/policy_vectors/nudge_policy.json tagged `phone` (or
 * `server`), plus the phone-side helpers that feed it.
 */
import {
  aggressiveToneLevel,
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
    channels: Channel[];
    subscriptions: { vector: VectorName; sensitivity: number; haptics: boolean; channel: Channel }[];
  };
  inputs: { t: number; events: { vector: VectorName; level: number; db_over_baseline?: number }[] }[];
  expected: { nudges: { channel: Channel; level: number; vectors: string[] }[]; levels: Record<string, number> }[];
}

const doc = loadFixture<{ _schema: { version: number }; cases: NudgeCase[] }>("nudge_policy.json");
expect(doc._schema.version).toBe(1);
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
    ]) {
      expect(names.has(n)).toBe(true);
    }
  });

  it.each(CASES.map((c) => [c.name, c] as const))("replays identically: %s", (_name, c) => {
    const policy = new NudgePolicy(c.config.subscriptions, c.config.cooldown_s, c.config.channels);
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
    const nudges = policy.onEvents(events, 3.0);
    expect(nudges).toEqual([{ channel: "A", level: 3, t: 3.0, vectors: ["aggressive_tone", "yelling"] }]);
    expect(policy.current()).toEqual({ A: 3 });
  });
});
