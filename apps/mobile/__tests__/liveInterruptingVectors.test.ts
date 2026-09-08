/**
 * src/live/nudgePolicy.ts interruptingEvents — bit-identical to
 * server/watch/vectors.py interrupting_events through the shared fixture
 * server/tests/fixtures/policy_vectors/interrupting.json (the codebase's
 * cross-runtime contract pattern, like nudge_policy.json).
 */
import * as fs from "fs";
import * as path from "path";
import { INTERRUPT_LEVELS, interruptingEvents } from "../src/live/nudgePolicy";

const FIXTURE = path.resolve(__dirname, "../../../server/tests/fixtures/policy_vectors/interrupting.json");

interface Case {
  name: string;
  self: [number, number][];
  other: [number, number][];
  events: { level: number; t: number; value: number }[];
}

describe("interruptingEvents (shared contract)", () => {
  const fx = JSON.parse(fs.readFileSync(FIXTURE, "utf8")) as { levels: [number, number][]; cases: Case[] };

  it("uses the fixture's ladder", () => {
    expect(INTERRUPT_LEVELS).toEqual(fx.levels);
  });

  for (const c of fx.cases) {
    it(c.name, () => {
      const got = interruptingEvents(
        c.self.map(([start, end]) => ({ start, end })),
        c.other.map(([start, end]) => ({ start, end })),
      );
      expect(got.map((e) => ({ level: e.level, t: e.t, value: e.value }))).toEqual(c.events);
      expect(got.every((e) => e.vector === "interrupting")).toBe(true);
    });
  }
});
