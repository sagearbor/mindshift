/**
 * Text-heat lexicon (src/live/heatLexicon.ts): a curated phrase list that
 * scores an on-device transcript unit for counterproductive conduct —
 * contempt, blame absolutes, hostile commands, threats, directed profanity,
 * accusations of lying — and a buzz gate over it. DARK: FastLoopDeps.
 * heatLexiconBuzz, default off.
 *
 * Measured 2026-10-10 (tmp/recordings/fable_heat/REPORT.md, $0): at the
 * shipped threshold 1.5 it fired 10 times over the English dev+held-out
 * clips, 10/10 on units BOTH blind judges rated heated (5/10 on units both
 * rated clearly counterproductive), 0 times on judged-calm material, and
 * 0.87/h across 22 h of ordinary SBCSAE conversation.
 */
import { FastLoop } from "../src/live/fastLoop";
import {
  HEAT_LEXICON_DEFAULTS,
  HeatLexiconBuzz,
  scoreHeatText,
} from "../src/live/heatLexicon";
import vectors from "./fixtures/heatLexicon.vectors.json";

describe("scoreHeatText", () => {
  it("flags the canonical counterproductive lines at or above the shipped threshold", () => {
    for (const t of [
      "you re an idiot",
      "shut up",
      "it s your fault for getting pregnant",
      "you re probably lying about that mom",
      "arp you are a liar i m sorry that the truth hurts you",
      "don t start on me and i said stay out of it",
      "fuck you",
    ]) {
      expect(scoreHeatText(t).score).toBeGreaterThanOrEqual(HEAT_LEXICON_DEFAULTS.threshold);
    }
  });

  it("stays under the threshold on ordinary talk, third-person gossip and things (the SBCSAE false alarms)", () => {
    for (const t of [
      "we should look at the budget for next quarter",
      "you never have to use any batteries",
      "so what did we get out of all this",
      "he s really stupid",
      "well this son of a bitch weighed about fifteen pounds",
      "i don t ever remember us doing anything like that though",
      "that you lie flat",
      "is that enough",
    ]) {
      expect(scoreHeatText(t).score).toBeLessThan(HEAT_LEXICON_DEFAULTS.threshold);
    }
  });

  it("reports the categories it matched", () => {
    expect(scoreHeatText("you always do this and it s your fault").cats).toEqual(["C2"]);
    expect(scoreHeatText("shut up").cats).toEqual(["C3"]);
  });

  it("agrees with the Python reference on every evaluation vector (cross-language contract)", () => {
    const rows = vectors as { text: string; score: number; cats: string[] }[];
    expect(rows.length).toBeGreaterThan(150);
    const bad = rows.filter((r) => {
      const s = scoreHeatText(r.text);
      return Math.abs(s.score - r.score) > 1e-9 || s.cats.join(",") !== r.cats.join(",");
    });
    expect(bad.map((r) => [r.text, r.score, scoreHeatText(r.text).score])).toEqual([]);
  });

  it("is cheap: well under a millisecond per unit", () => {
    const rows = vectors as { text: string }[];
    const t0 = Date.now();
    for (let k = 0; k < 20; k++) for (const r of rows) scoreHeatText(r.text);
    const perUnitMs = (Date.now() - t0) / (20 * rows.length);
    expect(perUnitMs).toBeLessThan(1);
  });
});

describe("HeatLexiconBuzz", () => {
  it("ships the DEV-tuned defaults", () => {
    expect(HEAT_LEXICON_DEFAULTS).toEqual({ threshold: 1.5, cooldownS: 20 });
  });

  it("buzzes once per cooldown and never on a calm line", () => {
    const b = new HeatLexiconBuzz();
    expect(b.observe("we should look at the budget", 10)).toBe(false);
    expect(b.observe("you re an idiot", 12)).toBe(true);
    expect(b.observe("shut up", 20)).toBe(false); // inside the 20 s cooldown
    expect(b.observe("shut up", 33)).toBe(true);
    expect(b.observe("", 60)).toBe(false);
  });
});

describe("FastLoop heatLexiconBuzz wiring", () => {
  it("is off unless the flag is set", () => {
    const base = { vad: null as never, embedder: null, labeler: null } as never;
    expect(FastLoop.heatLexiconFrom(undefined)).toBeNull();
    expect(FastLoop.heatLexiconFrom(false)).toBeNull();
    expect(FastLoop.heatLexiconFrom(true)?.opts).toEqual(HEAT_LEXICON_DEFAULTS);
    expect(FastLoop.heatLexiconFrom({ threshold: 2 })?.opts).toEqual({ threshold: 2, cooldownS: 20 });
    void base;
  });
});
