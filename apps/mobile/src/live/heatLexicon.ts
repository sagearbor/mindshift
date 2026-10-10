/**
 * TEXT-HEAT LEXICON — "something counterproductive was just SAID", from the
 * on-device transcript alone, in microseconds, no model and no cloud.
 * DARK: FastLoopDeps.heatLexiconBuzz, default off.
 *
 * Why words: loudness measures arousal and fires on delighted shouting as
 * hard as on anger (2026-09-10), and the frozen audio embeddings did not
 * transfer across rooms (2026-10-07). The owner's framing (2026-10-10): a
 * rare but ACCURATE intervention beats a frequent one, and "the words alone
 * could be found to be counterproductive". So this is a curated phrase list
 * — contempt, blame absolutes ("it's your fault"), hostile commands ("shut
 * up"), threats, profanity AIMED at a person, accusations of lying — each
 * with a weight; a unit's score is the sum of the weights of the patterns it
 * matches, and the buzz gate asks for >= threshold, then a cooldown.
 *
 * The list is generated from tmp/recordings/fable_heat/lexicon.py (v3) into
 * heatLexicon.lexicon.json; __tests__/heatLexicon.test.ts replays the Python
 * scorer's vectors, so both runtimes score identically.
 *
 * Measured (REPORT.md there, $0, two blind LLM judges on 53 English clips,
 * split by session before any label existed): at threshold 1.5 the detector
 * fired 10 times on dev+held-out, every fire on a unit BOTH judges rated
 * heated (10/10), half on units both rated clearly counterproductive, zero
 * fires on judge-calm material, and 0.87 fires/h across 22 h of ordinary
 * SBCSAE conversation (mostly jokes among friends and third-person swearing).
 * Reach is deliberately low: ~1 in 3 clear moments, ~1 in 6 mild ones.
 * Delay from onset: ~6 s at turn end; ~4 s if run on interim text.
 *
 * What it is NOT: about the wearer (no identity), and not a measure of
 * anger — a heated TOPIC scores 0; only heated CONDUCT scores.
 */
import LEXICON from "./heatLexicon.lexicon.json";

export interface HeatLexiconEntry {
  pattern: string;
  weight: number;
  cat: string;
}

export interface HeatTextScore {
  /** Sum of the weights of the matched patterns (0 = nothing matched). */
  score: number;
  /** Matched categories C1..C7 (DEFINITION.md in the experiment dir), sorted, unique. */
  cats: string[];
  /** The matched snippets, for logs. */
  hits: string[];
}

const COMPILED: { rx: RegExp; weight: number; cat: string }[] = (
  LEXICON as { patterns: HeatLexiconEntry[] }
).patterns.map((p) => ({ rx: new RegExp(p.pattern, "i"), weight: p.weight, cat: p.cat }));

/** Score one transcript unit. Case-insensitive; tolerant of ASR apostrophe
 *  loss ("you re", "don t"), which the patterns were written for. */
export function scoreHeatText(text: string): HeatTextScore {
  let score = 0;
  const cats = new Set<string>();
  const hits: string[] = [];
  if (!text) return { score: 0, cats: [], hits: [] };
  const t = text.toLowerCase();
  for (const p of COMPILED) {
    const m = p.rx.exec(t);
    if (m) {
      score += p.weight;
      cats.add(p.cat);
      hits.push(m[0]);
    }
  }
  return { score, cats: [...cats].sort(), hits };
}

export interface HeatLexiconBuzzOptions {
  /** Minimum unit score to buzz. 1.5 = one strong marker or two weak ones. */
  threshold: number;
  /** Seconds of silence from this gate after a buzz (a tirade counts once). */
  cooldownS: number;
}

export const HEAT_LEXICON_DEFAULTS: HeatLexiconBuzzOptions = { threshold: 1.5, cooldownS: 20 };

export class HeatLexiconBuzz {
  readonly opts: HeatLexiconBuzzOptions;
  private lastBuzzT = -Infinity;
  /** The last unit that crossed the threshold (for diagnostics). */
  last: { t: number; score: HeatTextScore } | null = null;

  constructor(opts: Partial<HeatLexiconBuzzOptions> = {}) {
    this.opts = { ...HEAT_LEXICON_DEFAULTS, ...opts };
  }

  /** Feed one finalized (or interim) transcript unit ending at audio time `t` (s). */
  observe(text: string, t: number): boolean {
    const s = scoreHeatText(text);
    if (s.score < this.opts.threshold) return false;
    this.last = { t, score: s };
    if (t - this.lastBuzzT < this.opts.cooldownS) return false;
    this.lastBuzzT = t;
    return true;
  }
}
