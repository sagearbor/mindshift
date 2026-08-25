/**
 * Pleasantness scoreboard — PRD §6's per-turn score (0–100) computed from
 * what the phone already has for a live turn: the local/cloud suggestion's
 * `text_tone` (warmth, defensiveness, sarcasm, frustration, label), the
 * turn's prosody (loudness against the speaker's own running baseline,
 * speech rate) and the conversation's turn balance.
 *
 * This is a competition to be NICER, not to win: every dimension rewards
 * kindness, calm and taking turns, and a person only "leads" when they are
 * clearly ahead (≥ 3 points) — noise never crowns anyone.
 *
 * Mapping (weights per PRD §6; every rule is pinned by the golden vectors in
 * server/tests/fixtures/policy_vectors/pleasantness.json, which the Python
 * twin `server/pleasantness.py` replays too, so the live board and the
 * post-session view agree to the point):
 *
 *   warmth           30%  = text_tone.warmth
 *   constructiveness 25%  = 100 − text_tone.defensiveness
 *   calmness         20%  = (100 − frustration | neutral prior 70 when only
 *                            prosody is measurable) − loudness penalty
 *                            (4 pts/dB over +3 dB above the speaker's own
 *                            baseline, max 30) − 10 when > 3.5 words/s
 *   respect          15%  = 100 − text_tone.sarcasm; a contempt/dismissive
 *                            label caps it at 20
 *   engagement       10%  = turn balance over the last 6 turns (50/50 → 100);
 *                            null with one voice in the window
 *
 * Score = weighted mean over the dimensions actually measured (weights
 * renormalized), half-up rounded; null when none of the four content
 * dimensions is present (turn balance alone never scores a turn). Missing
 * inputs are honest nulls, never 0.
 *
 * Pure (no React, no I/O) so the same code scores the live loop, the replay
 * of a stored session and the tests.
 */

export interface PleasantnessTone {
  warmth?: number | null;
  defensiveness?: number | null;
  sarcasm?: number | null;
  sadness?: number | null;
  frustration?: number | null;
  label?: string | null;
}

export interface PleasantnessProsody {
  rms_dbfs?: number | null;
  pitch_hz?: number | null;
  speech_rate?: number | null;
}

export type Dimension = "warmth" | "constructiveness" | "calmness" | "respect" | "engagement";

export const DIMENSIONS: readonly Dimension[] = [
  "warmth",
  "constructiveness",
  "calmness",
  "respect",
  "engagement",
];

export const WEIGHTS: Record<Dimension, number> = {
  warmth: 0.3,
  constructiveness: 0.25,
  calmness: 0.2,
  respect: 0.15,
  engagement: 0.1,
};

export const NEUTRAL_CALM_PRIOR = 70;
export const LOUD_DB_FREE = 3;
export const LOUD_PENALTY_PER_DB = 4;
export const LOUD_PENALTY_MAX = 30;
export const FAST_RATE_WPS = 3.5;
export const FAST_PENALTY = 10;
export const CONTEMPT_RESPECT_CAP = 20;
export const CONTEMPT_LABELS: ReadonlySet<string> = new Set([
  "contempt",
  "contemptuous",
  "dismissive",
  "hostile",
  "mocking",
]);
export const BALANCE_WINDOW = 6;
export const CURRENT_WINDOW = 5;
export const SERIES_LENGTH = 10;
export const LEAD_MIN_MARGIN = 3;

export type Dims = Record<Dimension, number | null>;

export interface TurnScore {
  dims: Dims;
  /** 0–100, or null when the turn carried nothing scoreable. */
  score: number | null;
}

export interface PersonScore {
  /** The raw speaker label the tracker keys by. */
  speaker: string;
  /** Mean of the last CURRENT_WINDOW scored turns; null when none. */
  current: number | null;
  /** The last SERIES_LENGTH scored turns' scores, oldest first. */
  series: number[];
  /** How many of this speaker's turns scored. */
  scoredTurns: number;
}

export interface Lead {
  speaker: string;
  margin: number;
}

export interface Scoreboard {
  people: PersonScore[];
  /** Null when fewer than two people have scores or they're within
   *  LEAD_MIN_MARGIN of each other. */
  lead: Lead | null;
}

/** Half-up rounding for non-negative values (Math.round already is; the
 *  Python twin needs floor(x + 0.5) to avoid banker's rounding). */
export function roundHalfUp(x: number): number {
  return Math.floor(x + 0.5);
}

function clamp(x: number, lo = 0, hi = 100): number {
  return Math.max(lo, Math.min(hi, x));
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function cleanLabel(v: unknown): string | null {
  if (typeof v !== "string") return null;
  const s = v.trim().toLowerCase();
  return s || null;
}

/** Per-speaker running mean of rms_dbfs over PRIOR turns (the loudness
 *  baseline the penalty is measured against). */
export class LoudnessBaselines {
  private readonly sums = new Map<string, { sum: number; n: number }>();

  /** The baseline BEFORE folding this turn in; null with no prior sample. */
  baselineFor(speaker: string): number | null {
    const s = this.sums.get(speaker);
    return s && s.n > 0 ? s.sum / s.n : null;
  }

  observe(speaker: string, rmsDbfs: number | null): void {
    if (rmsDbfs === null) return;
    const s = this.sums.get(speaker) ?? { sum: 0, n: 0 };
    s.sum += rmsDbfs;
    s.n += 1;
    this.sums.set(speaker, s);
  }

  rename(from: string, to: string): void {
    const s = this.sums.get(from);
    if (!s) return;
    this.sums.delete(from);
    const t = this.sums.get(to);
    this.sums.set(to, t ? { sum: t.sum + s.sum, n: t.n + s.n } : s);
  }
}

/** Turn-balance engagement over a window of speaker labels (the current
 *  turn included). Null unless ≥ 2 turns from ≥ 2 speakers are in it. */
export function engagementFromWindow(window: readonly string[], speaker: string): number | null {
  if (window.length < 2) return null;
  const distinct = new Set(window);
  if (distinct.size < 2) return null;
  const mine = window.filter((s) => s === speaker).length;
  const share = mine / window.length;
  return clamp(roundHalfUp(100 - 200 * Math.abs(share - 0.5)));
}

export interface ScoreTurnContext {
  /** Mean rms_dbfs of this speaker's prior turns; null with none. */
  baselineRms: number | null;
  /** Already-computed engagement (see engagementFromWindow). */
  engagement: number | null;
}

/** Score ONE turn from its tone + prosody and the tracker-supplied context. */
export function scoreTurn(
  textTone: PleasantnessTone | null | undefined,
  prosody: PleasantnessProsody | null | undefined,
  ctx: ScoreTurnContext,
): TurnScore {
  const tone = textTone ?? {};
  const pros = prosody ?? {};
  const warmthIn = num(tone.warmth);
  const defensiveness = num(tone.defensiveness);
  const sarcasm = num(tone.sarcasm);
  const frustration = num(tone.frustration);
  const label = cleanLabel(tone.label);
  const rms = num(pros.rms_dbfs);
  const rate = num(pros.speech_rate);

  const warmth = warmthIn === null ? null : clamp(warmthIn);
  const constructiveness = defensiveness === null ? null : clamp(100 - defensiveness);

  let respect = sarcasm === null ? null : clamp(100 - sarcasm);
  if (label !== null && CONTEMPT_LABELS.has(label)) {
    respect = respect === null ? CONTEMPT_RESPECT_CAP : Math.min(respect, CONTEMPT_RESPECT_CAP);
  }

  // Calmness: penalties are only "measurable" with something to measure —
  // a loudness delta needs a baseline; a speed penalty needs a rate.
  let loudPenalty = 0;
  let loudMeasurable = false;
  if (rms !== null && ctx.baselineRms !== null) {
    loudMeasurable = true;
    const over = rms - ctx.baselineRms;
    if (over > LOUD_DB_FREE) {
      loudPenalty = Math.min(LOUD_PENALTY_MAX, roundHalfUp(LOUD_PENALTY_PER_DB * (over - LOUD_DB_FREE)));
    }
  }
  const fastMeasurable = rate !== null;
  const fastPenalty = rate !== null && rate > FAST_RATE_WPS ? FAST_PENALTY : 0;
  let calmness: number | null = null;
  if (frustration !== null) {
    calmness = clamp(100 - frustration - loudPenalty - fastPenalty);
  } else if (loudMeasurable || fastMeasurable) {
    calmness = clamp(NEUTRAL_CALM_PRIOR - loudPenalty - fastPenalty);
  }

  const dims: Dims = {
    warmth,
    constructiveness,
    calmness,
    respect,
    engagement: ctx.engagement,
  };
  const content = [warmth, constructiveness, calmness, respect].some((v) => v !== null);
  if (!content) return { dims, score: null };
  let weighted = 0;
  let weightSum = 0;
  for (const d of DIMENSIONS) {
    const v = dims[d];
    if (v === null) continue;
    weighted += WEIGHTS[d] * v;
    weightSum += WEIGHTS[d];
  }
  return { dims, score: clamp(roundHalfUp(weighted / weightSum)) };
}

/** The lead line from a people list (exported so a stored board can be
 *  re-read without a tracker). */
export function leadOf(people: readonly PersonScore[]): Lead | null {
  const scored = people
    .filter((p): p is PersonScore & { current: number } => typeof p.current === "number")
    .sort((a, b) => b.current - a.current);
  if (scored.length < 2) return null;
  const margin = scored[0].current - scored[1].current;
  return margin >= LEAD_MIN_MARGIN ? { speaker: scored[0].speaker, margin } : null;
}

/**
 * Session-scoped tracker: feed turns in order, read the board any time.
 * Keys people by the RAW speaker label; `rename` folds a label into another
 * when the user names a speaker mid-call (or a label is revised).
 */
export class PleasantnessTracker {
  private readonly baselines = new LoudnessBaselines();
  private readonly window: string[] = [];
  private readonly order: string[] = [];
  private readonly scores = new Map<string, number[]>();

  observe(
    speaker: string,
    textTone: PleasantnessTone | null | undefined,
    prosody: PleasantnessProsody | null | undefined,
  ): TurnScore {
    this.window.push(speaker);
    if (this.window.length > BALANCE_WINDOW) this.window.shift();
    const engagement = engagementFromWindow(this.window, speaker);
    const result = scoreTurn(textTone, prosody, {
      baselineRms: this.baselines.baselineFor(speaker),
      engagement,
    });
    this.baselines.observe(speaker, num(prosody?.rms_dbfs));
    if (!this.order.includes(speaker)) this.order.push(speaker);
    const list = this.scores.get(speaker) ?? [];
    if (result.score !== null) list.push(result.score);
    this.scores.set(speaker, list);
    return result;
  }

  /** Fold `from` into `to` (a mid-call "that's Mom" on Speaker B). */
  rename(from: string, to: string): void {
    if (from === to) return;
    for (let i = 0; i < this.window.length; i++) if (this.window[i] === from) this.window[i] = to;
    this.baselines.rename(from, to);
    const fromScores = this.scores.get(from);
    if (fromScores) {
      this.scores.delete(from);
      this.scores.set(to, [...(this.scores.get(to) ?? []), ...fromScores]);
    }
    const idx = this.order.indexOf(from);
    if (idx >= 0) {
      if (this.order.includes(to)) this.order.splice(idx, 1);
      else this.order[idx] = to;
    }
  }

  board(): Scoreboard {
    const people = this.order.map((speaker) => personScore(speaker, this.scores.get(speaker) ?? []));
    return { people, lead: leadOf(people) };
  }
}

export function personScore(speaker: string, scored: readonly number[]): PersonScore {
  const recent = scored.slice(-CURRENT_WINDOW);
  const current = recent.length ? roundHalfUp(recent.reduce((a, b) => a + b, 0) / recent.length) : null;
  return { speaker, current, series: scored.slice(-SERIES_LENGTH), scoredTurns: scored.length };
}

/** One stored/live turn as the batch scorer reads it. */
export interface ScorableTurn {
  speaker: string;
  text_tone?: PleasantnessTone | null;
  prosody?: PleasantnessProsody | null;
}

/** Score a whole transcript (a stored live session) in one pass. */
export function scoreSession(turns: readonly ScorableTurn[]): { perTurn: TurnScore[]; board: Scoreboard } {
  const tracker = new PleasantnessTracker();
  const perTurn = turns.map((t) => tracker.observe(t.speaker, t.text_tone ?? null, t.prosody ?? null));
  return { perTurn, board: tracker.board() };
}
