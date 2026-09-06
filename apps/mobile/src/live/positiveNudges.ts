/**
 * Positive nudges — the four codes that tell you what you did WELL.
 *
 * Every nudge shipped before this one was a complaint: too loud, too hot, you
 * cut in, you're hogging. A coach that only ever says "stop that" is one a
 * person turns off. D / E / R / K (nudge vocabulary, owner-approved
 * 2026-09-06) are the other half:
 *
 *   📉 D  De-escalated  heat dropped a level within two turns of a spike
 *   👂 E  Listened      you let them finish a long turn without cutting in
 *   🤝 R  Repair        you validated or apologised and their tone softened
 *   🧘 K  Calm streak   N minutes with no escalation — a SUMMARY badge, never live
 *
 * Rules that keep praise from becoming its own nag:
 *   - one positive cue per two minutes across D/E/R together (the shared
 *     `PositiveNudgeGate`); a detection the cap drops is still REPORTED, with
 *     `delivered: false`, so a replay can show what the user nearly felt;
 *   - the cue is soft (nudgeVocabulary.ts) and unleveled — a "level 2 well
 *     done" is not a thing;
 *   - K never buzzes at all.
 *
 * Pure and language-neutral, like the rest of the policy brain: the contract
 * is server/tests/fixtures/policy_vectors/positive_nudges.json, replayed by
 * __tests__/positiveNudges.test.ts and by
 * server/tests/test_positive_nudges_vectors.py against the Python mirror
 * server/positive_nudges.py.
 */
import { aggressiveToneLevel } from "./nudgePolicy";
import { PositiveNudgeGate } from "./nudgeVocabulary";

export type PositiveCode = "D" | "E" | "R";

/**
 * One finalized turn, reduced to what the positive detectors need. Built from
 * a `LocalTurn` by [positiveTurnFromLocal] on the phone and from the analysis
 * rows on the server — neither detector ever sees audio.
 */
export interface PositiveTurn {
  index: number;
  start: number;
  end: number;
  /** Whose voice the loop believes this is — a cluster label or a person's
   *  name. [coalesceTurns] groups on it: a turn ends when someone ELSE
   *  speaks, which is the definition code E depends on. */
  speaker: string;
  isSelf: boolean;
  text: string;
  /** The loudness rung this turn raised (yellingLevel over the speaker's own
   *  baseline), or null when loudness could not be measured. */
  loudLevel: number | null;
  /** AGGRESSION: max(frustration, defensiveness), 0..100, or null when the
   *  turn was not scored. This is the self side's heat — the same rung the
   *  live policy's `aggressive_tone` vector reads. */
  toneHeat: number | null;
  /** NEGATIVE AFFECT: max(frustration, defensiveness, sadness), 0..100, or
   *  null. What R measures the other person's softening on — and the reason
   *  it is a separate number: someone you have just shouted at usually goes
   *  HURT, not aggressive, so an aggression-only reading would score a real
   *  repair as no change at all (measured on the couple_escalation fixture,
   *  2026-09-06: their aggression moved 5 points while their sadness moved
   *  65). Heat is never measured on this — being sad is not being heated. */
  toneNegativity: number | null;
  /** This self turn started inside the previous partner turn and kept going
   *  long enough to raise a `cut_in` (nudgePolicy.ts `interruptingEvents`). */
  cutIn: boolean;
}

export interface PositiveNudge {
  code: PositiveCode;
  /** Session seconds the detection landed on. */
  t: number;
  /** The turn that earned it. */
  turnIndex: number;
  /** False when the two-minute cap dropped it — detected, not felt. */
  delivered: boolean;
  /** One line a report (or Developer mode) can show verbatim. */
  detail: string;
}

/** K is not an event: it is a number on the summary. */
export interface CalmStreak {
  /** The longest run of session seconds with no alert nudge. */
  longestS: number;
  /** Whether that run earns the 🧘 badge. */
  badge: boolean;
}

export interface PositiveResult {
  nudges: PositiveNudge[];
  calm: CalmStreak;
}

// ---------------------------------------------------------------------------
// Constants — each one pinned by the fixture on both runtimes.
// ---------------------------------------------------------------------------

/** A partner turn at least this long is one you had to actively let run. The
 *  CANDOR median turn is a couple of seconds; 12 s is someone holding the
 *  floor, which is exactly when the urge to cut in shows up. */
export const LISTEN_MIN_TURN_S = 12.0;

/** How many of YOUR turns a de-escalation may take. "Within two turns of a
 *  spike" (owner, 2026-09-06): later than that and the calm is not a recovery,
 *  it is just a different part of the conversation. A turn whose heat could
 *  not be measured still spends one of the two — silence is not evidence. */
export const DEESCALATION_WINDOW_TURNS = 2;

/** How far the other person's tone must fall for "they softened" to be a
 *  measurement rather than noise (0..100 points). */
export const REPAIR_SOFTEN_DROP = 15;

/** No escalation for this long earns the 🧘 badge. */
export const CALM_STREAK_MIN_S = 300.0;

/**
 * Repair language: apology and validation, the two moves that actually turn a
 * fight around. Matched as substrings of the lowercased, punctuation-stripped
 * turn — deliberately a small, literal list rather than a classifier, because
 * a false "nice repair!" after something that was not one is worse than
 * saying nothing. Apostrophe-less spellings are listed because on-device STT
 * drops them.
 */
export const REPAIR_PHRASES: readonly string[] = [
  "im sorry",
  "i am sorry",
  "i apologize",
  "i apologise",
  "my bad",
  "that was on me",
  "youre right",
  "you are right",
  "thats fair",
  "that is fair",
  "fair enough",
  "i hear you",
  "i understand",
  "i get that",
  "that makes sense",
  "i see what you mean",
  "i shouldnt have",
  "i should not have",
  "i didnt mean",
  "i did not mean",
  "let me try again",
];

/** Lowercase, drop everything that is not a letter/digit/space, collapse runs
 *  of spaces — so "I'm sorry." and "im  sorry" both match "im sorry". */
export function normalizeForRepair(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9 ]+/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/** True when a turn says one of the repair moves. */
export function hasRepairLanguage(text: string): boolean {
  const norm = normalizeForRepair(text);
  if (!norm) return false;
  return REPAIR_PHRASES.some((p) => norm.includes(p));
}

/** The alert level a self turn raised: the higher of its loudness rung and
 *  its text-tone rung (the same two ladders the live policy maxes). Null when
 *  neither could be measured — never 0-as-a-guess. */
export function selfHeatLevel(turn: PositiveTurn): number | null {
  const tone = turn.toneHeat === null ? null : aggressiveToneLevel(turn.toneHeat, null);
  if (turn.loudLevel === null && tone === null) return null;
  return Math.max(turn.loudLevel ?? 0, tone ?? 0);
}

// ---------------------------------------------------------------------------
// Fragments -> turns
// ---------------------------------------------------------------------------

/**
 * Merge consecutive same-speaker fragments into the conversational turns the
 * detectors reason about.
 *
 * This is not cosmetic. The live loop's "turn" is a VAD fragment — it cuts at
 * a 300 ms pause — so someone holding the floor for sixteen seconds arrives as
 * five three-second pieces, and code E ("you let them finish a LONG turn")
 * would never fire at all. A turn ends when somebody ELSE starts talking;
 * that is what this restores.
 *
 * Merged fields take the worst/widest reading, because a turn is as loud as
 * its loudest moment and as hot as its hottest: `loudLevel`, `toneHeat` and
 * `toneNegativity` are maxima over the scored fragments (null only when NO
 * fragment was scored — never 0-as-a-guess), `cutIn` is true if any fragment
 * cut in, and `index` stays the FIRST fragment's so the turn is still
 * addressable in the report.
 */
export function coalesceTurns(fragments: readonly PositiveTurn[]): PositiveTurn[] {
  const out: PositiveTurn[] = [];
  for (const f of fragments) {
    const prev = out[out.length - 1];
    if (!prev || prev.speaker !== f.speaker) {
      out.push({ ...f });
      continue;
    }
    const widen = (a: number | null, b: number | null) => (a === null ? b : b === null ? a : Math.max(a, b));
    out[out.length - 1] = {
      ...prev,
      end: Math.max(prev.end, f.end),
      text: prev.text && f.text ? `${prev.text} ${f.text}` : prev.text || f.text,
      loudLevel: widen(prev.loudLevel, f.loudLevel),
      toneHeat: widen(prev.toneHeat, f.toneHeat),
      toneNegativity: widen(prev.toneNegativity, f.toneNegativity),
      cutIn: prev.cutIn || f.cutIn,
    };
  }
  return out;
}

// ---------------------------------------------------------------------------
// The detectors
// ---------------------------------------------------------------------------

/**
 * Run the positive detectors over a whole conversation.
 *
 * `alertNudgeTimes` are the session seconds at which ALERT nudges fired (the
 * live policy's escalations); K is the longest gap between them, bounded by
 * `sessionEndS`. Turns must be in time order.
 */
export function detectPositiveNudges(
  turns: readonly PositiveTurn[],
  alertNudgeTimes: readonly number[] = [],
  sessionEndS: number | null = null,
  capS?: number,
): PositiveResult {
  const gate = new PositiveNudgeGate(capS);
  const out: PositiveNudge[] = [];

  // D: the most recent self turn that raised heat, and how many self turns
  // have gone by since. Reset whenever heat is raised again at or above it —
  // a spike that keeps repeating is not a recovery in progress.
  let spikeLevel: number | null = null;
  let selfTurnsSinceSpike = 0;

  // R: the other person's NEGATIVE AFFECT before and after the self turn that
  // tried to repair. `pendingRepair` is the self turn we are waiting on the
  // partner to answer; `lastPartnerNegativity` is how they sounded BEFORE it.
  let lastPartnerNegativity: number | null = null;
  let pendingRepair: { turn: PositiveTurn; before: number } | null = null;

  const emit = (code: PositiveCode, t: number, turnIndex: number, detail: string) => {
    out.push({ code, t, turnIndex, delivered: gate.admit(t), detail });
  };

  for (const turn of turns) {
    if (turn.isSelf) {
      const heat = selfHeatLevel(turn);

      // --- D: did this turn pull the heat back down? ---
      if (spikeLevel !== null) {
        selfTurnsSinceSpike += 1;
        if (heat !== null && heat < spikeLevel) {
          emit(
            "D",
            turn.end,
            turn.index,
            `heat ${spikeLevel} -> ${heat} within ${selfTurnsSinceSpike} turn${selfTurnsSinceSpike === 1 ? "" : "s"} of the spike`,
          );
          spikeLevel = null;
          selfTurnsSinceSpike = 0;
        } else if (selfTurnsSinceSpike >= DEESCALATION_WINDOW_TURNS) {
          // The window closed without a measured drop. If this turn is itself
          // a spike it becomes the new one below; otherwise we stop watching.
          spikeLevel = null;
          selfTurnsSinceSpike = 0;
        }
      }
      if (heat !== null && heat >= 1 && (spikeLevel === null || heat >= spikeLevel)) {
        spikeLevel = heat;
        selfTurnsSinceSpike = 0;
      }

      // --- R: remember an attempt, to be judged by their next turn ---
      if (hasRepairLanguage(turn.text) && lastPartnerNegativity !== null) {
        pendingRepair = { turn, before: lastPartnerNegativity };
      }
      continue;
    }

    // --- partner turn ---
    // E: a long turn you did not cut into. Judged at its end, because a turn
    // that has ended can no longer be interrupted.
    if (turn.end - turn.start >= LISTEN_MIN_TURN_S && !cutIntoBy(turns, turn)) {
      emit(
        "E",
        turn.end,
        turn.index,
        `let a ${Math.round(turn.end - turn.start)} s turn finish with no cut-in`,
      );
    }

    // R: their answer to the repair attempt. "Their tone softened NEXT turn"
    // means the very next partner turn — so the attempt is spent here either
    // way, including when that turn was never scored (an unmeasured turn is
    // not evidence of softening, and waiting for a later one would let a
    // repair claim credit for a mood change minutes afterwards).
    if (pendingRepair) {
      if (
        turn.toneNegativity !== null &&
        pendingRepair.before - turn.toneNegativity >= REPAIR_SOFTEN_DROP
      ) {
        emit(
          "R",
          turn.end,
          pendingRepair.turn.index,
          `repair language, then their tone fell ${Math.round(pendingRepair.before - turn.toneNegativity)} points`,
        );
      }
      pendingRepair = null;
    }
    if (turn.toneNegativity !== null) lastPartnerNegativity = turn.toneNegativity;
  }

  return { nudges: out, calm: calmStreak(alertNudgeTimes, sessionEndS ?? lastEnd(turns)) };
}

/** True when any SELF turn started strictly inside `partner` and was tagged a
 *  cut-in — you talked over them, not just past their last word. */
function cutIntoBy(turns: readonly PositiveTurn[], partner: PositiveTurn): boolean {
  return turns.some((t) => t.isSelf && t.cutIn && t.start > partner.start && t.start < partner.end);
}

function lastEnd(turns: readonly PositiveTurn[]): number {
  return turns.length === 0 ? 0 : Math.max(...turns.map((t) => t.end));
}

/**
 * 🧘 K — the longest stretch of the session with no alert nudge, and whether
 * it earns the badge. Counted from 0 to the first alert, between consecutive
 * alerts, and from the last alert to `sessionEndS`, so a session that was calm
 * throughout is one long streak rather than none.
 */
export function calmStreak(alertNudgeTimes: readonly number[], sessionEndS: number): CalmStreak {
  const marks = [...alertNudgeTimes].filter((t) => Number.isFinite(t)).sort((a, b) => a - b);
  let longest = 0;
  let prev = 0;
  for (const t of marks) {
    longest = Math.max(longest, t - prev);
    prev = t;
  }
  longest = Math.max(longest, sessionEndS - prev);
  longest = Math.max(0, longest);
  return { longestS: longest, badge: longest >= CALM_STREAK_MIN_S };
}

// ---------------------------------------------------------------------------
// Live
// ---------------------------------------------------------------------------

/**
 * The live wrapper: feed it each finalized turn and it hands back the positive
 * nudges that are NEW since the last call.
 *
 * It re-runs [detectPositiveNudges] over the whole conversation each time
 * rather than keeping its own incremental state machine. That is deliberate,
 * and it is the point: what the phone does live is then the SAME function a
 * replay runs over the recorded file, so the nudge report cannot claim a
 * behaviour the device does not have. A conversation is a few hundred turns —
 * the cost is nothing next to the STT and LLM calls on the same path.
 *
 * `alertNudgeTimes` (for K) are pushed in by the caller as the alert policy
 * escalates, since those come from a different machine entirely.
 */
export class LivePositiveNudger {
  private readonly turns: PositiveTurn[] = [];
  private readonly alerts: number[] = [];
  private emitted = 0;

  constructor(private readonly capS?: number) {}

  /** Record an ALERT escalation, so K can measure the quiet between them. */
  onAlert(t: number): void {
    this.alerts.push(t);
  }

  /** Record a finalized FRAGMENT (the loop's own turn); returns the positives
   *  the conversation newly earned once fragments are coalesced back into
   *  turns (in
   *  emission order, including any the cap dropped — the caller buzzes only
   *  for `delivered`). */
  onTurn(turn: PositiveTurn): PositiveNudge[] {
    this.turns.push(turn);
    const all = detectPositiveNudges(coalesceTurns(this.turns), this.alerts, null, this.capS).nudges;
    const fresh = all.slice(this.emitted);
    this.emitted = all.length;
    return fresh;
  }

  /** 🧘 for the session summary. `sessionEndS` bounds the last quiet run. */
  calm(sessionEndS: number): CalmStreak {
    return calmStreak(this.alerts, sessionEndS);
  }
}
