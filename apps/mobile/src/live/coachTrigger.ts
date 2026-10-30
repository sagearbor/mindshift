/**
 * QUIET COACH — the cheap trigger gate in front of every free-form coach line.
 *
 * Before 2026-10-29 the coach answered every substantive turn: the phone's
 * local LLM line was voiced whenever it produced one, and the cloud LLM ran
 * on every turn (hundreds of calls an hour; the server's speak gate then
 * threw most lines away). For an outside tester the rule is the owner's:
 * a rare but accurate intervention beats a frequent one. So a free-form line
 * is only VOICED (phone) / only GENERATED (server, server/coach_trigger.py)
 * when a trigger holds on the turn:
 *
 *   heat      the words score >= 1.5 on the heated-words lexicon
 *             (live/heatLexicon.ts — the same list the buzz uses)
 *   floor     the mid-turn floor cue fired during this turn (live/floorTick.ts)
 *   question  someone other than the wearer asked a direct question
 *   library   (server only) the turn mentions a phrase from a selected
 *             Coach library item — the phone never holds the library text
 *
 * The phone sends the triggers it saw on `turn_local.triggers`; the server
 * unions them with its own. `isDirectQuestion` mirrors the server's
 * `coach_trigger.is_direct_question`; both are pinned by the shared vectors
 * in __tests__/fixtures/coachTrigger.vectors.json.
 */
import { scoreHeatText, HEAT_LEXICON_DEFAULTS } from "./heatLexicon";

export type CoachTrigger = "heat" | "floor" | "question" | "library";

export interface QuietCoachOptions {
  /** Heat lexicon score that counts as a trigger (the buzz threshold). */
  heatMin: number;
  heat: boolean;
  floor: boolean;
  question: boolean;
}

export const QUIET_COACH_DEFAULTS: QuietCoachOptions = {
  heatMin: HEAT_LEXICON_DEFAULTS.threshold,
  heat: true,
  floor: true,
  question: true,
};

const WH = "(?:what|why|how|when|where|who|whom|whose|which)";
const AUX =
  "(?:do|does|did|are|is|was|were|can|could|would|will|should|have|has|had|" +
  "shall|may|might|must|won't|wont|don't|dont|didn't|didnt|aren't|arent|isn't|isnt|" +
  "can't|cant|wouldn't|wouldnt|shouldn't|shouldnt|haven't|havent)";
const OPENER = new RegExp(`^(?:(?:so|and|but|okay|ok|well|um|uh|hey|then|like|now)[, ]+)*(?:${WH}|${AUX})\\b`, "i");
const YOU = /\b(?:you|your|yours|yourself|you're|youre|u|ya)\b/i;
const TAG = new RegExp(
  "^(?:(?:you know|right|okay|ok|yeah|huh|eh|no|yes|really|isn't it|isnt it|" +
    "don't you think|dont you think|you see|see|or what|what|sorry|pardon)[ ,.!?]*)+$",
  "i",
);
const TRAILING_TAG = /[, ]+(?:you know|right|okay|ok|yeah|huh|eh)\s*\?*$/i;

// No lookbehind / Unicode property escapes: plain regex for Hermes.
function words(s: string): string[] {
  return s.match(/[A-Za-z0-9_'À-￿]+/g) ?? [];
}

/** Last sentence: a boundary is [.!?] followed by whitespace (the server
 *  splits on the same boundary). */
function lastSentence(text: string): string {
  const parts = (text.trim().match(/(?:[^.!?]|[.!?](?=\S))+[.!?]*/g) ?? [])
    .map((p) => p.trim())
    .filter(Boolean);
  return parts.length ? parts[parts.length - 1] : "";
}

export const QUESTION_MIN_WORDS = 5;
const DISCOURSE =
  "(?:So|Well|Okay|OK|Ok|And|But|Now|Then|Like|Um|Uh|Hey|Yes|No|Yeah|Look|Listen|Honestly|" +
  "Seriously|Sorry|Please|Oh|Wait|Right|Alright|Anyway|Actually|Man|Dude|Guys|God)";
// A question put to a NAMED person ("Where'd you go, Mel?") is casual
// address to whoever was named, not a moment to coach the wearer.
const UNPUNCT_Q = new RegExp(
  `^(?:(?:so|and|but|okay|ok|well|um|uh|hey|then|now)[, ]+)*(?:${WH}\\s+)?${AUX}\\s+(?:you|u|ya)\\b`,
  "i",
);
const VOCATIVE_END =/,\s*[A-Z][a-z]+\s*[?!.]*$/;
const VOCATIVE_START = new RegExp(`^(?:[Hh]ey\\s+)?(?!${DISCOURSE}\\b)[A-Z][a-z]+,\\s`);

/** The turn's LAST sentence is a question put to the listener (v2, see the
 *  server's coach_trigger.is_direct_question — same rule, same vectors):
 *  addresses "you", >= QUESTION_MIN_WORDS words, ends in "?" or opens with
 *  an interrogative, not addressed to a name, not a tag question. */
export function isDirectQuestion(text: string): boolean {
  const last = lastSentence(text ?? "");
  if (!last || TAG.test(last)) return false;
  const body = last.replace(TRAILING_TAG, "").trim();
  if (words(body).length < QUESTION_MIN_WORDS) return false;
  if (VOCATIVE_END.test(body) || VOCATIVE_START.test(body)) return false;
  if (!YOU.test(body)) return false;
  const opener = OPENER.test(body);
  const endsQ = last.endsWith("?");
  if (endsQ && body !== last.replace(/\?+$/, "").trim() && !opener) return false;
  // Without a "?" (on-device STT drops punctuation) only an inverted
  // question counts ("do you…", "what do you…"); a bare wh-word opener is
  // usually a clause ("when you fold it open").
  return endsQ || UNPUNCT_Q.test(body);
}

/** Every phone-visible trigger on one finalized turn. */
export function turnTriggers(
  input: { text: string; isSelf: boolean | null; floorFired: boolean },
  opts: QuietCoachOptions = QUIET_COACH_DEFAULTS,
): CoachTrigger[] {
  const out: CoachTrigger[] = [];
  if (opts.heat && input.text && scoreHeatText(input.text).score >= opts.heatMin) out.push("heat");
  if (opts.floor && input.floorFired) out.push("floor");
  if (opts.question && input.isSelf !== true && isDirectQuestion(input.text)) out.push("question");
  return out;
}

/** A FastLoopDeps.quietCoach flag value -> options (null = off). */
export function quietCoachFrom(flag: boolean | Partial<QuietCoachOptions> | undefined): QuietCoachOptions | null {
  if (!flag) return null;
  return { ...QUIET_COACH_DEFAULTS, ...(typeof flag === "object" ? flag : {}) };
}
