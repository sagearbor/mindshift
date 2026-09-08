/**
 * NaturalTurn port — segmenting speech into psychologically meaningful
 * turns (Cooney & Reece 2025, Sci Reports 41598-025-24381-1; upstream
 * python: github.com/betterup/natural-turn-transcription, MIT). Constants
 * and decision order mirror the upstream `natural_turn` preset
 * (transcription/transcript_config.py + text.py::determine_utterance_type)
 * so our tags mean the same thing their published transcripts mean.
 *
 * Why: raw ASR treats every listener "yeah"/"uh-huh" (~1,000/hour in real
 * conversation) as a full turn — median turn collapses to 0.74 s and the
 * coach can answer an "mhm". NaturalTurn recovers 6.6 s median turns and
 * is the version of the data where turn length actually correlates with
 * enjoyment (r=.14 vs r=.002 on raw turns).
 *
 * Two consumers:
 * - LIVE (one mic, sequential VAD spans — overlap containment is not
 *   observable): `liveTurnKind` tags each finalized turn "backchannel" |
 *   "primary" from its words + duration. fastLoop uses it to keep the
 *   coach, the speaker clusters, and the stats off listener noises.
 * - BATCH (channelized or diarized utterances with real overlap, e.g. the
 *   server's transcripts or CANDOR-style data): `labelTurns` applies the
 *   full containment rule, then `mergePrimaries` joins a speaker's
 *   consecutive primary turns across pauses <= MAX_PAUSE_SECONDS.
 */

/** Upstream BACKCHANNEL_CUES (single tokens; the two-word "mm hmm" cue is
 *  inert upstream too — their Matcher builds one-token patterns). */
export const BACKCHANNEL_CUES: ReadonlySet<string> = new Set([
  "a", "ah", "alright", "awesome", "cool", "dope", "e", "exactly", "god",
  "gotcha", "huh", "hmm", "mhm", "mm", "mmm", "nice", "oh", "okay",
  "really", "right", "sick", "sucks", "sure", "uh", "um", "wow", "yeah",
  "yep", "yes", "yup",
  // ASR (AWS/Deepgram) writes "ok" not "okay"; the upstream literal cue
  // never matches it on real transcripts (found mining CANDOR, 2026-09-05).
  "ok",
]);

/** Upstream NOT_BACKCHANNEL_CUES: a short turn STARTING with one of these
 *  is someone starting a real thought ("and then…", "but I…"), never a
 *  backchannel. */
export const NOT_BACKCHANNEL_CUES: ReadonlySet<string> = new Set([
  "and", "but", "i", "i'm", "it", "it's", "like", "so", "that", "that's",
  "we", "we're", "well", "you", "you're",
]);

export const BACKCHANNEL_WORD_MAX = 3;
export const BACKCHANNEL_SECOND_MAX = 0.0; // natural_turn preset default
export const BACKCHANNEL_PROPORTION = 0.5;
/** Same-speaker primary turns closer than this merge into one turn. */
export const MAX_PAUSE_SECONDS = 1.5;

export type UtteranceKind = "primary" | "backchannel" | "secondary" | "other";

/** Lowercased word tokens with punctuation stripped (mirror of spacy's
 *  non-punct tokens for our purposes; apostrophes stay inside words so
 *  "i'm" matches the cue list). */
export function wordsOf(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9']+(?:-[a-z0-9']+)*/g) ?? []).filter(Boolean);
}

/**
 * Upstream `determine_utterance_type`, same decision order:
 * 1. every word a cue -> backchannel (however long);
 * 2. more than WORD_MAX words (and longer than SECOND_MAX) -> secondary;
 * 3. first word a not-cue -> secondary;
 * 4. cue proportion >= PROPORTION -> backchannel; else other.
 * Returns null for an utterance with no words.
 */
export function classifyUtterance(text: string, durationSeconds: number): UtteranceKind | null {
  const words = wordsOf(text);
  if (words.length === 0) return null;
  let cues = 0;
  for (const w of words) if (BACKCHANNEL_CUES.has(w)) cues++;
  const prop = cues / words.length;
  if (prop === 1) return "backchannel";
  if (words.length > BACKCHANNEL_WORD_MAX && durationSeconds > BACKCHANNEL_SECOND_MAX) {
    return "secondary";
  }
  if (NOT_BACKCHANNEL_CUES.has(words[0])) return "secondary";
  return prop >= BACKCHANNEL_PROPORTION ? "backchannel" : "other";
}

/** The live tag for a finalized single-mic turn. Without channel overlap
 *  there is no "secondary" to detect — a turn either is a listener noise
 *  (backchannel) or it is speech the session should count and may coach. */
export function liveTurnKind(text: string, durationSeconds: number): "primary" | "backchannel" {
  return classifyUtterance(text, durationSeconds) === "backchannel" ? "backchannel" : "primary";
}

// ---------------------------------------------------------------------------
// Batch path (channelized/diarized utterances with real overlap)
// ---------------------------------------------------------------------------

export interface Utterance {
  speaker: string;
  start: number;
  end: number;
  text: string;
}

export interface LabeledUtterance extends Utterance {
  kind: UtteranceKind;
  /** Index (into the sorted utterance list) of the primary turn this
   *  non-primary utterance interjects; null for primary turns. */
  interjects: number | null;
}

/**
 * Upstream `_label_turns`: sort by start; an utterance that begins before
 * an earlier utterance ends AND ends within it (start2 < stop1 &&
 * stop2 <= stop1) is non-primary, attached to that turn; the forward scan
 * stops at the first utterance that is not contained (upstream `break`).
 * Non-primary utterances are then classified; primaries stay "primary".
 */
export function labelTurns(utterances: Utterance[]): LabeledUtterance[] {
  const sorted = [...utterances].sort((a, b) => a.start - b.start || a.end - b.end);
  const out: LabeledUtterance[] = sorted.map((u) => ({ ...u, kind: "primary", interjects: null }));
  for (let i = 0; i < out.length; i++) {
    if (out[i].interjects !== null) continue; // already claimed by an earlier turn
    for (let j = i + 1; j < out.length; j++) {
      if (out[j].start < out[i].end && out[j].end <= out[i].end) {
        out[j].interjects = i;
        out[j].kind = classifyUtterance(out[j].text, out[j].end - out[j].start) ?? "other";
      } else {
        break; // first non-contained utterance ends this turn's window
      }
    }
  }
  return out;
}

export interface MergedTurn {
  speaker: string;
  start: number;
  end: number;
  text: string;
  /** Non-primary utterances (backchannels etc.) that rode inside or
   *  between this turn's merged parts. */
  attached: LabeledUtterance[];
  /** How many primary utterances merged into this turn. */
  parts: number;
}

/**
 * Upstream merge: consecutive PRIMARY turns by the same speaker join when
 * the pause between them is <= maxPause. Non-primary utterances never
 * break a merge — they attach to the merged turn (upstream reassigns
 * their turn_id to the interjected primary).
 */
export function mergePrimaries(
  labeled: LabeledUtterance[],
  maxPause: number = MAX_PAUSE_SECONDS,
): MergedTurn[] {
  const out: MergedTurn[] = [];
  let current: MergedTurn | null = null;
  for (const u of labeled) {
    if (u.kind !== "primary") {
      if (current) current.attached.push(u);
      else out[out.length - 1]?.attached.push(u);
      continue;
    }
    if (current && u.speaker === current.speaker && u.start - current.end <= maxPause) {
      current.text = `${current.text} ${u.text}`.trim();
      current.end = Math.max(current.end, u.end);
      current.parts += 1;
      continue;
    }
    current = { speaker: u.speaker, start: u.start, end: u.end, text: u.text, attached: [], parts: 1 };
    out.push(current);
  }
  return out;
}

/** One-call convenience for the compare harness and (later) the server
 *  parity port: utterances in, NaturalTurn-merged turns out. */
export function naturalTurns(utterances: Utterance[], maxPause: number = MAX_PAUSE_SECONDS): MergedTurn[] {
  return mergePrimaries(labelTurns(utterances), maxPause);
}
