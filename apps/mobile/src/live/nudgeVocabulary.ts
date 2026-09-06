/**
 * The nudge VOCABULARY — the TypeScript mirror of
 * server/tests/fixtures/policy_vectors/nudge_vocabulary.json (owner-approved
 * 2026-09-06), alongside server/nudge_vocabulary.py and the watch's
 * NudgeVocabulary.kt. __tests__/nudgeVocabulary.test.ts replays the fixture
 * against this file, so the emoji you see on the phone, the emoji the watch
 * draws and the emoji a report prints can never drift apart.
 *
 * Division of labour: `nudgePolicy.ts` decides WHEN something fires and at
 * what level; this file decides what that firing is CALLED, what it LOOKS
 * like and what it FEELS like. Nothing here reads a threshold, and nothing
 * in nudgePolicy.ts hardcodes an emoji or a millisecond.
 */

/** One-letter code — stable forever; the compact form in reports, chart
 *  markers and telemetry. */
export type NudgeCode = "H" | "D" | "C" | "A" | "E" | "R" | "K" | "P";

export type NudgePolarity = "alert" | "positive";
export type NudgeColor = "red" | "green" | "neutral";

/**
 * One playable cue. `timingsMs` alternates OFF, ON, OFF, ON, … starting with
 * an initial delay (always 0) — which is exactly React Native's Android
 * `Vibration.vibrate(pattern)` shape, so it is passed verbatim. `amplitudes`
 * (0 in the OFF slots, 1..255 in the ON slots) is what the watch's
 * `VibrationEffect.createWaveform` uses; RN cannot vary amplitude, so the
 * phone feels the RHYTHM only. That is why every level difference is a
 * rhythm difference.
 */
export interface HapticWaveform {
  timingsMs: number[];
  amplitudes: number[];
}

export interface NudgeVocabularyEntry {
  code: NudgeCode;
  vector: string;
  icon: string;
  color: NudgeColor;
  name: string;
  meaning: string;
  polarity: NudgePolarity;
  /** Detector vector names that raise this code (nudgePolicy.ts /
   *  server/watch/vectors.py). Empty for the positive codes, whose detectors
   *  live in positiveNudges.ts. */
  sources: string[];
  /** The on-screen line, or null when the icon alone is the message. */
  flashText: string | null;
  /** Never fires live — a badge on the post-session summary (K). */
  summaryOnly: boolean;
  /** The phone has no sensor for it (P: heart rate). */
  watchOnly: boolean;
  levels: number[];
  /** Level -> waveform, or null for a code that never buzzes (K). */
  haptic: Record<number, HapticWaveform> | null;
}

/** Pattern-wide never-merge silence floor (the watch's HapticPatterns.MIN_GAP_MS). */
export const MIN_GAP_MS = 170;

/** At most one positive haptic per this many seconds, across D/E/R together —
 *  so praise can never become its own nag. */
export const POSITIVE_CAP_S = 120.0;

/** The one vector whose intra-cue gap is deliberately under MIN_GAP_MS: a
 *  lub-dub only reads as a heartbeat when the two beats nearly merge. */
export const HAPTIC_GAP_EXCEPTIONS: readonly string[] = ["pulse"];

/** Owner order (2026-09-06). */
export const NUDGE_VOCABULARY: readonly NudgeVocabularyEntry[] = [
  {
    code: "H",
    vector: "heated",
    icon: "📈",
    color: "red",
    name: "Heated",
    meaning:
      "You got loud, or your words got hot. Yelling and aggressive tone are one family to the user (the detectors stay separate underneath).",
    polarity: "alert",
    sources: ["yelling", "aggressive_tone", "activation"],
    flashText: "Take it down a notch",
    summaryOnly: false,
    watchOnly: false,
    levels: [1, 2, 3],
    haptic: {
      // A RISING ramp: the cue itself builds, and the LEVEL is how many taps
      // it takes to get there. This is the shipped watch channel-A ladder.
      1: { timingsMs: [0, 75], amplitudes: [0, 255] },
      2: { timingsMs: [0, 75, 170, 75], amplitudes: [0, 210, 0, 255] },
      3: { timingsMs: [0, 100, 170, 100, 170, 100], amplitudes: [0, 200, 0, 230, 0, 255] },
    },
  },
  {
    code: "D",
    vector: "de_escalated",
    icon: "📉",
    color: "green",
    name: "De-escalated",
    meaning: "Heat or loudness dropped a level within two turns of a spike — you pulled it back.",
    polarity: "positive",
    sources: [],
    flashText: "Nice recovery",
    summaryOnly: false,
    watchOnly: false,
    levels: [1],
    // A FALLING ramp — the mirror of H, and the only cue that fades.
    haptic: { 1: { timingsMs: [0, 70, 170, 70, 170, 70], amplitudes: [0, 190, 0, 140, 0, 90] } },
  },
  {
    code: "C",
    vector: "cut_in",
    icon: "✂️",
    color: "red",
    name: "Cut in",
    meaning:
      "You started talking over them and kept going — sustained overlap, not the brief overlap of ordinary engagement.",
    polarity: "alert",
    sources: ["interrupting"],
    flashText: "Let them finish",
    summaryOnly: false,
    watchOnly: false,
    levels: [1, 2, 3],
    haptic: {
      // `• —`, `• • —`, `• • • —`: short tap(s) then one long "stop" buzz.
      1: { timingsMs: [0, 60, 170, 260], amplitudes: [0, 255, 0, 200] },
      2: { timingsMs: [0, 60, 170, 60, 170, 260], amplitudes: [0, 255, 0, 255, 0, 200] },
      3: { timingsMs: [0, 60, 170, 60, 170, 60, 170, 260], amplitudes: [0, 255, 0, 255, 0, 255, 0, 200] },
    },
  },
  {
    code: "A",
    vector: "hogging",
    icon: "🎤",
    color: "red",
    name: "Hogging",
    meaning: "You have been taking most of the airtime over the last two minutes.",
    polarity: "alert",
    sources: ["airtime"],
    flashText: "Give them the floor",
    summaryOnly: false,
    watchOnly: false,
    levels: [1, 2, 3],
    haptic: {
      // Slow `— — —`: long, unhurried buzzes — "you're talking a lot" is not
      // an emergency, so it must not feel like H's crisp taps.
      1: { timingsMs: [0, 250, 300, 250], amplitudes: [0, 180, 0, 180] },
      2: { timingsMs: [0, 250, 300, 250, 300, 250], amplitudes: [0, 180, 0, 180, 0, 180] },
      3: { timingsMs: [0, 250, 300, 250, 300, 250, 300, 250], amplitudes: [0, 180, 0, 180, 0, 180, 0, 180] },
    },
  },
  {
    code: "E",
    vector: "listened",
    icon: "👂",
    color: "green",
    name: "Listened",
    meaning: "You let them finish a long turn without cutting in.",
    polarity: "positive",
    sources: [],
    flashText: null,
    summaryOnly: false,
    watchOnly: false,
    levels: [1],
    // Soft `••` — deliberately the same cue as R: the wrist says "that was
    // good", the screen says which good thing.
    haptic: { 1: { timingsMs: [0, 50, 170, 50], amplitudes: [0, 110, 0, 110] } },
  },
  {
    code: "R",
    vector: "repair",
    icon: "🤝",
    color: "green",
    name: "Repair",
    meaning: "You validated or apologised and their tone softened on the next turn.",
    polarity: "positive",
    sources: [],
    flashText: null,
    summaryOnly: false,
    watchOnly: false,
    levels: [1],
    haptic: { 1: { timingsMs: [0, 50, 170, 50], amplitudes: [0, 110, 0, 110] } },
  },
  {
    code: "K",
    vector: "calm_streak",
    icon: "🧘",
    color: "green",
    name: "Calm streak",
    meaning: "N minutes with no escalation at all.",
    polarity: "positive",
    sources: [],
    flashText: null,
    summaryOnly: true,
    watchOnly: false,
    levels: [1],
    // No cue, ever — buzzing someone to say nothing happened is a nag.
    haptic: null,
  },
  {
    code: "P",
    vector: "pulse",
    icon: "❤️",
    color: "red",
    name: "Pulse",
    meaning: "Your heart rate jumped well over your resting rate (+15 / +25 / +35 bpm).",
    polarity: "alert",
    sources: ["hr_spike"],
    flashText: null,
    summaryOnly: false,
    watchOnly: true,
    levels: [1, 2, 3],
    haptic: {
      // Lub-dub: a short beat then a longer one 120 ms apart — under
      // MIN_GAP_MS on purpose, because the near-merge is the heartbeat.
      1: { timingsMs: [0, 90, 120, 150], amplitudes: [0, 190, 0, 240] },
      2: { timingsMs: [0, 90, 120, 150, 400, 90, 120, 150], amplitudes: [0, 190, 0, 240, 0, 190, 0, 240] },
      3: {
        timingsMs: [0, 90, 120, 150, 400, 90, 120, 150, 400, 90, 120, 150],
        amplitudes: [0, 190, 0, 240, 0, 190, 0, 240, 0, 190, 0, 240],
      },
    },
  },
];

const BY_CODE = new Map<string, NudgeVocabularyEntry>(NUDGE_VOCABULARY.map((e) => [e.code, e]));
const BY_VECTOR = new Map<string, NudgeVocabularyEntry>(NUDGE_VOCABULARY.map((e) => [e.vector, e]));
const BY_SOURCE = new Map<string, NudgeVocabularyEntry>();
for (const e of NUDGE_VOCABULARY) for (const s of e.sources) BY_SOURCE.set(s, e);

export function vocabularyForCode(code: string): NudgeVocabularyEntry | null {
  return BY_CODE.get(code) ?? null;
}

/** By vocabulary vector id ("heated") OR by the detector vector that raises
 *  it ("yelling", "aggressive_tone", …) — callers hold both kinds of name. */
export function vocabularyFor(vector: string): NudgeVocabularyEntry | null {
  return BY_VECTOR.get(vector) ?? BY_SOURCE.get(vector) ?? null;
}

/** The icon for a detector vector or vocabulary id, or null when unknown —
 *  never a fallback emoji, so a missing mapping is visible instead of silently
 *  mislabelled. */
export function iconFor(vector: string): string | null {
  return vocabularyFor(vector)?.icon ?? null;
}

/**
 * The cue to play for a code at a level, or null when that code never buzzes
 * (K) or the level is out of range. Positives are unleveled: any level maps
 * to their single cue, because a "level 2 well done" is not a thing.
 */
export function hapticFor(code: string, level: number): HapticWaveform | null {
  const entry = BY_CODE.get(code);
  if (!entry?.haptic) return null;
  if (entry.polarity === "positive") return entry.haptic[1] ?? null;
  if (!Number.isFinite(level)) return null;
  return entry.haptic[Math.trunc(level)] ?? null;
}

/**
 * The strongest code a set of firing detector vectors maps to, for the
 * screen/wrist. `vectors` is a NudgeEvent's `vectors` list. Ties break on the
 * owner's order (H before C before A before P), which is also worst-first.
 */
export function codeForVectors(vectors: readonly string[]): NudgeCode | null {
  let best: NudgeVocabularyEntry | null = null;
  let bestRank = Number.POSITIVE_INFINITY;
  for (const v of vectors) {
    const e = vocabularyFor(v);
    if (!e) continue;
    const rank = NUDGE_VOCABULARY.indexOf(e);
    if (rank < bestRank) {
      best = e;
      bestRank = rank;
    }
  }
  return best?.code ?? null;
}

/**
 * The positive-nudge rate limit: at most one positive cue per
 * [POSITIVE_CAP_S] across D/E/R together. `>=` (not `>`) so a caller ticking
 * at exactly the cadence fires on the tick — the same direction as the
 * watch's NudgeHapticSchedule.reminderDue.
 */
export class PositiveNudgeGate {
  private lastT: number | null = null;

  constructor(private readonly capS = POSITIVE_CAP_S) {}

  /** True (and the clock resets) when this positive offer may be delivered. */
  admit(t: number): boolean {
    if (this.lastT !== null && t - this.lastT < this.capS) return false;
    this.lastT = t;
    return true;
  }

  /** Seconds until the next positive may fire, or 0 when one may fire now. */
  waitS(t: number): number {
    if (this.lastT === null) return 0;
    return Math.max(0, this.capS - (t - this.lastT));
  }
}
