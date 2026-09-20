/**
 * In-person overlap probe (DARK) — the single-mic best-effort answer to
 * "was I talking over someone?" that the exact call-mode path cannot give
 * us in a room (docs/plans/2026-09-04-naturalturn-conversation-quality.md,
 * WS2/A).
 *
 * A phone mic mixes both voices into one stream, so the VAD segmenter
 * produces one continuous span while two people talk at once. This probe
 * slides a 1.5 s window across a LONG self turn, embeds each window with
 * the same ECAPA model the loop already runs, and scores it read-only
 * against the user's print and the other voices the session knows
 * (SpeakerLabeler.scoreWindow). A window that sits between the two
 * (both scores present, neither clearly ahead) is "mixed" — the acoustic
 * signature of overlapping speech. Mixed seconds and the longest mixed run
 * are recorded on the turn and shown in Developer mode; nothing nudges on
 * them until real sessions show the numbers are trustworthy (false
 * steamroll nudges in a family conversation would cost more trust than
 * the feature earns). When validated, the longest run maps straight onto
 * the shared `interrupting` ladder (nudgePolicy.ts INTERRUPT_LEVELS).
 */

/** Only turns at least this long are probed (short turns can't contain a
 *  sustained overlap, and each window costs one ECAPA pass ~65 ms). */
export const OVERLAP_PROBE_MIN_SECONDS = 4;
export const OVERLAP_WINDOW_SECONDS = 1.5;
export const OVERLAP_HOP_SECONDS = 0.5;
/** Probe at most the LAST this-many windows of a very long turn. */
export const OVERLAP_MAX_WINDOWS = 20;
/** Self-vs-other separation needed to call a window one voice. */
export const OVERLAP_MARGIN = 0.1;
/** Below this on BOTH sides a window is noise / neither known voice. */
export const OVERLAP_MIN_SCORE = 0.2;

export type WindowVoice = "self" | "other" | "mixed" | "unclear";

export interface WindowScore {
  /** Window start, seconds from the turn's start. */
  start: number;
  /** Cosine vs the user's print (null: no print). */
  self: number | null;
  /** Best cosine vs any other known voice (null: none known yet). */
  otherMax: number | null;
}

export interface OverlapSummary {
  windows: number;
  voices: WindowVoice[];
  /** Mixed windows × hop. */
  mixedSeconds: number;
  /** Longest run of consecutive mixed windows × hop. */
  longestMixedRunSeconds: number;
}

export function classifyWindow(s: WindowScore): WindowVoice {
  const self = s.self ?? -1;
  const other = s.otherMax ?? -1;
  if (Math.max(self, other) < OVERLAP_MIN_SCORE) return "unclear";
  if (self - other >= OVERLAP_MARGIN) return "self";
  if (other - self >= OVERLAP_MARGIN) return "other";
  return "mixed";
}

export function summarizeOverlap(scores: WindowScore[], hop = OVERLAP_HOP_SECONDS): OverlapSummary {
  const voices = scores.map(classifyWindow);
  let mixed = 0;
  let run = 0;
  let longest = 0;
  for (const v of voices) {
    if (v === "mixed") {
      mixed += 1;
      run += 1;
      if (run > longest) longest = run;
    } else {
      run = 0;
    }
  }
  const r3 = (x: number) => Math.round(x * 1000) / 1000;
  return {
    windows: voices.length,
    voices,
    mixedSeconds: r3(mixed * hop),
    longestMixedRunSeconds: r3(longest * hop),
  };
}

/** Window starts (seconds) covering the LAST `maxWindows` windows of a
 *  turn, in time order. Empty when the turn is shorter than one window. */
export function windowStarts(
  totalSeconds: number,
  window = OVERLAP_WINDOW_SECONDS,
  hop = OVERLAP_HOP_SECONDS,
  maxWindows = OVERLAP_MAX_WINDOWS,
): number[] {
  if (totalSeconds < window) return [];
  const starts: number[] = [];
  for (let s = totalSeconds - window; s >= -1e-9 && starts.length < maxWindows; s -= hop) {
    starts.push(Math.max(0, Math.round(s * 1000) / 1000));
  }
  return starts.reverse();
}

export interface ProbeOptions {
  window?: number;
  hop?: number;
  maxWindows?: number;
}

/**
 * Embed each window (caller supplies the model) and score it read-only
 * (caller supplies the labeler's scoreWindow). Null when the turn is too
 * short to window. Never throws on a single failed window — that window
 * is simply skipped.
 */
export async function probeOverlapAsync(
  pcm: Float32Array,
  sampleRate: number,
  embed: (window: Float32Array) => Promise<ArrayLike<number>>,
  score: (embedding: ArrayLike<number>) => { self: number | null; otherMax: number | null },
  opts: ProbeOptions = {},
): Promise<OverlapSummary | null> {
  const window = opts.window ?? OVERLAP_WINDOW_SECONDS;
  const hop = opts.hop ?? OVERLAP_HOP_SECONDS;
  const total = pcm.length / sampleRate;
  const starts = windowStarts(total, window, hop, opts.maxWindows);
  if (starts.length === 0) return null;
  const n = Math.round(window * sampleRate);
  const scores: WindowScore[] = [];
  for (const start of starts) {
    const a = Math.round(start * sampleRate);
    const slice = pcm.subarray(a, Math.min(pcm.length, a + n));
    try {
      const emb = await embed(slice);
      const s = score(emb);
      scores.push({ start, self: s.self, otherMax: s.otherMax });
    } catch {
      // Unembeddable window: skipped, never a guess.
    }
  }
  if (scores.length === 0) return null;
  return summarizeOverlap(scores, hop);
}
