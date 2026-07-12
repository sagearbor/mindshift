/**
 * Single source of truth for per-speaker colors. Lifted out of LiveTranscript
 * so the live transcript, the post-session HeatChart, and its legend all key
 * a speaker to the *same* hue — a speaker who is blue in the live view stays
 * blue in the dynamics chart, which is what makes the two screens feel like one
 * conversation rather than two unrelated visualizations.
 */

// Diarized live audio labels its speakers "Speaker A"/"Speaker B"; pin those to
// the house pair so the two most common cases are stable regardless of hashing.
const SPEAKER_COLORS: Record<string, string> = {
  "Speaker A": "#4A90D9",
  "Speaker B": "#E85D75",
};

// Palette reused for any other/named speakers. Ordered so the first two match
// the pinned pair above, keeping a 2-speaker conversation visually consistent
// whether its speakers are "Speaker A/B" or real names.
export const SPEAKER_PALETTE = [
  "#4A90D9", // primary blue
  "#E85D75", // rose
  "#10B981", // green
  "#F59E0B", // amber
  "#8B5CF6", // violet
];

/**
 * Deterministic color for a speaker label. Pinned labels win; everything else
 * hashes its characters into the palette so the same name always yields the
 * same color across renders and screens (no random assignment).
 */
export function getSpeakerColor(speaker: string): string {
  if (SPEAKER_COLORS[speaker]) return SPEAKER_COLORS[speaker];
  const hash = speaker
    .split("")
    .reduce((acc, c) => acc + c.charCodeAt(0), 0);
  return SPEAKER_PALETTE[hash % SPEAKER_PALETTE.length];
}
