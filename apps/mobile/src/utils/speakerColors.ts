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
 *
 * CAVEAT (confirmed by direct testing against the project's real fixture —
 * server/tests/fixtures/audio/test_recording_family_real_meta.json, speakers
 * "Sage"/"Asher"): with only 5 palette slots, this hash CAN and DOES collide
 * for two real, distinct names — "Sage" and "Asher" both land on the same
 * bucket. Calling this function separately for two different speakers is NOT
 * safe when you need them visually distinguishable; use
 * `resolveSpeakerColors` for that (it starts from this same hash but breaks
 * any collision within the given speaker list).
 */
export function getSpeakerColor(speaker: string): string {
  if (SPEAKER_COLORS[speaker]) return SPEAKER_COLORS[speaker];
  const hash = speaker
    .split("")
    .reduce((acc, c) => acc + c.charCodeAt(0), 0);
  return SPEAKER_PALETTE[hash % SPEAKER_PALETTE.length];
}

/**
 * Collision-free color assignment for ONE set of speakers (one conversation).
 *
 * getSpeakerColor() alone hashes into only 5 palette buckets (7 counting the
 * two pinned house colors) — deterministic and stable, but with that few
 * buckets two arbitrary real names CAN legitimately hash to the SAME color
 * (confirmed: "Sage" and "Asher" both resolve to "#8B5CF6"). Two different
 * speakers then render in the IDENTICAL color, which is indistinguishable
 * from "the wrong speaker's color at a turn transition" — the color simply
 * never changes at the boundary between them. This is the investigated,
 * confirmed root cause behind that report.
 *
 * This resolves colors for the speakers in ONE conversation so that within
 * that set, no two distinct speakers ever share a color: each speaker starts
 * from its own getSpeakerColor() hash (unchanged, so the common case — a
 * conversation with no collisions, or one using the pinned "Speaker A"/
 * "Speaker B" pair — renders EXACTLY as before); only when a LATER speaker's
 * hash collides with a color an EARLIER speaker in this same list already
 * claimed does it deterministically walk to the next unclaimed color in a
 * fixed pool (the two pinned house colors, then the palette, in that order).
 * `speakers` should be in first-appearance order (stable across a given
 * conversation's renders), which keeps this a pure function of that ordered
 * list — same input, same output, always.
 */
export function resolveSpeakerColors(speakers: string[]): Map<string, string> {
  const pool = [SPEAKER_COLORS["Speaker A"], SPEAKER_COLORS["Speaker B"], ...SPEAKER_PALETTE];
  const colorOf = new Map<string, string>();
  const claimed = new Set<string>();
  for (const speaker of speakers) {
    if (colorOf.has(speaker)) continue; // dedupe repeats in the input list
    let color = getSpeakerColor(speaker);
    if (claimed.has(color)) {
      const free = pool.find((c) => !claimed.has(c));
      // If every pool color is already claimed (more distinct speakers than
      // colors — an unusual conversation size), fall back to the hash color:
      // a rare repeat among many speakers is preferable to losing a speaker's
      // color entirely.
      if (free) color = free;
    }
    claimed.add(color);
    colorOf.set(speaker, color);
  }
  return colorOf;
}
