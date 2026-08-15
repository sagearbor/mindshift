/** Which recording engine drives the audio recorder screen. */
export type RecorderEngineKind = "stream" | "recorder";

/**
 * The escape hatch. "stream" is the v2 gapless engine (continuous PCM stream
 * → WAV files, zero-gap rotation, ~5 s crash-loss bound). "recorder" is the
 * v1 engine (platform recorder restarted every ~5 min: ~0.2 s gap per
 * rotation, up to a segment lost on crash) — kept selectable so a
 * device-specific failure of the realtime stream API has a one-line fallback.
 */
export const RECORDER_ENGINE: RecorderEngineKind = "stream";
