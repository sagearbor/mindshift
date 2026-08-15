/**
 * Sequential single-speaker playback — the "▶ Play this voice" audition.
 *
 * Given one speaker's turns, plays ONLY their segments back to back: seek to a
 * segment's start, play through it, jump straight to the next segment's start
 * (skipping everyone else's turns and the silence), and pause after the last.
 * This is the listening tool behind naming a voice for the household voice
 * library: isolate a speaker on the heat chart, hear exactly them, then decide
 * who they are.
 *
 * The engine is deliberately player-agnostic and free of React/timers: it is
 * driven by the position updates the app already has (MediaPlayer's ~4Hz
 * onPositionChange poll), against a minimal seek/play/pause seam. That keeps
 * the chaining logic fully unit-testable with a fake player, and reusable by
 * the upcoming naming flow.
 */

/** One playable slice of the recording, in seconds. */
export interface AuditionSegment {
  start: number;
  end: number;
}

/** The turn shape the audition needs (a subset of the API's TranscriptTurn). */
export interface AuditionTurn {
  speaker: string;
  start_time: number;
  end_time: number;
}

/** The minimal player surface the audition drives — MediaPlayerHandle
 *  satisfies it directly. */
export interface SegmentPlayer {
  seek(seconds: number): void;
  play(): void;
  pause(): void;
}

/**
 * Position tolerance (seconds) when deciding a segment is finished. The
 * position feed is a ~4Hz poll, so an exact `>= end` comparison would often
 * overshoot into the next speaker's first syllable before the tick lands;
 * ending a hair early keeps the audition honest to ONE voice.
 */
export const SEGMENT_END_EPS = 0.05;

export interface SegmentAudition {
  /** Seek to the first segment and start playing. No-ops into `onEnded` when
   *  there are no segments. */
  start(): void;
  /** User-initiated stop: pause immediately and go inert. */
  stop(): void;
  /** Feed the current playback position (seconds); advances the chain. */
  handlePosition(seconds: number): void;
  isActive(): boolean;
}

/** ONLY `speaker`'s turns, as chronologically ordered segments. Zero-length
 *  (or negative — defensive) turns are dropped: there is nothing to hear. */
export function speakerSegments(
  turns: AuditionTurn[],
  speaker: string,
): AuditionSegment[] {
  return turns
    .filter((t) => t.speaker === speaker && t.end_time > t.start_time)
    .map((t) => ({ start: t.start_time, end: t.end_time }))
    .sort((a, b) => a.start - b.start);
}

/**
 * Build an audition run over `segments`. `onEnded` fires exactly once when the
 * run finishes — naturally (played through the last segment) or via `stop()` —
 * so UI state can mirror the engine without tracking which way it ended.
 */
export function createSegmentAudition(
  player: SegmentPlayer,
  segments: AuditionSegment[],
  onEnded?: () => void,
): SegmentAudition {
  let active = false;
  let current = 0;

  return {
    start() {
      if (segments.length === 0) {
        // Nothing to play — never touch the player, but tell the UI honestly.
        onEnded?.();
        return;
      }
      current = 0;
      active = true;
      player.seek(segments[0].start);
      player.play();
    },

    stop() {
      if (!active) return;
      active = false;
      player.pause();
      onEnded?.();
    },

    handlePosition(seconds: number) {
      if (!active) return;
      // A stale pre-seek poll tick (or any position before the current
      // segment's end) never advances the chain.
      if (seconds < segments[current].end - SEGMENT_END_EPS) return;
      current += 1;
      if (current < segments.length) {
        // Jump the gap; playback keeps running.
        player.seek(segments[current].start);
      } else {
        active = false;
        player.pause();
        onEnded?.();
      }
    },

    isActive: () => active,
  };
}
