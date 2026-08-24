/**
 * Turn segmentation: per-frame speech flags -> (start_s, end_s) turns, with
 * server/watch/diarize.py::speech_segments' merge/drop rules:
 *
 * - adjacent speech runs separated by a gap <= `mergeGapSeconds` (inclusive)
 *   are ONE turn;
 * - merged spans shorter than `minSeconds` (strict <) are dropped as
 *   coughs/clicks — merging happens BEFORE this filter.
 *
 * `energySpeechSegments` is the whole-clip port (the golden-vector parity
 * target, driven by server/tests/fixtures/policy_vectors/vad_segments.json);
 * `StreamingSegmenter` applies the identical rules online so the live loop
 * can finalize a turn the moment its trailing silence exceeds the merge gap
 * — and a test pins that the two agree on every fixture case.
 */
import { energyIsSpeech, ENERGY_FRAME_SECONDS, SILENCE_FLOOR_DBFS } from "./vad";

export interface Span {
  start: number;
  end: number;
}

export interface SegmenterConfig {
  mergeGapSeconds: number;
  minSeconds: number;
}

export const MERGE_GAP_SECONDS = 0.3;
export const MIN_SEGMENT_SECONDS = 0.6;

export const DEFAULT_SEGMENTER_CONFIG: SegmenterConfig = {
  mergeGapSeconds: MERGE_GAP_SECONDS,
  minSeconds: MIN_SEGMENT_SECONDS,
};

/** Frame-run -> raw spans, then the merge and min-duration passes. */
export function mergeAndDrop(raw: Span[], cfg: SegmenterConfig): Span[] {
  if (raw.length === 0) return [];
  const merged: Span[] = [{ ...raw[0] }];
  for (let i = 1; i < raw.length; i++) {
    const { start, end } = raw[i];
    const prev = merged[merged.length - 1];
    if (start - prev.end <= cfg.mergeGapSeconds) {
      prev.end = end;
    } else {
      merged.push({ start, end });
    }
  }
  return merged.filter((s) => s.end - s.start >= cfg.minSeconds);
}

/**
 * Spans from per-frame flags. `boundaries[i]` / `boundaries[i+1]` are frame
 * i's start/end in seconds (explicit cumulative timestamps, so a trailing
 * sub-frame is still counted — see the Python).
 */
export function spansFromFlags(
  isSpeech: boolean[],
  boundaries: number[],
  cfg: SegmenterConfig,
): Span[] {
  const raw: Span[] = [];
  let runStart: number | null = null;
  for (let i = 0; i < isSpeech.length; i++) {
    if (isSpeech[i] && runStart === null) {
      runStart = i;
    } else if (!isSpeech[i] && runStart !== null) {
      raw.push({ start: boundaries[runStart], end: boundaries[i] });
      runStart = null;
    }
  }
  if (runStart !== null) {
    raw.push({ start: boundaries[runStart], end: boundaries[isSpeech.length] });
  }
  return mergeAndDrop(raw, cfg);
}

export interface EnergySegmenterConfig extends SegmenterConfig {
  floorDbfs: number;
  frameSeconds: number;
}

export const DEFAULT_ENERGY_SEGMENTER_CONFIG: EnergySegmenterConfig = {
  floorDbfs: SILENCE_FLOOR_DBFS,
  frameSeconds: ENERGY_FRAME_SECONDS,
  ...DEFAULT_SEGMENTER_CONFIG,
};

/** Whole-clip port of diarize.py::speech_segments over int16 PCM. */
export function energySpeechSegments(
  pcm: ArrayLike<number>,
  sr: number,
  cfg: EnergySegmenterConfig = DEFAULT_ENERGY_SEGMENTER_CONFIG,
): Span[] {
  const frameSamples = Math.max(1, Math.round(cfg.frameSeconds * sr));
  const nFrames = Math.floor(pcm.length / frameSamples);
  const trailing = pcm.length - nFrames * frameSamples;
  if (nFrames === 0 && trailing === 0) return [];

  const flags: boolean[] = [];
  const boundaries: number[] = [0];
  const slice = (a: number, b: number): number[] => {
    const out = new Array<number>(b - a);
    for (let i = a; i < b; i++) out[i - a] = pcm[i];
    return out;
  };
  for (let i = 0; i < nFrames; i++) {
    flags.push(
      energyIsSpeech(slice(i * frameSamples, (i + 1) * frameSamples), cfg.floorDbfs),
    );
    boundaries.push(boundaries[boundaries.length - 1] + cfg.frameSeconds);
  }
  if (trailing > 0) {
    flags.push(energyIsSpeech(slice(nFrames * frameSamples, pcm.length), cfg.floorDbfs));
    boundaries.push(boundaries[boundaries.length - 1] + trailing / sr);
  }
  return spansFromFlags(flags, boundaries, cfg);
}

/**
 * Online form of the same rules. Feed frames in order; a finalized turn comes
 * back the moment it can no longer be extended (silence past the merge gap)
 * — or from `flush()` at end of stream. Sub-minimum spans are dropped
 * exactly as offline.
 */
export class StreamingSegmenter {
  private current: Span | null = null;

  constructor(private readonly cfg: SegmenterConfig = DEFAULT_SEGMENTER_CONFIG) {}

  /** The open (not yet finalized) speech run, if any. */
  get active(): Span | null {
    return this.current ? { ...this.current } : null;
  }

  /** True while the last frame was speech (or a within-gap pause). */
  get inSpeech(): boolean {
    return this.current !== null;
  }

  push(isSpeech: boolean, frameStart: number, frameEnd: number): Span | null {
    if (isSpeech) {
      if (this.current === null) {
        this.current = { start: frameStart, end: frameEnd };
        return null;
      }
      if (frameStart - this.current.end <= this.cfg.mergeGapSeconds) {
        this.current.end = frameEnd;
        return null;
      }
      // Cannot happen if silence frames were fed in between (they finalize
      // first), but stay correct for a caller that skips silent frames.
      const done = this.current;
      this.current = { start: frameStart, end: frameEnd };
      return this.keep(done);
    }
    if (this.current !== null && frameEnd - this.current.end > this.cfg.mergeGapSeconds) {
      const done = this.current;
      this.current = null;
      return this.keep(done);
    }
    return null;
  }

  flush(): Span | null {
    const done = this.current;
    this.current = null;
    return done ? this.keep(done) : null;
  }

  reset() {
    this.current = null;
  }

  private keep(span: Span): Span | null {
    return span.end - span.start >= this.cfg.minSeconds ? span : null;
  }
}
