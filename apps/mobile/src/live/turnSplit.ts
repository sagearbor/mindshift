/**
 * Speaker-change splitting INSIDE one VAD span.
 *
 * The segmenter merges speech runs separated by <= 0.3 s, so two people
 * trading lines quickly come out as ONE turn: on the recording corpus 38% of
 * phone turns spanned two or more real voices, and the turn then carries one
 * speaker label (usually the wrong one for half of it). This pass embeds
 * ECAPA windows (`windowSec`, every `hopSec`) across a long span; for each
 * candidate boundary t it compares the window ending at t with the window
 * starting at t, and cuts where their cosine falls below `threshold`
 * (lowest first, recursively, up to `maxPieces`, never leaving a piece under
 * `minPieceSec`). Cost: ~(span - window) / hop extra embeddings per long
 * span (~40 ms each on the phone), paid only for spans >= `minSpanSec`.
 *
 * Pure selection (`chooseSplits`) is separate from the embedding pass so the
 * rule is unit-tested without a model.
 */
import type { Span } from "./segmenter";
import type { Embedder } from "./speakerId";

export interface TurnSplitConfig {
  windowSec: number;
  hopSec: number;
  /** Spans shorter than this are never examined. */
  minSpanSec: number;
  /** Cut where cosine(left window, right window) is below this. */
  threshold: number;
  minPieceSec: number;
  maxPieces: number;
  /** Re-join adjacent pieces whose whole-piece embeddings are at least this
   *  similar (null = no validation pass). */
  pieceMergeCos: number | null;
}

/** Tuned 2026-10-10 on the 5 DEV corpus items with ground-truth segment
 *  times (landscape D-006..D-015, phone-only replays with the AGC on): 2 s
 *  windows keep the wearer's self-time recall (1.5 s windows cost 7 points),
 *  cos < 0.15 beats 0.1/0.2/0.25, and the piece-validation pass did not pay.
 *  D-009 vs D-000: time-weighted purity 0.641 -> 0.701, self-time precision
 *  0.587 -> 0.718, self-time recall 0.422 -> 0.435; fragmentation of
 *  single-voice segments >= 2 s 31.5% -> 42.4%. */
export const TURN_SPLIT_DEFAULTS: TurnSplitConfig = {
  windowSec: 2.0,
  hopSec: 0.5,
  minSpanSec: 4.0,
  threshold: 0.15,
  minPieceSec: 2.0,
  maxPieces: 4,
  pieceMergeCos: null,
};

/** Production switch (FastLoop deps.turnSplit overrides). DARK: turning it
 *  on moves every pinned replay fixture (maggiano3's baseline included) and
 *  adds ~(span - 2 s) / 0.5 s ECAPA passes per long turn — owner's call. */
export const TURN_SPLIT_ENABLED = false;

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Cut times (span-relative seconds, ascending) from window embeddings.
 * `starts[i]` is window i's start (relative to the span); windows are
 * `cfg.windowSec` long. Embeddings are assumed L2-normalised.
 */
export function chooseSplits(
  starts: number[],
  embs: Float32Array[],
  spanStart: number,
  spanEnd: number,
  cfg: TurnSplitConfig,
): number[] {
  const byStart = new Map<string, number>();
  starts.forEach((s, i) => byStart.set(s.toFixed(3), i));
  // Candidate boundaries: left window ends at t, right window starts at t.
  const cands: { t: number; cos: number }[] = [];
  starts.forEach((s, i) => {
    const t = s + cfg.windowSec;
    const j = byStart.get(t.toFixed(3));
    if (j === undefined) return;
    cands.push({ t, cos: dot(embs[i], embs[j]) });
  });
  const cuts: number[] = [];
  const pieces: [number, number][] = [[spanStart, spanEnd]];
  while (cuts.length + 1 < cfg.maxPieces) {
    let best: { t: number; cos: number; piece: number } | null = null;
    for (const c of cands) {
      if (c.cos >= cfg.threshold) continue;
      const k = pieces.findIndex(([a, b]) => c.t - cfg.windowSec >= a - 1e-9 && c.t + cfg.windowSec <= b + 1e-9);
      if (k < 0) continue;
      const [a, b] = pieces[k];
      if (c.t - a < cfg.minPieceSec - 1e-9 || b - c.t < cfg.minPieceSec - 1e-9) continue;
      if (best === null || c.cos < best.cos) best = { ...c, piece: k };
    }
    if (best === null) break;
    const [a, b] = pieces[best.piece];
    pieces.splice(best.piece, 1, [a, best.t], [best.t, b]);
    cuts.push(best.t);
  }
  return cuts.sort((x, y) => x - y);
}

/**
 * The span as one or more pieces (absolute seconds) split at speaker
 * changes. `pcm` is exactly the span's audio (16 kHz float32). Any failure
 * (embedder error) returns the span whole — never a guess.
 */
export async function splitSpanBySpeaker(
  pcm: Float32Array,
  span: Span,
  embedder: Embedder,
  cfg: TurnSplitConfig = TURN_SPLIT_DEFAULTS,
  sampleRate = 16000,
): Promise<Span[]> {
  const dur = span.end - span.start;
  if (dur < cfg.minSpanSec || dur < 2 * cfg.minPieceSec) return [{ ...span }];
  const starts: number[] = [];
  for (let s = 0; s + cfg.windowSec <= dur + 1e-9; s += cfg.hopSec) starts.push(Math.round(s * 1000) / 1000);
  if (starts.length < 2) return [{ ...span }];
  const win = Math.round(cfg.windowSec * sampleRate);
  const embs: Float32Array[] = [];
  try {
    for (const s of starts) {
      const a = Math.round(s * sampleRate);
      embs.push(await embedder.embed(pcm.subarray(a, Math.min(pcm.length, a + win)), sampleRate));
    }
  } catch {
    return [{ ...span }];
  }
  const cuts = chooseSplits(starts, embs, 0, dur, cfg);
  if (cuts.length === 0) return [{ ...span }];
  let out: Span[] = [];
  let prev = span.start;
  for (const c of cuts) {
    const t = Math.round((span.start + c) * 1000) / 1000;
    out.push({ start: prev, end: t });
    prev = t;
  }
  out.push({ start: prev, end: span.end });
  if (cfg.pieceMergeCos !== null) {
    try {
      out = await mergeSameVoicePieces(pcm, span, out, embedder, cfg.pieceMergeCos, sampleRate);
    } catch {
      return [{ ...span }];
    }
  }
  return out;
}

/** Validation pass: embed each whole piece and re-join neighbours whose
 *  pooled voices still look alike (cosine >= `mergeCos`) — a dip between two
 *  1.5 s windows is not proof of a second voice. */
export async function mergeSameVoicePieces(
  pcm: Float32Array,
  span: Span,
  pieces: Span[],
  embedder: Embedder,
  mergeCos: number,
  sampleRate = 16000,
): Promise<Span[]> {
  const embOf = (p: Span) =>
    embedder.embed(
      pcm.subarray(Math.round((p.start - span.start) * sampleRate), Math.round((p.end - span.start) * sampleRate)),
      sampleRate,
    );
  const out: Span[] = [{ ...pieces[0] }];
  let prevEmb = await embOf(out[0]);
  for (let i = 1; i < pieces.length; i++) {
    const e = await embOf(pieces[i]);
    if (dot(prevEmb, e) >= mergeCos) {
      out[out.length - 1].end = pieces[i].end;
      prevEmb = await embOf(out[out.length - 1]);
    } else {
      out.push({ ...pieces[i] });
      prevEmb = e;
    }
  }
  return out;
}
