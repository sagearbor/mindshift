/**
 * Speaker-change splitting inside one VAD span (turnSplit.ts): two people
 * trading lines with < 0.3 s between them come out of the segmenter as ONE
 * span (38% of phone turns on the corpus spanned >= 2 real voices). ECAPA
 * windows either side of each candidate boundary; split where their cosine
 * drops below the threshold. Single-speaker spans must stay whole.
 */
import {
  chooseSplits,
  splitSpanBySpeaker,
  TURN_SPLIT_DEFAULTS,
  type TurnSplitConfig,
} from "../src/live/turnSplit";
import type { Embedder } from "../src/live/speakerId";

const cfg: TurnSplitConfig = { ...TURN_SPLIT_DEFAULTS, threshold: 0.3 };
const unit = (v: number[]) => {
  const n = Math.hypot(...v);
  return Float32Array.from(v.map((x) => x / n));
};
const A = unit([1, 0, 0]);
const B = unit([0, 1, 0]);

/** Window starts every hop over [0, dur - win]; voice = A before `change`, B after. */
function windows(dur: number, change: number | null) {
  const starts: number[] = [];
  const embs: Float32Array[] = [];
  for (let s = 0; s + cfg.windowSec <= dur + 1e-9; s += cfg.hopSec) {
    starts.push(s);
    const mid = s + cfg.windowSec / 2;
    embs.push(change !== null && mid > change ? B : A);
  }
  return { starts, embs };
}

describe("chooseSplits", () => {
  it("splits a two-voice span at the change", () => {
    const { starts, embs } = windows(8, 4);
    const cuts = chooseSplits(starts, embs, 0, 8, cfg);
    expect(cuts.length).toBe(1);
    expect(Math.abs(cuts[0] - 4)).toBeLessThanOrEqual(cfg.hopSec);
  });

  it("keeps a single-voice span whole", () => {
    const { starts, embs } = windows(10, null);
    expect(chooseSplits(starts, embs, 0, 10, cfg)).toEqual([]);
  });

  it("finds A-B-A as two cuts, capped by maxPieces", () => {
    const starts: number[] = [];
    const embs: Float32Array[] = [];
    for (let s = 0; s + cfg.windowSec <= 12 + 1e-9; s += cfg.hopSec) {
      starts.push(s);
      const mid = s + cfg.windowSec / 2;
      embs.push(mid > 4 && mid < 8 ? B : A);
    }
    const cuts = chooseSplits(starts, embs, 0, 12, cfg);
    expect(cuts.length).toBe(2);
    expect(chooseSplits(starts, embs, 0, 12, { ...cfg, maxPieces: 2 }).length).toBe(1);
  });

  it("never leaves a piece shorter than minPieceSec", () => {
    const { starts, embs } = windows(8, 0.9);
    const cuts = chooseSplits(starts, embs, 0, 8, cfg);
    for (const c of cuts) expect(c).toBeGreaterThanOrEqual(cfg.minPieceSec);
  });
});

describe("splitSpanBySpeaker", () => {
  /** A fake embedder that reads the voice from the first sample's sign. */
  const embedder: Embedder = {
    embed: async (pcm: Float32Array) => {
      let pos = 0;
      for (const x of pcm) if (x > 0) pos++;
      return pos > pcm.length / 2 ? A : B;
    },
  } as Embedder;
  const pcmFor = (dur: number, change: number | null) => {
    const n = Math.round(dur * 16000);
    const pcm = new Float32Array(n);
    for (let i = 0; i < n; i++) pcm[i] = change !== null && i / 16000 > change ? -0.1 : 0.1;
    return pcm;
  };

  it("returns pieces covering the span exactly", async () => {
    const span = { start: 10, end: 18 };
    const pieces = await splitSpanBySpeaker(pcmFor(8, 4), span, embedder, cfg);
    expect(pieces.length).toBe(2);
    expect(pieces[0].start).toBe(10);
    expect(pieces[pieces.length - 1].end).toBe(18);
    expect(pieces[0].end).toBe(pieces[1].start);
    expect(Math.abs(pieces[0].end - 14)).toBeLessThanOrEqual(cfg.hopSec);
  });

  it("leaves short spans and single voices alone (no embedding for short ones)", async () => {
    let calls = 0;
    const counting: Embedder = { embed: async (p: Float32Array) => (calls++, embedder.embed(p, 16000)) } as Embedder;
    expect(await splitSpanBySpeaker(pcmFor(2, null), { start: 0, end: 2 }, counting, cfg)).toEqual([{ start: 0, end: 2 }]);
    expect(calls).toBe(0);
    expect(await splitSpanBySpeaker(pcmFor(9, null), { start: 0, end: 9 }, counting, cfg)).toEqual([{ start: 0, end: 9 }]);
  });

  it("the validation pass re-joins pieces whose whole-piece voices match", async () => {
    // Window embedder says A|B at 4 s, but whole pieces both read as A.
    let call = 0;
    const fickle: Embedder = {
      embed: async (p: Float32Array) => {
        call++;
        if (p.length === Math.round(cfg.windowSec * 16000)) return embedder.embed(p, 16000);
        return A;
      },
    } as Embedder;
    const pcm = pcmFor(8, 4);
    expect((await splitSpanBySpeaker(pcm, { start: 0, end: 8 }, fickle, cfg)).length).toBe(2);
    expect(await splitSpanBySpeaker(pcm, { start: 0, end: 8 }, fickle, { ...cfg, pieceMergeCos: 0.5 })).toEqual([
      { start: 0, end: 8 },
    ]);
    expect(call).toBeGreaterThan(0);
    // ...and keeps a real change.
    expect((await splitSpanBySpeaker(pcm, { start: 0, end: 8 }, embedder, { ...cfg, pieceMergeCos: 0.5 })).length).toBe(2);
  });

  it("an embedder failure keeps the span whole", async () => {
    const broken: Embedder = { embed: async () => Promise.reject(new Error("x")) } as unknown as Embedder;
    expect(await splitSpanBySpeaker(pcmFor(8, 4), { start: 0, end: 8 }, broken, cfg)).toEqual([{ start: 0, end: 8 }]);
  });
});
