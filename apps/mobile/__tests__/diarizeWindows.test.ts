/**
 * The phone's window pass (src/live/diarizeWindows.ts) on synthetic input:
 * the speech gate, the window grid + hop widening at the 600-window cap, a
 * three-voice clip end to end (with a fake embedder that maps each voice to
 * its own direction), cancellation, and the numpy-faithful helpers.
 */
import {
  MAX_WINDOWS,
  diarizeWindows,
  eigengapK,
  frameRms,
  kmeansSeeded,
  modeFilter,
  percentileLinear,
  refineAffinity,
  speechMask,
  speechRmsThreshold,
  symmetricEigen,
  windowGrid,
  windowIsSpeech,
  windowLabelRuns,
  type EmbedBatch,
} from "../src/live/diarizeWindows";

const SR = 16000;

/** Deterministic pseudo-noise so tests are repeatable. */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** A clip of `voices` blocks (label, seconds, amplitude); silence between. */
function synthClip(blocks: { voice: number; seconds: number; amp?: number }[], gapSeconds = 0.4): { pcm: Float32Array; truth: [number, number, number][] } {
  const rnd = lcg(7);
  const total = blocks.reduce((s, b) => s + b.seconds + gapSeconds, 0);
  const pcm = new Float32Array(Math.round(total * SR));
  const truth: [number, number, number][] = [];
  let t = 0;
  for (const b of blocks) {
    const a = Math.round(t * SR);
    const e = Math.round((t + b.seconds) * SR);
    const f0 = 120 + 60 * b.voice;
    for (let i = a; i < e; i++) {
      const x = i / SR;
      pcm[i] = (b.amp ?? 0.2) * (Math.sin(2 * Math.PI * f0 * x) + 0.3 * Math.sin(2 * Math.PI * 2.7 * f0 * x)) + 0.01 * (rnd() - 0.5);
    }
    truth.push([t, t + b.seconds, b.voice]);
    t += b.seconds + gapSeconds;
  }
  return { pcm, truth };
}

/** Fake ECAPA: each voice → its own unit direction (+ a little noise), read
 *  off the dominant pitch of the chunk. */
function fakeEmbedder(): EmbedBatch {
  const rnd = lcg(11);
  return async (chunks) =>
    chunks.map((c) => {
      // Dominant pitch by zero crossings of the low-passed signal.
      let crossings = 0;
      for (let i = 1; i < c.length; i++) if ((c[i - 1] < 0) !== (c[i] < 0)) crossings++;
      const f0 = (crossings / 2) * (SR / c.length);
      const voice = Math.max(0, Math.min(2, Math.round((f0 - 120) / 60)));
      const v = new Float32Array(8);
      v[voice] = 1;
      for (let i = 0; i < 8; i++) v[i] += 0.05 * (rnd() - 0.5);
      return v;
    });
}

function coverage(segments: [number, number, number][], truth: [number, number, number][]): number {
  // Frame accuracy under the best 1:1 mapping over truth frames (10 ms).
  const labels = [...new Set(segments.map((s) => s[2]))];
  const truthLabels = [...new Set(truth.map((s) => s[2]))];
  const at = (segs: [number, number, number][], t: number) => segs.find(([s, e]) => t >= s && t < e)?.[2];
  let best = 0;
  const perms = (rest: number[], chosen: number[]) => {
    if (rest.length === 0 || chosen.length === labels.length) {
      const m = new Map(labels.map((l, i) => [l, chosen[i]]));
      let hit = 0;
      let total = 0;
      for (const [s, e, l] of truth) {
        for (let t = s + 0.005; t < e; t += 0.01) {
          total++;
          const p = at(segments, t);
          if (p !== undefined && m.get(p) === l) hit++;
        }
      }
      best = Math.max(best, hit / total);
      return;
    }
    rest.forEach((r, i) => perms([...rest.slice(0, i), ...rest.slice(i + 1)], [...chosen, r]));
  };
  perms(truthLabels, []);
  return best;
}

describe("percentileLinear", () => {
  it("matches numpy's linear interpolation", () => {
    expect(percentileLinear([1, 2, 3, 4], 50)).toBe(2.5);
    expect(percentileLinear([1, 2, 3, 4, 5], 10)).toBeCloseTo(1.4, 12);
    expect(percentileLinear([5, 1, 3], 80)).toBeCloseTo(4.2, 12);
    expect(percentileLinear([7], 80)).toBe(7);
  });
});

describe("speech gate (speaker_id.speech_rms_threshold semantics)", () => {
  it("is the absolute floor on a silent clip, floor-relative in a noisy room, capped at the ceiling", () => {
    expect(speechRmsThreshold(new Float64Array(0))).toBe(0.003);
    expect(speechRmsThreshold([0, 0, 0, 0.5])).toBe(0.003);
    expect(speechRmsThreshold(new Array(100).fill(0.01))).toBeCloseTo(0.015, 12);
    expect(speechRmsThreshold(new Array(100).fill(0.5))).toBe(0.03);
  });

  it("frames a clip at 30 ms and gates windows by ≥30 % speech frames", () => {
    const { pcm } = synthClip([{ voice: 0, seconds: 1.5 }, { voice: 1, seconds: 1.5 }], 1.5);
    const rms = frameRms(pcm, SR);
    expect(rms.length).toBe(Math.floor(pcm.length / 480));
    const speech = speechMask(pcm, SR);
    expect(speech.frameSeconds).toBe(0.03);
    expect(speech.threshold).toBeGreaterThanOrEqual(0.003);
    // Window inside the first voice block: speech; inside the 1.5 s gap: not.
    expect(windowIsSpeech(speech, 0, 24000, SR)).toBe(true);
    expect(windowIsSpeech(speech, Math.round(1.5 * SR), 24000, SR)).toBe(false);
  });
});

describe("window grid", () => {
  it("lays 1.5 s windows every 0.25 s from the clip start", () => {
    const g = windowGrid(30 * SR, SR);
    expect(g.windowSamples).toBe(24000);
    expect(g.hopSamples).toBe(4000);
    expect(g.starts[0]).toBe(0);
    expect(g.starts[1]).toBe(4000);
    expect(g.starts.length).toBe(Math.floor((30 * SR - 24000) / 4000) + 1);
    expect(g.widened).toBe(false);
  });

  it("widens the hop by the smallest integer factor that fits under the 600-window cap", () => {
    // 10 minutes at 0.25 s ≈ 2395 windows → factor 4 → hop 1.0 s → ≤ 600.
    const g = windowGrid(600 * SR, SR);
    expect(g.totalAtRequestedHop).toBeGreaterThan(MAX_WINDOWS);
    expect(g.widened).toBe(true);
    expect(g.hopSamples).toBe(16000);
    expect(g.starts.length).toBeLessThanOrEqual(MAX_WINDOWS);
    expect(g.starts.length).toBe(Math.floor((600 * SR - 24000) / 16000) + 1);
  });
});

describe("refineAffinity / eigengap / k-means", () => {
  it("finds k = 3 for three well-separated directions", () => {
    const embs: Float32Array[] = [];
    const rnd = lcg(3);
    for (let i = 0; i < 30; i++) {
      const v = new Float32Array(4);
      v[i % 3] = 1;
      for (let t = 0; t < 4; t++) v[t] += 0.02 * (rnd() - 0.5);
      embs.push(v);
    }
    const aff = refineAffinity(embs);
    expect(eigengapK(aff, 8).k).toBe(3);
    const labels = kmeansSeeded(
      embs.map((e) => Float64Array.from(e)),
      3,
    );
    const byVoice = [0, 1, 2].map((v) => new Set(labels.filter((_, i) => i % 3 === v)));
    byVoice.forEach((s) => expect(s.size).toBe(1));
    expect(new Set(labels).size).toBe(3);
  });

  it("symmetricEigen recovers a known spectrum", () => {
    // diag(1, 2, 3) rotated by a Givens rotation in the (0, 2) plane.
    const c = Math.cos(0.7);
    const s = Math.sin(0.7);
    const R = [
      [c, 0, -s],
      [0, 1, 0],
      [s, 0, c],
    ];
    const D = [1, 2, 3];
    const data = new Float64Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) data[i * 3 + j] = R[i].reduce((acc, _, t) => acc + R[i][t] * D[t] * R[j][t], 0);
    const e = symmetricEigen({ n: 3, data });
    expect(Array.from(e.values).map((x) => Math.round(x * 1e9) / 1e9)).toEqual([1, 2, 3]);
    // Eigenvector for λ=3 is column 2 of R (up to sign).
    const v = [e.vectors[0][2], e.vectors[1][2], e.vectors[2][2]];
    const dot = Math.abs(v[0] * -s + v[2] * c);
    expect(dot).toBeCloseTo(1, 9);
  });

  it("eigengap on fewer than two windows is k = 1", () => {
    expect(eigengapK({ n: 1, data: new Float64Array([1]) }, 8).k).toBe(1);
  });
});

describe("modeFilter + windowLabelRuns", () => {
  it("mode-filters over temporal neighbours and keeps a tied own label", () => {
    const starts = [0, 0.25, 0.5, 0.75, 1.0, 5.0];
    expect(modeFilter([0, 0, 1, 0, 0, 1], starts, 0.25)).toEqual([0, 0, 0, 0, 0, 1]);
    // A lone window far away keeps its label (its neighbourhood is itself).
    expect(modeFilter([1, 1, 0, 0], [0, 0.25, 3, 3.25], 0.25)).toEqual([1, 1, 0, 0]);
  });

  it("labels every 10 ms frame by the nearest window centre and absorbs runs < 0.5 s", () => {
    const starts = [0, 0.25, 0.5, 0.75, 1.0, 1.25];
    const runs = windowLabelRuns([0, 0, 0, 1, 1, 1], starts, 1.5, 0, 3.0);
    expect(runs.length).toBe(2);
    expect(runs[0][0]).toBe(0);
    expect(runs[0][1]).toBeCloseTo(1.38, 9);
    expect(runs[0][2]).toBe(0);
    expect(runs[1][0]).toBeCloseTo(1.38, 9);
    expect(runs[1][1]).toBe(3.0);
    expect(runs[1][2]).toBe(1);
    // A 0.25 s blip is absorbed into the longer neighbour.
    const blip = windowLabelRuns([0, 0, 1, 0, 0, 0], starts, 1.5, 0, 3.0);
    expect(blip).toEqual([[0, 3.0, 0]]);
    // No windows → one run.
    expect(windowLabelRuns([], [], 1.5, 0, 2)).toEqual([[0, 2, 0]]);
  });
});

describe("diarizeWindows end to end", () => {
  it("separates a synthetic three-voice clip (k = 3, ≥ 0.9 frame accuracy)", async () => {
    const { pcm, truth } = synthClip(
      [
        { voice: 0, seconds: 10 },
        { voice: 1, seconds: 9 },
        { voice: 2, seconds: 10 },
        { voice: 0, seconds: 8 },
        { voice: 1, seconds: 9 },
        { voice: 2, seconds: 10 },
      ],
      1.5, // a 1.5 s pause: windows fully inside it must be gated out
    );
    const progress: string[] = [];
    const r = await diarizeWindows(pcm, SR, fakeEmbedder(), { onProgress: (p) => progress.push(p.stage) });
    expect(r.k).toBe(3);
    expect(r.kEigengap).toBe(3);
    expect(r.hopSeconds).toBe(0.25);
    expect(r.windows).toBeGreaterThan(50);
    expect(r.windows).toBeLessThan(r.totalWindows); // the gaps are gated out
    expect(r.embeddings.length).toBe(r.windows);
    expect(r.embedMs.length).toBe(r.windows);
    expect(coverage(r.segments, truth)).toBeGreaterThanOrEqual(0.9);
    expect(r.segments[0][0]).toBe(0);
    expect(r.segments[r.segments.length - 1][1]).toBeCloseTo(pcm.length / SR, 6);
    expect(progress).toEqual(expect.arrayContaining(["gate", "embed", "cluster", "smooth"]));
    expect(r.timings.totalMs).toBeGreaterThanOrEqual(0);
  });

  it("hands the embedder OWNED zero-offset chunks (the native ORT binding ignores byteOffset)", async () => {
    // onnxruntime-react-native builds the tensor from `data.buffer` and drops
    // the view's byteOffset: a `subarray` view would embed the clip's first
    // 1.5 s for every window (Pixel 10, 2026-08-30: "1 voice found").
    const { pcm } = synthClip([{ voice: 0, seconds: 4 }, { voice: 1, seconds: 4 }]);
    const seen: { offset: number; owns: boolean; len: number; first: number }[] = [];
    const inner = fakeEmbedder();
    const spy: EmbedBatch = async (chunks, sr) => {
      for (const c of chunks) seen.push({ offset: c.byteOffset, owns: c.byteLength === c.buffer.byteLength, len: c.length, first: c[0] });
      return inner(chunks, sr);
    };
    const r = await diarizeWindows(pcm, SR, spy);
    expect(seen.length).toBe(r.windows);
    expect(seen.length).toBeGreaterThan(10);
    expect(seen.every((s) => s.offset === 0 && s.owns && s.len === 24000)).toBe(true);
    // And they are DIFFERENT windows, not copies of the first one.
    const firsts = seen.map((s, i) => pcm[Math.round(r.starts[i] * SR)]);
    expect(seen.map((s) => s.first)).toEqual(firsts);
  });

  it("honours the window cap by widening the hop", async () => {
    const { pcm } = synthClip([{ voice: 0, seconds: 20 }, { voice: 1, seconds: 20 }]);
    const r = await diarizeWindows(pcm, SR, fakeEmbedder(), { maxWindows: 40 });
    expect(r.totalWindows).toBeGreaterThan(40);
    expect(r.windows).toBeLessThanOrEqual(40);
    expect(r.hopSeconds).toBeGreaterThan(0.25);
    expect(r.k).toBe(2);
  });

  it("cancels between embedding batches with an AbortError", async () => {
    const { pcm } = synthClip([{ voice: 0, seconds: 6 }, { voice: 1, seconds: 6 }]);
    const signal = { aborted: false };
    const embed: EmbedBatch = async (chunks, sr) => {
      signal.aborted = true;
      return fakeEmbedder()(chunks, sr);
    };
    await expect(diarizeWindows(pcm, SR, embed, { signal })).rejects.toMatchObject({ name: "AbortError" });
  });

  it("a clip with no speech windows yields one run and k = 0", async () => {
    const pcm = new Float32Array(3 * SR);
    const embed = jest.fn<Promise<ArrayLike<number>[]>, [Float32Array[], number]>();
    const r = await diarizeWindows(pcm, SR, embed);
    expect(embed).not.toHaveBeenCalled();
    expect(r.windows).toBe(0);
    expect(r.segments).toEqual([[0, 3, 0]]);
  });
});
