/**
 * src/live/overlapProbe.ts — the dark single-mic overlap probe: window
 * classification, the summary, windowing, and the async orchestration.
 */
import {
  OVERLAP_HOP_SECONDS,
  OVERLAP_WINDOW_SECONDS,
  classifyWindow,
  probeOverlapAsync,
  summarizeOverlap,
  windowStarts,
} from "../src/live/overlapProbe";

describe("classifyWindow", () => {
  it("one voice clearly ahead -> that voice; both present and close -> mixed; both weak -> unclear", () => {
    expect(classifyWindow({ start: 0, self: 0.7, otherMax: 0.3 })).toBe("self");
    expect(classifyWindow({ start: 0, self: 0.25, otherMax: 0.6 })).toBe("other");
    expect(classifyWindow({ start: 0, self: 0.55, otherMax: 0.5 })).toBe("mixed");
    expect(classifyWindow({ start: 0, self: 0.1, otherMax: 0.15 })).toBe("unclear");
    expect(classifyWindow({ start: 0, self: 0.7, otherMax: null })).toBe("self"); // nobody else known yet
    expect(classifyWindow({ start: 0, self: null, otherMax: null })).toBe("unclear");
  });
});

describe("summarizeOverlap", () => {
  it("counts mixed windows and the longest consecutive run in seconds", () => {
    const w = (self: number, other: number, i: number) => ({ start: i * 0.5, self, otherMax: other });
    const s = summarizeOverlap([
      w(0.7, 0.2, 0), // self
      w(0.5, 0.5, 1), // mixed
      w(0.5, 0.48, 2), // mixed
      w(0.52, 0.5, 3), // mixed
      w(0.7, 0.1, 4), // self
      w(0.45, 0.5, 5), // mixed
    ]);
    expect(s.windows).toBe(6);
    expect(s.voices).toEqual(["self", "mixed", "mixed", "mixed", "self", "mixed"]);
    expect(s.mixedSeconds).toBe(2.0);
    expect(s.longestMixedRunSeconds).toBe(1.5);
  });
});

describe("windowStarts", () => {
  it("covers the last windows of a turn in time order, none for a turn shorter than one window", () => {
    expect(windowStarts(1.0)).toEqual([]);
    expect(windowStarts(2.5)).toEqual([0, 0.5, 1.0]);
    expect(windowStarts(30, OVERLAP_WINDOW_SECONDS, OVERLAP_HOP_SECONDS, 4)).toEqual([27, 27.5, 28, 28.5]);
  });
});

describe("probeOverlapAsync", () => {
  it("embeds each window, scores it, skips failures, and summarizes", async () => {
    const sr = 16000;
    const pcm = new Float32Array(3 * sr); // 3.0 s -> windows at 0, 0.5, 1.0, 1.5
    const embedded: number[] = [];
    const embed = async (w: Float32Array) => {
      embedded.push(w.length);
      if (embedded.length === 2) throw new Error("bad window");
      return new Float32Array([embedded.length]);
    };
    const score = (e: ArrayLike<number>) => (e[0] === 4 ? { self: 0.5, otherMax: 0.5 } : { self: 0.8, otherMax: 0.2 });
    const r = await probeOverlapAsync(pcm, sr, embed, score);
    expect(embedded).toEqual([24000, 24000, 24000, 24000]);
    expect(r!.windows).toBe(3); // one skipped
    expect(r!.voices).toEqual(["self", "self", "mixed"]);
    expect(r!.mixedSeconds).toBe(0.5);
  });

  it("null for a turn too short to window", async () => {
    const r = await probeOverlapAsync(new Float32Array(8000), 16000, async () => new Float32Array(1), () => ({ self: 1, otherMax: 0 }));
    expect(r).toBeNull();
  });
});
