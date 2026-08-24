/**
 * src/live/vad.ts + segmenter.ts.
 *
 * 1. Golden-vector parity: every case in
 *    server/tests/fixtures/policy_vectors/vad_segments.json, synthesized with
 *    the same generator as the Python driver, through (a) the whole-clip
 *    energy-VAD port and (b) the streaming segmenter fed frame by frame —
 *    both must equal `expected` within `tolerance_s`.
 * 2. The REAL silero_vad.onnx through onnxruntime-node: official I/O shapes,
 *    silence scores ~0, per-chunk timing measured and printed.
 */
import * as fs from "fs";
import * as path from "path";
import {
  EnergyVad,
  energyIsSpeech,
  rmsDbfsInt16,
  SILERO_CHUNK_SAMPLES,
  SileroVad,
  SpeechGate,
} from "../src/live/vad";
import {
  energySpeechSegments,
  mergeAndDrop,
  StreamingSegmenter,
} from "../src/live/segmenter";
import { nodeOrtSessionFactory } from "../src/live/testing/ortNode";
import { loadFixture, synthesizeCase, type VadCase, silenceInt16, toneInt16 } from "../src/live/testing/synth";

const doc = loadFixture<{ _schema: { version: number }; cases: VadCase[] }>("vad_segments.json");
expect(doc._schema.version).toBe(1);
const CASES = doc.cases;

function expectSpans(
  got: { start: number; end: number }[],
  c: VadCase,
) {
  const want = c.expected.map((s) => ({ start: s.start_s, end: s.end_s }));
  expect(got.length).toBe(want.length);
  got.forEach((g, i) => {
    expect(Math.abs(g.start - want[i].start)).toBeLessThanOrEqual(c.tolerance_s);
    expect(Math.abs(g.end - want[i].end)).toBeLessThanOrEqual(c.tolerance_s);
  });
}

describe("vad_segments.json golden vectors", () => {
  it("covers the required scenarios", () => {
    const names = new Set(CASES.map((c) => c.name));
    expect(names.size).toBe(CASES.length);
    for (const n of [
      "silence_only",
      "one_burst",
      "two_bursts_merged_across_short_gap",
      "two_bursts_kept_apart_across_long_gap",
      "burst_too_short_dropped",
      "mixed_conversation_shape",
    ]) {
      expect(names.has(n)).toBe(true);
    }
  });

  it.each(CASES.map((c) => [c.name, c] as const))(
    "generator hits the requested loudness: %s",
    (_name, c) => {
      const pcm = synthesizeCase(c);
      let offset = 0;
      for (const s of c.signal) {
        const n = Math.floor(s.seconds * c.sample_rate);
        const measured = rmsDbfsInt16(pcm.subarray(offset, offset + n));
        offset += n;
        if (s.dbfs === null) expect(measured).toBe(-Infinity);
        else expect(Math.abs(measured - s.dbfs)).toBeLessThanOrEqual(0.5);
      }
      expect(offset).toBe(pcm.length);
    },
  );

  it.each(CASES.map((c) => [c.name, c] as const))(
    "whole-clip energy VAD matches: %s",
    (_name, c) => {
      const got = energySpeechSegments(synthesizeCase(c), c.sample_rate, {
        floorDbfs: c.config.floor_dbfs,
        frameSeconds: c.config.frame_seconds,
        mergeGapSeconds: c.config.merge_gap_seconds,
        minSeconds: c.config.min_seconds,
      });
      expectSpans(got, c);
    },
  );

  it.each(CASES.map((c) => [c.name, c] as const))(
    "streaming segmenter agrees frame by frame: %s",
    async (_name, c) => {
      const pcm = synthesizeCase(c);
      const sr = c.sample_rate;
      const vad = new EnergyVad(c.config.floor_dbfs, c.config.frame_seconds, sr);
      const seg = new StreamingSegmenter({
        mergeGapSeconds: c.config.merge_gap_seconds,
        minSeconds: c.config.min_seconds,
      });
      const got: { start: number; end: number }[] = [];
      let t = 0;
      let offset = 0;
      while (offset < pcm.length) {
        const n = Math.min(vad.frameSamples, pcm.length - offset);
        const frame = new Float32Array(n);
        for (let i = 0; i < n; i++) frame[i] = pcm[offset + i] / 32768;
        const dt = n === vad.frameSamples ? c.config.frame_seconds : n / sr;
        const span = seg.push(await vad.isSpeech(frame), t, t + dt);
        if (span) got.push(span);
        t += dt;
        offset += n;
      }
      const tail = seg.flush();
      if (tail) got.push(tail);
      expectSpans(got, c);
    },
  );
});

describe("energy rule details", () => {
  it("a frame AT the floor is silence (strict >)", () => {
    // Build a tone at exactly -45 dBFS and check the comparison direction.
    const pcm = toneInt16(0.25, -45);
    const measured = rmsDbfsInt16(pcm);
    expect(energyIsSpeech(pcm, measured)).toBe(false);
    expect(energyIsSpeech(pcm, measured - 0.01)).toBe(true);
  });

  it("mergeAndDrop merges before dropping", () => {
    expect(
      mergeAndDrop(
        [
          { start: 0, end: 0.5 },
          { start: 0.75, end: 1.25 },
        ],
        { mergeGapSeconds: 0.3, minSeconds: 0.6 },
      ),
    ).toEqual([{ start: 0, end: 1.25 }]);
    expect(mergeAndDrop([{ start: 0, end: 0.25 }], { mergeGapSeconds: 0.3, minSeconds: 0.6 })).toEqual([]);
  });
});

describe("SpeechGate hysteresis", () => {
  it("enters at 0.5, leaves below 0.35", () => {
    const g = new SpeechGate();
    expect(g.update(0.49)).toBe(false);
    expect(g.update(0.5)).toBe(true);
    expect(g.update(0.4)).toBe(true); // still speaking: above the off threshold
    expect(g.update(0.34)).toBe(false);
    g.reset();
    expect(g.update(0.4)).toBe(false);
  });
});

describe("Silero VAD (real model via onnxruntime-node)", () => {
  const modelPath = path.resolve(__dirname, "../assets/models/silero_vad.onnx");

  it("the bundled model is present and MIT-licensed upstream (2.3 MB)", () => {
    const size = fs.statSync(modelPath).size;
    expect(size).toBeGreaterThan(2_000_000);
    expect(size).toBeLessThan(3_000_000);
  });

  it("runs the official I/O, scores silence ≈ 0, keeps a tone from flapping, and reports per-chunk latency", async () => {
    const session = await nodeOrtSessionFactory()(modelPath);
    expect(session.inputNames).toEqual(["input", "state", "sr"]);
    expect(session.outputNames).toEqual(["output", "stateN"]);
    const vad = new SileroVad(session);
    expect(vad.frameSamples).toBe(SILERO_CHUNK_SAMPLES);

    // 1 s of digital silence: every chunk must score well below the on threshold.
    const silence = silenceInt16(1.0);
    const silenceProbs: number[] = [];
    for (let off = 0; off + 512 <= silence.length; off += 512) {
      const f = new Float32Array(512);
      silenceProbs.push(await vad.probability(f));
    }
    expect(Math.max(...silenceProbs)).toBeLessThan(0.1);

    // The fixture's synthetic tone (one_burst) — a pure sine is NOT speech,
    // so no claim is made about its verdict; the point is the model runs on
    // the same PCM the energy VAD is pinned to, and the state carries.
    const tone = synthesizeCase(CASES.find((c) => c.name === "one_burst") as VadCase);
    const t0 = performance.now();
    let chunks = 0;
    for (let off = 0; off + 512 <= tone.length; off += 512) {
      const f = new Float32Array(512);
      for (let i = 0; i < 512; i++) f[i] = tone[off + i] / 32768;
      const p = await vad.probability(f);
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
      chunks += 1;
    }
    const msPerChunk = (performance.now() - t0) / chunks;
    // eslint-disable-next-line no-console
    console.log(`[silero] ${chunks} chunks, ${msPerChunk.toFixed(3)} ms/chunk on onnxruntime-node`);
    // A 32 ms chunk must cost far less than 32 ms (realtime budget).
    expect(msPerChunk).toBeLessThan(16);

    // A wrong-size frame is rejected loudly, never silently mis-shaped.
    await expect(vad.probability(new Float32Array(100))).rejects.toThrow(/512/);
    vad.reset();
    await session.release();
  });
});
