/**
 * src/live/speakerId.ts — vector math + clustering parity with
 * server/speaker_id.py / server/watch/diarize.py, and the session labeler
 * over enrolled voiceprints. Synthetic 192-d embeddings throughout.
 */
import * as fs from "fs";
import * as path from "path";
import {
  assignSpeakers,
  cosine,
  EcapaEmbedder,
  ECAPA_DIM,
  l2Normalize,
  MATCH_THRESHOLD,
  runningMeanEmbedding,
  SpeakerLabeler,
  StaticVoiceprintStore,
  unknownLabel,
} from "../src/live/speakerId";
import type { OnnxSession, OnnxTensor } from "../src/live/ort";
import { nodeOrtSessionFactory } from "../src/live/testing/ortNode";
import { sineF32, unitVector } from "../src/live/testing/synth";

const D = ECAPA_DIM;

describe("vector math", () => {
  it("l2Normalize yields unit length and leaves a zero vector alone", () => {
    const v = l2Normalize([3, 4]);
    expect(v[0]).toBeCloseTo(0.6);
    expect(v[1]).toBeCloseTo(0.8);
    expect(Array.from(l2Normalize([0, 0]))).toEqual([0, 0]);
  });

  it("cosine is 1 for parallel, 0 for orthogonal / mismatched shapes", () => {
    expect(cosine([1, 0], [2, 0])).toBeCloseTo(1);
    expect(cosine([1, 0], [0, 1])).toBeCloseTo(0);
    expect(cosine([1, 0], [1, 0, 0])).toBe(0);
    expect(cosine([], [])).toBe(0);
  });

  it("runningMeanEmbedding is count-weighted and renormalized", () => {
    const a = l2Normalize([1, 0]);
    const b = l2Normalize([0, 1]);
    const m1 = runningMeanEmbedding(null, 0, b);
    expect(Array.from(m1)).toEqual(Array.from(b));
    const m2 = runningMeanEmbedding(a, 3, b); // (3a + b)/4, normalized
    expect(m2[0]).toBeCloseTo(3 / Math.sqrt(10));
    expect(m2[1]).toBeCloseTo(1 / Math.sqrt(10));
  });
});

describe("assignSpeakers (diarize.py parity)", () => {
  it("labels self only against a real print, clusters others, keeps nulls", () => {
    const self = unitVector(D, 0);
    const selfish = unitVector(D, 0, 0.15, 2);
    const other = unitVector(D, 1);
    const otherish = unitVector(D, 1, 0.15, 3);
    const third = unitVector(D, 2);
    expect(assignSpeakers([selfish, other, otherish, null, third], self)).toEqual([
      "self",
      "other-1",
      "other-1",
      null,
      "other-2",
    ]);
  });

  it("never says self without a print", () => {
    expect(assignSpeakers([unitVector(D, 0)], null)).toEqual(["other-1"]);
  });
});

describe("SpeakerLabeler", () => {
  const you = { personId: "p-you", displayName: "You", isSelf: true, embedding: unitVector(D, 0) };
  const mom = { personId: "p-mom", displayName: "Mom", isSelf: false, embedding: unitVector(D, 1) };

  it("matches enrolled people greedily above MATCH_THRESHOLD", () => {
    const lab = new SpeakerLabeler([you, mom]);
    const v1 = lab.label(unitVector(D, 0, 0.15, 11));
    expect(v1.speaker).toBe("You");
    expect(v1.isSelf).toBe(true);
    expect(v1.personId).toBe("p-you");
    expect(v1.score as number).toBeGreaterThanOrEqual(MATCH_THRESHOLD);
    const v2 = lab.label(unitVector(D, 1, 0.15, 12));
    expect(v2.speaker).toBe("Mom");
    expect(v2.isSelf).toBe(false);
  });

  it("clusters strangers as Speaker A/B and knows they are not self", () => {
    const lab = new SpeakerLabeler([you, mom]);
    const a1 = lab.label(unitVector(D, 5));
    const a2 = lab.label(unitVector(D, 5, 0.15, 13));
    const b1 = lab.label(unitVector(D, 6));
    expect(a1.speaker).toBe("Speaker A");
    expect(a1.score).toBeNull(); // minted a cluster: no join score
    expect(a2.speaker).toBe("Speaker A");
    expect(a2.score as number).toBeGreaterThan(0.55);
    expect(b1.speaker).toBe("Speaker B");
    expect([a1.isSelf, a2.isSelf, b1.isSelf]).toEqual([false, false, false]);
    expect(unknownLabel(2)).toBe("Speaker C");
  });

  it("with nobody enrolled as self, isSelf is null (no honest basis)", () => {
    const lab = new SpeakerLabeler([mom]);
    expect(lab.label(unitVector(D, 7)).isSelf).toBeNull();
    const none = new SpeakerLabeler([]);
    expect(none.label(unitVector(D, 7))).toMatchObject({ speaker: "Speaker A", isSelf: null });
  });

  it("no embedding => Unknown, nothing decided", () => {
    const lab = new SpeakerLabeler([you]);
    expect(lab.label(null)).toEqual({ speaker: "Unknown", personId: null, displayName: null, isSelf: null, score: null });
  });

  it("reset forgets session clusters but not enrollment", () => {
    const lab = new SpeakerLabeler([you]);
    lab.label(unitVector(D, 5));
    lab.label(unitVector(D, 6));
    lab.reset();
    expect(lab.label(unitVector(D, 6)).speaker).toBe("Speaker A");
    expect(lab.enrolledCount).toBe(1);
  });

  it("StaticVoiceprintStore lists what it was given", async () => {
    expect(await new StaticVoiceprintStore([you]).list()).toEqual([you]);
  });
});

describe("EcapaEmbedder", () => {
  it("feeds f32 [1, T] to the session and normalizes the first output", async () => {
    const seen: Record<string, OnnxTensor>[] = [];
    const session: OnnxSession = {
      inputNames: ["wav"],
      outputNames: ["embedding"],
      async run(feeds) {
        seen.push(feeds);
        return { embedding: { type: "float32", data: Float32Array.from({ length: D }, (_, i) => (i === 3 ? 4 : 0)), dims: [1, D] } };
      },
      async release() {},
    };
    const emb = await new EcapaEmbedder(session).embed(sineF32(200, 0.5), 16000);
    expect(seen[0].wav.dims).toEqual([1, 8000]);
    expect(emb.length).toBe(D);
    expect(emb[3]).toBeCloseTo(1);
    await expect(new EcapaEmbedder(session).embed(sineF32(200, 0.5), 44100)).rejects.toThrow(/16 kHz/);
  });

  // Foundation B's scripts/export_ecapa_onnx.py produces this file; when it
  // exists locally the real model runs here (parity: same speaker → high
  // cosine, different frequency content → lower). Skipped honestly otherwise.
  const candidates = [
    path.resolve(__dirname, "../assets/models/ecapa.onnx"),
    path.resolve(__dirname, "../../../server/.ecapa_cache/ecapa.onnx"),
    path.resolve(__dirname, "../../../tmp/ecapa.onnx"),
  ];
  const ecapaPath = candidates.find((p) => fs.existsSync(p));
  const maybe = ecapaPath ? it : it.skip;
  maybe("real ECAPA export: 192-d unit output, self-similar > cross-similar", async () => {
    const session = await nodeOrtSessionFactory()(ecapaPath as string);
    const embedder = new EcapaEmbedder(session);
    const t0 = performance.now();
    const a1 = await embedder.embed(sineF32(140, 1.5, 0.4), 16000);
    const a2 = await embedder.embed(sineF32(145, 1.5, 0.4), 16000);
    const b = await embedder.embed(sineF32(320, 1.5, 0.4), 16000);
    // eslint-disable-next-line no-console
    console.log(`[ecapa] ${((performance.now() - t0) / 3).toFixed(1)} ms per 1.5 s segment on onnxruntime-node`);
    expect(a1.length).toBe(D);
    expect(cosine(a1, a2)).toBeGreaterThan(cosine(a1, b));
    await session.release();
  });
});
