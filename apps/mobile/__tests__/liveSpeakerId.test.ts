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
  MIN_CLUSTER_SECONDS,
  l2Normalize,
  MATCH_THRESHOLD,
  runningMeanEmbedding,
  SpeakerLabeler,
  StaticVoiceprintStore,
  unknownLabel,
} from "../src/live/speakerId";
import type { OnnxSession, OnnxTensor } from "../src/live/ort";
import { nodeOrtSessionFactory } from "../src/live/testing/ortNode";
import { AUDIO_FIXTURES_DIR, loadWav16k, sineF32, unitVector } from "../src/live/testing/synth";
import { findEcapaModel } from "../src/live/replay/sceneReplay";

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

  it("a segment under MIN_CLUSTER_SECONDS may match but never founds a cluster (Unknown, isSelf null)", () => {
    const lab = new SpeakerLabeler([you, mom]);
    const short = lab.label(unitVector(D, 7), 1.0);
    expect(short).toEqual({ speaker: "Unknown", personId: null, displayName: null, isSelf: null, score: null });
    // Nothing was minted: the next long stranger is the FIRST cluster.
    const long = lab.label(unitVector(D, 7), 2.0);
    expect(long.speaker).toBe("Speaker A");
    expect(long.isSelf).toBe(false);
    // Short segments still join an existing cluster or match a print.
    expect(lab.label(unitVector(D, 7, 0.15, 21), 0.8).speaker).toBe("Speaker A");
    expect(lab.label(unitVector(D, 0, 0.15, 22), 0.8)).toMatchObject({ speaker: "You", isSelf: true });
    // No duration given => no guard (batch callers with known-good segments).
    expect(lab.label(unitVector(D, 9)).speaker).toBe("Speaker B");
    expect(MIN_CLUSTER_SECONDS).toBe(1.5);
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

  // Cross-runtime parity, the property on-device speaker-ID depends on: the
  // ONNX export (server/ecapa_onnx.py — served by GET /models/ecapa.onnx)
  // run under onnxruntime-node must land on the SAME point in embedding
  // space as the server's torch model for the same real speech, else a
  // server-enrolled print scores lower on the phone for no visible reason.
  // The torch side is the committed fixture written by
  // `scripts/export_ecapa_onnx.py --reference-json`; the audio is the real
  // family/poker recordings under server/tests/fixtures/audio, read with
  // the same int16/32768 arithmetic. Skipped honestly (never fake-passed)
  // when the ~80 MB export isn't on this machine: produce it with
  // `tmp/venv-voice/bin/python scripts/export_ecapa_onnx.py` (it lands in
  // server/.ecapa_cache under its revision-stamped name) or via the server.
  interface Reference {
    revision: string;
    dim: number;
    slices: { fixture: string; label: string; start_s: number; end_s: number; embedding: number[] }[];
  }
  const reference = JSON.parse(
    fs.readFileSync(path.resolve(__dirname, "fixtures/ecapa_reference.json"), "utf8"),
  ) as Reference;
  // Same lookup the replay harness uses (revision-pinned cache first, then
  // the main checkout's cache when running inside a git worktree).
  const ecapaPath = findEcapaModel();
  const fixturesPresent = reference.slices.every((s) => fs.existsSync(path.join(AUDIO_FIXTURES_DIR, s.fixture)));
  const maybe = ecapaPath && fixturesPresent ? it : it.skip;

  maybe("real ECAPA export matches the server's torch embeddings on real speech (cosine > 0.999)", async () => {
    const session = await nodeOrtSessionFactory()(ecapaPath as string);
    const embedder = new EcapaEmbedder(session);
    const audio = new Map<string, Float32Array>();
    const pcmFor = (name: string) => {
      let pcm = audio.get(name);
      if (!pcm) {
        pcm = loadWav16k(path.join(AUDIO_FIXTURES_DIR, name));
        audio.set(name, pcm);
      }
      return pcm;
    };
    // Warm the session so the first timed slice isn't paying setup cost.
    await embedder.embed(pcmFor(reference.slices[0].fixture).subarray(0, 16000 * 1.5), 16000);

    const byLabel = new Map<string, Float32Array>();
    let worst = 1;
    const perSecondAndAHalf: number[] = [];
    for (const slice of reference.slices) {
      const clip = pcmFor(slice.fixture).subarray(slice.start_s * 16000, slice.end_s * 16000);
      const t0 = performance.now();
      const got = await embedder.embed(clip, 16000);
      const ms = performance.now() - t0;
      if (slice.end_s - slice.start_s === 1.5) perSecondAndAHalf.push(ms);
      expect(got.length).toBe(reference.dim);
      let norm = 0;
      for (const x of got) norm += x * x;
      expect(Math.sqrt(norm)).toBeCloseTo(1, 3);
      const cos = cosine(got, slice.embedding);
      worst = Math.min(worst, cos);
      byLabel.set(slice.label, got);
      // eslint-disable-next-line no-console
      console.log(`[ecapa] ${slice.label.padEnd(24)} ${(slice.end_s - slice.start_s).toFixed(1)}s cosine(onnx-node, torch)=${cos.toFixed(6)} ${ms.toFixed(1)} ms`);
      expect(cos).toBeGreaterThan(0.999);
    }
    const avg = perSecondAndAHalf.reduce((a, b) => a + b, 0) / Math.max(1, perSecondAndAHalf.length);
    // eslint-disable-next-line no-console
    console.log(`[ecapa] worst cosine ${worst.toFixed(6)}; ${avg.toFixed(1)} ms per 1.5 s slice on onnxruntime-node`);

    // The space still discriminates: the owner's two slices are far closer
    // to each other than to his son's — the relationship MATCH_THRESHOLD
    // relies on (same assertion as the server's parity test).
    const sageA = byLabel.get("Sage 0-5s") as Float32Array;
    const sageB = byLabel.get("Sage 10-15s") as Float32Array;
    const asher = byLabel.get("Asher 5-10s") as Float32Array;
    expect(cosine(sageA, sageB)).toBeGreaterThan(cosine(sageA, asher) + 0.2);
    // And a SpeakerLabeler seeded with the server's reference print for
    // Sage recognizes his phone-side embedding as him — the actual product
    // path (server-enrolled voiceprint, on-device turn embedding).
    const sageRef = reference.slices.find((s) => s.label === "Sage 0-5s") as Reference["slices"][number];
    const labeler = new SpeakerLabeler([
      { personId: "sage", displayName: "You", isSelf: true, embedding: sageRef.embedding },
    ]);
    expect(labeler.label(sageB)).toMatchObject({ speaker: "You", isSelf: true });
    expect(labeler.label(asher)).toMatchObject({ speaker: "Speaker A", isSelf: false });
    await session.release();
  });
});
