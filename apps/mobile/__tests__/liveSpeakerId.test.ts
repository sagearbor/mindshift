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
  CROSS_MATCH_MARGIN,
  CROSS_MATCH_MIN_SETTINGS,
  CROSS_MATCH_THRESHOLD,
  EcapaEmbedder,
  ECAPA_DIM,
  identifyClusters,
  MIN_CLUSTER_SECONDS,
  l2Normalize,
  MATCH_THRESHOLD,
  runningMeanEmbedding,
  SpeakerLabeler,
  StaticVoiceprintStore,
  unknownLabel,
  type EnrolledPerson,
} from "../src/live/speakerId";
import type { OnnxSession, OnnxTensor } from "../src/live/ort";
import { nodeOrtSessionFactory } from "../src/live/testing/ortNode";
import { AUDIO_FIXTURES_DIR, loadWav16k, sineF32, unitVector, vectorAtCosine } from "../src/live/testing/synth";
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

interface ParityCase {
  name: string;
  people: Record<string, { display_name: string; is_self: boolean; settings: number; embedding: number[] }>;
  speakers: Record<string, number[]>;
  expected: {
    matched: Record<string, string>;
    basis: Record<string, "absolute" | "contrast" | null>;
    scores: Record<string, Record<string, number>>;
  };
}

describe("identifyClusters (speaker_id.identify_from_embeddings parity)", () => {
  // Generated FROM the Python (server/tests/test_voice_cross_match.py pins
  // it in the other direction): absolute, the poker-shape contrast match,
  // and the margin / floor / settings / solo / one-to-one rejections.
  const fixture = JSON.parse(
    fs.readFileSync(path.join(__dirname, "fixtures", "speakerCrossMatch.json"), "utf8"),
  ) as { constants: Record<string, number>; cases: ParityCase[] };

  it("uses the server's constants", () => {
    expect(fixture.constants).toEqual({
      match_threshold: MATCH_THRESHOLD,
      cross_match_threshold: CROSS_MATCH_THRESHOLD,
      cross_match_margin: CROSS_MATCH_MARGIN,
      cross_match_min_settings: CROSS_MATCH_MIN_SETTINGS,
    });
    expect(fixture.cases.map((c) => c.name)).toEqual(
      expect.arrayContaining(["absolute_0.80_vs_0.10", "contrast_poker_0.42_vs_0.19_0.12", "reject_margin_0.45_vs_0.35"]),
    );
  });

  it.each(fixture.cases.map((c) => [c.name, c] as const))("%s", (_name, c) => {
    const people: EnrolledPerson[] = Object.entries(c.people).map(([personId, p]) => ({
      personId,
      displayName: p.display_name,
      isSelf: p.is_self,
      embedding: p.embedding,
      settings: p.settings,
    }));
    const clusters = new Map(Object.entries(c.speakers));
    const got = identifyClusters(clusters, people);
    const matched: Record<string, string> = {};
    const basis: Record<string, string | null> = {};
    for (const label of clusters.keys()) basis[label] = got.get(label)?.basis ?? null;
    for (const [label, id] of got) {
      matched[label] = id.personId;
      expect(id.score).toBeCloseTo(c.expected.scores[label][id.personId], 4);
    }
    expect(matched).toEqual(c.expected.matched);
    expect(basis).toEqual(c.expected.basis);
  });

  it("an unknown or zero settings count is one (contrast off), like the server", () => {
    const you: EnrolledPerson = { personId: "self", displayName: "You", isSelf: true, embedding: [1, 0] };
    const clusters = new Map<string, number[]>([["A", [0.42, Math.sqrt(1 - 0.42 ** 2)]], ["B", [0.19, Math.sqrt(1 - 0.19 ** 2)]]]);
    expect(identifyClusters(clusters, [you]).size).toBe(0);
    expect(identifyClusters(clusters, [{ ...you, settings: 0 }]).size).toBe(0);
    expect(identifyClusters(clusters, [{ ...you, settings: null }]).size).toBe(0);
    expect(identifyClusters(clusters, [{ ...you, settings: 2 }].map((p) => p)).get("A")).toEqual({
      personId: "self",
      basis: "contrast",
      score: 0.42,
    });
  });
});

describe("SpeakerLabeler", () => {
  const you = { personId: "p-you", displayName: "You", isSelf: true, embedding: unitVector(D, 0) };
  const mom = { personId: "p-mom", displayName: "Mom", isSelf: false, embedding: unitVector(D, 1) };
  // The owner's real cross-room print: two recordings pooled.
  const youAway = { ...you, settings: 2 };

  it("contrast: a cluster gains self once a second cluster exists to contrast against (the poker-night shape)", () => {
    const lab = new SpeakerLabeler([youAway]);
    const owner = vectorAtCosine(D, 0.42, 0, 1);
    const stranger = vectorAtCosine(D, 0.19, 0, 2);
    // Alone, 0.42 is an honest miss: no second voice to contrast with.
    const first = lab.label(owner, 2.0);
    expect(first).toMatchObject({ speaker: "Speaker A", personId: null, isSelf: false, basis: null });
    expect(lab.clusterAssignments().size).toBe(0);
    const rev0 = lab.identityRevision;
    // The stranger's cluster supplies the contrast: A is now the owner.
    const second = lab.label(stranger, 2.0);
    expect(second).toMatchObject({ speaker: "Speaker B", personId: null, isSelf: false, basis: null });
    expect(lab.identityRevision).toBe(rev0 + 1);
    expect(Array.from(lab.clusterAssignments().entries())).toEqual([
      ["Speaker A", { personId: "p-you", basis: "contrast", score: 0.42, displayName: "You", isSelf: true }],
    ]);
    // The owner's next turn carries the person on the RAW label (the wire
    // key never changes), with the contrast score and basis.
    const third = lab.label(owner, 2.0);
    expect(third).toEqual({
      speaker: "Speaker A",
      personId: "p-you",
      displayName: "You",
      isSelf: true,
      score: 0.42,
      selfScore: expect.closeTo(0.42, 5),
      basis: "contrast",
    });
    expect(lab.identityRevision).toBe(rev0 + 1); // nothing moved
  });

  it("contrast never fires on a single-recording print, below the floor, or inside the margin", () => {
    for (const [print, a, b] of [
      [you, 0.42, 0.19], // settings 1 (default)
      [youAway, 0.39, 0.05], // under the 0.40 floor
      [youAway, 0.45, 0.35], // 0.10 gap: inside different-people noise
    ] as const) {
      const lab = new SpeakerLabeler([print]);
      lab.label(vectorAtCosine(D, a, 0, 1), 2.0);
      lab.label(vectorAtCosine(D, b, 0, 2), 2.0);
      expect(lab.clusterAssignments().size).toBe(0);
      expect(lab.label(vectorAtCosine(D, a, 0, 1), 2.0)).toMatchObject({ speaker: "Speaker A", isSelf: false, basis: null });
    }
  });

  it("revises: a later cluster that beats the earlier one by the margin takes the person (a person is one voice)", () => {
    const lab = new SpeakerLabeler([youAway]);
    lab.label(vectorAtCosine(D, 0.42, 0, 1), 2.0); // A
    lab.label(vectorAtCosine(D, 0.19, 0, 2), 2.0); // B -> A is self by contrast
    expect(lab.clusterAssignments().get("Speaker A")?.personId).toBe("p-you");
    const rev = lab.identityRevision;
    // C scores 0.60: beats A's 0.42 by 0.18 >= margin — self moves to C.
    const c = lab.label(vectorAtCosine(D, 0.6, 0, 3), 2.0);
    expect(c).toMatchObject({ speaker: "Speaker C", personId: "p-you", isSelf: true, score: 0.6, basis: "contrast" });
    expect(lab.identityRevision).toBe(rev + 1);
    expect(Array.from(lab.clusterAssignments().keys())).toEqual(["Speaker C"]);
    // A is an unidentified voice again — honestly not self (a self print exists).
    expect(lab.label(vectorAtCosine(D, 0.42, 0, 1), 2.0)).toMatchObject({ speaker: "Speaker A", personId: null, isSelf: false, basis: null });
  });

  it("the absolute path is untouched: >= 0.65 matches outright, founds no cluster, basis absolute", () => {
    const lab = new SpeakerLabeler([you]); // single recording is enough
    expect(lab.label(vectorAtCosine(D, 0.7, 0, 1), 2.0)).toEqual({
      speaker: "You",
      personId: "p-you",
      displayName: "You",
      isSelf: true,
      score: expect.closeTo(0.7, 5),
      selfScore: expect.closeTo(0.7, 5),
      basis: "absolute",
    });
    expect(lab.clusterAssignments().size).toBe(0);
    expect(lab.label(unitVector(D, 5), 2.0).speaker).toBe("Speaker A");
  });

  it("reset forgets identities and the revision counter", () => {
    const lab = new SpeakerLabeler([youAway]);
    lab.label(vectorAtCosine(D, 0.42, 0, 1), 2.0);
    lab.label(vectorAtCosine(D, 0.19, 0, 2), 2.0);
    expect(lab.clusterAssignments().size).toBe(1);
    lab.reset();
    expect(lab.clusterAssignments().size).toBe(0);
    expect(lab.identityRevision).toBe(0);
  });

  it("matches enrolled people greedily above MATCH_THRESHOLD", () => {
    const lab = new SpeakerLabeler([you, mom]);
    const v1 = lab.label(unitVector(D, 0, 0.15, 11));
    expect(v1.speaker).toBe("You");
    expect(v1.isSelf).toBe(true);
    expect(v1.personId).toBe("p-you");
    expect(v1.basis).toBe("absolute");
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
    expect(short).toEqual({ speaker: "Unknown", personId: null, displayName: null, isSelf: null, score: null,
      selfScore: 0, basis: null });
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
    expect(lab.label(null)).toEqual({ speaker: "Unknown", personId: null, displayName: null, isSelf: null, score: null, basis: null });
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
