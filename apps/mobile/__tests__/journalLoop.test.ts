/**
 * live/journalLoop.ts with injected seams: the energy VAD over synthetic
 * PCM, a fake embedder that tells the owner (loud tone) from another voice
 * (quiet tone), and the real SpeakerLabeler with an enrolled owner print.
 *
 * Pins the rules the owner is relying on: only `isSelf === true` stretches
 * are handed to `keep` (with ±1 s of context, bounded by silence), other
 * voices are discarded and never written, sub-1.0 s stretches are never
 * even embedded, and one embedding is computed per completed segment.
 */
import { JournalLoop, type KeptSegmentMeta } from "../src/live/journalLoop";
import { EnergyVad } from "../src/live/vad";
import { SpeakerLabeler, type Embedder } from "../src/live/speakerId";
import { silenceInt16, toneInt16, unitVector } from "../src/live/testing/synth";

const SR = 16000;
const DIM = 192;
const SELF_VEC = unitVector(DIM, 0);
const OTHER_VEC = unitVector(DIM, 1);

/** Loud tone = the owner; quiet tone = someone else. */
class AmplitudeEmbedder implements Embedder {
  calls = 0;
  async embed(pcm: Float32Array): Promise<Float32Array> {
    this.calls += 1;
    let acc = 0;
    for (let i = 0; i < pcm.length; i++) acc += pcm[i] * pcm[i];
    const rms = Math.sqrt(acc / Math.max(1, pcm.length));
    return rms > 0.1 ? Float32Array.from(SELF_VEC) : Float32Array.from(OTHER_VEC);
  }
}

const SELF_DB = -10; // rms ≈ 0.22
const OTHER_DB = -25; // rms ≈ 0.04

function labelerWithSelf(): SpeakerLabeler {
  return new SpeakerLabeler([
    { personId: "self", displayName: "You", isSelf: true, embedding: SELF_VEC, settings: 3 },
  ]);
}

interface Harness {
  loop: JournalLoop;
  embedder: AmplitudeEmbedder;
  kept: { pcm: Int16Array; meta: KeptSegmentMeta }[];
  discarded: { reason: string; isSelf: boolean | null }[];
  /** Feed PCM in 100 ms buffers, advancing the fake wall clock with it and
   *  letting the loop process each buffer before the next (as real time does). */
  feed(pcm: Int16Array): Promise<void>;
  wall: { ms: number };
}

const BASE_WALL = Date.UTC(2026, 7, 30, 9, 0, 0);

function harness(opts: { labeler?: SpeakerLabeler; embedder?: Embedder | null } = {}): Harness {
  const embedder = new AmplitudeEmbedder();
  const kept: Harness["kept"] = [];
  const discarded: Harness["discarded"] = [];
  const wall = { ms: BASE_WALL };
  const loop = new JournalLoop({
    vad: new EnergyVad(-45, 0.032),
    embedder: opts.embedder === undefined ? embedder : opts.embedder,
    labeler: opts.labeler ?? labelerWithSelf(),
    keep: (pcm, meta) => kept.push({ pcm, meta }),
    onDiscard: (m) => discarded.push({ reason: m.reason, isSelf: m.isSelf }),
    now: () => wall.ms,
  });
  let fed = 0;
  return {
    loop,
    embedder,
    kept,
    discarded,
    wall,
    async feed(pcm) {
      for (let off = 0; off < pcm.length; off += 1600) {
        const chunk = pcm.subarray(off, Math.min(off + 1600, pcm.length));
        fed += chunk.length;
        wall.ms = BASE_WALL + (fed / SR) * 1000;
        loop.pushSamples(chunk);
        await loop.settle();
      }
    },
  };
}

function rmsOf(pcm: Int16Array, from: number, to: number): number {
  let acc = 0;
  for (let i = from; i < to; i++) acc += pcm[i] * pcm[i];
  return Math.sqrt(acc / Math.max(1, to - from)) / 32768;
}

describe("JournalLoop", () => {
  it("keeps the owner's stretch with ~1 s of context on each side and stamps it on the clock", async () => {
    const h = harness();
    h.loop.start();
    await h.feed(silenceInt16(1.0));
    await h.feed(toneInt16(2.0, SELF_DB));
    await h.feed(silenceInt16(3.0));
    await h.loop.settle();

    expect(h.embedder.calls).toBe(1);
    expect(h.discarded).toEqual([]);
    expect(h.kept).toHaveLength(1);
    const { pcm, meta } = h.kept[0];
    // 1 s lead (all there was) + ~2 s speech + 1 s trail.
    expect(pcm.length / SR).toBeCloseTo(4.0, 0);
    expect(meta.leadSeconds).toBeCloseTo(1.0, 1);
    expect(meta.speechSeconds).toBeCloseTo(2.0, 1);
    expect(meta.trailSeconds).toBeCloseTo(1.0, 1);
    expect(meta.basis).toBe("absolute");
    expect(meta.score).toBeGreaterThanOrEqual(0.65);
    // The speech began ~1 s into the session on the wall clock.
    expect(meta.startWallMs).toBeGreaterThan(BASE_WALL + 800);
    expect(meta.startWallMs).toBeLessThan(BASE_WALL + 1200);
    // The audio is real: silent lead, loud middle, silent trail.
    expect(rmsOf(pcm, 0, Math.round(0.9 * SR))).toBeLessThan(0.001);
    expect(rmsOf(pcm, Math.round(1.5 * SR), Math.round(2.5 * SR))).toBeGreaterThan(0.1);
    expect(rmsOf(pcm, pcm.length - Math.round(0.9 * SR), pcm.length)).toBeLessThan(0.001);

    const stats = h.loop.statsSnapshot;
    expect(stats.selfCount).toBe(1);
    expect(stats.selfSeconds).toBeCloseTo(2.0, 1);
    expect(stats.lastSelfWallMs).not.toBeNull();
    expect(stats.listeningSeconds).toBeCloseTo(6.0, 1);
  });

  it("discards another voice — nothing is ever handed to keep", async () => {
    const h = harness();
    h.loop.start();
    await h.feed(silenceInt16(0.5));
    await h.feed(toneInt16(2.0, OTHER_DB));
    await h.feed(silenceInt16(2.0));
    await h.loop.settle();

    expect(h.embedder.calls).toBe(1);
    expect(h.kept).toEqual([]);
    expect(h.discarded).toEqual([{ reason: "other", isSelf: false }]);
    expect(h.loop.statsSnapshot.selfCount).toBe(0);
    expect(h.loop.statsSnapshot.discardedCount).toBe(1);
  });

  it("never embeds a stretch shorter than 1.0 s", async () => {
    const h = harness();
    h.loop.start();
    await h.feed(silenceInt16(0.5));
    await h.feed(toneInt16(0.7, SELF_DB));
    await h.feed(silenceInt16(2.0));
    await h.loop.settle();
    expect(h.embedder.calls).toBe(0);
    expect(h.kept).toEqual([]);
    expect(h.discarded).toEqual([]);
  });

  it("bounds the trailing context by the next speech onset and the lead by the previous run", async () => {
    const h = harness();
    h.loop.start();
    await h.feed(silenceInt16(1.0));
    await h.feed(toneInt16(2.0, SELF_DB)); // owner 1.0–3.0
    await h.feed(silenceInt16(0.5)); // 3.0–3.5
    await h.feed(toneInt16(2.0, OTHER_DB)); // other 3.5–5.5
    await h.feed(silenceInt16(0.5)); // 5.5–6.0
    await h.feed(toneInt16(1.5, SELF_DB)); // owner 6.0–7.5
    await h.feed(silenceInt16(2.0));
    await h.loop.settle();

    expect(h.embedder.calls).toBe(3);
    expect(h.discarded).toEqual([{ reason: "other", isSelf: false }]);
    expect(h.kept).toHaveLength(2);
    const first = h.kept[0];
    expect(first.meta.trailSeconds).toBeCloseTo(0.5, 1);
    expect(first.pcm.length / SR).toBeCloseTo(3.5, 0);
    const second = h.kept[1];
    // The lead reaches back only to the end of the other voice's run.
    expect(second.meta.leadSeconds).toBeCloseTo(0.5, 1);
    expect(second.meta.speechSeconds).toBeCloseTo(1.5, 1);
    expect(second.meta.trailSeconds).toBeCloseTo(1.0, 1);
    // The other voice's audio was never written: the first chunk's tail is silent.
    expect(rmsOf(first.pcm, first.pcm.length - Math.round(0.4 * SR), first.pcm.length)).toBeLessThan(0.001);
  });

  it("writes the open stretch on stop with whatever trailing audio exists", async () => {
    const h = harness();
    h.loop.start();
    await h.feed(silenceInt16(0.5));
    await h.feed(toneInt16(1.5, SELF_DB));
    const stats = await h.loop.stop();
    expect(h.kept).toHaveLength(1);
    expect(h.kept[0].meta.trailSeconds).toBeCloseTo(0, 1);
    expect(h.kept[0].meta.speechSeconds).toBeCloseTo(1.5, 1);
    expect(stats.selfCount).toBe(1);
    expect(h.loop.isRunning).toBe(false);
  });

  it("keeps nothing without an enrolled owner print (isSelf is never true)", async () => {
    const h = harness({ labeler: new SpeakerLabeler([]) });
    h.loop.start();
    await h.feed(silenceInt16(0.5));
    await h.feed(toneInt16(2.0, SELF_DB));
    await h.feed(silenceInt16(2.0));
    await h.loop.settle();
    expect(h.kept).toEqual([]);
    expect(h.discarded).toEqual([{ reason: "unidentified", isSelf: null }]);
  });

  it("discards everything when there is no embedder at all", async () => {
    const h = harness({ embedder: null });
    h.loop.start();
    await h.feed(silenceInt16(0.5));
    await h.feed(toneInt16(2.0, SELF_DB));
    await h.feed(silenceInt16(2.0));
    await h.loop.settle();
    expect(h.kept).toEqual([]);
    expect(h.discarded).toEqual([{ reason: "no-embedder", isSelf: null }]);
  });

  it("accepts a contrast self match once a second cluster exists", async () => {
    // The owner's print from another room: cosine ~0.5 against it, well
    // under the 0.65 bar; a second (other) voice makes the contrast rule
    // applicable (settings ≥ 2, margin ≥ 0.15).
    const selfPrint = unitVector(DIM, 0);
    const selfLive = new Float32Array(DIM);
    selfLive[0] = 0.5;
    selfLive[2] = Math.sqrt(1 - 0.25);
    const other = unitVector(DIM, 1);
    const embedder: Embedder & { calls: number } = {
      calls: 0,
      async embed(pcm) {
        this.calls += 1;
        let acc = 0;
        for (let i = 0; i < pcm.length; i++) acc += pcm[i] * pcm[i];
        return Math.sqrt(acc / pcm.length) > 0.1 ? Float32Array.from(selfLive) : Float32Array.from(other);
      },
    };
    const labeler = new SpeakerLabeler([
      { personId: "self", displayName: "You", isSelf: true, embedding: selfPrint, settings: 3 },
    ]);
    const h = harness({ labeler, embedder });
    h.loop.start();
    await h.feed(silenceInt16(0.5));
    await h.feed(toneInt16(2.0, SELF_DB)); // alone: absolute path only → not kept
    await h.feed(silenceInt16(1.0));
    await h.feed(toneInt16(2.0, OTHER_DB)); // a second voice → contrast possible
    await h.feed(silenceInt16(1.0));
    await h.feed(toneInt16(2.0, SELF_DB)); // now the owner by contrast
    await h.feed(silenceInt16(2.0));
    await h.loop.settle();
    // The FIRST lone segment is kept by the journal's solo rule (selfScore
    // 0.5 >= 0.4, print pools >= 2 recordings, no second cluster yet); once a
    // second cluster exists the contrast rule takes over.
    expect(h.kept).toHaveLength(2);
    expect(h.kept[0].meta.basis).toBe("solo");
    expect(h.kept[1].meta.basis).toBe("contrast");
    // One "other" voice remains discarded; the segment that used to be the
    // second discard is now the solo keep above.
    expect(h.discarded.map((d) => d.reason)).toEqual(["other"]);
  });
});


describe("JournalLoop — solo self rule", () => {
  function fakeLabeler(opts: { selfScore: number; clusterCount?: number; selfSettings?: number }) {
    return {
      label: () => ({
        speaker: "Speaker A", personId: null, displayName: null,
        isSelf: false as const, score: 0.1, basis: null, selfScore: opts.selfScore,
      }),
      reset: () => {},
      clusterCount: opts.clusterCount ?? 1,
      selfSettings: opts.selfSettings ?? 2,
      hasSelfPrint: true,
    } as unknown as SpeakerLabeler;
  }

  async function run(labeler: SpeakerLabeler) {
    const h = harness({ labeler });
    h.loop.start();
    await h.feed(silenceInt16(0.5));
    await h.feed(toneInt16(2.0, SELF_DB));
    await h.feed(silenceInt16(2.0));
    await h.loop.settle();
    return h;
  }

  it("keeps a lone voice at selfScore >= 0.40 with a >= 2-recording print, basis solo", async () => {
    const h = await run(fakeLabeler({ selfScore: 0.45 }));
    expect(h.kept).toHaveLength(1);
    expect(h.kept[0].meta.basis).toBe("solo");
    expect(h.kept[0].meta.score).toBeCloseTo(0.45, 5);
  });

  it("discards a lone voice below the solo threshold", async () => {
    const h = await run(fakeLabeler({ selfScore: 0.35 }));
    expect(h.kept).toHaveLength(0);
    expect(h.discarded.map((d) => d.reason)).toEqual(["other"]);
  });

  it("no solo keep with a single-recording print", async () => {
    const h = await run(fakeLabeler({ selfScore: 0.55, selfSettings: 1 }));
    expect(h.kept).toHaveLength(0);
  });

  it("no solo keep once a second cluster exists (contrast owns it)", async () => {
    const h = await run(fakeLabeler({ selfScore: 0.45, clusterCount: 2 }));
    expect(h.kept).toHaveLength(0);
    expect(h.discarded.map((d) => d.reason)).toEqual(["other"]);
  });
});
