/**
 * FastLoopDeps.floorTick — the mid-turn floor cue wired into the real loop:
 * synthetic PCM through the energy VAD, a fake embedder whose "voice" is the
 * window's loudness (wearer at -20 dBFS, the other person at -32 dBFS), and
 * a real SpeakerLabeler holding the wearer's print.
 */
import { FastLoop } from "../src/live/fastLoop";
import { FLOOR_CUE_TEXT, type FloorFire } from "../src/live/floorTick";
import { EnergyVad } from "../src/live/vad";
import { cloudProvider, ProviderChain } from "../src/live/localLlm";
import { SpeakerLabeler, type Embedder } from "../src/live/speakerId";
import type { LiveMode } from "../src/live/localLlm";
import type { LocalTurn } from "../src/live/fastLoop";
import { silenceInt16, toneInt16, unitVector } from "../src/live/testing/synth";

const D = 16;
const WEARER_DB = -20;
const OTHER_DB = -32;

/** "Voice" = loudness: louder than -26 dBFS is the wearer's print. */
const loudnessEmbedder: Embedder = {
  embed: async (pcm) => {
    let sq = 0;
    for (let i = 0; i < pcm.length; i++) sq += pcm[i] * pcm[i];
    const db = 10 * Math.log10(sq / Math.max(1, pcm.length) + 1e-12);
    return db > -26 ? unitVector(D, 0) : unitVector(D, 1);
  },
};

interface H {
  loop: FastLoop;
  cues: { fire: FloorFire; turnsSoFar: number }[];
  spoken: string[];
  haptics: { level: number; code: string | null }[];
  turns: LocalTurn[];
}

function harness(opts: { floorTick?: boolean; embedder?: Embedder | null; enrolled?: boolean } = {}): H {
  const h: H = { loop: undefined as unknown as FastLoop, cues: [], spoken: [], haptics: [], turns: [] };
  const you = { personId: "p-you", displayName: "You", isSelf: true, embedding: unitVector(D, 0) };
  const embedder = opts.embedder === undefined ? loudnessEmbedder : opts.embedder;
  h.loop = new FastLoop({
    vad: new EnergyVad(-45, 0.032),
    embedder,
    labeler: embedder ? new SpeakerLabeler(opts.enrolled === false ? [] : [you]) : null,
    recognizer: null,
    llm: new ProviderChain([cloudProvider()]),
    instantHeat: false,
    speak: (t) => h.spoken.push(t),
    send: () => {},
    onTurn: (t) => h.turns.push(t),
    haptics: { nudge: async (level, code) => { h.haptics.push({ level, code: code ?? null }); } },
    onFloorCue: (fire) => h.cues.push({ fire, turnsSoFar: h.turns.length }),
    ...(opts.floorTick === undefined ? {} : { floorTick: opts.floorTick }),
    sttGraceMs: 50,
    pollMs: 5,
  });
  return h;
}

function push(loop: FastLoop, pcm: Int16Array) {
  for (let off = 0; off < pcm.length; off += 1600) loop.pushSamples(pcm.subarray(off, Math.min(off + 1600, pcm.length)));
}

async function play(h: H, parts: Int16Array[], mode: LiveMode = "earpiece") {
  await h.loop.start({ sessionId: "floor", mode, empathy: 50 });
  // One second at a time, settling in between, like a real-time mic: the
  // loop keeps only `historySeconds` of PCM, and a window is sliced when its
  // frame is processed, not when it was pushed.
  for (const p of parts) {
    for (let off = 0; off < p.length; off += 16000) {
      push(h.loop, p.subarray(off, Math.min(off + 16000, p.length)));
      await h.loop.settle();
    }
  }
  await h.loop.stop();
}

jest.setTimeout(60_000);

describe("FastLoop floorTick", () => {
  it("is OFF by default: a 60 s wearer monologue gets no floor cue", async () => {
    const h = harness();
    await play(h, [toneInt16(60, WEARER_DB), silenceInt16(1)]);
    expect(h.cues).toEqual([]);
    expect(h.spoken).not.toContain(FLOOR_CUE_TEXT);
    expect(h.haptics.filter((x) => x.code === "A")).toEqual([]);
  });

  it("a 60 s monologue cues at ~45 s, mid-turn — before the turn has even closed", async () => {
    const h = harness({ floorTick: true });
    await play(h, [toneInt16(50, WEARER_DB), silenceInt16(0.6), toneInt16(10, WEARER_DB), silenceInt16(1)]);
    expect(h.cues).toHaveLength(1);
    const { fire, turnsSoFar } = h.cues[0];
    expect(fire.t).toBeGreaterThanOrEqual(45);
    expect(fire.t).toBeLessThanOrEqual(46.5);
    expect(fire.text).toBe(FLOOR_CUE_TEXT);
    // The 50 s span closes at ~50.3 s: the cue did not wait for it.
    expect(turnsSoFar).toBe(0);
    expect(fire.t).toBeLessThan(h.turns[0].endTime);
    // Delivery: the hogging buzz at once, the line in the next pause.
    expect(h.haptics).toContainEqual({ level: 1, code: "A" });
    expect(h.spoken).toEqual([FLOOR_CUE_TEXT]);
  });

  it("a 30 s wearer turn never cues", async () => {
    const h = harness({ floorTick: true });
    await play(h, [toneInt16(30, WEARER_DB), silenceInt16(10)]);
    expect(h.cues).toEqual([]);
  });

  it("another voice taking the floor mid-span resets the clock", async () => {
    // One unbroken VAD span (no pause): wearer 30 s, other 5 s, wearer 30 s.
    // Whole-turn identity would only ever see one turn; the tick sees the
    // switch, so the second wearer stretch starts a fresh 45 s clock.
    const h = harness({ floorTick: true });
    await play(h, [toneInt16(30, WEARER_DB), toneInt16(5, OTHER_DB), toneInt16(30, WEARER_DB), silenceInt16(1)]);
    expect(h.cues).toEqual([]);
  });

  it("with no voice model (unknown wearer) it never cues", async () => {
    const h = harness({ floorTick: true, embedder: null });
    await play(h, [toneInt16(60, WEARER_DB), silenceInt16(1)]);
    expect(h.cues).toEqual([]);
  });

  it("with no wearer print enrolled it never cues", async () => {
    const h = harness({ floorTick: true, enrolled: false });
    await play(h, [toneInt16(60, WEARER_DB), silenceInt16(1)]);
    expect(h.cues).toEqual([]);
  });

  it("one long monologue gets ONE cue (once per run)", async () => {
    const h = harness({ floorTick: true });
    await play(h, [toneInt16(90, WEARER_DB), silenceInt16(1)]);
    expect(h.cues.map((c) => Math.round(c.fire.t))).toEqual([45]);
  });

  it("therapist mode buzzes but never speaks", async () => {
    const h = harness({ floorTick: true });
    await play(h, [toneInt16(50, WEARER_DB), silenceInt16(1)], "therapist");
    expect(h.cues).toHaveLength(1);
    expect(h.haptics).toContainEqual({ level: 1, code: "A" });
    expect(h.spoken).toEqual([]);
  });
});
