/**
 * FastLoop's mid-call naming hooks (cluster → person binding): a bound raw
 * label carries its person from the next turn on while the WIRE label stays
 * raw; the person's pooled session audio is kept for enrollment; a print
 * learned from it makes later voiceprint matches land on the same raw
 * label; naming someone as "me" makes every other voice honestly not-me;
 * and the prompt says "Mom", not "Speaker B".
 */
import { FastLoop, type LocalTurn } from "../src/live/fastLoop";
import { EnergyVad } from "../src/live/vad";
import { FakeSpeechRecognizer } from "../src/live/stt";
import { ProviderChain, parseSuggestionJson, type SuggestInput } from "../src/live/localLlm";
import { SpeakerLabeler, type Embedder, type EnrolledPerson } from "../src/live/speakerId";
import type { TurnLocalEvent } from "../src/live/types";
import { silenceInt16, toneInt16, unitVector } from "../src/live/testing/synth";

const DIM = 8;
/** An embedder that answers from a script of vectors, in call order. */
class ScriptedEmbedder implements Embedder {
  calls: Float32Array[] = [];
  constructor(private readonly queue: Float32Array[]) {}
  async embed(pcm: Float32Array): Promise<Float32Array> {
    this.calls.push(pcm);
    const next = this.queue.shift();
    if (!next) throw new Error("no scripted embedding left");
    return next;
  }
}

const GOOD = '{"suggestion":"Say: I hear you.","tone":{"warmth":50,"frustration":20,"label":"calm"}}';

function build(embedder: Embedder, people: EnrolledPerson[] = []) {
  const rec = new FakeSpeechRecognizer();
  const turns: LocalTurn[] = [];
  const sent: TurnLocalEvent[] = [];
  const prompts: SuggestInput[] = [];
  const labeler = new SpeakerLabeler(people);
  const loop = new FastLoop({
    vad: new EnergyVad(-45, 0.032),
    embedder,
    labeler,
    recognizer: rec,
    llm: new ProviderChain([
      {
        name: "os",
        isAvailable: async () => true,
        suggest: async (input) => {
          prompts.push(input);
          return parseSuggestionJson(GOOD);
        },
      },
    ]),
    // This suite reuses one constant coach line to assert KINDS; the
    // repeat-gate (covered in liveFastLoop) would silence the repeats.
    repeatGate: null,
    speak: () => {},
    send: (e) => sent.push(e),
    onTurn: (t) => turns.push(t),
    sttGraceMs: 50,
    pollMs: 5,
    selfSpeakerFallback: null,
  });
  return { loop, rec, turns, sent, prompts, labeler };
}

/** One spoken turn of `seconds` with the recognizer's words, then a pause. */
async function speak(h: ReturnType<typeof build>, seconds: number, text: string) {
  h.loop.pushSamples(toneInt16(seconds, -20));
  h.rec.emit({ text, isFinal: true });
  h.loop.pushSamples(silenceInt16(0.5));
  await h.loop.settle();
}

describe("FastLoop mid-call naming", () => {
  it("binds a raw cluster to a person: later turns carry the person, the wire label stays raw, the prompt uses the name", async () => {
    const a = unitVector(DIM, 0);
    const b = unitVector(DIM, 1);
    const h = build(new ScriptedEmbedder([a, b, b, a]));
    await h.loop.start({ sessionId: "s", mode: "speaker", empathy: 50 });
    await speak(h, 2.0, "hi mom");
    await speak(h, 2.0, "you never call");
    expect(h.turns.map((t) => t.speaker)).toEqual(["Speaker A", "Speaker B"]);
    expect(h.turns[1].personId).toBeNull();
    expect(h.prompts[1].speaker).toBe("Speaker B");

    h.loop.bindSpeaker("Speaker B", { personId: "mom", displayName: "Mom", isSelf: false });
    // Past turns are re-attributed for the session record …
    expect(h.turns[1]).toMatchObject({ speaker: "Speaker B", personId: "mom", displayName: "Mom", isSelf: false });
    expect(h.loop.bindingOf("Speaker B")).toEqual({ personId: "mom", displayName: "Mom", isSelf: false });
    expect(h.loop.displayNameOf("Speaker B")).toBe("Mom");
    expect(h.loop.displayNameOf("Speaker A")).toBe("Speaker A");

    // … and the next turn on that cluster carries the person on the wire.
    await speak(h, 2.0, "that's not fair");
    expect(h.turns[2]).toMatchObject({ speaker: "Speaker B", personId: "mom", displayName: "Mom", isSelf: false });
    expect(h.sent[2]).toMatchObject({ speaker: "Speaker B", speaker_person_id: "mom", is_self: false });
    expect(h.prompts[2].speaker).toBe("Mom");
    // Earlier context lines are renamed too.
    expect(h.prompts[2].context.map((c) => c.speaker)).toEqual(["Speaker A", "Mom"]);

    // An unbound cluster is untouched (no self bound → is_self stays null).
    await speak(h, 2.0, "i know");
    expect(h.turns[3]).toMatchObject({ speaker: "Speaker A", personId: null, isSelf: null });
    await h.loop.stop();
  });

  it("pools each speaker's audio (bounded) and embeds it for enrollment", async () => {
    const a = unitVector(DIM, 0);
    const b = unitVector(DIM, 1);
    const pooled = unitVector(DIM, 2);
    const emb = new ScriptedEmbedder([a, b, b, pooled]);
    const h = build(emb);
    await h.loop.start({ sessionId: "s", mode: "speaker", empathy: 50 });
    await speak(h, 2.0, "one");
    await speak(h, 1.8, "two");
    await speak(h, 1.6, "three");
    expect(h.loop.speakerAudioSeconds("Speaker A")).toBeCloseTo(2.0, 0);
    expect(h.loop.speakerAudioSeconds("Speaker B")).toBeCloseTo(3.4, 0);
    expect(h.loop.speakerAudio("Speaker B").length).toBe(Math.round(h.loop.speakerAudioSeconds("Speaker B") * 16000));
    expect(h.loop.speakerAudio("Nobody").length).toBe(0);
    const print = await h.loop.embedSpeaker("Speaker B");
    expect(print).toBe(pooled);
    expect(emb.calls[3].length).toBe(h.loop.speakerAudio("Speaker B").length);
    expect(await h.loop.embedSpeaker("Nobody")).toBeNull();
    await h.loop.stop();
  });

  it("caps the per-speaker pool at speakerAudioSeconds, keeping the most recent audio", async () => {
    const h = build(new ScriptedEmbedder([unitVector(DIM, 0), unitVector(DIM, 0), unitVector(DIM, 0)]));
    // Rebuild with a 3 s cap through the deps seam.
    const capped = new FastLoop({
      vad: new EnergyVad(-45, 0.032),
      embedder: null,
      labeler: null,
      recognizer: h.rec,
      llm: new ProviderChain([]),
      speak: () => {},
      send: () => {},
      onTurn: () => {},
      sttGraceMs: 50,
      pollMs: 5,
      speakerAudioSeconds: 3,
    });
    await capped.start({ sessionId: "s", mode: "speaker", empathy: 50 });
    for (let i = 0; i < 3; i++) {
      capped.pushSamples(toneInt16(2.0, -20));
      capped.pushSamples(silenceInt16(0.5));
      await capped.settle();
    }
    expect(capped.speakerAudioSeconds("Unknown")).toBeLessThanOrEqual(3.0);
    expect(capped.speakerAudioSeconds("Unknown")).toBeGreaterThan(1.5);
    await capped.stop();
  });

  it("a print learned mid-call makes later VOICE matches land on the same raw label", async () => {
    const a = unitVector(DIM, 0);
    const b = unitVector(DIM, 1);
    const bAgain = unitVector(DIM, 1, 0.05, 7);
    const h = build(new ScriptedEmbedder([a, b, bAgain]));
    await h.loop.start({ sessionId: "s", mode: "speaker", empathy: 50 });
    await speak(h, 2.0, "one");
    await speak(h, 2.0, "two");
    h.loop.bindSpeaker(
      "Speaker B",
      { personId: "mom", displayName: "Mom", isSelf: false },
      { personId: "mom", displayName: "Mom", isSelf: false, embedding: b },
    );
    expect(h.labeler.enrolledCount).toBe(1);
    // The labeler now matches Mom by voiceprint (speaker would be "Mom") —
    // the loop maps it back to the raw label the session has been using.
    await speak(h, 2.0, "three");
    expect(h.turns[2]).toMatchObject({ speaker: "Speaker B", personId: "mom", displayName: "Mom" });
    expect(h.turns[2].matchScore).toBeGreaterThan(0.9);
    await h.loop.stop();
  });

  it("naming a voice as 'me' makes the other voices honestly not-me and re-binding a person frees the old label", async () => {
    const a = unitVector(DIM, 0);
    const b = unitVector(DIM, 1);
    const h = build(new ScriptedEmbedder([a, b, a, b]));
    await h.loop.start({ sessionId: "s", mode: "speaker", empathy: 50 });
    await speak(h, 2.0, "one");
    await speak(h, 2.0, "two");
    h.loop.bindSpeaker("Speaker B", { personId: "self", displayName: "You", isSelf: true });
    await speak(h, 2.0, "three"); // Speaker A again
    expect(h.turns[2]).toMatchObject({ speaker: "Speaker A", isSelf: false, personId: null });
    expect(h.sent[2].is_self).toBe(false);
    // Oops — it was actually Speaker A who is me.
    h.loop.bindSpeaker("Speaker A", { personId: "self", displayName: "You", isSelf: true });
    expect(h.loop.bindingOf("Speaker B")).toBeNull();
    expect(h.loop.bindingOf("Speaker A")?.isSelf).toBe(true);
    await speak(h, 2.0, "four"); // Speaker B
    expect(h.turns[3]).toMatchObject({ speaker: "Speaker B", isSelf: false, personId: null });
    expect(h.turns[3].suggestionKind).toBe("response");
    await h.loop.stop();
  });

  it("bindings and pools reset at the next start", async () => {
    const h = build(new ScriptedEmbedder([unitVector(DIM, 0), unitVector(DIM, 0)]));
    await h.loop.start({ sessionId: "s1", mode: "speaker", empathy: 50 });
    await speak(h, 2.0, "one");
    h.loop.bindSpeaker("Speaker A", { personId: "mom", displayName: "Mom", isSelf: false });
    await h.loop.stop();
    await h.loop.start({ sessionId: "s2", mode: "speaker", empathy: 50 });
    expect(h.loop.bindingOf("Speaker A")).toBeNull();
    expect(h.loop.speakerAudioSeconds("Speaker A")).toBe(0);
    await h.loop.stop();
  });
});
