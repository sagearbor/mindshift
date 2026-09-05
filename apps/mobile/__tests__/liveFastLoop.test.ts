/**
 * src/live/fastLoop.ts — the whole on-device loop end to end with synthetic
 * PCM through the energy VAD (32 ms frames), a fake recognizer, fake LLM
 * providers, fake embedder, and spies on speak / send / haptics.
 */
import { FastLoop, MAX_EMBED_SECONDS, formatLatencyLog, prosodyHint, type LocalTurn } from "../src/live/fastLoop";
import { EnergyVad } from "../src/live/vad";
import { FakeSpeechRecognizer } from "../src/live/stt";
import { cloudProvider, ProviderChain, parseSuggestionJson, type SuggestionProvider } from "../src/live/localLlm";
import { SpeakerLabeler, type Embedder } from "../src/live/speakerId";
import { NudgePolicy, phoneNudgePolicy, type NudgeEvent, type VectorEvent } from "../src/live/nudgePolicy";
import { l2Normalize } from "../src/live/speakerId";
import type { TurnLocalEvent } from "../src/live/types";
import { silenceInt16, toneInt16, unitVector, vectorAtCosine } from "../src/live/testing/synth";

const GOOD = '{"suggestion":"Tell her you miss the calls too.","tone":{"warmth":30,"frustration":20,"label":"hurt"}}';
const LOUD = '{"suggestion":"ease up","tone":{"warmth":5,"frustration":95,"defensiveness":80,"label":"angry"}}';

function okProvider(json = GOOD, gate?: () => Promise<void>): SuggestionProvider {
  return {
    name: "os",
    isAvailable: async () => true,
    suggest: async () => {
      if (gate) await gate();
      return parseSuggestionJson(json);
    },
  };
}

interface Harness {
  loop: FastLoop;
  rec: FakeSpeechRecognizer;
  spoken: string[];
  sent: TurnLocalEvent[];
  turns: LocalTurn[];
  nudges: NudgeEvent[];
  haptic: number[];
}

function harness(opts: {
  provider?: SuggestionProvider;
  recognizer?: FakeSpeechRecognizer | null;
  embedder?: Embedder | null;
  labeler?: SpeakerLabeler | null;
} = {}): Harness {
  const rec = opts.recognizer === undefined ? new FakeSpeechRecognizer() : opts.recognizer;
  const h: Harness = {
    loop: undefined as unknown as FastLoop,
    rec: rec as FakeSpeechRecognizer,
    spoken: [],
    sent: [],
    turns: [],
    nudges: [],
    haptic: [],
  };
  h.loop = new FastLoop({
    vad: new EnergyVad(-45, 0.032),
    embedder: opts.embedder ?? null,
    labeler: opts.labeler ?? null,
    recognizer: rec,
    llm: new ProviderChain([opts.provider ?? okProvider(), cloudProvider()]),
    speak: (t) => h.spoken.push(t),
    send: (e) => h.sent.push(e),
    onTurn: (t) => h.turns.push(t),
    onNudge: (n) => h.nudges.push(n),
    haptics: { nudge: async (level) => { h.haptic.push(level); } },
    sttGraceMs: 150,
    pollMs: 5,
  });
  return h;
}

/** Push int16 audio in ~100 ms chunks, like expo-audio delivers it. */
function push(loop: FastLoop, pcm: Int16Array) {
  for (let off = 0; off < pcm.length; off += 1600) {
    loop.pushSamples(pcm.subarray(off, Math.min(off + 1600, pcm.length)));
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** okProvider with a per-call counter in the text: the CoachRepeatGate
 *  (identical lines within 45 s become silence) never fires, so tests
 *  about coaching KINDS keep seeing every kind. */
function variedProvider(base = okProvider()) {
  let n = 0;
  return {
    ...base,
    suggest: async (req: Parameters<typeof base.suggest>[0]) => {
      const r = await base.suggest(req);
      n++;
      // Fully distinct lines (no shared word bigrams): a mere suffix keeps
      // Jaccard >= 0.5 and the gate still fires.
      const LINES = [
        "Try asking how that felt.",
        "Give her a moment to answer.",
        "Name one thing you appreciated.",
        "Ask what she needs right now.",
        "Slow down and breathe together.",
      ];
      return r ? { ...r, suggestion: LINES[(n - 1) % LINES.length] } : r;
    },
  };
}

describe("FastLoop", () => {
  it("finalizes a turn, aligns STT text, coaches, speaks, sends turn_local, logs latency", async () => {
    const h = harness();
    await h.loop.start({ sessionId: "s1", mode: "earpiece", empathy: 60 });
    expect(h.rec.started).toBe(true);

    push(h.loop, toneInt16(1.0, -20));
    h.rec.emit({
      text: "you never call",
      isFinal: true,
      segments: [
        { startTimeMillis: 100, endTimeMillis: 400, segment: "you" },
        { startTimeMillis: 400, endTimeMillis: 700, segment: "never" },
        { startTimeMillis: 700, endTimeMillis: 950, segment: "call" },
      ],
    });
    push(h.loop, silenceInt16(0.5));
    await h.loop.settle();

    expect(h.turns).toHaveLength(1);
    const t = h.turns[0];
    expect(t.text).toBe("you never call");
    expect(t.transcriptFinal).toBe(true);
    expect(t.startTime).toBeCloseTo(0, 1);
    expect(t.endTime).toBeCloseTo(1.0, 1);
    expect(t.speaker).toBe("Unknown"); // no embedder => no identity
    expect(t.isSelf).toBeNull();
    expect(t.suggestion).toBe("Tell her you miss the calls too.");
    expect(t.suggestionKind).toBe("response");
    expect(t.provider).toBe("os");
    expect(t.prosody.rms_dbfs).toBeCloseTo(-20, 0);
    expect(t.prosody.speech_rate as number).toBeCloseTo(3, 0); // 3 words over a ~1.0 s (frame-aligned) span
    expect(t.spoken).toBe(true);
    expect(h.spoken).toEqual(["Tell her you miss the calls too."]);

    expect(h.sent).toHaveLength(1);
    expect(h.sent[0]).toMatchObject({
      type: "turn_local",
      session_id: "s1",
      speaker: "Unknown",
      speaker_person_id: null,
      is_self: null,
      text: "you never call",
      transcript_source: "on-device",
      suggestion: "Tell her you miss the calls too.",
      suggestion_source: "on-device",
      tts_source: "on-device",
      text_tone: expect.objectContaining({ frustration: 20, label: "hurt" }),
    });

    const l = h.loop.latencyLog[0];
    expect(l.toSpeakMs).not.toBeNull();
    expect(l.provider).toBe("os");
    expect(l.held).toBe(false);
    expect(formatLatencyLog(h.loop.latencyLog)).toMatch(/1 turns, median segment-end→speak \d+ ms/);
    expect(formatLatencyLog([])).toMatch(/no turns/);

    const summary = await h.loop.stop();
    expect(summary.turns).toHaveLength(1);
    expect(summary.sttAvailable).toBe(true);
    expect(h.rec.stopped).toBe(true);
  });

  it("holds a suggestion while speech is active and speaks it once the VAD goes quiet", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => { release = r; });
    const h = harness({ provider: okProvider(GOOD, () => gate) });
    await h.loop.start({ sessionId: "s2", mode: "speaker", empathy: 50 });

    push(h.loop, toneInt16(1.0, -20));
    h.rec.emit({ text: "hello there dear", isFinal: true });
    push(h.loop, silenceInt16(0.5)); // turn 1 finalizes, LLM now blocked on `gate`
    await sleep(20);
    push(h.loop, toneInt16(0.6, -20)); // the other person starts talking again
    await sleep(20);
    release(); // suggestion for turn 1 arrives mid-speech
    await sleep(20);
    expect(h.spoken).toEqual([]); // held: never talk over live speech
    push(h.loop, silenceInt16(0.5)); // VAD goes quiet
    await h.loop.settle();
    expect(h.spoken[0]).toBe("Tell her you miss the calls too.");
    expect(h.loop.latencyLog[0].held).toBe(true);
    expect(h.loop.latencyLog[0].toSpeakMs).not.toBeNull();
    await h.loop.stop();
  });

  it("therapist mode never speaks but still reports turns and sends turn_local", async () => {
    const h = harness();
    await h.loop.start({ sessionId: "s3", mode: "therapist", empathy: 50 });
    push(h.loop, toneInt16(1.0, -20));
    h.rec.emit({ text: "I feel unheard", isFinal: true });
    push(h.loop, silenceInt16(0.5));
    await h.loop.settle();
    expect(h.turns[0].suggestion).toBe("Tell her you miss the calls too.");
    expect(h.turns[0].spoken).toBe(false);
    expect(h.spoken).toEqual([]);
    expect(h.sent).toHaveLength(1);
    expect(h.loop.latencyLog[0].toSpeakMs).toBeNull();
    await h.loop.stop();
  });

  it("identifies the enrolled self, coaches with a nudge, and fires haptics when the policy escalates", async () => {
    const D = 192;
    const you = { personId: "p-you", displayName: "You", isSelf: true, embedding: unitVector(D, 0) };
    const mom = { personId: "p-mom", displayName: "Mom", isSelf: false, embedding: unitVector(D, 1) };
    // Embedder answers from a queue: self, self, self (loud).
    const queue = [unitVector(D, 0, 0.2, 3), unitVector(D, 0, 0.2, 4), unitVector(D, 0, 0.2, 5)];
    const embedder: Embedder = { embed: async () => queue.shift() ?? unitVector(D, 9) };
    const h = harness({ provider: okProvider(LOUD), embedder, labeler: new SpeakerLabeler([you, mom]) });
    await h.loop.start({ sessionId: "s4", mode: "earpiece", empathy: 50 });

    for (const dbfs of [-30, -30, -10]) {
      push(h.loop, toneInt16(1.0, dbfs));
      h.rec.emit({ text: "stop doing that", isFinal: true });
      push(h.loop, silenceInt16(0.5));
      await h.loop.settle();
    }
    expect(h.turns.map((t) => t.speaker)).toEqual(["You", "You", "You"]);
    expect(h.turns.map((t) => t.isSelf)).toEqual([true, true, true]);
    // Identical coach lines within the 45 s repeat cooldown become silence
    // (CoachRepeatGate): only the first "ease up" is delivered.
    expect(h.turns.map((t) => t.suggestionKind)).toEqual(["nudge", null, null]);
    expect(h.sent[0]).toMatchObject({ speaker: "You", speaker_person_id: "p-you", is_self: true });
    expect(h.sent[0].speaker_match_score as number).toBeGreaterThan(0.65);
    // aggressive_tone level 3 already on turn 1 (frustration 95) => haptic 3 once;
    // sustained level 3 afterwards is silent (policy hysteresis).
    expect(h.haptic).toEqual([3]);
    expect(h.nudges).toHaveLength(1);
    expect(h.nudges[0]).toMatchObject({ channel: "A", level: 3 });
    // Spoken once; the repeats were gated (the exact owner bug from
    // 2026-08-26: the same line on consecutive fragments).
    expect(h.spoken).toEqual(["ease up"]);
    await h.loop.stop();
  });

  it("without a voiceprint verdict, the 'Speaker A is you' convention decides the coaching kind (never is_self on the wire)", async () => {
    const D = 192;
    const queue = [unitVector(D, 0), unitVector(D, 5)];
    const embedder: Embedder = { embed: async () => queue.shift() ?? unitVector(D, 9) };
    const h = harness({ provider: variedProvider(), embedder, labeler: new SpeakerLabeler([]) });
    await h.loop.start({ sessionId: "s5", mode: "earpiece", empathy: 50 });
    for (let i = 0; i < 2; i++) {
      // >= MIN_CLUSTER_SECONDS: long enough to found an unknown cluster.
      push(h.loop, toneInt16(2.0, -20));
      h.rec.emit({ text: "some words here", isFinal: true });
      push(h.loop, silenceInt16(0.5));
      await h.loop.settle();
    }
    expect(h.turns.map((t) => t.speaker)).toEqual(["Speaker A", "Speaker B"]);
    expect(h.turns.map((t) => t.suggestionKind)).toEqual(["nudge", "response"]);
    expect(h.sent.map((e) => e.is_self)).toEqual([null, null]);
    await h.loop.stop();
  });

  it("a cooldown decay updates the screen (onNudge) but never buzzes, whoever was speaking", async () => {
    const D = 192;
    const you = { personId: "p-you", displayName: "You", isSelf: true, embedding: unitVector(D, 0) };
    const queue: Float32Array[] = [unitVector(D, 0), unitVector(D, 5), unitVector(D, 0)];
    const embedder: Embedder = { embed: async () => queue.shift() ?? unitVector(D, 5) };
    const h = harness({ provider: okProvider(LOUD), embedder, labeler: new SpeakerLabeler([you]) });
    // Cooldown 20 s: after a level-2 tone nudge, a turn > 20 s later decays it.
    (h.loop as unknown as { policy: unknown }).policy = phoneNudgePolicy(20);
    await h.loop.start({ sessionId: "s5d", mode: "earpiece", empathy: 50 });
    push(h.loop, toneInt16(2.0, -20)); // self, aggressive tone -> level 3 event
    h.rec.emit({ text: "shouting words here", isFinal: true });
    push(h.loop, silenceInt16(0.5));
    await h.loop.settle();
    expect(h.haptic).toEqual([3]);
    push(h.loop, silenceInt16(21)); // cooldown elapses in audio time
    push(h.loop, toneInt16(2.0, -20)); // a stranger's turn: the policy decays 3 -> 2
    h.rec.emit({ text: "other person talking", isFinal: true });
    push(h.loop, silenceInt16(0.5));
    await h.loop.settle();
    expect(h.nudges.map((n) => [n.level, n.vectors.length])).toEqual([
      [3, 1],
      [2, 0],
    ]);
    expect(h.haptic).toEqual([3]); // the decay did not buzz
    await h.loop.stop();
  });

  it("instant tier: the loudness nudge fires while STT/LLM are still pending (never waits ~7 s)", async () => {
    const D = 192;
    const you = { personId: "p-you", displayName: "You", isSelf: true, embedding: unitVector(D, 0) };
    const embedder: Embedder = { embed: async () => unitVector(D, 0, 0.2, 3) };
    let releaseLlm: () => void = () => {};
    let llmCalls = 0;
    const base = okProvider();
    const slow = {
      ...base,
      suggest: async (req: Parameters<typeof base.suggest>[0]) => {
        llmCalls++;
        if (llmCalls >= 3) await new Promise<void>((r) => { releaseLlm = r; });
        return base.suggest(req);
      },
    };
    const h = harness({ provider: slow, embedder, labeler: new SpeakerLabeler([you]) });
    await h.loop.start({ sessionId: "s-instant", mode: "earpiece", empathy: 50 });
    // Two calm turns build the personal loudness baseline (median -30 dBFS).
    for (const dbfs of [-30, -30]) {
      push(h.loop, toneInt16(1.0, dbfs));
      h.rec.emit({ text: "calm words here", isFinal: true });
      push(h.loop, silenceInt16(0.5));
      await h.loop.settle();
    }
    expect(h.nudges).toHaveLength(0);
    // The loud turn: +20 dB over baseline. Its text arrives, the LLM HANGS —
    // the yelling nudge must fire anyway, before the turn finalizes.
    push(h.loop, toneInt16(1.0, -10));
    h.rec.emit({ text: "stop yelling at me", isFinal: true });
    push(h.loop, silenceInt16(0.5));
    await sleep(80);
    // The HAPTIC (the buzz the user feels) fires now, while STT/LLM are still
    // pending — no ~7 s wait. The on-screen nudge lands with the combined
    // policy tick when the turn finalizes.
    expect(h.haptic).toEqual([3]);
    expect(h.turns).toHaveLength(2); // the full pipeline is still on the LLM
    releaseLlm();
    await h.loop.settle();
    expect(h.turns).toHaveLength(3);
    expect(h.nudges.some((n) => n.level === 3)).toBe(true); // screen nudge arrived
    await h.loop.stop();
  });

  it("a backchannel ('yeah') is recorded but never coached — the LLM is not even asked", async () => {
    let llmCalls = 0;
    const base = okProvider();
    const counting = { ...base, suggest: async (req: Parameters<typeof base.suggest>[0]) => { llmCalls++; return base.suggest(req); } };
    const h = harness({ provider: counting });
    await h.loop.start({ sessionId: "s-bc", mode: "earpiece", empathy: 50 });
    push(h.loop, toneInt16(0.6, -20));
    h.rec.emit({ text: "yeah", isFinal: true });
    push(h.loop, silenceInt16(0.5));
    await h.loop.settle();
    expect(h.turns).toHaveLength(1);
    expect(h.turns[0].kind).toBe("backchannel");
    expect(h.turns[0].suggestion).toBeNull();
    expect(llmCalls).toBe(0);
    expect(h.spoken).toEqual([]);
    // A real sentence right after IS coached.
    push(h.loop, toneInt16(1.0, -20));
    h.rec.emit({ text: "you never call me back", isFinal: true });
    push(h.loop, silenceInt16(0.5));
    await h.loop.settle();
    expect(h.turns[1].kind).toBe("primary");
    expect(llmCalls).toBe(1);
    await h.loop.stop();
  });

  it("CoachRepeatGate: the same line twice within the cooldown is delivered once", async () => {
    const h = harness(); // constant-text provider on purpose
    await h.loop.start({ sessionId: "s-rg", mode: "earpiece", empathy: 50 });
    for (let i = 0; i < 2; i++) {
      push(h.loop, toneInt16(1.0, -20));
      h.rec.emit({ text: "you never call", isFinal: true });
      push(h.loop, silenceInt16(0.5));
      await h.loop.settle();
    }
    expect(h.turns.map((t) => t.suggestion)).toEqual(["Tell her you miss the calls too.", null]);
    expect(h.spoken).toEqual(["Tell her you miss the calls too."]);
    await h.loop.stop();
  });

  describe("vocal activation (live/activation.ts)", () => {
    const D = 192;
    const you = { personId: "p-you", displayName: "You", isSelf: true, embedding: unitVector(D, 0) };
    /** A policy that records the vector events it is fed. */
    class SpyPolicy extends NudgePolicy {
      seen: VectorEvent[][] = [];
      override onEvents(events: VectorEvent[], t: number): NudgeEvent[] {
        this.seen.push(events);
        return super.onEvents(events, t);
      }
    }
    const spy = () => new SpyPolicy([{ vector: "yelling" }, { vector: "aggressive_tone" }, { vector: "activation" }], 20, ["A"]);

    it("is measured on the user's own turns, null on others', and DARK by default (no policy vector)", async () => {
      const queue = [unitVector(D, 0, 0.2, 3), unitVector(D, 5)];
      const embedder: Embedder = { embed: async () => queue.shift() ?? unitVector(D, 9) };
      const h = harness({ embedder, labeler: new SpeakerLabeler([you]) });
      const policy = spy();
      (h.loop as unknown as { policy: unknown }).policy = policy;
      await h.loop.start({ sessionId: "s-act", mode: "earpiece", empathy: 50 });
      for (const text of ["stop doing that", "some other person"]) {
        push(h.loop, toneInt16(2.0, -20));
        h.rec.emit({ text, isFinal: true });
        push(h.loop, silenceInt16(0.5));
        await h.loop.settle();
      }
      expect(h.turns[0].isSelf).toBe(true);
      expect(h.turns[0].activation).not.toBeNull();
      expect(h.turns[0].activation!.probability).toBeGreaterThanOrEqual(0);
      expect(h.turns[0].activation!.probability).toBeLessThanOrEqual(1);
      expect(h.turns[1].activation).toBeNull(); // not the user
      // Dark: the policy never sees an "activation" vector.
      expect(policy.seen.flat().some((e) => e.vector === "activation")).toBe(false);
      await h.loop.stop();
    });

    it("with activationNudges the vector reaches the policy (still only on self turns)", async () => {
      const embedder: Embedder = { embed: async () => unitVector(D, 0, 0.2, 3) };
      const policy = spy();
      const rec = new FakeSpeechRecognizer();
      const turns: LocalTurn[] = [];
      const loop = new FastLoop({
        vad: new EnergyVad(-45, 0.032),
        embedder,
        labeler: new SpeakerLabeler([you]),
        recognizer: rec,
        llm: new ProviderChain([okProvider(), cloudProvider()]),
        speak: () => {},
        send: () => {},
        onTurn: (t) => turns.push(t),
        policy,
        activationNudges: true,
        sttGraceMs: 150,
        pollMs: 5,
      });
      await loop.start({ sessionId: "s-act2", mode: "earpiece", empathy: 50 });
      push(loop, toneInt16(2.0, -20));
      rec.emit({ text: "stop doing that", isFinal: true });
      push(loop, silenceInt16(0.5));
      await loop.settle();
      const act = policy.seen.flat().filter((e) => e.vector === "activation");
      expect(act).toHaveLength(1);
      expect(act[0].level).toBe(turns[0].activation!.level);
      expect(act[0].value).toBeCloseTo(turns[0].activation!.probability, 10);
      await loop.stop();
    });
  });

  describe("single-mic overlap probe (live/overlapProbe.ts, dark)", () => {
    const D = 192;
    const you = { personId: "p-you", displayName: "You", isSelf: true, embedding: unitVector(D, 0) };
    const mom = { personId: "p-mom", displayName: "Mom", isSelf: false, embedding: unitVector(D, 1) };
    const blend = () => l2Normalize(Float32Array.from(unitVector(D, 0), (v, i) => v + unitVector(D, 1)[i]));

    it("a long self turn is probed window by window; mixed-voice windows are counted; short turns are not probed; nothing nudges", async () => {
      // First embed = the turn's identity (self). Then one embed per probe
      // window (5 s turn -> 8 windows): self, self, blend x4, self, self.
      const queue = [unitVector(D, 0, 0.2, 3), unitVector(D, 0), unitVector(D, 0), blend(), blend(), blend(), blend(), unitVector(D, 0), unitVector(D, 0)];
      const embedder: Embedder = { embed: async () => queue.shift() ?? unitVector(D, 0) };
      const h = harness({ embedder, labeler: new SpeakerLabeler([you, mom]) });
      await h.loop.start({ sessionId: "s-ovl", mode: "earpiece", empathy: 50 });
      push(h.loop, toneInt16(5.0, -20));
      h.rec.emit({ text: "and another thing i have been meaning to say for a while now", isFinal: true });
      push(h.loop, silenceInt16(0.5));
      await h.loop.settle();
      const t0 = h.turns[0];
      expect(t0.isSelf).toBe(true);
      expect(t0.overlap).not.toBeNull();
      expect(t0.overlap!.windows).toBe(8);
      expect(t0.overlap!.voices.filter((v) => v === "mixed")).toHaveLength(4);
      expect(t0.overlap!.longestMixedRunSeconds).toBe(2.0);
      queue.push(unitVector(D, 0, 0.2, 4));
      push(h.loop, toneInt16(1.0, -20));
      h.rec.emit({ text: "short one", isFinal: true });
      push(h.loop, silenceInt16(0.5));
      await h.loop.settle();
      expect(h.turns[1].overlap).toBeNull();
      expect(h.nudges).toHaveLength(0); // dark
      await h.loop.stop();
    });
  });

  it("a sub-1.5 s fragment that matches nobody is Unknown (no cluster, is_self null) and is not coached as self", async () => {
    // Replay-harness finding: before the guard, every short fragment minted
    // a fresh "Speaker X" (13 clusters for 2 voices on the couple scene).
    const D = 192;
    const you = { personId: "p-you", displayName: "You", isSelf: true, embedding: unitVector(D, 0) };
    const embedder: Embedder = { embed: async () => unitVector(D, 5) }; // a stranger
    const h = harness({ provider: variedProvider(), embedder, labeler: new SpeakerLabeler([you]) });
    await h.loop.start({ sessionId: "s5b", mode: "earpiece", empathy: 50 });
    push(h.loop, toneInt16(1.0, -20));
    h.rec.emit({ text: "some words here", isFinal: true });
    push(h.loop, silenceInt16(0.5));
    await h.loop.settle();
    push(h.loop, toneInt16(2.0, -20));
    h.rec.emit({ text: "more words here now", isFinal: true });
    push(h.loop, silenceInt16(0.5));
    await h.loop.settle();
    expect(h.turns.map((t) => t.speaker)).toEqual(["Unknown", "Speaker A"]);
    expect(h.turns.map((t) => t.isSelf)).toEqual([null, false]);
    expect(h.turns.map((t) => t.suggestionKind)).toEqual(["response", "response"]);
    expect(h.sent.map((e) => e.is_self)).toEqual([null, false]);
    await h.loop.stop();
  });

  it("speakQuietMs holds a suggestion until the VAD has been quiet that long (a 300 ms pause is not a turn end)", async () => {
    const h = harness();
    const loop = new FastLoop({
      vad: new EnergyVad(-45, 0.032),
      embedder: null,
      labeler: null,
      recognizer: h.rec,
      llm: new ProviderChain([okProvider(), cloudProvider()]),
      speak: (t) => h.spoken.push(t),
      send: (e) => h.sent.push(e),
      onTurn: (t) => h.turns.push(t),
      sttGraceMs: 150,
      pollMs: 5,
      speakQuietMs: 600,
    });
    await loop.start({ sessionId: "s5c", mode: "speaker", empathy: 50 });
    push(loop, toneInt16(1.0, -20));
    h.rec.emit({ text: "you never call", isFinal: true });
    // The segmenter closes the turn after 0.3 s of silence; 0.5 s is not
    // yet 0.6 s of quiet, so the suggestion is HELD, not spoken.
    push(loop, silenceInt16(0.5));
    await loop.settle();
    expect(h.turns).toHaveLength(1);
    expect(h.turns[0].suggestion).toBe("Tell her you miss the calls too.");
    expect(h.spoken).toEqual([]);
    expect(h.turns[0].latency.held).toBe(true);
    // Another 0.3 s of silence crosses the quiet threshold: released.
    push(loop, silenceInt16(0.3));
    await loop.settle();
    expect(h.spoken).toEqual(["Tell her you miss the calls too."]);
    expect(h.turns[0].spoken).toBe(true);
    await loop.stop();
  });

  it("with no recognizer (a loop built without STT): empty text, no LLM call, turn_local still sent; stop() flushes an open turn", async () => {
    const calls: string[] = [];
    const h = harness({
      recognizer: null,
      provider: { name: "os", isAvailable: async () => true, suggest: async () => (calls.push("x"), parseSuggestionJson(GOOD)) },
    });
    await h.loop.start({ sessionId: "s6", mode: "earpiece", empathy: 50 });
    push(h.loop, toneInt16(1.2, -20)); // no trailing silence: still open at stop
    const summary = await h.loop.stop();
    expect(summary.sttAvailable).toBe(false);
    expect(summary.turns).toHaveLength(1);
    expect(summary.turns[0].text).toBe("");
    expect(summary.turns[0].suggestion).toBeNull();
    expect(calls).toEqual([]);
    // Deliberately no STT (the replay harness's transcript-less scenes):
    // the turn record still flows. Contrast the two tests below, where a
    // recognizer that DIED must not claim words it never heard.
    expect(h.sent[0]).toMatchObject({ text: "", suggestion: null, suggestion_source: null });
    expect(h.spoken).toEqual([]);
  });

  it("a recognizer that fails to START sends no turn_local (an empty-text one would suppress the server's transcript)", async () => {
    const rec = new FakeSpeechRecognizer();
    rec.start = async () => { throw new Error("not available"); };
    const h = harness({ recognizer: rec });
    await h.loop.start({ sessionId: "s6b", mode: "earpiece", empathy: 50 });
    push(h.loop, toneInt16(1.0, -20));
    push(h.loop, silenceInt16(0.5));
    await h.loop.settle();
    expect(h.turns).toHaveLength(1);
    expect(h.turns[0].text).toBe("");
    // audio_pipeline._covered_by_local_range would drop the server's own
    // Deepgram segment for this span — with nothing heard on-device the
    // phone must not claim the words.
    expect(h.sent).toEqual([]);
    await h.loop.stop();
  });

  it("uses interim STT text at the grace deadline and flags it non-final; recognizer errors are surfaced once", async () => {
    const errors: string[] = [];
    const rec = new FakeSpeechRecognizer();
    const h = harness({ recognizer: rec });
    const loop = new FastLoop({
      vad: new EnergyVad(-45, 0.032),
      embedder: null,
      labeler: null,
      recognizer: rec,
      llm: new ProviderChain([cloudProvider()]),
      speak: (t) => h.spoken.push(t),
      send: (e) => h.sent.push(e),
      onTurn: (t) => h.turns.push(t),
      onSttError: (code) => errors.push(code),
      sttGraceMs: 60,
      pollMs: 5,
    });
    await loop.start({ sessionId: "s7", mode: "earpiece", empathy: 50 });
    push(loop, toneInt16(1.0, -20));
    rec.emit({ text: "I was just", isFinal: false });
    push(loop, silenceInt16(0.5));
    await loop.settle();
    expect(h.turns[0].text).toBe("I was just");
    expect(h.turns[0].transcriptFinal).toBe(false);
    expect(h.turns[0].provider).toBe("cloud"); // chain ended at cloud: nothing local
    expect(h.turns[0].suggestion).toBeNull();
    // Transient ends ("no-speech" after a pause) are the recognizer's own
    // business (it restarts itself, see expoStt.ts); anything it reports
    // here is fatal.
    rec.emitError("service-not-allowed");
    expect(errors).toEqual(["service-not-allowed"]);
    const s = await loop.stop();
    expect(s.sttAvailable).toBe(false);
  });

  it("after a fatal STT error mid-session, later turns send no turn_local (the server's transcript covers them)", async () => {
    const h = harness();
    await h.loop.start({ sessionId: "s7b", mode: "earpiece", empathy: 50 });
    push(h.loop, toneInt16(1.0, -20));
    h.rec.emit({ text: "first words", isFinal: true });
    push(h.loop, silenceInt16(0.5));
    await h.loop.settle();
    expect(h.sent).toHaveLength(1);
    h.rec.emitError("audio-capture", "mic lost");
    push(h.loop, toneInt16(1.0, -20));
    push(h.loop, silenceInt16(0.5));
    await h.loop.settle();
    expect(h.turns).toHaveLength(2);
    expect(h.turns[1].text).toBe("");
    expect(h.sent).toHaveLength(1); // nothing claimed for the second span
    await h.loop.stop();
  });

  it("re-bases the aligner's clock when the recognizer restarts itself (Android word timings restart at 0)", async () => {
    const h = harness();
    await h.loop.start({ sessionId: "s7c", mode: "earpiece", empathy: 50 });
    // 3 s into the session the native recognizer ends (a pause) and is
    // brought back: its word timings now count from session second 3.
    push(h.loop, silenceInt16(3.0));
    await h.loop.settle();
    h.rec.emitRestart();
    push(h.loop, toneInt16(1.0, -20));
    h.rec.emit({
      text: "after restart",
      isFinal: true,
      segments: [
        { startTimeMillis: 100, endTimeMillis: 500, segment: "after" },
        { startTimeMillis: 500, endTimeMillis: 900, segment: "restart" },
      ],
    });
    push(h.loop, silenceInt16(0.5));
    await h.loop.settle();
    expect(h.turns).toHaveLength(1);
    expect(h.turns[0].startTime).toBeCloseTo(3.0, 1);
    expect(h.turns[0].text).toBe("after restart");
    expect(h.turns[0].transcriptFinal).toBe(true);
    await h.loop.stop();
  });

  it("a recognizer that fails to start degrades to text-less turns", async () => {
    const rec = new FakeSpeechRecognizer();
    rec.start = async () => { throw new Error("not available"); };
    const errors: string[] = [];
    const h = harness({ recognizer: rec });
    (h.loop as unknown as { deps: { onSttError: (c: string) => void } }).deps.onSttError = (c) => errors.push(c);
    await h.loop.start({ sessionId: "s8", mode: "earpiece", empathy: 50 });
    expect(errors).toEqual(["start-failed"]);
    push(h.loop, toneInt16(1.0, -20));
    push(h.loop, silenceInt16(0.5));
    await h.loop.settle();
    expect(h.turns[0].text).toBe("");
    await h.loop.stop();
  });

  it("ignores audio before start and after stop", async () => {
    const h = harness();
    push(h.loop, toneInt16(1.0, -20));
    await h.loop.start({ sessionId: "s9", mode: "earpiece", empathy: 50 });
    await h.loop.stop();
    push(h.loop, toneInt16(1.0, -20));
    await h.loop.settle();
    expect(h.turns).toEqual([]);
    expect(h.loop.isRunning).toBe(false);
  });

  it("a suggestion still held for a quiet moment is dropped at stop(), never spoken after Stop", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => { release = r; });
    const h = harness({ provider: okProvider(GOOD, () => gate) });
    await h.loop.start({ sessionId: "s10", mode: "earpiece", empathy: 50 });
    push(h.loop, toneInt16(1.0, -20));
    h.rec.emit({ text: "hello there dear", isFinal: true });
    push(h.loop, silenceInt16(0.5)); // turn 1 finalizes, LLM blocked
    await sleep(20);
    push(h.loop, toneInt16(0.6, -20)); // someone starts talking again
    await sleep(20);
    release();
    await sleep(20);
    expect(h.spoken).toEqual([]); // held
    await h.loop.stop(); // the user pressed Stop while it was held
    expect(h.spoken).toEqual([]);
    expect(h.loop.latencyLog[0].held).toBe(true);
    expect(h.loop.latencyLog[0].toSpeakMs).toBeNull();
  });

  it("offerSpeech: a cloud line obeys the same never-over-speech rule and is refused in therapist mode", async () => {
    const h = harness({ provider: cloudProvider() });
    await h.loop.start({ sessionId: "s11", mode: "speaker", empathy: 50 });
    // Quiet: spoken at once.
    expect(h.loop.offerSpeech("Cloud: say something kind.")).toBe(true);
    expect(h.spoken).toEqual(["Cloud: say something kind."]);
    // Mid-speech: held until the VAD goes quiet.
    push(h.loop, toneInt16(0.6, -20));
    await h.loop.settle();
    expect(h.loop.offerSpeech("Cloud: held line.")).toBe(true);
    expect(h.spoken).toHaveLength(1);
    push(h.loop, silenceInt16(0.5));
    await h.loop.settle();
    expect(h.spoken[1]).toBe("Cloud: held line.");
    await h.loop.stop();

    const t = harness({ provider: cloudProvider() });
    await t.loop.start({ sessionId: "s12", mode: "therapist", empathy: 50 });
    expect(t.loop.offerSpeech("Cloud: never aloud here.")).toBe(false);
    expect(t.spoken).toEqual([]);
    await t.loop.stop();
    expect(t.loop.offerSpeech("after stop")).toBe(false);
  });

  it("setSelfSpeakerFallback flips which unknown cluster is coached as self (the screen's You: A ⇄ B chip)", async () => {
    const D = 192;
    const queue = [unitVector(D, 0), unitVector(D, 5), unitVector(D, 0, 0.05, 2), unitVector(D, 5, 0.05, 3)];
    const embedder: Embedder = { embed: async () => queue.shift() ?? unitVector(D, 9) };
    const h = harness({ provider: variedProvider(okProvider(LOUD)), embedder, labeler: new SpeakerLabeler([]) });
    await h.loop.start({ sessionId: "s13", mode: "earpiece", empathy: 50 });
    const turn = async () => {
      // >= MIN_CLUSTER_SECONDS: long enough to found an unknown cluster.
      push(h.loop, toneInt16(2.0, -20));
      h.rec.emit({ text: "some words here", isFinal: true });
      push(h.loop, silenceInt16(0.5));
      await h.loop.settle();
    };
    await turn(); // Speaker A (self by convention) -> nudge
    await turn(); // Speaker B -> response
    h.loop.setSelfSpeakerFallback("Speaker B");
    await turn(); // Speaker A -> now the OTHER person -> response
    await turn(); // Speaker B -> now self -> nudge
    expect(h.turns.map((t) => t.speaker)).toEqual(["Speaker A", "Speaker B", "Speaker A", "Speaker B"]);
    expect(h.turns.map((t) => t.suggestionKind)).toEqual(["nudge", "response", "response", "nudge"]);
    // Haptics only ever on the coached user's own turns (level-3 tone on
    // the first self turn, then policy hysteresis).
    expect(h.haptic).toEqual([3]);
    await h.loop.stop();
  });

  it("a VAD that throws is swapped for the energy rule mid-session and the UI is told once", async () => {
    const degraded: string[] = [];
    const rec = new FakeSpeechRecognizer();
    const h = harness({ recognizer: rec });
    let calls = 0;
    const brokenVad = {
      frameSamples: 512,
      isSpeech: async () => {
        calls += 1;
        throw new Error("ORT session lost");
      },
      reset() {},
    };
    const loop = new FastLoop({
      vad: brokenVad,
      embedder: null,
      labeler: null,
      recognizer: rec,
      llm: new ProviderChain([okProvider(), cloudProvider()]),
      speak: (t) => h.spoken.push(t),
      send: (e) => h.sent.push(e),
      onTurn: (t) => h.turns.push(t),
      onDegrade: (stage, reason) => degraded.push(`${stage}:${reason}`),
      sttGraceMs: 150,
      pollMs: 5,
    });
    await loop.start({ sessionId: "s14", mode: "earpiece", empathy: 50 });
    push(loop, toneInt16(1.0, -20));
    rec.emit({ text: "still heard", isFinal: true });
    push(loop, silenceInt16(0.5));
    await loop.settle();
    expect(calls).toBe(1); // the broken detector is never asked again
    expect(loop.isVadDegraded).toBe(true);
    expect(degraded).toEqual(["vad:ORT session lost"]);
    expect(h.turns).toHaveLength(1);
    expect(h.turns[0].text).toBe("still heard");
    expect(h.turns[0].suggestion).toBe("Tell her you miss the calls too.");
    await loop.stop();
  });

  describe("cross-recording (contrast) self identity", () => {
    const D = 192;
    // The owner's print pooled from TWO recordings (settings 2) — the real
    // 2026-08-27 shape: poker night scores the owner 0.42, the runner-up 0.19.
    const youAway = { personId: "p-you", displayName: "You", isSelf: true, embedding: unitVector(D, 0), settings: 2 };
    const owner = () => vectorAtCosine(D, 0.42, 0, 1);
    const stranger = () => vectorAtCosine(D, 0.19, 0, 2);

    async function speak(h: Harness, text: string) {
      push(h.loop, toneInt16(2.0, -20)); // >= MIN_CLUSTER_SECONDS: can found a cluster
      h.rec.emit({ text, isFinal: true });
      push(h.loop, silenceInt16(0.5));
      await h.loop.settle();
    }

    function withQueue(queue: Float32Array[], people: (typeof youAway)[]) {
      const embedder: Embedder = { embed: async () => queue.shift() ?? unitVector(D, 9) };
      // variedProvider: these tests assert coaching KINDS turn by turn.
      return harness({ provider: variedProvider(), embedder, labeler: new SpeakerLabeler(people) });
    }

    it("(a) settings=2, clusters at 0.42 / 0.19: the first is self by contrast once the second exists", async () => {
      const h = withQueue([owner(), stranger(), owner()], [youAway]);
      await h.loop.start({ sessionId: "cx-a", mode: "earpiece", empathy: 50 });
      await speak(h, "one voice alone");
      // Alone: an honest miss (nothing to contrast with) — not self on the wire.
      expect(h.sent[0]).toMatchObject({ speaker: "Speaker A", speaker_person_id: null, is_self: false, speaker_match_basis: null });
      await speak(h, "a second voice");
      expect(h.sent[1]).toMatchObject({ speaker: "Speaker B", speaker_person_id: null, is_self: false });
      // The earlier turn is revised in place (the session record follows).
      expect(h.turns[0]).toMatchObject({ speaker: "Speaker A", personId: "p-you", displayName: "You", isSelf: true, matchBasis: "contrast" });
      expect(h.turns[0].matchScore).toBeCloseTo(0.42, 4);
      await speak(h, "the owner again");
      expect(h.turns[2]).toMatchObject({ speaker: "Speaker A", personId: "p-you", isSelf: true, matchBasis: "contrast", suggestionKind: "nudge" });
      expect(h.sent[2]).toMatchObject({
        speaker: "Speaker A", // the raw label stays the wire key
        speaker_person_id: "p-you",
        is_self: true,
        speaker_match_basis: "contrast",
      });
      expect(h.sent[2].speaker_match_score).toBeCloseTo(0.42, 4);
      await h.loop.stop();
    });

    it("(b) settings=1: the same scores never make anyone self", async () => {
      const h = withQueue([owner(), stranger(), owner()], [{ ...youAway, settings: 1 }]);
      await h.loop.start({ sessionId: "cx-b", mode: "earpiece", empathy: 50 });
      for (const t of ["one", "two", "three"]) await speak(h, t);
      expect(h.turns.map((t) => [t.speaker, t.isSelf, t.personId, t.matchBasis])).toEqual([
        ["Speaker A", false, null, null],
        ["Speaker B", false, null, null],
        ["Speaker A", false, null, null],
      ]);
      expect(h.sent.every((e) => e.is_self === false && e.speaker_match_basis === null)).toBe(true);
      await h.loop.stop();
    });

    it("(c) a single cluster at 0.55 is not self until a second cluster appears", async () => {
      const h = withQueue([vectorAtCosine(D, 0.55, 0, 1), vectorAtCosine(D, 0.55, 0, 1), vectorAtCosine(D, 0.05, 0, 2)], [youAway]);
      await h.loop.start({ sessionId: "cx-c", mode: "earpiece", empathy: 50 });
      await speak(h, "just me so far");
      await speak(h, "still just me");
      expect(h.turns.map((t) => [t.speaker, t.isSelf, t.matchBasis])).toEqual([
        ["Speaker A", false, null],
        ["Speaker A", false, null],
      ]);
      await speak(h, "someone else");
      expect(h.turns.map((t) => [t.speaker, t.isSelf, t.personId, t.matchBasis])).toEqual([
        ["Speaker A", true, "p-you", "contrast"],
        ["Speaker A", true, "p-you", "contrast"],
        ["Speaker B", false, null, null],
      ]);
      await h.loop.stop();
    });

    it("(d) revision: a later cluster at 0.60 beats the earlier 0.42 by the margin, so self moves", async () => {
      const h = withQueue([owner(), stranger(), vectorAtCosine(D, 0.6, 0, 3), owner()], [youAway]);
      await h.loop.start({ sessionId: "cx-d", mode: "earpiece", empathy: 50 });
      await speak(h, "first voice");
      await speak(h, "second voice");
      expect(h.turns[0]).toMatchObject({ speaker: "Speaker A", isSelf: true, matchBasis: "contrast" });
      await speak(h, "third voice, closer to the print");
      expect(h.turns[2]).toMatchObject({ speaker: "Speaker C", personId: "p-you", isSelf: true, matchBasis: "contrast" });
      expect(h.turns[2].matchScore).toBeCloseTo(0.6, 4);
      // A person is one voice: Speaker A gave self up (and is honestly not-self).
      expect(h.turns[0]).toMatchObject({ speaker: "Speaker A", personId: null, displayName: null, isSelf: false, matchScore: null, matchBasis: null });
      await speak(h, "first voice again");
      expect(h.turns[3]).toMatchObject({ speaker: "Speaker A", personId: null, isSelf: false, matchBasis: null, suggestionKind: "response" });
      expect(h.sent[3]).toMatchObject({ speaker: "Speaker A", speaker_person_id: null, is_self: false, speaker_match_basis: null });
      await h.loop.stop();
    });

    it("(e) absolute: 0.70 matches outright with a single cluster, exactly as before", async () => {
      const h = withQueue([vectorAtCosine(D, 0.7, 0, 1)], [{ ...youAway, settings: 1 }]);
      await h.loop.start({ sessionId: "cx-e", mode: "earpiece", empathy: 50 });
      await speak(h, "me, clearly");
      expect(h.turns[0]).toMatchObject({ speaker: "You", personId: "p-you", isSelf: true, matchBasis: "absolute", suggestionKind: "nudge" });
      expect(h.sent[0]).toMatchObject({ speaker: "You", speaker_person_id: "p-you", is_self: true, speaker_match_basis: "absolute" });
      expect(h.sent[0].speaker_match_score).toBeCloseTo(0.7, 4);
      await h.loop.stop();
    });

    it("a user's own binding of a cluster outranks the contrast inference on that label", async () => {
      const h = withQueue([owner(), stranger(), owner()], [youAway]);
      await h.loop.start({ sessionId: "cx-f", mode: "earpiece", empathy: 50 });
      await speak(h, "first voice");
      h.loop.bindSpeaker("Speaker A", { personId: "p-mom", displayName: "Mom", isSelf: false });
      await speak(h, "second voice");
      // The contrast rule would call A the owner; the user said it is Mom.
      expect(h.turns[0]).toMatchObject({ speaker: "Speaker A", personId: "p-mom", displayName: "Mom", isSelf: false });
      await speak(h, "first voice again");
      expect(h.turns[2]).toMatchObject({ speaker: "Speaker A", personId: "p-mom", displayName: "Mom", isSelf: false, matchBasis: null });
      expect(h.sent[2]).toMatchObject({ speaker: "Speaker A", speaker_person_id: "p-mom", is_self: false, speaker_match_basis: null });
      await h.loop.stop();
    });
  });

  it("bounds what a long turn hands to the embedder (last MAX_EMBED_SECONDS) and still measures loudness over all of it", async () => {
    const D = 192;
    const seen: number[] = [];
    const embedder: Embedder = {
      embed: async (pcm) => {
        seen.push(pcm.length);
        return unitVector(D, 0);
      },
    };
    const h = harness({ embedder, labeler: new SpeakerLabeler([]) });
    await h.loop.start({ sessionId: "s15", mode: "earpiece", empathy: 50 });
    push(h.loop, toneInt16(14.0, -20)); // a 14 s monologue
    h.rec.emit({ text: "long", isFinal: true });
    push(h.loop, silenceInt16(0.5));
    await h.loop.settle();
    // The identity embed is bounded; the dark overlap probe then embeds its
    // own 1.5 s windows over this long self-by-convention turn.
    expect(seen[0]).toBe(MAX_EMBED_SECONDS * 16000);
    expect(seen.slice(1).every((n) => n === 1.5 * 16000)).toBe(true);
    expect(h.turns[0].prosody.rms_dbfs).toBeCloseTo(-20, 0);
    expect(h.turns[0].endTime - h.turns[0].startTime).toBeCloseTo(14, 0);
    await h.loop.stop();
  });
});

describe("prosodyHint", () => {
  it("names loud/quiet and fast/slow, nothing when unremarkable", () => {
    expect(prosodyHint({ rms_dbfs: -10, pitch_hz: null, speech_rate: 4 })).toBe("loud, fast");
    expect(prosodyHint({ rms_dbfs: -40, pitch_hz: null, speech_rate: 1 })).toBe("quiet, slow");
    expect(prosodyHint({ rms_dbfs: -25, pitch_hz: null, speech_rate: 2.5 })).toBeUndefined();
    expect(prosodyHint({ rms_dbfs: null, pitch_hz: null, speech_rate: null })).toBeUndefined();
  });
});
