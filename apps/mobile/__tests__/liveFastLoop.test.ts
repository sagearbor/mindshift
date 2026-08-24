/**
 * src/live/fastLoop.ts — the whole on-device loop end to end with synthetic
 * PCM through the energy VAD (32 ms frames), a fake recognizer, fake LLM
 * providers, fake embedder, and spies on speak / send / haptics.
 */
import { FastLoop, formatLatencyLog, prosodyHint, type LocalTurn } from "../src/live/fastLoop";
import { EnergyVad } from "../src/live/vad";
import { FakeSpeechRecognizer } from "../src/live/stt";
import { cloudProvider, ProviderChain, parseSuggestionJson, type SuggestionProvider } from "../src/live/localLlm";
import { SpeakerLabeler, type Embedder } from "../src/live/speakerId";
import type { NudgeEvent } from "../src/live/nudgePolicy";
import type { TurnLocalEvent } from "../src/live/types";
import { silenceInt16, toneInt16, unitVector } from "../src/live/testing/synth";

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
    expect(h.turns.map((t) => t.suggestionKind)).toEqual(["nudge", "nudge", "nudge"]);
    expect(h.sent[0]).toMatchObject({ speaker: "You", speaker_person_id: "p-you", is_self: true });
    expect(h.sent[0].speaker_match_score as number).toBeGreaterThan(0.65);
    // aggressive_tone level 3 already on turn 1 (frustration 95) => haptic 3 once;
    // sustained level 3 afterwards is silent (policy hysteresis).
    expect(h.haptic).toEqual([3]);
    expect(h.nudges).toHaveLength(1);
    expect(h.nudges[0]).toMatchObject({ channel: "A", level: 3 });
    // Spoken nudges are the short delivery cue.
    expect(h.spoken).toEqual(["ease up", "ease up", "ease up"]);
    await h.loop.stop();
  });

  it("without a voiceprint verdict, the 'Speaker A is you' convention decides the coaching kind (never is_self on the wire)", async () => {
    const D = 192;
    const queue = [unitVector(D, 0), unitVector(D, 5)];
    const embedder: Embedder = { embed: async () => queue.shift() ?? unitVector(D, 9) };
    const h = harness({ embedder, labeler: new SpeakerLabeler([]) });
    await h.loop.start({ sessionId: "s5", mode: "earpiece", empathy: 50 });
    for (let i = 0; i < 2; i++) {
      push(h.loop, toneInt16(1.0, -20));
      h.rec.emit({ text: "some words here", isFinal: true });
      push(h.loop, silenceInt16(0.5));
      await h.loop.settle();
    }
    expect(h.turns.map((t) => t.speaker)).toEqual(["Speaker A", "Speaker B"]);
    expect(h.turns.map((t) => t.suggestionKind)).toEqual(["nudge", "response"]);
    expect(h.sent.map((e) => e.is_self)).toEqual([null, null]);
    await h.loop.stop();
  });

  it("with no recognizer: empty text, no LLM call, turn_local still sent; stop() flushes an open turn", async () => {
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
    expect(h.sent[0]).toMatchObject({ text: "", suggestion: null, suggestion_source: null });
    expect(h.spoken).toEqual([]);
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
    rec.emitError("no-speech"); // idle, not broken
    rec.emitError("service-not-allowed");
    expect(errors).toEqual(["service-not-allowed"]);
    const s = await loop.stop();
    expect(s.sttAvailable).toBe(false);
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
});

describe("prosodyHint", () => {
  it("names loud/quiet and fast/slow, nothing when unremarkable", () => {
    expect(prosodyHint({ rms_dbfs: -10, pitch_hz: null, speech_rate: 4 })).toBe("loud, fast");
    expect(prosodyHint({ rms_dbfs: -40, pitch_hz: null, speech_rate: 1 })).toBe("quiet, slow");
    expect(prosodyHint({ rms_dbfs: -25, pitch_hz: null, speech_rate: 2.5 })).toBeUndefined();
    expect(prosodyHint({ rms_dbfs: null, pitch_hz: null, speech_rate: null })).toBeUndefined();
  });
});
