/**
 * src/live/replay/* — the harness's own pieces, without any model: the WAV
 * reader, the three meta shapes, the virtual clock, the scripted recognizer
 * / provider, and every scoring rule on hand-built turns. Runs everywhere
 * (no ECAPA needed); the scene/real replays live in replay.scenes.test.ts
 * and replay.real.test.ts.
 */
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { parseWav, readWav16kMono, int16ToFloat32 } from "../src/live/replay/wav";
import { parseSceneMeta, speakerSeconds } from "../src/live/replay/meta";
import { InflightTracker, VirtualClock } from "../src/live/replay/virtualClock";
import {
  REFUSAL_TEXT,
  ScriptedRecognizer,
  scriptedProvider,
  SpokenLog,
  TONE_BY_EMOTION,
  toneForTurns,
  turnsInText,
  recordingPolicy,
} from "../src/live/replay/fakes";
import {
  levelSatisfies,
  matchScriptTurns,
  median,
  percentile,
  scoreAttribution,
  scoreBoundaries,
  scoreLatency,
  scoreNudges,
  scoreSpeaking,
} from "../src/live/replay/score";
import { AUDIO_FIXTURES_DIR, loadScene } from "../src/live/replay/sceneReplay";
import { parseArgs as parseArgsForTest } from "../src/live/replay/cli";
import type { LocalTurn } from "../src/live/fastLoop";
import { phoneNudgePolicy } from "../src/live/nudgePolicy";
import { ProviderChain, cloudProvider } from "../src/live/localLlm";
import type { EnrollmentRecord } from "../src/live/replay/enroll";

// ---------------------------------------------------------------------------
// WAV
// ---------------------------------------------------------------------------

function wavBuffer(samples: number[], sampleRate = 16000, channels = 1): Buffer {
  const data = Buffer.alloc(samples.length * 2);
  samples.forEach((s, i) => data.writeInt16LE(s, i * 2));
  const header = Buffer.alloc(44);
  header.write("RIFF", 0, "ascii");
  header.writeUInt32LE(36 + data.length, 4);
  header.write("WAVE", 8, "ascii");
  header.write("fmt ", 12, "ascii");
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * channels * 2, 28);
  header.writeUInt16LE(channels * 2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36, "ascii");
  header.writeUInt32LE(data.length, 40);
  return Buffer.concat([header, data]);
}

describe("wav reader", () => {
  it("parses 16 kHz mono int16 and rejects other rates with the ffmpeg hint", () => {
    const wav = parseWav(wavBuffer([0, 1000, -1000, 32767, -32768]));
    expect(wav).toMatchObject({ channels: 1, sampleRate: 16000, bitsPerSample: 16, frames: 5 });
    expect(Array.from(wav.samples)).toEqual([0, 1000, -1000, 32767, -32768]);
    expect(int16ToFloat32(wav.samples)[3]).toBeCloseTo(32767 / 32768, 6);

    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "replay-wav-"));
    const bad = path.join(dir, "bad.wav");
    fs.writeFileSync(bad, wavBuffer([0, 0], 48000, 2));
    expect(() => readWav16kMono(bad)).toThrow(/ffmpeg -i <input> -ac 1 -ar 16000/);
    expect(() => parseWav(Buffer.from("not a wav at all"))).toThrow(/RIFF/);
  });

  it("reads the scene pack and the real recordings at the contract rate (the 24 kHz OpenAI fixtures are refused)", () => {
    for (const f of ["scene_couple_escalation", "scene_family3", "scene_meeting4", "family_real", "poker6_real"]) {
      const pcm = readWav16kMono(path.join(AUDIO_FIXTURES_DIR, `test_recording_${f}.wav`));
      expect(pcm.length).toBeGreaterThan(16000 * 5);
    }
    expect(() => readWav16kMono(path.join(AUDIO_FIXTURES_DIR, "test_recording_openai.wav"))).toThrow(/24000 Hz/);
  });
});

// ---------------------------------------------------------------------------
// Meta
// ---------------------------------------------------------------------------

describe("scene meta -> replay script", () => {
  it("scene pack: turn times follow from duration_sec + silence_gap_sec (test_diarize_scenes._build_turns)", () => {
    const s = loadScene("scene_couple_escalation").script;
    expect(s.turns).toHaveLength(13);
    expect(s.selfSpeaker).toBe("Speaker A");
    expect(s.speakers).toEqual(["Speaker A", "Speaker B"]);
    expect(s.voices).toEqual({ "Speaker A": "onyx", "Speaker B": "coral" });
    expect(s.turns[0]).toMatchObject({ start: 0, end: 4.0966, emotionCoarse: "neutral", scriptedEmotion: "calm_open" });
    expect(s.turns[1].start).toBeCloseTo(4.0966 + 0.4, 4);
    expect(s.expectedNudges.map((n) => [n.afterTurnIndex, n.level])).toEqual([
      [4, "mild"],
      [6, "strong"],
      [8, "strong"],
    ]);
    expect(s.approxBoundaries).toBe(false);
    expect(s.hasText).toBe(true);
    // Last turn ends at the WAV length (the pack's sum(duration) + gaps invariant).
    const wavSeconds = loadScene("scene_couple_escalation").pcm.length / 16000;
    expect(s.turns[12].end).toBeCloseTo(wavSeconds, 1);
    expect(speakerSeconds(s, "Speaker A")).toBeCloseTo(33.97, 1);
  });

  it("real recording with pipeline turns: explicit start/end, self supplied by the caller", () => {
    const s = loadScene("family_real", { selfSpeaker: "Sage" }).script;
    expect(s.turns).toHaveLength(8);
    expect(s.turns[3]).toMatchObject({ speaker: "Sage", start: 10.48, end: 15.125 });
    expect(s.selfSpeaker).toBe("Sage");
    expect(s.speakers).toEqual(["Sage", "Asher"]);
    expect(s.voices).toEqual({ Sage: null, Asher: null });
    expect(s.expectedNudges).toEqual([]);
    expect(() => parseSceneMeta({ turns: [{ speaker: "A", start_time: 0, end_time: 1 }] }, { selfSpeaker: "Nobody" })).toThrow(/self speaker/);
  });

  it("real recording with approx_turns: approximate boundaries, owner as self, no text", () => {
    const s = loadScene("poker6_real").script;
    expect(s.turns).toHaveLength(6);
    expect(s.selfSpeaker).toBe("Player6");
    expect(s.approxBoundaries).toBe(true);
    expect(s.boundarySlackSec).toBe(2);
    expect(s.hasText).toBe(false);
    expect(s.turns[5]).toMatchObject({ speaker: "Player6", start: 25, end: 30.12 });
  });

  it("a hand-written phone-capture meta is the same shape", () => {
    const s = parseSceneMeta(
      {
        self_speaker: "You",
        turns: [
          { speaker: "Mom", text: "how are you", start_time: 0.5, end_time: 2.0 },
          { speaker: "You", text: "fine mom", start_time: 2.6, end_time: 3.4, emotion_coarse: "angry" },
        ],
        expected_nudges: [{ after_turn_index: 1, level: "mild" }],
      },
      { name: "mom_call" },
    );
    expect(s.name).toBe("mom_call");
    expect(s.turns[1].emotionCoarse).toBe("angry");
    expect(s.expectedNudges).toEqual([{ afterTurnIndex: 1, level: "mild", reason: null }]);
    expect(() => parseSceneMeta({ turns: [] })).toThrow(/non-empty/);
    expect(() => parseSceneMeta({ turns: [{ speaker: "A", duration_sec: 1 }] })).toThrow(/silence_gap_sec/);
    expect(() => parseSceneMeta({ turns: [{ speaker: "A", start_time: 0, end_time: 1 }], expected_nudges: [{ after_turn_index: 3, level: "mild" }] })).toThrow(/out of range/);
  });
});

// ---------------------------------------------------------------------------
// Virtual clock
// ---------------------------------------------------------------------------

describe("VirtualClock", () => {
  it("fires timers in due order at their due time, never early, and lets chained timers land", async () => {
    const clock = new VirtualClock();
    const log: string[] = [];
    void clock.sleep(300).then(() => log.push(`a@${clock.now()}`));
    void clock.sleep(100).then(async () => {
      log.push(`b@${clock.now()}`);
      await clock.sleep(150); // due at 250, registered while advancing
      log.push(`c@${clock.now()}`);
    });
    await clock.advanceTo(50);
    expect(log).toEqual([]);
    await clock.advanceTo(1000);
    expect(log).toEqual(["b@100", "c@250", "a@300"]);
    expect(clock.now()).toBe(1000);
    expect(clock.pendingTimers).toBe(0);
    await expect(clock.advanceTo(999)).rejects.toThrow(/cannot go back/);
  });

  it("InflightTracker.quiesce waits for tracked work and what it triggers; wall time is accounted", async () => {
    const tracker = new InflightTracker();
    let done = 0;
    const work = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
    void tracker.track("a", work(20)).then(() => {
      done += 1;
      void tracker.track("b", work(20)).then(() => (done += 1));
    });
    await tracker.quiesce();
    expect(done).toBe(2);
    expect(tracker.calls).toEqual({ a: 1, b: 1 });
    expect(tracker.wallMs.a).toBeGreaterThanOrEqual(15);
  });
});

// ---------------------------------------------------------------------------
// Scripted STT / LLM / TTS
// ---------------------------------------------------------------------------

describe("ScriptedRecognizer", () => {
  const script = parseSceneMeta({
    self_speaker: "A",
    turns: [
      { speaker: "A", text: "one two three four", start_time: 0, end_time: 2.0 },
      { speaker: "B", text: "five six", start_time: 2.5, end_time: 3.5 },
    ],
  });

  it("emits interims while a turn runs and one word-timed final `finalLatencyMs` after it ends", async () => {
    const rec = new ScriptedRecognizer(script, { finalLatencyMs: 500, interimEveryMs: 500, wordTimings: true }, () => 0);
    const got: { text: string; isFinal: boolean; segments?: unknown[] }[] = [];
    rec.onResult((e) => got.push(e));
    await rec.start();
    for (let t = 0.1; t <= 4.2; t += 0.1) rec.tick(Math.round(t * 10) / 10);
    const finals = got.filter((e) => e.isFinal);
    expect(finals.map((e) => e.text)).toEqual(["one two three four", "five six"]);
    expect(rec.emitted.filter((e) => e.isFinal).map((e) => e.at)).toEqual([2.5, 4.0]);
    const segs = finals[0].segments as { startTimeMillis: number; endTimeMillis: number; segment: string }[];
    expect(segs.map((s) => s.segment)).toEqual(["one", "two", "three", "four"]);
    expect(segs[0]).toEqual({ startTimeMillis: 0, endTimeMillis: 500, segment: "one" });
    expect(segs[3]).toEqual({ startTimeMillis: 1500, endTimeMillis: 2000, segment: "four" });
    const interims = got.filter((e) => !e.isFinal).map((e) => e.text);
    expect(interims.length).toBeGreaterThan(0);
    expect(interims[0]).toBe("one");
    expect(interims).toContain("one two three");
    // Untimed (iOS-style) finals carry no segments.
    const ios = new ScriptedRecognizer(script, { finalLatencyMs: 0, interimEveryMs: 0, wordTimings: false });
    const iosGot: { isFinal: boolean; segments?: unknown }[] = [];
    ios.onResult((e) => iosGot.push(e));
    await ios.start();
    ios.tick(5);
    expect(iosGot).toHaveLength(2);
    expect(iosGot.every((e) => e.isFinal && e.segments === undefined)).toBe(true);
  });
});

describe("scriptedProvider + tone", () => {
  const script = parseSceneMeta({
    self_speaker: "A",
    turns: [
      { speaker: "A", text: "You never listen to me.", start_time: 0, end_time: 2, emotion_coarse: "angry" },
      { speaker: "B", text: "I am trying, really.", start_time: 2.5, end_time: 4, emotion_coarse: "sad" },
      { speaker: "A", text: "Okay.", start_time: 4.5, end_time: 5, emotion_coarse: "neutral" },
    ],
  });

  it("finds the scripted turns inside STT text (fragments, interims, merged spans) and scores the strongest emotion", () => {
    expect(turnsInText(script, "you never listen to me").map((t) => t.index)).toEqual([0]);
    expect(turnsInText(script, "never listen to").map((t) => t.index)).toEqual([0]); // fragment
    expect(turnsInText(script, "You never listen to me. I am trying, really.").map((t) => t.index)).toEqual([0, 1]); // merged
    expect(turnsInText(script, "")).toEqual([]);
    expect(toneForTurns(turnsInText(script, "You never listen to me. I am trying, really."))).toEqual(TONE_BY_EMOTION.angry);
    expect(toneForTurns([])).toMatchObject({ frustration: null, label: null });
    // The wire contract is int | None: every score is an integer 0..100.
    for (const tone of Object.values(TONE_BY_EMOTION)) {
      for (const v of [tone.warmth, tone.defensiveness, tone.sarcasm, tone.sadness, tone.frustration]) {
        expect(Number.isInteger(v)).toBe(true);
        expect(v as number).toBeGreaterThanOrEqual(0);
        expect(v as number).toBeLessThanOrEqual(100);
      }
    }
  });

  it("answers after its virtual latency, refuses every k-th call, and the real ProviderChain falls through", async () => {
    const clock = new VirtualClock();
    const osP = scriptedProvider(script, clock, { name: "os", latencyMs: 400, refuseEveryK: 2 });
    const bundled = scriptedProvider(script, clock, { name: "bundled", latencyMs: 700 });
    const chain = new ProviderChain([osP, bundled, cloudProvider()], ["os", "bundled", "cloud"], () => clock.now());
    const input = { text: "You never listen to me.", speaker: "A", isSelf: true, empathy: 50, context: [], mode: "earpiece" as const };
    // The chain reaches the provider's sleep() a microtask after the call:
    // give it one macrotask before moving the clock (what the replay's
    // quiesce step does).
    const flush = () => new Promise((r) => setTimeout(r, 0));
    const p1 = chain.suggest(input);
    await flush();
    await clock.advanceTo(399);
    let settled = false;
    void p1.then(() => (settled = true));
    await flush();
    expect(settled).toBe(false);
    await clock.advanceTo(400);
    const r1 = await p1;
    expect(r1.provider).toBe("os");
    expect(r1.output).toMatchObject({ suggestion: "ease up", textTone: TONE_BY_EMOTION.angry });
    // Second call refuses -> bundled answers 700 ms later (1100 ms total).
    const p2 = chain.suggest({ ...input, isSelf: false, speaker: "B", text: "I am trying, really." });
    await flush();
    await clock.advanceTo(400 + 400); // os refuses here...
    await flush();
    await clock.advanceTo(400 + 400 + 700); // ...bundled answers here
    const r2 = await p2;
    expect(r2.provider).toBe("bundled");
    expect(r2.attempts.map((a) => a.outcome)).toEqual(["refused", "ok"]);
    expect(r2.attempts[0].detail).toBe(REFUSAL_TEXT);
    expect(r2.output?.suggestion).toMatch(/^Try: "I hear you — I am trying, really/);
    expect(osP.calls.map((c) => c.outcome)).toEqual(["ok", "refused"]);
    // Bundled unavailable: the refusal reaches the cloud (null output).
    const noBundled = new ProviderChain(
      [scriptedProvider(script, clock, { name: "os", latencyMs: 0, refuseEveryK: 1 }), scriptedProvider(script, clock, { name: "bundled", latencyMs: 0, available: false }), cloudProvider()],
      ["os", "bundled", "cloud"],
      () => clock.now(),
    );
    const p3 = noBundled.suggest(input);
    await flush();
    await clock.advanceTo(clock.now() + 1);
    const r3 = await p3;
    expect(r3).toMatchObject({ provider: "cloud", output: null });
  });

  it("SpokenLog records the virtual instant and the VAD verdict the loop knew; recordingPolicy logs raw levels", () => {
    const clock = new VirtualClock();
    let known: boolean | null = true;
    const log = new SpokenLog(clock, () => known);
    log.speak("hi");
    known = false;
    void clock.sleep(0);
    log.speak("there");
    expect(log.lines).toEqual([
      { text: "hi", atMs: 0, atSec: 0, vadSpeechKnown: true },
      { text: "there", atMs: 0, atSec: 0, vadSpeechKnown: false },
    ]);
    const policy = recordingPolicy(phoneNudgePolicy());
    policy.onEvents([{ vector: "yelling", level: 2, t: 1 }, { vector: "aggressive_tone", level: 1, t: 1 }], 1);
    policy.onEvents([], 2);
    expect(policy.log.map((c) => [c.rawLevel, c.emitted.map((e) => e.level), c.levelAfter])).toEqual([
      [2, [2], 2],
      [0, [], 2],
    ]);
    expect(policy.current()).toEqual({ A: 2 });
  });
});

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

function turn(i: number, start: number, end: number, speaker: string, extra: Partial<LocalTurn> = {}): LocalTurn {
  const personId = extra.personId ?? null;
  return {
    index: i,
    matchBasis: null,
    speaker,
    text: extra.text ?? "words",
    transcriptFinal: true,
    startTime: start,
    endTime: end,
    isSelf: extra.isSelf ?? null,
    displayName: extra.displayName ?? null,
    personId,
    matchScore: extra.matchScore ?? null,
    prosody: { rms_dbfs: -20, pitch_hz: null, speech_rate: null },
    textTone: null,
    suggestion: extra.suggestion ?? null,
    suggestionKind: extra.suggestionKind ?? null,
    provider: extra.provider ?? "os",
    spoken: extra.spoken ?? false,
    latency: {
      turn: i,
      segmentEndMs: end * 1000 + 330,
      prosodyMs: 0,
      speakerMs: 40,
      sttWaitMs: extra.latency?.sttWaitMs ?? 100,
      llmMs: extra.latency?.llmMs ?? 400,
      toSpeakMs: extra.latency?.toSpeakMs ?? null,
      provider: extra.provider ?? "os",
      held: extra.latency?.held ?? false,
    },
  };
}

const enrolledSelf: EnrollmentRecord = {
  personId: "p-A",
  displayName: "A",
  isSelf: true,
  embedding: new Float32Array(2),
  fromScene: "x",
  fromSpeaker: "A",
  crossScene: true,
  turnsUsed: [0],
  seconds: 5,
};

describe("scoring", () => {
  const script = parseSceneMeta({
    self_speaker: "A",
    turns: [
      { speaker: "A", text: "t0", start_time: 0, end_time: 2, emotion_coarse: "angry" },
      { speaker: "B", text: "t1", start_time: 2.5, end_time: 4 },
      { speaker: "C", text: "t2", start_time: 4.5, end_time: 6, emotion_coarse: "angry" },
      { speaker: "A", text: "t3", start_time: 6.5, end_time: 8 },
      { speaker: "B", text: "t4", start_time: 8.5, end_time: 10 },
    ],
    expected_nudges: [{ after_turn_index: 0, level: "strong" }],
  });

  it("attribution: enrolled names exact, unknown clusters best-permutation, missing turns wrong", () => {
    const turns = [
      turn(0, 0.1, 1.9, "A", { personId: "p-A", isSelf: true }),
      turn(1, 2.6, 3.9, "Speaker A"), // unknown cluster 0 == B
      turn(2, 4.6, 5.9, "Speaker B"), // unknown cluster 1 == C
      turn(3, 6.6, 7.9, "Speaker B"), // self mislabeled as cluster 1
      // t4 never segmented
    ];
    const a = scoreAttribution(script, turns, [enrolledSelf]);
    expect(a.mapping).toEqual({ "?Speaker A": "B", "?Speaker B": "C" });
    expect(a.perTurn.map((t) => t.ok)).toEqual([true, true, true, false, false]);
    expect(a).toMatchObject({ correct: 3, total: 5, selfCorrect: 1, selfTotal: 2, enrolledCorrect: 1, enrolledTotal: 2, unknownClusters: 2, speakersDetected: 3 });
    expect(matchScriptTurns(script, turns)).toEqual([0, 1, 2, 3, null]);
    // The loop's "Speaker A" cluster is never confused with the enrolled "A".
    expect(a.perTurn[1].predicted).toBe("?Speaker A");
    // With nobody enrolled every label is a cluster: 3 clusters over 3 truth speakers.
    const none = scoreAttribution(script, turns.map((t) => ({ ...t, personId: null, speaker: t.index === 0 ? "Speaker C" : t.speaker })), []);
    expect(none.correct).toBe(3);
  });

  it("boundaries: errors, split / merged / unmatched / extra", () => {
    const turns = [
      turn(0, 0.05, 1.0, "A"),
      turn(1, 1.3, 2.1, "A"), // t0 split in two
      turn(2, 2.4, 6.1, "B"), // t1 + t2 merged
      turn(3, 6.5, 8.0, "A"),
      turn(4, 11, 12, "B"), // nothing scripted here
    ];
    const b = scoreBoundaries(script, turns);
    expect(b.split).toBe(1);
    expect(b.merged).toBe(1);
    expect(b.unmatched).toBe(1); // t4
    expect(b.extra).toBe(1);
    expect(b.startErrMs).toHaveLength(4);
    expect(b.medianStartMs).toBeCloseTo(median([50, 100, 100, 0]), 6);
    expect(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90)).toBe(9);
    expect(Number.isNaN(median([]))).toBe(true);
  });

  it("nudges: hits by escalation level, misses, false positives on non-self turns coached as self, silent hits", () => {
    const turns = [turn(0, 0, 2, "A"), turn(1, 2.5, 4, "B"), turn(2, 4.5, 6, "C"), turn(3, 6.5, 8, "A"), turn(4, 8.5, 10, "B")];
    const policy = recordingPolicy(phoneNudgePolicy());
    policy.onEvents([{ vector: "aggressive_tone", level: 2, t: 2 }], 2); // t0: strong -> hit
    policy.onEvents([], 4);
    policy.onEvents([{ vector: "aggressive_tone", level: 2, t: 6 }], 6); // t2 (C) wrongly coached as self -> FP, and silent (level held)
    policy.onEvents([{ vector: "yelling", level: 0, t: 8 }], 8); // self, calm
    policy.onEvents([], 10);
    const n = scoreNudges(script, turns, policy.log);
    expect(n).toMatchObject({ hits: 1, misses: 0, falsePositives: 1, hitsSilent: 0 });
    expect(n.perTurn.map((t) => t.verdict)).toEqual(["hit", "quiet", "fp", "quiet", "quiet"]);
    expect(n.hapticsFired).toEqual([{ t: 2, level: 2, scriptTurn: 0 }]);
    expect(levelSatisfies(1, "mild")).toBe(true);
    expect(levelSatisfies(1, "strong")).toBe(false);
    // A strong expectation met only at level 1 is a miss.
    const weak = recordingPolicy(phoneNudgePolicy());
    weak.onEvents([{ vector: "aggressive_tone", level: 1, t: 2 }], 2);
    expect(scoreNudges(script, turns, weak.log)).toMatchObject({ hits: 0, misses: 1 });
  });

  it("speaking: the invariant uses what the loop knew; scripted overlap is informational", () => {
    const turns = [turn(0, 0, 2, "A", { suggestion: "s0", spoken: true }), turn(1, 2.5, 4, "B", { suggestion: "s1", spoken: false, latency: { held: true } as LocalTurn["latency"] })];
    const spoken = [
      { text: "s0", atMs: 2400, atSec: 2.4, vadSpeechKnown: false },
      { text: "x", atMs: 3000, atSec: 3.0, vadSpeechKnown: true },
    ];
    const vad = { speechAt: (t: number) => t >= 2.5 && t < 4 };
    const s = scoreSpeaking(script, turns, spoken, vad);
    expect(s).toMatchObject({ spoken: 2, overVadSpeech: 1, overVadTimeline: 1, overScriptedSpeech: 1, held: 1, dropped: 1 });
  });

  it("latency: medians over spoken turns, provider histogram, textless / interim counts", () => {
    const turns = [
      turn(0, 0, 2, "A", { latency: { toSpeakMs: 900 } as LocalTurn["latency"], provider: "os" }),
      turn(1, 2.5, 4, "B", { latency: { toSpeakMs: 1500 } as LocalTurn["latency"], provider: "bundled" }),
      turn(2, 4.5, 6, "C", { text: "", provider: "none" }),
    ];
    turns[1].transcriptFinal = false;
    const l = scoreLatency(script, turns);
    expect(l).toMatchObject({ turns: 3, spokenTurns: 2, toSpeakMedianMs: 1200, toSpeakMaxMs: 1500, textless: 1, interimOnly: 1 });
    expect(l.providers).toEqual({ os: 1, bundled: 1, none: 1 });
    expect(l.segmentCloseMedianMs).toBeCloseTo(330, 6);
  });
});

describe("cli argument parsing", () => {
  it("maps flags onto replay options and rejects unknown ones", () => {
    const a = parseArgsForTest(["scene_family3", "--mode", "therapist", "--stt-latency", "900", "--refuse-every", "3", "--no-bundled", "--speak-quiet", "600", "--dump", "/tmp/x"]);
    expect(a).toMatchObject({ scene: "scene_family3", mode: "therapist", sttLatency: 900, refuseEvery: 3, bundled: false, speakQuiet: 600, dump: "/tmp/x" });
    expect(() => parseArgsForTest(["--bogus"])).toThrow(/unknown option/);
    expect(() => parseArgsForTest([])).toThrow(/help/);
  });
});
