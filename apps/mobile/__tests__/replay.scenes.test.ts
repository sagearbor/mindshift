/**
 * The scene pack through the REAL fast loop (Silero + segmenter + ECAPA +
 * aligner + provider chain + nudge policy) on a virtual clock — pinned at
 * the MEASURED numbers (2026-08-24, onnxruntime-node, ECAPA export
 * 0f99f2d0…). The rule from server/tests/test_diarize_scenes.py applies:
 * a pin that regresses is investigated, not lowered; one that improves is
 * raised.
 *
 * What the numbers say (see the PR for the full tables):
 * - The segmenter closes a turn after a 300 ms pause, and TTS sentence
 *   pauses are 0.4–0.8 s, so every scripted turn is cut into 2–3 fragments
 *   (couple: 30 loop turns for 13 scripted). Suggestions are therefore
 *   produced per fragment and mostly HELD until the next pause — which is
 *   usually a pause INSIDE the other person's turn.
 * - Self is enrolled CROSS-SCENE (the demo case). Fragments >= ~2 s match
 *   the print (cosine 0.66–0.79); shorter ones don't, and a shout does not
 *   (meeting4 turn 13).
 * - The nudge timeline holds on couple (3/3) and family3 (1/1, no false
 *   positive on the teen's shout); meeting4's shouted spike is missed
 *   because the shouting voice does not match the calm print.
 *
 * Skipped honestly when the 80 MB ECAPA export is not on this machine.
 */
import {
  findEcapaModel,
  formatReport,
  loadModels,
  loadScene,
  replayScene,
  summaryLine,
  writeTurnLocalDump,
  DEFAULT_REPLAY_OPTIONS,
  REPLAY_OUT_DIR,
  type LoadedModels,
  type ReplayResult,
  type SceneInput,
} from "../src/live/replay/sceneReplay";
import { SCENE_PACK } from "../src/live/replay/cli";
import { MATCH_THRESHOLD } from "../src/live/speakerId";

const ecapaPath = findEcapaModel();
const maybe = ecapaPath ? describe : describe.skip;

/** The wire shape every event must have (the pydantic side is
 *  server/tests/test_replay_turn_local_contract.py over the dump). */
function expectTurnLocalShape(r: ReplayResult) {
  expect(r.sent).toHaveLength(r.turns.length);
  for (const e of r.sent) {
    expect(e.type).toBe("turn_local");
    expect(e.session_id).toBe(`replay-${r.scene}-${r.mode}`);
    expect(e.transcript_source).toBe("on-device");
    expect(e.tts_source).toBe("on-device");
    expect(e.end_time).toBeGreaterThanOrEqual(e.start_time);
    expect(e.start_time).toBeGreaterThanOrEqual(0);
    expect(e.suggestion_source).toBe(e.suggestion ? "on-device" : null);
    if (e.text_tone) {
      for (const v of [e.text_tone.warmth, e.text_tone.defensiveness, e.text_tone.frustration]) {
        expect(v === null || Number.isInteger(v)).toBe(true);
      }
    }
    if (e.is_self === true) expect(e.speaker_person_id).not.toBeNull();
    if (e.speaker_person_id === null) expect(e.is_self).not.toBe(true);
  }
}

function expectInvariants(r: ReplayResult) {
  // Never speak over live speech (as the loop knew it) — every mode.
  expect(r.speaking.overVadSpeech).toBe(0);
  // Every spoken line came from a suggestion the loop produced.
  for (const l of r.spoken) expect(r.turns.some((t) => t.suggestion === l.text)).toBe(true);
  // Haptics fire exactly for the policy's escalation events (never on a decay).
  expect(r.haptics).toEqual(r.nudges.filter((n) => n.level > 0 && n.vectors.length > 0).map((n) => n.level));
  expect(r.nudgeScore.hapticsFired.map((h) => h.level)).toEqual(r.haptics);
  if (r.mode === "therapist") {
    expect(r.spoken).toEqual([]);
    expect(r.turns.filter((t) => t.spoken)).toHaveLength(0);
  }
  expectTurnLocalShape(r);
}

maybe("scene pack replay (real Silero + ECAPA, scripted STT/LLM, virtual clock)", () => {
  let models: LoadedModels;
  const scenes: Record<string, SceneInput> = {};
  const pool = (name: string) => SCENE_PACK.filter((n) => n !== name).map((n) => scenes[n]);

  beforeAll(async () => {
    models = await loadModels({ ...DEFAULT_REPLAY_OPTIONS, ortFactory: null, ecapaPath });
    for (const n of SCENE_PACK) scenes[n] = loadScene(n);
  }, 60_000);

  const run = (name: string, extra: Parameters<typeof replayScene>[1]) =>
    replayScene(scenes[name], { models, enrollFrom: pool(name), ...extra });

  it("scene_couple_escalation / earpiece: 11/13 attribution (self 6/7), 3/3 nudges, no false positives, spoken never over speech", async () => {
    const r = await run("scene_couple_escalation", { mode: "earpiece" });
    console.log(formatReport(r));
    console.log(summaryLine(r));
    expectInvariants(r);
    expect(r.capability.enrolled).toHaveLength(1);
    expect(r.capability.enrolled[0]).toMatchObject({ displayName: "Speaker A", isSelf: true, crossScene: true, fromScene: "scene_family3", turnsUsed: [2, 5, 9, 11, 14] });
    // Attribution: the print is from ANOTHER scene; 6 of 7 self turns match
    // it (the miss is the closing 1.3 s fragment), and the partner is now ONE
    // clean unknown cluster. With the live CLUSTER_THRESHOLD lowered to 0.40
    // (real short-turn same-voice similarity), the partner's fragments no
    // longer split off a stray cluster: 2 speakers, correct 12/13 (was 3/11).
    expect(r.attribution).toMatchObject({ correct: 12, total: 13, selfCorrect: 6, selfTotal: 7, speakersDetected: 2, unknownClusters: 1 });
    const selfScores = r.turns.filter((t) => t.isSelf).map((t) => t.matchScore as number);
    expect(Math.min(...selfScores)).toBeGreaterThanOrEqual(MATCH_THRESHOLD);
    // Fragmentation: 30 loop turns, 12 of 13 scripted turns split, none merged.
    expect(r.turns).toHaveLength(30);
    expect(r.boundaries).toMatchObject({ split: 12, merged: 0, unmatched: 0, extra: 0 });
    expect(r.boundaries.medianEndMs).toBeLessThan(50); // the LAST fragment ends where the script does
    // Nudges: mild@4, strong@6, strong@8 all reached (two silently: the
    // policy holds level 2, so no new haptic); the loud/tense partner turns
    // never nudge.
    expect(r.nudgeScore).toMatchObject({ hits: 3, misses: 0, falsePositives: 0, hitsSilent: 2 });
    expect(r.nudgeScore.perTurn.filter((t) => t.expected).map((t) => t.level)).toEqual([2, 2, 2]);
    expect(r.nudgeScore.hapticsFired.map((h) => [h.level, h.scriptTurn])).toEqual([[2, 4]]);
    // The cooldown decay (2 -> 1 twenty seconds later) reaches the screen, not the motor.
    expect(r.nudges.map((n) => [n.level, n.vectors])).toEqual([
      [2, ["aggressive_tone"]],
      [1, []],
    ]);
    // Speech: every suggestion was held (the next fragment starts within
    // 0.4 s) and voiced at the following pause — 17 of 29 inside a scripted turn.
    expect(r.speaking).toMatchObject({ spoken: 29, held: 29, dropped: 1, overVadSpeech: 0, overScriptedSpeech: 17 });
    expect(r.latency).toMatchObject({ toSpeakMedianMs: 1600, toSpeakMaxMs: 4100, textless: 0, interimOnly: 17 });
    expect(r.latency.providers).toEqual({ os: 30 });
    expect(r.stt.finals).toBe(13);
    writeTurnLocalDump(r, REPLAY_OUT_DIR);
  }, 60_000);

  it("scene_couple_escalation / speaker: identical to earpiece (both voice; the mode only changes the audio route)", async () => {
    const r = await run("scene_couple_escalation", { mode: "speaker" });
    expectInvariants(r);
    expect(r.attribution.correct).toBe(12);
    expect(r.speaking.spoken).toBe(29);
    expect(r.latency.toSpeakMedianMs).toBe(1600);
  }, 60_000);

  it("scene_couple_escalation / therapist: never speaks, still coaches on screen and sends every turn_local", async () => {
    const r = await run("scene_couple_escalation", { mode: "therapist", enroll: "all" });
    console.log(summaryLine(r));
    expectInvariants(r);
    expect(r.spoken).toEqual([]);
    expect(r.turns.filter((t) => t.suggestion)).toHaveLength(30);
    expect(r.sent.filter((e) => e.suggestion)).toHaveLength(30);
    // Both partners enrolled (the therapist setup): the partner's print is
    // pooled from her voice in this scene (coral appears nowhere else).
    expect(r.capability.enrolled.map((e) => [e.displayName, e.crossScene])).toEqual([
      ["Speaker A", true],
      ["Speaker B", false],
    ]);
    // 10/13 named exactly: the partner's same-scene print (first 10 s of
    // her) misses her two shortest fragments; self is as in earpiece.
    expect(r.attribution).toMatchObject({ enrolledCorrect: 10, enrolledTotal: 13, selfCorrect: 6 });
    expect(r.nudgeScore).toMatchObject({ hits: 3, misses: 0, falsePositives: 0 });
    writeTurnLocalDump(r, REPLAY_OUT_DIR);
  }, 60_000);

  it("scene_couple_escalation: a refusing OS model falls through to the bundled model, or to the cloud when there is none", async () => {
    const r = await run("scene_couple_escalation", { mode: "earpiece", osRefuseEveryK: 3 });
    expectInvariants(r);
    expect(r.providerCalls).toEqual({ os: 30, osRefused: 10, bundled: 10 });
    expect(r.latency.providers).toEqual({ os: 20, bundled: 10 });
    expect(r.turns.filter((t) => t.suggestion)).toHaveLength(30);
    // Refused turns pay both latencies (400 + 700 ms).
    const viaBundled = r.turns.filter((t) => t.provider === "bundled");
    expect(viaBundled.every((t) => t.latency.llmMs === 1100)).toBe(true);
    expect(viaBundled.every((t) => t.suggestion !== null && !/can't help/.test(t.suggestion))).toBe(true);

    const cloud = await run("scene_couple_escalation", { mode: "earpiece", osRefuseEveryK: 3, bundledAvailable: false });
    expectInvariants(cloud);
    expect(cloud.latency.providers).toEqual({ os: 20, cloud: 10 });
    expect(cloud.turns.filter((t) => t.provider === "cloud").every((t) => t.suggestion === null)).toBe(true);
    expect(cloud.sent.filter((e) => e.suggestion === null && e.suggestion_source === null)).toHaveLength(10);
    expect(cloud.speaking.spoken).toBeLessThan(r.speaking.spoken);
  }, 90_000);

  it("scene_couple_escalation: speakQuietMs=600 stops most mid-turn whispers (29 -> 10 spoken) at the cost of dropping superseded ones", async () => {
    const r = await run("scene_couple_escalation", { mode: "earpiece", speakQuietMs: 600 });
    expectInvariants(r);
    expect(r.speaking).toMatchObject({ spoken: 10, held: 29, dropped: 20, overVadSpeech: 0, overScriptedSpeech: 8 });
    expect(r.latency.toSpeakMedianMs).toBe(2050);
    // Coaching itself is unchanged: same turns, same nudges.
    expect(r.attribution.correct).toBe(12);
    expect(r.nudgeScore).toMatchObject({ hits: 3, misses: 0, falsePositives: 0 });
  }, 60_000);

  it("scene_couple_escalation: STT finalization latency vs the loop's 700 ms grace (interim text is used past it; nothing is lost)", async () => {
    const out: string[] = [];
    for (const finalLatencyMs of [200, 500, 1200]) {
      const r = await run("scene_couple_escalation", { mode: "earpiece", stt: { ...DEFAULT_REPLAY_OPTIONS.stt, finalLatencyMs } });
      expectInvariants(r);
      out.push(`stt ${finalLatencyMs} ms: textless ${r.latency.textless} interim-only ${r.latency.interimOnly} stt-wait median ${r.latency.sttWaitMedianMs} first-words median ${r.latency.toSpeakMedianMs}`);
      expect(r.latency.textless).toBe(0);
      expect(r.turns.filter((t) => t.suggestion)).toHaveLength(30);
      if (finalLatencyMs === 200) expect(r.latency.interimOnly).toBe(17);
      if (finalLatencyMs === 1200) expect(r.latency.interimOnly).toBe(30);
    }
    console.log(out.join("\n"));
    // iOS-style untimed finals: text is attributed by window instead of
    // word timing — still every turn gets words.
    const ios = await run("scene_couple_escalation", { mode: "earpiece", stt: { ...DEFAULT_REPLAY_OPTIONS.stt, wordTimings: false } });
    expectInvariants(ios);
    expect(ios.latency.textless).toBe(0);
  }, 120_000);

  it("scene_family3 / earpiece: 0.25 s gaps merge neighbours (6 merged), 9/15 attribution, the teen's shout does NOT nudge, self flare does (1/1)", async () => {
    const r = await run("scene_family3", { mode: "earpiece" });
    console.log(formatReport(r));
    console.log(summaryLine(r));
    expectInvariants(r);
    expect(r.capability.enrolled[0]).toMatchObject({ crossScene: true, fromScene: "scene_couple_escalation" });
    expect(r.attribution).toMatchObject({ correct: 9, total: 15, selfCorrect: 3, selfTotal: 5, speakersDetected: 6, unknownClusters: 5 });
    expect(r.turns).toHaveLength(24);
    expect(r.boundaries).toMatchObject({ split: 11, merged: 6, unmatched: 0 });
    expect(r.nudgeScore).toMatchObject({ hits: 1, misses: 0, falsePositives: 0 });
    expect(r.nudgeScore.perTurn[7].level).toBe(0); // the teen's shout_angry
    expect(r.nudgeScore.perTurn[6].level).toBe(0); // the other parent's tense_rising
    expect(r.nudgeScore.perTurn[9]).toMatchObject({ expected: "mild", level: 2, verdict: "hit" });
    expect(r.speaking).toMatchObject({ spoken: 18, held: 23, overVadSpeech: 0 });
    expect(r.latency).toMatchObject({ toSpeakMedianMs: 1400, toSpeakMaxMs: 2700, textless: 0 });
    writeTurnLocalDump(r, REPLAY_OUT_DIR);
    const t = await run("scene_family3", { mode: "therapist" });
    expectInvariants(t);
  }, 90_000);

  it("scene_meeting4 / earpiece: 11/17 attribution, self 2/5 — the shout/apology don't match the calm print (missed nudge), and this 4-party meeting is where the live CLUSTER_THRESHOLD=0.48 costs some separation (its voices sit unusually close); documented trade-off for the couples/family core", async () => {
    const r = await run("scene_meeting4", { mode: "earpiece" });
    console.log(formatReport(r));
    console.log(summaryLine(r));
    expectInvariants(r);
    expect(r.attribution).toMatchObject({ correct: 11, total: 17, selfCorrect: 2, selfTotal: 5, speakersDetected: 7, unknownClusters: 6 });
    expect(r.turns).toHaveLength(34);
    expect(r.boundaries).toMatchObject({ split: 14, merged: 0, unmatched: 0 });
    // mild@11 hit; strong@13 missed: the 2 s shouted fragment scored below
    // MATCH_THRESHOLD against the calm cross-scene print and became an
    // unknown cluster, so the loop did not treat it as the coached user.
    expect(r.nudgeScore).toMatchObject({ hits: 1, misses: 1, falsePositives: 0 });
    expect(r.nudgeScore.perTurn[11]).toMatchObject({ expected: "mild", verdict: "hit", level: 2 });
    expect(r.nudgeScore.perTurn[13]).toMatchObject({ expected: "strong", verdict: "miss", level: 0 });
    expect(r.nudgeScore.perTurn[7].level).toBe(0); // Speaker D's tense_rising: no nudge
    const shout = r.attribution.perTurn[13];
    expect(shout.predicted?.startsWith("?")).toBe(true);
    expect(r.speaking).toMatchObject({ spoken: 30, held: 33, overVadSpeech: 0 });
    expect(r.latency).toMatchObject({ toSpeakMedianMs: 2100, toSpeakMaxMs: 3400, textless: 0 });
    writeTurnLocalDump(r, REPLAY_OUT_DIR);
    const t = await run("scene_meeting4", { mode: "therapist" });
    expectInvariants(t);
  }, 90_000);

  it("with speaker-ID off (no model) every turn is Unknown and nobody is ever coached as self", async () => {
    const r = await replayScene(scenes.scene_couple_escalation, {
      mode: "earpiece",
      models: { ...models, embedder: null, ecapaPath: null },
      enrollFrom: pool("scene_couple_escalation"),
    });
    expectInvariants(r);
    expect(r.capability.speakerId).toBe(false);
    expect(new Set(r.turns.map((t) => t.speaker))).toEqual(new Set(["Unknown"]));
    expect(r.turns.every((t) => t.isSelf === null && t.suggestionKind !== "nudge")).toBe(true);
    expect(r.nudgeScore).toMatchObject({ hits: 0, misses: 3 });
  }, 60_000);
});
