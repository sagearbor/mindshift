/**
 * The two REAL recordings through the fast loop (measured 2026-08-24 and
 * pinned — investigate a regression, raise an improvement):
 *
 * - test_recording_family_real.wav (owner + son, self = "Sage"). Self is
 *   enrolled CROSS-RECORDING from his poker-night turn (poker6 Player6):
 *   the demo case, enrolled in one room and matched in another. It does
 *   NOT match: cosine 0.08–0.29 against his family turns (MATCH_THRESHOLD
 *   0.65), while the same-recording Sage turns score 0.49–0.68 among
 *   themselves and 0.01–0.11 against Asher. Same-recording enrollment
 *   matches 3/3. Enrollment must happen in the demo's own acoustic setting
 *   — see the PR.
 * - test_recording_poker6_real.wav (6 real men, no transcript). Online
 *   clustering finds 6 unknown clusters for 6 voices after the
 *   short-segment guard (7 before), 5/6 by best permutation; the owner's
 *   own turn does not match his family-recording print (same finding).
 *
 * Skipped honestly when the ECAPA export is absent.
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
  type SceneInput,
} from "../src/live/replay/sceneReplay";
import { KNOWN_VOICE_PAIRS } from "../src/live/replay/cli";

const ecapaPath = findEcapaModel();
const maybe = ecapaPath ? describe : describe.skip;

maybe("real recordings replay (real Silero + ECAPA)", () => {
  let models: LoadedModels;
  let family: SceneInput;
  let poker: SceneInput;

  beforeAll(async () => {
    models = await loadModels({ ...DEFAULT_REPLAY_OPTIONS, ortFactory: null, ecapaPath });
    family = loadScene("family_real", { selfSpeaker: "Sage" });
    poker = loadScene("poker6_real");
  }, 60_000);

  it("family_real: self enrolled from the poker recording does not match (cross-recording ceiling); same-recording enrollment matches 3/3", async () => {
    const cross = await replayScene(family, { mode: "earpiece", models, enrollFrom: [poker], voicePairs: KNOWN_VOICE_PAIRS });
    console.log(formatReport(cross));
    console.log(summaryLine(cross));
    expect(cross.capability.enrolled[0]).toMatchObject({ displayName: "Sage", isSelf: true, crossScene: true, fromScene: "poker6_real", fromSpeaker: "Player6" });
    expect(cross.attribution).toMatchObject({ selfCorrect: 0, selfTotal: 3, correct: 1, total: 8, unknownClusters: 3 });
    expect(cross.turns.every((t) => t.isSelf !== true)).toBe(true);
    // Two real voices, 6 loop turns: the son's two sub-second interjections
    // ("and yeah.", "And") are below the segmenter's 0.6 s minimum, and his
    // first reply is merged into the owner's opening turn (no pause between).
    expect(cross.turns).toHaveLength(6);
    expect(cross.boundaries).toMatchObject({ merged: 1, unmatched: 2, extra: 0 });
    expect(cross.speaking.overVadSpeech).toBe(0);
    expect(cross.latency.textless).toBe(0);
    expect(cross.sent).toHaveLength(6);
    writeTurnLocalDump(cross, REPLAY_OUT_DIR);

    const same = await replayScene(family, { mode: "earpiece", models, enrollFrom: [] });
    console.log(summaryLine(same));
    expect(same.capability.enrolled[0]).toMatchObject({ crossScene: false, fromScene: "family_real", turnsUsed: [0, 3, 6] });
    expect(same.attribution).toMatchObject({ selfCorrect: 3, selfTotal: 3, correct: 4, total: 8 });
    expect(same.turns.filter((t) => t.isSelf === true)).toHaveLength(3);
    expect(same.attribution.perTurn.filter((t) => t.truth === "Asher" && t.ok)).toHaveLength(1);
  }, 90_000);

  it("poker6_real: 6 real voices, no transcript -> 6 unknown clusters, 5/6 by best permutation, nothing spoken", async () => {
    const r = await replayScene(poker, { mode: "earpiece", models, enrollFrom: [family], voicePairs: KNOWN_VOICE_PAIRS });
    console.log(formatReport(r));
    console.log(summaryLine(r));
    expect(r.capability.enrolled[0]).toMatchObject({ displayName: "Player6", crossScene: true, fromScene: "family_real", fromSpeaker: "Sage" });
    expect(r.attribution).toMatchObject({ correct: 5, total: 6, selfCorrect: 0, selfTotal: 1, unknownClusters: 6, speakersDetected: 6 });
    expect(r.turns).toHaveLength(7);
    // The 1.3 s fragment of Player2 is "Unknown" (short-segment guard) rather than a 7th cluster.
    expect(r.turns.filter((t) => t.speaker === "Unknown")).toHaveLength(1);
    // No text in the meta => no STT, no LLM, no speech; turn_local still flows.
    expect(r.stt).toEqual({ emitted: 0, finals: 0 });
    expect(r.latency.providers).toEqual({ none: 7 });
    expect(r.spoken).toEqual([]);
    expect(r.sent).toHaveLength(7);
    expect(r.sent.every((e) => e.text === "" && e.suggestion === null)).toBe(true);
    expect(r.script.approxBoundaries).toBe(true);
    writeTurnLocalDump(r, REPLAY_OUT_DIR);
  }, 60_000);
});
