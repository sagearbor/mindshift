/**
 * The acoustic instant tier WIRED IN — does it actually run inside the live
 * loop, on the rolling 2 s window, once a second?
 *
 * It is dark: nothing escalates on it, and the scene pack's nudge pins below
 * must be exactly what they were before it existed. What this file checks is
 * that the measurement reaches the places a later confirm/veto policy will
 * read it from — `ReplayResult.heat` (the per-second window record) and
 * `LocalTurn.instantHeat` (the turn summary the nudge report renders).
 *
 * Skipped honestly when the 80 MB ECAPA export is not on this machine; the
 * window arithmetic itself is covered model-free in instantTier.test.ts.
 */
import {
  DEFAULT_REPLAY_OPTIONS,
  findEcapaModel,
  loadModels,
  loadScene,
  replayScene,
  type LoadedModels,
  type SceneInput,
} from "../src/live/replay/sceneReplay";
import { SCENE_PACK } from "../src/live/replay/cli";
import { HEAT_TICK_SECONDS, HEAT_WINDOW_SECONDS } from "../src/live/fastLoop";
import { buildNudgeReport } from "../src/live/replay/nudgeReport";

const ecapaPath = findEcapaModel();
const maybe = ecapaPath ? describe : describe.skip;

maybe("instant tier inside the live loop", () => {
  let models: LoadedModels;
  const scenes: Record<string, SceneInput> = {};

  beforeAll(async () => {
    models = await loadModels({ ...DEFAULT_REPLAY_OPTIONS, ortFactory: null, ecapaPath });
    for (const n of SCENE_PACK) scenes[n] = loadScene(n);
  }, 60_000);

  it("scores the rolling 2 s window once a second, and the turns carry it", async () => {
    const scene = scenes.scene_couple_escalation;
    const r = await replayScene(scene, {
      models,
      mode: "earpiece",
      enrollFrom: SCENE_PACK.filter((n) => n !== "scene_couple_escalation").map((n) => scenes[n]),
    });

    // --- the per-second window record ------------------------------------
    expect(r.heat.length).toBeGreaterThan(10);
    const seconds = r.heat.map((w) => w.t);
    // strictly increasing, on whole-second boundaries, never before a full
    // window's worth of audio exists
    expect([...seconds].sort((a, b) => a - b)).toEqual(seconds);
    expect(new Set(seconds).size).toBe(seconds.length);
    for (const w of r.heat) {
      expect(w.t).toBeGreaterThanOrEqual(HEAT_WINDOW_SECONDS);
      expect(Number.isInteger(w.t / HEAT_TICK_SECONDS)).toBe(true);
      expect(w.score).toBeGreaterThan(0);
      expect(w.score).toBeLessThan(1);
      expect(w.frames).toBeGreaterThan(0);
    }
    // it does not fire on every second of the scene: a window with no speech
    // in it is skipped rather than scored on room tone
    expect(r.heat.length).toBeLessThanOrEqual(Math.ceil(r.durationSec));
    // and it did find voiced frames in a scene that is nothing but talking
    expect(r.heat.some((w) => w.voicedFrames > 0)).toBe(true);

    // --- the turn summary -------------------------------------------------
    const scored = r.turns.filter((t) => t.instantHeat !== null);
    expect(scored.length).toBeGreaterThan(0);
    for (const t of scored) {
      expect(t.instantHeat as number).toBeGreaterThan(0);
      expect(t.instantHeat as number).toBeLessThan(1);
    }

    // --- and it reaches the nudge report ----------------------------------
    const card = buildNudgeReport(r);
    const frags = card.turns.flatMap((t) => t.fragments);
    expect(frags.some((f) => f.instantHeat !== null)).toBe(true);
    expect(card.turns.some((t) => t.instantHeatMax !== null)).toBe(true);

    // --- DARK: the shipped ladder is untouched ----------------------------
    // The pins live in replay.scenes.test.ts; this is the one-line version —
    // the acoustic tier changed nothing about what actually buzzed.
    expect(r.nudgeScore).toMatchObject({ hits: 3, misses: 0, falsePositives: 0 });
    expect(r.haptics).toEqual(
      r.nudges.filter((n) => n.level > 0 && n.vectors.length > 0).map((n) => n.level),
    );
  }, 120_000);

  it("can be switched off entirely", async () => {
    const scene = scenes.scene_family3;
    const r = await replayScene(scene, { models, mode: "earpiece", instantHeat: false });
    expect(r.heat).toEqual([]);
    for (const t of r.turns) expect(t.instantHeat).toBeNull();
  }, 120_000);
});
