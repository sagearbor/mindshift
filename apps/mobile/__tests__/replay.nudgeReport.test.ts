/**
 * The nudge verification gate — every nudge behaviour checked FROM RECORDED
 * FILES, repeatably, with a report a human reads in a minute
 * (tmp/nudge-report-<YYYYMMDD>.html + .json; owner, 2026-09-06: "I cannot
 * be the limiting factor to test manually").
 *
 * Runs the three TTS scenes + the owner's real family recording through the
 * REAL fast loop (replay.scenes.test.ts / replay.real.test.ts setups), builds
 * a NudgeReport per scene and asserts, per scene:
 *   - expected nudges hit >= the number replay.scenes.test.ts PINS — read
 *     from that file's source, so this gate can never drift weaker than it;
 *   - false positives == 0;
 *   - when a loud self turn exists, the instant loudness haptic fired
 *     before that turn's LLM-tier nudge.
 *
 * File writing is on by default locally; MINDSHIFT_NUDGE_REPORT=0 (or CI)
 * skips it; MINDSHIFT_NUDGE_REPORT_DIR picks the folder (default
 * <repo>/tmp). The server-side AMI section is included when
 * tmp/nudge-report/ami_vectors.json exists (tmp/nudge-report/ami_vectors.py).
 * Skipped honestly without the ECAPA export (~60 s per scene with it).
 */
import * as fs from "fs";
import * as path from "path";
import {
  DEFAULT_REPLAY_OPTIONS,
  findEcapaModel,
  loadModels,
  loadScene,
  REPO_ROOT,
  replayScene,
  summaryLine,
  type LoadedModels,
  type ReplayResult,
  type SceneInput,
} from "../src/live/replay/sceneReplay";
import { RAVDESS_SCENE, SCENE_PACK } from "../src/live/replay/cli";
import { parseSceneMeta } from "../src/live/replay/meta";
import { cosine, CROSS_MATCH_THRESHOLD, MATCH_THRESHOLD } from "../src/live/speakerId";
import {
  buildNudgeReport,
  callModeInterrupting,
  gateFailures,
  renderNudgeReportHtml,
  type AmiVectorsJson,
  type NudgeGate,
  type NudgeReport,
} from "../src/live/replay/nudgeReport";

const ecapaPath = findEcapaModel();
const maybe = ecapaPath ? describe : describe.skip;

/**
 * The RAVDESS scene lives in tmp/, never in the repo: its source corpus is
 * CC BY-NC-SA 4.0 (non-commercial, share-alike), so a derivative WAV must not
 * sit inside a commercial product tree. Rebuild it in a few seconds with
 * `python scripts/make_ravdess_scene.py`; without it this one test skips
 * honestly, exactly like the whole file does without the ECAPA model.
 */
const RAVDESS_WAV = path.join(REPO_ROOT, "tmp", "ravdess-scene", `test_recording_${RAVDESS_SCENE}.wav`);
const ravdessWav = fs.existsSync(RAVDESS_WAV) ? RAVDESS_WAV : null;
const withRavdess = ravdessWav ? it : it.skip;

const WRITE = process.env.MINDSHIFT_NUDGE_REPORT !== "0" && !process.env.CI;
const OUT_DIR = process.env.MINDSHIFT_NUDGE_REPORT_DIR ?? path.join(REPO_ROOT, "tmp");
const AMI_JSON = process.env.MINDSHIFT_AMI_VECTORS_JSON ?? path.join(OUT_DIR, "nudge-report", "ami_vectors.json");

function dateStamp(d = new Date()): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}`;
}

/**
 * The hits replay.scenes.test.ts pins for each scene's earpiece run, read
 * from its SOURCE: `expect(r.nudgeScore).toMatchObject({ hits: N, …` inside
 * the `it("<scene> / earpiece` block. A rename there fails loudly here
 * rather than silently weakening the gate.
 */
export function pinnedHitsFromScenesTest(src: string, scenes: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const scene of scenes) {
    const start = src.indexOf(`it("${scene} / earpiece`);
    if (start < 0) throw new Error(`replay.scenes.test.ts: no earpiece block for ${scene}`);
    const next = src.indexOf("\n  it(", start + 1);
    const block = src.slice(start, next < 0 ? undefined : next);
    const m = /nudgeScore\)\.toMatchObject\(\{\s*hits:\s*(\d+)/.exec(block);
    if (!m) throw new Error(`replay.scenes.test.ts: no pinned nudge hits in the ${scene} earpiece block`);
    out[scene] = Number(m[1]);
  }
  return out;
}

/** The positive half of the vocabulary — K never reaches a haptic sink. */
const POSITIVE_CODES = ["D", "E", "R"];

const PINNED = pinnedHitsFromScenesTest(fs.readFileSync(path.join(__dirname, "replay.scenes.test.ts"), "utf8"), SCENE_PACK);

/**
 * Ground truth for the REAL family recording (its fixture meta carries no
 * `expected_nudges`: the server's meta schema ties that key to scripted
 * emotions, which a real recording has none of). Measured 2026-09-06 by
 * this replay: on turn 6 the owner stages the argument out loud ("…and
 * I'm arguing now") and raises his voice to -15.1 dBFS, +6.2 dB over his
 * own running baseline (-19.0 / -23.5 on his first two turns) — the
 * instant loudness tier's level-1 rung, with no text tone in play. Before
 * this was written down the turn scored as a false positive. The son's
 * turns must never nudge.
 */
const FAMILY_REAL_NUDGES = [
  { afterTurnIndex: 6, level: "mild" as const, reason: "owner raises his voice (+6.2 dB over baseline) while saying 'I'm arguing now'" },
];

/**
 * The RAVDESS scene's ground truth is in its OWN meta (written by
 * scripts/make_ravdess_scene.py from the corpus's per-clip labels), so unlike
 * family_real nothing has to be restated here — `expected_nudges` and
 * `expected_positive_nudges` both come off disk.
 */
const GATES: Record<string, NudgeGate> = {
  ...Object.fromEntries(SCENE_PACK.map((s) => [s, { minHits: PINNED[s] }])),
  family_real: { minHits: FAMILY_REAL_NUDGES.length },
  // 0, not 1: the shout this fixture exists for is a DOCUMENTED MISS today —
  // the loop does not recognise the coached user while they are shouting. The
  // test below measures exactly why, and asserts the miss, so the day that is
  // fixed this file fails loudly instead of silently passing at a weaker bar.
  [RAVDESS_SCENE]: { minHits: 0 },
};

describe("nudge report: pure pieces", () => {
  it("reads the pinned hits from replay.scenes.test.ts (3, 1, 1 as of 2026-09-06 — raise there, never lower here)", () => {
    expect(PINNED).toEqual({ scene_couple_escalation: 3, scene_family3: 1, scene_meeting4: 1 });
  });

  it("the call-mode column fires the steamroll vector where the ground truth overlaps for >= 2 s", () => {
    const script = parseSceneMeta(
      {
        self_speaker: "Me",
        turns: [
          { speaker: "Them", text: "a long point being made", start_time: 0, end_time: 8 },
          { speaker: "Me", text: "talking over them", start_time: 3, end_time: 9 }, // 5 s of mutual speech
          { speaker: "Them", text: "brief", start_time: 12, end_time: 14 },
          { speaker: "Me", text: "yeah", start_time: 13.5, end_time: 14.5 }, // 0.5 s: normal overlap, nothing
        ],
      },
      { name: "synthetic" },
    );
    expect(callModeInterrupting(script)).toEqual([{ vector: "interrupting", level: 2, t: 3, value: 5 }]);
  });
});

maybe("nudge verification from recorded files (real Silero + ECAPA, scripted STT/LLM, virtual clock)", () => {
  let models: LoadedModels;
  const scenes: Record<string, SceneInput> = {};
  const reports: NudgeReport[] = [];
  const results: Record<string, ReplayResult> = {};
  const pool = (name: string) => SCENE_PACK.filter((n) => n !== name).map((n) => scenes[n]);

  beforeAll(async () => {
    models = await loadModels({ ...DEFAULT_REPLAY_OPTIONS, ortFactory: null, ecapaPath });
    for (const n of SCENE_PACK) scenes[n] = loadScene(n);
    scenes.family_real = loadScene("family_real", { selfSpeaker: "Sage", expectedNudges: FAMILY_REAL_NUDGES });
    expect(scenes.family_real.script.expectedNudges).toHaveLength(GATES.family_real.minHits);
    if (ravdessWav) scenes[RAVDESS_SCENE] = loadScene(ravdessWav);
  }, 60_000);

  /**
   * ⚡ The OUT-OF-CORPUS half of the activation gate (the in-corpus half lives
   * in activationGate.test.ts, where it passes at AUC 1.000 with zero false
   * flags over 288 RAVDESS clips).
   *
   * These are the same numbers over real speech, through the real loop. They
   * are PINNED rather than merely asserted non-zero, because the whole point
   * is that they must go to zero before `activationNudges` can flip — an
   * improvement to the classifier has to show up here as a diff.
   *
   * As of 2026-09-06: three false flags across the pack, two of them on the
   * owner's own family recording, at LEVEL 2, on "Okay, this is Sage talking
   * or dad" and "I'm about to head off" — ordinary calm narration. (Its third
   * measured self turn is the one that genuinely escalates, so it is not
   * counted here.) The diagnosis is in
   * FastLoopDeps.activationNudges' doc: the model's two strongest features are
   * voiced/unvoiced DURATION, standardized against RAVDESS clips that carry a
   * second of silence at each end, so a real conversational turn — nearly all
   * speech — reads as "worked up" for its length alone.
   */
  const ACTIVATION_FALSE_FLAGS: Record<string, number> = {
    scene_couple_escalation: 0,
    scene_family3: 0,
    scene_meeting4: 0,
    scene_ravdess_pair: 0,
    family_real: 0,
  };

  afterAll(() => {
    if (reports.length === 0) return;
    for (const rep of reports) {
      const want = ACTIVATION_FALSE_FLAGS[rep.scene];
      if (want === undefined) continue;
      const got = rep.scorecard.activationFalseFlags;
      console.log(`⚡ activation false flags on ${rep.scene}: ${got} (turns #${rep.scorecard.activationFlaggedTurns.join(", #")})`);
      if (want !== got) {
        throw new Error(
          `⚡ activation false flags on ${rep.scene}: ${got}, pinned ${want} ` +
            `(turns #${rep.scorecard.activationFlaggedTurns.join(", #")}). If this went DOWN the ` +
            "classifier improved — update the pin, and once every scene reaches 0 the " +
            "activationNudges flag can finally flip (see FastLoopDeps.activationNudges).",
        );
      }
    }
    const generatedAt = new Date().toISOString();
    let ami: AmiVectorsJson | null = null;
    if (fs.existsSync(AMI_JSON)) ami = JSON.parse(fs.readFileSync(AMI_JSON, "utf8")) as AmiVectorsJson;
    const html = renderNudgeReportHtml(reports, { generatedAt, gates: GATES, ami });
    for (const rep of reports) {
      const s = rep.scorecard;
      console.log(
        `${rep.scene}: hits ${s.hits}/${s.expected} (pinned >= ${GATES[rep.scene].minHits}) miss ${s.misses} fp ${s.falsePositives}; ` +
          `instant haptics ${s.instantHaptics}${s.instantLeadMs ? ` lead median ${s.instantLeadMs.median} ms` : ""}; LLM lag median ${s.policyLagMs?.median ?? "-"} ms; ` +
          `call-mode interrupting ${s.callModeInterrupting}; activation max ${s.activationMaxProbability ?? "-"}; overlap probed ${s.overlapProbed}\n` +
          `   🎧 earpiece: instant ${s.earpiece.instantHaptics}, screen nudges ${s.earpiece.screenNudges}, spoken nudge lines ${s.earpiece.spokenNudges} (median ${s.earpiece.nudgeToSpeakMs?.median ?? "-"} ms after the turn), responses ${s.earpiece.spokenResponses}\n` +
          `   ⌚ watch: buzzes ${s.watch.buzzes} ${JSON.stringify(s.watch.byVector)}; watch-only turns [${s.watch.watchOnlyTurns}], earpiece-only turns [${s.watch.earpieceOnlyTurns}], both [${s.watch.bothTurns}]; all on self turns ${s.watch.allOnSelfTurns}`,
      );
    }
    if (!WRITE) {
      console.log("nudge report: file writing skipped (MINDSHIFT_NUDGE_REPORT=0 or CI)");
      return;
    }
    fs.mkdirSync(OUT_DIR, { recursive: true });
    const stem = path.join(OUT_DIR, `nudge-report-${dateStamp()}`);
    fs.writeFileSync(`${stem}.html`, html);
    fs.writeFileSync(`${stem}.json`, JSON.stringify({ generatedAt, gates: GATES, scenes: reports, ami }, null, 2) + "\n");
    console.log(`nudge report: ${stem}.html (+ .json${ami ? ", AMI section included" : ", no AMI json found"})`);
  });

  const gateScene = (r: ReplayResult) => {
    console.log(summaryLine(r));
    const rep = buildNudgeReport(r);
    results[r.scene] = r;
    reports.push(rep);
    const gate = GATES[r.scene];
    const s = rep.scorecard;
    // Bookkeeping the report relies on: every haptic is classified, and every
    // escalation the policy emitted has exactly one haptic behind it (the
    // instant tier's, or the policy tier's own).
    expect(rep.haptics).toHaveLength(r.haptics.length);
    expect(rep.haptics.every((h) => h.loopTurn !== null)).toBe(true);
    const escalations = rep.nudges.filter((n) => !n.decay);
    expect(rep.haptics.length).toBe(escalations.length);
    for (const h of rep.haptics.filter((h) => h.tier === "instant")) {
      expect(h.leadOverPolicyMs).not.toBeNull();
      expect(h.leadOverPolicyMs as number).toBeGreaterThan(0);
    }
    // The gate itself.
    expect(gateFailures(rep, gate)).toEqual([]);
    expect(s.hits).toBeGreaterThanOrEqual(gate.minHits);
    expect(s.hits).toBe(r.nudgeScore.hits);
    expect(s.falsePositives).toBe(0);
    if (s.instantHaptics > 0) expect(s.instantBeforePolicy).toBe(true);
    // ⌚ The watch lane never buzzes on someone else's turn.
    expect(rep.watch.buzzes.filter((b) => !b.onSelfTurn)).toEqual([]);
    expect(rep.watch.allOnSelfTurns).toBe(true);
    // 🎧 Every instant haptic is on the earpiece lane of exactly one fragment.
    expect(rep.turns.flatMap((t) => t.fragments).filter((f) => f.earpiece.instantHaptic).length + rep.extraFragments.filter((f) => f.earpiece.instantHaptic).length).toBe(s.instantHaptics);
    // 💚 Positives are a separate lane on the same sink: every DELIVERED one
    // has a soft cue behind it, none is leveled, and each is reported.
    expect(rep.positives.filter((p) => p.delivered).length).toBe(r.positiveHaptics.length);
    for (const h of r.positiveHaptics) {
      expect(h.level).toBe(1);
      expect(POSITIVE_CODES).toContain(h.code);
    }
    expect(rep.positives.every((p) => p.detail.length > 0)).toBe(true);
    expect(
      Object.values(s.positives.delivered).reduce((a, b) => a + b, 0) + s.positives.suppressed,
    ).toBe(rep.positives.length);
    return rep;
  };

  it("scene_couple_escalation / earpiece: 3 expected nudges hit, no false positive", async () => {
    const rep = gateScene(await replayScene(scenes.scene_couple_escalation, { mode: "earpiece", models, enrollFrom: pool("scene_couple_escalation") }));
    expect(rep.turns.filter((t) => t.expected).map((t) => t.verdict)).toEqual(["hit", "hit", "hit"]);
    // The LLM tier's own latency shows in the report: STT grace + os model.
    expect(rep.scorecard.policyLagMs?.median).toBeGreaterThan(300);
  }, 120_000);

  it("scene_family3 / earpiece: the self flare is hit, the teen's shout never nudges", async () => {
    const rep = gateScene(await replayScene(scenes.scene_family3, { mode: "earpiece", models, enrollFrom: pool("scene_family3") }));
    expect(rep.turns[9].verdict).toBe("hit");
    expect(rep.turns[7].level).toBe(0);
  }, 120_000);

  withRavdess("scene_ravdess_pair / earpiece: REAL voices — a 15 s turn the user lets run (E) and the recovery after the shout (D); the shout itself is the DOCUMENTED MISS, and this measures exactly why", async () => {
    // The one fixture with a real dynamic range and a real long turn.
    // Replayed WITHOUT the TTS pack's enrollment pool: these are different
    // human beings, so the loop has to find them from this recording alone.
    const r = await replayScene(scenes[RAVDESS_SCENE], { mode: "earpiece", models, enrollFrom: [] });
    const rep = gateScene(r);
    const meta = JSON.parse(
      fs.readFileSync(RAVDESS_WAV.replace(/\.wav$/, "_meta.json"), "utf8"),
    ) as {
      measured: { angry_spike_db_over_baseline: number; longest_partner_turn_sec: number };
      expected_positive_nudges: { code: string }[];
    };
    // The two properties no TTS scene has, asserted from the fixture's own
    // measurements so a regenerated fixture cannot quietly lose them.
    expect(meta.measured.angry_spike_db_over_baseline).toBeGreaterThan(14);
    expect(meta.measured.longest_partner_turn_sec).toBeGreaterThanOrEqual(12);

    // ---------------------------------------------------------------------
    // THE DOCUMENTED MISS (2026-09-06). The loudest moment in the recording —
    // a real +28 dB shout — raises NO nudge, because the loop does not
    // recognise the coached user while they are shouting. Not a threshold to
    // nudge down: see the measurement below.
    // ---------------------------------------------------------------------
    const shout = rep.turns.find((t) => t.expected === "strong")!;
    expect(shout.verdict).toBe("miss");
    expect(shout.isSelf).toBe(true);
    // …and the loop put it on somebody else's lane entirely:
    expect(shout.fragments.every((f) => !f.coachedAsSelf)).toBe(true);
    expect(rep.scorecard.falsePositives).toBe(0);

    // WHY, measured here so the diagnosis can never go stale. A shouted turn
    // sits at ~0.36 cosine to the same person's calm print — far below the
    // 0.65 absolute bar and even below the 0.40 contrast bar — while calm
    // turns sit at ~0.9. It is still an order of magnitude closer to the user
    // than to the other speaker (~0.05), so the information IS there; what
    // blocks it is that `identifyClusters` gives each person at most ONE
    // cluster, and the calm cluster has already taken them. Lowering either
    // bar to ~0.36 would start attributing strangers' shouting to the user,
    // which is the one failure a nudge must never make — so the fix is
    // multi-prototype voiceprints (a person owning a calm AND a raised
    // cluster), not a threshold change. That is a cross-runtime contract
    // change (speakerId.ts + server/speaker_id.py + speakerCrossMatch.json)
    // and is deliberately NOT bundled into this change.
    const emb = models.embedder!;
    const scene = scenes[RAVDESS_SCENE];
    const pcmAt = (a: number, b: number) => scene.pcmF32.subarray(Math.round(a * 16000), Math.round(b * 16000));
    const selfTurns = scene.script.turns.filter((t) => t.speaker === scene.script.selfSpeaker);
    const calm = selfTurns.filter((t) => t.emotionCoarse !== "angry");
    const angry = selfTurns.find((t) => t.emotionCoarse === "angry")!;
    const calmVecs = await Promise.all(calm.map((t) => emb.embed(pcmAt(t.start, t.end), 16000)));
    const calmCentroid = new Float32Array(calmVecs[0].length);
    for (const v of calmVecs) for (let i = 0; i < v.length; i++) calmCentroid[i] += v[i] / calmVecs.length;
    const shoutVec = await emb.embed(pcmAt(angry.start, angry.end), 16000);
    const partnerVecs = await Promise.all(
      scene.script.turns.filter((t) => t.speaker !== scene.script.selfSpeaker).slice(0, 3).map((t) => emb.embed(pcmAt(t.start, t.end), 16000)),
    );
    const toSelf = cosine(shoutVec, calmCentroid);
    const toPartner = Math.max(...partnerVecs.map((v) => cosine(shoutVec, v)));
    console.log(
      `ravdess shout identity: cosine to own calm print ${toSelf.toFixed(3)} ` +
        `(absolute bar ${MATCH_THRESHOLD}, contrast bar ${CROSS_MATCH_THRESHOLD}), to the other speaker ${toPartner.toFixed(3)}; ` +
        `calm-to-calm ${calmVecs.map((v) => cosine(v, calmCentroid).toFixed(2)).join("/")}`,
    );
    expect(toSelf).toBeLessThan(MATCH_THRESHOLD); // the miss
    expect(toSelf).toBeLessThan(CROSS_MATCH_THRESHOLD); // and the contrast rule can't save it
    expect(toSelf).toBeGreaterThan(toPartner + 0.2); // but the signal is unambiguous
    for (const v of calmVecs) expect(cosine(v, calmCentroid)).toBeGreaterThan(0.8);

    // 💚 👂 E works end to end on real voices: a 15 s turn the loop's VAD cut
    // into six fragments, coalesced back into one turn nobody interrupted.
    const listened = rep.positives.find((p) => p.code === "E");
    expect(listened).toBeDefined();
    expect(listened!.delivered).toBe(true);
    expect(listened!.detail).toMatch(/1[2-9] s turn finish with no cut-in/);

    // 📉 D is the COLLATERAL DAMAGE of the same identity miss, and worth
    // stating separately because it is the more expensive half: the fixture's
    // spec expects a de-escalation right after the shout, and the user really
    // does go from a +28 dB shout to their quietest turn in the recording —
    // but a recovery needs a spike to recover FROM, and the spike was filed
    // under someone else. So one identity failure costs both the nudge that
    // should have fired AND the credit for pulling it back. Fixing the
    // voiceprint fixes both, and this assertion flips when it does.
    expect(meta.expected_positive_nudges.map((p) => p.code).sort()).toEqual(["D", "E"]);
    expect(rep.positives.map((p) => p.code)).toEqual(["E"]);
  }, 120_000);

  it("scene_meeting4 / earpiece: mild@11 hit; strong@13 is the documented miss (the shout does not match the calm print), still no false positive", async () => {
    const rep = gateScene(await replayScene(scenes.scene_meeting4, { mode: "earpiece", models, enrollFrom: pool("scene_meeting4") }));
    expect(rep.turns[11].verdict).toBe("hit");
    expect(rep.turns[13].verdict).toBe("miss");
  }, 120_000);

  it("family_real / earpiece (owner + son, self enrolled from this recording): the owner's 'I'm arguing now' turn (+6 dB) buzzes the instant tier ~0.7 s before the LLM tier; the son never nudges", async () => {
    const rep = gateScene(await replayScene(scenes.family_real, { mode: "earpiece", models, enrollFrom: [] }));
    expect(rep.scorecard.attribution.selfCorrect).toBe(3);
    expect(rep.scorecard.coachedFragments).toBe(3);
    const arguing = rep.turns[6];
    expect(arguing).toMatchObject({ speaker: "Sage", expected: "mild", verdict: "hit", instantLevelMax: 1, earpieceLevel: 1 });
    expect(arguing.dbOverBaselineMax as number).toBeGreaterThanOrEqual(6);
    // 🎧 the loudness tier is the ONLY nudge here (no text tone on a real
    // recording): instant buzz first, then the screen, then the spoken line.
    const frag = arguing.fragments.find((f) => f.earpiece.instantHaptic) as NonNullable<(typeof arguing.fragments)[number]>;
    expect(frag.earpiece.instantHaptic).toEqual({ atSec: expect.any(Number), level: 1 });
    expect(frag.earpiece.screenNudges).toEqual([{ atSec: expect.any(Number), level: 1, vectors: ["yelling"] }]);
    expect(frag.earpiece.spokenAtSec as number).toBeGreaterThan(frag.earpiece.screenNudges[0].atSec);
    expect(frag.earpiece.screenNudges[0].atSec).toBeGreaterThan(frag.earpiece.instantHaptic!.atSec);
    expect(rep.scorecard.instantLeadMs?.median as number).toBeGreaterThanOrEqual(500);
    // The son's turns: no level anywhere, on either lane.
    for (const t of rep.turns.filter((t) => t.speaker === "Asher")) expect([t.level, t.earpieceLevel, t.watchLevel]).toEqual([0, 0, 0]);
    // ⌚ the same +6 dB turn is a wrist buzz too (loudness rides both lanes) —
    // unless the airtime rule already holds the lane at L3 from the FIRST
    // turn (share = 100 % of one sentence), the engine artefact the report
    // flags; either way every watch buzz sits on the owner's own turns.
    expect(rep.watch.buzzes.length).toBeGreaterThan(0);
  }, 120_000);

  it("the page renders every scene, the legend block and the How-to-test card, with no document skeleton of its own", () => {
    expect(reports.length).toBeGreaterThan(0);
    const html = renderNudgeReportHtml(reports, { gates: GATES, ami: null });
    expect(html.startsWith("<title>")).toBe(true);
    expect(html).not.toMatch(/<html|<body|<!doctype/i);
    expect(html).toContain('<details class="legend">');
    expect(html).toContain("How to test");
    for (const rep of reports) expect(html).toContain(rep.scene);
    expect(html).toContain("⚡");
    expect(html).toContain("<svg");
    // Every feature row carries the four-column tag.
    expect(html.match(/class="tag"/g)?.length ?? 0).toBeGreaterThanOrEqual(reports.length * 7);
  });
});
