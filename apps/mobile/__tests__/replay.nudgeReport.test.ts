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
import { SCENE_PACK } from "../src/live/replay/cli";
import { parseSceneMeta } from "../src/live/replay/meta";
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

const GATES: Record<string, NudgeGate> = {
  ...Object.fromEntries(SCENE_PACK.map((s) => [s, { minHits: PINNED[s] }])),
  family_real: { minHits: FAMILY_REAL_NUDGES.length },
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
  }, 60_000);

  afterAll(() => {
    if (reports.length === 0) return;
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
