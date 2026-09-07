/**
 * ⚡ The vocal-activation GATE — does the dark classifier deserve to nudge?
 *
 * `activation.ts` ships DARK: the phone measures "how worked-up does this
 * voice sound" on every one of the user's turns, records it, shows it in
 * Developer mode, and never buzzes about it. The owner's condition for
 * turning it on (2026-09-06) is a FILE gate, not an opinion:
 *
 *   ROC-AUC >= 0.75 separating heated from calm turns, AND zero flags on
 *   calm turns at the shipped ladder.
 *
 * This measures both, on the phone's OWN TypeScript implementation, over real
 * human speech (RAVDESS, 24 professional actors). Two things that makes it
 * different from the 0.82 grouped-CV number in tmp/ravdess/analysis:
 *
 *  1. It runs `activationFeatures` + `activationProbability` — the PORT — not
 *     sklearn. A number that only exists in Python cannot gate a phone.
 *  2. It asks the PRODUCT's question. The model was trained on RAVDESS's
 *     intensity label (strong vs normal, pooled across every emotion,
 *     including strong SADNESS and strong HAPPINESS, which a coach must
 *     never nudge). The gate instead scores "angry, strong" against "calm or
 *     neutral, normal" — heated versus not — which is the only comparison the
 *     ladder's thresholds will ever be asked to make in a real conversation.
 *
 * VERDICT (2026-09-06): the corpus half passes at AUC 1.000 with zero false
 * flags. The OUT-OF-CORPUS half — the same classifier run through the real
 * loop over the owner's own family recording and the TTS scenes — fails
 * catastrophically, and lives in replay.nudgeReport.test.ts where those
 * replays already happen. Both halves must pass before `activationNudges`
 * flips; the last test here pins that it has not.
 *
 * Skips honestly without the corpus (tmp/ravdess is a local, CC BY-NC-SA
 * download — see scripts/make_ravdess_scene.py).
 */
import * as fs from "fs";
import * as path from "path";
import {
  ACTIVATION_LEVELS,
  ACTIVATION_WINDOW_SECONDS,
  activationLevel,
  turnActivationAsync,
} from "../src/live/activation";
import { readCorpusWavTo16k } from "../src/live/replay/corpusWav";
import { REPO_ROOT } from "../src/live/replay/sceneReplay";

const RAVDESS = path.join(REPO_ROOT, "tmp", "ravdess", "audio");
const haveCorpus = fs.existsSync(RAVDESS);
const maybe = haveCorpus ? describe : describe.skip;

/** The owner's bar (2026-09-06). Raise, never lower. */
const MIN_AUC = 0.75;

/** RAVDESS: 03-01-<emotion>-<intensity>-<statement>-<repetition>-<actor>.wav */
interface Clip {
  file: string;
  emotion: string;
  intensity: string;
  actor: number;
}

function listClips(): Clip[] {
  const out: Clip[] = [];
  for (const dir of fs.readdirSync(RAVDESS).sort()) {
    const actorDir = path.join(RAVDESS, dir);
    if (!fs.statSync(actorDir).isDirectory()) continue;
    for (const name of fs.readdirSync(actorDir).sort()) {
      if (!name.endsWith(".wav")) continue;
      const p = name.replace(/\.wav$/, "").split("-");
      if (p.length !== 7) continue;
      out.push({ file: path.join(actorDir, name), emotion: p[2], intensity: p[3], actor: Number(p[6]) });
    }
  }
  return out;
}

/** Area under the ROC curve, by the rank (Mann-Whitney) identity — exact, and
 *  it handles ties the way sklearn does (average ranks). */
export function rocAuc(scores: number[], labels: number[]): number {
  const idx = scores.map((s, i) => [s, i] as const).sort((a, b) => a[0] - b[0]);
  const ranks = new Array<number>(scores.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[idx[k][1]] = avg;
    i = j + 1;
  }
  const pos = labels.reduce((a, b) => a + b, 0);
  const neg = labels.length - pos;
  if (pos === 0 || neg === 0) return NaN;
  let sumPos = 0;
  for (let k = 0; k < labels.length; k++) if (labels[k] === 1) sumPos += ranks[k];
  return (sumPos - (pos * (pos + 1)) / 2) / (pos * neg);
}

maybe("⚡ vocal-activation gate (RAVDESS, the phone's own implementation)", () => {
  // "Heated" = angry at strong intensity. "Calm" = calm or neutral at normal
  // intensity. Everything else (sad, fearful, happy, disgust, surprised) is
  // deliberately EXCLUDED: they are neither what the ladder should fire on nor
  // what it should stay silent through, and folding them in would make the
  // number look better or worse for reasons the product does not care about.
  const clips = haveCorpus ? listClips() : [];
  const heated = clips.filter((c) => c.emotion === "05" && c.intensity === "02");
  const calm = clips.filter((c) => (c.emotion === "02" || c.emotion === "01") && c.intensity === "01");

  let probs: { p: number; label: number; clip: Clip }[] = [];

  beforeAll(async () => {
    // The PHONE's entry point, not the raw feature extractor: it windows to
    // the last ACTIVATION_WINDOW_SECONDS, which is load-bearing. The two
    // strongest coefficients are voiced/unvoiced DURATION, so an unwindowed
    // clip is scored partly on how long it happens to be — measured here
    // 2026-09-06: without the window, two ordinary calm clips that simply ran
    // long scored p = 0.997 and p = 0.9998 and would have buzzed at level 3.
    probs = [];
    for (const clip of [...heated, ...calm]) {
      const pcm = readCorpusWavTo16k(clip.file);
      const act = await turnActivationAsync(pcm, 16000, { sleep: async () => {} });
      probs.push({ p: act?.probability ?? 0, label: clip.emotion === "05" ? 1 : 0, clip });
    }
  }, 300_000);

  it("has enough of both classes, across every actor, for the number to mean anything", () => {
    // 24 actors x 2 statements x 2 repetitions = 96 angry-strong clips;
    // calm+neutral at normal intensity is twice that.
    expect(heated.length).toBe(96);
    expect(calm.length).toBe(192);
    expect(new Set(clips.map((c) => c.actor)).size).toBe(24);
  });

  it(`separates heated from calm at ROC-AUC >= ${MIN_AUC}`, () => {
    const auc = rocAuc(probs.map((r) => r.p), probs.map((r) => r.label));
    const meanHeated = probs.filter((r) => r.label === 1).reduce((a, r) => a + r.p, 0) / heated.length;
    const meanCalm = probs.filter((r) => r.label === 0).reduce((a, r) => a + r.p, 0) / calm.length;
    console.log(
      `activation gate: AUC ${auc.toFixed(3)} over ${heated.length} heated / ${calm.length} calm clips ` +
        `(mean p heated ${meanHeated.toFixed(3)}, calm ${meanCalm.toFixed(3)})`,
    );
    expect(auc).toBeGreaterThanOrEqual(MIN_AUC);
  });

  it("never flags a calm turn at the shipped ladder — the condition for letting it nudge", () => {
    const flagged = probs.filter((r) => r.label === 0 && activationLevel(r.p) > 0);
    const firstRung = ACTIVATION_LEVELS[ACTIVATION_LEVELS.length - 1][0];
    console.log(
      `activation gate: ${flagged.length}/${calm.length} calm clips reach level >= 1 ` +
        `(first rung p >= ${firstRung}); worst calm p = ${Math.max(...probs.filter((r) => r.label === 0).map((r) => r.p)).toFixed(3)}`,
    );
    // A false "you're getting worked up" while somebody is speaking normally
    // is the one failure that makes a person switch the coach off.
    for (const r of flagged) {
      console.log(`  calm clip flagged: ${path.basename(r.clip.file)} p=${r.p.toFixed(4)} level=${activationLevel(r.p)}`);
    }
    expect(flagged.map((r) => path.basename(r.clip.file))).toEqual([]);
  });

  it("catches a useful share of the heated turns it is allowed to nudge on", () => {
    const caught = probs.filter((r) => r.label === 1 && activationLevel(r.p) > 0);
    const recall = caught.length / heated.length;
    console.log(`activation gate: recall on heated ${caught.length}/${heated.length} = ${(recall * 100).toFixed(1)}%`);
    // No bar asserted: with a zero-false-positive requirement, recall is
    // whatever the ladder's conservatism leaves. It is logged so the owner can
    // see what the silence costs, and so a threshold change shows up here.
    expect(recall).toBeGreaterThanOrEqual(0);
  });
});

describe("⚡ activation stays dark", () => {
  it("is off by default, and says why in the same place a reader would change it", () => {
    // A classifier that scores AUC 1.000 inside its own corpus and flags
    // "Okay, this is Sage talking, I'm about to head off" as level-2 worked-up
    // on a real recording is not ready to buzz anyone. The measurement that
    // says so lives in replay.nudgeReport.test.ts; the reasoning lives next to
    // the flag itself, so nobody can flip it without reading it.
    const src = fs.readFileSync(path.join(__dirname, "..", "src", "live", "fastLoop.ts"), "utf8");
    expect(src).toContain("this.activationNudges = deps.activationNudges ?? false;");
    expect(src).toMatch(/Default FALSE, and it must stay false until the gate/);
    expect(src).toMatch(/clip-shape detector/);
  });
});
