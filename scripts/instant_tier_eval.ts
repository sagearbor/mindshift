/**
 * INSTANT TIER — the real-audio gate, run through the SHIPPED TypeScript.
 *
 * Python picked the feature subset; this proves the phone code reproduces it
 * on actual corpus audio, not on a synthetic tone. It reads the raw 2 s
 * windows Python dumped (scripts/instant_tier_select.py dump-pcm, int16 LE @
 * 16 kHz under tmp/instant-tier/pcm — gitignored, no audio is committed) and
 * runs apps/mobile/src/live/instantTier.ts over every one of them.
 *
 *   npx tsx scripts/instant_tier_eval.ts --features   # dump TS features for
 *                                                     # the Python refit
 *   npx tsx scripts/instant_tier_eval.ts              # AUC gate + bench ->
 *                                                     # apps/mobile/__tests__/
 *                                                     #   fixtures/instantTier.eval.json
 *   npx tsx scripts/instant_tier_eval.ts --bench      # latency only
 *
 * The eval set is the one the plan asks for: every RAVDESS angry+happy clip
 * (the cross-corpus test — different actors, rooms, scripts) and 500 random
 * CREMA-D angry+happy clips (the within-corpus sanity check, on the corpus the
 * model was fitted on, so it is expected to be the easier number).
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  INSTANT_FEATURE_NAMES,
  extractInstantFeatures,
  instantHeatScore,
  type InstantFeatureName,
} from "../apps/mobile/src/live/instantTier";

const HERE = dirname(fileURLToPath(import.meta.url));
const WORKTREE = join(HERE, "..");
/** The corpora and the dumped PCM live in the MAIN checkout's tmp/, which a
 *  worktree does not carry. Walk up out of .claude/worktrees/<agent>/ if that
 *  is where we are. */
function mainRepoTmp(): string {
  let d = WORKTREE;
  for (let i = 0; i < 6; i++) {
    if (existsSync(join(d, "tmp/instant-tier/pcm"))) return join(d, "tmp/instant-tier");
    d = join(d, "..");
  }
  throw new Error(
    "tmp/instant-tier/pcm not found — run `python scripts/instant_tier_select.py dump-pcm` first",
  );
}

const WORK = mainRepoTmp();
const EVAL_OUT = join(WORKTREE, "apps/mobile/__tests__/fixtures/instantTier.eval.json");
const SR = 16000;

interface Row {
  key: string;
  corpus: string;
  speaker: string;
  emotion: string;
  is_angry: boolean;
}

function loadIndex(): Row[] {
  const p = join(WORK, "windows.json");
  if (!existsSync(p)) {
    throw new Error(`${p} not found — run \`python scripts/instant_tier_select.py dump-pcm\` first`);
  }
  return JSON.parse(readFileSync(p, "utf8")) as Row[];
}

function readPcm(key: string): Int16Array {
  const buf = readFileSync(join(WORK, "pcm", `${key}.pcm`));
  // Node buffers are byte-aligned but not necessarily 2-byte aligned inside
  // their pool, so copy rather than view.
  const out = new Int16Array(buf.byteLength / 2);
  for (let i = 0; i < out.length; i++) out[i] = buf.readInt16LE(i * 2);
  return out;
}

/** Area under the ROC curve via the rank-sum identity; ties get mid-ranks. */
export function auc(labels: number[], scores: number[]): number {
  const idx = scores.map((s, i) => [s, i] as const).sort((a, b) => a[0] - b[0]);
  const rank = new Float64Array(scores.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    const mid = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) rank[idx[k][1]] = mid;
    i = j + 1;
  }
  let sumPos = 0;
  let nPos = 0;
  for (let k = 0; k < labels.length; k++) {
    if (labels[k] === 1) {
      sumPos += rank[k];
      nPos++;
    }
  }
  const nNeg = labels.length - nPos;
  if (nPos === 0 || nNeg === 0) return NaN;
  return (sumPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

/** Deterministic pseudo-random ordering, so "500 random CREMA-D clips" is the
 *  same 500 on every machine and every run. */
function stableShuffleKey(key: string): string {
  return createHash("sha1").update(key).digest("hex");
}

function cmdFeatures(): void {
  const rows = loadIndex();
  const out: Record<string, number | null>[] = [];
  const t0 = Date.now();
  for (const r of rows) {
    const f = extractInstantFeatures(readPcm(r.key), SR);
    out.push({ key: r.key as unknown as null, ...f } as Record<string, number | null>);
  }
  const p = join(WORK, "ts_features.json");
  writeFileSync(p, JSON.stringify(out));
  console.log(`${rows.length} windows in ${((Date.now() - t0) / 1000).toFixed(1)} s -> ${p}`);
}

function bench(): { msPerWindow: number; windows: number } {
  const rows = loadIndex().slice(0, 200);
  const pcms = rows.map((r) => readPcm(r.key));
  for (const p of pcms.slice(0, 20)) instantHeatScore(p, SR); // warm the JIT + caches
  const t0 = process.hrtime.bigint();
  for (const p of pcms) instantHeatScore(p, SR);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  return { msPerWindow: ms / pcms.length, windows: pcms.length };
}

function cmdEval(): void {
  const rows = loadIndex().filter((r) => r.emotion === "angry" || r.emotion === "happy");
  const rav = rows.filter((r) => r.corpus === "ravdess");
  const crema = rows
    .filter((r) => r.corpus === "cremad")
    .sort((a, b) => stableShuffleKey(a.key).localeCompare(stableShuffleKey(b.key)))
    .slice(0, 500);

  const run = (set: Row[]) => {
    const labels: number[] = [];
    const scores: number[] = [];
    let unscored = 0;
    for (const r of set) {
      const { score } = instantHeatScore(readPcm(r.key), SR);
      if (score === null) {
        unscored++;
        continue;
      }
      labels.push(r.is_angry ? 1 : 0);
      scores.push(score);
    }
    return {
      n: labels.length,
      nAngry: labels.filter((l) => l === 1).length,
      unscored,
      aucAngryVsHappy: Number(auc(labels, scores).toFixed(4)),
    };
  };

  const ravRes = run(rav);
  const cremaRes = run(crema);
  const b = bench();

  const payload = {
    _comment:
      "Measured by scripts/instant_tier_eval.ts running apps/mobile/src/live/instantTier.ts " +
      "over the real corpus audio (2 s loudest window per clip, 16 kHz). RAVDESS is the " +
      "cross-corpus number: the model was fitted on CREMA-D only and has never seen a " +
      "RAVDESS actor, room or script. Regenerate with `npx tsx scripts/instant_tier_eval.ts`.",
    generated: new Date().toISOString().slice(0, 10),
    crossCorpus: { corpus: "ravdess", ...ravRes },
    withinCorpus: { corpus: "cremad", note: "trained on this corpus; not a generalisation claim", ...cremaRes },
    latency: {
      msPerWindow: Number(b.msPerWindow.toFixed(3)),
      windowsBenched: b.windows,
      node: process.version,
      note: "node on the dev Mac; a phone JS engine is slower, budget is 40 ms",
    },
  };
  mkdirSync(dirname(EVAL_OUT), { recursive: true });
  writeFileSync(EVAL_OUT, `${JSON.stringify(payload, null, 1)}\n`);
  console.log(JSON.stringify(payload, null, 1));
  console.log(`-> ${EVAL_OUT}`);
}

const arg = process.argv[2];
if (arg === "--features") cmdFeatures();
else if (arg === "--bench") console.log(bench());
else cmdEval();
