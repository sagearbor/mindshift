/**
 * Is the single-mic overlap probe good enough to NUDGE on?
 *
 * `overlapProbe.ts` has shipped DARK since 2026-09-05, with its own file
 * saying why: "nothing nudges on them until real sessions show the numbers
 * are trustworthy — false steamroll nudges in a family conversation would
 * cost more trust than the feature earns", and "when validated, the longest
 * run maps straight onto the shared `interrupting` ladder". This is that
 * validation, and it does not need real sessions — it needs real OVERLAP.
 *
 * The corpus: AMI ES2002a, a 21-minute 4-person meeting (CC BY 4.0). AMI
 * publishes each participant's own HEADSET track alongside the mixed room
 * mic, sample-aligned. That is the whole trick: running an energy VAD on each
 * headset separately gives true "who was talking when" — and therefore true
 * OVERLAP — with no hand annotation at all. Measured that way this meeting is
 * 8.2% overlapped, squarely inside the published range for AMI, which is the
 * evidence that the ground truth is sane before anything is scored against it.
 *
 * The probe then runs over the MIXED mic — one microphone, exactly what a
 * phone on a table hears — and every window is classified by the app's OWN
 * `classifyWindow`, not a reimplementation.
 *
 * Skips honestly without the corpus:
 *   python tmp/ami-overlap/fetch.py   (see that script; ~160 MB, gitignored)
 */
import * as fs from "fs";
import * as path from "path";
import { EcapaEmbedder, l2Normalize, cosine } from "../src/live/speakerId";
import {
  classifyWindow,
  OVERLAP_HOP_SECONDS,
  OVERLAP_MARGIN,
  OVERLAP_MIN_SCORE,
  OVERLAP_WINDOW_SECONDS,
  type WindowVoice,
} from "../src/live/overlapProbe";
import { readCorpusWavTo16k } from "../src/live/replay/corpusWav";
import { DEFAULT_REPLAY_OPTIONS, findEcapaModel, loadModels, REPO_ROOT } from "../src/live/replay/sceneReplay";

const DIR = path.join(REPO_ROOT, "tmp", "ami-overlap");
const MIX = path.join(REPO_ROOT, "tmp", "testaudio", "ES2002a.Mix-Headset.wav");
const GT = path.join(DIR, "ground_truth.json");
const have = Boolean(findEcapaModel()) && fs.existsSync(GT) && fs.existsSync(MIX);
const maybe = have ? describe : describe.skip;

const SR = 16000;
/** Minutes of the meeting to score. The whole 21 would be ~2500 ECAPA passes
 *  per speaker; this is enough for a stable rate and keeps the suite usable. */
const SLICE_START_S = 240;
const SLICE_END_S = 660;

interface GroundTruth {
  hop_s: number;
  speakers: Record<string, [number, number][]>;
}

/** Seconds of `spk` speech inside [a,b), and seconds where ANYONE else also
 *  spoke — the two numbers a window is judged on. */
function coverage(gt: GroundTruth, spk: string, a: number, b: number) {
  const overlapOf = (iv: [number, number][]) =>
    iv.reduce((s, [x, y]) => s + Math.max(0, Math.min(b, y) - Math.max(a, x)), 0);
  const self = overlapOf(gt.speakers[spk]);
  let other = 0;
  for (const [k, iv] of Object.entries(gt.speakers)) {
    if (k === spk) continue;
    other = Math.max(other, overlapOf(iv));
  }
  return { self, other };
}

maybe("single-mic overlap probe vs real overlapping speech (AMI ES2002a)", () => {
  const gt: GroundTruth = have ? JSON.parse(fs.readFileSync(GT, "utf8")) : { hop_s: 0, speakers: {} };
  const speakers = Object.keys(gt.speakers);
  let rows: {
    spk: string;
    start: number;
    self: number;
    otherMax: number;
    voice: WindowVoice;
    trueOverlap: boolean;
  }[] = [];

  beforeAll(async () => {
    const models = await loadModels({ ...DEFAULT_REPLAY_OPTIONS, ortFactory: null, ecapaPath: findEcapaModel() });
    const emb = models.embedder as EcapaEmbedder;
    const mix = readCorpusWavTo16k(MIX);

    // Enrol each speaker from their OWN headset — clean, single-voice audio,
    // which is what a real enrolment is. Pooled from their longest turns.
    const prints: Record<string, Float32Array> = {};
    for (let i = 0; i < speakers.length; i++) {
      const spk = speakers[i];
      const head = readCorpusWavTo16k(path.join(DIR, `ES2002a.Headset-${i}.wav`));
      const longest = [...gt.speakers[spk]].sort((x, y) => y[1] - y[0] - (x[1] - x[0])).slice(0, 6);
      const vecs: Float32Array[] = [];
      for (const [a, b] of longest) {
        const seg = head.subarray(Math.round(a * SR), Math.round(Math.min(b, a + 8) * SR));
        if (seg.length < SR) continue;
        vecs.push(await emb.embed(seg, SR));
      }
      const c = new Float32Array(vecs[0].length);
      for (const v of vecs) for (let k = 0; k < v.length; k++) c[k] += v[k] / vecs.length;
      prints[spk] = l2Normalize(c);
    }

    // Slide the probe's own window/hop across the MIXED mic.
    for (let t = SLICE_START_S; t + OVERLAP_WINDOW_SECONDS <= SLICE_END_S; t += OVERLAP_HOP_SECONDS) {
      const seg = mix.subarray(Math.round(t * SR), Math.round((t + OVERLAP_WINDOW_SECONDS) * SR));
      const v = await emb.embed(seg, SR);
      const scores: Record<string, number> = {};
      for (const spk of speakers) scores[spk] = cosine(v, prints[spk]);
      for (const spk of speakers) {
        const cov = coverage(gt, spk, t, t + OVERLAP_WINDOW_SECONDS);
        // Only judge windows this speaker actually dominates — the probe only
        // ever runs inside the coached user's OWN turn.
        const selfLed = cov.self >= OVERLAP_WINDOW_SECONDS * 0.5;
        if (!selfLed) continue;
        const otherMax = Math.max(...speakers.filter((s) => s !== spk).map((s) => scores[s]));
        rows.push({
          spk,
          start: t,
          self: scores[spk],
          otherMax,
          voice: classifyWindow({ start: t, self: scores[spk], otherMax }),
          // A real cut-in: someone else speaking for a third of the window.
          trueOverlap: cov.other >= OVERLAP_WINDOW_SECONDS / 3,
        });
      }
    }
  }, 1_800_000);

  it("has enough judged windows, from a meeting with realistic overlap", () => {
    expect(rows.length).toBeGreaterThan(200);
    const pos = rows.filter((r) => r.trueOverlap).length;
    console.log(
      `  judged ${rows.length} self-led windows across ${speakers.length} speakers; ` +
        `${pos} (${((pos / rows.length) * 100).toFixed(1)}%) contain real overlap`,
    );
    expect(pos).toBeGreaterThan(20);
  });

  it("scores the SHIPPED rule: does 'mixed' actually mean two people talking?", () => {
    const mixed = rows.filter((r) => r.voice === "mixed");
    const tp = mixed.filter((r) => r.trueOverlap).length;
    const fp = mixed.length - tp;
    const fn = rows.filter((r) => r.trueOverlap && r.voice !== "mixed").length;
    const precision = mixed.length ? tp / mixed.length : NaN;
    const recall = tp + fn ? tp / (tp + fn) : NaN;
    const byVoice: Record<string, number> = {};
    for (const r of rows) byVoice[r.voice] = (byVoice[r.voice] ?? 0) + 1;
    console.log(
      `  probe verdicts ${JSON.stringify(byVoice)}\n` +
        `  mixed=${mixed.length}  true-overlap ${tp}  false ${fp}  missed ${fn}\n` +
        `  PRECISION ${(precision * 100).toFixed(1)}%  (of the windows it calls "mixed", how many really are)\n` +
        `  RECALL    ${(recall * 100).toFixed(1)}%  (of the real overlaps, how many it catches)\n` +
        `  thresholds: margin ${OVERLAP_MARGIN}, min score ${OVERLAP_MIN_SCORE}`,
    );
    // Reported, not gated — the bar for SHIPPING it is decided below, on
    // precision, because a false "you talked over them" is the costly error.
    expect(rows.length).toBeGreaterThan(0);
  });

  it("sweeps the margin: is the signal there at ALL, at any threshold?", () => {
    // The shipped rule is one point on a curve. If no point on the curve is
    // usable, the probe is not a threshold problem — it is the wrong feature,
    // and that is worth knowing before anyone tunes it further.
    console.log("  margin  minScore   mixed  precision  recall");
    const results: { margin: number; minScore: number; precision: number; recall: number; n: number }[] = [];
    for (const minScore of [0.0, 0.1, 0.2]) {
      for (const margin of [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]) {
        const mixed = rows.filter(
          (r) => Math.max(r.self, r.otherMax) >= minScore && Math.abs(r.self - r.otherMax) < margin,
        );
        const tp = mixed.filter((r) => r.trueOverlap).length;
        const total = rows.filter((r) => r.trueOverlap).length;
        const precision = mixed.length ? tp / mixed.length : 0;
        const recall = total ? tp / total : 0;
        results.push({ margin, minScore, precision, recall, n: mixed.length });
        console.log(
          `  ${margin.toFixed(2)}    ${minScore.toFixed(2)}      ${String(mixed.length).padStart(4)}    ` +
            `${(precision * 100).toFixed(0).padStart(4)}%     ${(recall * 100).toFixed(0).padStart(4)}%`,
        );
      }
    }
    // The base rate — what a coin that always said "mixed" would score.
    const base = rows.filter((r) => r.trueOverlap).length / rows.length;
    const best = results.filter((r) => r.n >= 10).sort((a, b) => b.precision - a.precision)[0];
    console.log(
      `  base rate ${(base * 100).toFixed(1)}% — a rule is only worth anything ABOVE this.\n` +
        `  best precision with >=10 calls: ${(best.precision * 100).toFixed(0)}% at margin ${best.margin} / minScore ${best.minScore} (recall ${(best.recall * 100).toFixed(0)}%)`,
    );
    expect(results.length).toBeGreaterThan(0);
  });

  it("scores the LADDER, which is what would actually nudge: sustained runs", () => {
    // The product never asks "was this 1.5 s window overlapped". It asks "did
    // a sustained talk-over happen" — INTERRUPT_LEVELS starts at 2 s, i.e. a
    // run of consecutive mixed windows. Isolated noise should not cluster, so
    // runs can be far more precise than the windows they are made of. If they
    // are not, the probe cannot drive ✂️ at any setting.
    console.log("  margin  minRun(s)  runs  true-ovl  precision");
    const rowsBySpk = new Map<string, typeof rows>();
    for (const r of rows) {
      if (!rowsBySpk.has(r.spk)) rowsBySpk.set(r.spk, []);
      rowsBySpk.get(r.spk)!.push(r);
    }
    let best = { margin: 0, minRun: 0, runs: 0, tp: 0, precision: 0 };
    for (const margin of [0.15, 0.2, 0.25, 0.3]) {
      for (const minRun of [2, 3, 4]) {
        const need = Math.round(minRun / OVERLAP_HOP_SECONDS);
        let runs = 0;
        let tp = 0;
        for (const list of rowsBySpk.values()) {
          const sorted = [...list].sort((a, b) => a.start - b.start);
          let run: typeof rows = [];
          const flush = () => {
            if (run.length >= need) {
              runs += 1;
              // A run is right if MOST of it really was overlapped.
              if (run.filter((r) => r.trueOverlap).length >= run.length / 2) tp += 1;
            }
            run = [];
          };
          for (let i = 0; i < sorted.length; i++) {
            const contiguous = i === 0 || Math.abs(sorted[i].start - sorted[i - 1].start - OVERLAP_HOP_SECONDS) < 1e-6;
            const isMixed = Math.abs(sorted[i].self - sorted[i].otherMax) < margin;
            if (isMixed && contiguous) run.push(sorted[i]);
            else {
              flush();
              if (isMixed) run = [sorted[i]];
            }
          }
          flush();
        }
        const precision = runs ? tp / runs : 0;
        console.log(
          `  ${margin.toFixed(2)}      ${minRun}        ${String(runs).padStart(4)}    ${String(tp).padStart(4)}       ${(precision * 100).toFixed(0).padStart(4)}%`,
        );
        if (runs >= 5 && precision > best.precision) best = { margin, minRun, runs, tp, precision };
      }
    }
    console.log(
      `  BEST usable ladder point: margin ${best.margin}, min run ${best.minRun}s -> ` +
        `${best.runs} nudges, ${(best.precision * 100).toFixed(0)}% right`,
    );
    // THE SHIPPING DECISION. A false "you talked over them" is the costly
    // error — it accuses someone of something they did not do, mid-argument.
    // 80% is the bar: one wrong accusation in five is already a lot.
    const SHIP_BAR = 0.8;
    console.log(
      best.precision >= SHIP_BAR
        ? `  -> SHIPPABLE (>= ${SHIP_BAR * 100}%)`
        : `  -> NOT SHIPPABLE: best is ${(best.precision * 100).toFixed(0)}%, bar is ${SHIP_BAR * 100}%. Stays dark.`,
    );
    expect(best.runs).toBeGreaterThan(0);
  });
});
