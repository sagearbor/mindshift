/**
 * Step 2 of E: run the phone's REAL live speaker clustering over the bake-off
 * fixtures and emit `[[start, end, label], ...]` predictions for score.py.
 *
 * Three segmentations, one labeler (apps/mobile/src/live/speakerId.ts
 * `SpeakerLabeler`, no enrolled people — the bake-off approaches are all
 * unsupervised, so this is the like-for-like point):
 *
 *   live    the whole shipped loop through apps/mobile's replay harness:
 *           100 ms frames -> Silero VAD (silero_vad.onnx) -> StreamingSegmenter
 *           (merge gap 0.3 s, min 0.6 s) -> last <=10 s of the turn -> ECAPA
 *           ONNX (server/.ecapa_cache export, onnxruntime-node) -> labeler
 *           @ CLUSTER_THRESHOLD 0.48 with the 1.5 s founding guard.
 *   gt      the ground-truth utterance boundaries fed as the segments
 *           (what the loop would see with a perfect segmenter).
 *   energy  the phone's fallback energy VAD (`energySpeechSegments`:
 *           -45 dBFS floor, 0.25 s frames, same merge/min rules).
 *
 * For every segmentation the (start, end, seconds, embedding) sequence is
 * saved and re-labelled offline at 0.48 (must reproduce `live` exactly —
 * asserted) and at 0.55 (the pre-2026-08-26 value).
 *
 * Run from apps/mobile (node resolves onnxruntime-node from the repo root):
 *   cd apps/mobile && npx tsx ../../docs/research/2026-08-29-voice-separation/E-on-device/replay_live.ts
 */
import * as fs from "fs";
import * as path from "path";
import {
  DEFAULT_REPLAY_OPTIONS,
  findEcapaModel,
  loadModels,
  loadScene,
  replayScene,
  type LoadedModels,
} from "../../../../apps/mobile/src/live/replay/sceneReplay";
import {
  CLUSTER_THRESHOLD,
  MATCH_THRESHOLD,
  MIN_CLUSTER_SECONDS,
  SpeakerLabeler,
  type Embedder,
} from "../../../../apps/mobile/src/live/speakerId";
import { MAX_EMBED_SECONDS } from "../../../../apps/mobile/src/live/fastLoop";
import { energySpeechSegments } from "../../../../apps/mobile/src/live/segmenter";

const HERE = __dirname;
const REPO = path.resolve(HERE, "../../../..");
const TMP = path.join(REPO, "tmp/e-on-device");
const SR = 16000;
const THRESHOLDS = [CLUSTER_THRESHOLD, 0.55];

interface Fixture {
  scene: string;
  meta: string;
  gt: [number, number, string | string[]][];
  k_true: number;
  owner: string | null;
  seconds: number;
}
interface SegRec {
  start: number;
  end: number;
  seconds: number;
  embedSeconds: number;
  embedding: number[];
}

/** Records every embedding the loop asks for, in call order (one per turn). */
class RecordingEmbedder implements Embedder {
  readonly records: { seconds: number; emb: Float32Array; ms: number }[] = [];
  constructor(private readonly inner: Embedder) {}
  async embed(pcm: Float32Array, sampleRate: number): Promise<Float32Array> {
    const t0 = performance.now();
    const emb = await this.inner.embed(pcm, sampleRate);
    this.records.push({ seconds: pcm.length / sampleRate, emb, ms: performance.now() - t0 });
    return emb;
  }
}

function relabel(seq: SegRec[], threshold: number): string[] {
  const labeler = new SpeakerLabeler([], MATCH_THRESHOLD, threshold, MIN_CLUSTER_SECONDS);
  return seq.map((s) => labeler.label(Float32Array.from(s.embedding), s.seconds).speaker);
}

/** Drop "Unknown" (no identity claimed) — those frames score as unlabelled. */
function toPred(seq: SegRec[], labels: string[]): [number, number, string][] {
  const out: [number, number, string][] = [];
  seq.forEach((s, i) => {
    if (labels[i] !== "Unknown") out.push([round(s.start), round(s.end), labels[i]]);
  });
  return out;
}
const round = (x: number) => Math.round(x * 1000) / 1000;

async function embedSpans(
  embedder: Embedder,
  pcmF32: Float32Array,
  spans: { start: number; end: number }[],
): Promise<SegRec[]> {
  const out: SegRec[] = [];
  const maxSamples = MAX_EMBED_SECONDS * SR;
  for (const sp of spans) {
    const a = Math.round(sp.start * SR);
    const b = Math.min(pcmF32.length, Math.round(sp.end * SR));
    let pcm = pcmF32.subarray(a, b);
    // Same tail cap the loop applies (fastLoop.finalizeTurn).
    if (pcm.length > maxSamples) pcm = pcm.subarray(pcm.length - maxSamples);
    const emb = await embedder.embed(pcm, SR);
    out.push({ start: sp.start, end: sp.end, seconds: sp.end - sp.start, embedSeconds: pcm.length / SR, embedding: Array.from(emb) });
  }
  return out;
}

async function main() {
  const fixtures = JSON.parse(fs.readFileSync(path.join(TMP, "fixtures.json"), "utf8")) as Record<string, Fixture>;
  const ecapaPath = findEcapaModel();
  if (!ecapaPath) throw new Error("no ECAPA export on this machine");
  const models: LoadedModels = await loadModels({ ...DEFAULT_REPLAY_OPTIONS, ortFactory: null, ecapaPath });
  if (!models.embedder) throw new Error("ECAPA failed to load");
  console.log(`ECAPA: ${ecapaPath}`);
  console.log(`labeler: CLUSTER_THRESHOLD=${CLUSTER_THRESHOLD} MIN_CLUSTER_SECONDS=${MIN_CLUSTER_SECONDS} MAX_EMBED_SECONDS=${MAX_EMBED_SECONDS}; thresholds tried ${THRESHOLDS.join(", ")}`);

  const summary: Record<string, unknown> = {};
  for (const [name, fx] of Object.entries(fixtures)) {
    const scene = loadScene(fx.scene, { metaPath: fx.meta, selfSpeaker: null });
    const rec = new RecordingEmbedder(models.embedder);
    const wall0 = performance.now();
    const r = await replayScene(scene, {
      mode: "therapist",
      models: { ...models, embedder: rec },
      enroll: "none",
      speakerCostMs: 0,
    });
    const wallMs = performance.now() - wall0;
    if (rec.records.length !== r.turns.length) {
      throw new Error(`${name}: ${rec.records.length} embeddings for ${r.turns.length} turns`);
    }
    const liveSeq: SegRec[] = r.turns.map((t, i) => ({
      start: t.startTime,
      end: t.endTime,
      seconds: t.endTime - t.startTime,
      embedSeconds: rec.records[i].seconds,
      embedding: Array.from(rec.records[i].emb),
    }));
    const liveLabels = r.turns.map((t) => t.speaker);
    // The offline re-run at the shipped threshold must be the live result.
    const check = relabel(liveSeq, CLUSTER_THRESHOLD);
    if (check.join("|") !== liveLabels.join("|")) {
      throw new Error(`${name}: offline relabel differs from live: ${check.join(",")} vs ${liveLabels.join(",")}`);
    }

    const gtSpans = fx.gt.map(([s, e]) => ({ start: s, end: e }));
    const gtSeq = await embedSpans(models.embedder, scene.pcmF32, gtSpans);
    const energySpans = energySpeechSegments(scene.pcm, SR);
    const energySeq = await embedSpans(models.embedder, scene.pcmF32, energySpans);

    const variants: Record<string, SegRec[]> = { live: liveSeq, gt: gtSeq, energy: energySeq };
    const preds: Record<string, string> = {};
    const info: Record<string, unknown> = {};
    for (const [vname, seq] of Object.entries(variants)) {
      for (const thr of THRESHOLDS) {
        const labels = vname === "live" && thr === CLUSTER_THRESHOLD ? liveLabels : relabel(seq, thr);
        const key = `${vname}@${thr.toFixed(2)}`;
        const file = path.join(HERE, `pred_${name}_${vname}_t${thr.toFixed(2).replace(".", "")}.json`);
        fs.writeFileSync(file, JSON.stringify(toPred(seq, labels)) + "\n");
        preds[key] = path.basename(file);
        info[key] = {
          n_segments: seq.length,
          n_unknown: labels.filter((l) => l === "Unknown").length,
          labels,
        };
      }
    }
    fs.writeFileSync(
      path.join(HERE, `segments_${name}.json`),
      JSON.stringify({ fixture: name, variants: Object.fromEntries(Object.entries(variants).map(([k, v]) => [k, v.map((s) => ({ ...s, embedding: s.embedding.map((x) => Math.round(x * 1e5) / 1e5) }))])) }),
    );
    const turnSeconds = liveSeq.map((s) => s.seconds);
    const embedSeconds = liveSeq.map((s) => s.embedSeconds);
    const ecapaMs = rec.records.map((x) => x.ms);
    summary[name] = {
      seconds: fx.seconds,
      k_true: fx.k_true,
      gt_segments: fx.gt.length,
      live: {
        turns: liveSeq.length,
        turns_per_minute: round((liveSeq.length / fx.seconds) * 60),
        turn_seconds: turnSeconds.map(round),
        embedded_seconds_total: round(embedSeconds.reduce((a, b) => a + b, 0)),
        embedded_seconds_per_minute: round((embedSeconds.reduce((a, b) => a + b, 0) / fx.seconds) * 60),
        ecapa_ms_per_turn_mac: ecapaMs.map((x) => Math.round(x)),
        vad_calls: r.wall.calls.vad ?? null,
        vad_wall_ms_mac: Math.round(r.wall.byStage.vad ?? 0),
        ecapa_wall_ms_mac: Math.round(r.wall.byStage.ecapa ?? 0),
        replay_wall_ms_mac: Math.round(wallMs),
      },
      energy_segments: energySeq.length,
      preds,
      info,
    };
    console.log(
      `${name.padEnd(15)} ${fx.seconds.toFixed(1)} s  live turns ${liveSeq.length} (${liveLabels.join(",")})  gt ${gtSeq.length}  energy ${energySeq.length}  ecapa ${Math.round(r.wall.byStage.ecapa ?? 0)} ms / vad ${Math.round(r.wall.byStage.vad ?? 0)} ms (${r.wall.calls.vad} frames)`,
    );
  }
  fs.writeFileSync(path.join(HERE, "replay_summary.json"), JSON.stringify(summary, null, 1) + "\n");
  console.log("wrote replay_summary.json");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
