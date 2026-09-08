/**
 * F — approach B's window pass + spectral clustering, as PORTED TO THE PHONE
 * (apps/mobile/src/live/diarizeWindows.ts), replayed in node over the
 * bake-off fixtures with the phone's own ECAPA path (the served ONNX export
 * under onnxruntime-node, one window per `EcapaEmbedder.embed` call — the
 * batch-1 shape the app runs) and scored with ../score.py.
 *
 * Writes:
 *   pred_<fixture>_b8.json     the port at B's k range (eigengap over 1..8, no floor)
 *   pred_<fixture>_prod6.json  the same embeddings clustered with production's
 *                              k clamp (eigengap over 1..6, floor 2 — diarize_local)
 *   replay_summary.json        windows / gate / k / eigenvalues / timings per fixture
 *   tmp/f-device-b/emb_<fixture>.json   the window embeddings (for dump_parity.py)
 *
 * Run from apps/mobile (needs tmp/e-on-device/fixtures.json from
 * ../E-on-device/prep_fixtures.py and an ECAPA export on this machine):
 *   cd apps/mobile && npx tsx ../../docs/research/2026-08-29-voice-separation/F-device-b/replay_b.ts
 */
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { AUDIO_FIXTURES_DIR, findEcapaModel } from "../../../../apps/mobile/src/live/replay/sceneReplay";
import { int16ToFloat32, readWav16kMono } from "../../../../apps/mobile/src/live/replay/wav";
import { nodeOrtSessionFactory } from "../../../../apps/mobile/src/live/testing/ortNode";
import { EcapaEmbedder } from "../../../../apps/mobile/src/live/speakerId";
import {
  clusterWindows,
  diarizeWindows,
  windowLabelRuns,
  type ClusterOptions,
  type EmbedBatch,
  type Segment,
} from "../../../../apps/mobile/src/live/diarizeWindows";

const HERE = __dirname;
const REPO = path.resolve(HERE, "../../../..");
const TMP_E = path.join(REPO, "tmp/e-on-device");
const TMP_F = path.join(REPO, "tmp/f-device-b");
const SR = 16000;

/** The two k policies scored side by side. */
const VARIANTS: Record<string, ClusterOptions> = {
  b8: { maxSpeakers: 8, minSpeakers: 1 },
  prod6: { maxSpeakers: 6, minSpeakers: 2 },
};

interface Fixture {
  scene: string;
  meta: string;
  gt: [number, number, string | string[]][];
  k_true: number;
  owner: string | null;
  seconds: number;
}

function scenePath(scene: string): string {
  return scene.includes("/") ? scene : path.join(AUDIO_FIXTURES_DIR, `test_recording_${scene}.wav`);
}

function stats(ms: number[]) {
  const s = [...ms].sort((a, b) => a - b);
  const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  const r = (x: number) => Math.round(x * 100) / 100;
  return s.length === 0
    ? { n: 0, mean_ms: null, median_ms: null, p90_ms: null }
    : { n: s.length, mean_ms: r(s.reduce((a, b) => a + b, 0) / s.length), median_ms: r(q(0.5)), p90_ms: r(q(0.9)) };
}

const round = (x: number, d = 3) => Math.round(x * 10 ** d) / 10 ** d;
const toPred = (segs: Segment[]) => segs.map(([s, e, l]) => [round(s), round(e), `Speaker ${String.fromCharCode(65 + l)}`]);

async function main() {
  const fixtures = JSON.parse(fs.readFileSync(path.join(TMP_E, "fixtures.json"), "utf8")) as Record<string, Fixture>;
  const ecapaPath = findEcapaModel();
  if (!ecapaPath) throw new Error("no ECAPA export on this machine");
  const factory = nodeOrtSessionFactory();
  const embedder = new EcapaEmbedder(await factory(ecapaPath));
  fs.mkdirSync(TMP_F, { recursive: true });
  const cpus = os.cpus();
  const machine = { platform: `${os.platform()} ${os.release()}`, arch: os.arch(), cpu: cpus[0]?.model ?? "?", cores: cpus.length, node: process.version };
  console.log(`ECAPA: ${path.basename(ecapaPath)}  ${JSON.stringify(machine)}`);

  // Batch-1 embedding, exactly what the phone does (EcapaEmbedder.embed per window).
  const embedBatch: EmbedBatch = async (chunks, sr) => {
    const out: Float32Array[] = [];
    for (const c of chunks) out.push(await embedder.embed(Float32Array.from(c), sr));
    return out;
  };

  const summary: Record<string, unknown> = { machine, ecapa_model: path.basename(ecapaPath), fixtures: {} };
  for (const [name, fx] of Object.entries(fixtures)) {
    const pcm = int16ToFloat32(readWav16kMono(scenePath(fx.scene)));
    const wall0 = performance.now();
    const r = await diarizeWindows(pcm, SR, embedBatch, VARIANTS.b8);
    const wallMs = performance.now() - wall0;
    // Re-cluster the same embeddings under each k policy (no model calls).
    const embeddings = r.embeddings;
    const variants: Record<string, unknown> = {};
    for (const [vname, opts] of Object.entries(VARIANTS)) {
      const c = vname === "b8" ? r : clusterWindows(embeddings, r.starts, r.hopSeconds, opts);
      const segs = vname === "b8" ? r.segments : windowLabelRuns(c.labels, r.starts, r.windowSeconds, 0, r.durationSeconds);
      const file = path.join(HERE, `pred_${name}_${vname}.json`);
      fs.writeFileSync(file, JSON.stringify(toPred(segs)) + "\n");
      variants[vname] = { k: c.k, k_eigengap: c.kEigengap, eigenvalues: c.eigenvalues.map((x) => round(x, 6)), n_segments: segs.length, pred_file: path.basename(file) };
    }
    fs.writeFileSync(
      path.join(TMP_F, `emb_${name}.json`),
      JSON.stringify({
        fixture: name,
        sample_rate: SR,
        window_s: r.windowSeconds,
        hop_s: r.hopSeconds,
        duration_s: r.durationSeconds,
        starts: r.starts,
        embeddings: embeddings.map((e) => Array.from(e, (x) => Math.round(x * 1e7) / 1e7)),
      }),
    );
    const es = stats(r.embedMs);
    (summary.fixtures as Record<string, unknown>)[name] = {
      seconds: round(r.durationSeconds, 2),
      k_true: fx.k_true,
      speech_seconds: round(r.speechSeconds, 2),
      gate_rms: round(r.gate, 4),
      windows_kept: r.windows,
      windows_total: r.totalWindows,
      hop_s: r.hopSeconds,
      variants,
      embed_ms_per_window: es,
      timings_ms: { gate: round(r.timings.gateMs, 1), embed: round(r.timings.embedMs), cluster: round(r.timings.clusterMs), smooth: round(r.timings.smoothMs), total: round(r.timings.totalMs), wall: round(wallMs) },
    };
    console.log(
      `${name.padEnd(15)} ${r.durationSeconds.toFixed(1)} s  windows ${r.windows}/${r.totalWindows} hop ${r.hopSeconds}  gate ${r.gate.toFixed(4)}  ` +
        `k(b8)=${r.k} eig=${r.kEigengap}  k(prod6)=${(variants.prod6 as { k: number }).k}  embed ${es.mean_ms} ms/win  cluster ${Math.round(r.timings.clusterMs)} ms  wall ${(wallMs / 1000).toFixed(1)} s`,
    );
  }
  fs.writeFileSync(path.join(HERE, "replay_summary.json"), JSON.stringify(summary, null, 1) + "\n");
  console.log("wrote replay_summary.json");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
