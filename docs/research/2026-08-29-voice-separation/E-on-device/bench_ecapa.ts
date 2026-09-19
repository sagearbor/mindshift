/**
 * Step 3 of E: what approach B's window embeddings would cost on device.
 *
 * Times the SAME ECAPA ONNX export the phone downloads (GET /models/ecapa.onnx
 * -> server/.ecapa_cache/ecapa_<rev>.onnx) under onnxruntime-node on this
 * Mac's CPU: a 1.5 s window at 16 kHz, batch 1, 100 timed runs after 10
 * warm-ups, on real speech (sliding windows of family_real so no two runs see
 * identical input). Repeated with intra-op threads = 1 (one big core) and
 * the ORT default (all cores) to bracket what a phone's CPU EP would see.
 * Silero VAD (the loop's per-32 ms-frame cost) is timed the same way for a
 * "what the loop already pays" baseline. Extrapolations to windows-per-minute
 * at 0.25 s / 0.5 s hop are in the output; the Pixel factor is NOT measured
 * here (see README).
 *
 * Run from apps/mobile:
 *   npx tsx ../../docs/research/2026-08-29-voice-separation/E-on-device/bench_ecapa.ts
 */
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { findEcapaModel, SILERO_PATH } from "../../../../apps/mobile/src/live/replay/sceneReplay";
import { int16ToFloat32, readWav16kMono } from "../../../../apps/mobile/src/live/replay/wav";

const HERE = __dirname;
const REPO = path.resolve(HERE, "../../../..");
const SR = 16000;

// eslint-disable-next-line @typescript-eslint/no-require-imports
const ort = require("onnxruntime-node") as typeof import("onnxruntime-node");

interface Stats {
  n: number;
  mean_ms: number;
  median_ms: number;
  p90_ms: number;
  min_ms: number;
  max_ms: number;
}

function stats(ms: number[]): Stats {
  const s = [...ms].sort((a, b) => a - b);
  const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  const r = (x: number) => Math.round(x * 100) / 100;
  return { n: s.length, mean_ms: r(s.reduce((a, b) => a + b, 0) / s.length), median_ms: r(q(0.5)), p90_ms: r(q(0.9)), min_ms: r(s[0]), max_ms: r(s[s.length - 1]) };
}

async function timeEcapa(session: InstanceType<typeof ort.InferenceSession>, pcm: Float32Array, windowSec: number, warm: number, runs: number, hopSec = 0.25): Promise<Stats> {
  const win = Math.round(windowSec * SR);
  const hop = Math.round(hopSec * SR);
  const maxStart = pcm.length - win;
  const feed = (i: number) => {
    const start = (i * hop) % Math.max(1, maxStart);
    return { waveform: new ort.Tensor("float32", pcm.slice(start, start + win), [1, win]) };
  };
  for (let i = 0; i < warm; i++) await session.run(feed(i));
  const ms: number[] = [];
  for (let i = 0; i < runs; i++) {
    const f = feed(warm + i);
    const t0 = performance.now();
    await session.run(f);
    ms.push(performance.now() - t0);
  }
  return stats(ms);
}

async function timeSilero(session: InstanceType<typeof ort.InferenceSession>, pcm: Float32Array, warm: number, runs: number): Promise<Stats> {
  let state = new Float32Array(2 * 128);
  const input = new Float32Array(64 + 512);
  const ms: number[] = [];
  for (let i = 0; i < warm + runs; i++) {
    const off = (i * 512) % (pcm.length - 576);
    input.set(pcm.subarray(off, off + 576));
    const feeds = {
      input: new ort.Tensor("float32", input, [1, 576]),
      state: new ort.Tensor("float32", state, [2, 1, 128]),
      sr: new ort.Tensor("int64", BigInt64Array.from([BigInt(SR)]), []),
    };
    const t0 = performance.now();
    const out = await session.run(feeds);
    const dt = performance.now() - t0;
    state = Float32Array.from(out.stateN.data as Float32Array);
    if (i >= warm) ms.push(dt);
  }
  return stats(ms);
}

async function main() {
  const ecapaPath = findEcapaModel();
  if (!ecapaPath) throw new Error("no ECAPA export");
  const wav = readWav16kMono(path.join(REPO, "server/tests/fixtures/audio/test_recording_family_real.wav"));
  const pcm = int16ToFloat32(wav);
  const cpus = os.cpus();
  const machine = { platform: `${os.platform()} ${os.release()}`, arch: os.arch(), cpu: cpus[0]?.model ?? "?", cores: cpus.length, node: process.version, onnxruntime_node: (require("onnxruntime-node/package.json") as { version: string }).version };
  console.log(JSON.stringify(machine));

  const result: Record<string, unknown> = { machine, ecapa_model: path.basename(ecapaPath), ecapa_model_bytes: fs.statSync(ecapaPath).size, window_sec: 1.5, sample_rate: SR };
  for (const threads of [1, 0]) {
    const label = threads === 1 ? "intra_op_1_thread" : "intra_op_default";
    const opts = threads === 1 ? { intraOpNumThreads: 1, interOpNumThreads: 1 } : {};
    const session = await ort.InferenceSession.create(ecapaPath, opts);
    const w15 = await timeEcapa(session, pcm, 1.5, 10, 100);
    const other: Record<string, Stats> = {};
    for (const sec of [0.5, 1.0, 3.0, 5.0, 10.0]) other[`${sec}s`] = await timeEcapa(session, pcm, sec, 3, 20, 1.0);
    const perMin = (hop: number) => {
      const windows = Math.round(60 / hop);
      return { windows: windows, seconds_of_compute_mean: Math.round((windows * w15.mean_ms) / 10) / 100, seconds_of_compute_p90: Math.round((windows * w15.p90_ms) / 10) / 100 };
    };
    result[label] = {
      ecapa_1p5s_window: w15,
      ecapa_other_windows: other,
      per_minute_of_audio: { hop_0p25s: perMin(0.25), hop_0p5s: perMin(0.5) },
    };
    console.log(`[${label}] ECAPA 1.5 s: ${JSON.stringify(w15)}`);
    for (const [k, v] of Object.entries(other)) console.log(`[${label}] ECAPA ${k}: mean ${v.mean_ms} ms median ${v.median_ms}`);
    if (threads === 1) {
      const silero = await ort.InferenceSession.create(SILERO_PATH, opts);
      const sv = await timeSilero(silero, pcm, 20, 200);
      const framesPerMin = Math.round(60 / (512 / SR));
      result.silero_1_thread = { per_512_frame: sv, frames_per_minute: framesPerMin, seconds_of_compute_per_minute_mean: Math.round((framesPerMin * sv.mean_ms) / 10) / 100 };
      console.log(`[silero 1 thread] per 32 ms frame: ${JSON.stringify(sv)} -> ${framesPerMin} frames/min = ${result.silero_1_thread && (result.silero_1_thread as { seconds_of_compute_per_minute_mean: number }).seconds_of_compute_per_minute_mean} s compute/min`);
      await silero.release();
    }
    await session.release();
  }
  fs.writeFileSync(path.join(HERE, "bench_ecapa.json"), JSON.stringify(result, null, 1) + "\n");
  console.log("wrote bench_ecapa.json");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
