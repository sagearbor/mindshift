/**
 * The on-phone voice-separation orchestrator end to end on REAL pieces: the
 * served ECAPA export under onnxruntime-node, a real two-voice fixture as
 * the server's `?format=pcm16k` WAV, the production deviceDiarization →
 * diarizeWindows → EcapaEmbedder path — asserting k and that the window
 * embeddings are not one vector. Skipped when no ECAPA export is on this
 * machine (see replay/sceneReplay.ts findEcapaModel).
 *
 * Regression for the Pixel 10's first run (2026-08-30): "1 voice found
 * (eigengap 1)" on every recording because onnxruntime-react-native's JSI
 * binding creates the input tensor from `data.buffer` and ignores the typed
 * array's byteOffset — every `pcm.subarray(...)` window embedded the clip's
 * first 1.5 s. The second test runs the SAME orchestration through a session
 * wrapper that drops byteOffset exactly like the phone's binding, and must
 * still find two voices.
 */
import * as fs from "fs";
import * as path from "path";
import { runDeviceDiarization, meanPairwiseCosine, type DeviceDiarizationDeps, type LoadedEmbedder } from "../src/live/deviceDiarization";
import { EcapaEmbedder } from "../src/live/speakerId";
import { nodeOrtSessionFactory } from "../src/live/testing/ortNode";
import { AUDIO_FIXTURES_DIR, findEcapaModel } from "../src/live/replay/sceneReplay";
import type { OnnxSession } from "../src/live/ort";

const FIXTURE = path.join(AUDIO_FIXTURES_DIR, "test_recording_family_real.wav");
const ecapaPath = findEcapaModel();
const maybe = ecapaPath && fs.existsSync(FIXTURE) ? it : it.skip;

/** onnxruntime-react-native 1.24.3 (cpp/TensorUtils.cpp): `data = tensor.data.buffer.data()`. */
function phoneFaithful(session: OnnxSession): OnnxSession {
  return {
    ...session,
    run(feeds) {
      const rebased: typeof feeds = {};
      for (const [name, t] of Object.entries(feeds)) {
        const d = t.data as Float32Array;
        rebased[name] = { ...t, data: new Float32Array(d.buffer, 0, d.length) };
      }
      return session.run(rebased);
    },
  };
}

function deps(session: OnnxSession): DeviceDiarizationDeps {
  const loaded: LoadedEmbedder = { embedder: new EcapaEmbedder(session), modelRev: path.basename(ecapaPath ?? ""), modelSource: "test", release: async () => {} };
  return {
    getMediaUrl: async () => ({ url: "https://api.test/recordings/r1/media?tk=abc", expires_in: 900 }),
    // The server fixture IS a 16 kHz mono int16 WAV, i.e. the pcm16k shape
    // (within 1 LSB of the server's transcode — deviceDiarization.test.ts).
    downloadBytes: async () => new Uint8Array(fs.readFileSync(FIXTURE)),
    loadEmbedder: async () => loaded,
    deviceInfo: () => ({ platform: "android", osVersion: "16", model: "Pixel 10", userAgent: null }),
    now: () => new Date("2026-08-30T01:00:00.000Z"),
  };
}

describe("runDeviceDiarization on the real ECAPA export (family_real, two voices)", () => {
  jest.setTimeout(120000);
  let session: OnnxSession | null = null;

  beforeAll(async () => {
    if (ecapaPath) session = await nodeOrtSessionFactory()(ecapaPath);
  });
  afterAll(async () => {
    await session?.release();
  });

  maybe("finds two voices with distinct window embeddings (the replay's numbers)", async () => {
    const ev = await runDeviceDiarization("r1", { deps: deps(session!) }).promise;
    expect(ev.windows).toBe(110);
    expect(ev.windows_total).toBe(113);
    expect(ev.k).toBe(2);
    expect(ev.k_eigengap).toBe(2);
    expect(ev.mean_pairwise_cosine).not.toBeNull();
    expect(ev.mean_pairwise_cosine as number).toBeLessThan(0.5); // 0.187 on this machine
    expect(ev.mean_pairwise_cosine as number).toBeGreaterThan(-0.5);
    expect(ev.eigenvalues[0]).toBeGreaterThan(20);
    expect(ev.eigenvalues[1]).toBeGreaterThan(15); // the second voice's eigenvalue, not ~0
    expect(new Set(ev.segments.map((s) => s[2])).size).toBe(2);
  });

  maybe("still finds two voices through a session that drops byteOffset like the phone's binding", async () => {
    const ev = await runDeviceDiarization("r1", { deps: deps(phoneFaithful(session!)) }).promise;
    expect(ev.k).toBe(2);
    expect(ev.k_eigengap).toBe(2);
    expect(ev.mean_pairwise_cosine as number).toBeLessThan(0.5);
  });

  maybe("subarray views through the embedder no longer collapse — float32Tensor copies them before ORT sees them", async () => {
    // What the phone did before the fix, on the same model and audio: every
    // window's tensor pointed at the buffer's start. The seam (ort.ts
    // float32Tensor) now hands ORT an owned zero-offset array, so even a
    // byteOffset-dropping session embeds the right audio for a view.
    const embedder = new EcapaEmbedder(phoneFaithful(session!));
    const pcm = new Float32Array(16000 * 6);
    const src = fs.readFileSync(FIXTURE);
    for (let i = 0; i < pcm.length; i++) pcm[i] = src.readInt16LE(44 + i * 2) / 32768;
    const views = [0, 8000, 32000, 48000].map((s) => pcm.subarray(s, s + 24000));
    const owned = views.map((v) => v.slice());
    const viewEmbs = [];
    const ownedEmbs = [];
    for (const v of views) viewEmbs.push(await embedder.embed(v, 16000));
    for (const o of owned) ownedEmbs.push(await embedder.embed(o, 16000));
    expect(meanPairwiseCosine(viewEmbs) as number).toBeLessThan(0.95);
    expect(meanPairwiseCosine(ownedEmbs) as number).toBeLessThan(0.95);
    // And the two paths agree window for window.
    for (let i = 0; i < views.length; i++) {
      let dot = 0;
      for (let j = 0; j < viewEmbs[i].length; j++) dot += viewEmbs[i][j] * ownedEmbs[i][j];
      expect(dot).toBeGreaterThan(0.999);
    }
  });
});
