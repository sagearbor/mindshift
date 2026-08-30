/**
 * The on-phone voice-separation orchestrator (src/live/deviceDiarization.ts)
 * with every I/O piece faked: media URL → pcm16k download → the phone's
 * ECAPA → diarizeWindows → a DeviceDiarizationEvent; cancellation; the
 * phrased failures (model missing, 413, bad audio).
 */
import * as fs from "fs";
import * as path from "path";
import { buildWavHeader, int16ToBytes } from "../src/recorder/wav";
import { parseWav, parseWav16kMono } from "../src/recorder/wavParse";
import { parseWav as parseWavNode } from "../src/live/replay/wav";
import { AUDIO_FIXTURES_DIR } from "../src/live/replay/sceneReplay";
import {
  DeviceDiarizationError,
  runDeviceDiarization,
  type DeviceDiarizationDeps,
  type DeviceDiarizationProgress,
  type LoadedEmbedder,
} from "../src/live/deviceDiarization";
import type { Embedder } from "../src/live/speakerId";

const SR = 16000;

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** Two alternating voices (120 / 180 Hz) with 1.5 s pauses, as a 16 kHz mono WAV. */
function wavClip(blocks: { voice: number; seconds: number }[]): { bytes: Uint8Array; seconds: number } {
  const rnd = lcg(5);
  const gap = 1.5;
  const total = blocks.reduce((s, b) => s + b.seconds + gap, 0);
  const pcm = new Int16Array(Math.round(total * SR));
  let t = 0;
  for (const b of blocks) {
    const a = Math.round(t * SR);
    const e = Math.round((t + b.seconds) * SR);
    const f0 = 120 + 60 * b.voice;
    for (let i = a; i < e; i++) pcm[i] = Math.round(32767 * (0.2 * Math.sin((2 * Math.PI * f0 * i) / SR) + 0.005 * (rnd() - 0.5)));
    t += b.seconds + gap;
  }
  const body = int16ToBytes(pcm);
  const header = buildWavHeader(SR, body.byteLength);
  const bytes = new Uint8Array(header.byteLength + body.byteLength);
  bytes.set(header, 0);
  bytes.set(body, header.byteLength);
  return { bytes, seconds: total };
}

/** Fake ECAPA keyed on the chunk's dominant pitch. `views` counts chunks
 *  that were NOT owned zero-offset buffers — the shape the native ORT
 *  binding mis-reads (it drops byteOffset), so it must stay 0. */
function fakeEmbedder(): Embedder & { calls: number; views: number } {
  const rnd = lcg(9);
  const e = {
    calls: 0,
    views: 0,
    async embed(pcm: Float32Array): Promise<Float32Array> {
      e.calls++;
      if (pcm.byteOffset !== 0 || pcm.byteLength !== pcm.buffer.byteLength) e.views++;
      let crossings = 0;
      for (let i = 1; i < pcm.length; i++) if ((pcm[i - 1] < 0) !== (pcm[i] < 0)) crossings++;
      const f0 = (crossings / 2) * (SR / pcm.length);
      const voice = Math.max(0, Math.min(1, Math.round((f0 - 120) / 60)));
      const v = new Float32Array(6);
      v[voice] = 1;
      for (let i = 0; i < 6; i++) v[i] += 0.05 * (rnd() - 0.5);
      return v;
    },
  };
  return e;
}

function fakeDeps(clip: { bytes: Uint8Array }, overrides: Partial<DeviceDiarizationDeps> = {}) {
  const embedder = fakeEmbedder();
  const release = jest.fn().mockResolvedValue(undefined);
  const loaded: LoadedEmbedder = { embedder, modelRev: "rev-123", modelSource: "cached", release };
  const downloadUrls: string[] = [];
  const deps: DeviceDiarizationDeps = {
    getMediaUrl: jest.fn().mockResolvedValue({ url: "https://api.test/recordings/r1/media?tk=abc", expires_in: 900 }),
    downloadBytes: jest.fn(async (url: string, onProgress) => {
      downloadUrls.push(url);
      onProgress(clip.bytes.byteLength / 2, clip.bytes.byteLength);
      onProgress(clip.bytes.byteLength, clip.bytes.byteLength);
      return clip.bytes;
    }),
    loadEmbedder: jest.fn().mockResolvedValue(loaded),
    deviceInfo: () => ({ platform: "android", osVersion: "16", model: "Pixel 10", userAgent: null }),
    now: () => new Date("2026-08-30T01:00:00.000Z"),
    ...overrides,
  };
  return { deps, embedder, release, downloadUrls };
}

describe("wavParse", () => {
  it("round-trips the recorder's canonical WAV and rejects the wrong rate", () => {
    const { bytes, seconds } = wavClip([{ voice: 0, seconds: 1 }]);
    const w = parseWav(bytes);
    expect([w.channels, w.sampleRate, w.bitsPerSample]).toEqual([1, 16000, 16]);
    expect(w.seconds).toBeCloseTo(seconds, 6);
    const f = parseWav16kMono(bytes);
    expect(f.length).toBe(w.samples.length);
    expect(Math.max(...Array.from(f.subarray(0, 16000)))).toBeGreaterThan(0.15);
    const other = new Uint8Array(bytes);
    new DataView(other.buffer).setUint32(24, 44100, true);
    expect(() => parseWav16kMono(other)).toThrow(/16 kHz mono/);
    expect(() => parseWav(new Uint8Array([1, 2, 3]))).toThrow(/RIFF/);
  });

  it("reads a real `?format=pcm16k` WAV from the server's own writer, sample for sample", () => {
    // fixtures/pcm16k_family_real_6s.wav: the first 6 s of
    // server/tests/fixtures/audio/test_recording_family_real.wav through
    // server/audio_ingest.decode_to_pcm_16k + pcm_to_wav16 (what
    // GET /recordings/{id}/media?format=pcm16k returns). The server writes
    // int16 as round(x × 32767) of the float, so it lands within 1 LSB of
    // the original int16 — never more.
    const served = fs.readFileSync(path.join(__dirname, "fixtures/pcm16k_family_real_6s.wav"));
    const source = parseWavNode(fs.readFileSync(path.join(AUDIO_FIXTURES_DIR, "test_recording_family_real.wav")));
    expect(source.sampleRate).toBe(16000);
    // Exactly what the phone hands the parser: a Uint8Array over the XHR
    // ArrayBuffer — but also a view with a non-zero offset, which must not matter.
    const bytes = new Uint8Array(served.buffer, served.byteOffset, served.byteLength);
    const w = parseWav(bytes, "pcm16k");
    expect([w.channels, w.sampleRate, w.bitsPerSample]).toEqual([1, 16000, 16]);
    expect(w.samples.length).toBe(6 * 16000);
    expect(w.seconds).toBeCloseTo(6, 6);
    const f = parseWav16kMono(bytes, "pcm16k");
    expect(f.length).toBe(6 * 16000);
    expect(f.byteOffset).toBe(0);
    let maxAbs = 0;
    let maxDiffLsb = 0;
    for (let i = 0; i < f.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(f[i]));
      maxDiffLsb = Math.max(maxDiffLsb, Math.abs(w.samples[i] - source.samples[i]));
    }
    expect(maxAbs).toBeLessThanOrEqual(1);
    expect(maxAbs).toBeGreaterThan(0.1); // real speech, not silence or a scale slip
    expect(maxDiffLsb).toBeLessThanOrEqual(1);
    for (let i = 0; i < 100; i++) expect(Math.abs(w.samples[i] - source.samples[i])).toBeLessThanOrEqual(1);
    // The float mapping is the fast loop's own: int16 / 32768.
    const k = w.samples.findIndex((s) => Math.abs(s) > 1000);
    expect(f[k]).toBeCloseTo(w.samples[k] / 32768, 7);
  });
});

describe("runDeviceDiarization", () => {
  it("downloads the pcm16k variant, runs the engine on the phone's ECAPA and shapes the diagnostics event", async () => {
    const clip = wavClip([
      { voice: 0, seconds: 8 },
      { voice: 1, seconds: 8 },
      { voice: 0, seconds: 7 },
      { voice: 1, seconds: 8 },
    ]);
    const { deps, embedder, release, downloadUrls } = fakeDeps(clip);
    const progress: DeviceDiarizationProgress[] = [];
    const run = runDeviceDiarization("r1", { deps, onProgress: (p) => progress.push(p) });
    const ev = await run.promise;

    expect(downloadUrls).toEqual(["https://api.test/recordings/r1/media?tk=abc&format=pcm16k"]);
    expect(ev.recording_id).toBe("r1");
    expect(ev.engine).toBe("B");
    expect(ev.k).toBe(2);
    expect(ev.k_eigengap).toBe(2);
    expect(ev.hop_s).toBe(0.25);
    expect(ev.window_s).toBe(1.5);
    expect(ev.windows).toBeGreaterThan(50);
    expect(ev.windows).toBe(embedder.calls);
    // Every chunk crossed the embedder seam as an owned zero-offset buffer
    // (the native binding drops byteOffset — a view would embed the clip's
    // first window every time and the run would degenerate to one vector).
    expect(embedder.views).toBe(0);
    expect(ev.mean_pairwise_cosine).not.toBeNull();
    expect(ev.mean_pairwise_cosine as number).toBeGreaterThan(0);
    expect(ev.mean_pairwise_cosine as number).toBeLessThan(0.95);
    expect(ev.windows_total).toBeGreaterThanOrEqual(ev.windows);
    expect(ev.duration_s).toBeCloseTo(clip.seconds, 3);
    expect(ev.speech_s).toBeGreaterThan(25);
    expect(ev.segments[0][0]).toBe(0);
    expect(ev.segments[ev.segments.length - 1][1]).toBeCloseTo(clip.seconds, 2);
    expect(new Set(ev.segments.map((s) => s[2])).size).toBe(2);
    expect(ev.download_bytes).toBe(clip.bytes.byteLength);
    expect(ev.download_ms).toBeGreaterThanOrEqual(0);
    expect(ev.embed_ms_mean).not.toBeNull();
    expect(ev.embed_ms_p90).not.toBeNull();
    expect(ev.total_ms).toBeGreaterThanOrEqual(ev.cluster_ms);
    expect(ev.model_rev).toBe("rev-123");
    expect(ev.model_source).toBe("cached");
    expect(ev.device.model).toBe("Pixel 10");
    expect(ev.created_at).toBe("2026-08-30T01:00:00.000Z");
    expect(release).toHaveBeenCalledTimes(1);

    const phases = progress.map((p) => p.phase);
    expect(phases[0]).toBe("model");
    expect(phases).toEqual(expect.arrayContaining(["download", "gate", "embed", "cluster", "smooth"]));
    const dl = progress.filter((p) => p.phase === "download");
    expect(dl[dl.length - 1].fraction).toBe(1);
    expect(dl[dl.length - 1].detail).toMatch(/downloading audio .* MB \/ .* MB/);
  });

  it("refuses honestly when the model is not on the phone — and never downloads audio", async () => {
    const clip = wavClip([{ voice: 0, seconds: 3 }]);
    const { deps } = fakeDeps(clip, { loadEmbedder: jest.fn().mockResolvedValue({ unavailable: "offline and no cached model" }) });
    const run = runDeviceDiarization("r1", { deps });
    await expect(run.promise).rejects.toMatchObject({ code: "model-unavailable", message: expect.stringContaining("offline and no cached model") });
    expect(deps.getMediaUrl).not.toHaveBeenCalled();
    expect(deps.downloadBytes).not.toHaveBeenCalled();
  });

  it("passes the server's 413 through as a phrased 'too long' failure and releases the model", async () => {
    const clip = wavClip([{ voice: 0, seconds: 3 }]);
    const { deps, release } = fakeDeps(clip, {
      downloadBytes: jest.fn().mockRejectedValue(new DeviceDiarizationError("this recording is longer than 30 minutes — too big to separate on the phone", "too-long")),
    });
    await expect(runDeviceDiarization("r1", { deps }).promise).rejects.toMatchObject({ code: "too-long" });
    expect(release).toHaveBeenCalled();
  });

  it("reports unreadable audio and a failed link lookup with their reasons", async () => {
    const clip = wavClip([{ voice: 0, seconds: 3 }]);
    const bad = fakeDeps(clip, { downloadBytes: jest.fn().mockResolvedValue(new Uint8Array([1, 2, 3, 4])) });
    await expect(runDeviceDiarization("r1", { deps: bad.deps }).promise).rejects.toMatchObject({ code: "bad-audio" });
    const noLink = fakeDeps(clip, { getMediaUrl: jest.fn().mockRejectedValue(new Error("API error: 404")) });
    await expect(runDeviceDiarization("r1", { deps: noLink.deps }).promise).rejects.toMatchObject({ code: "http", message: expect.stringContaining("404") });
  });

  it("cancel() stops the run between windows with a 'cancelled' failure", async () => {
    const clip = wavClip([
      { voice: 0, seconds: 8 },
      { voice: 1, seconds: 8 },
    ]);
    const { deps, embedder, release } = fakeDeps(clip);
    let run: ReturnType<typeof runDeviceDiarization> | null = null;
    const inner = embedder.embed.bind(embedder);
    embedder.embed = async (pcm: Float32Array, sr: number) => {
      if (embedder.calls === 3) run?.cancel();
      return inner(pcm, sr);
    };
    run = runDeviceDiarization("r1", { deps });
    await expect(run.promise).rejects.toMatchObject({ code: "cancelled" });
    expect(embedder.calls).toBeLessThan(10);
    expect(release).toHaveBeenCalled();
  });
});
