/**
 * "Separate voices on this phone (engine B)" — run the bake-off's
 * transcript-free window engine (diarizeWindows.ts) over a STORED
 * recording's audio, on the phone, post-hoc:
 *
 *   media_url (same token as playback) + `?format=pcm16k`
 *     -> download the 16 kHz mono WAV (progress)         [downloadBytes]
 *     -> the phone's ECAPA (same path as the live loop)  [loadEmbedder]
 *     -> diarizeWindows(pcm)                             (cancellable)
 *     -> a DeviceDiarizationEvent (segments, k, timings, model, device)
 *
 * The event is what components/DeviceDiarizationRow.tsx draws and what it
 * POSTs through the diagnostics store so `scripts/diagnostics_tail.py` can
 * print — and score against a per-second rubric — the phone's own result.
 * Every I/O piece is an injectable dep (deviceDiarization.test.ts drives it
 * with fakes); the defaults are the app's real client / ORT / fetch. Never
 * blocks the UI thread for long: the embedder awaits between windows and
 * the run can be cancelled between batches.
 *
 * Two guards learned from the first Pixel 10 run (2026-08-30, "1 voice found
 * (eigengap 1)" on every recording): the native ORT binding ignores a typed
 * array's byteOffset, so every PCM chunk crosses the seam as an OWNED buffer
 * (`ownedFloat32`); and the event carries `mean_pairwise_cosine` so a run
 * whose windows all embed to the same vector is reported as the fault it is
 * (`embeddingsLookDegenerate`), never as one voice.
 */
import type { Embedder } from "./speakerId";
import { EcapaEmbedder } from "./speakerId";
import { diarizeWindows, type DiarizeProgress, type DiarizeWindowsResult } from "./diarizeWindows";
import { revisionFromEtag } from "./modelDownload";
import { parseWav16kMono } from "../recorder/wavParse";
import { getRecordingMediaUrl, pcm16kMediaUrl, type RecordingMediaUrl, type SpeakerSegmentInput } from "../api/client";
import { unknownLabel } from "./speakerId";
import { ECAPA_REVISION, authHeaders, ecapaModelUrl } from "../api/liveSessions";
import { collectDeviceInfo, type DeviceDiarizationEvent, type DeviceInfo } from "../diagnostics/diagnostics";

export type DownloadProgress = (loadedBytes: number, totalBytes: number | null) => void;

export interface LoadedEmbedder {
  embedder: Embedder;
  modelRev: string | null;
  modelSource: string | null;
  release: () => Promise<void>;
}

export interface DeviceDiarizationDeps {
  getMediaUrl: (recordingId: string) => Promise<RecordingMediaUrl>;
  downloadBytes: (url: string, onProgress: DownloadProgress, signal: { readonly aborted: boolean }) => Promise<Uint8Array>;
  /** The ECAPA session the live loop uses; a string is the honest reason it
   *  is not available (model not downloaded, ORT missing, …). */
  loadEmbedder: () => Promise<LoadedEmbedder | { unavailable: string }>;
  deviceInfo: () => DeviceInfo;
  now: () => Date;
}

export interface DeviceDiarizationProgress {
  phase: "download" | "model" | DiarizeProgress["stage"];
  /** 0..1 when known, else null (an indeterminate step). */
  fraction: number | null;
  detail: string;
}

export interface DeviceDiarizationRun {
  promise: Promise<DeviceDiarizationEvent>;
  cancel: () => void;
}

/** A failure with a code the row can phrase (model missing vs too long vs …). */
export class DeviceDiarizationError extends Error {
  constructor(
    message: string,
    readonly code: "model-unavailable" | "too-long" | "http" | "network" | "bad-audio" | "cancelled",
  ) {
    super(message);
    this.name = "DeviceDiarizationError";
  }
}

function fmtMb(bytes: number): string {
  return `${(bytes / 1e6).toFixed(1)} MB`;
}

/**
 * A Float32Array that owns its whole buffer from offset 0 — the only shape
 * that is safe to hand to the native ORT session. onnxruntime-react-native
 * (1.24.3, cpp/TensorUtils.cpp `createOrtValueFromJSTensor`) creates the
 * Ort::Value from `data.buffer` and never reads `byteOffset` / `length`, so
 * a `subarray` view silently embeds the START of the underlying buffer.
 * onnxruntime-node and onnxruntime-web honour the view, so this is invisible
 * in the replay and in tests unless asserted. The input is returned as is
 * when it already qualifies (no copy on the common path).
 */
export function ownedFloat32(pcm: Float32Array): Float32Array {
  return pcm.byteOffset === 0 && pcm.byteLength === pcm.buffer.byteLength ? pcm : pcm.slice();
}

/** Mean cosine over every pair of window embeddings (they arrive
 *  L2-normalised from diarizeWindows); null below two windows. */
export function meanPairwiseCosine(embs: ArrayLike<number>[]): number | null {
  const n = embs.length;
  if (n < 2) return null;
  const d = embs[0].length;
  let sum = 0;
  let pairs = 0;
  for (let i = 0; i < n; i++) {
    const a = embs[i];
    for (let j = i + 1; j < n; j++) {
      const b = embs[j];
      let dot = 0;
      for (let t = 0; t < d; t++) dot += a[t] * b[t];
      sum += dot;
      pairs++;
    }
  }
  return sum / pairs;
}

/**
 * Above this mean pairwise cosine the windows are not one voice — they are
 * one VECTOR: a model or audio-path fault (the byteOffset bug above scored
 * exactly 1.0000; the two-voice family_real fixture scores 0.19). The row
 * shows such a run as a warning, never as "1 voice found".
 */
export const DEGENERATE_EMBEDDING_COSINE = 0.95;

export function embeddingsLookDegenerate(ev: Pick<DeviceDiarizationEvent, "mean_pairwise_cosine" | "windows">): boolean {
  return ev.windows >= 2 && ev.mean_pairwise_cosine !== null && ev.mean_pairwise_cosine > DEGENERATE_EMBEDDING_COSINE;
}

/**
 * The engine's result as the server's speaker timeline
 * (`postReanalyzeWithSegments`): every run `[start, end, cluster]` becomes
 * `{start, end, label}` with the cluster named the way the strip's legend
 * names it ("Speaker A" …), in time order, zero-length runs dropped. Runs
 * are already non-overlapping (diarizeWindows emits contiguous runs).
 */
export function toSpeakerSegments(ev: Pick<DeviceDiarizationEvent, "segments">): SpeakerSegmentInput[] {
  return ev.segments
    .filter(([s, e]) => e > s)
    .map(([s, e, l]) => ({ start: s, end: e, label: unknownLabel(l) }))
    .sort((a, b) => a.start - b.start || a.end - b.end);
}

/** XMLHttpRequest when the platform has it (React Native does, with progress
 *  events); a plain fetch otherwise. */
export async function defaultDownloadBytes(url: string, onProgress: DownloadProgress, signal: { readonly aborted: boolean }): Promise<Uint8Array> {
  const XHR = (globalThis as { XMLHttpRequest?: typeof XMLHttpRequest }).XMLHttpRequest;
  if (XHR) {
    return new Promise<Uint8Array>((resolve, reject) => {
      const xhr = new XHR();
      xhr.open("GET", url, true);
      xhr.responseType = "arraybuffer";
      xhr.onprogress = (ev) => onProgress(ev.loaded, ev.lengthComputable ? ev.total : null);
      xhr.onerror = () => reject(new DeviceDiarizationError("download failed (network)", "network"));
      xhr.onabort = () => reject(new DeviceDiarizationError("cancelled", "cancelled"));
      xhr.onload = () => {
        if (xhr.status === 413) reject(new DeviceDiarizationError("this recording is longer than 30 minutes — too big to separate on the phone", "too-long"));
        else if (xhr.status < 200 || xhr.status >= 300) reject(new DeviceDiarizationError(`server answered ${xhr.status}`, "http"));
        else resolve(new Uint8Array(xhr.response as ArrayBuffer));
      };
      const poll = setInterval(() => {
        if (signal.aborted) {
          clearInterval(poll);
          xhr.abort();
        }
      }, 200);
      xhr.onloadend = () => clearInterval(poll);
      xhr.send();
    });
  }
  const res = await fetch(url);
  if (res.status === 413) throw new DeviceDiarizationError("this recording is longer than 30 minutes — too big to separate on the phone", "too-long");
  if (!res.ok) throw new DeviceDiarizationError(`server answered ${res.status}`, "http");
  const buf = await res.arrayBuffer();
  onProgress(buf.byteLength, buf.byteLength);
  return new Uint8Array(buf);
}

/** The live loop's own ECAPA path (ortNative → modelDownload), lazily. */
export async function defaultLoadEmbedder(): Promise<LoadedEmbedder | { unavailable: string }> {
  let native: typeof import("./ortNative") | null = null;
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    native = require("./ortNative") as typeof import("./ortNative");
  } catch {
    native = null;
  }
  if (!native) return { unavailable: "native ONNX Runtime unavailable on this build" };
  const loaded = await native.loadEcapaSession(ecapaModelUrl(), await authHeaders(false));
  if (!loaded.session || loaded.model.status !== "ready") {
    return { unavailable: loaded.model.status === "ready" ? "ONNX session failed" : loaded.model.reason };
  }
  const session = loaded.session;
  return {
    embedder: new EcapaEmbedder(session),
    modelRev: revisionFromEtag(loaded.model.etag) ?? ECAPA_REVISION,
    modelSource: loaded.model.source,
    release: () => session.release(),
  };
}

export const defaultDeps: DeviceDiarizationDeps = {
  getMediaUrl: getRecordingMediaUrl,
  downloadBytes: defaultDownloadBytes,
  loadEmbedder: defaultLoadEmbedder,
  deviceInfo: () => collectDeviceInfo(),
  now: () => new Date(),
};

function percentile(sorted: number[], p: number): number | null {
  if (sorted.length === 0) return null;
  return sorted[Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)))];
}

/** Shape the engine's result (+ download facts) as the diagnostics event. */
export function toDeviceDiarizationEvent(
  recordingId: string,
  r: DiarizeWindowsResult,
  download: { ms: number; bytes: number },
  model: { rev: string | null; source: string | null },
  device: DeviceInfo,
  createdAt: string,
): DeviceDiarizationEvent {
  const r1 = (x: number) => Math.round(x * 10) / 10;
  const r3 = (x: number) => Math.round(x * 1000) / 1000;
  const sorted = [...r.embedMs].sort((a, b) => a - b);
  const mean = sorted.length > 0 ? sorted.reduce((a, b) => a + b, 0) / sorted.length : null;
  const p90 = percentile(sorted, 0.9);
  const cos = meanPairwiseCosine(r.embeddings);
  return {
    recording_id: recordingId,
    engine: "B",
    k: r.k,
    k_eigengap: r.kEigengap,
    eigenvalues: r.eigenvalues.map((x) => Math.round(x * 1e6) / 1e6),
    mean_pairwise_cosine: cos === null ? null : Math.round(cos * 1e4) / 1e4,
    segments: r.segments.map(([s, e, l]) => [r3(s), r3(e), l]),
    windows: r.windows,
    windows_total: r.totalWindows,
    window_s: r.windowSeconds,
    hop_s: r.hopSeconds,
    gate_rms: Math.round(r.gate * 1e4) / 1e4,
    speech_s: r3(r.speechSeconds),
    duration_s: r3(r.durationSeconds),
    download_ms: Math.round(download.ms),
    download_bytes: download.bytes,
    embed_ms_mean: mean === null ? null : r1(mean),
    embed_ms_p90: p90 === null ? null : r1(p90),
    cluster_ms: Math.round(r.timings.clusterMs + r.timings.smoothMs),
    total_ms: Math.round(download.ms + r.timings.totalMs),
    model_rev: model.rev,
    model_source: model.source,
    device,
    created_at: createdAt,
  };
}

function nowMs(): number {
  return typeof performance !== "undefined" && typeof performance.now === "function" ? performance.now() : Date.now();
}

/**
 * Start a run. Returns immediately with the promise and a `cancel()`; the
 * promise rejects with a DeviceDiarizationError (code "cancelled" after
 * cancel(), a phrased code otherwise).
 */
export function runDeviceDiarization(
  recordingId: string,
  opts: { onProgress?: (p: DeviceDiarizationProgress) => void; deps?: Partial<DeviceDiarizationDeps> } = {},
): DeviceDiarizationRun {
  const deps: DeviceDiarizationDeps = { ...defaultDeps, ...opts.deps };
  const progress = opts.onProgress ?? (() => {});
  const signal = { aborted: false };
  const check = () => {
    if (signal.aborted) throw new DeviceDiarizationError("cancelled", "cancelled");
  };

  const promise = (async (): Promise<DeviceDiarizationEvent> => {
    // 1. The model first: if it is not on the phone there is no point downloading audio.
    progress({ phase: "model", fraction: null, detail: "loading the voice model" });
    const loaded = await deps.loadEmbedder();
    if ("unavailable" in loaded) {
      throw new DeviceDiarizationError(`the voice model isn't ready on this phone (${loaded.unavailable})`, "model-unavailable");
    }
    try {
      check();
      // 2. Audio.
      progress({ phase: "download", fraction: 0, detail: "downloading audio" });
      const t0 = nowMs();
      let media: RecordingMediaUrl;
      try {
        media = await deps.getMediaUrl(recordingId);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        throw new DeviceDiarizationError(`couldn't get the audio link (${msg})`, "http");
      }
      check();
      let bytes: Uint8Array;
      try {
        bytes = await deps.downloadBytes(pcm16kMediaUrl(media), (loadedBytes, total) => {
          progress({
            phase: "download",
            fraction: total ? Math.min(1, loadedBytes / total) : null,
            detail: total ? `downloading audio ${fmtMb(loadedBytes)} / ${fmtMb(total)}` : `downloading audio ${fmtMb(loadedBytes)}`,
          });
        }, signal);
      } catch (err) {
        if (err instanceof DeviceDiarizationError) throw err;
        const msg = err instanceof Error ? err.message : String(err);
        throw new DeviceDiarizationError(`download failed (${msg})`, "network");
      }
      const downloadMs = nowMs() - t0;
      check();
      let pcm: Float32Array;
      try {
        pcm = parseWav16kMono(bytes, "recording audio");
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        throw new DeviceDiarizationError(`the audio couldn't be read (${msg})`, "bad-audio");
      }
      // 3. The engine, one window per embed call (the live loop's shape).
      // Every chunk crosses the native seam as an owned, zero-offset buffer
      // (see ownedFloat32) — diarizeWindows already slices, this is the guard.
      const embedBatch = async (chunks: Float32Array[], sr: number) => {
        const out: Float32Array[] = [];
        for (const c of chunks) {
          check();
          out.push(await loaded.embedder.embed(ownedFloat32(c), sr));
        }
        return out;
      };
      let result: DiarizeWindowsResult;
      try {
        result = await diarizeWindows(pcm, 16000, embedBatch, {
          signal,
          onProgress: (p) =>
            progress({
              phase: p.stage,
              fraction: p.total > 0 ? p.done / p.total : null,
              detail: p.stage === "embed" ? `listening to window ${p.done} of ${p.total}` : p.stage === "gate" ? "finding speech" : p.stage === "cluster" ? "grouping voices" : "smoothing",
            }),
        });
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") throw new DeviceDiarizationError("cancelled", "cancelled");
        throw err;
      }
      return toDeviceDiarizationEvent(
        recordingId,
        result,
        { ms: downloadMs, bytes: bytes.byteLength },
        { rev: loaded.modelRev, source: loaded.modelSource },
        deps.deviceInfo(),
        deps.now().toISOString(),
      );
    } finally {
      await loaded.release().catch(() => {});
    }
  })();

  return {
    promise,
    cancel: () => {
      signal.aborted = true;
    },
  };
}
