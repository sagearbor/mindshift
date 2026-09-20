/**
 * Production ONNX Runtime seam over `onnxruntime-react-native` (1.24.3;
 * CPU/XNNPACK — see docs/research/on-device-stack-2026-08-24.md on why not
 * CoreML/NNAPI for these models) plus the two model files the fast loop
 * runs:
 *
 * - Silero VAD ships INSIDE the app (assets/models/silero_vad.onnx, 2.3 MB,
 *   MIT). Metro bundles `.onnx` as an asset (metro.config.js) and expo-asset
 *   materializes it on disk; ORT-RN needs a plain filesystem path, so the
 *   `file://` prefix is stripped (microsoft/onnxruntime#27062).
 * - ECAPA (voiceprints) is downloaded from the server at runtime into the
 *   app's document directory (`GET|HEAD /models/ecapa.onnx`, produced by
 *   server/ecapa_onnx.py — ~80 MB, far too much to bundle) with the
 *   download-once + ETag-revalidate protocol in modelDownload.ts. Absence
 *   (server 503, offline with no cache, older server) degrades to
 *   "speaker-ID off" with a reason, never a crash.
 *
 * Never imported by tests (they use testing/ortNode.ts and drive
 * modelDownload.ts with fakes); every export here returns null / an
 * `unavailable` result instead of throwing when a native piece is missing.
 */
import { Asset } from "expo-asset";
import { Directory, File, Paths } from "expo-file-system";
import * as ort from "onnxruntime-react-native";
import type { OnnxSession, OnnxSessionFactory, OrtRuntimeLike } from "./ort";
import { wrapOrtRuntime } from "./ort";
import {
  ECAPA_FILENAME,
  resolveEcapaModel,
  type EcapaModelResult,
  type FetchLike,
  type ModelFileStat,
  type ModelFileStore,
} from "./modelDownload";

export { ECAPA_FILENAME };

/** Fixed-shape models on CPU: the safe default on both platforms. */
const SESSION_OPTIONS = {
  executionProviders: ["cpu"],
  graphOptimizationLevel: "all",
};

export function nativeOrtSessionFactory(): OnnxSessionFactory {
  return wrapOrtRuntime(ort as unknown as OrtRuntimeLike, SESSION_OPTIONS);
}

function stripFileScheme(uri: string): string {
  return uri.startsWith("file://") ? uri.slice("file://".length) : uri;
}

/** The bundled Silero VAD, materialized to a path ORT can open; null when
 *  the asset can't be resolved (e.g. web). */
export async function sileroModelPath(): Promise<string | null> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const module = require("../../assets/models/silero_vad.onnx") as number;
    const asset = Asset.fromModule(module);
    await asset.downloadAsync();
    return asset.localUri ? stripFileScheme(asset.localUri) : null;
  } catch {
    return null;
  }
}

export async function loadSileroSession(
  factory: OnnxSessionFactory = nativeOrtSessionFactory(),
): Promise<OnnxSession | null> {
  const path = await sileroModelPath();
  if (!path) return null;
  try {
    return await factory(path);
  } catch (err) {
    console.warn("[live] Silero VAD failed to load — falling back to energy VAD:", err);
    return null;
  }
}

/**
 * modelDownload.ts's filesystem seam over expo-file-system's `File` /
 * `Directory` (SDK 57 object API), rooted at `<document dir>/models/` —
 * persistent across launches and OTA updates, excluded from nothing the
 * user would notice (it is regenerable, not user data).
 */
export function expoModelFileStore(): ModelFileStore {
  const dir = new Directory(Paths.document, "models");
  const file = (name: string) => new File(dir, name);
  const stat = (f: File): ModelFileStat => ({ exists: f.exists, size: f.exists ? (f.size ?? 0) : 0 });
  const ensureDir = () => {
    if (!dir.exists) dir.create({ intermediates: true, idempotent: true });
  };
  return {
    async stat(name) {
      return stat(file(name));
    },
    pathOf(name) {
      return stripFileScheme(file(name).uri);
    },
    async readText(name) {
      const f = file(name);
      return f.exists ? await f.text() : null;
    },
    async writeText(name, text) {
      ensureDir();
      file(name).write(text);
    },
    async download(url, name, headers) {
      ensureDir();
      // Delete any stale temp from an interrupted earlier attempt BEFORE
      // downloading: `idempotent` should overwrite, but on a Pixel 10 the
      // native downloadFileAsync was rejecting ("Call to function
      // 'FileSystem…'") against a left-over destination, and a truncated
      // native message is all the capability card can show. Removing it first
      // makes the call deterministic; the concise native reason is preserved
      // for the (rare) genuine failure.
      const dest = file(name);
      try {
        if (dest.exists) dest.delete();
      } catch {
        // best-effort; downloadFileAsync will still try
      }
      try {
        const downloaded = await File.downloadFileAsync(url, dest, {
          headers,
          idempotent: true,
        });
        return stat(downloaded);
      } catch (err) {
        // Native rejections read "Call to function 'X' has been rejected → Y";
        // keep just Y so the surfaced reason is the actual cause, not the
        // wrapper (which the capability card truncates).
        const raw = err instanceof Error ? err.message : String(err);
        const concise = raw.includes("→") ? raw.slice(raw.indexOf("→") + 1).trim() : raw;
        throw new Error(concise || raw);
      }
    },
    async move(from, to) {
      const dest = file(to);
      if (dest.exists) dest.delete();
      file(from).move(dest);
    },
    async remove(name) {
      const f = file(name);
      if (f.exists) f.delete();
    },
  };
}

/**
 * Resolve the ECAPA model on disk (download once, revalidate by ETag on
 * each launch). Never throws; see modelDownload.ts for the outcomes.
 */
export async function ecapaModel(
  url: string,
  headers: Record<string, string> = {},
  fetchImpl: FetchLike = fetch as unknown as FetchLike,
): Promise<EcapaModelResult> {
  try {
    return await resolveEcapaModel({ url, headers, fetch: fetchImpl, store: expoModelFileStore() });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { status: "unavailable", code: "network", reason: `model store failed (${msg})` };
  }
}

/** Back-compat shape: just the path, or null. */
export async function ecapaModelPath(url: string, headers: Record<string, string> = {}): Promise<string | null> {
  const result = await ecapaModel(url, headers);
  return result.status === "ready" ? result.path : null;
}

export interface EcapaLoad {
  session: OnnxSession | null;
  model: EcapaModelResult;
}

export async function loadEcapaSession(
  url: string,
  headers: Record<string, string> = {},
  factory: OnnxSessionFactory = nativeOrtSessionFactory(),
): Promise<EcapaLoad> {
  const model = await ecapaModel(url, headers);
  if (model.status !== "ready") return { session: null, model };
  try {
    return { session: await factory(model.path), model };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.warn("[live] ECAPA model failed to load — speaker-ID disabled:", err);
    return {
      session: null,
      model: { status: "unavailable", code: "bad-download", reason: `ONNX Runtime rejected the model (${msg})` },
    };
  }
}
