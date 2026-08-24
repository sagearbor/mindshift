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
 *   app's document directory (`GET /models/ecapa.onnx`, produced by
 *   Foundation B's scripts/export_ecapa_onnx.py) — ~20 MB is too much to
 *   bundle, and absence must degrade to "speaker-ID off", not a crash.
 *
 * Never imported by tests (they use testing/ortNode.ts); every export here
 * returns null instead of throwing when a native piece is missing.
 */
import { Asset } from "expo-asset";
import { Directory, File, Paths } from "expo-file-system";
import * as ort from "onnxruntime-react-native";
import type { OnnxSession, OnnxSessionFactory, OrtRuntimeLike } from "./ort";
import { wrapOrtRuntime } from "./ort";

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

export const ECAPA_FILENAME = "ecapa.onnx";

/**
 * Resolve the ECAPA model on disk, downloading it from `url` on first use.
 * Null (never a throw) when the download fails or the endpoint is absent —
 * speaker-ID is simply disabled for this session.
 */
export async function ecapaModelPath(url: string, headers: Record<string, string> = {}): Promise<string | null> {
  try {
    const dir = new Directory(Paths.document, "models");
    if (!dir.exists) dir.create();
    const file = new File(dir, ECAPA_FILENAME);
    if (file.exists && (file.size ?? 0) > 1_000_000) return stripFileScheme(file.uri);
    const downloaded = await File.downloadFileAsync(url, file, { headers, idempotent: true });
    if (!downloaded.exists || (downloaded.size ?? 0) < 1_000_000) {
      // A 404 body or an HTML error page is not a model.
      try {
        downloaded.delete();
      } catch {
        // Nothing to clean up.
      }
      return null;
    }
    return stripFileScheme(downloaded.uri);
  } catch {
    return null;
  }
}

export async function loadEcapaSession(
  url: string,
  headers: Record<string, string> = {},
  factory: OnnxSessionFactory = nativeOrtSessionFactory(),
): Promise<OnnxSession | null> {
  const path = await ecapaModelPath(url, headers);
  if (!path) return null;
  try {
    return await factory(path);
  } catch (err) {
    console.warn("[live] ECAPA model failed to load — speaker-ID disabled:", err);
    return null;
  }
}
