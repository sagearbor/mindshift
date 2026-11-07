/**
 * Where the two PRIVATE Codex models live on the phone, and how we find them.
 *
 * The weights are never in the repo, the app bundle or the server: the owner
 * sideloads two files (docs/plans/2026-11-07-emotion-dev-mode.md, "Setup"):
 *
 *   emotion-tiny-int8.onnx   (~10 MB)  — the one that nudges
 *   emotion-small-int8.onnx  (~95 MB)  — silent, background
 *
 * into the app's external files folder (adb push works with no root; the app
 * can read its own folder with no permission), or into its documents
 * `emotion-models/` directory. Either is used IN PLACE (no 105 MB copy).
 * Absence is a status, never an error: the mode then says "models missing".
 */
import { MODEL_FILES, MODEL_MIN_BYTES } from "./config";

export type ModelKind = "tiny" | "small";

export type ModelStatus =
  | { status: "ready"; path: string; bytes: number; dir: string }
  | { status: "missing"; reason: string };

export interface ModelProbeFs {
  stat(uri: string): { exists: boolean; size: number };
}

export function stripFileScheme(uri: string): string {
  return uri.startsWith("file://") ? uri.slice("file://".length) : uri;
}

/** Candidate folders, in preference order. */
export function modelSearchDirs(opts: { documentDirUri: string; androidExternalDirUri: string | null }): string[] {
  const dirs: string[] = [];
  if (opts.androidExternalDirUri) dirs.push(opts.androidExternalDirUri);
  dirs.push(`${opts.documentDirUri.replace(/\/$/, "")}/emotion-models`);
  return dirs;
}

export function androidExternalEmotionDir(applicationId: string): string {
  return `file:///storage/emulated/0/Android/data/${applicationId}/files/emotion`;
}

export function resolveEmotionModel(kind: ModelKind, fs: ModelProbeFs, dirs: string[]): ModelStatus {
  const name = MODEL_FILES[kind];
  const seen: string[] = [];
  for (const dir of dirs) {
    const uri = `${dir}/${name}`;
    let st: { exists: boolean; size: number };
    try {
      st = fs.stat(uri);
    } catch {
      continue; // unreadable folder (e.g. external storage not reachable): try the next
    }
    if (!st.exists) continue;
    if (st.size < MODEL_MIN_BYTES[kind]) {
      seen.push(`${name} in ${dir} is only ${st.size} bytes (truncated copy?)`);
      continue;
    }
    return { status: "ready", path: stripFileScheme(uri), bytes: st.size, dir };
  }
  return {
    status: "missing",
    reason: seen.length ? seen.join("; ") : `${name} not found (looked in ${dirs.join(", ")})`,
  };
}

export function resolveBoth(fs: ModelProbeFs, dirs: string[]): { tiny: ModelStatus; small: ModelStatus } {
  return { tiny: resolveEmotionModel("tiny", fs, dirs), small: resolveEmotionModel("small", fs, dirs) };
}
