/**
 * Native glue for the in-app model picker: system file picker -> native file
 * copy into `<documents>/emotion-models/` (the resolver's second search path).
 * `File.copy` is a native copy, so the ~95 MB SMALL model is never loaded into
 * JS memory. All decisions live in modelImport.ts (unit-tested); this file is
 * the thin I/O layer, exercised only on a device.
 */
import * as DocumentPicker from "expo-document-picker";
import { Directory, File, Paths } from "expo-file-system";
import { Platform } from "react-native";
import { MODEL_FILES } from "./config";
import { androidExternalEmotionDir, modelSearchDirs, resolveBoth, type ModelKind, type ModelStatus } from "./modelFiles";
import { checkModelSize, checkStorage, validatePickedModel } from "./modelImport";
import * as Application from "expo-application";

export type ImportResult = { ok: true; message: string } | { ok: false; message: string };

function searchDirs(): string[] {
  const id = Platform.OS === "android" ? Application.applicationId : null;
  return modelSearchDirs({ documentDirUri: Paths.document.uri, androidExternalDirUri: id ? androidExternalEmotionDir(id) : null });
}

/** Where each model currently resolves (same search order the runtime uses). */
export function probeModels(): { tiny: ModelStatus; small: ModelStatus } | null {
  if (Platform.OS === "web") return null;
  try {
    return resolveBoth(
      {
        stat: (uri) => {
          const f = new File(uri);
          return { exists: f.exists, size: f.exists ? (f.size ?? 0) : 0 };
        },
      },
      searchDirs(),
    );
  } catch {
    return null;
  }
}

export async function pickAndImportModel(kind: ModelKind): Promise<ImportResult> {
  if (Platform.OS === "web") return { ok: false, message: "Model files can only be added in the native app." };
  let picked: DocumentPicker.DocumentPickerResult;
  try {
    // copyToCacheDirectory:true lets the picker do the (native) copy and hands back a plain file:// path;
    // copying straight from a content:// URI failed on device ("FileSystemFile.delete ... does not exist").
    picked = await DocumentPicker.getDocumentAsync({ type: "*/*", copyToCacheDirectory: true, multiple: false });
  } catch (e) {
    return { ok: false, message: `Could not open the file picker: ${e instanceof Error ? e.message : String(e)}` };
  }
  if (picked.canceled || !picked.assets?.[0]) return { ok: false, message: "Cancelled — no file chosen." };
  const asset = picked.assets[0];

  let src: File;
  let size: number | null = typeof asset.size === "number" ? asset.size : null;
  try {
    src = new File(asset.uri);
    if (size === null) size = src.exists ? (src.size ?? null) : null;
  } catch (e) {
    return { ok: false, message: `Could not read the chosen file: ${e instanceof Error ? e.message : String(e)}` };
  }

  const check = validatePickedModel(kind, { name: asset.name ?? null, size });
  if (!check.ok) return { ok: false, message: check.reason };

  if (size !== null) {
    let free: number | null = null;
    try {
      free = Paths.availableDiskSpace;
    } catch {
      free = null;
    }
    const lowStorage = checkStorage(size, free);
    if (lowStorage) return { ok: false, message: lowStorage };
  }

  const dir = new Directory(`${Paths.document.uri.replace(/\/$/, "")}/emotion-models`);
  const dest = new File(dir, MODEL_FILES[kind]);
  const safeDelete = (f: File) => {
    try {
      f.delete();
    } catch {
      // already gone / not deletable: nothing to clean
    }
  };
  try {
    if (!dir.exists) dir.create({ intermediates: true });
    safeDelete(dest);
    try {
      src.move(dest); // same-filesystem rename of the picker's cache copy: no second 95 MB copy
    } catch {
      src.copy(dest);
      safeDelete(src);
    }
    const copied = dest.size ?? 0;
    const sizeProblem = checkModelSize(kind, copied);
    if (sizeProblem) {
      safeDelete(dest);
      return { ok: false, message: sizeProblem };
    }
    return { ok: true, message: `${kind.toUpperCase()} model saved (${(copied / (1024 * 1024)).toFixed(1)} MB). Tap Check again.` };
  } catch (e) {
    safeDelete(dest);
    const msg = e instanceof Error ? e.message : String(e);
    return { ok: false, message: /space|ENOSPC/i.test(msg) ? `Out of storage while copying: ${msg}` : `Copy failed: ${msg}` };
  }
}
