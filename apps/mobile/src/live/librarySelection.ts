/**
 * Which coach-library items the user picked for Live Coach, remembered on
 * this device so the next session starts with the same selection.
 *
 * Wire: `library_item_ids` on the WebSocket config frame (the first frame,
 * and a config update whenever the selection changes mid-session). Ids only:
 * titles and contents stay on the server and never reach diagnostics.
 */
import { Platform } from "react-native";
import { Paths } from "expo-file-system";
import type { RecorderFs } from "../recorder/types";
import { ExpoRecorderFs } from "../recorder/expoFs";

/** The server's cap on one session's selection. */
export const MAX_LIBRARY_SELECTION = 20;

export const LIBRARY_SELECTION_FILE_NAME = "live-library-selection.json";
const WEB_KEY = "mindshift.liveLibrarySelection.v1";

export interface LibrarySelectionStore {
  load(): string[];
  save(ids: string[]): void;
}

/** Deduplicated string ids, capped at MAX_LIBRARY_SELECTION. */
export function normalizeSelection(ids: unknown): string[] {
  if (!Array.isArray(ids)) return [];
  const out: string[] = [];
  for (const v of ids) {
    if (typeof v === "string" && v && !out.includes(v)) out.push(v);
    if (out.length >= MAX_LIBRARY_SELECTION) break;
  }
  return out;
}

function parse(text: string | null | undefined): string[] {
  if (!text) return [];
  try {
    return normalizeSelection(JSON.parse(text));
  } catch {
    return [];
  }
}

export function createFsLibrarySelectionStore(fs: RecorderFs, fileUri: string): LibrarySelectionStore {
  return {
    load() {
      try {
        return fs.exists(fileUri) ? parse(fs.readText(fileUri)) : [];
      } catch {
        return [];
      }
    },
    save(ids) {
      try {
        const clean = normalizeSelection(ids);
        if (clean.length === 0) {
          if (fs.exists(fileUri)) fs.deleteRecursive(fileUri);
          return;
        }
        fs.writeText(fileUri, JSON.stringify(clean));
      } catch {
        // Remembering is a convenience; the session still gets the ids.
      }
    },
  };
}

function webStorage(): Storage | null {
  try {
    return (globalThis as { localStorage?: Storage }).localStorage ?? null;
  } catch {
    return null;
  }
}

/** Device store: a file in the app's document dir (native) or localStorage. */
export function defaultLibrarySelectionStore(): LibrarySelectionStore {
  if (Platform.OS === "web") {
    return {
      load: () => {
        try {
          return parse(webStorage()?.getItem(WEB_KEY));
        } catch {
          return [];
        }
      },
      save: (ids) => {
        try {
          const clean = normalizeSelection(ids);
          if (clean.length) webStorage()?.setItem(WEB_KEY, JSON.stringify(clean));
          else webStorage()?.removeItem(WEB_KEY);
        } catch {
          // Fail-open.
        }
      },
    };
  }
  try {
    const base = Paths.document.uri;
    const uri = `${base.endsWith("/") ? base.slice(0, -1) : base}/${LIBRARY_SELECTION_FILE_NAME}`;
    return createFsLibrarySelectionStore(new ExpoRecorderFs(), uri);
  } catch {
    return { load: () => [], save: () => {} };
  }
}
