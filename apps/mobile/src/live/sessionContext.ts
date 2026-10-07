/**
 * What the user tells the coach about THIS conversation, both optional and
 * generic by default (no relationship is ever assumed — the old client
 * default was "Husband / Wife"):
 *
 * - `relationship` — who they are talking with (child, partner, parent,
 *   coworker, friend, other). Wire: `relationship` on the WebSocket config
 *   frame and on POST /sessions/live; omitted when unset.
 * - `session_context` — free text ("Meeting with my boss to ask for a raise;
 *   this year I shipped…"), capped at SESSION_CONTEXT_MAX_CHARS (the server
 *   rejects longer). Wire: `session_context` on the config frame (and a
 *   config update when edited mid-session) and on POST /sessions/live;
 *   omitted when empty. Private: never in diagnostics.
 */
import { Platform } from "react-native";
import { Paths } from "expo-file-system";
import type { RecorderFs } from "../recorder/types";
import { ExpoRecorderFs } from "../recorder/expoFs";

export const RELATIONSHIPS = ["child", "partner", "parent", "coworker", "friend", "other"] as const;
export type Relationship = (typeof RELATIONSHIPS)[number];

export function isRelationship(v: unknown): v is Relationship {
  return typeof v === "string" && (RELATIONSHIPS as readonly string[]).includes(v);
}

/** Hard cap on the user-written context (the server rejects longer). */
export const SESSION_CONTEXT_MAX_CHARS = 4000;

/**
 * How much of the context goes into the ON-DEVICE coach prompt. Gemini Nano
 * (ML Kit GenAI) and Apple's on-device model take only a few thousand tokens
 * of input, the prompt already carries the system rules + recent turns, and
 * on-device latency grows with prompt length (a 4 s whisper budget). 600
 * characters ≈ 150 tokens keeps the prompt well inside that window; the
 * full text still goes to the server's coach.
 */
export const ON_DEVICE_CONTEXT_MAX_CHARS = 600;

/** Enforce the cap (the UI also blocks typing past it). */
export function clampSessionContext(text: string): string {
  return text.length > SESSION_CONTEXT_MAX_CHARS ? text.slice(0, SESSION_CONTEXT_MAX_CHARS) : text;
}

/** The context trimmed for the on-device prompt, cut at a word with "…". */
export function onDeviceSessionContext(text: string | null | undefined): string | null {
  const t = (text ?? "").replace(/\s+/g, " ").trim();
  if (!t) return null;
  if (t.length <= ON_DEVICE_CONTEXT_MAX_CHARS) return t;
  const cut = t.slice(0, ON_DEVICE_CONTEXT_MAX_CHARS - 1);
  const space = cut.lastIndexOf(" ");
  return `${space > ON_DEVICE_CONTEXT_MAX_CHARS * 0.6 ? cut.slice(0, space) : cut}…`;
}

// --- Remembered on this device (so the user can reuse it) -----------------

export const SESSION_CONTEXT_FILE_NAME = "live-session-context.txt";
const WEB_KEY = "mindshift.liveSessionContext.v1";

export interface SessionContextStore {
  load(): string;
  save(text: string): void;
  clear(): void;
}

export function createFsSessionContextStore(fs: RecorderFs, fileUri: string): SessionContextStore {
  return {
    load() {
      try {
        return fs.exists(fileUri) ? clampSessionContext(fs.readText(fileUri)) : "";
      } catch {
        return "";
      }
    },
    save(text) {
      try {
        if (!text) {
          if (fs.exists(fileUri)) fs.deleteRecursive(fileUri);
          return;
        }
        fs.writeText(fileUri, clampSessionContext(text));
      } catch {
        // Remembering is a convenience; the session still gets the text.
      }
    },
    clear() {
      try {
        if (fs.exists(fileUri)) fs.deleteRecursive(fileUri);
      } catch {
        // Nothing to clear.
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
export function defaultSessionContextStore(): SessionContextStore {
  if (Platform.OS === "web") {
    return {
      load: () => {
        try {
          return clampSessionContext(webStorage()?.getItem(WEB_KEY) ?? "");
        } catch {
          return "";
        }
      },
      save: (text) => {
        try {
          if (text) webStorage()?.setItem(WEB_KEY, clampSessionContext(text));
          else webStorage()?.removeItem(WEB_KEY);
        } catch {
          // Fail-open.
        }
      },
      clear: () => {
        try {
          webStorage()?.removeItem(WEB_KEY);
        } catch {
          // Fail-open.
        }
      },
    };
  }
  try {
    const base = Paths.document.uri;
    const uri = `${base.endsWith("/") ? base.slice(0, -1) : base}/${SESSION_CONTEXT_FILE_NAME}`;
    return createFsSessionContextStore(new ExpoRecorderFs(), uri);
  } catch {
    return { load: () => "", save: () => {}, clear: () => {} };
  }
}
