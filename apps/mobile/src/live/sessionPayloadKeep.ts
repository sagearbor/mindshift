/**
 * Debug safety net for POST /sessions/live (2026-10-07, dx-NCRN-SAQE: the
 * POST went out with `turns: []`, got a 422, and the session was simply
 * gone — no local copy, no retry).
 *
 * Behind ONE flag, KEEP_LIVE_SESSION_PAYLOAD_FOR_DEBUG (owner-approved for
 * now; flip to false to turn all of it off):
 *   - the finished session payload is written to the phone (document dir,
 *     `live-session-payloads/<session_id>.json`, newest MAX_KEPT_PAYLOADS
 *     kept) together with every POST attempt's outcome;
 *   - a POST that failed for a transient reason (network, 5xx, 429) is
 *     retried after POST_RETRY_DELAYS_MS; a 4xx such as 422 is not (the
 *     same body would be refused again) — it is kept for debugging instead.
 *
 * Independent of the flag, the hook never sends `turns: []`: with no
 * on-device turns the POST is skipped and the reason recorded
 * (`empty_turns_reason`).
 *
 * The file holds the transcript of a private conversation. It lives in the
 * app's own sandbox, is never uploaded by this module, and is capped.
 */
import { Platform } from "react-native";
import { Paths } from "expo-file-system";
import type { RecorderFs } from "../recorder/types";
import { ExpoRecorderFs } from "../recorder/expoFs";
import type { LiveSessionBody } from "../api/liveSessions";

/** THE flag: keep a local copy of each finished session payload and retry a
 *  transiently failed POST. Debug aid; on while the owner is testing. */
export const KEEP_LIVE_SESSION_PAYLOAD_FOR_DEBUG = true;

export const PAYLOAD_DIR_NAME = "live-session-payloads";
export const MAX_KEPT_PAYLOADS = 10;
/** Waits before each retry of a transiently failed POST (3 retries). */
export const POST_RETRY_DELAYS_MS: readonly number[] = [2000, 5000, 15000];

export interface PostAttempt {
  at: string;
  status: "created" | "unsupported" | "failed" | "skipped";
  error?: string;
}

export interface KeptSessionPayload {
  saved_at: string;
  body: LiveSessionBody;
  /** Why `body.turns` is empty, when it is (the POST is then skipped). */
  empty_turns_reason: string | null;
  attempts: PostAttempt[];
}

export interface SessionPayloadKeeper {
  /** Write (or overwrite) the record for this session. Never throws;
   *  returns the file uri or null. */
  save(record: KeptSessionPayload): string | null;
}

/** "API error: 422" → 422; anything else (network text) → null. */
export function httpStatusOf(error: string): number | null {
  const m = /API error: (\d{3})/.exec(error);
  return m ? Number(m[1]) : null;
}

/** Worth retrying: no HTTP status (network), 5xx, or 429. */
export function isRetryablePostFailure(error: string): boolean {
  const status = httpStatusOf(error);
  if (status === null) return true;
  return status >= 500 || status === 429;
}

function safeName(sessionId: string): string {
  return sessionId.replace(/[^A-Za-z0-9._-]/g, "_").slice(0, 120) || "session";
}

export function createSessionPayloadKeeper(fs: RecorderFs, dir: string): SessionPayloadKeeper {
  return {
    save(record) {
      try {
        fs.ensureDir(dir);
        const uri = `${dir}/${safeName(record.body.session_id)}.json`;
        fs.writeText(uri, JSON.stringify(record, null, 1));
        // Keep the newest MAX_KEPT_PAYLOADS (names carry the session id,
        // which is `live-<epoch ms>`, so they sort by age).
        const names = fs.listFileNames(dir).filter((n) => n.endsWith(".json")).sort();
        for (const old of names.slice(0, Math.max(0, names.length - MAX_KEPT_PAYLOADS))) {
          fs.deleteRecursive(`${dir}/${old}`);
        }
        return uri;
      } catch (err) {
        console.warn("[sessionPayloadKeep] could not keep the session payload:", err);
        return null;
      }
    },
  };
}

/** The production keeper (null in the browser, or when the flag is off). */
export function openDefaultSessionPayloadKeeper(): SessionPayloadKeeper | null {
  if (!KEEP_LIVE_SESSION_PAYLOAD_FOR_DEBUG || Platform.OS === "web") return null;
  try {
    const base = Paths.document.uri;
    const dir = `${base.endsWith("/") ? base.slice(0, -1) : base}/${PAYLOAD_DIR_NAME}`;
    return createSessionPayloadKeeper(new ExpoRecorderFs(), dir);
  } catch {
    return null;
  }
}
