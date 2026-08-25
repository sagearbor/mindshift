/**
 * DELETE /me — self-serve account deletion (server/routers/account.py).
 *
 * Its own module rather than another export in the 2.5k-line client.ts: this
 * is the one call in the app that cannot be undone, and it has two
 * requirements nothing else here has —
 *
 * 1. a FORCE-REFRESHED ID token. The server's `get_fresh_uid` refuses a token
 *    minted more than a few minutes ago, so we ask Firebase for a brand-new
 *    one immediately before the call (`getFreshToken(true)` →
 *    `getIdToken(true)`) rather than reusing the ~1h-cached one every other
 *    request happily sends.
 * 2. a typed confirmation in the body. The UI makes the user type DELETE; we
 *    send exactly that string, and the server 422s anything else.
 *
 * Honest errors: the thrown Error carries `.status` so the screen can tell a
 * signed-out/stale-token 401 from a 500 whose detail names which tier failed,
 * and its message is the server's own explanation when there is one.
 */
import { getFreshToken } from "../auth/authToken";

const API_URL = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

/** Per-category counts of what the server actually erased. The keys mirror
 *  `account_deletion.COUNT_KEYS`; every one is always present, so a 0 means
 *  "none of these", never "unknown". Typed loosely on purpose — the UI shows a
 *  fixed, human-worded list and never enumerates whatever the server sent. */
export interface DeleteAccountCounts {
  [category: string]: number;
}

export interface DeleteAccountResult {
  deleted: boolean;
  /** False on an idempotent repeat: the data walk ran and found nothing left. */
  firebase_user_deleted: boolean;
  counts: DeleteAccountCounts;
}

export type ApiError = Error & { status?: number };

/** The exact string the server demands, and the exact string the UI asks the
 *  user to type. One constant so the two can never drift. */
export const DELETE_CONFIRMATION = "DELETE";

/**
 * Permanently delete the signed-in account and everything under it.
 *
 * Resolves with the server's summary of what was removed. Rejects with an
 * Error carrying `.status` on any non-2xx — including the deliberate 500 the
 * server returns when a storage tier failed, which means NOTHING was
 * half-deleted silently and the user should retry.
 */
export async function deleteAccount(): Promise<DeleteAccountResult> {
  // Force-refresh: see (1) in the module docstring. A signed-out caller has no
  // token at all — send the request anyway and let the server answer its own
  // 401 rather than fabricating one here (house rule, same as me.ts).
  const token = await getFreshToken(true);
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const res = await fetch(`${API_URL}/me`, {
    method: "DELETE",
    headers,
    body: JSON.stringify({ confirm: DELETE_CONFIRMATION }),
  });

  if (!res.ok) {
    const err = new Error(
      (await errorMessage(res)) || `API error: ${res.status}`,
    ) as ApiError;
    err.status = res.status;
    throw err;
  }
  return (await res.json()) as DeleteAccountResult;
}

/** The server's own explanation, when it sent one. FastAPI's `detail` is a
 *  plain string for the 401/422/429 cases and an object (with a `message`) for
 *  the partial-failure 500; both are read, neither is invented. */
async function errorMessage(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as {
      detail?: string | { message?: string };
    };
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail.message === "string") return detail.message;
  } catch {
    // Non-JSON body — the caller falls back to the status line.
  }
  return "";
}
