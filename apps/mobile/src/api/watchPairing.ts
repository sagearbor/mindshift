/**
 * Phase 3 Slice 1 — "Set up your watch": claim the 6-character pairing code
 * the watch displays. Mirrors the unified backend's `POST /me/pair/claim`
 * (server/watch/routers/pairing.py) exactly: full-auth Bearer token, JSON
 * body `{ code }`, and — unlike most of this file's other POSTs — the caller
 * gets back a plain `{ ok, detail? }` rather than a thrown error, because
 * every failure here (bad/expired code, already claimed, lockout) is a
 * NORMAL, expected outcome of a human mistyping a code, not an exceptional
 * one. The screen never polls: the watch is the one polling
 * `GET /me/pair/status`, so this call either succeeds once or reports why it
 * didn't.
 */
import { getFreshToken } from "../auth/authToken";

const API_URL = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

/** Result of a claim attempt. `detail` is always present on failure — the
 *  message to show the user, honest either way (see claimWatchPairing). */
export interface ClaimPairingResult {
  ok: boolean;
  detail?: string;
}

/** Fresh Firebase ID token as Bearer auth, mirroring client.ts's authHeaders
 *  (omitted when signed out, so the server answers its own 401 rather than
 *  the client fabricating one). */
async function authHeaders(): Promise<Record<string, string>> {
  const token = await getFreshToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

/** Best-effort read of the server's `{ detail: "..." }` body; undefined on a
 *  non-JSON or missing body rather than throwing. */
async function readDetail(res: Response): Promise<string | undefined> {
  try {
    const j = (await res.json()) as { detail?: unknown };
    return typeof j?.detail === "string" ? j.detail : undefined;
  } catch {
    return undefined;
  }
}

/**
 * POST /me/pair/claim — redeem the code the watch is showing for the
 * signed-in user's account. Success mints the watch's device token
 * server-side (the watch itself picks it up within ~10s via its own poll of
 * `GET /me/pair/status`); this call never sees that token and never polls.
 *
 * Every expected failure resolves (never throws) with an honest, human
 * message:
 *   - 404 (code not found — wrong or already expired) / 409 (already
 *     claimed): both mean "that code won't work anymore" from the human's
 *     seat, so both get the same friendly retry copy rather than the raw
 *     server strings (which are written for logs, not a person staring at a
 *     watch face).
 *   - 429 (per-account lockout, see pairing.py's FIX ROUND 2/3): the
 *     server's own detail is already the honest, specific thing to say, so
 *     it's shown verbatim.
 *   - 401 (shouldn't happen from a screen gated behind sign-in, but honest
 *     regardless): a plain "you need to be signed in" message.
 *   - anything else: the server's detail when it sent one, else a generic
 *     "try again" naming the status — never a fabricated success.
 */
export async function claimWatchPairing(
  code: string,
): Promise<ClaimPairingResult> {
  const res = await fetch(`${API_URL}/me/pair/claim`, {
    method: "POST",
    headers: await authHeaders(),
    body: JSON.stringify({ code }),
  });

  if (res.ok) {
    return { ok: true };
  }

  if (res.status === 404 || res.status === 409) {
    // Consume the body (even though we don't render it) so a real server
    // failure isn't masked by an unhandled rejection elsewhere.
    await readDetail(res);
    return {
      ok: false,
      detail:
        "That code wasn't recognized. It may be wrong, expired, or already used — check the watch for a fresh code and try again.",
    };
  }

  if (res.status === 429) {
    const detail = await readDetail(res);
    return {
      ok: false,
      detail: detail ?? "Too many failed attempts. Please try again later.",
    };
  }

  if (res.status === 401) {
    await readDetail(res);
    return { ok: false, detail: "You need to be signed in to pair a watch." };
  }

  const detail = await readDetail(res);
  return {
    ok: false,
    detail: detail ?? `Something went wrong (error ${res.status}). Please try again.`,
  };
}

/** Result of `disconnectWatch()` — `count` is the number of device tokens
 *  actually removed (0 is a valid, non-error outcome: nothing was paired). */
export interface DisconnectWatchResult {
  disconnected: boolean;
  count: number;
}

/**
 * DELETE /me/watch-pairing — "Disconnect this watch": revoke every device
 * token bound to the signed-in account (server/watch/routers/rest.py's
 * `disconnect_watch`). This is a pure auth-revoke, never a data deletion —
 * recordings, growth, and everything else stay exactly as they were, and
 * pairing a watch again (even a different one) immediately sees all the
 * same cloud data.
 *
 * Unlike `claimWatchPairing` (whose failures are normal, expected human
 * mistakes), a disconnect failure is unexpected — mirrors client.ts's
 * `forgetVoice`: throws `API error: <status>` on any non-OK response so the
 * confirm-then-delete UI can surface an honest error rather than silently
 * pretending success.
 */
export async function disconnectWatch(): Promise<DisconnectWatchResult> {
  const res = await fetch(`${API_URL}/me/watch-pairing`, {
    method: "DELETE",
    headers: await authHeaders(),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return (await res.json()) as DisconnectWatchResult;
}
