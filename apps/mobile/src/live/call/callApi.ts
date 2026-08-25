/**
 * REST half of in-app calls (types.ts has the wire summary). Mirrors
 * api/liveSessions.ts: a fresh Firebase ID token as Bearer auth; every
 * failure is an Error with an honest message (404 = the server predates
 * calls) — the hook turns it into `CallView.error`, never a fake call.
 */
import { authHeaders } from "../../api/liveSessions";
import type { CallCreated, CallRole } from "./types";

const API_URL = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

export interface CallApi {
  create(maxParticipants?: number): Promise<CallCreated>;
  /** `idOrCode` is what the user has: the join code from a link, or a
   *  call id. The server accepts either in the path. `role` says which seat
   *  the joiner takes (a therapist link encodes ?role=therapist). */
  join(idOrCode: string, role?: CallRole): Promise<CallCreated>;
  end(callId: string): Promise<void>;
}

function parseCreated(body: unknown): CallCreated {
  const b = (body ?? {}) as Record<string, unknown>;
  const callId = typeof b.call_id === "string" ? b.call_id : "";
  if (!callId) throw new Error("the server's reply had no call_id");
  return {
    callId,
    joinCode: typeof b.join_code === "string" ? b.join_code : callId,
    joinUrl: typeof b.join_url === "string" ? b.join_url : "",
  };
}

async function describeFailure(res: Response, what: string): Promise<Error> {
  if (res.status === 404) return new Error(`${what}: this server has no in-app calls yet (404)`);
  if (res.status === 401) return new Error(`${what}: sign in again (401)`);
  let detail = "";
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // Not JSON.
  }
  return new Error(`${what}: ${res.status}${detail ? ` ${detail}` : ""}`);
}

export const callApi: CallApi = {
  async create(maxParticipants) {
    const res = await fetch(`${API_URL}/calls`, {
      method: "POST",
      headers: await authHeaders(),
      body: JSON.stringify(maxParticipants ? { max_participants: maxParticipants } : {}),
    });
    if (!res.ok) throw await describeFailure(res, "couldn't start a call");
    return parseCreated(await res.json());
  },
  async join(idOrCode, role) {
    const key = encodeURIComponent(idOrCode.trim());
    const res = await fetch(`${API_URL}/calls/${key}/join`, {
      method: "POST",
      headers: await authHeaders(),
      body: JSON.stringify(role ? { role } : {}),
    });
    if (res.status === 404) throw new Error("no call with that code (it may have ended)");
    if (!res.ok) throw await describeFailure(res, "couldn't join the call");
    return parseCreated(await res.json());
  },
  async end(callId) {
    try {
      await fetch(`${API_URL}/calls/${encodeURIComponent(callId)}/end`, {
        method: "POST",
        headers: await authHeaders(),
        body: JSON.stringify({}),
      });
    } catch {
      // Best effort: the socket closing ends it server-side too.
    }
  },
};
