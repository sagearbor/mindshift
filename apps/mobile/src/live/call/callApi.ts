/**
 * REST half of in-app calls (server/routers/calls.py; the wire summary is
 * in types.ts). Mirrors api/liveSessions.ts: a fresh Firebase ID token as
 * Bearer auth; every failure is an Error with an honest message (404 = the
 * server predates calls) — the hook turns it into `CallView.error`, never a
 * fake call. `CallOut` is typed from the generated OpenAPI schema.
 */
import { authHeaders } from "../../api/liveSessions";
import type { components } from "../../api/generated/openapi";
import type { CallCreated, CallRole, IceConfig, IceServer } from "./types";

const API_URL = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

export type CallOut = components["schemas"]["CallOut"];
export type IceConfigOut = components["schemas"]["IceConfigOut"];

export interface CallApi {
  /** `POST /calls` — the caller becomes the host (slot A). */
  create(options?: { displayName?: string; maxParticipants?: 2 | 3 }): Promise<CallCreated>;
  /** `POST /calls/join` — by the code from the invite link, taking the seat
   *  the link encodes (participant = slot B, therapist = slot C). */
  join(joinCode: string, role?: CallRole, displayName?: string): Promise<CallCreated>;
  /** `POST /calls/{id}/end` — hang up for everyone; best effort. */
  end(callId: string): Promise<void>;
  /** `GET /calls/ice` — the ICE servers WITHOUT creating a call, for the
   *  connectivity pre-flight on the idle screen. */
  ice(): Promise<IceConfig>;
}

export function fromIceConfigOut(body: Partial<IceConfigOut> | null | undefined): IceConfig {
  const b = body ?? {};
  return {
    iceServers: Array.isArray(b.ice_servers) ? (b.ice_servers as IceServer[]) : [],
    turnConfigured: b.turn_configured === true,
    credentialMode: typeof b.turn_credential_mode === "string" ? b.turn_credential_mode : "none",
    ttlSeconds: typeof b.ttl_seconds === "number" ? b.ttl_seconds : null,
  };
}

export function fromCallOut(body: Partial<CallOut> | null | undefined): CallCreated {
  const b = body ?? {};
  const callId = typeof b.call_id === "string" ? b.call_id : "";
  if (!callId) throw new Error("the server's reply had no call_id");
  return {
    callId,
    joinCode: typeof b.join_code === "string" ? b.join_code : callId,
    joinUrl: typeof b.join_url === "string" ? b.join_url : "",
    selfLabel: typeof b.self_label === "string" ? b.self_label : null,
    selfRole: b.self_role === "therapist" ? "therapist" : "participant",
    iceServers: Array.isArray(b.ice_servers) ? (b.ice_servers as IceServer[]) : [],
  };
}

async function describeFailure(res: Response, what: string): Promise<Error> {
  let detail = "";
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // Not JSON.
  }
  if (res.status === 404 && !detail) return new Error(`${what}: this server has no in-app calls yet (404)`);
  if (res.status === 401) return new Error(`${what}: sign in again (401)`);
  return new Error(`${what}: ${detail || `HTTP ${res.status}`}`);
}

export const callApi: CallApi = {
  async create(options = {}) {
    const res = await fetch(`${API_URL}/calls`, {
      method: "POST",
      headers: await authHeaders(),
      body: JSON.stringify({
        ...(options.displayName ? { display_name: options.displayName } : {}),
        ...(options.maxParticipants ? { max_participants: options.maxParticipants } : {}),
      }),
    });
    if (!res.ok) throw await describeFailure(res, "couldn't start a call");
    return fromCallOut((await res.json()) as CallOut);
  },
  async join(joinCode, role, displayName) {
    const res = await fetch(`${API_URL}/calls/join`, {
      method: "POST",
      headers: await authHeaders(),
      body: JSON.stringify({
        join_code: joinCode.trim(),
        ...(role ? { role } : {}),
        ...(displayName ? { display_name: displayName } : {}),
      }),
    });
    if (res.status === 404) throw new Error("no call with that code (check it, or it may have ended)");
    if (res.status === 410) throw new Error("that call has ended or expired");
    if (!res.ok) throw await describeFailure(res, "couldn't join the call");
    return fromCallOut((await res.json()) as CallOut);
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
  async ice() {
    const res = await fetch(`${API_URL}/calls/ice`, { headers: await authHeaders() });
    if (!res.ok) throw await describeFailure(res, "couldn't fetch the ICE servers");
    return fromIceConfigOut((await res.json()) as IceConfigOut);
  },
};
