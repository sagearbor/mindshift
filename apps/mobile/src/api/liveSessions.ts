/**
 * Server calls for the on-device live path (Track 3-mobile).
 *
 * - `postLiveSession` — Track 2's `POST /sessions/live`: the phone's own
 *   record of a session (turn_local dicts + any cloud tone/identity events)
 *   so an episode exists even when the cloud never heard the audio. A 404
 *   (endpoint not deployed yet) resolves to null rather than failing the
 *   session — the transcript is already on screen.
 * - `fetchVoiceprints` — Foundation B's enrolled-people list (person_id,
 *   display_name, is_self, 192-d embedding). Tolerant of absence: [] means
 *   "match nobody", never a guess.
 * - `ecapaModelUrl` / `authHeaders` — where the ECAPA ONNX export lives
 *   (`GET /models/ecapa.onnx`), downloaded by src/live/ortNative.ts.
 *
 * Mirrors me.ts / client.ts: fresh Firebase ID token as Bearer auth when
 * signed in; absent otherwise so the server answers its own 401.
 */
import { getFreshToken } from "../auth/authToken";
import type { TurnLocalEvent, ToneFlagEvent, SpeakerIdentityEvent } from "../live/types";
import type { EnrolledPerson } from "../live/speakerId";

const API_URL = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

export async function authHeaders(json = true): Promise<Record<string, string>> {
  const token = await getFreshToken();
  const headers: Record<string, string> = json ? { "Content-Type": "application/json" } : {};
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

export type LiveSessionMode = "earpiece" | "speaker" | "therapist";

export interface LiveSessionBody {
  session_id: string;
  started_at: string;
  ended_at: string;
  mode: LiveSessionMode;
  turns: TurnLocalEvent[];
  tone_flags?: ToneFlagEvent[];
  speaker_identities?: SpeakerIdentityEvent[];
}

export type PostLiveSessionResult =
  | { status: "created"; episodeId: string }
  | { status: "unsupported" } // 404: endpoint not deployed yet
  | { status: "failed"; error: string };

export async function postLiveSession(body: LiveSessionBody): Promise<PostLiveSessionResult> {
  try {
    const res = await fetch(`${API_URL}/sessions/live`, {
      method: "POST",
      headers: await authHeaders(),
      body: JSON.stringify(body),
    });
    if (res.status === 404) return { status: "unsupported" };
    if (!res.ok) return { status: "failed", error: `API error: ${res.status}` };
    const data = (await res.json()) as { episode_id?: string };
    return { status: "created", episodeId: data.episode_id ?? "" };
  } catch (err) {
    return { status: "failed", error: err instanceof Error ? err.message : String(err) };
  }
}

/** Foundation B's list endpoint. Overridable so the path can follow the
 *  server without a client release once it lands. */
export const VOICEPRINTS_PATH = "/voice/people";

interface VoiceprintWire {
  person_id: string;
  display_name?: string | null;
  is_self?: boolean;
  embedding?: number[] | null;
  voiceprint?: number[] | null;
}

export async function fetchVoiceprints(path = VOICEPRINTS_PATH): Promise<EnrolledPerson[]> {
  try {
    const res = await fetch(`${API_URL}${path}`, { method: "GET", headers: await authHeaders(false) });
    if (!res.ok) return [];
    const data = (await res.json()) as VoiceprintWire[] | { people?: VoiceprintWire[] };
    const list = Array.isArray(data) ? data : (data.people ?? []);
    const people: EnrolledPerson[] = [];
    for (const p of list) {
      const emb = p.embedding ?? p.voiceprint;
      if (!p.person_id || !Array.isArray(emb) || emb.length === 0) continue;
      people.push({
        personId: p.person_id,
        displayName: p.display_name || (p.is_self ? "You" : p.person_id),
        isSelf: Boolean(p.is_self),
        embedding: emb,
      });
    }
    return people;
  } catch {
    return [];
  }
}

export function ecapaModelUrl(): string {
  return `${API_URL}/models/ecapa.onnx`;
}
