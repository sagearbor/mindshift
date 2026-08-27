/**
 * Server calls for the on-device live path (Track 3-mobile).
 *
 * - `postLiveSession` — Track 2's `POST /sessions/live`: the phone's own
 *   record of a session (turn_local dicts + any cloud tone/identity events)
 *   so an episode exists even when the cloud never heard the audio. A 404
 *   (endpoint not deployed yet) resolves to null rather than failing the
 *   session — the transcript is already on screen.
 * - `fetchVoiceprints` — Foundation B's enrolled-people list with the
 *   on-device opt-in: `GET /voice/people?include_embeddings=true` returns,
 *   for the signed-in account's OWN enrolled people only, the blended
 *   L2-normalized voiceprint (`embedding`, `dim`, `model`) the server itself
 *   matches with (server/routers/voice.py — the default response still
 *   never carries one). People without a usable vector are skipped: []
 *   means "match nobody", never a guess. The result also says WHY it is
 *   empty (older server, 401, network) so the session status line can.
 * - `ecapaModelUrl` / `authHeaders` — where the ECAPA ONNX export is
 *   served (`GET|HEAD /models/ecapa.onnx`, server/routers/models.py),
 *   downloaded once + ETag-revalidated by src/live/modelDownload.ts via
 *   src/live/ortNative.ts.
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

/** Wire value of the session mode (`speaker` = the "In person" mode; `call`
 *  = an in-app call — server/routers/sessions.py LiveMode). */
export type LiveSessionMode = "earpiece" | "speaker" | "therapist" | "call";

/** A speaker the user NAMED during (or right after) the session — stored
 *  server-side as the human-assertion rung (manual / manual-person), the
 *  same one "Who is this?" writes on a stored recording. */
export interface LiveSpeakerLabel {
  display_name: string;
  /** An enrolled person id (or "self"); null for a name-only label. */
  person_id: string | null;
  is_self: boolean;
}

export interface LiveSessionBody {
  session_id: string;
  started_at: string;
  ended_at: string;
  mode: LiveSessionMode;
  turns: TurnLocalEvent[];
  tone_flags?: ToneFlagEvent[];
  speaker_identities?: SpeakerIdentityEvent[];
  /** Raw wire label → the name/person the user gave it mid-call. Omitted
   *  when nobody was named (older servers ignore the key). */
  speaker_labels?: Record<string, LiveSpeakerLabel>;
}

export type PostLiveSessionResult =
  | {
      status: "created";
      episodeId: string;
      /** Therapist emails the server auto-shared the episode with at
       *  ingest (the patient's linked therapist, auto-share on). Empty on
       *  older servers and when nobody was granted. */
      sharedWith: string[];
    }
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
    const data = (await res.json()) as { episode_id?: string; shared_with?: unknown };
    const sharedWith = Array.isArray(data.shared_with)
      ? data.shared_with.filter((x): x is string => typeof x === "string")
      : [];
    return { status: "created", episodeId: data.episode_id ?? "", sharedWith };
  } catch (err) {
    return { status: "failed", error: err instanceof Error ? err.message : String(err) };
  }
}

/** Foundation B's list endpoint, with the embeddings opt-in. */
export const VOICEPRINTS_PATH = "/voice/people?include_embeddings=true";

interface VoiceprintWire {
  person_id: string;
  display_name?: string | null;
  is_self?: boolean;
  embedding?: number[] | null;
  voiceprint?: number[] | null;
  dim?: number | null;
  model?: string | null;
}

export interface VoiceprintsResult {
  people: EnrolledPerson[];
  /** Why the list may be empty/partial: null on a clean answer. */
  error: string | null;
}

/** Parse the wire people list into labeler input — exported for tests and
 *  for any future cached copy. A person whose vector length disagrees with
 *  the server's own `dim` is dropped (a corrupt print must not silently
 *  match nobody / everybody). */
export function parseVoiceprints(data: unknown): EnrolledPerson[] {
  const list: VoiceprintWire[] = Array.isArray(data)
    ? (data as VoiceprintWire[])
    : (((data as { people?: VoiceprintWire[] } | null)?.people) ?? []);
  const people: EnrolledPerson[] = [];
  for (const p of list) {
    if (!p || typeof p !== "object") continue;
    const emb = p.embedding ?? p.voiceprint;
    if (!p.person_id || !Array.isArray(emb) || emb.length === 0) continue;
    if (typeof p.dim === "number" && p.dim > 0 && p.dim !== emb.length) continue;
    if (!emb.every((x) => typeof x === "number" && Number.isFinite(x))) continue;
    people.push({
      personId: p.person_id,
      displayName: p.display_name || (p.is_self ? "You" : p.person_id),
      isSelf: Boolean(p.is_self),
      embedding: emb,
      model: p.model ?? null,
      dim: typeof p.dim === "number" ? p.dim : emb.length,
    });
  }
  return people;
}

export async function fetchVoiceprints(path = VOICEPRINTS_PATH): Promise<VoiceprintsResult> {
  try {
    const res = await fetch(`${API_URL}${path}`, { method: "GET", headers: await authHeaders(false) });
    if (res.status === 404) return { people: [], error: "server has no people endpoint (404)" };
    if (res.status === 401 || res.status === 403) return { people: [], error: `not signed in (${res.status})` };
    if (!res.ok) return { people: [], error: `people endpoint answered ${res.status}` };
    return { people: parseVoiceprints(await res.json()), error: null };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { people: [], error: `voiceprints unreachable (${msg})` };
  }
}

// The pinned ECAPA revision the server exports and the phone's voiceprints are
// tied to (server: speaker_id.ECAPA_REVISION). Bump BOTH this and the uploaded
// file when the pin changes.
const ECAPA_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286";

export function ecapaModelUrl(): string {
  // Served from Firebase Hosting (public CDN), NOT the Cloud Run API. The API's
  // GET /models/ecapa.onnx returned 500 on the 84 MB body from a real Pixel 10
  // (HEAD 200 / GET 500, deterministic — Cloud-Run-specific, the route serves
  // fine locally, 2026-08-26), which left speaker-ID off and every voice
  // labelled "Unknown". Hosting handles the large file with Range support and
  // an ETag, so modelDownload.ts's download-once + revalidate works unchanged;
  // the revision in the path versions it. Overridable for tests/self-host.
  const override = process.env.EXPO_PUBLIC_ECAPA_URL;
  if (override) return override;
  return `https://arborfam-hub.web.app/models/ecapa_${ECAPA_REVISION}.onnx`;
}

/** One enrolled person, as the pre-session "who's here" strip shows it
 *  (no embedding — that opt-in is `fetchVoiceprints`). */
export interface VoicePerson {
  personId: string;
  displayName: string;
  isSelf: boolean;
  enrollCount: number;
}

export interface VoicePeopleResult {
  people: VoicePerson[];
  /** Why the list may be empty: null on a clean answer. */
  error: string | null;
}

/** `GET /voice/people` (default, embedding-free) — the account's enrolled
 *  people, owner first. Read-only consumer; enrolment itself lives in the
 *  People / Voice settings screens. Never throws. */
export async function listVoicePeople(): Promise<VoicePeopleResult> {
  try {
    const res = await fetch(`${API_URL}/voice/people`, {
      method: "GET",
      headers: await authHeaders(false),
    });
    if (res.status === 404) return { people: [], error: "server has no people endpoint (404)" };
    if (res.status === 401 || res.status === 403) return { people: [], error: `not signed in (${res.status})` };
    if (!res.ok) return { people: [], error: `people endpoint answered ${res.status}` };
    const data = (await res.json()) as { people?: VoiceprintWire[] & { enroll_count?: number }[] };
    const list = Array.isArray(data?.people) ? data.people : [];
    const people: VoicePerson[] = [];
    for (const p of list as (VoiceprintWire & { enroll_count?: number })[]) {
      if (!p || typeof p !== "object" || !p.person_id) continue;
      people.push({
        personId: p.person_id,
        displayName: p.display_name || (p.is_self ? "You" : p.person_id),
        isSelf: Boolean(p.is_self),
        enrollCount: typeof p.enroll_count === "number" ? p.enroll_count : 0,
      });
    }
    return { people, error: null };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { people: [], error: `people unreachable (${msg})` };
  }
}
