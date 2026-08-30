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
import { Platform } from "react-native";
import { File as FSFile, FileMode } from "expo-file-system";
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
  /** Distinct recordings pooled into the print (gates the contrast match). */
  settings?: number | null;
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
      // Absent on an older server => 1: the contrast match stays off for
      // that print rather than trusting a count nobody reported.
      settings: typeof p.settings === "number" && p.settings > 0 ? Math.trunc(p.settings) : 1,
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
export const ECAPA_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286";

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
  /** Distinct recordings the print pools (0 when the server predates it). */
  settings: number;
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
        settings: typeof p.settings === "number" ? p.settings : 0,
      });
    }
    return { people, error: null };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { people: [], error: `people unreachable (${msg})` };
  }
}

// --- Session audio: attach the phone's kept WAV to a saved live episode ------
// The Live Coach keeps the session's mic audio on the phone (live/
// liveAudioKeeper.ts) and, once `POST /sessions/live` has produced an episode,
// uploads it here so the episode gains replay / re-analyze / voice learning:
//   POST /sessions/{recording_id}/audio      multipart `file` (session.wav)
// Direct up to LIVE_AUDIO_DIRECT_MAX_BYTES (margin under the server's 25 MB
// cap); larger files take the existing chunked flow —
//   POST /uploads/start → PUT /uploads/{id}/chunks/{i} → POST /uploads/{id}/complete
// with `attach_to_recording_id` on the complete call. Both answer the same
// body. Every request carries an X-Request-ID and a fresh Bearer token, with
// one forced-refresh retry on 401 (client.ts uploadFetch, which isn't
// exported — the minimal loop is mirrored here rather than editing client.ts).
// Non-OK answers throw an Error carrying `.status` (client.ts's idiom);
// a network rejection throws with status 0. Idempotent server-side, so the
// hook's retry simply calls again.

/** Files at or under this size go up in one multipart POST. */
export const LIVE_AUDIO_DIRECT_MAX_BYTES = 24 * 1024 * 1024;
export const LIVE_AUDIO_FILE_NAME = "session.wav";
export const LIVE_AUDIO_MIME = "audio/wav";

export interface LiveSessionAudioResult {
  recording_id: string;
  media_type: "audio";
  duration_seconds: number;
  size_bytes: number;
  stored_variants: string[];
}

export type LiveSessionAudioUploader = (
  recordingId: string,
  fileUri: string,
  bytes: number,
) => Promise<LiveSessionAudioResult>;

function apiError(status: number, message = `API error: ${status}`): Error & { status: number } {
  const err = new Error(message) as Error & { status: number };
  err.status = status;
  return err;
}

function newRequestId(): string {
  let id = "";
  for (let i = 0; i < 32; i += 1) id += Math.floor(Math.random() * 16).toString(16);
  return id;
}

interface AudioUploadAttempt {
  requestId: string;
  authRetried: boolean;
}

async function audioUploadFetch(
  attempt: AudioUploadAttempt,
  url: string,
  init: {
    method: string;
    body?: BodyInit;
    contentType?: "application/json" | "application/octet-stream" | null;
  },
): Promise<Response> {
  let forcedToken: string | null = null;
  for (;;) {
    const token = forcedToken ?? (await getFreshToken());
    const headers: Record<string, string> = { "X-Request-ID": attempt.requestId };
    if (init.contentType) headers["Content-Type"] = init.contentType;
    if (token) headers.Authorization = `Bearer ${token}`;
    let res: Response;
    try {
      res = await fetch(url, { method: init.method, headers, body: init.body });
    } catch (cause) {
      throw apiError(
        0,
        `Upload network failure: ${cause instanceof Error ? cause.message : String(cause)}`,
      );
    }
    if (res.status === 401 && !attempt.authRetried) {
      attempt.authRetried = true;
      forcedToken = await getFreshToken(true);
      continue;
    }
    return res;
  }
}

async function postLiveSessionAudioDirect(
  attempt: AudioUploadAttempt,
  recordingId: string,
  fileUri: string,
): Promise<Response> {
  const form = new FormData();
  if (Platform.OS === "web") {
    const blob = await (await fetch(fileUri)).blob();
    form.append("file", blob, LIVE_AUDIO_FILE_NAME);
  } else {
    // expo-file-system's File implements Blob; Expo's WinterCG fetch streams
    // it from disk (the { uri, name, type } descriptor is rejected there).
    form.append("file", new FSFile(fileUri) as unknown as Blob, LIVE_AUDIO_FILE_NAME);
  }
  return audioUploadFetch(
    attempt,
    `${API_URL}/sessions/${encodeURIComponent(recordingId)}/audio`,
    { method: "POST", body: form as unknown as BodyInit, contentType: null },
  );
}

/** Successive byte ranges of the kept file (native: a seekable
 *  expo-file-system handle; web: Blob slices). */
async function openChunkReader(fileUri: string): Promise<{
  read(start: number, length: number): Promise<Uint8Array>;
  close(): void;
}> {
  if (Platform.OS === "web") {
    const blob = await (await fetch(fileUri)).blob();
    return {
      async read(start, length) {
        return new Uint8Array(await blob.slice(start, start + length).arrayBuffer());
      },
      close() {},
    };
  }
  const handle = new FSFile(fileUri).open(FileMode.ReadOnly);
  return {
    async read(start, length) {
      handle.offset = start;
      return handle.readBytes(length);
    },
    close() {
      handle.close();
    },
  };
}

const AUDIO_CHUNK_RETRY_BACKOFF_MS = [400, 800];

async function postLiveSessionAudioChunked(
  attempt: AudioUploadAttempt,
  recordingId: string,
  fileUri: string,
  bytes: number,
): Promise<Response> {
  const startRes = await audioUploadFetch(attempt, `${API_URL}/uploads/start`, {
    method: "POST",
    contentType: "application/json",
    body: JSON.stringify({
      filename: LIVE_AUDIO_FILE_NAME,
      content_type: LIVE_AUDIO_MIME,
      total_bytes: bytes,
      consent: true,
      store: true,
    }),
  });
  if (!startRes.ok) throw apiError(startRes.status);
  const { upload_id, chunk_bytes, expected_chunks } = (await startRes.json()) as {
    upload_id: string;
    chunk_bytes: number;
    expected_chunks: number;
  };
  const base = `${API_URL}/uploads/${encodeURIComponent(upload_id)}`;
  const abort = async () => {
    try {
      await audioUploadFetch(attempt, base, { method: "DELETE", contentType: "application/json" });
    } catch {
      // Best effort — the original failure is what the caller needs to see.
    }
  };
  try {
    const reader = await openChunkReader(fileUri);
    try {
      for (let index = 0; index < expected_chunks; index += 1) {
        const start = index * chunk_bytes;
        const chunk = await reader.read(start, Math.min(chunk_bytes, bytes - start));
        for (let tryIndex = 0; ; tryIndex += 1) {
          const canRetry = tryIndex < AUDIO_CHUNK_RETRY_BACKOFF_MS.length;
          let status: number;
          try {
            const res = await audioUploadFetch(attempt, `${base}/chunks/${index}`, {
              method: "PUT",
              body: chunk as unknown as BodyInit,
              contentType: "application/octet-stream",
            });
            if (res.ok) break;
            status = res.status;
          } catch (err) {
            if (!canRetry) throw err;
            status = 0;
          }
          const retryable = status === 0 || status >= 500 || status === 429 || status === 408;
          if (!canRetry || !retryable) throw apiError(status);
          await new Promise((r) => setTimeout(r, AUDIO_CHUNK_RETRY_BACKOFF_MS[tryIndex]));
        }
      }
    } finally {
      reader.close();
    }
    const completeRes = await audioUploadFetch(attempt, `${base}/complete`, {
      method: "POST",
      contentType: "application/json",
      body: JSON.stringify({ attach_to_recording_id: recordingId }),
    });
    if (!completeRes.ok) throw apiError(completeRes.status);
    return completeRes;
  } catch (err) {
    await abort();
    throw err;
  }
}

/**
 * Attach the kept session WAV to the saved live episode `recordingId`.
 * Direct multipart when `bytes` fits under the direct cap (a 413 from the
 * server re-routes through the chunked flow), chunked otherwise. Resolves
 * with the server's body; throws with `.status` on any non-OK answer
 * (404 unknown / not a live episode, 422 undecodable, 0 network).
 */
export async function postLiveSessionAudio(
  recordingId: string,
  fileUri: string,
  bytes: number,
): Promise<LiveSessionAudioResult> {
  const attempt: AudioUploadAttempt = { requestId: newRequestId(), authRetried: false };
  let res: Response | null = null;
  if (bytes <= LIVE_AUDIO_DIRECT_MAX_BYTES) {
    res = await postLiveSessionAudioDirect(attempt, recordingId, fileUri);
    if (res.status !== 413) {
      if (!res.ok) throw apiError(res.status);
      return (await res.json()) as LiveSessionAudioResult;
    }
  }
  res = await postLiveSessionAudioChunked(attempt, recordingId, fileUri, bytes);
  return (await res.json()) as LiveSessionAudioResult;
}
