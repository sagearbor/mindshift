import { Platform } from "react-native";
import type { Suggestion } from "../components/SuggestionCard";
import { getFreshToken } from "../auth/authToken";

const API_URL =
  process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Request body for POST /respond. Mirrors the server's RespondRequest
 * (server/main.py): the server wants the single latest utterance as
 * `transcript_turn`, the empathy slider as an int 0–100, the role, and any
 * prior conversation as free-text `context`.
 */
export interface RespondRequest {
  transcript_turn: string;
  role: string;
  empathy_slider: number;
  context?: string;
}

/** Raw JSON returned by the server: suggestions are plain strings. */
interface RespondServerResponse {
  suggestions: string[];
  tone_score: Record<string, number>;
}

/** Parsed result the app consumes: suggestions carried as {text, tone}. */
export interface RespondResult {
  suggestions: Suggestion[];
  toneScore: Record<string, number>;
}

/**
 * Maps the empathy slider (0–100) to the coaching-stance label shown on each
 * suggestion. Mirrors useAudioStream's empathyTone: it describes how the
 * suggestion was generated, not a claim about detected tone (the server's
 * /respond returns suggestions as bare strings with no per-item tone).
 */
export function empathyTone(slider: number): string {
  if (slider <= 20) return "assertive";
  if (slider <= 50) return "balanced";
  if (slider <= 80) return "empathetic";
  return "validating";
}

// --- POST /analyze (Conversation Dynamics post-session analysis) ------------
// These types mirror the backend's /analyze response contract exactly. They are
// intentionally faithful: nullable fields (interruptions, coupling.strength,
// leader, follow_rate) stay `| null` so the UI can *omit* rather than fabricate
// a value the model couldn't determine — the house "no fake data" rule.

/** One turn as sent to /analyze. start/end times are optional (present only
 *  for diarized live audio; absent for pasted transcripts). */
export interface AnalyzeTurnInput {
  speaker: string;
  text: string;
  start_time?: number;
  end_time?: number;
}

/** Emotional/behavioral markers the backend may attach to a turn. */
export type ConversationMarker =
  | "criticism"
  | "contempt"
  | "defensiveness"
  | "stonewalling"
  | "repair_attempt"
  | "validation";

/**
 * Prosody summary for a turn, derived from the recording's audio track by the
 * /analyze/upload endpoint. Each field is a coarse categorical label. The whole
 * object is `null` (or absent) when prosody couldn't be measured — degraded
 * audio, or a text-only /analyze that never had sound to begin with — so the UI
 * omits the chips rather than inventing a reading (the house "no fake data" rule).
 */
export interface Voice {
  energy_label: "quiet" | "normal" | "loud";
  // Null when the turn had too little voiced speech to measure pitch — the
  // server refuses to invent a reading for silence/noise.
  pitch_label: "low" | "mid" | "high" | null;
  rate_label: "slow" | "normal" | "fast";
}

export interface AnalyzePerTurn {
  index: number;
  speaker: string;
  heat: number; // 0–100
  markers: string[]; // subset of ConversationMarker
  is_spike: boolean;
  trigger_phrase: string | null;
  // Optional so a pre-voice server (or the text-only /analyze path) simply omits
  // it; `null` when audio was present but prosody couldn't be measured.
  voice?: Voice | null;
}

export interface HorsemenCounts {
  criticism: number;
  contempt: number;
  defensiveness: number;
  stonewalling: number;
}

export interface AnalyzePerSpeaker {
  turns: number;
  talk_share: number; // 0–1
  avg_heat: number;
  peak_heat: number;
  peak_turn_index: number;
  heat_variance: number;
  interruptions: number | null; // null when the backend can't infer it
  horsemen: HorsemenCounts;
  repair_attempts: number;
  repairs_accepted: number;
}

export interface AnalyzeDynamics {
  coupling: {
    strength: number | null;
    leader: string | null;
    description: string;
  };
  deescalation: {
    who_first: string | null;
    follow_rate: number | null;
    description: string;
  };
  triggers: {
    phrase: string;
    speaker: string;
    turn_index: number;
    heat_delta: number;
  }[];
  requests: { speaker: string; request: string; outcome: string }[];
}

/**
 * Per-speaker "report card" grade. `score` is an ABSOLUTE 0–100 measure of
 * conduct (higher = better) — intentionally comparable across speakers (owner's
 * product decision), so the UI renders it plainly without softening qualifiers.
 */
export interface ReportCard {
  score: number; // 0–100, higher = better conduct
  headline: string;
  did_well: string;
  work_on: string;
}

export interface AnalyzeResult {
  per_turn: AnalyzePerTurn[];
  per_speaker: Record<string, AnalyzePerSpeaker>;
  // Optional at the type level to match runtime reality: a pre-v2 server omits
  // this field entirely (the UI already guards and omits the section).
  report_cards?: Record<string, ReportCard>;
  dynamics: AnalyzeDynamics;
  narrative: string;
}

/**
 * One transcribed turn returned by /analyze/upload. The client didn't have the
 * transcript (the server produced it from the recording), so unlike
 * AnalyzeTurnInput the timing fields are always present — the server diarized
 * real utterance boundaries, which is what lets /analyze compute honest
 * interruption stats downstream.
 */
export interface TurnOut {
  speaker: string;
  text: string;
  start_time: number;
  end_time: number;
}

/**
 * Result of POST /analyze/upload. It's a normal AnalyzeResult (its per_turn may
 * additionally carry `voice`) PLUS the server-produced `turns` transcript the
 * client never had, and an optional `voice_analysis` note explaining — honestly,
 * in plain words — when prosody was unavailable. `stored`/`recording_id`/
 * `storage_note` report what the server actually did with the file: storing
 * only ever happens when the caller both consented and asked to store, and
 * `storage_note` carries a plain-language reason (not fabricated) whenever it
 * didn't happen.
 */
export type UploadAnalyzeResult = AnalyzeResult & {
  turns: TurnOut[];
  voice_analysis?: string;
  stored: boolean;
  recording_id: string | null;
  storage_note: string | null;
};

/** Consent/storage choices for a recording upload. `consent` defaults to
 *  false (nothing is stored without it); `store` defaults to true (storage is
 *  attempted whenever consent is given, unless the caller opts out). */
export interface UploadAnalyzeOptions {
  consent?: boolean;
  store?: boolean;
}

/**
 * POST /analyze/upload — analyze a recording (audio or video; the server
 * extracts the audio track, transcribes, and analyzes it). Sent as
 * multipart/form-data so the binary file streams up untouched.
 *
 * `file` is the platform-native handle: on web it's a `File` (append it
 * directly); on native it's the local file URI string, appended as React
 * Native's `{ uri, name, type }` form-part object. We deliberately do NOT set a
 * Content-Type header — `fetch` must set `multipart/form-data` itself so it can
 * append the correct boundary. Bearer auth mirrors the other calls; a non-OK
 * response throws `API error: <status>` (401/413/422/429/502/503) so the caller
 * surfaces an honest, mapped message rather than a fabricated analysis.
 *
 * `consent`/`store` are sent as the literal strings "true"/"false" (multipart
 * form fields have no boolean type). The server only stores the recording when
 * BOTH are true; without consent, upload/analysis still proceeds but nothing
 * is retained (see `options.consent`'s default of false).
 */
export async function postAnalyzeUpload(
  file: string | File,
  name: string,
  mimeType: string,
  context?: string,
  options?: UploadAnalyzeOptions,
): Promise<UploadAnalyzeResult> {
  const token = await getFreshToken();
  const headers: Record<string, string> = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const form = new FormData();
  if (Platform.OS === "web") {
    // On web `file` is a real File; append it directly (its own name is used,
    // but we pass `name` too for servers that read the part's filename).
    form.append("file", file as File, name);
  } else {
    // React Native's FormData accepts a { uri, name, type } descriptor for a
    // local file; the bridge streams it from disk without loading it into JS.
    form.append(
      "file",
      { uri: file as string, name, type: mimeType } as unknown as Blob,
    );
  }
  if (context !== undefined) {
    form.append("context", context);
  }
  const consent = options?.consent ?? false;
  const store = options?.store ?? true;
  form.append("consent", consent ? "true" : "false");
  form.append("store", store ? "true" : "false");

  const res = await fetch(`${API_URL}/analyze/upload`, {
    method: "POST",
    headers,
    body: form,
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }

  return (await res.json()) as UploadAnalyzeResult;
}

/** One simulated turn from /analyze/counterfactual: a hypothetical heat value
 *  at that turn's conversation-wide index, from the pivot to the last turn. */
export interface SimulatedTurn {
  index: number;
  speaker: string;
  heat: number; // 0–100
}

/** Result of POST /analyze/counterfactual — a "what if this turn had been said
 *  differently" projection. Never fabricated client-side; comes wholesale from
 *  the server, disclaimer included, and is rendered verbatim. */
export interface CounterfactualResult {
  pivot_index: number;
  rewritten_text: string;
  rationale: string;
  simulated_per_turn: SimulatedTurn[];
  disclaimer: string;
}

/**
 * POST /analyze — post-session "Conversation Dynamics" analysis. Follows
 * postRespond exactly: a fresh Firebase ID token as Bearer auth (omitted when
 * signed out, so the server answers its own 401), JSON body, and a thrown
 * `API error: <status>` on any non-OK response so the caller can surface an
 * honest error state (401/429/502/413) rather than a fabricated result.
 */
export async function postAnalyze(
  turns: AnalyzeTurnInput[],
  context?: string,
): Promise<AnalyzeResult> {
  const token = await getFreshToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const res = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    headers,
    // Only include `context` when provided — keeps the body shape minimal and
    // matches the optional field in the contract.
    body: JSON.stringify(context !== undefined ? { turns, context } : { turns }),
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }

  return (await res.json()) as AnalyzeResult;
}

/**
 * POST /analyze/counterfactual — the "what if this turn had been said
 * differently?" projection. Follows postAnalyze exactly: fresh Firebase ID
 * token as Bearer auth (omitted when signed out so the server answers its own
 * 401), JSON body carrying the full `turns` (same shape incl. optional
 * start/end times) plus the `pivot_index` to rewrite, and a thrown
 * `API error: <status>` on any non-OK response (401/422/429/502/413) so the
 * caller can show an honest inline error rather than a fabricated simulation.
 */
export async function postCounterfactual(
  turns: AnalyzeTurnInput[],
  pivotIndex: number,
  context?: string,
): Promise<CounterfactualResult> {
  const token = await getFreshToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const res = await fetch(`${API_URL}/analyze/counterfactual`, {
    method: "POST",
    headers,
    body: JSON.stringify(
      context !== undefined
        ? { turns, pivot_index: pivotIndex, context }
        : { turns, pivot_index: pivotIndex },
    ),
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }

  return (await res.json()) as CounterfactualResult;
}

// --- Media replay (stored recordings + synced heat graph) -------------------
// These types mirror the PINNED persistence contract the backend is being built
// against. Like the /analyze types they are faithful to the wire shape; the
// client methods follow postAnalyze exactly (fresh Bearer token, thrown
// `API error: <status>` on any non-OK so callers surface honest 401/404/503
// states instead of a fabricated recording).

export type MediaType = "audio" | "video";

/** One entry in GET /recordings — enough to render a list row without fetching
 *  the full recording. */
export interface RecordingSummary {
  id: string;
  created_at: string; // ISO-8601 timestamp
  filename: string;
  media_type: MediaType;
  duration_seconds: number;
  has_analysis: boolean;
}

/** One stored turn. Unlike a pasted transcript, a recorded conversation always
 *  carries timing (that's what the playhead syncs against). */
export interface RecordingTurn {
  speaker: string;
  text: string;
  start_time: number;
  end_time: number;
}

/** GET /recordings/{id} — the summary fields plus the analyzed transcript and
 *  full dynamics analysis. `analysis.per_turn` is index-aligned with `turns`. */
export interface RecordingDetail extends RecordingSummary {
  turns: RecordingTurn[];
  analysis: AnalyzeResult;
}

/** GET /recordings/{id}/media_url — a short-lived signed URL for playback. */
export interface RecordingMediaUrl {
  url: string;
  expires_in: number; // seconds until the signed URL expires
}

/**
 * Shared auth-header builder: a fresh Firebase ID token as Bearer (omitted when
 * signed out so the server answers its own 401). Mirrors the inline header
 * blocks in postAnalyze/postCounterfactual so every recordings call authes
 * identically.
 */
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

/**
 * GET /recordings — list the caller's stored recordings. Throws
 * `API error: <status>` on any non-OK (401 signed out, 503 storage not
 * configured) so the UI shows an honest state rather than an empty list.
 */
export async function listRecordings(): Promise<RecordingSummary[]> {
  const res = await fetch(`${API_URL}/recordings`, {
    method: "GET",
    headers: await authHeaders(),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  const data = (await res.json()) as { recordings?: RecordingSummary[] };
  return Array.isArray(data.recordings) ? data.recordings : [];
}

/**
 * GET /recordings/{id} — the full recording: metadata, analyzed transcript, and
 * dynamics analysis. Throws on any non-OK (401/404/503).
 */
export async function getRecording(id: string): Promise<RecordingDetail> {
  const res = await fetch(`${API_URL}/recordings/${encodeURIComponent(id)}`, {
    method: "GET",
    headers: await authHeaders(),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return (await res.json()) as RecordingDetail;
}

/**
 * GET /recordings/{id}/media_url — a short-lived signed URL to stream the media.
 * Fetched separately from the detail so the (larger) analysis payload can render
 * while playback is still resolving, and so an expiring URL can be re-fetched
 * without re-downloading the analysis. Throws on any non-OK (401/404/503).
 */
export async function getRecordingMediaUrl(
  id: string,
): Promise<RecordingMediaUrl> {
  const res = await fetch(
    `${API_URL}/recordings/${encodeURIComponent(id)}/media_url`,
    {
      method: "GET",
      headers: await authHeaders(),
    },
  );
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return (await res.json()) as RecordingMediaUrl;
}

/**
 * DELETE /recordings/{id} — remove a stored recording. Resolves on the 204 the
 * contract specifies; throws `API error: <status>` on any non-OK (401/404/503)
 * so the caller can keep the row and surface the failure honestly.
 */
export async function deleteRecording(id: string): Promise<void> {
  const res = await fetch(`${API_URL}/recordings/${encodeURIComponent(id)}`, {
    method: "DELETE",
    headers: await authHeaders(),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
}

export async function postRespond(
  payload: RespondRequest,
): Promise<RespondResult> {
  // A fresh Firebase ID token authenticates the request; the backend verifies
  // it and scopes data to the token's uid. Null when signed out — the header is
  // then omitted and the server answers with its own 401.
  const token = await getFreshToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const res = await fetch(`${API_URL}/respond`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }

  const data = (await res.json()) as RespondServerResponse;
  const tone = empathyTone(payload.empathy_slider);
  const suggestions: Suggestion[] = Array.isArray(data.suggestions)
    ? data.suggestions.map((text) => ({ text, tone }))
    : [];

  return { suggestions, toneScore: data.tone_score ?? {} };
}
