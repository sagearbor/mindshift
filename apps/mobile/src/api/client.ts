import { Platform } from "react-native";
import { File as FSFile, FileMode } from "expo-file-system";
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
 *  attempted whenever consent is given, unless the caller opts out).
 *
 *  `title` is an OPTIONAL human name for the conversation ("Sunday budget
 *  talk"). It's forwarded to the server so a stored recording can be titled
 *  instead of showing the raw upload filename ("photos_share.mp4"). Servers that
 *  don't yet support titling simply ignore the extra field (no schema is
 *  `extra="forbid"`), so sending it is forward-compatible and harmless. */
export interface UploadAnalyzeOptions {
  consent?: boolean;
  store?: boolean;
  title?: string;
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
  // Optional human title for the stored recording — ignored by servers that
  // don't support it yet (extra form fields are dropped, not rejected).
  if (options?.title !== undefined && options.title !== "") {
    form.append("title", options.title);
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

// --- Chunked upload (large recordings) --------------------------------------
// Some hosting layers (Cloud Run / the app's ingress) reject request bodies over
// ~32MB *before* the server can answer 413 — a 100MB phone video would surface a
// generic transport failure, not an honest size message. The chunked path splits
// the file into server-sized slices so no single request trips that ceiling:
//   POST   /uploads/start            → { upload_id, chunk_bytes, expected_chunks }
//   PUT    /uploads/{id}/chunks/{i}  raw octet-stream body → 204/200
//   POST   /uploads/{id}/complete    → the full UploadAnalyzeResult
//   DELETE /uploads/{id}             → abort (best-effort on any failure)
// `complete` can take minutes for a long video (server-side transcription), so we
// deliberately set no client-side timeout on it.

/** Options for a chunked upload. `consent`/`store` are sent as real JSON booleans
 *  in the /uploads/start body (unlike the multipart /analyze/upload path, which
 *  can only carry strings). `onProgress` receives a 0→1 fraction after each chunk
 *  lands so the UI can render an honest progress bar. */
export interface ChunkedUploadOptions {
  consent: boolean;
  store: boolean;
  context?: string;
  // Optional human title for the stored recording (see UploadAnalyzeOptions.title);
  // ignored by servers that don't support titling yet.
  title?: string;
  onProgress?: (fraction: number) => void;
}

/** Response of POST /uploads/start. */
interface UploadStartResult {
  upload_id: string;
  chunk_bytes: number;
  expected_chunks: number;
}

/**
 * Reads successive byte ranges of the picked file, uniformly across platforms:
 *
 *   - Web: `file` is a DOM `File` (a Blob); `File.slice(start, end)` +
 *     `arrayBuffer()` yields the bytes without loading the whole file.
 *   - Native: `file` is a `file://` URI string. expo-file-system's modern `File`
 *     API opens a read handle whose `offset` we seek and `readBytes(length)`
 *     returns a `Uint8Array` directly — no base64 encode/decode round-trip (the
 *     legacy `readAsStringAsync({ encoding: base64, position, length })` path
 *     would have forced one). The handle is opened once and reused across chunks,
 *     then closed by `close()`.
 *
 * Both branches return a `Uint8Array`, which React Native's networking layer
 * accepts as a raw binary fetch body (it base64-encodes it for the native bridge
 * internally), so the PUT sends application/octet-stream on web and native alike.
 */
interface ChunkReader {
  read(start: number, length: number): Promise<Uint8Array>;
  close(): void;
}

function createChunkReader(file: string | File): ChunkReader {
  if (Platform.OS === "web") {
    const blob = file as File;
    return {
      async read(start, length) {
        const buf = await blob.slice(start, start + length).arrayBuffer();
        return new Uint8Array(buf);
      },
      close() {},
    };
  }
  const handle = new FSFile(file as string).open(FileMode.ReadOnly);
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

/** Bearer header for a raw-binary chunk PUT (octet-stream, not JSON). A fresh
 *  token is fetched per chunk so a multi-minute upload survives token refresh. */
async function octetStreamHeaders(): Promise<Record<string, string>> {
  const token = await getFreshToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/octet-stream",
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

/**
 * Upload a large recording in chunks, then analyze it. Mirrors
 * `postAnalyzeUpload`'s result and honesty contract (a non-OK response throws
 * `API error: <status>` — 401/404/409/413/503 — so the caller surfaces a mapped
 * message rather than a fabricated analysis), but streams the file so no single
 * request exceeds the platform's body ceiling.
 *
 * On ANY failure after the upload has started, a best-effort DELETE aborts the
 * partial upload server-side before the honest error is rethrown, so we never
 * leave orphaned chunks behind.
 */
/**
 * Best-effort abort so a failed upload doesn't leave orphaned chunks. Any error
 * from the abort itself is swallowed — the original failure is what the caller
 * needs to see.
 */
async function abortChunkedUpload(uploadId: string): Promise<void> {
  try {
    await fetch(`${API_URL}/uploads/${encodeURIComponent(uploadId)}`, {
      method: "DELETE",
      headers: await authHeaders(),
    });
  } catch {
    // ignore
  }
}

/**
 * Phase 1 of a chunked upload: negotiate a session and PUT every slice. Returns
 * the server's `upload_id` once all parts have landed. On a start failure there
 * is nothing server-side to abort; on a PUT failure the partial upload is aborted
 * before the honest error is rethrown. Shared by the synchronous-complete and
 * job-complete flows so the byte-streaming logic lives in one place.
 */
async function uploadFileInChunks(
  file: string | File,
  name: string,
  mimeType: string,
  sizeBytes: number,
  opts: ChunkedUploadOptions,
): Promise<string> {
  const startBody: Record<string, unknown> = {
    filename: name,
    content_type: mimeType,
    total_bytes: sizeBytes,
    consent: opts.consent,
    store: opts.store,
  };
  if (opts.context !== undefined) {
    startBody.context = opts.context;
  }
  if (opts.title !== undefined && opts.title !== "") {
    startBody.title = opts.title;
  }
  const startRes = await fetch(`${API_URL}/uploads/start`, {
    method: "POST",
    headers: await authHeaders(),
    body: JSON.stringify(startBody),
  });
  if (!startRes.ok) {
    // Nothing was created server-side yet, so there's nothing to abort.
    throw new Error(`API error: ${startRes.status}`);
  }
  const { upload_id, chunk_bytes, expected_chunks } =
    (await startRes.json()) as UploadStartResult;

  const reader = createChunkReader(file);
  try {
    for (let index = 0; index < expected_chunks; index += 1) {
      const start = index * chunk_bytes;
      const length = Math.min(chunk_bytes, sizeBytes - start);
      const chunk = await reader.read(start, length);
      const res = await fetch(
        `${API_URL}/uploads/${encodeURIComponent(upload_id)}/chunks/${index}`,
        {
          method: "PUT",
          headers: await octetStreamHeaders(),
          body: chunk as unknown as BodyInit,
        },
      );
      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }
      opts.onProgress?.((index + 1) / expected_chunks);
    }
  } catch (err) {
    await abortChunkedUpload(upload_id);
    throw err;
  } finally {
    reader.close();
  }
  return upload_id;
}

export async function postAnalyzeUploadChunked(
  file: string | File,
  name: string,
  mimeType: string,
  sizeBytes: number,
  opts: ChunkedUploadOptions,
): Promise<UploadAnalyzeResult> {
  const uploadId = await uploadFileInChunks(file, name, mimeType, sizeBytes, opts);
  try {
    // Complete: the server transcribes + analyzes and returns the full result.
    // No client timeout — a long video can legitimately take minutes here (this
    // synchronous completion is exactly the multi-minute request the job path
    // below exists to replace on servers that support it).
    const completeRes = await fetch(
      `${API_URL}/uploads/${encodeURIComponent(uploadId)}/complete`,
      {
        method: "POST",
        headers: await authHeaders(),
      },
    );
    if (!completeRes.ok) {
      throw new Error(`API error: ${completeRes.status}`);
    }
    return (await completeRes.json()) as UploadAnalyzeResult;
  } catch (err) {
    await abortChunkedUpload(uploadId);
    throw err;
  }
}

/** A chunked upload that completes as a submit-and-poll JOB. Either the server
 *  accepted the job (poll `jobId` via {@link getAnalyzeJob}) or — on an older
 *  server / storage off (the job endpoint 404s/503s) — it fell back to the
 *  synchronous complete and already has the `result`. */
export type ChunkedJobOutcome =
  | { jobId: string }
  | { result: UploadAnalyzeResult };

/**
 * Chunked upload whose completion is a background JOB (fixes the multi-minute
 * synchronous `/complete` that Android backgrounding routinely kills). Streams
 * the file exactly as {@link postAnalyzeUploadChunked}, then POSTs
 * `/uploads/{id}/complete/jobs`:
 *   - 202 → returns `{ jobId }`; the server now owns the parts (it cleans them
 *     up when the job finishes), so the caller polls and never aborts.
 *   - 404/503 (old server / storage off) → falls back to the synchronous
 *     `/complete` and returns `{ result }` — the parts are still present.
 * Any other failure aborts the partial upload and rethrows an honest error.
 */
export async function postAnalyzeUploadChunkedJob(
  file: string | File,
  name: string,
  mimeType: string,
  sizeBytes: number,
  opts: ChunkedUploadOptions,
): Promise<ChunkedJobOutcome> {
  const uploadId = await uploadFileInChunks(file, name, mimeType, sizeBytes, opts);
  const jobRes = await fetch(
    `${API_URL}/uploads/${encodeURIComponent(uploadId)}/complete/jobs`,
    { method: "POST", headers: await authHeaders() },
  );
  if (jobRes.ok) {
    // Job accepted — the server owns the parts now; do NOT abort.
    const { job_id } = (await jobRes.json()) as JobCreated;
    return { jobId: job_id };
  }
  if (jobRes.status === 404 || jobRes.status === 503) {
    // Jobs unavailable (old server / storage off) — synchronous fallback. The
    // parts are still on the server; complete() aborts on its own failure.
    try {
      const completeRes = await fetch(
        `${API_URL}/uploads/${encodeURIComponent(uploadId)}/complete`,
        { method: "POST", headers: await authHeaders() },
      );
      if (!completeRes.ok) {
        throw new Error(`API error: ${completeRes.status}`);
      }
      return { result: (await completeRes.json()) as UploadAnalyzeResult };
    } catch (err) {
      await abortChunkedUpload(uploadId);
      throw err;
    }
  }
  // A real error on the job POST — abort the upload and surface it.
  await abortChunkedUpload(uploadId);
  throw new Error(`API error: ${jobRes.status}`);
}

/** Options for analyzing a remote recording by link. Same consent/store meaning
 *  as an upload; both are sent as real JSON booleans. */
export interface AnalyzeLinkOptions {
  consent: boolean;
  store: boolean;
  context?: string;
  // Optional human title for the stored recording (see UploadAnalyzeOptions.title);
  // ignored by servers that don't support titling yet.
  title?: string;
}

/**
 * POST /analyze/link — analyze a recording the server fetches from a URL (a
 * direct file link, a Google Drive share link, or a Google Photos share link of
 * a single video; the server downloads, extracts audio, transcribes, and
 * analyzes). Returns the same `UploadAnalyzeResult` as the upload paths.
 *
 * Unlike the other calls, a non-OK response's *body message* is user-facing: the
 * server writes 422 (not a direct link / blocked URL / unrecognised share page)
 * and 413 (over the size cap) messages for humans, so we surface the server's
 * `detail` verbatim on the thrown error (as `.detail`) while still carrying the
 * numeric `.status` for the generic branches (401/429/502/503). Callers render
 * `.detail` when present and fall back to a mapped message otherwise.
 */
export async function postAnalyzeLink(
  url: string,
  options: AnalyzeLinkOptions,
): Promise<UploadAnalyzeResult> {
  const body: Record<string, unknown> = {
    url,
    consent: options.consent,
    store: options.store,
  };
  if (options.context !== undefined) {
    body.context = options.context;
  }
  if (options.title !== undefined && options.title !== "") {
    body.title = options.title;
  }

  const res = await fetch(`${API_URL}/analyze/link`, {
    method: "POST",
    headers: await authHeaders(),
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    // FastAPI surfaces user-facing errors as { detail: "..." }; keep that text
    // so 422/413 explanations reach the user verbatim.
    let detail: string | undefined;
    try {
      const j = (await res.json()) as { detail?: unknown };
      if (typeof j?.detail === "string") detail = j.detail;
    } catch {
      // Non-JSON body — fall back to the status-only message.
    }
    const err = new Error(detail ?? `API error: ${res.status}`) as Error & {
      status?: number;
      detail?: string;
    };
    err.status = res.status;
    err.detail = detail;
    throw err;
  }

  return (await res.json()) as UploadAnalyzeResult;
}

// --- Submit-and-poll analysis jobs ------------------------------------------
// A link download or a chunked-upload completion is a multi-minute synchronous
// request today; Android backgrounding / socket loss routinely kills the
// response the server already finished producing, so the user sees an error on
// work that actually succeeded. The job endpoints run that same pipeline as a
// background task and expose staged progress the client polls (~every 3s),
// decoupling the result from one fragile long-lived connection.

/** The server-reported lifecycle of an analysis job. `stalled` is COMPUTED by
 *  the server on read (a non-terminal job whose state stopped advancing — e.g. an
 *  in-process task orphaned by an instance restart), so the client never spins
 *  forever. */
export type JobStatus =
  | "queued"
  | "downloading"
  | "transcribing"
  | "analyzing"
  | "storing"
  | "done"
  | "failed"
  | "stalled";

/** 202 body of the job-submit endpoints — the id to poll with. */
export interface JobCreated {
  job_id: string;
}

/** GET /analyze/jobs/{id} — a job's staged progress (and its result once done). */
export interface AnalyzeJobState {
  job_id: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  stage_started_at: string | null;
  // A human string for the current stage (e.g. "38 MB downloaded"), never a
  // fabricated percentage — the server is honest about what it actually knows.
  progress_note: string | null;
  // Known once the recording has been decoded/transcribed; drives the client
  // ETA. Null until then.
  duration_seconds: number | null;
  // Download heartbeat for the `downloading` stage: bytes fetched so far and the
  // total when the source advertised a Content-Length. Both OPTIONAL — a server
  // without download heartbeats omits them entirely, so the client must render
  // defensively (absent ⇒ no byte bar). Never a fabricated percentage.
  bytes_downloaded?: number | null;
  bytes_total?: number | null;
  // Honest failure detail — the SAME message the synchronous path would 4xx/5xx
  // with. Null unless `status === "failed"`.
  error: string | null;
  // The full analysis, present ONLY when `status === "done"`.
  result: UploadAnalyzeResult | null;
}

/**
 * POST /analyze/link/jobs — submit a link analysis as a background job. Returns
 * `{ job_id }` (202) to poll via {@link getAnalyzeJob}. Throws `API error:
 * <status>` on any non-OK with the numeric `.status` attached, so the caller can
 * FALL BACK to the synchronous {@link postAnalyzeLink} on 404/503 (old server /
 * storage off) and surface a 422/413 `.detail` verbatim otherwise.
 */
export async function postAnalyzeLinkJob(
  url: string,
  options: AnalyzeLinkOptions,
): Promise<JobCreated> {
  const body: Record<string, unknown> = {
    url,
    consent: options.consent,
    store: options.store,
  };
  if (options.context !== undefined) {
    body.context = options.context;
  }
  if (options.title !== undefined && options.title !== "") {
    body.title = options.title;
  }
  const res = await fetch(`${API_URL}/analyze/link/jobs`, {
    method: "POST",
    headers: await authHeaders(),
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw await jobPostError(res);
  }
  return (await res.json()) as JobCreated;
}

/**
 * POST /uploads/{id}/complete/jobs — submit a completed chunked upload's
 * analysis as a background job. Used by {@link postAnalyzeUploadChunkedJob}
 * after the parts are uploaded; returns `{ job_id }` (202) or throws
 * `API error: <status>` (with `.status`) so the orchestrator can fall back to
 * the synchronous complete on 404/503.
 */
export async function postUploadCompleteJob(
  uploadId: string,
): Promise<JobCreated> {
  const res = await fetch(
    `${API_URL}/uploads/${encodeURIComponent(uploadId)}/complete/jobs`,
    { method: "POST", headers: await authHeaders() },
  );
  if (!res.ok) {
    throw await jobPostError(res);
  }
  return (await res.json()) as JobCreated;
}

/**
 * GET /analyze/jobs/{id} — poll a job's state. Throws `API error: <status>` on
 * any non-OK (401/404/503) so the caller can stop polling and surface an honest
 * state rather than an eternal spinner.
 */
export async function getAnalyzeJob(jobId: string): Promise<AnalyzeJobState> {
  const res = await fetch(
    `${API_URL}/analyze/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET", headers: await authHeaders() },
  );
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return (await res.json()) as AnalyzeJobState;
}

/** Build an Error for a failed job-submit POST, carrying `.status` (and the
 *  server's `.detail` when present, e.g. a link 422/413 written for the user)
 *  so callers can both branch on the code and render the message verbatim. */
async function jobPostError(res: Response): Promise<Error> {
  let detail: string | undefined;
  try {
    const j = (await res.json()) as { detail?: unknown };
    if (typeof j?.detail === "string") detail = j.detail;
  } catch {
    // Non-JSON body — fall back to the status-only message.
  }
  const err = new Error(detail ?? `API error: ${res.status}`) as Error & {
    status?: number;
    detail?: string;
  };
  err.status = res.status;
  err.detail = detail;
  return err;
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
  // Optional human title the user gave the conversation (e.g. via the Analyze
  // "Name this conversation" field or an in-place rename). Absent on servers
  // that don't support titling yet — the UI falls back to `filename`.
  title?: string | null;
  media_type: MediaType;
  // Null when the server couldn't determine a duration (decode degraded and
  // the transcript carried no end time) — type matches the wire honestly.
  duration_seconds: number | null;
  has_analysis: boolean;
  // Provenance: "upload" | "link" (present on newer servers).
  source_type?: string;
}

/** One stored turn. Unlike a pasted transcript, a recorded conversation always
 *  carries timing (that's what the playhead syncs against). */
export interface RecordingTurn {
  speaker: string;
  text: string;
  start_time: number;
  end_time: number;
}

/** Provenance of a stored recording. `upload` = we hold the only copy;
 *  `link` = the user linked their own hosted media (a durable share/direct URL
 *  we can re-resolve for HD replay from the original source). */
export interface RecordingSource {
  type: "upload" | "link";
  // The durable user-provided URL for a link source; null for an upload (or a
  // link recording stored before the url was kept).
  url: string | null;
  original_filename?: string | null;
}

/** GET /recordings/{id} — the summary fields plus the analyzed transcript and
 *  full dynamics analysis. `analysis.per_turn` is index-aligned with `turns`. */
export interface RecordingDetail extends RecordingSummary {
  turns: RecordingTurn[];
  // Null on the wire when a recording was stored without a completed analysis
  // (rare); the UI must treat it as absent rather than assume it.
  analysis: AnalyzeResult | null;
  // Provenance (type/url). Optional: older servers omit it entirely, in which
  // case replay uses the stored-derivative path (as for an upload).
  source?: RecordingSource;
}

/** GET /recordings/{id}/media_url — a short-lived signed URL for playback. */
export interface RecordingMediaUrl {
  url: string;
  expires_in: number; // seconds until the signed URL expires
}

/** GET /recordings/{id}/source_url — the CURRENT direct media URL for a
 *  link-sourced recording, re-resolved server-side from the durable share link
 *  so the client can stream the user's own HD original straight from its CDN.
 *  The URL `may expire`; the caller refetches (or falls back to the derivative)
 *  on failure. */
export interface RecordingSourceUrl {
  url: string;
  content_type: string | null; // best-effort hint; null when unknown
  expires_hint: string;
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
 * GET /recordings/{id}/source_url — resolve the CURRENT direct media URL for a
 * LINK-sourced recording so the player can stream the user's own HD original
 * straight from its CDN (we never proxy the bytes). Only meaningful when the
 * detail's `source.type === "link"`; the server answers 404 for an upload or a
 * link recording whose url wasn't kept. Throws `API error: <status>` on any
 * non-OK (401/404/422/502/503) so the caller can fall back to the stored
 * derivative (`getRecordingMediaUrl`) rather than showing a broken player.
 */
export async function getRecordingSourceUrl(
  id: string,
): Promise<RecordingSourceUrl> {
  const res = await fetch(
    `${API_URL}/recordings/${encodeURIComponent(id)}/source_url`,
    {
      method: "GET",
      headers: await authHeaders(),
    },
  );
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return (await res.json()) as RecordingSourceUrl;
}

/** Result of PATCH /recordings/{id}/source — the recording is now link-sourced,
 *  so subsequent replay resolves the user's own hosted original (HD-first). */
export interface PatchSourceResult {
  type: "link";
  url: string;
  original_filename: string | null;
}

/**
 * PATCH /recordings/{id}/source — attach (or replace) the durable share/direct
 * link to the user's OWN hosted original for a stored recording, so replay can
 * stream it in HD instead of our stored derivative. Body is `{ url }`.
 *
 * Like postAnalyzeLink, a 422 means the link is unusable and its `detail` is
 * written for the user — surfaced verbatim on the thrown error (`.detail`) while
 * `.status` carries the code for the generic 404/503 branches. On 200 the server
 * echoes the new link source; the caller refetches the recording so HD-first
 * playback kicks in immediately.
 */
export async function patchRecordingSource(
  id: string,
  url: string,
): Promise<PatchSourceResult> {
  const res = await fetch(
    `${API_URL}/recordings/${encodeURIComponent(id)}/source`,
    {
      method: "PATCH",
      headers: await authHeaders(),
      body: JSON.stringify({ url }),
    },
  );
  if (!res.ok) {
    // A 422 carries a user-facing explanation (unusable link) — keep it verbatim.
    let detail: string | undefined;
    try {
      const j = (await res.json()) as { detail?: unknown };
      if (typeof j?.detail === "string") detail = j.detail;
    } catch {
      // Non-JSON body — fall back to the status-only message.
    }
    const err = new Error(detail ?? `API error: ${res.status}`) as Error & {
      status?: number;
      detail?: string;
    };
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  return (await res.json()) as PatchSourceResult;
}

/** Result of PATCH /recordings/{id} with a `{ title }` body — the recording's
 *  new human name. */
export interface PatchTitleResult {
  id: string;
  title: string;
}

/**
 * PATCH /recordings/{id} — rename a stored recording (body `{ title }`), so
 * replay/list show the user's name instead of the raw upload filename.
 *
 * CAPABILITY-GATED: as of this writing the backend has no title field or this
 * route (the recordings path only supports GET/DELETE + PATCH …/source), so the
 * server answers 4xx (typically 405/404). The caller MUST catch that and show an
 * honest "renaming isn't supported yet" message rather than pretend it worked —
 * the thrown Error carries `.status` so the caller can tell an unsupported route
 * (4xx) from a transient failure (5xx/network). Once the backend adds the field
 * this call starts succeeding with no client change.
 */
export async function patchRecordingTitle(
  id: string,
  title: string,
): Promise<PatchTitleResult> {
  const res = await fetch(`${API_URL}/recordings/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: await authHeaders(),
    body: JSON.stringify({ title }),
  });
  if (!res.ok) {
    const err = new Error(`API error: ${res.status}`) as Error & {
      status?: number;
    };
    err.status = res.status;
    throw err;
  }
  return (await res.json()) as PatchTitleResult;
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

// --- Voice enrollment ("This is me" → auto-label "You") --------------------

/** GET /voice/profile — whether the server can do voice ID and whether this
 *  user has enrolled. `available` is false when the optional embedding deps
 *  aren't installed server-side; the UI hides the "This is me" affordance then.
 *  The embedding vector itself is never returned. */
export interface VoiceProfile {
  available: boolean;
  storage_enabled: boolean;
  enrolled: boolean;
  enroll_count: number;
  updated_at?: string | null;
  model?: string | null;
  dim?: number | null;
}

/** POST /voice/enroll response — confirmation the voiceprint was saved/refined. */
export interface EnrollResult {
  enrolled: boolean;
  speaker: string;
  enroll_count: number;
  dim: number;
  updated_at: string;
  // Plain-language statement of exactly what is stored (biometric transparency).
  stored: string;
}

/**
 * GET /voice/profile — voice-ID availability + this user's enrollment status.
 * Never throws for the "not available"/"not enrolled" cases (those are normal
 * states carried in the body); throws only on a real transport/auth failure.
 */
export async function getVoiceProfile(): Promise<VoiceProfile> {
  const res = await fetch(`${API_URL}/voice/profile`, {
    method: "GET",
    headers: await authHeaders(),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return (await res.json()) as VoiceProfile;
}

/**
 * POST /voice/enroll — "This is me": teach the server your voice from one
 * diarized speaker in a stored recording. The error carries `.status` so the UI
 * can distinguish an honest 503 (voice ID not available on this server) or 422
 * (too little of that speaker's voice) from a transient failure.
 */
export async function enrollVoice(
  recordingId: string,
  speaker: string,
): Promise<EnrollResult> {
  const res = await fetch(`${API_URL}/voice/enroll`, {
    method: "POST",
    headers: await authHeaders(),
    body: JSON.stringify({ recording_id: recordingId, speaker }),
  });
  if (!res.ok) {
    let detail = "";
    try {
      detail = ((await res.json()) as { detail?: string }).detail ?? "";
    } catch {
      // non-JSON body — leave detail empty
    }
    const err = new Error(detail || `API error: ${res.status}`) as Error & {
      status?: number;
    };
    err.status = res.status;
    throw err;
  }
  return (await res.json()) as EnrollResult;
}

/**
 * DELETE /voice/voiceprint — "Forget my voice": really delete the stored
 * biometric signature. Resolves to whether one existed (idempotent); throws
 * `API error: <status>` on a non-OK so the UI can surface it honestly.
 */
export async function forgetVoice(): Promise<{ deleted: boolean }> {
  const res = await fetch(`${API_URL}/voice/voiceprint`, {
    method: "DELETE",
    headers: await authHeaders(),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return (await res.json()) as { deleted: boolean };
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
