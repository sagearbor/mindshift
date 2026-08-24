/**
 * The two-sided therapist setup (server/routers/therapist.py):
 *
 * - patient side: `GET/PUT/PATCH/DELETE /therapist/link` — name ONE
 *   therapist account by email; "share sessions automatically" is a flag
 *   on that link. The link grants nothing by itself: every episode the
 *   therapist sees is the existing per-episode share grant (client.ts's
 *   postShare), made automatically at ingest when the flag is on.
 * - therapist side: `GET /therapist/patients` (+ accept / decline).
 * - viewer-private notes on an episode: `GET/PUT /therapist/notes/{id}`.
 *
 * Auth mirrors client.ts's authHeaders. Every non-OK throws
 * `Error("API error: <status>")` with `.status` (and the server's `detail`
 * verbatim as `.detail` when it wrote a user-facing one — "no MindShift
 * account with that email", "you can't be your own therapist").
 */
import { getFreshToken } from "../auth/authToken";

const API_URL = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

export type LinkStatus = "pending" | "accepted";

export interface TherapistLink {
  linked: boolean;
  therapist_email?: string | null;
  status?: LinkStatus;
  auto_share?: boolean;
  created_at?: string | null;
  accepted_at?: string | null;
}

export interface PatientLink {
  patient_uid: string;
  patient_email: string | null;
  status: LinkStatus;
  auto_share: boolean;
  created_at: string | null;
  accepted_at: string | null;
}

export interface SessionNote {
  episode_id: string;
  text: string;
  updated_at: string | null;
}

export interface ApiError extends Error {
  status?: number;
  detail?: string;
}

async function authHeaders(json = true): Promise<Record<string, string>> {
  const token = await getFreshToken();
  const headers: Record<string, string> = json ? { "Content-Type": "application/json" } : {};
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

async function raise(res: Response): Promise<never> {
  let detail: string | undefined;
  try {
    const j = (await res.json()) as { detail?: unknown };
    if (typeof j?.detail === "string") detail = j.detail;
  } catch {
    // Non-JSON body — status-only message.
  }
  const err = new Error(detail ?? `API error: ${res.status}`) as ApiError;
  err.status = res.status;
  err.detail = detail;
  throw err;
}

export async function getTherapistLink(): Promise<TherapistLink> {
  const res = await fetch(`${API_URL}/therapist/link`, {
    method: "GET",
    headers: await authHeaders(false),
  });
  if (!res.ok) return raise(res);
  const data = (await res.json()) as TherapistLink;
  return { ...data, linked: data.linked === true };
}

export async function setTherapistLink(email: string): Promise<TherapistLink> {
  const res = await fetch(`${API_URL}/therapist/link`, {
    method: "PUT",
    headers: await authHeaders(),
    body: JSON.stringify({ email }),
  });
  if (!res.ok) return raise(res);
  return (await res.json()) as TherapistLink;
}

export async function setAutoShare(autoShare: boolean): Promise<TherapistLink> {
  const res = await fetch(`${API_URL}/therapist/link`, {
    method: "PATCH",
    headers: await authHeaders(),
    body: JSON.stringify({ auto_share: autoShare }),
  });
  if (!res.ok) return raise(res);
  return (await res.json()) as TherapistLink;
}

export async function unlinkTherapist(): Promise<void> {
  const res = await fetch(`${API_URL}/therapist/link`, {
    method: "DELETE",
    headers: await authHeaders(false),
  });
  if (!res.ok) return raise(res);
}

export async function listPatients(): Promise<PatientLink[]> {
  const res = await fetch(`${API_URL}/therapist/patients`, {
    method: "GET",
    headers: await authHeaders(false),
  });
  if (!res.ok) return raise(res);
  const data = (await res.json()) as { patients?: PatientLink[] };
  return Array.isArray(data.patients) ? data.patients : [];
}

export async function acceptPatient(patientUid: string): Promise<PatientLink> {
  const res = await fetch(
    `${API_URL}/therapist/patients/${encodeURIComponent(patientUid)}/accept`,
    { method: "POST", headers: await authHeaders(false) },
  );
  if (!res.ok) return raise(res);
  return (await res.json()) as PatientLink;
}

export async function declinePatient(patientUid: string): Promise<void> {
  const res = await fetch(
    `${API_URL}/therapist/patients/${encodeURIComponent(patientUid)}/decline`,
    { method: "POST", headers: await authHeaders(false) },
  );
  if (!res.ok) return raise(res);
}

export async function getSessionNote(episodeId: string): Promise<SessionNote> {
  const res = await fetch(
    `${API_URL}/therapist/notes/${encodeURIComponent(episodeId)}`,
    { method: "GET", headers: await authHeaders(false) },
  );
  if (!res.ok) return raise(res);
  return (await res.json()) as SessionNote;
}

export async function putSessionNote(
  episodeId: string,
  text: string,
): Promise<SessionNote> {
  const res = await fetch(
    `${API_URL}/therapist/notes/${encodeURIComponent(episodeId)}`,
    { method: "PUT", headers: await authHeaders(), body: JSON.stringify({ text }) },
  );
  if (!res.ok) return raise(res);
  return (await res.json()) as SessionNote;
}
