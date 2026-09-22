/**
 * The two-sided therapist setup (server/routers/therapist.py):
 *
 * - patient side: `GET/PUT/PATCH/DELETE /therapist/link` — name ONE
 *   therapist account by email; "share sessions automatically" is a flag
 *   on that link. The link grants nothing by itself: every episode the
 *   therapist sees is the existing per-episode share grant (client.ts's
 *   postShare), made automatically at ingest when the flag is on.
 * - therapist side: `GET /therapist/patients` (+ accept / decline / seen).
 * - viewer-private notes on an episode: `GET/PUT /therapist/notes/{id}`.
 * - consent (server/consent.py): `POST /therapist/consent` grants or revokes
 *   ONE scope. `episodes` = they may read stored sessions (granted by
 *   PUT /therapist/link itself); `live` = they may observe a call as it
 *   happens. The exact sentence the patient must be shown comes back on
 *   `GET /therapist/link` as `consent.scopes[scope].disclosure` — LINKED OR
 *   NOT — and the client renders the server's wording, never its own copy:
 *   the stored record cites a `text_version`, and a client-side sentence
 *   would silently drift from it.
 *
 * Auth mirrors client.ts's authHeaders. Every non-OK throws
 * `Error("API error: <status>")` with `.status` (and the server's `detail`
 * verbatim as `.detail` when it wrote a user-facing one — "no MindShift
 * account with that email", "you can't be your own therapist").
 */
import { getFreshToken } from "../auth/authToken";

const API_URL = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

export type LinkStatus = "pending" | "accepted";

/** The two things a patient can agree to, separately (server/consent.py). */
export type ConsentScope = "episodes" | "live";

/** One scope of `GET /therapist/link`'s consent block. `disclosure` is the
 *  wording in force TODAY (render this, never a client copy);
 *  `granted_text_version` is the wording that was on screen when they
 *  agreed — different values mean the sentence has been reworded since. */
export interface ConsentScopeView {
  granted: boolean;
  at?: string | null;
  granted_text_version?: string | null;
  disclosure?: string | null;
}

export interface ConsentView {
  text_version?: string | null;
  scopes?: Partial<Record<ConsentScope, ConsentScopeView>>;
}

export interface TherapistLink {
  linked: boolean;
  therapist_email?: string | null;
  status?: LinkStatus;
  auto_share?: boolean;
  created_at?: string | null;
  accepted_at?: string | null;
  /** Present on any server that has consent; absent on an older one. */
  consent?: ConsentView;
}

export interface PatientLink {
  patient_uid: string;
  patient_email: string | null;
  status: LinkStatus;
  auto_share: boolean;
  created_at: string | null;
  accepted_at: string | null;
  /** Which scopes this patient has consented to — a therapist must be able
   *  to see that a patient revoked live observation. */
  consent_scopes?: ConsentScope[];
  /** When THIS therapist last marked the patient read (POST .../seen). */
  last_seen_at?: string | null;
}

/** The disclosure for one scope, or "" when the server did not send one.
 *  Never falls back to a hard-coded sentence: the empty string is the
 *  client's honest "I was not told what to show", and the caller must
 *  refuse to take consent rather than invent the wording. */
export function disclosureFor(
  link: TherapistLink | null | undefined,
  scope: ConsentScope,
): string {
  const text = link?.consent?.scopes?.[scope]?.disclosure;
  return typeof text === "string" ? text : "";
}

export function consentGranted(
  link: TherapistLink | null | undefined,
  scope: ConsentScope,
): boolean {
  return link?.consent?.scopes?.[scope]?.granted === true;
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

/** `POST /therapist/consent` — grant or revoke ONE scope, and get the whole
 *  link view back (including the consent block) so the card never guesses
 *  what the server stored. */
export async function setTherapistConsent(
  scope: ConsentScope,
  granted: boolean,
): Promise<TherapistLink> {
  const res = await fetch(`${API_URL}/therapist/consent`, {
    method: "POST",
    headers: await authHeaders(),
    body: JSON.stringify({ scope, granted }),
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

/** `POST /therapist/patients/{uid}/seen` — "I have read this patient up to
 *  now". Stored on the link, so every device this therapist uses agrees
 *  about which patients have something new. */
export async function markPatientSeen(patientUid: string): Promise<string | null> {
  const res = await fetch(
    `${API_URL}/therapist/patients/${encodeURIComponent(patientUid)}/seen`,
    { method: "POST", headers: await authHeaders(false) },
  );
  if (!res.ok) return raise(res);
  const body = (await res.json()) as { last_seen_at?: unknown };
  return typeof body?.last_seen_at === "string" ? body.last_seen_at : null;
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
