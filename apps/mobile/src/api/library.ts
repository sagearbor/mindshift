/**
 * Coach knowledge library — `/library/items` (server/routers/library.py).
 *
 * The user saves material once (typed notes, uploaded PDF / .txt / .md /
 * .docx / .csv / .xlsx) and picks items for a Live Coach session, which sends
 * their ids as `library_item_ids` on the WebSocket config frame. The server
 * keeps the text; the phone only ever holds ids, titles and status.
 *
 * Honest errors: every non-2xx rejects with an Error carrying `.status` and
 * the server's own `detail` as its message (413 "file is larger than…",
 * 422 "50 items", unreadable file…), never an invented one.
 *
 * Private: titles and contents never go into diagnostics or logs.
 */
import { Platform } from "react-native";
import { File as FSFile } from "expo-file-system";
import { getFreshToken } from "../auth/authToken";
import { reportPlanLimit } from "../store/planLimitStore";

const API_URL = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

export type LibraryKind = "note" | "document" | "table";
export type LibraryStatus = "processing" | "ready" | "failed";

export interface LibraryItem {
  id: string;
  title: string;
  kind: LibraryKind;
  chars: number;
  status: LibraryStatus;
  error?: string | null;
  created_at: string;
  updated_at: string;
  /** Chunks + vectors exist (retrieval can use this item). */
  indexed: boolean;
  /** Containing project; null/absent = top level (older servers omit it). */
  project_id?: string | null;
}

/** A folder. `parent_id` null = top level; nesting is capped server-side. */
export interface LibraryProject {
  id: string;
  name: string;
  parent_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface LibraryItemDetail extends LibraryItem {
  filename?: string | null;
  content_type?: string | null;
  chunk_count: number;
  /** The first ~2,000 characters of the text. */
  preview: string;
  preview_truncated: boolean;
  /** The whole text — only when fetched with `{ full: true }` (`?full=1`),
   *  capped server-side at the 2 MB item limit. Absent from older servers. */
  text?: string | null;
}

export interface LibraryLimits {
  max_file_bytes: number;
  max_text_chars: number;
  max_items: number;
}

export interface LibraryList {
  items: LibraryItem[];
  /** Folders (absent from older servers = none). */
  projects?: LibraryProject[];
  /** False when the server cannot search inside documents (no embedding
   *  service): large selections can't be used yet, small ones still can. */
  retrieval_available: boolean;
  limits: LibraryLimits;
}

/** What expo-document-picker hands back for one asset (the fields we use). */
export interface PickedLibraryFile {
  uri: string;
  name: string;
  mimeType?: string | null;
  /** Web only: the real File object. */
  file?: File | null;
}

export type LibraryApiError = Error & { status?: number };

/** The honest line shown when the server cannot search inside documents
 *  (`retrieval_available: false`). */
export const RETRIEVAL_UNAVAILABLE_TEXT =
  "Search unavailable: large documents can't be used yet. Notes and short documents still work.";

async function headers(json: boolean): Promise<Record<string, string>> {
  const token = await getFreshToken();
  const h: Record<string, string> = json ? { "Content-Type": "application/json" } : {};
  if (token) h.Authorization = `Bearer ${token}`;
  return h;
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    // Typed plan-limit 429 (server/plans.py): open the upgrade sheet and use
    // the server's friendly sentence as the error text.
    const planMsg = reportPlanLimit(detail);
    if (planMsg) return planMsg;
    if (Array.isArray(detail) && detail[0] && typeof detail[0].msg === "string") return detail[0].msg;
  } catch {
    // Non-JSON body: fall back to the status line.
  }
  return "";
}

async function call<T>(path: string, init: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    const err = new Error(
      (await errorMessage(res)) || `Library request failed (${res.status})`,
    ) as LibraryApiError;
    err.status = res.status;
    throw err;
  }
  return (await res.json()) as T;
}

function itemPath(id: string): string {
  return `/library/items/${encodeURIComponent(id)}`;
}

export async function listLibrary(): Promise<LibraryList> {
  return call<LibraryList>("/library/items", { method: "GET", headers: await headers(false) });
}

/** Item detail. `{ full: true }` also asks for the whole text (`?full=1`),
 *  which the edit flow needs for notes longer than the ~2,000-char preview. */
export async function getLibraryItem(
  id: string,
  opts?: { full?: boolean },
): Promise<LibraryItemDetail> {
  const path = opts?.full ? `${itemPath(id)}?full=1` : itemPath(id);
  return call<LibraryItemDetail>(path, { method: "GET", headers: await headers(false) });
}

export async function createLibraryNote(
  title: string,
  text: string,
  projectId?: string | null,
): Promise<LibraryItem> {
  const body: Record<string, unknown> = { kind: "note", title, text };
  if (projectId) body.project_id = projectId;
  return call<LibraryItem>("/library/items", {
    method: "POST",
    headers: await headers(true),
    body: JSON.stringify(body),
  });
}

/** "Brief.final.pdf" -> "Brief.final"; a name with no extension is kept. */
export function titleFromFilename(name: string): string {
  const base = (name || "").split(/[\\/]/).pop() || "";
  const dot = base.lastIndexOf(".");
  const stem = dot > 0 ? base.slice(0, dot) : base;
  return stem.trim();
}

/** Multipart upload of a picked file. No Content-Type header: fetch sets the
 *  multipart boundary itself. Native sends the file as an expo-file-system
 *  File (a Blob streamed from disk; Expo's fetch rejects the legacy
 *  `{uri, name, type}` descriptor, see client.ts postAnalyzeUpload). */
export async function uploadLibraryFile(
  picked: PickedLibraryFile,
  title?: string,
  projectId?: string | null,
): Promise<LibraryItem> {
  const form = new FormData();
  if (Platform.OS === "web" && picked.file) {
    form.append("file", picked.file, picked.name);
  } else {
    form.append("file", new FSFile(picked.uri) as unknown as Blob, picked.name);
  }
  // Explicit title always: on Android the FormData filename arg is ignored
  // for an expo File blob, so the server would title it from the cache-copy
  // UUID. The server's filename stem stays as the fallback.
  const finalTitle = (title && title.trim()) || titleFromFilename(picked.name);
  if (finalTitle) form.append("title", finalTitle);
  if (projectId) form.append("project_id", projectId);
  return call<LibraryItem>("/library/items", {
    method: "POST",
    headers: await headers(false),
    body: form,
  });
}

/** Rename (any item) and/or replace the text (notes only). */
export async function updateLibraryItem(
  id: string,
  patch: { title?: string; text?: string; project_id?: string | null },
): Promise<LibraryItem> {
  return call<LibraryItem>(itemPath(id), {
    method: "PATCH",
    headers: await headers(true),
    body: JSON.stringify(patch),
  });
}

export async function deleteLibraryItem(id: string): Promise<{ deleted: boolean; id: string }> {
  return call(itemPath(id), { method: "DELETE", headers: await headers(false) });
}

export async function createLibraryProject(
  name: string,
  parentId?: string | null,
): Promise<LibraryProject> {
  return call<LibraryProject>("/library/projects", {
    method: "POST",
    headers: await headers(true),
    body: JSON.stringify({ name, parent_id: parentId ?? null }),
  });
}

/** Rename and/or move (`parent_id: null` = top level). */
export async function updateLibraryProject(
  id: string,
  patch: { name?: string; parent_id?: string | null },
): Promise<LibraryProject> {
  return call<LibraryProject>(`/library/projects/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: await headers(true),
    body: JSON.stringify(patch),
  });
}

/** Deleting a non-empty project is refused (409) unless `contents` is
 *  "move_to_parent" (files and sub-projects move up one level). */
export async function deleteLibraryProject(
  id: string,
  contents?: "move_to_parent",
): Promise<{ deleted: boolean; id: string }> {
  const q = contents ? `?contents=${contents}` : "";
  return call(`/library/projects/${encodeURIComponent(id)}${q}`, {
    method: "DELETE",
    headers: await headers(false),
  });
}
