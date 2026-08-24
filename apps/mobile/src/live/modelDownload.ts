/**
 * Download-once + revalidate-on-launch for the ECAPA ONNX model the fast
 * loop's speaker-ID runs (`GET|HEAD /models/ecapa.onnx`, server/routers/
 * models.py). Pure: the filesystem and `fetch` are injected seams, so the
 * whole protocol is unit-tested with fakes (liveModelDownload.test.ts) and
 * `ortNative.ts` only supplies the expo-file-system implementation.
 *
 * Why HEAD-first: the server answers `If-None-Match` from the pinned model
 * revision alone (its `ETag`), so a launch re-check is one tiny request with
 * no body. React Native's `fetch` buffers a whole response before resolving,
 * so a conditional GET that turned out to be a 200 would pull all ~80 MB
 * through JS memory — the file body is fetched ONLY through the native
 * downloader (`File.downloadFileAsync`) once HEAD says the revision changed
 * (or nothing is cached yet). The ETag is remembered in a sidecar next to
 * the model so the next launch can revalidate.
 *
 * Honesty rules (the reasons surface in the session's "On-device: …" status
 * line, never as an error toast — speaker-ID is optional):
 *
 * - 503 = the server has no model and can't make one (voice deps absent):
 *   speaker-ID stays off; the server's reason header is passed through.
 * - offline with a cached model: use it (`cached-offline`) — the model is a
 *   pure function of the revision, a day-old copy is not stale in any way
 *   that matters; offline without one: off.
 * - a download whose size disagrees with `Content-Length` (or is
 *   implausibly small — an HTML error page) is discarded, and any previous
 *   good copy is kept: a bad download never replaces a working model.
 */

export const ECAPA_FILENAME = "ecapa.onnx";
/** The real export is ~80 MB; a 404 page or a truncated stream is not a model. */
export const MIN_MODEL_BYTES = 1_000_000;

export interface ModelFileStat {
  exists: boolean;
  size: number;
}

/** The slice of a filesystem the download protocol needs. `ortNative.ts`
 *  implements it over expo-file-system; tests use an in-memory map. */
export interface ModelFileStore {
  stat(name: string): Promise<ModelFileStat>;
  /** Filesystem path ORT can open (no `file://`). */
  pathOf(name: string): string;
  readText(name: string): Promise<string | null>;
  writeText(name: string, text: string): Promise<void>;
  /** Fetch `url` straight to disk under `name` (overwriting). Resolves with
   *  the resulting file's stat; may throw on a network/HTTP failure. */
  download(url: string, name: string, headers: Record<string, string>): Promise<ModelFileStat>;
  move(from: string, to: string): Promise<void>;
  remove(name: string): Promise<void>;
}

export interface HeadResponseLike {
  status: number;
  ok: boolean;
  headers: { get(name: string): string | null };
}

export type FetchLike = (
  url: string,
  init: { method: string; headers: Record<string, string> },
) => Promise<HeadResponseLike>;

export type EcapaModelSource = "cached" | "cached-offline" | "downloaded" | "updated";

export type EcapaUnavailableCode =
  | "server-503"
  | "auth"
  | "not-found"
  | "http"
  | "network"
  | "bad-download";

export type EcapaModelResult =
  | { status: "ready"; path: string; source: EcapaModelSource; etag: string | null }
  | { status: "unavailable"; code: EcapaUnavailableCode; reason: string };

export interface ResolveEcapaModelOptions {
  url: string;
  headers: Record<string, string>;
  fetch: FetchLike;
  store: ModelFileStore;
  filename?: string;
  minBytes?: number;
}

export function etagSidecarName(filename: string): string {
  return `${filename}.etag`;
}

/** The revision inside a (possibly weak, possibly quoted) ETag — what the
 *  server pins the model to, and what a person's `model` field ends with. */
export function revisionFromEtag(etag: string | null | undefined): string | null {
  if (!etag) return null;
  let tag = etag.trim();
  if (tag.startsWith("W/")) tag = tag.slice(2);
  if (tag.startsWith('"') && tag.endsWith('"')) tag = tag.slice(1, -1);
  return tag || null;
}

function describeHttp(status: number): string {
  return `model endpoint answered ${status}`;
}

/**
 * Make sure the ECAPA model is on disk and current. Never throws; every
 * failure mode becomes an `unavailable` result with a human-readable reason.
 */
export async function resolveEcapaModel(opts: ResolveEcapaModelOptions): Promise<EcapaModelResult> {
  const filename = opts.filename ?? ECAPA_FILENAME;
  const minBytes = opts.minBytes ?? MIN_MODEL_BYTES;
  const { store } = opts;
  const sidecar = etagSidecarName(filename);

  let cached: ModelFileStat;
  try {
    cached = await store.stat(filename);
  } catch {
    cached = { exists: false, size: 0 };
  }
  const haveFile = cached.exists && cached.size >= minBytes;
  let etag: string | null = null;
  if (haveFile) {
    try {
      etag = (await store.readText(sidecar))?.trim() || null;
    } catch {
      etag = null;
    }
  }

  // 1. Revalidate (or discover) with HEAD.
  let head: HeadResponseLike;
  try {
    head = await opts.fetch(opts.url, {
      method: "HEAD",
      headers: { ...opts.headers, ...(etag ? { "If-None-Match": etag } : {}) },
    });
  } catch (err) {
    if (haveFile) {
      return { status: "ready", path: store.pathOf(filename), source: "cached-offline", etag };
    }
    const msg = err instanceof Error ? err.message : String(err);
    return { status: "unavailable", code: "network", reason: `offline and no cached model (${msg})` };
  }

  if (head.status === 304 && haveFile) {
    return { status: "ready", path: store.pathOf(filename), source: "cached", etag };
  }
  if (head.status === 503) {
    const reason = head.headers.get("x-model-unavailable") || "server has no ECAPA model (503)";
    return { status: "unavailable", code: "server-503", reason };
  }
  if (head.status === 401 || head.status === 403) {
    return { status: "unavailable", code: "auth", reason: `not signed in (${head.status})` };
  }
  if (head.status === 404) {
    return { status: "unavailable", code: "not-found", reason: "server has no model endpoint (404)" };
  }
  if (!head.ok) {
    // A 304 with nothing cached is a server quirk we treat like any other
    // non-success: fall through to "unavailable" rather than guess.
    return { status: "unavailable", code: "http", reason: describeHttp(head.status) };
  }

  const serverEtag = head.headers.get("etag");
  if (haveFile && etag && serverEtag && serverEtag === etag) {
    return { status: "ready", path: store.pathOf(filename), source: "cached", etag };
  }

  // 2. Download to a temp name; only a verified file replaces the old one.
  const expected = Number(head.headers.get("content-length"));
  const tmp = `${filename}.download`;
  let got: ModelFileStat;
  try {
    got = await store.download(opts.url, tmp, opts.headers);
  } catch (err) {
    await store.remove(tmp).catch(() => {});
    if (haveFile) {
      return { status: "ready", path: store.pathOf(filename), source: "cached-offline", etag };
    }
    const msg = err instanceof Error ? err.message : String(err);
    return { status: "unavailable", code: "network", reason: `model download failed (${msg})` };
  }
  const sizeOk =
    got.exists && got.size >= minBytes && (!(expected > 0) || got.size === expected);
  if (!sizeOk) {
    await store.remove(tmp).catch(() => {});
    if (haveFile) {
      // Keep the known-good copy rather than a truncated update.
      return { status: "ready", path: store.pathOf(filename), source: "cached", etag };
    }
    return {
      status: "unavailable",
      code: "bad-download",
      reason: `model download was ${got.size} bytes (expected ${expected > 0 ? expected : `≥ ${minBytes}`})`,
    };
  }
  try {
    await store.move(tmp, filename);
    if (serverEtag) await store.writeText(sidecar, serverEtag);
    else await store.remove(sidecar).catch(() => {});
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { status: "unavailable", code: "bad-download", reason: `could not store the model (${msg})` };
  }
  return {
    status: "ready",
    path: store.pathOf(filename),
    source: haveFile ? "updated" : "downloaded",
    etag: serverEtag,
  };
}
