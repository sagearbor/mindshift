/**
 * src/live/modelDownload.ts — the download-once + ETag-revalidate protocol
 * for GET|HEAD /models/ecapa.onnx, driven with an in-memory file store and a
 * scripted fetch. Every branch the session status line can report:
 * fresh download, cached + 304, cached + new revision, 503 (server has no
 * model) with the reason header, 401/404/other HTTP, offline with and
 * without a cache, and a bad download that must never replace a good model.
 */
import {
  ECAPA_FILENAME,
  etagSidecarName,
  resolveEcapaModel,
  revisionFromEtag,
  type HeadResponseLike,
  type ModelFileStat,
  type ModelFileStore,
} from "../src/live/modelDownload";

const MIN = 100; // keep fake "models" small
const GOOD = 1234;
const ETAG_A = '"rev-a"';
const ETAG_B = '"rev-b"';

class MemoryStore implements ModelFileStore {
  files = new Map<string, { size: number; text?: string }>();
  downloads: { url: string; name: string; headers: Record<string, string> }[] = [];
  /** What the next download produces (size), or an Error to throw. */
  nextDownload: number | Error = GOOD;

  private statOf(name: string): ModelFileStat {
    const f = this.files.get(name);
    return f ? { exists: true, size: f.size } : { exists: false, size: 0 };
  }
  async stat(name: string) {
    return this.statOf(name);
  }
  pathOf(name: string) {
    return `/docs/models/${name}`;
  }
  async readText(name: string) {
    return this.files.get(name)?.text ?? null;
  }
  async writeText(name: string, text: string) {
    this.files.set(name, { size: text.length, text });
  }
  async download(url: string, name: string, headers: Record<string, string>) {
    this.downloads.push({ url, name, headers });
    if (this.nextDownload instanceof Error) throw this.nextDownload;
    this.files.set(name, { size: this.nextDownload });
    return this.stat(name);
  }
  async move(from: string, to: string) {
    const f = this.files.get(from);
    if (!f) throw new Error(`no ${from}`);
    this.files.delete(from);
    this.files.set(to, f);
  }
  async remove(name: string) {
    this.files.delete(name);
  }
}
function head(status: number, headers: Record<string, string> = {}): HeadResponseLike {
  const lower = new Map(Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v]));
  return { status, ok: status >= 200 && status < 300, headers: { get: (n) => lower.get(n.toLowerCase()) ?? null } };
}

function scripted(responses: (HeadResponseLike | Error)[]) {
  const calls: { url: string; init: { method: string; headers: Record<string, string> } }[] = [];
  const fetch = async (url: string, init: { method: string; headers: Record<string, string> }) => {
    calls.push({ url, init });
    const next = responses.shift();
    if (!next) throw new Error("unexpected fetch");
    if (next instanceof Error) throw next;
    return next;
  };
  return { fetch, calls };
}

const URL = "https://api.example/models/ecapa.onnx";
const AUTH = { Authorization: "Bearer tok" };

function withCache(store: MemoryStore, etag: string | null = ETAG_A) {
  store.files.set(ECAPA_FILENAME, { size: GOOD });
  if (etag) store.files.set(etagSidecarName(ECAPA_FILENAME), { size: etag.length, text: etag });
}

describe("revisionFromEtag", () => {
  it("strips quotes and weak markers", () => {
    expect(revisionFromEtag('"abc"')).toBe("abc");
    expect(revisionFromEtag('W/"abc"')).toBe("abc");
    expect(revisionFromEtag("abc")).toBe("abc");
    expect(revisionFromEtag("")).toBeNull();
    expect(revisionFromEtag(null)).toBeNull();
  });
});

describe("resolveEcapaModel", () => {
  it("downloads once on a cold start: HEAD with auth, then the body to a temp name, then the ETag sidecar", async () => {
    const store = new MemoryStore();
    const { fetch, calls } = scripted([head(200, { ETag: ETAG_A, "Content-Length": String(GOOD) })]);
    const res = await resolveEcapaModel({ url: URL, headers: AUTH, fetch, store, minBytes: MIN });
    expect(res).toEqual({ status: "ready", path: `/docs/models/${ECAPA_FILENAME}`, source: "downloaded", etag: ETAG_A });
    expect(calls).toHaveLength(1);
    expect(calls[0].init.method).toBe("HEAD");
    expect(calls[0].init.headers).toEqual(AUTH); // no If-None-Match without a cache
    expect(store.downloads).toEqual([{ url: URL, name: `${ECAPA_FILENAME}.download`, headers: AUTH }]);
    expect(store.files.get(ECAPA_FILENAME)?.size).toBe(GOOD);
    expect(store.files.has(`${ECAPA_FILENAME}.download`)).toBe(false);
    expect(store.files.get(etagSidecarName(ECAPA_FILENAME))?.text).toBe(ETAG_A);
  });

  it("revalidates a cached model with If-None-Match and keeps it on 304 (no download)", async () => {
    const store = new MemoryStore();
    withCache(store);
    const { fetch, calls } = scripted([head(304, { ETag: ETAG_A })]);
    const res = await resolveEcapaModel({ url: URL, headers: AUTH, fetch, store, minBytes: MIN });
    expect(res).toMatchObject({ status: "ready", source: "cached", etag: ETAG_A });
    expect(calls[0].init.headers).toEqual({ ...AUTH, "If-None-Match": ETAG_A });
    expect(store.downloads).toEqual([]);
  });

  it("treats a 200 with the same ETag as cached too", async () => {
    const store = new MemoryStore();
    withCache(store);
    const { fetch } = scripted([head(200, { ETag: ETAG_A, "Content-Length": String(GOOD) })]);
    const res = await resolveEcapaModel({ url: URL, headers: AUTH, fetch, store, minBytes: MIN });
    expect(res).toMatchObject({ status: "ready", source: "cached" });
    expect(store.downloads).toEqual([]);
  });

  it("re-downloads when the server's revision changed and replaces the sidecar", async () => {
    const store = new MemoryStore();
    withCache(store);
    store.nextDownload = GOOD + 10;
    const { fetch } = scripted([head(200, { ETag: ETAG_B, "Content-Length": String(GOOD + 10) })]);
    const res = await resolveEcapaModel({ url: URL, headers: AUTH, fetch, store, minBytes: MIN });
    expect(res).toMatchObject({ status: "ready", source: "updated", etag: ETAG_B });
    expect(store.files.get(ECAPA_FILENAME)?.size).toBe(GOOD + 10);
    expect(store.files.get(etagSidecarName(ECAPA_FILENAME))?.text).toBe(ETAG_B);
  });

  it("a cached file without a sidecar is revalidated without If-None-Match", async () => {
    const store = new MemoryStore();
    withCache(store, null);
    const { fetch, calls } = scripted([head(200, { ETag: ETAG_A, "Content-Length": String(GOOD) })]);
    const res = await resolveEcapaModel({ url: URL, headers: AUTH, fetch, store, minBytes: MIN });
    expect(calls[0].init.headers).toEqual(AUTH);
    // No etag to compare: the server copy is fetched and the sidecar written.
    expect(res).toMatchObject({ status: "ready", source: "updated", etag: ETAG_A });
    expect(store.files.get(etagSidecarName(ECAPA_FILENAME))?.text).toBe(ETAG_A);
  });

  it("503 => unavailable with the server's reason, speaker-ID stays off", async () => {
    const store = new MemoryStore();
    const { fetch } = scripted([head(503, { "X-Model-Unavailable": "voice deps not installed" })]);
    const res = await resolveEcapaModel({ url: URL, headers: AUTH, fetch, store, minBytes: MIN });
    expect(res).toEqual({ status: "unavailable", code: "server-503", reason: "voice deps not installed" });
    expect(store.downloads).toEqual([]);
    // Without the header there is still a plain-language reason.
    const plain = await resolveEcapaModel({ url: URL, headers: AUTH, fetch: scripted([head(503)]).fetch, store, minBytes: MIN });
    expect(plain).toMatchObject({ status: "unavailable", code: "server-503", reason: expect.stringContaining("503") });
  });

  it("401 / 404 / other statuses name the cause", async () => {
    const store = new MemoryStore();
    for (const [status, code] of [[401, "auth"], [403, "auth"], [404, "not-found"], [500, "http"]] as const) {
      const res = await resolveEcapaModel({ url: URL, headers: AUTH, fetch: scripted([head(status)]).fetch, store, minBytes: MIN });
      expect(res).toMatchObject({ status: "unavailable", code, reason: expect.stringContaining(String(status)) });
    }
  });

  it("offline: uses a cached model, otherwise reports network", async () => {
    const cached = new MemoryStore();
    withCache(cached);
    const res = await resolveEcapaModel({ url: URL, headers: AUTH, fetch: scripted([new Error("Network request failed")]).fetch, store: cached, minBytes: MIN });
    expect(res).toMatchObject({ status: "ready", source: "cached-offline", etag: ETAG_A });

    const empty = new MemoryStore();
    const off = await resolveEcapaModel({ url: URL, headers: AUTH, fetch: scripted([new Error("Network request failed")]).fetch, store: empty, minBytes: MIN });
    expect(off).toMatchObject({ status: "unavailable", code: "network", reason: expect.stringContaining("Network request failed") });
  });

  it("a truncated download is discarded and never replaces a good model", async () => {
    const store = new MemoryStore();
    withCache(store);
    store.nextDownload = 50; // shorter than Content-Length and below minBytes
    const { fetch } = scripted([head(200, { ETag: ETAG_B, "Content-Length": String(GOOD) })]);
    const res = await resolveEcapaModel({ url: URL, headers: AUTH, fetch, store, minBytes: MIN });
    expect(res).toMatchObject({ status: "ready", source: "cached", etag: ETAG_A });
    expect(store.files.get(ECAPA_FILENAME)?.size).toBe(GOOD);
    expect(store.files.has(`${ECAPA_FILENAME}.download`)).toBe(false);
    expect(store.files.get(etagSidecarName(ECAPA_FILENAME))?.text).toBe(ETAG_A);

    const cold = new MemoryStore();
    cold.nextDownload = GOOD - 1; // disagrees with Content-Length
    const bad = await resolveEcapaModel({ url: URL, headers: AUTH, fetch: scripted([head(200, { ETag: ETAG_A, "Content-Length": String(GOOD) })]).fetch, store: cold, minBytes: MIN });
    expect(bad).toMatchObject({ status: "unavailable", code: "bad-download" });
    expect(cold.files.has(ECAPA_FILENAME)).toBe(false);
  });

  it("a download that throws keeps the cached model, else reports network", async () => {
    const store = new MemoryStore();
    withCache(store);
    store.nextDownload = new Error("socket closed");
    const { fetch } = scripted([head(200, { ETag: ETAG_B, "Content-Length": String(GOOD) })]);
    const res = await resolveEcapaModel({ url: URL, headers: AUTH, fetch, store, minBytes: MIN });
    expect(res).toMatchObject({ status: "ready", source: "cached-offline" });

    const cold = new MemoryStore();
    cold.nextDownload = new Error("socket closed");
    const off = await resolveEcapaModel({ url: URL, headers: AUTH, fetch: scripted([head(200, { ETag: ETAG_A })]).fetch, store: cold, minBytes: MIN });
    expect(off).toMatchObject({ status: "unavailable", code: "network", reason: expect.stringContaining("socket closed") });
  });

  it("an undersized cached file is treated as absent and re-fetched", async () => {
    const store = new MemoryStore();
    store.files.set(ECAPA_FILENAME, { size: 10 }); // an old 404 page, say
    const { fetch, calls } = scripted([head(200, { ETag: ETAG_A, "Content-Length": String(GOOD) })]);
    const res = await resolveEcapaModel({ url: URL, headers: AUTH, fetch, store, minBytes: MIN });
    expect(calls[0].init.headers).toEqual(AUTH);
    expect(res).toMatchObject({ status: "ready", source: "downloaded" });
    expect(store.files.get(ECAPA_FILENAME)?.size).toBe(GOOD);
  });
});
