/**
 * modelDownload.ts's `ModelFileStore` for the browser, so the web build runs
 * the SAME download-once + ETag-revalidate protocol the phone does for the
 * ~80 MB ECAPA ONNX model (`GET|HEAD /models/ecapa.onnx`) — one tested
 * state machine, two storage backends.
 *
 * Storage: the Cache API (`caches.open("mindshift-models")`), which iOS
 * Safari, Chrome and Firefox all keep across visits (Safari evicts it after
 * a couple of weeks without a visit — that just means one more download).
 * Each "file" is a synthetic same-origin Response under
 * `/__mindshift_models/<name>`; the ETag sidecar is a tiny text entry next
 * to it. When the Cache API is missing (an old private-mode Safari, an
 * insecure origin), an in-memory map stands in: the model downloads every
 * session and nothing is persisted — still correct, just slower.
 *
 * `download` streams the body so the caller can show a one-time progress
 * line ("Downloading voice model … 42 %"); a `content-length` mismatch is
 * caught by modelDownload.ts (a bad download never replaces a good one).
 *
 * Extra to the seam: `readBytes(name)` — ORT-web builds a session from
 * bytes, not a path (there is no filesystem in a browser).
 */
import type { ModelFileStat, ModelFileStore } from "./modelDownload";

export const MODEL_CACHE_NAME = "mindshift-models";
const KEY_PREFIX = "/__mindshift_models/";

export interface DownloadProgress {
  received: number;
  /** Bytes expected from `content-length`; null when the server didn't say. */
  total: number | null;
}

/** The slice of the Cache API we use (structural so tests can fake it). */
export interface CacheLike {
  match(key: string): Promise<ResponseLike | undefined>;
  put(key: string, response: ResponseLike): Promise<void>;
  delete(key: string): Promise<boolean>;
}

export interface ResponseLike {
  ok?: boolean;
  status?: number;
  headers: { get(name: string): string | null };
  arrayBuffer(): Promise<ArrayBuffer>;
  text(): Promise<string>;
  body?: { getReader(): { read(): Promise<{ done: boolean; value?: Uint8Array }> } } | null;
}

export interface WebModelStoreOptions {
  /** Defaults to `caches.open(MODEL_CACHE_NAME)`; null => in-memory only. */
  openCache?: () => Promise<CacheLike | null>;
  fetch?: (url: string, init: { method: string; headers: Record<string, string> }) => Promise<ResponseLike>;
  onProgress?: (p: DownloadProgress) => void;
  /** Build a Response to store (seam: Node has no Response in old Jest envs). */
  makeResponse?: (body: Uint8Array | string, headers: Record<string, string>) => ResponseLike;
}

export interface WebModelStore extends ModelFileStore {
  readBytes(name: string): Promise<Uint8Array | null>;
  /** True when entries persist across page loads (Cache API present). */
  readonly persistent: Promise<boolean>;
}

function defaultOpenCache(): Promise<CacheLike | null> {
  try {
    const c = (globalThis as { caches?: { open(name: string): Promise<CacheLike> } }).caches;
    if (!c || typeof c.open !== "function") return Promise.resolve(null);
    return c.open(MODEL_CACHE_NAME).catch(() => null);
  } catch {
    return Promise.resolve(null);
  }
}

function defaultMakeResponse(body: Uint8Array | string, headers: Record<string, string>): ResponseLike {
  const Ctor = (globalThis as { Response?: new (b: unknown, init: { headers: Record<string, string> }) => ResponseLike }).Response;
  if (!Ctor) throw new Error("Response is not available in this environment");
  return new Ctor(body, { headers });
}

/** Memory-backed CacheLike — the fallback, and what tests inspect. */
export class MemoryCache implements CacheLike {
  readonly entries = new Map<string, ResponseLike>();
  async match(key: string) {
    return this.entries.get(key);
  }
  async put(key: string, response: ResponseLike) {
    this.entries.set(key, response);
  }
  async delete(key: string) {
    return this.entries.delete(key);
  }
}

/** A minimal Response stand-in for environments without one (Jest). */
export function bytesResponse(body: Uint8Array | string, headers: Record<string, string> = {}): ResponseLike {
  const bytes = typeof body === "string" ? new TextEncoder().encode(body) : body;
  const lower: Record<string, string> = {};
  for (const [k, v] of Object.entries(headers)) lower[k.toLowerCase()] = v;
  return {
    ok: true,
    status: 200,
    headers: { get: (n) => lower[n.toLowerCase()] ?? null },
    async arrayBuffer() {
      return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
    },
    async text() {
      return new TextDecoder().decode(bytes);
    },
    body: null,
  };
}

export function webModelStore(options: WebModelStoreOptions = {}): WebModelStore {
  const memory = new MemoryCache();
  const cachePromise: Promise<CacheLike> = (options.openCache ?? defaultOpenCache)()
    .then((c) => c ?? memory)
    .catch(() => memory);
  const persistent = cachePromise.then((c) => c !== memory);
  const fetchImpl =
    options.fetch ??
    ((url, init) => (fetch as unknown as (u: string, i: unknown) => Promise<ResponseLike>)(url, init));
  const makeResponse = options.makeResponse ?? defaultMakeResponse;
  const keyOf = (name: string) => `${KEY_PREFIX}${name}`;

  async function statOf(name: string): Promise<ModelFileStat> {
    const cache = await cachePromise;
    const hit = await cache.match(keyOf(name));
    if (!hit) return { exists: false, size: 0 };
    const declared = Number(hit.headers.get("x-mindshift-size") ?? hit.headers.get("content-length"));
    if (Number.isFinite(declared) && declared > 0) return { exists: true, size: declared };
    const buf = await hit.arrayBuffer();
    return { exists: true, size: buf.byteLength };
  }

  return {
    persistent,
    async stat(name) {
      return statOf(name);
    },
    pathOf(name) {
      return keyOf(name);
    },
    async readText(name) {
      const cache = await cachePromise;
      const hit = await cache.match(keyOf(name));
      return hit ? await hit.text() : null;
    },
    async writeText(name, text) {
      const cache = await cachePromise;
      await cache.put(keyOf(name), makeResponse(text, { "content-type": "text/plain" }));
    },
    async readBytes(name) {
      const cache = await cachePromise;
      const hit = await cache.match(keyOf(name));
      if (!hit) return null;
      return new Uint8Array(await hit.arrayBuffer());
    },
    async download(url, name, headers) {
      const res = await fetchImpl(url, { method: "GET", headers });
      if (res.ok === false || (typeof res.status === "number" && res.status >= 400)) {
        throw new Error(`model endpoint answered ${res.status}`);
      }
      const declared = Number(res.headers.get("content-length"));
      const total = Number.isFinite(declared) && declared > 0 ? declared : null;
      let bytes: Uint8Array;
      if (res.body && typeof res.body.getReader === "function") {
        const reader = res.body.getReader();
        const chunks: Uint8Array[] = [];
        let received = 0;
        options.onProgress?.({ received: 0, total });
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) {
            chunks.push(value);
            received += value.byteLength;
            options.onProgress?.({ received, total });
          }
        }
        bytes = new Uint8Array(received);
        let off = 0;
        for (const c of chunks) {
          bytes.set(c, off);
          off += c.byteLength;
        }
      } else {
        bytes = new Uint8Array(await res.arrayBuffer());
        options.onProgress?.({ received: bytes.byteLength, total });
      }
      const cache = await cachePromise;
      await cache.put(
        keyOf(name),
        makeResponse(bytes, {
          "content-type": "application/octet-stream",
          "x-mindshift-size": String(bytes.byteLength),
        }),
      );
      return { exists: true, size: bytes.byteLength };
    },
    async move(from, to) {
      const cache = await cachePromise;
      const hit = await cache.match(keyOf(from));
      if (!hit) throw new Error(`nothing stored under ${from}`);
      const bytes = new Uint8Array(await hit.arrayBuffer());
      await cache.put(
        keyOf(to),
        makeResponse(bytes, {
          "content-type": "application/octet-stream",
          "x-mindshift-size": String(bytes.byteLength),
        }),
      );
      await cache.delete(keyOf(from));
    },
    async remove(name) {
      const cache = await cachePromise;
      await cache.delete(keyOf(name));
    },
  };
}

/** "Downloading voice model … 42 %" / "… 12 MB" for the status line. */
export function describeProgress(p: DownloadProgress): string {
  if (p.total && p.total > 0) {
    const pct = Math.min(100, Math.round((p.received / p.total) * 100));
    return `Downloading voice model (one time) … ${pct} %`;
  }
  return `Downloading voice model (one time) … ${(p.received / (1024 * 1024)).toFixed(0)} MB`;
}
