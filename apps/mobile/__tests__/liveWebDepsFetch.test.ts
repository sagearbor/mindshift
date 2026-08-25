/**
 * src/live/webDeps.ts — the browser's bare `fetch` must never be handed to
 * `resolveEcapaModel` as an options property. Found by the WebKit smoke
 * (scripts/web_smoke_webkit.mjs) on https://arborfam-hub.web.app: the
 * pre-flight panel read "Speaker-ID ✗ offline and no cached model (Can only
 * call Window.fetch on instances of Window)" because `opts.fetch(url, …)`
 * invoked `window.fetch` with the options object as `this`, which WebKit
 * (and Chrome, as "Illegal invocation") rejects. Speaker-ID was therefore
 * off in every browser, and the reason on screen blamed the network.
 */
import { ECAPA_FILENAME, resolveEcapaModel, type FetchLike } from "../src/live/modelDownload";
import type { WebModelStore } from "../src/live/modelStoreWeb";
import { probeWebFastLoopCapabilities } from "../src/live/webDeps";
import type { OrtWebRuntime } from "../src/live/ortWeb";

jest.mock("../src/api/liveSessions", () => ({
  ecapaModelUrl: () => "https://api.example/models/ecapa.onnx",
  authHeaders: async () => ({ Authorization: "Bearer test" }),
  fetchVoiceprints: async () => ({ people: [], model: null, error: null }),
}));

/** A `fetch` with WebKit's `this` check: throws unless called on the window
 *  (or with no receiver at all, which every browser accepts). */
function strictWindowFetch(calls: { url: string; method: string; self: unknown }[]) {
  return function fetchLike(this: unknown, url: string, init?: { method?: string }) {
    calls.push({ url, method: init?.method ?? "GET", self: this });
    if (this !== undefined && this !== globalThis) {
      return Promise.reject(new TypeError("Can only call Window.fetch on instances of Window"));
    }
    return Promise.resolve({
      ok: true,
      status: 200,
      headers: { get: (n: string) => ({ etag: '"rev-1"', "content-length": "2000000" } as Record<string, string>)[n.toLowerCase()] ?? null },
    });
  };
}

class MemoryStore implements WebModelStore {
  files = new Map<string, { size: number; text?: string }>();
  readonly persistent = Promise.resolve(false);
  async stat(name: string) {
    const f = this.files.get(name);
    return f ? { exists: true, size: f.size } : { exists: false, size: 0 };
  }
  pathOf(name: string) {
    return `cache://${name}`;
  }
  async readText(name: string) {
    return this.files.get(name)?.text ?? null;
  }
  async writeText(name: string, text: string) {
    this.files.set(name, { size: text.length, text });
  }
  async download(_url: string, name: string) {
    this.files.set(name, { size: 2_000_000 });
    return { exists: true, size: 2_000_000 };
  }
  async move(from: string, to: string) {
    const f = this.files.get(from);
    if (f) this.files.set(to, f);
    this.files.delete(from);
  }
  async remove(name: string) {
    this.files.delete(name);
  }
  async readBytes(name: string) {
    return this.files.has(name) ? new Uint8Array(8) : null;
  }
}

/** The smallest OrtWebRuntime that builds a session from bytes. */
function fakeOrt(): OrtWebRuntime {
  const session = {
    inputNames: ["input"],
    outputNames: ["embedding"],
    run: async () => ({ embedding: { data: new Float32Array(192), dims: [1, 192] } }),
    release: async () => {},
  };
  return {
    env: { wasm: {} },
    InferenceSession: { create: async () => session },
    Tensor: class {
      constructor(
        public type: string,
        public data: unknown,
        public dims: number[],
      ) {}
    },
  } as unknown as OrtWebRuntime;
}

describe("webDeps hands resolveEcapaModel a fetch that survives WebKit's receiver check", () => {
  const originalFetch = globalThis.fetch;
  const calls: { url: string; method: string; self: unknown }[] = [];
  beforeEach(() => {
    calls.length = 0;
    (globalThis as { fetch: unknown }).fetch = strictWindowFetch(calls);
  });
  afterEach(() => {
    (globalThis as { fetch: unknown }).fetch = originalFetch;
  });

  it("resolveEcapaModel itself never invokes opts.fetch with the options as `this`", async () => {
    const result = await resolveEcapaModel({
      url: "https://api.example/models/ecapa.onnx",
      headers: {},
      fetch: globalThis.fetch as unknown as FetchLike, // the bare global, deliberately
      store: new MemoryStore(),
      minBytes: 1,
    });
    expect(calls.map((c) => c.method)).toEqual(["HEAD"]);
    expect(calls[0].self === undefined || calls[0].self === globalThis).toBe(true);
    expect(result.status).toBe("ready");
  });

  it("probeWebFastLoopCapabilities does not report the speaker-ID as offline because of the receiver", async () => {
    const store = new MemoryStore();
    const caps = await probeWebFastLoopCapabilities({
      loadOrt: async () => ({ ort: fakeOrt(), reason: null }),
      sileroUrl: null,
      store,
    });
    const head = calls.find((c) => c.method === "HEAD");
    expect(head).toBeDefined();
    expect(head!.self === undefined || head!.self === globalThis).toBe(true);
    expect(caps.speakerId.reason ?? "").not.toMatch(/Window\.fetch|Illegal invocation/);
    // The model landed in the store: the HEAD + download path completed.
    expect(store.files.has(ECAPA_FILENAME)).toBe(true);
  });
});
