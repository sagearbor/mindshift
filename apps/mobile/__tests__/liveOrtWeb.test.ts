/**
 * src/live/ortWeb.ts + modelStoreWeb.ts + webDeps.ts — the browser ONNX
 * Runtime seam (script injection, wasm config, sessions from a URL or from
 * bytes), the Cache-API model store driving modelDownload.ts's protocol,
 * and the web fast-loop builder's degradation ladder — all over fakes.
 */
import {
  loadSileroSessionWeb,
  loadWebOrt,
  resetWebOrtForTests,
  webOrtSessionFactory,
  type OrtWebRuntime,
} from "../src/live/ortWeb";
import { bytesResponse, describeProgress, MemoryCache, webModelStore } from "../src/live/modelStoreWeb";
import { resolveEcapaModel, ECAPA_FILENAME, etagSidecarName } from "../src/live/modelDownload";
import { createWebFastLoop } from "../src/live/webDeps";
import { FakeSpeechRecognizer } from "../src/live/stt";
import { SILERO_CHUNK_SAMPLES } from "../src/live/vad";

jest.mock("../src/api/liveSessions", () => ({
  __esModule: true,
  ecapaModelUrl: () => "https://api.test/models/ecapa.onnx",
  authHeaders: async () => ({ Authorization: "Bearer t" }),
  fetchVoiceprints: async () => ({
    people: [
      { personId: "p1", displayName: "Me", isSelf: true, embedding: new Float32Array(192).fill(0.1), model: "ecapa@rev1" },
    ],
    error: null,
  }),
}));

// --- a fake `globalThis.ort` --------------------------------------------------

function fakeOrt(opts: { failFor?: (model: unknown) => boolean } = {}): OrtWebRuntime & { created: unknown[] } {
  const created: unknown[] = [];
  return {
    created,
    env: { wasm: {} },
    Tensor: class {
      constructor(
        public type: string,
        public data: Float32Array | BigInt64Array,
        public dims: readonly number[],
      ) {}
    },
    InferenceSession: {
      async create(model: unknown) {
        created.push(model);
        if (opts.failFor?.(model)) throw new Error("bad model");
        return {
          inputNames: ["input", "state", "sr"],
          outputNames: ["output", "stateN"],
          async run() {
            return {
              output: { type: "float32", data: new Float32Array([0.9]), dims: [1, 1] },
              stateN: { type: "float32", data: new Float32Array(256), dims: [2, 1, 128] },
            };
          },
          async release() {},
        };
      },
    },
  };
}

class FakeScript {
  src = "";
  async = false;
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
}

function fakeDocument(onAppend: (s: FakeScript) => void) {
  return {
    createElement: () => new FakeScript() as unknown as HTMLElement,
    head: { appendChild: (s: unknown) => onAppend(s as FakeScript) } as unknown as HTMLHeadElement,
  } as unknown as Pick<Document, "createElement" | "head">;
}

beforeEach(() => resetWebOrtForTests());

describe("loadWebOrt", () => {
  it("injects the self-hosted script once and configures single-threaded wasm from its directory", async () => {
    const g: Record<string, unknown> = {};
    const appended: FakeScript[] = [];
    const doc = fakeDocument((s) => {
      appended.push(s);
      g.ort = fakeOrt();
      s.onload?.();
    });
    const a = loadWebOrt({ document: doc, global: g, scriptUrl: "/ort/ort.wasm.min.js" });
    const b = loadWebOrt({ document: doc, global: g, scriptUrl: "/ort/ort.wasm.min.js" });
    const [ra, rb] = await Promise.all([a, b]);
    expect(appended).toHaveLength(1);
    expect(appended[0].src).toBe("/ort/ort.wasm.min.js");
    expect(ra.ort).not.toBeNull();
    expect(rb.ort).toBe(ra.ort);
    expect(ra.ort!.env.wasm.wasmPaths).toBe("/ort/");
    expect(ra.ort!.env.wasm.numThreads).toBe(1);
  });

  it("reports a failed script load with a reason and allows a retry", async () => {
    const g: Record<string, unknown> = {};
    let calls = 0;
    const doc = fakeDocument((s) => {
      calls += 1;
      if (calls === 1) s.onerror?.();
      else {
        g.ort = fakeOrt();
        s.onload?.();
      }
    });
    const first = await loadWebOrt({ document: doc, global: g });
    expect(first.ort).toBeNull();
    expect(first.reason).toMatch(/failed to load/);
    const second = await loadWebOrt({ document: doc, global: g });
    expect(second.ort).not.toBeNull();
  });

  it("without a document (SSR/Jest) it says so instead of throwing", async () => {
    const r = await loadWebOrt({ global: {}, document: undefined });
    expect(r.ort).toBeNull();
    expect(r.reason).toMatch(/no document/);
  });
});

describe("webOrtSessionFactory", () => {
  it("builds sessions from a URL (Silero asset) and from bytes (cached ECAPA)", async () => {
    const ort = fakeOrt();
    const factory = webOrtSessionFactory(ort);
    const fromUrl = await factory("/assets/silero_vad.onnx");
    const bytes = new Uint8Array([1, 2, 3]);
    const fromBytes = await factory(bytes);
    expect(ort.created).toEqual(["/assets/silero_vad.onnx", bytes]);
    const out = await fromUrl.run({
      input: { type: "float32", data: new Float32Array(576), dims: [1, 576] },
    });
    expect(Number(out.output.data[0])).toBeCloseTo(0.9);
    await fromBytes.release();
  });

  it("loadSileroSessionWeb: null without a URL or when ORT rejects the model", async () => {
    const ort = fakeOrt({ failFor: (m) => m === "/bad.onnx" });
    const factory = webOrtSessionFactory(ort);
    expect(await loadSileroSessionWeb(factory, null)).toBeNull();
    const warn = jest.spyOn(console, "warn").mockImplementation(() => {});
    expect(await loadSileroSessionWeb(factory, "/bad.onnx")).toBeNull();
    warn.mockRestore();
    expect(await loadSileroSessionWeb(factory, "/good.onnx")).not.toBeNull();
  });
});

describe("webModelStore + resolveEcapaModel", () => {
  const MODEL = new Uint8Array(1_500_000).fill(7);

  function serverFetch(opts: { etag?: string; headStatus?: number } = {}) {
    const etag = opts.etag ?? '"rev1"';
    return jest.fn(async (url: string, init: { method: string; headers: Record<string, string> }) => {
      if (init.method === "HEAD") {
        if (init.headers["If-None-Match"] === etag) {
          return { status: 304, ok: false, headers: { get: () => null } };
        }
        const status = opts.headStatus ?? 200;
        return {
          status,
          ok: status < 400,
          headers: {
            get: (n: string) =>
              ({ etag, "content-length": String(MODEL.byteLength), "x-model-unavailable": "no torch" })[
                n.toLowerCase()
              ] ?? null,
          },
        };
      }
      // GET: stream the body in two chunks so progress is observable.
      let i = 0;
      const chunks = [MODEL.subarray(0, 1_000_000), MODEL.subarray(1_000_000)];
      return {
        status: 200,
        ok: true,
        headers: { get: (n: string) => (n.toLowerCase() === "content-length" ? String(MODEL.byteLength) : null) },
        body: { getReader: () => ({ read: async () => (i < chunks.length ? { done: false, value: chunks[i++] } : { done: true }) }) },
        arrayBuffer: async () => MODEL.buffer,
        text: async () => "",
      };
    });
  }

  it("downloads once with progress, caches with the ETag, then revalidates to a 304", async () => {
    const cache = new MemoryCache();
    const progress: string[] = [];
    const fetchImpl = serverFetch();
    const store = webModelStore({
      openCache: async () => cache,
      makeResponse: bytesResponse,
      fetch: fetchImpl as never,
      onProgress: (p) => progress.push(describeProgress(p)),
    });
    const first = await resolveEcapaModel({ url: "https://api.test/models/ecapa.onnx", headers: {}, fetch: fetchImpl, store });
    expect(first).toMatchObject({ status: "ready", source: "downloaded", etag: '"rev1"' });
    expect(progress[0]).toMatch(/0 %/);
    expect(progress[progress.length - 1]).toMatch(/100 %/);
    expect(await store.persistent).toBe(true);
    const bytes = await store.readBytes(ECAPA_FILENAME);
    expect(bytes?.byteLength).toBe(MODEL.byteLength);
    expect(await store.readText(etagSidecarName(ECAPA_FILENAME))).toBe('"rev1"');
    expect(cache.entries.has(`/__mindshift_models/${ECAPA_FILENAME}.download`)).toBe(false);

    const second = await resolveEcapaModel({ url: "https://api.test/models/ecapa.onnx", headers: {}, fetch: fetchImpl, store });
    expect(second).toMatchObject({ status: "ready", source: "cached" });
    // Only the HEADs after the first download — no second body fetch.
    expect(fetchImpl.mock.calls.filter(([, init]) => init.method === "GET")).toHaveLength(1);
  });

  it("falls back to memory when the Cache API is missing and passes the 503 reason through", async () => {
    const store = webModelStore({ openCache: async () => null, makeResponse: bytesResponse });
    expect(await store.persistent).toBe(false);
    const r = await resolveEcapaModel({
      url: "https://api.test/models/ecapa.onnx",
      headers: {},
      fetch: serverFetch({ headStatus: 503 }),
      store,
    });
    expect(r).toEqual({ status: "unavailable", code: "server-503", reason: "no torch" });
  });
});

describe("createWebFastLoop", () => {
  const silence = new Int16Array(SILERO_CHUNK_SAMPLES * 4);

  function handlers() {
    const sent: unknown[] = [];
    const status: string[] = [];
    return {
      sent,
      status,
      h: {
        speak: jest.fn(),
        send: (e: unknown) => sent.push(e),
        onTurn: jest.fn(),
        onStatus: (line: string) => status.push(line),
        recognizer: new FakeSpeechRecognizer(),
      },
    };
  }

  it("with ORT + a cached model: Silero (wasm), speaker-ID on, LLM cloud, primed browser STT", async () => {
    const ort = fakeOrt();
    const cache = new MemoryCache();
    const MODEL = new Uint8Array(1_500_000);
    const fetchImpl = jest.fn(async (_url: string, init: { method: string }) =>
      init.method === "HEAD"
        ? { status: 200, ok: true, headers: { get: (n: string) => ({ etag: '"rev1"', "content-length": String(MODEL.byteLength) })[n.toLowerCase()] ?? null } }
        : { status: 200, ok: true, headers: { get: () => null }, body: null, arrayBuffer: async () => MODEL.buffer, text: async () => "" },
    );
    const store = webModelStore({ openCache: async () => cache, makeResponse: bytesResponse, fetch: fetchImpl as never });
    const { h, status } = handlers();
    const build = await createWebFastLoop(h, {
      loadOrt: async () => ({ ort, reason: null }),
      sileroUrl: "/assets/silero_vad.onnx",
      store,
      fetch: fetchImpl as never,
    });
    expect(build.capabilities.vad).toBe("silero");
    expect(build.capabilities.speakerId.active).toBe(true);
    expect(build.capabilities.speakerId.enrolled).toBe(1);
    expect(build.capabilities.llm).toEqual(["cloud"]);
    expect(build.status).toMatch(/Silero VAD \(wasm\)/);
    expect(build.status).toMatch(/speaker-ID on \(1 enrolled, model downloaded\)/);
    expect(build.status).toMatch(/browser speech recognition/);
    expect(status[0]).toMatch(/Loading on-device models/);
    // The ECAPA session came from the cached BYTES, Silero from its URL.
    expect(ort.created).toContain("/assets/silero_vad.onnx");
    expect(ort.created.some((m) => m instanceof Uint8Array && m.byteLength === MODEL.byteLength)).toBe(true);
    // The primed recognizer is the one the loop starts.
    await build.loop.start({ sessionId: "s", mode: "therapist", empathy: 50 });
    expect(h.recognizer.started).toBe(true);
    build.loop.pushSamples(silence);
    await build.loop.stop();
  });

  it("without ORT: energy VAD, speaker-ID off with the reason, everything else works", async () => {
    const warn = jest.spyOn(console, "warn").mockImplementation(() => {});
    const log = jest.spyOn(console, "log").mockImplementation(() => {});
    const { h } = handlers();
    const build = await createWebFastLoop(h, {
      loadOrt: async () => ({ ort: null, reason: "ONNX Runtime script failed to load (/ort/ort.wasm.min.js)" }),
    });
    warn.mockRestore();
    log.mockRestore();
    expect(build.capabilities.vad).toBe("energy");
    expect(build.capabilities.speakerId.active).toBe(false);
    expect(build.capabilities.speakerId.reason).toMatch(/failed to load/);
    expect(build.status).toMatch(/energy VAD · speaker-ID off/);
    await build.loop.start({ sessionId: "s", mode: "earpiece", empathy: 50 });
    await build.loop.stop();
  });

  it("a failed model download leaves speaker-ID off but keeps Silero", async () => {
    const log = jest.spyOn(console, "log").mockImplementation(() => {});
    const { h } = handlers();
    const build = await createWebFastLoop(h, {
      loadOrt: async () => ({ ort: fakeOrt(), reason: null }),
      sileroUrl: "/assets/silero_vad.onnx",
      store: webModelStore({ openCache: async () => null, makeResponse: bytesResponse }),
      fetch: (async () => {
        throw new Error("offline");
      }) as never,
    });
    log.mockRestore();
    expect(build.capabilities.vad).toBe("silero");
    expect(build.capabilities.speakerId.active).toBe(false);
    expect(build.capabilities.speakerId.reason).toMatch(/offline and no cached model/);
  });
});
