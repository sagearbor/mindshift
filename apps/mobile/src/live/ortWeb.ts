/**
 * ONNX Runtime seam for the WEB build (iOS Safari is the target — the
 * therapist runs MindShift from https://arborfam-hub.web.app with no App
 * Store install), over `onnxruntime-web`'s WebAssembly backend.
 *
 * How the runtime gets onto the page: NOT through Metro. onnxruntime-web's
 * ESM entry loads its wasm glue with a dynamic `import(url)` that Metro
 * cannot bundle, so the pinned package's `ort.wasm.min.js` (IIFE, defines
 * `globalThis.ort`), its `.mjs` glue and the 12 MB `.wasm` are copied into
 * `apps/mobile/public/ort/` at build time (scripts/web_copy_ort.mjs, run by
 * `npm run build:web`) and the script is injected once, on first use, from
 * the site's own origin. The npm dependency exists only to pin the version.
 *
 * Threads: multi-threaded wasm needs `crossOriginIsolated`, i.e. COOP/COEP
 * headers on the host. Those headers break Firebase's Google sign-in popup
 * (COOP `same-origin` severs `window.opener`), so the site stays
 * un-isolated and the loop runs SINGLE-THREADED on purpose — Silero is
 * < 1 ms per 32 ms chunk either way, and an ECAPA embedding of a 2 s turn
 * is a few hundred ms, which runs in parallel with the STT wait anyway.
 * `numThreads` is pinned to 1 so ORT never even warns.
 *
 * Every export here degrades to null / an honest reason instead of throwing:
 * no wasm => energy VAD + speaker-ID off, exactly like the native ladder.
 */
import type { OnnxSession, OnnxSessionFactory, OrtRuntimeLike } from "./ort";
import { wrapOrtRuntime } from "./ort";

/** Where `scripts/web_copy_ort.mjs` puts the runtime (served by hosting). */
export const ORT_WEB_ASSET_DIR = "/ort/";
export const ORT_WEB_SCRIPT_URL = `${ORT_WEB_ASSET_DIR}ort.wasm.min.js`;

/** `globalThis.ort` as `ort.wasm.min.js` defines it — the structural slice we touch. */
export interface OrtWebRuntime extends OrtRuntimeLike {
  env: {
    wasm: {
      wasmPaths?: string | Record<string, string>;
      numThreads?: number;
      simd?: boolean;
      proxy?: boolean;
    };
    logLevel?: string;
  };
}

/** Fixed-shape models, wasm EP, no graph surprises. */
const SESSION_OPTIONS = {
  executionProviders: ["wasm"],
  graphOptimizationLevel: "all",
};

export interface LoadWebOrtOptions {
  scriptUrl?: string;
  /** Seam for tests: where the `<script>` goes and what it resolves to. */
  document?: Pick<Document, "createElement" | "head">;
  global?: Record<string, unknown>;
  /** Give up after this long (a stalled 12 MB fetch must not hang start). */
  timeoutMs?: number;
}

export type WebOrtLoad = { ort: OrtWebRuntime; reason: null } | { ort: null; reason: string };

let pending: Promise<WebOrtLoad> | null = null;

function configure(ort: OrtWebRuntime, assetDir: string): OrtWebRuntime {
  try {
    ort.env.wasm.wasmPaths = assetDir;
    ort.env.wasm.numThreads = 1;
    // Never route through a worker: the page's own thread is fine for
    // these models and iOS Safari's worker + wasm story is the flaky one.
    ort.env.wasm.proxy = false;
  } catch {
    // A frozen env (older runtime) still works with its defaults.
  }
  return ort;
}

/**
 * Make `onnxruntime-web` available on this page, injecting its script once.
 * Never rejects: resolves `{ ort: null, reason }` when the runtime can't be
 * loaded (no DOM, the asset isn't deployed, a network failure, a timeout).
 */
export function loadWebOrt(options: LoadWebOrtOptions = {}): Promise<WebOrtLoad> {
  const g = options.global ?? (globalThis as Record<string, unknown>);
  const existing = g.ort as OrtWebRuntime | undefined;
  if (existing && existing.InferenceSession) {
    return Promise.resolve({ ort: configure(existing, assetDirOf(options.scriptUrl)), reason: null });
  }
  if (pending) return pending;
  const doc = options.document ?? (typeof document !== "undefined" ? document : null);
  if (!doc) {
    return Promise.resolve({ ort: null, reason: "no document to load ONNX Runtime into" });
  }
  const url = options.scriptUrl ?? ORT_WEB_SCRIPT_URL;
  pending = new Promise<WebOrtLoad>((resolve) => {
    let settled = false;
    const finish = (result: WebOrtLoad) => {
      if (settled) return;
      settled = true;
      resolve(result);
    };
    let script: HTMLScriptElement;
    try {
      script = doc.createElement("script");
    } catch (err) {
      finish({ ort: null, reason: `could not create a script element (${String(err)})` });
      return;
    }
    script.src = url;
    script.async = true;
    script.onload = () => {
      const loaded = g.ort as OrtWebRuntime | undefined;
      if (loaded && loaded.InferenceSession) {
        finish({ ort: configure(loaded, assetDirOf(url)), reason: null });
      } else {
        finish({ ort: null, reason: `${url} loaded but defined no ONNX Runtime` });
      }
    };
    script.onerror = () => finish({ ort: null, reason: `ONNX Runtime script failed to load (${url})` });
    const timer = setTimeout(
      () => finish({ ort: null, reason: `ONNX Runtime script timed out (${url})` }),
      options.timeoutMs ?? 20000,
    );
    // Don't keep a page alive for this in environments with timer unref.
    (timer as unknown as { unref?: () => void }).unref?.();
    try {
      doc.head.appendChild(script);
    } catch (err) {
      finish({ ort: null, reason: `could not inject the ONNX Runtime script (${String(err)})` });
    }
  }).then((result) => {
    if (result.ort === null) pending = null; // let a later session retry
    return result;
  });
  return pending;
}

function assetDirOf(scriptUrl: string | undefined): string {
  if (!scriptUrl) return ORT_WEB_ASSET_DIR;
  const slash = scriptUrl.lastIndexOf("/");
  return slash >= 0 ? scriptUrl.slice(0, slash + 1) : ORT_WEB_ASSET_DIR;
}

/** Test hook: forget an in-flight/failed load. */
export function resetWebOrtForTests() {
  pending = null;
}

export function webOrtSessionFactory(ort: OrtWebRuntime): OnnxSessionFactory {
  return wrapOrtRuntime(ort, SESSION_OPTIONS);
}

/**
 * URL of the bundled Silero VAD on the web build. Metro emits `.onnx` as an
 * asset (metro.config.js) and expo-asset resolves it to the exported
 * `/assets/...` URL; null when that resolution fails.
 */
export function sileroModelUrl(): string | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { Asset } = require("expo-asset") as typeof import("expo-asset");
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const module = require("../../assets/models/silero_vad.onnx") as number;
    const asset = Asset.fromModule(module);
    const uri = asset.uri || asset.localUri;
    return typeof uri === "string" && uri ? uri : null;
  } catch {
    return null;
  }
}

export async function loadSileroSessionWeb(
  factory: OnnxSessionFactory,
  url: string | null = sileroModelUrl(),
): Promise<OnnxSession | null> {
  if (!url) return null;
  try {
    return await factory(url);
  } catch (err) {
    console.warn("[live/web] Silero VAD failed to load — falling back to energy VAD:", err);
    return null;
  }
}
