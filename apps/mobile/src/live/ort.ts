/**
 * The ONNX Runtime seam of the on-device fast loop.
 *
 * Everything that runs a model (Silero VAD, the ECAPA voiceprint) talks to
 * this tiny `OnnxSession` interface, never to `onnxruntime-react-native`
 * directly. Two implementations exist:
 *
 * - `ortNative.ts` — production, over `onnxruntime-react-native` (CPU/XNNPACK,
 *   fixed-shape inputs; see docs/research/on-device-stack-2026-08-24.md).
 * - `ortWeb.ts` — the web build (iOS Safari / Chrome), over `onnxruntime-web`
 *   (wasm, single-threaded) loaded at runtime from the site's own /ort/.
 * - `testing/ortNode.ts` — Jest, over `onnxruntime-node` (a devDependency),
 *   so the test suite runs the REAL models on the REAL fixture PCM instead of
 *   a stub that fabricates probabilities.
 *
 * All three packages expose the same `onnxruntime-common` API shape
 * (`InferenceSession.create` + `Tensor`), so one adapter (`wrapOrtRuntime`)
 * serves them all — the seam exists so the pure logic never imports either.
 */

export type OnnxTensorType = "float32" | "int64";

export interface OnnxTensor {
  readonly type: OnnxTensorType;
  readonly data: Float32Array | BigInt64Array;
  readonly dims: readonly number[];
}

export interface OnnxSession {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Record<string, OnnxTensor>): Promise<Record<string, OnnxTensor>>;
  /** Release native resources. Must never throw. */
  release(): Promise<void>;
}

/** A model as ORT accepts it: a filesystem path (native), a URL (web —
 *  the bundled Silero asset), or the raw bytes (web — the ECAPA download
 *  read back from the browser cache). */
export type OnnxModelSource = string | Uint8Array;

/** Builds a session from a model path / URL / byte buffer. */
export type OnnxSessionFactory = (model: OnnxModelSource) => Promise<OnnxSession>;

export function float32Tensor(
  data: Float32Array,
  dims: readonly number[],
): OnnxTensor {
  // onnxruntime-react-native (≤1.24) reads a typed array's underlying
  // ArrayBuffer and IGNORES byteOffset/byteLength — a `subarray` view is
  // silently replaced by the start of its parent buffer. Measured 2026-08-30:
  // every 1.5 s window of a recording embedded the recording's first 1.5 s
  // ("1 voice found", eigengap 1, cosine 1.000). Hand ORT an OWNED,
  // zero-offset array whenever the view is not the whole buffer, so no
  // caller can hit this again (the live loop passes subarray tails too).
  const owned =
    data.byteOffset !== 0 || data.byteLength !== data.buffer.byteLength
      ? data.slice()
      : data;
  return { type: "float32", data: owned, dims };
}

export function int64Scalar(value: number): OnnxTensor {
  return { type: "int64", data: BigInt64Array.from([BigInt(value)]), dims: [] };
}

// ---------------------------------------------------------------------------
// Adapter over the onnxruntime-common API surface (structural — no import).
// ---------------------------------------------------------------------------

/** The subset of an onnxruntime `Tensor` we read back. */
interface OrtTensorLike {
  type: string;
  data: unknown;
  dims: readonly number[];
}

interface OrtSessionLike {
  inputNames: readonly string[];
  outputNames: readonly string[];
  run(feeds: Record<string, OrtTensorLike>): Promise<Record<string, OrtTensorLike>>;
  release(): Promise<void>;
}

/** What `import * as ort from "onnxruntime-{node,react-native}"` gives us. */
export interface OrtRuntimeLike {
  InferenceSession: {
    create(model: OnnxModelSource, options?: unknown): Promise<OrtSessionLike>;
  };
  Tensor: new (
    type: string,
    data: Float32Array | BigInt64Array,
    dims: readonly number[],
  ) => OrtTensorLike;
}

/**
 * Wrap a real ORT runtime as an `OnnxSessionFactory`. `sessionOptions` is
 * passed straight through to `InferenceSession.create` (the RN build takes
 * execution-provider choices there).
 */
export function wrapOrtRuntime(
  ort: OrtRuntimeLike,
  sessionOptions?: unknown,
): OnnxSessionFactory {
  return async (model: OnnxModelSource): Promise<OnnxSession> => {
    const session = await ort.InferenceSession.create(model, sessionOptions);
    return {
      inputNames: session.inputNames,
      outputNames: session.outputNames,
      async run(feeds) {
        const ortFeeds: Record<string, OrtTensorLike> = {};
        for (const [name, t] of Object.entries(feeds)) {
          ortFeeds[name] = new ort.Tensor(t.type, t.data, t.dims);
        }
        const out = await session.run(ortFeeds);
        const result: Record<string, OnnxTensor> = {};
        for (const [name, t] of Object.entries(out)) {
          if (t.type === "int64") {
            result[name] = {
              type: "int64",
              data: t.data as BigInt64Array,
              dims: t.dims,
            };
          } else {
            // Every model we run outputs float32; anything else is coerced so
            // a caller reading `.data[0]` still gets a number, not a crash.
            const data =
              t.data instanceof Float32Array
                ? t.data
                : Float32Array.from(t.data as ArrayLike<number>);
            result[name] = { type: "float32", data, dims: t.dims };
          }
        }
        return result;
      },
      async release() {
        try {
          await session.release();
        } catch {
          // Already released / native gone — nothing left to free.
        }
      },
    };
  };
}
