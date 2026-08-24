/**
 * Jest-only ONNX Runtime over `onnxruntime-node` (a devDependency). Lives
 * under src/ so tsc type-checks it, but NOTHING in the app graph imports it —
 * Metro never sees the node binary. Tests use it to run the real
 * silero_vad.onnx (and the ECAPA export when present) on synthetic PCM.
 *
 * Realm note: jest runs each test file inside its own `vm` context, whose
 * `Float32Array` is a different constructor from the main Node realm's. The
 * native binding hands back main-realm typed arrays, and onnxruntime-common
 * validates them with `instanceof` against the context's constructor
 * (jest#7780) — so the first `session.run()` would throw "A float32 tensor's
 * data must be type of Float32Array". The fix is to make the test context use
 * the main realm's typed-array constructors before ORT is loaded; every
 * `new Float32Array(...)` in the code under test resolves the global at call
 * time, so the whole file agrees on one realm. Test-only, by construction.
 */
import type { OnnxSessionFactory, OrtRuntimeLike } from "../ort";
import { wrapOrtRuntime } from "../ort";

function unifyTypedArrayRealms() {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const vm = require("vm") as { runInThisContext(code: string): unknown };
  const g = globalThis as Record<string, unknown>;
  for (const name of ["Float32Array", "BigInt64Array", "Int16Array", "Uint8Array"]) {
    const outer = vm.runInThisContext(name);
    if (outer && g[name] !== outer) g[name] = outer;
  }
}

export function nodeOrtSessionFactory(): OnnxSessionFactory {
  unifyTypedArrayRealms();
  // Dynamic require keeps the native binding out of any bundler graph that
  // might statically walk this file.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const ort = require("onnxruntime-node") as OrtRuntimeLike;
  return wrapOrtRuntime(ort);
}
