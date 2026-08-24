#!/usr/bin/env node
/**
 * Copy the pinned onnxruntime-web runtime into apps/mobile/public/ort/ so
 * the web build can load it at runtime from its own origin
 * (src/live/ortWeb.ts injects /ort/ort.wasm.min.js; the wasm glue + binary
 * sit next to it). Metro can't bundle ORT's dynamic wasm import, and the
 * 12 MB binary must not go through the JS bundle anyway.
 *
 * Run by `npm run build:web` (apps/mobile) and scripts/web_deploy.sh. The
 * output directory is git-ignored: it is derived from node_modules.
 */
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const mobile = resolve(here, "..", "apps", "mobile");
const require = createRequire(join(mobile, "package.json"));

// package.json isn't in the package's `exports`; the wasm binary is, so
// locate the dist directory through it.
const dist = dirname(require.resolve("onnxruntime-web/ort-wasm-simd-threaded.wasm"));
const version = JSON.parse(readFileSync(join(dist, "..", "package.json"), "utf8")).version;

// The IIFE runtime (defines globalThis.ort), its ES-module wasm glue (loaded
// by the runtime with a dynamic import), and the wasm binary itself. Only
// the plain wasm backend — no WebGPU/JSEP (Safari's WebGPU is not where we
// want to debug) and no threads (see ortWeb.ts on COOP/COEP).
const FILES = ["ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"];

const out = join(mobile, "public", "ort");
mkdirSync(out, { recursive: true });
for (const name of FILES) {
  const src = join(dist, name);
  if (!existsSync(src)) {
    console.error(`web_copy_ort: ${src} is missing — onnxruntime-web ${version} changed its layout?`);
    process.exit(1);
  }
  copyFileSync(src, join(out, name));
}
writeFileSync(join(out, "VERSION"), `${version}\n`);
console.log(`web_copy_ort: onnxruntime-web ${version} -> ${out} (${FILES.join(", ")})`);
