#!/usr/bin/env node
/**
 * Smoke-test the exported web app (apps/mobile/dist) in headless Chrome
 * over the DevTools protocol — no puppeteer/playwright dependency.
 *
 *   node scripts/web_smoke.mjs            # after `npm run build:web`
 *   CHROME=/path/to/chrome node scripts/web_smoke.mjs
 *
 * Serves dist/ on a local port (SPA rewrite like Firebase Hosting), opens it
 * in an iPhone-sized viewport and checks, with a hard timeout:
 *   1. the app MOUNTS (the root renders text — the login screen) with no
 *      console errors other than a missing favicon;
 *   2. the self-hosted ONNX Runtime (/ort/ort.wasm.min.js) loads on that
 *      page and builds a Silero VAD session from the exported .onnx asset —
 *      the exact path src/live/ortWeb.ts takes on iOS Safari.
 * Exit code 0 = both passed. scripts/web_deploy.sh runs this before a
 * deploy when Chrome is present and skips it (with a note) otherwise.
 */
import { spawn } from "node:child_process";
import { createReadStream, existsSync, mkdtempSync, readdirSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const DIST = resolve(here, "..", "apps", "mobile", "dist");
const CHROME =
  process.env.CHROME ||
  ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome", "/usr/bin/chromium"].find(
    existsSync,
  );

if (!existsSync(join(DIST, "index.html"))) {
  console.error(`web_smoke: ${DIST}/index.html missing — run \`npm run build:web\` first`);
  process.exit(2);
}
if (!CHROME) {
  console.error("web_smoke: no Chrome found (set CHROME=/path/to/chrome)");
  process.exit(3);
}

const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".json": "application/json",
  ".css": "text/css",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".ico": "image/x-icon",
};

// --- static server with the hosting rewrite (** -> /index.html) -------------
const server = createServer((req, res) => {
  const url = decodeURIComponent((req.url ?? "/").split("?")[0]);
  let file = join(DIST, url);
  if (!file.startsWith(DIST)) {
    res.writeHead(403).end();
    return;
  }
  if (!existsSync(file) || statSync(file).isDirectory()) file = join(DIST, "index.html");
  res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" });
  createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;
const base = `http://127.0.0.1:${port}`;

// --- headless Chrome over CDP -------------------------------------------------
const profile = mkdtempSync(join(tmpdir(), "web-smoke-"));
const chrome = spawn(CHROME, [
  "--headless=new",
  "--disable-gpu",
  "--no-first-run",
  "--remote-debugging-port=0",
  `--user-data-dir=${profile}`,
  "about:blank",
]);
const wsUrl = await new Promise((resolvePromise, reject) => {
  let buf = "";
  chrome.stderr.on("data", (d) => {
    buf += d.toString();
    const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
    if (m) resolvePromise(m[1]);
  });
  chrome.on("exit", (c) => reject(new Error(`chrome exited ${c}\n${buf}`)));
  setTimeout(() => reject(new Error("no DevTools endpoint\n" + buf)), 20000);
});
const httpBase = wsUrl.replace(/^ws:\/\//, "http://").replace(/\/devtools\/browser\/.*/, "");
const targets = await (await fetch(`${httpBase}/json`)).json();
const page = targets.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0;
const pending = new Map();
const logs = [];
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    pending.get(msg.id)(msg);
    pending.delete(msg.id);
  } else if (msg.method === "Runtime.consoleAPICalled") {
    logs.push({ level: msg.params.type, text: msg.params.args.map((a) => a.value ?? a.description ?? "").join(" ") });
  } else if (msg.method === "Log.entryAdded") {
    logs.push({ level: msg.params.entry.level, text: `${msg.params.entry.text} ${msg.params.entry.url ?? ""}` });
  } else if (msg.method === "Runtime.exceptionThrown") {
    logs.push({ level: "error", text: `${msg.params.exceptionDetails.text} ${msg.params.exceptionDetails.exception?.description ?? ""}` });
  }
};
const send = (method, params = {}) =>
  new Promise((resolvePromise) => {
    const n = ++id;
    pending.set(n, resolvePromise);
    ws.send(JSON.stringify({ id: n, method, params }));
  });
const evaluate = async (expression) => {
  const r = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description ?? r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
const waitFor = async (expression, timeoutMs) => {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      const v = await evaluate(expression);
      if (v) return v;
    } catch {
      // still loading
    }
    await new Promise((r) => setTimeout(r, 400));
  }
  return null;
};

let failed = false;
const report = (ok, line) => {
  console.log(`${ok ? "PASS" : "FAIL"} ${line}`);
  if (!ok) failed = true;
};

try {
  await send("Runtime.enable");
  await send("Log.enable");
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 3, mobile: true });
  await send("Page.navigate", { url: `${base}/` });

  // 1. The app mounts.
  const text = await waitFor('document.getElementById("root") && document.getElementById("root").innerText.trim()', 40000);
  report(Boolean(text), `app mounted (root text: ${text ? JSON.stringify(text.slice(0, 60)) : "none"})`);
  const errors = logs.filter((l) => l.level === "error" && !/favicon/.test(l.text));
  report(errors.length === 0, `no console errors${errors.length ? ":\n  " + errors.map((e) => e.text).join("\n  ") : ""}`);

  // 2. ORT + Silero on that page, the way ortWeb.ts does it.
  const onnx = readdirSync(join(DIST, "assets", "assets", "models")).find((f) => f.endsWith(".onnx"));
  report(Boolean(onnx), `Silero asset exported (${onnx ?? "missing"})`);
  const ortResult = await evaluate(`(async () => {
    const t0 = performance.now();
    await new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = "/ort/ort.wasm.min.js"; s.onload = res; s.onerror = () => rej(new Error("script failed"));
      document.head.appendChild(s);
    });
    if (!globalThis.ort) throw new Error("globalThis.ort undefined");
    ort.env.wasm.wasmPaths = "/ort/"; ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
    const s = await ort.InferenceSession.create("/assets/assets/models/${onnx}", { executionProviders: ["wasm"] });
    const feeds = {
      input: new ort.Tensor("float32", new Float32Array(576), [1, 576]),
      state: new ort.Tensor("float32", new Float32Array(256), [2, 1, 128]),
      sr: new ort.Tensor("int64", BigInt64Array.from([16000n]), []),
    };
    const out = await s.run(feeds);
    return { ms: Math.round(performance.now() - t0), p: out.output.data[0], version: ort.env.versions.web, isolated: !!globalThis.crossOriginIsolated };
  })()`);
  report(
    ortResult && typeof ortResult.p === "number",
    `onnxruntime-web ${ortResult?.version} loaded + Silero session ran (p(silence)=${ortResult?.p?.toFixed(3)}, ${ortResult?.ms} ms, crossOriginIsolated=${ortResult?.isolated})`,
  );
} catch (err) {
  report(false, `smoke threw: ${err instanceof Error ? err.message : String(err)}`);
} finally {
  ws.close();
  chrome.kill();
  server.close();
}
process.exit(failed ? 1 : 0);
