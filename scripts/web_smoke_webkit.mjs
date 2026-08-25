#!/usr/bin/env node
/**
 * Smoke-test the web app in WebKit — the engine iPhone Safari uses — with
 * an iPhone 15 device profile. scripts/web_smoke.mjs covers headless
 * CHROME; this is the "what will the owner's mom actually see on her
 * iPhone" check, driven by Playwright's WebKit build.
 *
 *   node scripts/web_smoke_webkit.mjs                     # local export (apps/mobile/dist, served here)
 *   node scripts/web_smoke_webkit.mjs --url https://arborfam-hub.web.app   # the LIVE site
 *   node scripts/web_smoke_webkit.mjs --signup            # + sign in with a throwaway Firebase account
 *   WEB_SMOKE_EMAIL=… WEB_SMOKE_PASSWORD=… node scripts/web_smoke_webkit.mjs   # + sign in as an existing account
 *
 * Playwright is NOT a dependency of this repo: the script resolves it from
 * `npm exec -p playwright@latest` (cached under ~/.npm/_npx) and installs
 * the WebKit build on first use (`playwright install webkit`, ~80 MB).
 * Override the package spec with PLAYWRIGHT_PKG=playwright@1.62.1.
 *
 * What it checks (PASS / FAIL, plus INFO lines that are reported, not judged):
 *   1. the app MOUNTS with no console errors (favicon 404s excepted);
 *   2. the login screen renders at 393×852 with no horizontal overflow;
 *   3. the self-hosted ONNX Runtime + Silero VAD session runs in WebKit
 *      (single-threaded wasm — the site is not cross-origin isolated);
 *   4. with credentials: email/password sign-in works, the onboarding
 *      walkthrough (if shown) can be skipped, and Home renders;
 *   5. Live Coach: the pre-flight panel's rows ("On-device speech: …",
 *      Speaker-ID, Suggestions, Turn detection) are read back verbatim,
 *      plus a direct probe of `webkitSpeechRecognition.start()` — headless
 *      WebKit exposes the constructor, but whether Apple's recognizer is
 *      reachable there is reported honestly, never assumed;
 *   6. Therapist dashboard and Growth render at phone width with
 *      `document.scrollingElement.scrollWidth <= innerWidth`;
 *   7. sign-out returns to the login screen.
 * Screenshots (CSS-pixel scale, small) land in --shots DIR (default
 * docs/research/web-webkit/<local|live>/). A --signup account is deleted
 * at the end (Firebase accounts:delete) unless --keep-account.
 * Exit code 0 = every PASS/FAIL check passed.
 */
import { execFileSync } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, realpathSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(here, "..");
const DIST = join(ROOT, "apps", "mobile", "dist");

// --- args ---------------------------------------------------------------------
const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const opt = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : dflt;
};
if (flag("--help") || flag("-h")) {
  console.log(
    "usage: node scripts/web_smoke_webkit.mjs [--url URL] [--signup | WEB_SMOKE_EMAIL/WEB_SMOKE_PASSWORD] [--shots DIR] [--keep-account] [--label NAME]",
  );
  process.exit(0);
}
const URL_OPT = opt("--url", null);
const SIGNUP = flag("--signup");
const KEEP_ACCOUNT = flag("--keep-account");
const LABEL = opt("--label", URL_OPT ? "live" : "local");
const SHOTS = resolve(opt("--shots", join(ROOT, "docs", "research", "web-webkit", LABEL)));
const EMAIL = process.env.WEB_SMOKE_EMAIL ?? null;
const PASSWORD = process.env.WEB_SMOKE_PASSWORD ?? null;
// The app's PUBLIC Firebase web API key (apps/mobile/src/auth/firebaseConfig.ts).
const FIREBASE_API_KEY = process.env.EXPO_PUBLIC_FIREBASE_API_KEY ?? "AIzaSyAJA-C1dpMqpjmM9A7GIGb-IfsOJSl7XS4";
const SIGNUP_BASE = process.env.WEB_SMOKE_SIGNUP_EMAIL_BASE ?? "sagearbor@gmail.com";
const IDENTITY_TOOLKIT = "https://identitytoolkit.googleapis.com/v1";
const PLAYWRIGHT_PKG = process.env.PLAYWRIGHT_PKG ?? "playwright@latest";
const VIEWPORT = { width: 393, height: 852 };

// --- Playwright without a repo dependency -----------------------------------
async function loadPlaywright() {
  try {
    return await import("playwright");
  } catch {
    // Not installed here: resolve the npx-cached package's real path.
  }
  const bin = execFileSync("npm", ["exec", "--yes", `--package=${PLAYWRIGHT_PKG}`, "-c", "command -v playwright"], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
  }).trim();
  const pkgDir = dirname(realpathSync(bin)); // …/node_modules/playwright/cli.js -> its dir
  return import(pathToFileURL(join(pkgDir, "index.mjs")).href);
}
function installWebKit() {
  console.log("web_smoke_webkit: installing Playwright's WebKit build (one-time, ~80 MB) …");
  execFileSync("npm", ["exec", "--yes", `--package=${PLAYWRIGHT_PKG}`, "-c", "playwright install webkit"], { stdio: "inherit" });
}

// --- static server with the hosting rewrite (** -> /index.html) -------------
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
async function serveDist() {
  if (!existsSync(join(DIST, "index.html"))) {
    console.error(`web_smoke_webkit: ${DIST}/index.html missing — run \`npm run build:web\` in apps/mobile first (or pass --url)`);
    process.exit(2);
  }
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
  return { server, base: `http://127.0.0.1:${server.address().port}` };
}

// --- Firebase REST (throwaway account) ----------------------------------------
async function firebaseAuth(verb, email, password) {
  const res = await fetch(`${IDENTITY_TOOLKIT}/accounts:${verb}?key=${FIREBASE_API_KEY}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email, password, returnSecureToken: true }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`Firebase ${verb} failed: ${res.status} ${body?.error?.message ?? ""}`);
  return body;
}
async function firebaseDelete(idToken) {
  const res = await fetch(`${IDENTITY_TOOLKIT}/accounts:delete?key=${FIREBASE_API_KEY}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ idToken }),
  });
  return res.ok;
}
function throwawayEmail(base) {
  const [local, domain] = base.split("@");
  const hex = Array.from(crypto.getRandomValues(new Uint8Array(3)), (b) => b.toString(16).padStart(2, "0")).join("");
  return `${local}+webkit-${hex}@${domain}`;
}

// --- report -------------------------------------------------------------------
let failed = false;
const findings = [];
const report = (ok, line) => {
  console.log(`${ok ? "PASS" : "FAIL"} ${line}`);
  findings.push({ status: ok ? "PASS" : "FAIL", line });
  if (!ok) failed = true;
};
const info = (line) => {
  console.log(`INFO ${line}`);
  findings.push({ status: "INFO", line });
};
const tid = (id) => `[data-testid="${id}"]`;

// --- main ---------------------------------------------------------------------
const pw = await loadPlaywright();
const { webkit, devices } = pw.default ?? pw;
let browser;
try {
  browser = await webkit.launch();
} catch (err) {
  if (!/Executable doesn't exist|install/i.test(String(err?.message))) throw err;
  installWebKit();
  browser = await webkit.launch();
}

mkdirSync(SHOTS, { recursive: true });
const served = URL_OPT ? null : await serveDist();
const base = (URL_OPT ?? served.base).replace(/\/$/, "");
console.log(`web_smoke_webkit: ${base} · WebKit ${browser.version()} · shots -> ${SHOTS}`);

const iphone = devices["iPhone 15"];
const context = await browser.newContext({ ...iphone, viewport: VIEWPORT });
const page = await context.newPage();
const logs = [];
page.on("console", (m) => logs.push({ level: m.type(), text: m.text() }));
page.on("pageerror", (e) => logs.push({ level: "error", text: `uncaught: ${e.message}` }));
page.on("requestfailed", (r) => logs.push({ level: "requestfailed", text: `${r.url()} ${r.failure()?.errorText ?? ""}` }));
const consoleErrors = (since = 0) =>
  logs.slice(since).filter((l) => l.level === "error" && !/favicon/i.test(l.text));
const shot = async (name) => {
  const file = join(SHOTS, `${name}.png`);
  await page.screenshot({ path: file, scale: "css" });
  return file;
};
const overflow = () => page.evaluate(() => ({ scrollWidth: document.scrollingElement.scrollWidth, innerWidth: window.innerWidth }));
const overflowCheck = async (what) => {
  const o = await overflow();
  report(o.scrollWidth <= o.innerWidth, `${what}: no horizontal overflow (scrollWidth ${o.scrollWidth} <= innerWidth ${o.innerWidth})`);
};
const rowText = async (id) => {
  const el = page.locator(tid(id));
  if ((await el.count()) === 0) return null;
  return (await el.first().innerText()).replace(/\s+/g, " ").trim();
};
/** Open the hamburger catalog and pick a destination. */
const navigateTo = async (destId) => {
  await page.locator(tid("chrome-hamburger-button")).first().click();
  await page.locator(tid(`chrome-catalog-item-${destId}`)).first().click();
};

let account = null;
try {
  // 1. Mount.
  await page.goto(`${base}/`, { waitUntil: "load", timeout: 60000 });
  const mounted = await page
    .waitForFunction(() => (document.getElementById("root")?.innerText ?? "").trim().length > 0, null, { timeout: 40000 })
    .then(() => true)
    .catch(() => false);
  const rootText = mounted ? await page.evaluate(() => document.getElementById("root").innerText.trim()) : "";
  report(mounted, `app mounted (root text: ${JSON.stringify(rootText.slice(0, 60))})`);
  report(
    consoleErrors().length === 0,
    `no console errors on mount${consoleErrors().length ? ":\n  " + consoleErrors().map((e) => e.text).join("\n  ") : ""}`,
  );
  info(`UA ${await page.evaluate(() => navigator.userAgent)}`);

  // 2. Login screen at phone width.
  const loginVisible =
    (await page.locator(tid("email-input")).count()) > 0 &&
    (await page.locator(tid("password-input")).count()) > 0 &&
    (await page.locator(tid("submit-button")).count()) > 0;
  report(loginVisible, `login screen renders at ${VIEWPORT.width}×${VIEWPORT.height} (email, password, submit present)`);
  await overflowCheck("login");
  info(`screenshot ${await shot("01-login")}`);

  // Browser capabilities the live path depends on — reported, not judged.
  const caps = await page.evaluate(() => ({
    SpeechRecognition: typeof SpeechRecognition !== "undefined" ? "SpeechRecognition" : typeof webkitSpeechRecognition !== "undefined" ? "webkitSpeechRecognition" : "none",
    getUserMedia: Boolean(navigator.mediaDevices?.getUserMedia),
    AudioWorklet: typeof AudioWorkletNode !== "undefined",
    speechSynthesis: typeof speechSynthesis !== "undefined",
    cacheApi: typeof caches !== "undefined",
    wasm: typeof WebAssembly !== "undefined",
    sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    crossOriginIsolated: Boolean(globalThis.crossOriginIsolated),
    vibrate: typeof navigator.vibrate === "function",
  }));
  info(`browser APIs: ${JSON.stringify(caps)}`);

  // 3. ORT + Silero, the way src/live/ortWeb.ts loads it (self-hosted, wasm EP, 1 thread).
  const onnxName = await page.evaluate(async () => {
    // The exported asset name is hashed; find it from the bundle's manifest-free
    // location by listing what the app itself references.
    const scripts = Array.from(document.scripts).map((s) => s.src).filter(Boolean);
    for (const src of scripts) {
      const text = await fetch(src).then((r) => r.text()).catch(() => "");
      const m = text.match(/\/assets\/assets\/models\/[A-Za-z0-9_.-]+\.onnx/);
      if (m) return m[0];
    }
    return null;
  });
  report(Boolean(onnxName), `Silero asset referenced by the bundle (${onnxName ?? "not found"})`);
  const ortResult = await page
    .evaluate(async (onnxUrl) => {
      const t0 = performance.now();
      await new Promise((res, rej) => {
        const s = document.createElement("script");
        s.src = "/ort/ort.wasm.min.js";
        s.onload = res;
        s.onerror = () => rej(new Error("script failed"));
        document.head.appendChild(s);
      });
      if (!globalThis.ort) throw new Error("globalThis.ort undefined");
      ort.env.wasm.wasmPaths = "/ort/";
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      const s = await ort.InferenceSession.create(onnxUrl, { executionProviders: ["wasm"] });
      const feeds = {
        input: new ort.Tensor("float32", new Float32Array(576), [1, 576]),
        state: new ort.Tensor("float32", new Float32Array(256), [2, 1, 128]),
        sr: new ort.Tensor("int64", BigInt64Array.from([16000n]), []),
      };
      const out = await s.run(feeds);
      return { ms: Math.round(performance.now() - t0), p: out.output.data[0], version: ort.env.versions.web };
    }, onnxName ?? "")
    .catch((err) => ({ error: err.message }));
  report(
    typeof ortResult.p === "number",
    ortResult.error
      ? `onnxruntime-web in WebKit: ${ortResult.error}`
      : `onnxruntime-web ${ortResult.version} loaded + Silero session ran in WebKit (p(silence)=${ortResult.p.toFixed(3)}, ${ortResult.ms} ms)`,
  );

  // 4. Sign in.
  let email = EMAIL;
  let password = PASSWORD;
  if (SIGNUP) {
    email = throwawayEmail(SIGNUP_BASE);
    password = `Wk-${Array.from(crypto.getRandomValues(new Uint8Array(9)), (b) => b.toString(16).padStart(2, "0")).join("")}`;
    const created = await firebaseAuth("signUp", email, password);
    account = { email, idToken: created.idToken, uid: created.localId };
    info(`throwaway account ${email} (uid ${account.uid})`);
  }
  if (!email || !password) {
    info("no credentials (--signup or WEB_SMOKE_EMAIL/WEB_SMOKE_PASSWORD) — skipping the signed-in checks");
  } else {
    const before = logs.length;
    await page.locator(tid("email-input")).first().fill(email);
    await page.locator(tid("password-input")).first().fill(password);
    await page.locator(tid("submit-button")).first().click();
    const landed = await page
      .waitForSelector(`${tid("onboarding-screen")}, ${tid("app-chrome")}, ${tid("auth-error")}`, { timeout: 40000 })
      .catch(() => null);
    const authError = await rowText("auth-error");
    report(Boolean(landed) && !authError, `email/password sign-in${authError ? ` failed: ${authError}` : " landed"}`);
    if (await page.locator(tid("onboarding-screen")).count()) {
      info(`first-launch onboarding shown — screenshot ${await shot("02-onboarding")}`);
      await overflowCheck("onboarding");
      await page.locator(tid("onboarding-skip")).first().click();
    }
    const home = await page.waitForSelector(tid("app-chrome"), { timeout: 30000 }).catch(() => null);
    report(Boolean(home), "Home renders inside AppChrome after sign-in");
    await page.waitForTimeout(1500);
    await overflowCheck("home");
    info(`screenshot ${await shot("03-home")}`);

    // 5. Live Coach + the pre-flight panel.
    await navigateTo("coach");
    const preflight = await page.waitForSelector(tid("live-preflight"), { timeout: 30000 }).catch(() => null);
    report(Boolean(preflight), "Live Coach renders the pre-flight panel");
    // Wait for the probe (ORT + Silero + ECAPA download) to settle.
    const settled = await page
      .waitForFunction(
        () => !Array.from(document.querySelectorAll("div,span")).some((e) => e.childElementCount === 0 && /Loading models…/.test(e.textContent ?? "")),
        null,
        { timeout: 90000 },
      )
      .then(() => true)
      .catch(() => false);
    if (!settled) info("pre-flight still 'Loading models…' after 90 s");
    for (const id of ["preflight-stt", "preflight-speaker-id", "preflight-llm", "preflight-vad"]) {
      const t = await rowText(id);
      if (t !== null) info(`pre-flight ${id}: ${t}`);
    }
    const modeRow = await rowText("live-mode-row");
    info(`on-device switch row: ${modeRow ?? "not offered (liveCapable=false)"}`);
    const liveStatus = await rowText("live-status");
    if (liveStatus) info(`live-status: ${liveStatus}`);
    const sttRow = await rowText("preflight-stt");
    report(Boolean(sttRow && /On-device speech/.test(sttRow)), `pre-flight "On-device speech" row present${sttRow ? ` -> "${sttRow}"` : ""}`);
    await overflowCheck("live coach");
    info(`screenshot ${await shot("04-live-coach")}`);

    // Direct probe: can WebKit actually START Apple's recognizer here? The
    // constructor existing says nothing about the service.
    const sttProbe = await page.evaluate(
      () =>
        new Promise((resolveProbe) => {
          const Ctor = globalThis.SpeechRecognition ?? globalThis.webkitSpeechRecognition;
          if (!Ctor) return resolveProbe({ ctor: false });
          const events = [];
          let rec;
          try {
            rec = new Ctor();
          } catch (err) {
            return resolveProbe({ ctor: true, construct: String(err) });
          }
          rec.continuous = true;
          rec.interimResults = true;
          rec.onstart = () => events.push("start");
          rec.onerror = (e) => events.push(`error:${e.error}`);
          rec.onend = () => events.push("end");
          rec.onaudiostart = () => events.push("audiostart");
          const done = () => {
            try {
              rec.abort();
            } catch {
              // ignore
            }
            resolveProbe({ ctor: true, events });
          };
          try {
            rec.start();
          } catch (err) {
            events.push(`start threw: ${err?.message ?? err}`);
            return done();
          }
          setTimeout(done, 4000);
        }),
    );
    info(`webkitSpeechRecognition.start() probe (no user gesture, headless): ${JSON.stringify(sttProbe)}`);

    // 6. Therapist dashboard + Growth at phone width.
    await navigateTo("therapistDashboard");
    const dash = await page.waitForSelector(tid("therapist-dashboard"), { timeout: 30000 }).catch(() => null);
    report(Boolean(dash), "Therapist dashboard renders");
    await page.waitForTimeout(3000);
    await overflowCheck("therapist dashboard");
    info(`screenshot ${await shot("05-dashboard")}`);

    await navigateTo("growth");
    const growth = await page.waitForSelector(tid("growth-screen"), { timeout: 30000 }).catch(() => null);
    report(Boolean(growth), "Growth renders");
    await page.waitForFunction((sel) => !document.querySelector(sel), tid("growth-loading"), { timeout: 30000 }).catch(() => {});
    await page.waitForTimeout(500);
    await overflowCheck("growth");
    info(`screenshot ${await shot("06-growth")}`);

    // The local export still talks to the PRODUCTION API, whose CORS
    // allowlist (MINDSHIFT_ALLOWED_ORIGINS) has no 127.0.0.1 origin, so from
    // a local serve every API call dies in preflight. That is the host's
    // policy, not a web-app defect: reported, not failed, in local mode.
    const isCors = (t) => /Preflight response|access control checks/i.test(t);
    const errs = consoleErrors(before).filter((e) => URL_OPT || !isCors(e.text));
    const corsErrs = URL_OPT ? [] : consoleErrors(before).filter((e) => isCors(e.text));
    if (corsErrs.length) info(`${corsErrs.length} API calls blocked by CORS from the local origin (expected: the prod API allowlists only the hosted origin)`);
    report(errs.length === 0, `no console errors while signed in${errs.length ? ":\n  " + errs.map((e) => e.text).join("\n  ") : ""}`);
    const failedReqs = logs.slice(before).filter((l) => l.level === "requestfailed" && (URL_OPT || !isCors(l.text)));
    if (failedReqs.length) info(`failed requests:\n  ${failedReqs.map((r) => r.text).join("\n  ")}`);

    // 7. Sign out.
    await page.locator(tid("chrome-avatar-button")).first().click();
    await page.locator(tid("chrome-account-sign-out")).first().click();
    const loggedOut = await page.waitForSelector(tid("login-screen"), { timeout: 20000 }).catch(() => null);
    report(Boolean(loggedOut), "sign-out returns to the login screen");
  }
} catch (err) {
  report(false, `smoke threw: ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
  await shot("99-failure").catch(() => {});
} finally {
  if (account && !KEEP_ACCOUNT) {
    const ok = await firebaseDelete(account.idToken).catch(() => false);
    info(`throwaway account ${account.email} ${ok ? "deleted" : "NOT deleted — remove it in the Firebase console"}`);
  }
  await browser.close().catch(() => {});
  served?.server.close();
}
console.log(`\nweb_smoke_webkit: ${failed ? "FAILED" : "passed"} (${findings.filter((f) => f.status === "PASS").length} pass, ${findings.filter((f) => f.status === "FAIL").length} fail, ${findings.filter((f) => f.status === "INFO").length} info)`);
process.exit(failed ? 1 : 0);
