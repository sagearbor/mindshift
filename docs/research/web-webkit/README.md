# Web app in WebKit (iPhone Safari's engine) — first run, 2026-08-24

The hosted app (https://arborfam-hub.web.app) is how the owner's mom uses
MindShift: iPhone, Safari, no App Store. Until now it had only ever been
smoke-tested in headless **Chrome** (`scripts/web_smoke.mjs`). This is the
first run in **WebKit**, via Playwright's WebKit 26.5 build with the
`iPhone 15` device profile at 393×852 (iOS Safari UA, DPR 3, touch), against
both the local `expo export` and the live site. Rerun with:

```
node scripts/web_smoke_webkit.mjs --signup                                   # local export (needs npm run build:web)
node scripts/web_smoke_webkit.mjs --signup --url https://arborfam-hub.web.app # live
```

`--signup` mints a throwaway Firebase account over the REST API
(`sagearbor+webkit-<hex>@gmail.com`) and deletes it at the end. Playwright
is not a repo dependency — the script resolves it through
`npm exec --package=playwright@latest` and installs WebKit on first use.

Screenshots are CSS-pixel (393 px wide) captures from the run: `live/` is the
deployed site as of this run, `local/` the export built from this branch.

## Findings

| Area | iPhone Safari (WebKit) | Status |
|---|---|---|
| App mounts, no console errors | Mounts in < 3 s; zero console errors on the live site, signed out and signed in | works |
| Login screen at 393×852 | Hero, email/password, Sign In, Google, Apple (coming soon) all present; `scrollWidth == innerWidth` | works |
| Email/password sign-in | Firebase `signInWithEmailAndPassword` + `browserLocalPersistence` lands on the first-launch onboarding, Skip goes Home | works |
| Onboarding / Home / Live Coach / Therapist dashboard / Growth | All render at phone width with no horizontal overflow (hamburger catalog navigation works, tab bar renders) | works |
| Sign-out | Avatar menu → Log out returns to the login screen | works |
| ONNX Runtime (wasm) + Silero VAD | Self-hosted `/ort/ort.wasm.min.js` loads, single-threaded (no `SharedArrayBuffer`, `crossOriginIsolated=false` as designed), Silero session runs: p(silence)=0.002 in 130–660 ms | works |
| Live Coach pre-flight "On-device speech" | `✓ On-device speech ready` — `webkitSpeechRecognition` **is defined** in WebKit, so `detectLiveCapability()` says capable and the "On-device coaching" switch is offered, on by default | works (constructor); recognizer itself **unknown** |
| `webkitSpeechRecognition.start()` | Headless, no user gesture: no `start`, no `error`, no `end` within 4 s — the service is neither reachable nor refused. Apple's recognizer can only be judged on a real iPhone, in a tap handler (`primeWebRecognizer()` already does that) | unknown |
| Live Coach pre-flight "Speaker-ID" | **Broken before this branch**: `✗ Speaker-ID offline and no cached model (Can only call Window.fetch on instances of Window)`. `webDeps.ts` handed the bare `window.fetch` to `resolveEcapaModel`, which called it as `opts.fetch(...)` — WebKit rejects `fetch` with a non-window `this` (Chrome says "Illegal invocation"), so speaker-ID was silently off in every browser and the reason blamed the network | fixed here (not yet deployed) |
| ECAPA model on Safari after the fix | From the live origin in WebKit, `HEAD /models/ecapa.onnx` now answers 200 with an ETag and **content-length 84,188,645** — the first Live Coach open on the phone will pull ~84 MB into the Cache API. Works, but worth knowing on cellular | works (size flagged) |
| Speaker-ID end to end (embeddings on WebKit) | Needs the deployed fix plus enrolled voiceprints; not exercised | unknown |
| Microphone capture (`getUserMedia` + `AudioWorklet`) | Both APIs present; an actual session was not started (headless has no mic; it would also create server-side live sessions for a throwaway user) | unknown |
| Text-to-speech (`speechSynthesis`) | Present; not exercised (needs a session) | unknown |
| Haptics | `navigator.vibrate` absent — expected, webDeps wires `haptics: null` | n/a |
| Local export vs prod API | From a `127.0.0.1` origin every API call dies in CORS preflight (`MINDSHIFT_ALLOWED_ORIGINS` has only the hosted origin). Host policy, not an app bug; the script reports it as INFO in local mode | expected |

## Defects fixed on this branch

- `apps/mobile/src/live/webDeps.ts` — wrap the global `fetch` instead of
  passing it bare (`(url, init) => fetch(url, init)`).
- `apps/mobile/src/live/modelDownload.ts` — call the injected fetch through a
  local, never as `opts.fetch(...)`, so any caller passing a bare `fetch` is
  safe too.
- Regression test: `apps/mobile/__tests__/liveWebDepsFetch.test.ts` (a fake
  `fetch` with WebKit's receiver check; fails on `main`, passes here).

## Not verified (needs a real iPhone)

- Whether Apple's recognizer actually produces text in Safari for a
  continuous session, and how often it auto-ends (the `sttWeb.ts` restart
  path).
- Mic permission prompt + AudioWorklet capture at 44.1/48 kHz.
- That an 84 MB ECAPA download completes and persists in Safari's Cache API
  (Safari evicts site data after 7 days without a visit).
