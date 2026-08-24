# The web app as a first-class client: Safari on an iPhone (2026-08-24)

> The therapist (the owner's mom) uses MindShift from **Safari on her
> iPhone** — no App Store install, no Apple Developer account, just
> https://arborfam-hub.web.app (Firebase Hosting, `firebase.json` →
> `apps/mobile/dist`). This note records what the web build now does, how
> it is built and shipped, and — honestly — what works on iOS Safari
> versus what is known-limited. Nothing here was verified on a physical
> iPhone: the only test phone is a Pixel 10 and there is no iOS device or
> Xcode. Everything below was verified in headless Chrome against the real
> export plus Jest over fakes of the Safari APIs; the Safari-specific
> behaviours are from WebKit's documented/observed quirks, guarded in code.

## What ships

The on-device realtime fast loop (`apps/mobile/src/live/fastLoop.ts`,
unchanged) now runs in the browser through web implementations of its
seams:

| Seam | Native (phone app) | Web (Safari / Chrome) |
|---|---|---|
| ONNX Runtime | `ortNative.ts` over onnxruntime-react-native | `ortWeb.ts` over **onnxruntime-web (wasm, single-threaded)**, loaded at runtime from the site's own `/ort/` (copied from the pinned npm package by `scripts/web_copy_ort.mjs`; Metro cannot bundle ORT's dynamic wasm import) |
| Silero VAD | app asset via expo-asset → file path | the same `.onnx` asset, served from `/assets/…` → session from URL |
| ECAPA voiceprint model | `GET|HEAD /models/ecapa.onnx` → document dir, ETag revalidated | **the same protocol** (`modelDownload.ts`) over a Cache-API store (`modelStoreWeb.ts`): downloaded once (~80 MB, progress line "Downloading voice model (one time) … 42 %"), ETag-revalidated per session, session built from bytes; in-memory fallback when the Cache API is missing |
| Speech-to-text | expo-speech-recognition (Apple Speech / Android on-device) | **Web Speech API** (`sttWeb.ts`: `webkitSpeechRecognition` on iOS Safari) — continuous + interim, restarted on `end`, Safari's late/never finals synthesized with word timing over the window their interims were seen in |
| Coaching LLM | Gemini Nano / Apple FM / bundled → cloud | **cloud only** (no on-device model in a browser); the server's streaming suggestion (p50 ≈ 1.1 s in production) answers each `turn_local` |
| TTS | expo-speech (native) | expo-speech's web build → `window.speechSynthesis`, unlocked inside the Start tap (iOS drops the first utterance otherwise); earpiece/speaker modes only |
| Haptics | expo-haptics | none (iOS Safari has no vibration API); the on-screen nudge flash still shows |
| Mic capture | expo-audio PCM stream | `webAudioCapture.ts` (getUserMedia + AudioWorklet) → the hook resamples to **16 kHz mono int16** — the SAME frames go to the WebSocket AND the loop (`handleAudioBuffer`) |

Capability gate (`capability.ts` on web): capable when the browser has both
the Web Speech API and getUserMedia + AudioWorklet. Firefox (no Web Speech
API) gets the legacy server path with the reason on screen, exactly as
before this change.

Therapist mode (on-screen only, never speaks) is enforced in two places on
the web too: the loop never calls `speak` in therapist mode, and the hook
now refuses to voice the cloud's `suggestion` event while a therapist-mode
loop is active.

## Build + deploy (the owner runs this; agents never deploy)

```bash
# one-shot: build with the production env from apps/mobile/eas.json, verify, deploy
scripts/web_deploy.sh
# build + verify only
scripts/web_deploy.sh --dry-run
```

`npm run build:web` (apps/mobile) = copy the ORT runtime into
`public/ort/` (git-ignored, derived from node_modules) + `expo export
--platform web --output-dir dist`. The deploy script bakes
`EXPO_PUBLIC_API_URL` and `EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID` from
`eas.json`'s production profile and refuses to deploy if the bundle does
not contain the production API URL. `firebase.json` now sets long
immutable caching for `/ort/**`, `/_expo/static/**` and `/assets/**` and
an explicit `application/wasm` type.

**No COOP/COEP headers on purpose**: multi-threaded wasm would need
`crossOriginIsolated`, and COOP `same-origin` breaks the Google sign-in
popup. Single-threaded is fine for these models (Silero < 1 ms/chunk).

Server side: `server/main.py` now exposes `ETag`, `X-Model-Unavailable`
and `Content-Length` through CORS so the browser can revalidate the model
and read the 503 reason (they are not CORS-safelisted). The production
`MINDSHIFT_ALLOWED_ORIGINS` must include `https://arborfam-hub.web.app`
(it already does for the existing web sign-in / API calls to work).

## Verified locally (headless Chrome against the real export)

- The export loads and mounts (the login screen renders; no console errors).
- Bundle carries the production API URL, the Google web client id, `"/ort/"`
  and `webkitSpeechRecognition`; **zero** `localhost:8000` literals (the
  `||` fallback is folded away by the minifier).
- `/ort/ort.wasm.min.js` defines `globalThis.ort` (1.24.3); a Silero
  session builds from the exported asset URL (154 ms) and from bytes;
  0.87 ms/chunk single-threaded (`crossOriginIsolated=false`);
  p(silence) = 0.004; a Cache-API round trip of the model bytes works.
  (Real-speech probabilities are pinned by the Jest `liveVad` suite over
  onnxruntime-node — the same model file.) `scripts/web_smoke.mjs`
  re-runs the mount + ORT checks against `dist/` in headless Chrome and
  `scripts/web_deploy.sh` runs it before deploying when Chrome is present.
- Jest: `liveSttWeb`, `liveOrtWeb` (runtime loader, session factory,
  Cache-API model store + the download/revalidate protocol, the web loop
  builder's degradation ladder), `liveCapabilityWeb`, `useAudioStreamWebLive`
  (gesture priming, one frame stream → WS + loop, therapist never speaks,
  mic-released reporting, legacy path untouched). 112 suites / 1394 tests
  green; `tsc --noEmit` clean.

## iOS Safari: what should work vs. known limits

Works (by design, unverified on hardware):
- Sign in, therapist dashboard, session detail, growth — all fit a
  phone-width viewport (sparklines now size to the window; Export falls
  back to the clipboard when `navigator.share` isn't usable).
- Live Coach with the on-device loop: VAD/segmentation on the phone,
  speaker-ID once the model is cached, browser STT, cloud suggestions,
  spoken via the phone's own TTS (earpiece/speaker) or on screen only
  (therapist).
- Everything degrades with the reason on the "On-device: …" line — no
  wasm ⇒ energy VAD + speaker-ID off; no model ⇒ speaker-ID off; STT
  failure ⇒ the server's transcript takes over mid-session.

Known-limited:
- **Foreground only.** Locking the screen or switching apps releases the
  microphone (iOS). The app now says so (track `ended` → banner) and the
  idle explainer tells the user to keep the page open. Restart the session
  afterwards.
- **Two permission prompts** on the first Start (speech recognition +
  microphone), both gated on the tap — the code starts recognition and
  the AudioContext synchronously inside the tap for that reason.
- **Web Speech API is Apple's service, not on-device**: the audio for
  recognition goes to Apple. In the protocol `transcript_source:
  "on-device"` means "the client produced the words" (vs. the server's
  Deepgram) and is what the server keys on; the status line says
  "browser speech recognition".
- Recognition **restarts** every ~60 s / after pauses (Safari ends it);
  each restart loses a few hundred ms of speech. Finals are sometimes only
  flagged at `end`; the recognizer finalizes a pending interim itself and
  times it over the window its interims were seen in so the aligner does
  not smear it.
- Whether Safari lets `SpeechRecognition` and `getUserMedia` capture the
  mic **simultaneously** is the one thing that must be confirmed on a
  real iPhone. If recognition reports `audio-capture`/`not-allowed`, the
  session keeps running on the server transcript (banner says so).
- Speaker-ID's first session downloads ~80 MB (over Wi-Fi ideally); Safari
  evicts the Cache API after ~2 weeks without a visit, so it may download
  again later. Single-threaded ECAPA on a phone: a few hundred ms per turn
  (runs in parallel with the STT wait).
- No haptic nudges (no vibration API in iOS Safari).
- `speechSynthesis` voices are Safari's; volume follows the ringer switch
  on some iOS versions.

## Files

- New: `src/live/ortWeb.ts`, `src/live/modelStoreWeb.ts`,
  `src/live/sttWeb.ts`, `src/live/webDeps.ts`, `src/utils/webSpeech.ts`,
  `scripts/web_copy_ort.mjs`, `scripts/web_deploy.sh`, the four tests
  above.
- Touched: `src/live/ort.ts` (a session can be built from bytes),
  `src/live/capability.ts` (web gate), `src/live/defaultDeps.ts`
  (handler extras: `onStatus`, primed `recognizer`),
  `src/hooks/useAudioStream.ts` (web session wiring, therapist gate),
  `src/utils/webAudioCapture.ts` (track ended/muted + visibility resume),
  `LiveCoachScreen.tsx` (web foreground note), `TherapistDashboard.tsx` /
  `SessionDetail.tsx` (phone-width sparklines, web export fallback),
  `firebase.json`, `server/main.py`, `apps/mobile/package.json`
  (`build:web`, `onnxruntime-web` pinned), `.gitignore`.
