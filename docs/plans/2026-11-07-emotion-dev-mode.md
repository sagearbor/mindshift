# Emotion dev tester mode — phase 1 (2026-11-07)

Owner-only, opt-in, OFF by default. Settings -> Advanced -> (turn on Developer mode under Diagnostics) -> Experimental -> **Emotion dev tester**.
Purpose: wear the watch all day and judge two experimental voice-emotion models
against real life. It is a measuring instrument, not a product feature, and
nothing in it claims accuracy.

> **displayed | method | latency | computed** — see docs/ICON-LEGEND.md

| feature | tags |
|---|---|
| Emotion label + buzz on the wrist (TINY, rate-capped) | ⌚ \| 📳 \| 🧠 \| 📱 |
| Latest emotion + optional suggestion screen | ⌚ \| 📝 \| 🧠 \| 📱 |
| Pause / resume, three flags (✗ ✓ ignore) | ⌚ \| 📝 \| ⚡ \| 📱 |
| SMALL model, silent, logged | – \| 🔕 \| 🧠 \| 📱 |
| Flag clip (45 s before + 15 s after), disagreement clip | – \| 🔕 \| ⏳ \| 📱 |
| Settings panel (switch, buzz budget, usage, export, delete-all) | 📱 \| 📝 \| ⚡ \| 📱 |

## How to test (owner, ~5 min once set up)

1. Put the two model files on the phone (Setup, below). Merge + deploy server, publish the phone OTA, install the watch build (your calls; see "Rollout").
2. Phone: Settings -> Advanced -> Developer mode on -> Experimental -> **Emotion dev tester** -> on. Status should read "Waiting for a Live Coach session…".
3. Start Live Coach (that is the mic source). Status becomes "TINY nudging, SMALL running silently. N windows…". If it says "Models missing", the status line names the folder it looked in.
4. Watch in companion mode (so its server socket is open). Settings -> **Emotion dev tester** appears after the first frame arrives. Speak; a double-tap buzz + label means TINY thought something non-neutral twice in a row.
5. Tap ✗ / ✓ / Ignore on the watch (or the phone panel). An un-tapped buzz is logged "unrated".
6. Review: phone panel -> **Export log** (JSONL, probabilities + flags, no audio). Clips: `adb pull /sdcard/Android/data/com.sagearbor.mindshift.app/files/emotion/data/clips`.

## What it does

```
live mic (16 kHz mono int16, the frames Live Coach already has)
   -> hook.handleAudioBuffer -> emotionDevHub.pushPcm        (no-op when OFF)
        -> PcmRing (60 s, in memory) + ClipScheduler
        -> every 3 s: last 6 s -> speech-energy gate -> ONE log-mel (logMel.ts)
              -> TINY  (int8 ONNX)  -> NudgeGate -> buzz?  -> WatchLink -> watch
              -> SMALL (int8 ONNX)  silent
        -> both RAW probability vectors -> emotion-windows.jsonl
        -> nudge / disagreement / flag records -> emotion-log.jsonl (+ clips/*.wav)
```

* **Models** (Codex-trained Whisper encoder + 6-class head: anger, disgust, fear, happiness, neutral, sadness). TINY ~10 MB nudges; SMALL ~95 MB runs silently. They over-predict *neutral*; per-class logit offsets helped only on acted data, so `LOGIT_OFFSETS` in `src/emotion/config.ts` ships all-zero (a no-op) and the log holds RAW probabilities. Perceived-voice estimate only.
* **Frontend** `src/emotion/logMel.ts` reimplements HF `WhisperFeatureExtractor` (80 Slaney mels, n_fft 400, hop 160, periodic Hann, reflect centre-pad, log10, max-8 floor, (x+4)/4, 6 s zero pad, 600 frames, `valid_frames = min(300, ceil(n/320))`, no gain normalisation). Parity is pinned by `__tests__/emotionLogMel.test.ts` against a golden vector made by `scripts/emotion_golden.py` from MESS clip-04 (CC BY 4.0, `src/emotion/__fixtures__/ATTRIBUTION.md`): max abs error < 1e-3. With the real models present, `__tests__/emotionModelParity.test.ts` also checks TS-frontend + onnxruntime-node probabilities against Python's within 0.02 (it passed on the build machine, and skips — loudly — elsewhere; weights are never in the repo).
* **Nudge rule (TINY only)**: top class not neutral, probability >= 0.5, same class on 2 consecutive windows, at most `maxNudgesPerHour` (default 4, settable in the panel) and >= 120 s apart. All constants in `config.ts`.
* **Flags**: ✗ wrong, ✓ good (optional), ignore ("not a conversation, e.g. media"). A flag targets the latest buzz if < 10 min old, else the latest window; it saves **45 s before + 15 s after** (snapshotted at tap time; completes when the 15 s have arrived; shorter if less history exists). Pause drops the ring, so a flag cannot reach across a pause; flags are refused while paused.
* **Disagreement clip**: TINY and SMALL name different top classes, both >= 0.6 -> an 8 s clip, <= 20/day (`DailyCap`). Independent of whether TINY also buzzed.
* **Record** per window of interest: timestamp, both RAW probability vectors, nudge sent y/n (+ reason when not), flag (`bad|good|ignore|unrated`), clip file name.
* **Pause** on the watch and on the phone: stops listening AND buffering.
* **Media playing**: NOT detected. Android exposes it only through native `AudioManager.isMusicActive()` / `MediaSessionManager`; reading it needs a new phone native module, which changes the native fingerprint and forces a store build (see Transport). Use Pause or the Ignore flag. Noted for a later native release.

## Storage and privacy

* Audio ring is memory-only. Clips and logs are written **on the phone only**, under `.../files/emotion/data/` (Android external app folder, reachable by `adb pull`; falls back to the app documents folder `emotion-dev/` if that folder is not writable). Caps: clips 150 MB (oldest evicted), each log 8 MB (oldest half dropped).
* **Delete all** (panel) removes every clip and both logs. **Export log** is an explicit Share-sheet action and sends the JSONL log only. Audio never leaves the phone from this feature; there is no upload code path. Pulling clips is a manual `adb pull`.
* Server sees only {label, probability, nudge, paused, 80-char suggestion} in transit, plus pause/flag verbs; nothing is stored there (in-memory, 100 items, 15 min TTL).

## Setup: putting the (private) models on the phone — no git

The weights live in the private `mindshift-emotion-private` repo and must never be copied into this repo. On the Mac:

```
D=/sdcard/Android/data/com.sagearbor.mindshift.app/files/emotion
R=/Users/sagearbor/projects/githubs/mindshift-emotion-private/results
adb shell mkdir -p $D
adb push $R/tiny-finetuned-v1/export/emotion-int8.onnx  $D/emotion-tiny-int8.onnx
adb push $R/small-finetuned-v1/export/emotion-int8.onnx $D/emotion-small-int8.onnx
```

The app uses them in place (no 105 MB copy). Search order: that folder, then `<app documents>/emotion-models/`. A file smaller than 5 MB (tiny) / 50 MB (small) is reported as truncated. Without TINY the mode says "Models missing — …"; without SMALL it runs TINY-only and says so. Tap **Check again** after pushing. *Unverified: whether `adb push` into `Android/data` is allowed on the Pixel 10's Android version (restricted for some shells on 14+); if not, push to `/sdcard/Download/` and copy via the Files app into the app folder, or tell me and I will add an in-app file picker (`expo-document-picker` is already a dependency).*

## Transport (phone <-> watch): decision

Checked: the watch does **not** use the Wear Data Layer (no `play-services-wearable`); it talks to the server over OkHttp (`WatchApiClient`, `EpisodeWsClient`), and companion mode already keeps a socket open all day with the server relaying phone events to it (`server/watch/relay.py`).

Options: (a) **Data Layer `MessageClient`** — same `applicationId` (`com.sagearbor.mindshift.app`) on both, so feasible, local and private; but the phone side needs a **new native module**, and this repo's `nativeParity` guard (`native-fingerprint.json`) makes any `modules/**` change require an `expo.version` bump and a store build of **both** platforms. (b) **Existing server websocket** (chosen) — no phone native change, ships as an OTA, reuses auth and the socket the watch already holds.

Chosen (b). Phone -> watch: `POST /emotion/event` -> server pushes an `emotion` frame down the watch socket. Watch -> phone: `emotion_ctl` frame up the socket -> server queue -> phone polls `GET /emotion/ctl?after=<seq>` every 2 s. Cost: needs internet on both devices, and the server must be deployed (`server/watch/emotion_bridge.py`); latency is a few hundred ms plus up to 2 s polling for watch->phone; the relay registry is per-process like every other relay feature. The phone talks only to `WatchLink` (`watchLink.ts`) and the watch to `EmotionDevModel`, so a Data Layer transport can replace the server one later without touching logic. If the owner prefers it, a Data Layer version is a contained follow-up (new phone module + `play-services-wearable` + a `WearableListenerService`), at the price of that store build.

Testability: `FakeWatchLink`/`ServerWatchLink` with a fake `fetch` (TS), `EmotionDevModel` with a fake sender and `MockWebServer` (Kotlin), FastAPI `TestClient` websocket tests (Python).

## Phase 2 (not now): TINY on the watch itself

The seam is `EmotionEngine` (`src/emotion/engine.ts`): prepared 80x600 log-mel + `valid_frames` in, six probabilities out; everything above it (gates, caps, flags, store, transport) is engine-agnostic. On the watch, the equivalent is a Kotlin engine producing the same `EmotionFrame`s that `EmotionDevModel.onFrame` already consumes (today they come from the socket), so the screen, buzz and ctl verbs stay unchanged. Needed: Whisper log-mel in Kotlin (port `logMel.ts`; same golden vector), an ORT-Android build of the 10 MB int8 model, the watch mic, and a battery budget — none of which are measured here.

## Rollout (nothing done by this PR)

1. Merge; deploy the server (adds `/emotion/event`, `/emotion/ctl`, `emotion`/`emotion_ctl` frames). Prod env is wiped by `deploy_cloudrun.sh` — follow the usual deploy notes.
2. Phone: **OTA only** (JS-only change; no native inputs touched, `nativeParity` passes, no version bump made). Use `./scripts/ota_publish.sh`.
3. Watch: build/install the wear app (Kotlin changes; no version bump made).
4. Push the two model files.

## What I could not verify (honest list)

* **No phone, no watch, no Pixel** were available. Nothing here has run on a device: the ONNX sessions via `onnxruntime-react-native`, the sideload paths (`Android/data` writability), the `ExpoRecorderFs` on external URIs, the Settings panel on screen, the live-hook integration (2 added lines; type-checked, but the hook's jest suites cannot run in this checkout — see below), the watch screen/navigation, the double-tap buzz feel, and the end-to-end websocket round trip.
* **Latency and battery are unmeasured.** Only datum: the TS log-mel for one full 6 s window takes ~30 ms in Node/V8 (inside the jest run); Hermes on a phone will be slower, and TWO ONNX inferences per 3 s window plus the mic are real battery cost on a phone that is also running Live Coach. Expect to tune `hopS`, or gate harder, after a first day.
* **Audio source is Live Coach's mic only.** The mode is a consumer of the frames a live session already captures; with no live session there is no audio. It is not a standalone all-day background recorder. If the OS suspends the app screen-off, windows stop.
* **The models have not been validated on real speech.** On the one golden clip (MESS clip-04, human label: angry) TINY says neutral 0.52 / disgust 0.26 / anger 0.16 and SMALL says anger 0.52 / neutral 0.31 — a live illustration of why this tool logs raw probabilities and flags, and of the disagreement trigger. They hear any voice (TV, others), not just the wearer; the Ignore flag and Pause exist for that.
* The speech-energy gate (-45 dBFS, 30% of 100 ms frames) and every threshold are untuned guesses.
* Server relay is per-process; a multi-instance Cloud Run deployment without affinity could drop a push or hide a control (same limitation as the existing phone->watch relay).
* Test environment: 15 jest suites (the `useAudioStream*`, `LibraryScreen`, `liveCoachSafety`, … family) fail to even load here with `Cannot find module 'test-renderer'` from `@testing-library/react-native` — a missing dev dependency in this machine's shared `node_modules`, unrelated to this change; the other 221 suites pass (2468 tests), including `AdvancedScreen`, `nativeParity` and every new emotion test. Python: `server/tests/watch` 483 passed. Watch: `:wearApp:testDebugUnitTest` 465 tests, 0 failures (JDK 17).
