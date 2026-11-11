# Emotion on the watch (Phase 2) - 2026-11-10

Phase 1 (2026-11-07, `2026-11-07-emotion-dev-mode.md`) runs TINY and SMALL on the PHONE and only
buzzes the wrist. Phase 2 runs **TINY on the watch itself** (no phone, no server in the decision
loop). SMALL stays phone-only. Off by default; owner-only experiment; no accuracy claim.

## What runs where

```
mic (AudioRecord, 16 kHz mono, own thread)            EmotionLocalService (foreground, microphone)
  -> 6 s ring buffer, one tick every 3 s              LocalEmotionRunner
  -> speech-energy gate (-45 dBFS, 30% of 100 ms frames)   skip silence: no model call
  -> LogMel.compute()  (Kotlin port of logMel.ts)
  -> OnnxTinyEngine    (ONNX Runtime Android, 1 thread, int8 tiny)
  -> NudgeGate         (minProb 0.5, 2 consecutive windows, max 4/h, min gap 120 s, neutral never)
  -> HapticDirector.playEmotionNudge()   local double tap
  -> WindowLog         rolling emotion-windows.jsonl, numbers only
  -> every 5 min / on stop: POST /emotion/watch-windows -> server queue -> phone polls /emotion/ctl
```

Same inputs and outputs as the phone: `log_mel` float32 [1,80,600] + `valid_frames` int64 [1] ->
six probabilities (anger, disgust, fear, happiness, neutral, sadness).

## Parity (what is actually checked)

* `LogMelParityTest` runs the Kotlin log-mel on the **same golden file** the TS test uses
  (`apps/mobile/src/emotion/__fixtures__/clip04-*`, HF `WhisperFeatureExtractor` output): max abs
  diff < 1e-3, mean < 1e-4, same bar as `emotionLogMel.test.ts`.
* `EmotionModelParityTest` (optional, needs the private weights via `EMOTION_TINY_MODEL=...`) runs
  Kotlin log-mel -> desktop ONNX Runtime -> probabilities and compares with the reference
  probabilities recorded in `clip04-meta.json`. Run 2026-11-10: max |diff| 0.0095, same top class
  (neutral 0.520 vs 0.520). The residual is int8 kernel / runtime-version noise.
* `NudgeGateTest` pins the nudge defaults to the phone's `EMOTION_DEV_DEFAULTS`.

## Getting the model onto the watch: Hosting download, sideload as fallback

Owner approved hosting the tiny int8 file on Firebase Hosting, like the ECAPA model. `TinyModelSpec`
(ModelDownloader.kt) pins the filename `emotion-tiny-int8-<sha12>.onnx` and its sha256.
`scripts/web_deploy.sh` reads that pin, copies the local file into `dist/models/` (aborts if missing,
<= 5 MB, or the hash differs) - so the next normal web deploy publishes it:
`https://arborfam-hub.web.app/models/emotion-tiny-int8-ae4d75f72b2b.onnx`.

On the watch, turning the toggle on with no model present downloads that URL (status row shows
"Downloading model... N%"), verifies size and sha256, and installs it as `emotion-tiny-int8.onnx` in
the app files folder; a bad download (e.g. Hosting's SPA index.html) is deleted, never installed.
A file already there (sideload) is used as-is:

```
adb -s <watch> push emotion-tiny-int8.onnx /sdcard/Android/data/com.sagearbor.mindshift.app.debug/files/emotion/
```

(`.debug` = sideloaded dev build; Play id is `com.sagearbor.mindshift.app`.) Failures show in Settings
("Model download failed: ...").

## Size cost

ONNX Runtime Android adds native libs. The shipped build is limited to `arm64-v8a` + `armeabi-v7a`
(x86 emulators cannot run the mode). Release APK after R8: **48.5 MB** (debug 79.7 MB) (before ORT: not measured here).
Well under Play's limits but a large download for a watch; if that
matters, a follow-up can ship the mode in a separate flavor.

## Latency and battery: honest status

**No watch was reachable, so on-watch latency and battery are UNMEASURED.** What exists:

| number | value | where |
|---|---|---|
| tiny inference, desktop JVM, 1 thread | ~12 ms (median of 20) | `EmotionModelParityTest` output |
| log-mel, desktop JVM, 2.7 s clip | ~12 ms | same |
| watch CPUs vs a desktop core | typically several times slower | expectation, not data |

Rough expectation (a guess, to be replaced by the logs below): well under 1 s of compute per 3 s hop,
i.e. one busy core under ~25% on speech, ~0% on silence (gated). The mic stays open the whole time,
which is likely the larger battery cost. Do not quote any watch figure until the log says it.

To measure on the wrist, every classified window is logged with `ms` (log-mel + inference wall time)
and `b` (battery %), and Settings shows count, silent-skip count, buzzes, avg/max ms and unsynced rows.
Read the numbers back:

```
adb -s <watch> exec-out run-as com.sagearbor.mindshift.app.debug cat files/emotion/emotion-windows.jsonl > windows.jsonl
```

Battery: note `b` at start/end of a few hours with the mode on vs off (the sentinel off in both).

## Sync to the phone

Watch -> `POST /emotion/watch-windows` (device-token auth, <= 200 rows, rows that are not probability
vectors are dropped, never repaired) -> the account's ctl queue as one `windows` item -> the phone's
existing `GET /emotion/ctl` poller -> `EmotionDevController.handleControl` writes each row to the
Emotion dev log as `kind: "watch"` (small = null, never drives a buzz). Numbers only; there is no
audio field anywhere in the schema. Limits, deliberately accepted: the server queue is in-memory with
a 15-minute TTL and is per-process, so rows only reach the phone if the phone is polling within that
window; the watch's own file is the durable copy (`unsynced` count in Settings). The cursor advances
when the SERVER accepts a batch, not when the phone drains it.

## Settings toggle

Watch Settings -> **Emotion on watch** (default OFF). Turning it on asks for the mic permission if
needed, starts the foreground service (ongoing notification "Listening for emotion on this watch"),
and shows live status under the toggle. Off stops the service. `MainActivity.onStart` re-asserts the
pref, so the mode resumes after a reboot as soon as the app is opened.

## Known risks / unverified

1. Two `AudioRecord`s open at once (this service + the sentinel + the preview meter). Legal on
   Android 10+, but whether a watch's audio HAL keeps all of them fed is untested; the symptom is the
   status "Mic lost".
2. Mic stays open while ON, indefinitely - the biggest battery unknown.
3. ORT 1.24.3 int8 kernels on the watch's ARM cores: parity was only proven on desktop.
4. The service has no auto-start without the app being opened after a reboot (Android forbids
   starting a microphone foreground service from the background).
