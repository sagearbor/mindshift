# Realtime coaching build — three-track handoff (2026-08-24)

> **Read this if you're a fresh Claude Code session (or Sage) picking up the
> realtime work.** It supersedes the "poker6 diarization status" half of
> [2026-08-24-mac-transition-and-poker6-status.md](2026-08-24-mac-transition-and-poker6-status.md)
> (poker6 is RESOLVED, see §1). Plan of record:
> `~/.claude/plans/glittery-doodling-kay.md` on Sage's Mac (mirrored in §2).

## 0. The one-paragraph version

The app now has an **on-device-first realtime coaching loop** on the phone
(capture → Silero VAD → speaker-ID → on-device STT → local LLM → expo-speech
earpiece, all without the server in the critical path), the **server turned
into async enrichment** (cloud suggestion, audio tone, identity confirmation,
watch relay, latency telemetry), **live sessions flow into the same
over-time analysis** as uploads (Growth/YourDay/Dynamics/Therapist + a cached
"what you could have said" pass), and the **watch nudge** shares one
escalation policy with the phone via golden test vectors and now escalates
on tone, not just volume. Everything is merged to `main`; every suite is
green. What's left is on-device verification on real hardware (§6).

## 1. Poker6 — resolved (PR #129)

The shipped ECAPA k-search was rejecting the real 6th voice by a hair
(marginal cosine 0.301 vs the 0.30 bar; anchor 0.231 vs 0.20). Recalibrated
`STRONG_SEPARATION_COSINE` 0.30→0.32 and `NEW_VOICE_ANCHOR_COSINE` 0.20→0.24
against every real, checked-in fixture: **poker6 6/6 = 100%**, no new deps,
no manual speaker count (the owner's hard rule). Two tests that "protected"
the old values were built on a fixture the repo's own README says never to
use for diarization (`tmp/test_recording.wav`, gain/tempo-modulated single
voice) — repaired to real fixtures. pyannote.audio was evaluated and archived
(`docs/research/poker6-sliding-window/`): 83% at best, ~800 MB of deps,
over-segments 2-speaker audio in auto mode.

## 2. What shipped (in merge order, all squash-merged to `main`)

| PR | What | Key numbers |
|---|---|---|
| #130 Foundation A | canonical `server/nudge_policy.py` (watch module re-exports), golden vectors `server/tests/fixtures/policy_vectors/{nudge_policy,vad_segments}.json`, realtime WS models `TurnLocalEvent`/`ToneFlagEvent`/`SpeakerIdentityEvent`, `SuggestionEvent.suggestion_source` | 14 nudge + 11 VAD vector cases |
| #131 Foundation C | `server/tone_id.py` — wav2vec2-IEMOCAP audio tone, pinned, lazy, `MINDSHIFT_TONE_AUDIO=off\|dark\|on` | **ships DARK**: 40% 4-class / 50% arousal on our acted fixtures — it reads voice identity, not emotion. Gain-invariant, ~150 ms/turn CPU, 758 MB cache. Report: `docs/research/tone-audio/` |
| #132 Foundation B | `scripts/export_ecapa_onnx.py` (+ parity test), multi-person named voiceprints (`voiceprints/{uid}/{person_id}/profile.json`, `identify_speakers_multi`, `/voice/people` CRUD; legacy single print read as `self`) | ONNX 80 MB opset-17, cosine(onnx, torch) = **1.000000**, **~15 ms per 1.5 s clip** on CPU |
| #133 Foundation D | `scripts/make_test_recording_scenes.py` + 3 labeled scenes (2/3/4 voices, self voice, `emotion_coarse`, `expected_nudges`), `test_diarize_scenes.py` | couple 13/13, family3 15/15, meeting4 **11/17 (pinned ceiling — OpenAI TTS voices are near-twins; self still isolated 6/6)**; +6.7 MB; ~$0.13 TTS |
| #134 | watch enroll tests made venv-independent | — |
| #135 Track 1 | Kotlin replays the golden vectors (Gradle `syncPolicyVectors`), PRD §6 haptic schedule (pure mapper + wearApp), `server/watch/relay.py::push_turn_local` (self-turns only; tone raises level; max with dB path), `tone_escalation.json` | 0 vector mismatches; wear emulator boot **662 ms, 0 crashes**, UI rendered |
| #136 Track 3-server | `audio_pipeline.py`: per-stage latency log + `latency_summary` on `session_complete`, `turn_local` handling (Deepgram overlap suppression, cloud suggestion w/ tone context, `partial` streaming previews, server TTS off when local-first), PCM ring buffer enrichment (audio tone → `tone_flag` if surfaced, identity → `speaker_identity`, watch relay) | 37 new tests; legacy clients byte-identical |
| #137 Track 2 | `POST /sessions/live` (idempotent, episodes = recordings with `media_type: none`), `POST /episodes/{id}/reflect` (cached "could have said"), `GET /sessions` for the therapist dashboard, tone + per-person dimensions in `/growth` and the screens | scene-pack ingest test: escalation turns == `expected_nudges`, no non-self leakage |
| #138 Track 3-mobile | `apps/mobile/src/live/`: `prosody.ts`, `ort.ts` (+node impl for Jest), `vad.ts`/`segmenter.ts` (Silero v6 committed, 2.3 MB), `speakerId.ts`, `stt.ts` (expo-speech-recognition), `localLlm.ts` (expo-ai-kit: Gemini Nano → Apple FM → LiteRT-LM bundled → cloud, refusal falls through), `nudgePolicy.ts`, `fastLoop.ts` (earpiece / speaker / therapist modes), wired into `useAudioStream`/`LiveCoachScreen` behind an STT capability gate; `plugins/withOrtGradle9.js` | Silero **0.26 ms/chunk**; all vectors replay; Android debug APK **builds and boots** on `pixel10_api35`, 0 crashes |
| People labeling (`feat/people-labeling`) | "That's Mom" ONCE, recognized everywhere: `POST /voice/people/{id}/enroll-from-recording` (learn a voice from a stored recording's diarized speaker; honest 422s `[too-little-speech]` / `[sounds-like-someone-else]` / `[no-audio]` — see `speaker_id.enrollment_conflict`), `PATCH /voice/people/{id}` rename, the `manual-person` label rung (`PATCH …/speaker-labels` `people` map → meta.json `manual_speaker_people`; a manual "self" counts for /growth; person ids flow into `/growth` people rows + `GET /sessions` rows), mobile People screen (Settings → People, hamburger catalog), "Who is this?" sheet on Replay + the therapist SessionDetail | live-gated scene test: self enrolled from couple via the endpoint → found in family3 (only self); guard refuses the partner voice tapped as someone else |

Suites on `main` after #139: pytest **1514 passed** (1 known live-Deepgram
network test deselected — pre-existing vendor issue), Jest **108 suites /
1366 tests**, `tsc` clean, Gradle `:shared:allTests` + `:wearApp` unit green.
After people labeling: pytest **1531 passed**, Jest **113 suites / 1399 tests**.

## 3. What shipped DARK (built, measured, silenced — not forgotten)

- **Audio tone (wav2vec2-IEMOCAP)**: `MINDSHIFT_TONE_AUDIO=dark`. Computed
  and logged server-side per turn, never surfaced, never relayed to the
  watch. Text-tone (from the LLM call) is what drives tone today. Cheapest
  next experiment per the report: per-speaker logit normalization; better
  bet: a naturalistic dimensional arousal/valence model. **Ops note:** with
  voice deps installed the first ≥1 s `turn_local` triggers a one-time
  ~750 MB model fetch in a thread.
- **On-device speaker-ID**: implemented in #138, made live by **#139**
  (`GET /models/ecapa.onnx` with ETag/304 + one-time server-side export,
  `GET /voice/people?include_embeddings=true`, client download-once).
  Jest parity against the torch reference on real fixtures: cosine
  **1.000000**, **~20 ms per 1.5 s slice** (onnxruntime-node); server side
  15.7 ms. LiveCoachScreen's "On-device: …" line now says `speaker-ID on
  (N enrolled, model cached)` or the reason it's off. Still unmeasured on
  a real phone (§6).
- **Bundled LLM tier**: uses expo-ai-kit's `getRecommendedModel()`; Gemma 3
  1B was rejected because its HF download is gated (401) for end users.

## 4. Decisions worth knowing (so nobody re-litigates them)

- The fast loop never touches the server; the phone is the orchestrator.
  Older/incapable phones (no on-device STT) silently get the legacy server
  path — that IS the fallback, no separate code path.
- Cross-language sharing (Kotlin/TS/Python) is done with **golden test
  vectors**, not shared code. Add a case to the JSON → all three suites
  must pass it.
- One embedding model everywhere (the pinned speechbrain ECAPA, exported to
  ONNX) so phone/batch/watch voiceprints stay compatible. Not WeSpeaker,
  not FluidAudio.
- `turn_local.is_self` is a voiceprint verdict only; the watch escalates on
  self turns only (bias guard cases in `tone_escalation.json`).
- Never speak over live speech (earpiece and speaker-phone); therapist
  mode never speaks; cloud suggestion is voiced only if every local
  provider fell through.
- Native modules ⇒ **dev build**, never Expo Go, and never raw `eas update`
  (AGENTS.md).

## 5. Machine/toolchain state on Sage's new Mac (all installed 2026-08-24)

JDK 17 (`/opt/homebrew/opt/openjdk@17`), Android SDK at
`~/Library/Android/sdk` (platform 35 + 34, build-tools, emulator, `adb`,
cmdline-tools copied into the SDK root so `avdmanager` works), Android
Studio, AVDs `pixel10_api35` and `wear_os5`, `JAVA_HOME`/`ANDROID_HOME` in
`~/.zprofile`; `apps/watch/local.properties` (gitignored). Python venvs in
`tmp/` (`venv-voice` has torch+speechbrain+transformers+onnx/onnxruntime;
`venv-whisper` runs the plain suite; `venv-pyannote` is research-only).
`HF_TOKEN` in `.env`. A LaunchAgent keeps the Mac awake for Sage's account.
**Xcode 26.3 installed** later the same day (+ iOS 26.3 simulator runtime,
CocoaPods). `apps/mobile/ios` prebuilds and builds (PR #141 fixed the two
Podfile issues: workspace-hoisted ORT pod path, AppCheckCore static
frameworks); the Release app boots on the iPhone 17 Pro simulator. Run
`pod install` with `LANG=en_US.UTF-8`. Claude Code's sandbox shell does not source `~/.zprofile` — prefix
`export PATH="/opt/homebrew/bin:$PATH"`.

## 6. Only-Sage items (nothing else is blocked on these)

1. ~~Xcode~~ — done (see §5). The iOS providers still need a REAL iPhone
   15 Pro+ to exercise Apple FM / SpeechAnalyzer; the simulator only proves
   it compiles and boots.
2. **Pixel 10**: enable Developer options → USB debugging, plug in, tap
   Allow. A standalone release APK is already built and verified on the
   emulator (no Metro needed): `adb install -r
   ~/Desktop/mindshift-release-cf7310a.apk` (278 MB universal; points at
   the production Cloud Run URL from `eas.json`; signed with the debug
   keystore; frozen at cf7310a — no OTA will touch it). For a live-reload
   dev build instead: `cd apps/mobile && npx expo run:android --device`.
   On first launch the app downloads `ecapa.onnx` (~80 MB) once — which
   requires the server deploy below.
3. **Enroll voices** in the app: yourself ("You") and "Mom" as a named
   person (new multi-person enrollment in Voice settings) before the demo.
4. **Demo run** (speaker-phone mode): start a live session in `speaker`
   mode, call Mom on speaker, watch the on-screen transcript label
   self/Mom, hear suggestions only while you're silent; at session end the
   per-stage latency log prints (target < 1.5 s to first spoken words —
   this number has NOT been measured on real hardware yet) and the session
   appears in Your Day / Growth with the "could have said" reflections.
5. Apple Intelligence toggle on any iPhone 15 Pro+ you test with.
6. **Play Store service account** (≈10 min, once): the 1.17.0 production
   AAB is built on EAS but nobody can `eas submit` it until a Google Play
   service-account key exists — see §9 for the exact clicks. Until then
   Android installs stay "via link" (the `preview` APK).
7. **Wear app upload key**: `~/.config/gauge/gauge-upload.jks` and
   `apps/watch/keystore.properties` are not on this Mac (§9) — copy them
   from wherever the Gauge upload key lives before any watch release.

## 7. Next steps, in order

1. **Deploy note (Cloud Run):** each instance exports the 80 MB ONNX on
   first `/models/ecapa.onnx` request (tens of seconds, ephemeral FS).
   Pre-export into the image (`python -m server.ecapa_onnx` / the
   `scripts/export_ecapa_onnx.py` CLI) or set `MINDSHIFT_ECAPA_ONNX_PATH`;
   a torch-less image that ships the file is supported. Nothing
   diarization/realtime-related has been deployed to production yet.
2. Real-hardware pass on the Pixel 10: Gemini Nano Prompt API refusal
   behaviour on coaching prompts (unknown until tried), on-device STT
   quality, per-stage latency, whether speaker-ID separates self/Mom
   through a speaker-phone mic (poker6/family fixtures say ECAPA can).
3. If Nano refuses: expo-ai-kit's LiteRT-LM tier is the middle rung; cloud
   is the floor. Measure, don't assume.
4. Tone: run the per-speaker-normalization experiment from the tone
   report; flip `MINDSHIFT_TONE_AUDIO=on` only when it beats text-tone on
   the scene pack.
5. Apple Watch: accepted gap (~6 months).

## 8. Installable builds via EAS + OTA path from the new Mac (2026-08-24, evening)

### Runtime isolation (read before any OTA)
`app.json` uses `runtimeVersion: {policy: "appVersion"}`, so the runtime
version **is** `expo.version`. The Play build (1.16.0 / versionCode 31,
EAS build `16087820`, channel `production`) predates every new native
module (onnxruntime-react-native, expo-ai-kit, expo-speech-recognition,
expo-build-properties). `expo.version` is therefore bumped to **1.17.0**
(versionCode 32): an OTA published from `main` now carries runtime 1.17.0
and can never be delivered to the 1.16.0 Play build. Consequences:
- The 1.16.0 Play users stay frozen on the last 1.16.0 OTA
  (`0a49071e…`, 2026-08-23) until a 1.17.0 production build ships to Play
  (`eas build -p android --profile production` + `eas submit`). To hot-fix
  1.16.0 you would have to publish from a checkout with `version: 1.16.0`
  and no new-native imports — don't; ship the store build instead.
- Bump `expo.version` again any time a native module is added/removed.

### Android — Pixel 10 (no USB needed)
- `eas build -p android --profile preview` (APK, internal distribution,
  channel `preview`, prod Cloud Run URL baked from `eas.json`). Android
  keystore is on EAS servers ("Build Credentials EEdFzvqWfA") — nothing
  to migrate from the old Mac.
- Build `555b04e3-9a28-4efb-8bbf-e9a63e2dcac7` → install page **https://expo.dev/accounts/sagearbor/projects/mindshift/builds/555b04e3-9a28-4efb-8bbf-e9a63e2dcac7**
  (APK `https://expo.dev/artifacts/eas/mgVE7KfXfo5864WpZkKOzxNbCCmglFqxBYqVbxXRDnU.apk`, 292 MB, ~13 min cloud build, first attempt succeeded).
- This APK is signed with the real upload keystore, unlike the
  debug-signed `~/Desktop/mindshift-release-cf7310a.apk` from §6 — the
  two will NOT install over each other (signature mismatch); uninstall
  the other one first. Same for a Play-Store-installed MindShift: Play
  App Signing re-signs with Google's key, so this upload-key-signed APK
  reports "App not installed / signature mismatch" over it — uninstall
  the Play copy first (recordings sync from the server; local-only
  drafts are lost).

### OTA path from this Mac
- `scripts/ota_publish.sh` gained `OTA_DRY_RUN=1` (auth + project id +
  channel check, local `expo export`, greps the compiled .hbc for the
  baked `EXPO_PUBLIC_API_URL`) and `OTA_CHANNEL=<name>` (default
  `production`). Dry run passes on this Mac.
- Published once to the `preview` channel (runtime 1.17.0 — only the
  APK above can receive it): update group
  `f453712b-331c-448b-ab99-4de6807ba433`
  (android `01a035f1-4918-7dd4-a93d-81e3a9bba496`,
  ios `01a035f1-4918-70cf-825b-2596571c3463`), commit `e1d6c7d`.
- Day-to-day: `OTA_CHANNEL=preview ./scripts/ota_publish.sh "msg"` for the
  Pixel/Mom builds; plain `./scripts/ota_publish.sh "msg"` for Play once a
  1.17.0 production build exists there.

### iOS — Mom's iPhone (owner-only; nothing else blocks on it)
State: the EAS account `sagearbor` has **no Apple team linked**
(`eas device:list` → "No Apple teams found"), no iOS build has ever run,
so there is no distribution cert / provisioning profile. Creating them
needs an Apple ID + 2FA, so it was not attempted. The repo is prepared:
`ios.bundleIdentifier` `com.sagearbor.mindshift.app`, `buildNumber` "1"
(auto-incremented by the `production` profile's `autoIncrement`),
explicit `NSMicrophoneUsageDescription` /
`NSSpeechRecognitionUsageDescription` (App Store review rejects
expo-speech-recognition apps without them) and
`ITSAppUsesNonExemptEncryption: false` (skips the TestFlight export-
compliance question); `eas.json` `submit.production.ios` has
`appleId` / `ascAppId` / `appleTeamId` placeholders.

Prerequisite: Apple Developer Program membership (~$99/yr, developer.apple.com,
takes up to ~48 h to activate). Then, from `apps/mobile`:

1. `eas credentials -p ios` → pick `preview` (or `production`) → "Log in
   to your Apple Developer account" → Apple ID + 2FA code. Let EAS
   create the distribution certificate and provisioning profile.
2. Pick ONE of:
   - **TestFlight (recommended for Mom — she installs the free TestFlight
     app and taps a link; up to 90 days per build, no device registration):**
     `eas build -p ios --profile production` then `eas submit -p ios
     --profile production` (fill the three placeholders in `eas.json`
     first — `ascAppId` is the numeric App Store Connect app id after you
     create the app record at appstoreconnect.apple.com; or just run
     `eas submit` interactively and it will create the record). Add Mom
     as an internal or external tester in App Store Connect → TestFlight.
   - **Ad-hoc (internal distribution; no App Store Connect record, but
     Mom's UDID must be registered first):** `eas device:create` (choose
     "Website" → EAS prints a link/QR; Mom opens it on her iPhone, taps
     "Register", installs the profile), then `eas build -p ios --profile
     preview`; send her the build page's install link (Safari only).
     Ad-hoc profiles are capped at 100 devices/yr and each new device
     needs a rebuild.
3. Either way the iOS build will receive the same `preview` /
   `production` OTAs as Android (runtime 1.17.0 published for both
   platforms).

## 9. Play Store track — why the phone still installs "via link" (2026-08-24, night)

**Short answer:** the Play Store copy of MindShift is still 1.16.0
(versionCode 31, EAS build `16087820`). Everything since (all of §2, the
on-device modules) needs the 1.17.0 native runtime, and a 1.17.0 binary has
only ever been shipped as the internal-distribution `preview` APK (§8) —
i.e. "via link". Getting 1.17.0 onto Play is two commands, and the second
one is blocked on a one-time, owner-only Google credential.

### What was done
- **Production AAB built on EAS** (Play-ready, `distribution: store`,
  channel `production`, runtime 1.17.0, versionCode **33**, commit
  `f9ecb2a`): build `7e08138d-86d3-4d53-8d44-7ed6e035f926` →
  https://expo.dev/accounts/sagearbor/projects/mindshift/builds/7e08138d-86d3-4d53-8d44-7ed6e035f926
  Status **FINISHED** (queued 23:20 UTC, built in 8 min 45 s). Artifact:
  https://expo.dev/artifacts/eas/z2j11IIVrj4-8q7pMIIIa7EQnlCBCdHkDYo9HJ6x_kw.aab
  (also downloadable from the build page; EAS keeps it 30 days, until
  2026-09-23 — submit before then or rebuild).
  `eas.json`'s `production` profile has `autoIncrement: true` with
  `appVersionSource: local`, so the build bumped `android.versionCode`
  32 → 33 **in app.json**; that bump is committed with this section
  (otherwise the next build would produce a second versionCode 33 and Play
  would reject it).
- **Submit was NOT possible** — verified, not assumed: the EAS project's
  Android credentials hold only the build keystore ("Build Credentials
  EEdFzvqWfA"); `googleServiceAccountKeyForSubmissions` is `null` (queried
  via the EAS GraphQL API, since `eas credentials` has no non-interactive
  mode), and there is no service-account JSON anywhere on this Mac or in
  the repo (`eas.json` has no `serviceAccountKeyPath`). The 1.16.0 build
  was evidently uploaded to Play by hand. (An `eas submit --id 7e08138d…`
  attempt from the agent session was additionally blocked by the Claude
  Code permission classifier; with the key in place it is a plain command.)
- **OTA plumbing checked:** EAS channel `production` → branch
  `production` (last group `0a49071e…`, runtime 1.16.0). The new build
  listens on channel `production`, so once it is on Play a plain
  `./scripts/ota_publish.sh "msg"` (channel default `production`, runtime
  1.17.0) reaches it — no script/EAS change needed. Until then, every
  `production` OTA published from `main` is delivered to nobody (no
  1.17.0 store build installed anywhere) and the 1.16.0 Play users stay
  frozen on the 1.16.0 OTA of 2026-08-23. Do not try to "fix" that by
  publishing 1.16.0 OTAs — ship the store build instead.
- `.gitignore` now excludes `apps/mobile/play-service-account.json` and
  `*service-account*.json`; AGENTS.md "Deploying" has the two-command
  Play recipe.

### Owner steps (once, ~10 minutes; everything else is scriptable after)
Play Console access cannot be delegated to an agent — it needs your Google
login. The Play app record already exists (1.16.0 is live there).

1. **Google Cloud project + service account.** Play Console → *Setup → API
   access*. Link a Google Cloud project (the Firebase/Cloud Run project
   is fine; `gcloud config` on this Mac points at `arborfam-hub`). Under
   *Service accounts* click *Learn how to create service accounts* → it
   deep-links to Google Cloud IAM → *Create service account*, name it e.g.
   `eas-play-submit`, no roles needed at the project level → *Done*. Then
   on that account → *Keys → Add key → Create new key → JSON* — a
   `<project>-<hash>.json` downloads.
2. **Grant it Play access.** Back in Play Console *API access* → the new
   account appears under *Service accounts* → *Manage Play Console
   permissions* → *App permissions* tab → add **MindShift**
   (`com.sagearbor.mindshift.app`) → *Account permissions*: tick
   *Releases → Release to testing tracks* (and *Manage testing tracks and
   edit tester lists*; not *Release to production* — promotion stays a
   manual Play Console click) → *Invite user* / *Apply*. Google Play
   sometimes needs ~24 h before a brand-new service account is accepted by
   the publishing API ("The caller does not have permission") — just
   retry later.
3. **Hand the key to EAS (pick one):**
   - *Recommended — stored on EAS, works from any machine/agent:*
     `cd apps/mobile && eas credentials -p android` → *production* →
     *Google Service Account* → *Set up a Google Service Account Key for
     Play Store Submissions* → point it at the downloaded JSON. (Delete the
     local JSON afterwards, or keep it at the gitignored
     `apps/mobile/play-service-account.json`.)
   - *Local file only:* save it as `apps/mobile/play-service-account.json`
     (gitignored) and add `--key ./play-service-account.json` to the
     submit command below.
4. **Submit** (agent-runnable once step 3 is done):
   `cd apps/mobile && eas submit -p android --profile production
   --non-interactive --id 7e08138d-86d3-4d53-8d44-7ed6e035f926`
   (`--latest` also works once this AAB is the newest finished production
   build). `eas.json` `submit.production.android.track` is `internal`, so
   this lands on **Internal testing**, never production.
5. **Testers.** Play Console → *Testing → Internal testing* → *Testers*
   tab → create an email list with your and your son's Google accounts →
   *Save* → copy the *Join on Android* opt-in link and open it on each
   phone → *Become a tester* → Play Store then shows MindShift 1.17.0
   (install over the existing Play copy; it will NOT install over the
   `preview` APK — different signing key, uninstall that first). Mom is on
   iPhone: iOS is §8 (TestFlight), unaffected by any of this.
6. **Promote to production** when you're happy: Internal testing release
   → *Promote release → Production* (owner-only; agents never do this).

### Wear OS app (`apps/watch`, `com.sagearbor.gauge.wear`) — status only
Not buildable for Play on this Mac: `~/.config/gauge/` does not exist (no
`gauge-upload.jks`) and `apps/watch/keystore.properties` is absent
(gitignored, must be recreated by hand). `wearApp/build.gradle.kts` falls
back to debug signing when the properties file is missing, so a
`bundleRelease` here would produce an AAB Play rejects (wrong upload key).
Nothing was built. To enable: copy `gauge-upload.jks` to
`~/.config/gauge/` and write `apps/watch/keystore.properties`
(`storeFile=/Users/<you>/.config/gauge/gauge-upload.jks`,
`storePassword=…`, `keyAlias=…`, `keyPassword=…`), then
`cd apps/watch && JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home ANDROID_HOME=$HOME/Library/Android/sdk ./gradlew :wearApp:bundleRelease`
→ `apps/watch/wearApp/build/outputs/bundle/release/wearApp-release.aab`
(currently versionCode 11 / 0.4.1; bump both in `build.gradle.kts` first).
The same Play service account from the steps above can submit it if you
add the wear app under *App permissions* — but there is no EAS project
for the watch, so upload it in the Play Console UI (or a separate
`gradle-play-publisher` setup) rather than `eas submit`.

## 10. Two-sided "Sage + his therapist Mom" (branch `feat/therapist-two-sided`)

What it adds, on top of the tracks above (decisions are in the PR body):

- **Live Coach**: one explicit mode picker (Earpiece / Speaker-phone /
  Therapist) with a one-line hint each, remembered PER ACCOUNT
  (`src/live/modePrefs.ts`); an honest pre-flight card (on-device STT,
  speaker-ID + the reason it's off, local LLM provider or "cloud", VAD) via
  `probeFastLoopCapabilities()` + a "who's here" strip from `GET /voice/people`
  (read-only); "on-device"/"cloud" tags on every suggestion; an escalation
  counter in the session strip; therapist mode renders a two-column
  transcript (`TherapistTranscript`) and never speaks; session end shows a
  summary card (duration, turns per person, escalations, first-words
  median/best from `latencyLog`) with "Share with my therapist" when linked
  and not auto-shared.
- **Therapist link** (`server/therapist_links.py`, `routers/therapist.py`,
  Settings → "My therapist"): the patient names ONE therapist by account
  email; `auto_share` defaults on; ingest (live session or stored upload)
  grants the therapist with the EXISTING per-episode share (`add_share`) —
  no second sharing system. The therapist accepts/declines from the
  dashboard; pending links still auto-share (the patient chose the
  recipient, exactly like a manual share); decline deletes the link.
  Earlier episodes are never back-shared.
- **Therapist dashboard**: pending requests, patient list ("You" first,
  linked ✓ + counts), pull-to-refresh; a shared session's detail shows
  escalation markers, named people, and a viewer-private note
  (`/therapist/notes/{episode}`).
- **Patient after the session**: `POST /sessions/live` result → optimistic
  row in Your Day (`liveEpisodeStore`, server row wins), pull-to-refresh on
  Your Day / Growth / Dashboard, and Replay polls a still-"lite" live session
  (5 s × 12) until the reflection lands.
- Jest note: `jest-setup.ts` re-mocks RN's ScrollView so a `refreshControl`
  element never lands in host props (it broke snapshots + `JSON.stringify`).
