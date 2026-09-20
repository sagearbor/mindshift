# iOS readiness — `apps/mobile` (2026-09-20)

Goal: make the phone app buildable for iOS and **prove it compiles** with an
EAS cloud iOS **simulator** build — no Apple Developer account, no Xcode, no
iPhone needed for this step (the owner has none of the three on this Mac; the
Apple Developer Program enrollment is happening tomorrow). This doc is the
file-verified proof plus the owner's path from here to TestFlight/App Store.

Branch: `worktree-agent-ab1b7f64790fea9c4` (isolated worktree off
`fix/livecoach-followups`, merged with `eval/real-conversation-audit` per the
task's first step). No push; no `eas submit`; no Android build touched; no
`versionCode`/`android.version*` edited; no `eas credentials` run (nothing
that prompts for an Apple login).

## Verdict

**The app compiles for iOS and the EAS cloud simulator build succeeded on the
first attempt.** No compile-blocking native module gaps were found — the
codebase already wraps every platform-sensitive native call (`Platform.OS`
checks or a `tryRequire`-style optional load), which is exactly the pattern
this task asked for and it turned out to already be in place almost
everywhere. Three feature gaps are real but non-blocking (documented below,
each with a `Platform`/config note rather than a crash).

## What was audited

### `npx expo-doctor` — 16/21 checks passed, 5 failed (all pre-existing, none iOS-specific, none block a build)
1. **Schema: `splash` is an unrecognized top-level key.** Pre-existing (predates
   this work; Android already builds/ships with it). Expo SDK 57 moved splash
   screen config to the `expo-splash-screen` plugin; the top-level `splash` key
   still works (Android's builds prove it) but doctor now flags it as
   deprecated schema. Left as-is — out of scope for "make iOS buildable" and
   fixing it means adding a new plugin + testing both platforms' splash
   screens, which is a separate piece of work.
2. **Duplicate `react`/`react-native` across the npm workspace.** A stray
   `react@18.3.1`/`react-native@0.76.6` is hoisted to the workspace ROOT
   `node_modules` (a transitive peer from somewhere in the monorepo) while
   `apps/mobile` itself runs `react@19.2.3`/`react-native@0.86.0`. This is the
   exact hazard `plugins/withOrtGradle9.js` already works around for
   onnxruntime-react-native's Android JSI shim (fix #6 in that file's
   comments) — the iOS Podfile fix in the same plugin (#3) exists because of
   the same workspace-hoisting shape. Nothing new; not fixed here because
   de-duplicating a monorepo dependency tree is unrelated to iOS readiness
   and risks the Android build that already ships.
3. **Hermes V1 memory regression advisory** for `expo@57.0.2`
   (`250829098.0.14`; fix lands `.0.16`+). An SDK-wide advisory, not
   iOS-specific; upgrading `expo`/`react-native` patch versions is a
   standalone maintenance task with its own regression risk on Android.
4. **React Native Directory metadata gaps**: `react-native-webrtc` shows
   "untested on New Architecture"; `expo-ai-kit` and `onnxruntime-react-native`
   have no directory metadata at all (both are the on-device/AI-kit packages
   this project depends on directly for the fast loop — expected for
   lesser-known packages, not a signal of an iOS-specific problem).
5. **22 packages behind their SDK-57-expected patch/minor versions** (routine
   drift, e.g. `expo-camera` 57.0.1 vs expected 57.0.5). Not iOS-specific.

None of the above stopped `expo prebuild`'s iOS path or the EAS build below —
they're doctor's opinions about the dependency tree, not a build failure.

### `npx tsc --noEmit` — **clean, zero errors.**

### `npx jest` — **2138 total: 2133 passed, 5 skipped, 0 failed.**
One test (`__tests__/activationGate.test.ts`, the "activation is LIVE" suite)
calls the RAVDESS corpus loader unconditionally (unlike the file's other
`describe` block, which correctly gates on `fs.existsSync`) and failed with
`ENOENT` the first run — **not an iOS regression**: this worktree is isolated
from the main repo's gitignored `tmp/`, which is where `tmp/ravdess/audio`
lives (a local, CC-BY-NC-SA corpus download). Symlinked
`tmp/ravdess -> <main repo>/tmp/ravdess` (same pattern the task already asked
for with `node_modules`) and the suite passed. **No code change** — this is a
worktree-isolation artifact, not a product bug, though the missing
`haveCorpus` gate on that one `describe` block is a small pre-existing test
hygiene gap worth a follow-up.

### Native module audit (every entry in `package.json`)

| Package | iOS native support | Notes |
|---|---|---|
| `expo`, `expo-application`, `expo-auth-session`, `expo-battery`, `expo-constants`, `expo-crypto`, `expo-document-picker`, `expo-file-system`, `expo-haptics`, `expo-secure-store`, `expo-speech`, `expo-updates`, `expo-video`, `expo-web-browser` | ✅ full | Standard Expo SDK modules, iOS-first citizens. |
| `expo-audio` | ✅ full | Drives the recording/playback/call audio-session helpers in `src/utils/audioMode.ts` — already written with iOS in mind (`playsInSilentMode`, comments explicitly describing iOS session behavior vs Android's). |
| `expo-camera`, `expo-media-library` | ✅ full | Config-plugin permission strings already present; iOS `NSCameraUsageDescription` added this session (see below). |
| `expo-speech-recognition` | ✅ full (Apple `Speech` framework) | `expo-speech-recognition/ios` has native code; `androidSpeechServicePackages` plugin option is Android-only and harmless on iOS. |
| `expo-build-properties` | ✅ (build-time only) | Already sets `ios.useFrameworks: "static"` — required for several of the pods below to link cleanly. |
| `expo-ai-kit` | ✅ (Apple Intelligence / Foundation Models) | `src/live/localLlm.ts` + `src/live/defaultDeps.ts:230` already branch `Platform.OS === "ios" ? "apple-fm" : "mlkit"` — the iOS on-device LLM path was coded before this session; never exercised until now because iOS never built. |
| `onnxruntime-react-native` | ✅ (has `ios/` + podspec) | Hoisted to the **workspace root** `node_modules` (see doctor finding #2), which breaks the package's own generated Podfile line (`:path => '../node_modules/...'`, relative to `apps/mobile`, doesn't exist). `plugins/withOrtGradle9.js` fix (3) already rewrites that Podfile line to the real resolved path — written for this exact problem, and the EAS build below (which runs `pod install`) is the first real proof it works. |
| `react-native-webrtc` + `@config-plugins/react-native-webrtc` | ✅ (has podspec, config plugin) | In-app call transport; iOS permission strings already present in `app.json`. Flagged by expo-doctor as "untested on New Architecture" — a maintenance note, not blocking. |
| `@react-native-google-signin/google-signin` | ✅ (has podspec) | Its config plugin only validates that `iosUrlScheme` starts with `com.googleusercontent.apps.` — the current placeholder (`...REPLACE_WITH_IOS_CLIENT_ID`) satisfies that and does not fail prebuild. **Functional gap, not a compile gap** — see "Known iOS feature gaps" below. |
| `@react-native-community/slider`, `react-native-svg`, `react-native-safe-area-context` | ✅ full | Plain cross-platform RN native modules. |
| `firebase` (JS SDK, not `@react-native-firebase/*`) | ✅ (pure JS) | `src/auth/firebase.ts` uses the Firebase **Web** SDK with `expo-secure-store`-backed persistence on native — no `GoogleService-Info.plist` is required for Auth to work, which removes what would otherwise be the biggest iOS setup gap. |
| `onnxruntime-web`, `react-native-web`, `zustand` | N/A (web/cross-platform JS) | No native iOS surface. |

**Android-only surfaces already guarded, none of which needed new code:**
- `plugins/withOrtGradle9.js`'s `withMainApplication`/`withGradleProperties`/
  `withProjectBuildGradle` mods are Android-platform config-plugin hooks by
  construction — Expo simply never invokes them during an iOS prebuild.
- The nudge-vocabulary haptic patterns (`src/live/defaultDeps.ts`'s
  `expoHaptics.nudge`) already gate `Vibration.vibrate(pattern)` behind
  `Platform.OS === "android"` — iOS already falls back to `expo-haptics`'
  single impact, which is what shipped before the vocabulary existed.
- `src/nav/useAndroidBackHandler.ts` is already a documented no-op off
  Android ("No-op on iOS/web — there's no hardware back event there").
- The **watch bridge** (`WatchSetupScreen.tsx`, `useAudioStream.ts`'s
  `watchConnected` state) has **no native module at all** — pairing is pure
  server-mediated REST/WebSocket (`claimWatchPairing`, `disconnectWatch`,
  `watch_connected` frames). It compiles identically on every platform; see
  "Known iOS feature gaps" for the one UX loose end it leaves on iOS.
- All native-module access that genuinely can throw at import time
  (`onnxruntime-react-native`, `expo-ai-kit`) already goes through the
  `tryRequire()` helper in `src/live/defaultDeps.ts` (comment: "Native
  packages are resolved lazily... A missing package is just another rung
  down the degradation ladder"). This pattern predates this session and is
  exactly what the task asked to verify — it was already correct.

**No code changes were needed to make the iOS bundle compile.** The only
edits were `app.json`/`eas.json` configuration (below).

## Config changes made

### `apps/mobile/app.json`
- `ios.supportsTablet: false` (added).
- `ios.infoPlist.NSCameraUsageDescription` (added) — the app already asks for
  camera access (`expo-camera`, `AvatarCaptureScreen.tsx`, `RecordScreen.tsx`)
  and the top-level `ios.infoPlist` block didn't carry the key explicitly
  (the `expo-camera` plugin would inject it too, but declaring it directly
  keeps the two mic/speech/camera strings together and future-proofs against
  the plugin ever changing its default injection behavior).
- `ios.infoPlist.UIBackgroundModes: ["audio"]` (added) — matches exactly what
  `src/recorder/defaultDeps.ts`'s own comment already assumed
  ("iOS uses UIBackgroundModes audio and needs no notification permission")
  when it computed `recordingSessionPlan()`'s `backgroundCapable` flag for
  `os !== "android"`. Without this key that assumption is false and a
  backgrounded Live Coach / recording session would be silently killed by
  iOS; it is now true.
- `ios.bundleIdentifier` (`com.sagearbor.mindshift.app`, mirrors Android's
  package name) and the microphone/speech-recognition usage strings were
  **already present** from earlier work — confirmed, not re-added.
- `runtimeVersion.policy: "appVersion"` — confirmed this is a top-level Expo
  updates setting, not an Android-only one; it works identically for the iOS
  channel (`preview`'s update group already carries an `ios` entry from the
  2026-08-24 OTA publish — see the handoff doc §8).

### `apps/mobile/eas.json`
- New `ios-simulator` build profile: `extends: "preview"` (reuses the same
  internal distribution + `preview` channel + Cloud Run API URL as the
  Android preview APK) with `ios.simulator: true`. `eas config --profile
  ios-simulator --platform ios` was used to confirm this resolves to
  `distribution: internal`, `channel: preview`, `ios.simulator: true` before
  spending build minutes on it.
- `production.ios.autoIncrement: "buildNumber"` (added) — the profile's
  existing top-level `autoIncrement: true` already covers Android's
  `versionCode`; making the iOS side explicit means a future
  `eas build -p ios --profile production` bumps `ios.buildNumber` the same
  way the Android side bumps `versionCode`, without silently doing nothing
  for iOS. Confirmed via `eas config --profile production --platform ios`
  (`"autoIncrement": "buildNumber"` resolved correctly).
- `submit.production.ios` (`appleId`/`ascAppId`/`appleTeamId` placeholders)
  was **already present** from earlier work — left untouched; filling these
  in is an owner step (below) and `eas submit` was never run this session.

No `android.versionCode`, no `ios.buildNumber` on the `production` profile,
and no `eas credentials`/`eas submit` were touched.

## EAS iOS **simulator** build — the proof

```
npx eas build -p ios --profile ios-simulator --non-interactive --no-wait
```

| | |
|---|---|
| Build ID | `e6129bc7-ccd6-4b9b-94ed-1e0fadfc2983` |
| Status | **finished** (first attempt — no retries needed) |
| Platform / profile | iOS / `ios-simulator` (`isForIosSimulator: true`) |
| Distribution / channel | internal / `preview` |
| SDK / runtime / app version | 57.0.0 / 1.18.0 / 1.18.0 (buildNumber 1, unchanged) |
| Started → finished | 2026-09-20 01:28:36 → 01:34:15 (**≈5 min 39 s**) |
| Logs | https://expo.dev/accounts/sagearbor/projects/mindshift/builds/e6129bc7-ccd6-4b9b-94ed-1e0fadfc2983 |
| Application Archive (simulator `.app` in a `.tar.gz`) | https://expo.dev/artifacts/eas/2xMJohVehTgBAq_Nhie4PrE8Ej8SwRnB4Nr6apvpsGE.tar.gz |
| Fingerprint | `a07c3254cb162e1bcba7a947104a915bd303c986` |

Polled via `npx eas build:view <id> --json` every ~60 s, bounded loop, all in
the foreground (per this session's constraints — no background jobs, no
`eas build:wait`/`--wait`). The artifact is a real compiled `.app` (needs no
code signing for a simulator target, which is exactly why this build was
possible with zero Apple Developer account, zero Xcode, zero iPhone on this
Mac) — running it needs a Mac with Xcode's Simulator.app, which this Mac does
not have; the finished-build status plus the downloadable artifact is the
available proof, matching the task's own framing ("prove it compiles").

## Known iOS feature gaps (documented, not crashes)

1. **Google Sign-In on iOS is configured but not functional yet.**
   `@react-native-google-signin/google-signin`'s config plugin only validates
   that `iosUrlScheme` starts with `com.googleusercontent.apps.` — the
   current value ends `REPLACE_WITH_IOS_CLIENT_ID`, a placeholder, so the
   iOS build succeeds but a real device's OAuth redirect back into the app
   would never be caught by that URL scheme (`GoogleSignInButton.native.tsx`
   is a `.native.tsx` file, so it renders on iOS too, not just Android as its
   header comment implies). `firebaseConfig.ts` already defines
   `googleOAuth.iosClientId` from `EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID`, but
   nothing reads it yet (`GoogleSignin.configure` only passes `webClientId`,
   which is correct per the library's docs — the *scheme*, not the configure
   call, is what needs the real iOS client). **Owner step:** create an iOS
   OAuth client in Google Cloud Console (same project as the existing web
   client, `arborfam-hub`), then replace the `iosUrlScheme` plugin option in
   `app.json` with the real reversed client ID. Email/password sign-in is
   unaffected.
2. **"Set up your watch" (`AdvancedScreen.tsx` → `WatchSetupScreen.tsx`) shows
   on iOS and links to the Android Play Store listing for the Wear OS app
   (`com.sagearbor.gauge.wear`)** — tapping "Install the watch app" opens
   Safari to a Play Store URL that does nothing useful on an iPhone. This is
   a **dead end, not a crash**: no native module is involved (pairing is a
   pure REST/WebSocket flow), so nothing throws. Left un-guarded deliberately
   this session — `App.test.tsx` and `AdvancedScreen.test.tsx` already assert
   `advanced-watch-setup` is visible under jest-expo's **default test
   platform, which is iOS** (`Platform.OS === "android"` is set explicitly,
   per-test, only where Android-specific behavior is under test — see
   `AdvancedScreen.test.tsx:751`). Hiding the row behind
   `Platform.OS === "android"` would have broken roughly ten existing,
   currently-passing assertions across both files for a cosmetic gap with no
   functional risk. **Recommended follow-up** (separate PR, with the test
   updates budgeted in): gate that row on Android, or reword its copy on iOS
   to something honest like "Watch support is Android-only right now."
3. **No Apple Watch companion exists or is planned in this repo.** `apps/watch`
   is a Gradle/Kotlin Wear OS project (a ported "Gauge" app,
   `com.sagearbor.gauge.wear`), entirely separate from Expo/React Native. An
   Apple Watch app would need a **brand-new watchOS target** — SwiftUI +
   WatchConnectivity, built and signed via Xcode, added to a *new* Xcode
   workspace/project (Expo/EAS has no watchOS build support at all, managed
   or bare). That is a from-scratch native project, not a config change to
   this one, and stays explicitly **out of scope** here.

## Owner checklist — from here to TestFlight / App Store

1. **Apple Developer Program enrollment** (tomorrow, per plan).
   [developer.apple.com](https://developer.apple.com/programs/enroll/) — pick
   **Individual** (fastest: personal Apple ID, ~$99/yr, usually approved
   within minutes to ~48 h) unless MindShift needs to publish under a company
   name/D-U-N-S-verified organization, which adds paperwork and a longer
   review. Individual is almost certainly the right choice for a
   single-owner app like this one.
2. **Let EAS create the signing credentials** (first time only, needs an
   interactive Apple login + 2FA — do this yourself, not from an agent
   session): from `apps/mobile`,
   `npx eas credentials -p ios` → pick the `production` profile → "Log in to
   your Apple Developer account" → EAS creates the distribution certificate
   + provisioning profile and stores them on EAS's servers (nothing to keep
   track of locally).
3. **Real device build:** `npx eas build -p ios --profile production
   --non-interactive`. `autoIncrement: "buildNumber"` (added this session)
   bumps `ios.buildNumber` in `app.json` automatically — commit that bump
   afterwards, same convention as the Android `versionCode` bumps.
4. **TestFlight** (recommended first stop — Mom installs the free TestFlight
   app and taps a link):
   - Fill in `eas.json`'s `submit.production.ios` placeholders: `appleId`
     (the Apple ID email used for App Store Connect), `ascAppId` (create the
     app record at [appstoreconnect.apple.com](https://appstoreconnect.apple.com)
     first, or let an interactive `eas submit` create it and report the id
     back), `appleTeamId` (Apple Developer account → Membership).
   - `npx eas submit -p ios --profile production --non-interactive`.
   - App Store Connect → your app → TestFlight → add Mom as an internal or
     external tester (external testing needs a brief Apple review, ~24 h;
     internal is instant but capped to accounts on your Apple Developer team).
5. **App Store Connect listing** (only needed before a public App Store
   release, not for TestFlight): app name/subtitle, screenshots (iPhone
   sizes), description, keywords, support URL, **privacy policy URL** (the
   same one already live for Play — see `docs/legal/` / the GitHub Pages
   privacy page mentioned in recent commits), age rating questionnaire.
6. **App Privacy ("nutrition label") in App Store Connect** — mirror the
   already-answered Play Data Safety declarations in
   `docs/play/play-answers-mindshift.yaml` (re-verify it first — its own
   header flags it as stale versus the current versionCode): data collected
   —account email, microphone audio (processed on-device for coaching;
   recordings the user explicitly saves upload to the backend), optionally
   recorded video (camera), no advertising, no third-party data sharing
   beyond the stated backend. Translate each Play "data type" answer to its
   nearest App Store Connect privacy category; the *facts* are already
   decided, only the form is different.
7. **Apple Watch companion app** — not attempted, not started. Would require:
   a new Xcode project with a watchOS app target (SwiftUI), a WatchConnectivity
   bridge to the phone app (or its own network stack, mirroring what
   `apps/watch`'s Wear OS app already does against the same backend API), its
   own App Store Connect listing (Watch apps embedded in an iOS app still need
   their own binary built via Xcode — EAS cannot build this), and a full
   round of the same privacy/permissions work done here for the phone app.
   Budget this as a separate project, not a follow-up PR.

## Files touched this session

- `apps/mobile/app.json` — iOS `supportsTablet`, `NSCameraUsageDescription`,
  `UIBackgroundModes`.
- `apps/mobile/eas.json` — `ios-simulator` build profile,
  `production.ios.autoIncrement`.
- `docs/plans/2026-09-20-ios-readiness.md` — this file.
- No `apps/mobile/src/**` changes were needed (see "What was audited").
