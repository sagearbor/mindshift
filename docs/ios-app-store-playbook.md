# iOS / Apple App Store: a working playbook

A portable, repo-agnostic checklist for getting an app onto iOS, TestFlight and
the App Store. Two lanes are covered because both were actually used:

- **Flutter + local Xcode** — brought up on FitRival, Sept 2026.
- **Expo / React Native + EAS cloud builds** — brought up on MindShift,
  2026-09-21.

**Drop this file into any repo** (e.g. `docs/ios-app-store-playbook.md`) and
point a Claude Code session at it.

## Trust levels — read this first

| Section | Status |
|---|---|
| 1–8 (project config → simulator build) | **Verified** on Flutter. App built and launched on the iOS 26.3 simulator. |
| 9 (signing) | **Verified** on Expo/EAS, 2026-09-21, and it overturns what section 2 used to say — see below. |
| 10 (upload / TestFlight) | **Not executed.** Commands are a starting point. |
| 11–12 (listing, review notes) | **Documented, not executed.** |
| 13 (Guideline 4.8) | **Verified as a real trap** — caught a rejection-bound build on MindShift and was implemented. |

A session following this should say which half it's in rather than implying
everything is proven.

## What changed on 2026-09-21

The earlier version of this document said the one step an agent cannot do is
the Xcode sign-in. **That is no longer true, and it was the single biggest
constraint in the file.** With an App Store Connect API key you never sign into
Xcode at all: the distribution certificate and provisioning profile can be
minted directly from Apple's API, and the build can happen in the cloud. A
signed production IPA was produced start to finish with no interactive prompt.

The genuinely irreducible manual steps turned out to be two, and neither is the
Xcode login:

1. **Generating the API key** (App Store Connect → Users and Access →
   Integrations). One-time per Apple account.
2. **Creating the app record.** `POST /v1/apps` answers
   `FORBIDDEN_ERROR … The resource 'apps' does not allow 'CREATE'`, and the
   internal `/iris/v1/apps` path 404s under API-key auth. One-time per app.

Everything between those two is scriptable.

## Account facts (Sage's — reuse across apps)

- **Team ID:** `553Y9KWR84` (team name "Sage Arbor", Individual/Sole Proprietor)
- **Membership:** renews annually; activated 2026-09-21
- Bundle ID convention: `com.sagearbor.<app>.app`

Team IDs are not secret (they ship inside every app). The Apple ID password
never goes in a file, a repo, or a chat.

### Where credentials live (machine-level, shared by every repo)

Credentials are per-**account**, not per-repo, so they live in the home
directory and each repo symlinks or reads them. Set up once, reuse everywhere.

| What | Path | Used by |
|---|---|---|
| ASC API key | `~/.appstoreconnect/private_keys/AuthKey_<KEYID>.p8` (dir 700) | EAS, fastlane and altool all auto-discover this path |
| Key ids + team | `~/.config/asc/asc.env` (600) | `source` it before any iOS build or submit |
| Signing artefacts — certificate | `~/.config/ios-credentials/_team/` (700) | shared by **every** app on the team |
| Signing artefacts — profile | `~/.config/ios-credentials/<bundle id>/` (700) | `credentials.json` points at both |
| Play service accounts | `~/.config/play/<app>-sa.json` (600) | `scripts/play_publish.py`, `eas.json` |

`asc.env` exports `EXPO_ASC_KEY_ID`, `EXPO_ASC_ISSUER_ID`,
`EXPO_ASC_API_KEY_PATH`, `APPLE_TEAM_ID` **and `EXPO_APPLE_TEAM_ID`**.

### ⚠️ eas-cli wants the team id under its own name

Set `EXPO_APPLE_TEAM_ID` as well as `APPLE_TEAM_ID`. With the three
`EXPO_ASC_*` variables all set but the team id only under `APPLE_TEAM_ID`,
`eas submit` reports

> `App Store Connect credentials are incomplete, skipping TestFlight setup`

and then offers to generate a brand-new API key — which it does through an
interactive Apple ID login, i.e. straight back into the wall this whole
approach exists to avoid. The check is a four-way `&&` on
`EXPO_ASC_API_KEY_PATH`, `EXPO_ASC_KEY_ID`, `EXPO_ASC_ISSUER_ID` and
`EXPO_APPLE_TEAM_ID` (`build/submit/ios/ensureTestFlightSetup.js`), and it
names none of the four in the message. Verified on eas-cli 24.7.0.

Answer **no** to "Generate a new App Store Connect API Key?" — a second key
solves nothing and the generation path needs the password.

### ⚠️ Getting the .p8 onto the machine

The key is downloadable exactly once. Do **not** route it through a tool that
returns file contents inline (a Drive/Dropbox read tool), because the private
key then passes through the session transcript. Download it in the browser
straight to `~/Downloads` and move it with a shell command — the content never
enters the agent's context. In Claude Code, the Drive-download path is blocked
by the permission classifier for exactly this reason.

---

## 1. Prerequisites

```bash
xcodebuild -version      # Xcode 26.3 used here — NOT needed for the EAS lane
flutter --version        # Flutter lane only; 3.22.2 used here
pod --version            # Flutter lane only; CocoaPods 1.17.0
```

Apple Developer Program membership ($99/yr) is required for **anything with a
device**: signing, TestFlight, the App Store. It is **not** required for
sections 1–8 — do all of that before paying or while enrolment is pending.

The EAS lane needs no local Xcode at all. Installing it is still worth it for
the simulator, which is the only way to see the app without a device.

## 2. What an agent cannot do

Superseded — see "What changed" above. The list is now: generate the API key,
create the app record, and enable auth providers in the Firebase console.
Claude cannot enter passwords, so anything gated on a password is yours.

Do **not** design a flow around the Xcode login. It is avoidable.

## 3. Bundle ID and deployment target

**Flutter:** set `PRODUCT_BUNDLE_IDENTIFIER` in
`ios/Runner.xcodeproj/project.pbxproj` (three places — Debug, Release, Profile)
or via Xcode's General tab.

**Expo:** `expo.ios.bundleIdentifier` in `app.json`. Never edit the Xcode
project; `prebuild` regenerates it.

Deployment target must satisfy your heaviest plugin. Known floors:

| Plugin | Minimum iOS |
|---|---|
| `cloud_firestore` 6.x / `firebase_*` 13.x | **15.0** |
| `sign_in_with_apple` 7.x | 13.0 |
| Flutter default (unchanged) | 12.0 |

Flutter needs it in **both** places or the build fails confusingly:

```ruby
# ios/Podfile — line 2, uncomment and raise
platform :ios, '15.0'
```
```
# ios/Runner.xcodeproj/project.pbxproj — all three configurations
IPHONEOS_DEPLOYMENT_TARGET = 15.0;
```

## 4. Firebase on iOS (Flutter; skip if not using Firebase)

```bash
firebase apps:create IOS "<App Name> (iOS)" \
  --bundle-id com.example.myapp --project <firebase-project>
firebase apps:sdkconfig IOS <APP_ID> --project <firebase-project> \
  > ios/Runner/GoogleService-Info.plist
```

Then add the iOS block to `lib/firebase_options.dart`: a `case
TargetPlatform.iOS: return ios;` in the switch, plus a `static const
FirebaseOptions ios` whose values come from the plist (`API_KEY`,
`GOOGLE_APP_ID`, `GCM_SENDER_ID`, `PROJECT_ID`, `STORAGE_BUCKET`,
`CLIENT_ID`, `BUNDLE_ID`).

Expo with the Firebase **JS** SDK needs none of this — the config is plain
public client identifiers in a TS file, same values on every platform.

### ⚠️ Gotcha that will bite you (Flutter)

Writing the plist to disk is **not enough** — it must be a member of the Runner
target, or the app builds fine and then logs at runtime:

> `Could not locate configuration file: 'GoogleService-Info.plist'`

Either drag it into the Runner group in Xcode (tick *Copy items* + *Runner
target*), or patch `project.pbxproj` directly — three entries with one shared
pair of hex IDs:

```
/* 1. PBXBuildFile section */
<ID_B> /* GoogleService-Info.plist in Resources */ = {isa = PBXBuildFile; fileRef = <ID_A>; };

/* 2. PBXFileReference section */
<ID_A> /* GoogleService-Info.plist */ = {isa = PBXFileReference; lastKnownFileType = text.plist.xml; path = "GoogleService-Info.plist"; sourceTree = "<group>"; };

/* 3. Runner PBXGroup children AND the Resources PBXResourcesBuildPhase files */
<ID_A> /* GoogleService-Info.plist */,     # in the group
<ID_B> /* GoogleService-Info.plist in Resources */,   # in Resources
```

Verify by rebuilding and confirming the warning is gone from the run log.

## 5. Info.plist

Purpose strings — **required**, and App Review rejects vague ones. Say what you
read and why the user benefits. Flutter edits `ios/Runner/Info.plist`; Expo
puts the same keys under `expo.ios.infoPlist` in `app.json`.

```xml
<key>NSMicrophoneUsageDescription</key>
<string>MyApp uses your microphone to transcribe the conversation and coach you in real time.</string>
<key>NSHealthShareUsageDescription</key>
<string>MyApp reads your weight from Apple Health so you don't have to retype it.</string>
```

Google Sign-In needs the reversed client ID as a URL scheme (copy
`REVERSED_CLIENT_ID` verbatim from `GoogleService-Info.plist`):

```xml
<key>CFBundleURLTypes</key>
<array>
  <dict>
    <key>CFBundleTypeRole</key><string>Editor</string>
    <key>CFBundleURLSchemes</key>
    <array><string>com.googleusercontent.apps.NNNNNN-xxxxxxxx</string></array>
  </dict>
</array>
```

Saves an export-compliance question on every single upload:

```xml
<key>ITSAppUsesNonExemptEncryption</key><false/>
```

Background audio, if the app records or plays with the screen off:

```xml
<key>UIBackgroundModes</key><array><string>audio</string></array>
```

## 6. Entitlements

**Flutter:** create `ios/Runner/Runner.entitlements` and wire it into all three
build configurations in `project.pbxproj`:

```
CODE_SIGN_ENTITLEMENTS = Runner/Runner.entitlements;
DEVELOPMENT_TEAM = 553Y9KWR84;
```

**Expo:** use the first-class config fields instead of writing entitlements by
hand — e.g. `expo.ios.usesAppleSignIn: true` adds
`com.apple.developer.applesignin`.

### ⚠️ An entitlement is only half of it

The **App ID** must also carry the matching capability, and adding one
**invalidates every provisioning profile bound to that App ID**. Apple then
refuses a second profile with the same name, so a naive re-run fails on a name
conflict. Delete the invalid profile and reissue. Capability changes are
scriptable (section 9).

## 7. Sign in with Apple — see section 13

Moved, because it is a rejection cause rather than a config detail.

## 8. Build and run on the simulator

### ⚠️ Gotcha: Flutter skips `pod install`

`flutter build` only runs CocoaPods when it thinks the Podfile changed. After
adding a plugin you'll get `Module 'foo' not found`. Force it:

```bash
cd ios && pod install && cd ..
```

Then:

```bash
flutter build ios --simulator --no-codesign
xcrun simctl boot "iPhone 17 Pro"; open -a Simulator
xcrun simctl install booted build/ios/iphonesimulator/Runner.app
xcrun simctl launch booted com.example.myapp
xcrun simctl io booted screenshot /tmp/shot.png     # visual verification
```

Expo equivalent: an `ios-simulator` EAS profile with `ios.simulator: true`,
then install the resulting `.app` with the same `simctl` commands.

**Commit the CocoaPods integration artifacts** (Flutter) — a fresh clone fails
without them:

```
ios/Flutter/Debug.xcconfig        # gains the Pods-Runner include line
ios/Flutter/Release.xcconfig
ios/Runner.xcworkspace/contents.xcworkspacedata   # gains the Pods project ref
ios/Podfile.lock
```

Note: `Pods/` itself stays gitignored.

### What the simulator cannot do

HealthKit, push notifications, the camera, and real Sign in with Apple do not
work in the simulator. Those need a device via TestFlight.

**No iPhone at all?** An Apple Silicon Mac can run iPhone builds two ways:
the simulator (needs Xcode), or TestFlight for macOS, which installs iPhone
builds as long as the app record keeps "Make this app available on Mac"
enabled. Neither covers watch pairing, and background-audio behaviour differs.

### ⚠️ Gotcha: `osascript` hangs

Driving the Simulator with `osascript … System Events` blocks forever waiting
on an invisible Accessibility permission prompt. Use `xcrun simctl` for
everything, or `cliclick` (`brew install cliclick`) for taps.

---

## 9. Signing — fully scriptable ✅ *verified 2026-09-21*

### The blocker this replaces

`eas build -p ios --non-interactive` refuses to create a certificate:

> `Distribution Certificate is not validated for non-interactive builds.`
> `Credentials are not set up. Run this command again in interactive mode.`

and `eas credentials` has **no** non-interactive mode — its only flag is
`--platform`. So the documented EAS path cannot be driven unattended at all.
The fix is to stop asking EAS for credentials.

### What to do instead

Mint the artefacts against the App Store Connect API and hand the builder a
local `credentials.json`. In this repo that is one idempotent script:

```bash
source ~/.config/asc/asc.env
python3 scripts/ios_credentials_bootstrap.py \
    --bundle-id com.sagearbor.mindshift.app --name MindShift \
    --project-dir apps/mobile
```

It performs, in order:

1. `GET/POST /v1/bundleIds` — register the identifier (`seedId` = team ID).
2. `openssl genrsa` + `openssl req` — a local key and CSR.
3. `POST /v1/certificates` with `certificateType: IOS_DISTRIBUTION` and the CSR;
   the response carries the certificate as base64 DER.
4. `openssl pkcs12 -export -legacy` — build the `.p12`.
5. `POST /v1/profiles` with `profileType: IOS_APP_STORE`, related to the bundle
   ID and the certificate. No devices needed for an App Store profile.
6. Write `credentials.json`.

Then one switch in `eas.json`:

```json
"production": { "ios": { "credentialsSource": "local" } }
```

`credentials.json` holds the `.p12` password in clear text — gitignore it.

### ⚠️ Run it with `/usr/bin/python3`

PyJWT and requests are installed in the **system** interpreter on this machine,
not in Homebrew's. With `/opt/homebrew/bin` ahead of `/usr/bin` on PATH the
script dies on `import jwt`, which reads as a bug in the script rather than the
wrong python. It now says which it is, but the shorter answer is to invoke
`/usr/bin/python3` explicitly in any scripted use.

### ⚠️ `-legacy` is not optional

OpenSSL 3 no longer writes the PKCS#12 algorithms Apple's tooling expects.
Without `-legacy` the `.p12` is silently the wrong flavour.

### ⚠️ One certificate for the whole team — not one per app

Apple caps distribution certificates at **two per team**, and a single
certificate signs **every** app on the team. The first version of
`ios_credentials_bootstrap.py` kept the certificate under the per-bundle-id
directory, so its "reuse what we already minted" check missed on every new app
and minted a fresh one: app #2 consumed the last slot and app #3 could not sign
at all. Fixed 2026-09-21 — the layout is now

```
~/.config/ios-credentials/_team/          dist.key, dist.cer, dist.p12,
                                          p12_password.txt, cert_id.txt
~/.config/ios-credentials/<bundle id>/    AppStore.mobileprovision
```

A certificate left over from the old layout is copied into `_team/` on the next
run (non-destructive — the old files stay put, unused). The script also refuses
to mint when the account already has two, listing them instead of letting Apple
answer with an opaque 409.

Certificates last one year; re-run to roll them.

### More than one Xcode target (watch app, share extension, App Clip)

Every target has its own bundle identifier, so every target needs its own
provisioning profile — but they all share the one team certificate. The script
takes `--watch-target <XcodeTargetName>` (sugar for
`--extra-target .watchkitapp=<name>`), registers `<bundle id>.watchkitapp`,
mints a second `IOS_APP_STORE` profile, and switches `credentials.json` to
EAS's **multi-target** form:

```bash
/usr/bin/python3 scripts/ios_credentials_bootstrap.py \
    --bundle-id com.sagearbor.mindshift.app --name MindShift \
    --project-dir apps/mobile \
    --xcode-target MindShift --watch-target MindShiftWatch
```

```json
{"ios": {"MindShift":      {"provisioningProfilePath": "…", "distributionCertificate": {…}},
         "MindShiftWatch": {"provisioningProfilePath": "…", "distributionCertificate": {…}}}}
```

The multi-target form **replaces** the single-target one — `ios.provisioning‑
ProfilePath` at the top level stops being read — and its keys are **Xcode target
names**, not bundle ids and not app names. Read them out of the project rather
than guessing:

```bash
grep productName apps/mobile/ios/*.xcodeproj/project.pbxproj
```

### ⚠️ `IOS_APP_STORE` is the profile type for a watch app too

There is no watch-specific `profileType`. Apple's portal text says an iOS
profile covers "iOS and watchOS apps and App Clips", and the
`POST /v1/profiles` for `com.sagearbor.mindshift.app.watchkitapp` with
`profileType: IOS_APP_STORE` was accepted on 2026-09-22. Profile *names* are
unique per team, so name extra profiles after the target
("MindShiftWatch App Store"), not the app.

### ⚠️ For an Expo CNG project, eas-cli finds targets in app.json, not the pbxproj

`ios/` does not exist when `eas build` starts, so eas-cli enumerates iOS targets
from `extra.eas.build.experimental.ios.appExtensions` in the evaluated Expo
config. A target that is missing there gets no credentials however correct
`credentials.json` is. Config plugins that generate targets
(`@bacons/apple-targets`, `expo-targets`) write that key themselves; confirm
with `npx expo config --type introspect`, and check the `targetName` there
matches the `credentials.json` key exactly.

### ⚠️ A watch app that does not match the phone's version is rejected

Apple compares the two bundles at upload: `ITMS-90473` for
`CFBundleShortVersionString`, `ITMS-90379` for `CFBundleVersion`.
`@bacons/apple-targets@5.0.0` hardcodes `MARKETING_VERSION = "1.0"` for every
target it generates, and a `watch` target has no real `Info.plist`
(`GENERATE_INFOPLIST_FILE = YES`), so the literal goes straight into the
shipped plist. Fix it with a follow-on config plugin — see
`apps/mobile/plugins/withWatchTargetVersion.js`, which must be listed **before**
`@bacons/apple-targets` because config-plugin mods run in reverse registration
order.

Prove it locally instead of paying for a cloud build to find out:

```bash
npx expo prebuild -p ios --clean
grep MARKETING_VERSION ios/*.xcodeproj/project.pbxproj
xcodebuild -project ios/App.xcodeproj -target <WatchTarget> -sdk watchos \
  -destination 'generic/platform=watchOS' CODE_SIGNING_ALLOWED=NO build
/usr/libexec/PlistBuddy -c Print ios/build/Release-watchos/<WatchTarget>.app/Info.plist
```

The last command shows the *built* plist, which is the thing Apple actually
validates. No watchOS simulator runtime is needed for any of this — only the
watchOS SDK, which ships with Xcode.

### Adding a capability (Sign in with Apple, push, HealthKit)

```
POST /v1/bundleIdCapabilities
{"data":{"type":"bundleIdCapabilities",
         "attributes":{"capabilityType":"APPLE_ID_AUTH",
                       "settings":[{"key":"APPLE_ID_AUTH_APP_CONSENT",
                                    "options":[{"key":"PRIMARY_APP_CONSENT"}]}]},
         "relationships":{"bundleId":{"data":{"type":"bundleIds","id":"<ID>"}}}}}
```

Omitting `settings` returns `409 ENTITY_ERROR — "Please select at least one
configuration for Sign In with Apple."` Afterwards every profile on that App ID
reads `profileState: INVALID`; delete and reissue (section 6).

### Measured result

Production IPA, Expo SDK 57, signed with the above: **5 min 19 s on EAS, first
try**, no prompt, no local Xcode.

## 10. Upload and TestFlight *(not yet executed)*

Two routes. Both authenticate with the API key, so neither needs a password.

```bash
# Expo / EAS
source ~/.config/asc/asc.env
npx eas-cli submit -p ios --latest

# Flutter, or any local .ipa (needs Xcode's command line tools)
flutter build ipa --export-method app-store
xcrun altool --upload-app -f build/ios/ipa/MyApp.ipa -t ios \
  --apiKey "$EXPO_ASC_KEY_ID" --apiIssuer "$EXPO_ASC_ISSUER_ID"
```

Put the `.p8` at `~/.appstoreconnect/private_keys/` (not `~/.config/appstore/`)
— that is the path `altool`, `fastlane` and EAS all search automatically.

Processing takes 5–30 min, then add testers. Internal testers (up to 100, same
team) need **no review**, so TestFlight is reachable before any of the listing
work in section 11. External testers need a short Beta App Review.

> **Note for agents:** in Claude Code, `eas submit` is blocked by the
> permission classifier as an outward-facing publish. Hand the command to the
> owner rather than routing around it.

## 11. App Store Connect listing *(not yet executed)*

- **Create the app record** — the one step with no API. Name, primary language,
  bundle ID, SKU. Do this before the first upload.
- **Screenshots**: 6.7" iPhone is the only strictly required size now; generate
  with `xcrun simctl io booted screenshot` on a matching simulator.
- **Privacy nutrition labels**: the same answers as Google Play's Data Safety
  form — keep one source doc and fill both from it.
- **Privacy policy URL**: required, must be live before submission.
- **Account deletion**: if the app has accounts, an in-app deletion path is
  mandatory (Guideline 5.1.1(v)). Document where it is in the review notes.

## 12. Review notes that prevent rejections

Fill the "Notes" field. Cheap insurance:

- A demo account if anything is behind a login. Apple *will* reject for
  "couldn't sign in". **A guest mode is better than a demo account** — it
  removes the credential from the process entirely, and it is a real feature
  rather than reviewer scaffolding.
- Where to find each sensitive capability — exact tap path.
- Why each permission is needed, in one line each.
- For HealthKit: state that data is read-only and never used for advertising.

Typical review: 1–3 days. A rejection is a conversation, not a verdict — reply
in Resolution Center.

## 13. Sign in with Apple — usually mandatory ✅ *verified as a real trap*

**Guideline 4.8:** offer any third-party sign-in (Google, Facebook) and you
**must** also offer Sign in with Apple. This caught a MindShift build that was
otherwise ready to submit.

**A disabled "coming soon" button does not satisfy it, and makes things worse** —
a visibly dead control advertising a future feature is its own rejection risk
under Guideline 2.1 App Completeness. Either implement it or remove the
third-party sign-in from the iOS build; a placeholder is the one option that
fails twice.

It does **not** block internal TestFlight, which has no review. It blocks App
Store submission.

### The nonce is the part that goes wrong

Generate a random raw nonce. Hand Apple its **SHA-256 hash**; hand Firebase the
**raw** value. Firebase hashes it again and compares against the hash Apple
embedded in the signed token. Swapping them — natural, since both are "the
nonce" — fails as `auth/invalid-credential`. Pin it with a test.

**Expo:**

```ts
const rawNonce = Crypto.randomUUID();
const hashedNonce = await Crypto.digestStringAsync(
  Crypto.CryptoDigestAlgorithm.SHA256, rawNonce);
const credential = await AppleAuthentication.signInAsync({
  requestedScopes: [/* EMAIL, FULL_NAME */], nonce: hashedNonce,
});
await signInWithCredential(auth, new OAuthProvider("apple.com").credential({
  idToken: credential.identityToken, rawNonce,
}));
```

**Flutter:** `sign_in_with_apple: ^7.0.1` + `crypto: ^3.0.3`. Name the helper
distinctly — `generateNonce` collides with `sign_in_with_apple`'s own export.

### Three things that are easy to miss

- Gate on `AppleAuthentication.isAvailableAsync()` (Expo) or
  `Platform.isIOS || Platform.isMacOS` (Flutter), and treat a dismissed sheet
  (`ERR_REQUEST_CANCELED` / `AuthorizationErrorCode.canceled`) as a normal
  cancel, not an error.
- **Enable Apple as a provider in the Firebase console.** The entitlement, the
  App ID capability and the code can all be correct and Firebase will still
  reject a valid Apple token with `auth/operation-not-allowed`. Surface that
  message honestly rather than a generic retry. This is an owner step — an
  agent cannot enable auth providers.
- The **Services ID and private key** fields on Firebase's Apple provider are
  for Apple's redirect-based OAuth flow, i.e. **web and Android**. A native
  iOS-only integration does not need them, which removes a whole credential
  from the setup. If you do offer Apple sign-in on web, you need both.
- Apple releases an identity token **once per authorization**, so a stashed
  Apple credential cannot be replayed later the way a Google one can. Don't
  build a pending-credential linking path for it.

---

## Quick reference — files touched

**Flutter**

```
ios/Podfile                              # platform :ios, 'NN.0'
ios/Podfile.lock                         # COMMIT
ios/Flutter/{Debug,Release}.xcconfig     # COMMIT (Pods include lines)
ios/Runner.xcworkspace/contents.xcworkspacedata   # COMMIT (Pods ref)
ios/Runner/Info.plist                    # purpose strings, URL schemes, encryption
ios/Runner/Runner.entitlements           # NEW — capabilities
ios/Runner/GoogleService-Info.plist      # NEW — and add to the Xcode target!
ios/Runner.xcodeproj/project.pbxproj     # target, team, entitlements, plist ref
lib/firebase_options.dart                # iOS block
pubspec.yaml                             # sign_in_with_apple, crypto
```

**Expo / EAS**

```
app.json                  # ios.bundleIdentifier, buildNumber, infoPlist,
                          # usesAppleSignIn, UIBackgroundModes
eas.json                  # production.ios.credentialsSource: "local"
credentials.json          # GITIGNORE — generated, holds the .p12 password
package.json              # expo-apple-authentication, expo-crypto
```

Nothing under `ios/` is edited by hand in the Expo lane; `prebuild`
regenerates it on every cloud build.

## Order of operations

1. Sections 3–8 — no Apple account needed. Get it building on the simulator.
2. Human enrols in the Apple Developer Program and generates an ASC API key.
3. Section 9 — signing, scripted. Section 13 if you offer any third-party login.
4. Human creates the app record; then section 10 uploads to TestFlight.
5. Sections 11–12, then submit.

Steps 1 and 2 are independent: do the code while enrolment processes. Step 4's
app record is the only thing between a green build and TestFlight, so create it
early rather than discovering it at upload time.
