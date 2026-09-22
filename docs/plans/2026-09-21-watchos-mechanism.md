# Adding a watchOS target to apps/mobile — mechanism, cost, risk

Date: 2026-09-21 · Branch: `ship/heat-judge-20260920` · Scope: feasibility only, no implementation.

---

## Recommendation (confidence: medium-high on mechanism, medium on effort)

**Use a config plugin that generates the watch target during prebuild — do not commit `ios/`, do not "eject".**
Run a 1-day go/no-go spike first, on `@bacons/apple-targets@5.0.0` as primary with
`expo-targets@1.9.2` as the named fallback if prebuild breaks on SDK 57.

Then extend `scripts/ios_credentials_bootstrap.py` to mint a **second** bundle id
(`com.sagearbor.mindshift.app.watchkitapp`) and a **second** `IOS_APP_STORE` provisioning
profile, and rewrite `credentials.json` in EAS's **multi-target** form keyed by Xcode target
name. That combination — local credentials + multi-target `credentials.json` — is the *only*
route with reported success for watch targets on EAS; EAS's managed credential service is
documented by its own users as unable to do it
([eas-cli#2578](https://github.com/expo/eas-cli/issues/2578)).

**The watch app itself will be pure Swift/SwiftUI with zero React Native.** Everything on the
wrist is a second codebase talking to the phone over WatchConnectivity. That, not the build
plumbing, is the cost.

**Top risk:** we cannot verify it. No Apple Watch, no iPhone. WCSession between paired
simulators is documented-flaky, and the `Embed Watch Content` phase is device-only. This
violates the repo's own standing rule ("every behavior proven by file replay + report + test
gate") in a way no amount of work on our side fixes.

**Showstopper:** none absolute. Two hard gates that must be cleared in the spike, in order:
ITMS-90473 (version mismatch, see §2.4) and the RN 0.86 framework-embed regression (§1.1).

---

## 0. Ground truth for this repo (verified, read from files)

| Fact | Where |
| --- | --- |
| Expo `57.0.2`, `@expo/prebuild-config` `57.0.4`, `@expo/config-plugins` `57.0.2`, RN `0.86.0`, React `19.2.3` | `node_modules/*/package.json` |
| Bundle id `com.sagearbor.mindshift.app`, version `1.18.0`, buildNumber `3` | `apps/mobile/app.json` |
| **Xcode target name is `MindShift`** (single target, `productName = MindShift`) | `apps/mobile/ios/MindShift.xcodeproj/project.pbxproj` |
| `production.ios.credentialsSource = "local"`, `autoIncrement: "buildNumber"` | `apps/mobile/eas.json` |
| `credentials.json` is in **single-target** form (`ios.provisioningProfilePath`, `ios.distributionCertificate`) | `apps/mobile/credentials.json` |
| `apps/mobile/ios/` and `apps/mobile/android/` are **gitignored** (`.gitignore:261-262`) → this is a CNG project; native dirs exist locally only as prebuild output | `.gitignore` |
| 8 config plugins in `app.json` (expo-audio, expo-video, expo-camera, expo-media-library, google-signin, `@config-plugins/react-native-webrtc`, …) — all regenerate `ios/` on prebuild | `apps/mobile/app.json` |
| Xcode **26.3** (17C529). watchOS SDK **26.2** + watchsimulator26.2 present | `xcodebuild -showsdks` |
| **No watchOS simulator runtime installed** — only `iOS 26.3`. Watch device *types* exist (Series 9/10/11), but zero watch runtimes and zero bootable watch devices | `xcrun simctl list runtimes` / `list devices available` |

Live read-only check against App Store Connect (GET only, nothing created):

```
bundleIds:  com.sagearbor.mindshift.app   platform=UNIVERSAL  name=MindShift
            com.sagearbor.fitrival.app    platform=UNIVERSAL  name=FitRival
profiles:   MindShift App Store   type=IOS_APP_STORE  state=ACTIVE  platform=IOS
            FitRival App Store    type=IOS_APP_STORE  state=ACTIVE  platform=IOS
```

So: one profile today, `IOS_APP_STORE`, and the bundle id already sits at `UNIVERSAL` platform.
There is no watch-specific `profileType`; Apple's own portal text says an iOS profile covers
"iOS and watchOS apps and App Clips"
([Apple Developer Help](https://developer.apple.com/help/account/provisioning-profiles/create-a-development-provisioning-profile/)).
**Inferred (not proven): `IOS_APP_STORE` is the correct `profileType` for the watch bundle id too.**
Cheap to falsify in the spike — the POST either succeeds or 409s.

---

## 1. The options

### 1.1 `@bacons/apple-targets` — primary candidate

Verified from the npm registry and the published 5.0.0 tarball (I unpacked and read `build/`):

- **Latest `5.0.0`, published 2026-07-17.** 50 versions. peerDeps `{"expo": ">=52"}`.
  1.67M downloads/month. 1384 stars, 72 open issues, not archived.
  ([npm](https://www.npmjs.com/package/@bacons/apple-targets) ·
  [GitHub](https://github.com/EvanBacon/expo-apple-targets))
- **`watch` is a first-class target type.** `build/target.d.ts`:
  ```ts
  readonly watch: {
    readonly productType: "com.apple.product-type.application";
    readonly needsEmbeddedSwift: true;
    readonly displayName: "Watch";
  };
  ```
  plus `watch-widget` (WidgetKit complication). `packages/create-target/templates/watch/`
  ships `index.swift` + `content.swift` + a SwiftUI preview asset catalog.
- **Does it survive `prebuild`?** Yes — that's the entire design. It is a config plugin that
  rewrites the pbxproj on every prebuild, with your Swift sources living *outside* `ios/`
  (in `targets/<name>/`). Per the README: on prebuild "the plugin will generate the Xcode
  project and link the target files to the project."
- **Does it work with EAS cloud builds?** Yes, and the mechanism is explicit. `build/with-eas-credentials.js`
  writes `extra.eas.build.experimental.ios.appExtensions` entries of
  `{ targetName, bundleIdentifier, parentBundleIdentifier, entitlements }`, derived from the
  generated pbxproj (`getEASCredentialsForXcodeProject`, using `target.props.productName`).
  That matters because **eas-cli resolves iOS targets for a managed/CNG project from exactly
  that config key**, not from a pbxproj that doesn't exist yet
  ([`packages/eas-cli/src/project/ios/target.ts`](https://github.com/expo/eas-cli/blob/main/packages/eas-cli/src/project/ios/target.ts):
  `resolveManagedProjectTargetsAsync()` reads `exp.extra?.eas?.build?.experimental?.ios?.appExtensions`).
  README: "The codesigning is theoretically handled entirely by EAS Build."
- **Watch build settings it emits** (`build/configuration-list.js:263`, `createWatchAppConfigurationList`):
  `SDKROOT: "watchos"`, `TARGETED_DEVICE_FAMILY: "4"`, `WATCHOS_DEPLOYMENT_TARGET: "9.4"` (overridable),
  `INFOPLIST_KEY_WKCompanionAppBundleIdentifier` set from the main target's
  `PRODUCT_BUNDLE_IDENTIFIER` — so the companion link is handled for us.

**Maintenance concerns — these are the reason for a spike gate, not a decision:**

| Concern | Evidence |
| --- | --- |
| Last release and last commit are both 2026-07-17. **~2 months idle.** | GitHub `pushed_at` / commit log |
| **No SDK 57 statement anywhere.** The most recent compat work references *Expo 56* ("bump ExtensionStorage podspec to iOS 16.4 for Expo 56", #195). Issue [#196](https://github.com/EvanBacon/expo-apple-targets/issues/196) "Compatibility with expo sdk 56" is still open. | issues/commits |
| It declares `@expo/prebuild-config: ~55.0.6` as a runtime dependency while our tree has `57.0.4` → a second, older copy of prebuild-config in `node_modules`. | 5.0.0 `package.json` vs our `node_modules` |
| ~9 PRs sitting unmerged since May–Aug 2026, incl. [#209](https://github.com/EvanBacon/expo-apple-targets/issues/209) (undeclared runtime deps — same failure mode that broke prebuild for every consumer on SDK 56), [#206](https://github.com/EvanBacon/expo-apple-targets/issues/206)/[#201](https://github.com/EvanBacon/expo-apple-targets/issues/201) (incremental prebuild crashes with `Cannot read properties of undefined (reading 'removeFromProject')`). | open PR list |
| **[#194](https://github.com/EvanBacon/expo-apple-targets/issues/194) — open, zero comments:** adding a target on RN 0.83.6 produced a build that installs and then dies at launch with `dyld: Library not loaded … ReactNativeDependencies.framework`. We are on **RN 0.86**. Reported for a widget extension; a watch app is a separate watchOS binary that does not embed RN, so it may not apply — but the plugin's pbxproj surgery does touch the host target's embed phases. **Unknown until smoke-tested.** | issue #194 |
| Watch-specific open issues: [#147](https://github.com/EvanBacon/expo-apple-targets/issues/147)/[#148](https://github.com/EvanBacon/expo-apple-targets/issues/148) CFBundleVersion mismatch, [#171](https://github.com/EvanBacon/expo-apple-targets/issues/171) watch target's pbxproj is missing the `Frameworks` build phase entirely (blocks SPM/framework deps on the watch side), [#207](https://github.com/EvanBacon/expo-apple-targets/issues/207) Liquid Glass watch icon. | issue search |

### 1.2 `expo-targets` (csark0812) — named fallback

- **Latest `1.9.2`, published 2026-09-15 (six days ago).** 45 versions. peerDeps `{"expo": "*", "expo-constants": "*"}`.
  97 stars, **0 open issues**, active.
  ([npm](https://www.npmjs.com/package/expo-targets) · [GitHub](https://github.com/csark0812/expo-targets))
- **README states verbatim: "Tested on Expo SDK 57."** That is the single thing
  `@bacons/apple-targets` cannot claim, and it is the thing most likely to bite us.
- `watch` **and** `watch-widget` are shipped types with runnable examples
  (`examples/watch/targets/watch/{target.config.json,ios/WatchApp.swift}`) and simulator
  journeys. Their maturity table marks `watch`/`watch-widget` "✅ watchOS (paired sim DoD)".
- Same EAS mechanism: `plugin/src/ios/config-plugins/withEASCredentials.ts` writes
  `extra.eas.build.experimental.ios.appExtensions` with `{targetName, bundleIdentifier, entitlements}`.
  It writes **no** `parentBundleIdentifier` and does **not** special-case watch.
- Their own `docs/limits.md` records the honest limit: **"Watch / watch-widget … Hard stop:
  Device-only Embed Watch Content; Smart Stack needs user-added complication."**
- **Cost of choosing it:** 5.5k downloads/month vs 1.67M. One maintainer, young API, breaking
  changes likely. Naming convention differs (config `name: "MyShare"` → Xcode product
  `MyShareTarget`), which changes the `credentials.json` key.

### 1.3 CNG off — commit a native `ios/` directory

Mechanically the *easiest* way to get a watch target (File → New Target in Xcode, done) and
the one EAS handles via the `resolveBareProjectTargetsAsync()` path — it reads the real pbxproj,
so it sees the watch target with no `appExtensions` declaration needed. eas-cli
[#795](https://github.com/expo/eas-cli/issues/795) is the historical bug report for exactly this
shape, closed as fixed in 2021 (workaround at the time: add the WatchKit target to the iOS app's
dependency list).

**Why not:** `apps/mobile/ios/` is currently gitignored, and we run **8 config plugins** that
regenerate it. Committing `ios/` means every future change to `expo-audio` background recording,
the WebRTC plugin, camera permissions, or an SDK 57→58 upgrade becomes a hand-merge into a
checked-in Xcode project instead of a regenerated one. That is a permanent tax on the whole app
to get one extra target. Take this route only if both plugins fail the spike.

### 1.4 "Full ejection"

Not a distinct option any more. `expo eject` was removed; SDK 50+ has only prebuild/CNG on or
off. §1.3 *is* ejection. Nothing further to evaluate.

---

## 2. Signing — what our bootstrap script must grow

### 2.1 `credentials.json` multi-target: **confirmed supported**

Verified from Expo's docs ([Local credentials](https://docs.expo.dev/app-signing/local-credentials/)),
quoted verbatim:

> "If your iOS app is using App Extensions like Share Extension, Widget Extension, and so on,
> you need to provide credentials for every target of the Xcode project. This is necessary
> because each extension is identified by an individual bundle identifier."

```json
{
  "ios": {
    "multitarget":     { "provisioningProfilePath": "…", "distributionCertificate": { "path": "…", "password": "…" } },
    "shareextension":  { "provisioningProfilePath": "…", "distributionCertificate": { "path": "…", "password": "…" } }
  }
}
```

Keys are **Xcode target names**. Note the shape change: the multi-target form replaces the
single-target form entirely — `ios.provisioningProfilePath` at the top level is no longer read.
So this is a **rewrite** of `credentials.json`, not an addition, and the script must emit:

```json
{
  "ios": {
    "MindShift": { "provisioningProfilePath": "~/.config/ios-credentials/com.sagearbor.mindshift.app/AppStore.mobileprovision",
                   "distributionCertificate": { "path": "~/.config/ios-credentials/_team/dist.p12", "password": "…" } },
    "watch":     { "provisioningProfilePath": "~/.config/ios-credentials/com.sagearbor.mindshift.app.watchkitapp/AppStore.mobileprovision",
                   "distributionCertificate": { "path": "~/.config/ios-credentials/_team/dist.p12", "password": "…" } }
  }
}
```

`"MindShift"` is proven from our pbxproj. `"watch"` is whatever the plugin names the target —
`@bacons` defaults to a sanitized directory name, `expo-targets` appends `Target`. **Must be
read back out of the generated pbxproj, never guessed.**

**Independent confirmation that this is the route that works.** eas-cli
[#2578](https://github.com/expo/eas-cli/issues/2578) is a watch target failing with
`error: No profiles for 'xx.watchkitapp' were found`. The reporter's conclusion after trying
EAS-managed credentials (2024-10-14 and again 2025-07-21):

> "We couldn't find a way to handle this case with credentials managed by EAS. … we ended up
> using iOS 'local' credentials definition for multiple targets, like explained in Expo
> documentation: docs.expo.dev/app-signing/local-credentials/#multi-target-project"

We are *already* on local credentials for unrelated reasons (the script's docstring: EAS refuses
to mint a distribution certificate non-interactively). That accident of history is what makes
this feasible at all.

### 2.2 Changes to `scripts/ios_credentials_bootstrap.py`

Small and mechanical. Roughly:

1. **New arg** `--watch-target NAME` (or `--extra-target bundleSuffix=targetName`), so the
   script stays generic and reusable across repos as its docstring promises.
2. `ensure_bundle_id(asc, f"{bundle_id}.watchkitapp", f"{name} Watch", team)` — a second
   registration. Existing function works unchanged; `platform: "IOS"` is what it already POSTs
   and Apple normalizes to `UNIVERSAL` (observed on both existing bundle ids).
3. `ensure_profile(...)` a second time with `profile_name = f"{name} Watch App Store"` and
   `profileType: "IOS_APP_STORE"`, into `~/.config/ios-credentials/<watch bundle id>/`.
   The existing per-bundle-id directory layout already accommodates this — the comment at
   line 299 ("only the profile is per bundle id") is exactly right.
4. **No certificate change.** One `IOS_DISTRIBUTION` cert signs both targets; the Apple
   two-per-team cap logic at lines 193-206 stays untouched. Both `credentials.json` entries
   point at the same `_team/dist.p12`.
5. **Rewrite the `credentials.json` writer** (lines 313-318) into the multi-target form, with
   the main app's Xcode target name as a required input.

Estimated diff: ~60 lines. This is the cheapest part of the whole project.

### 2.3 The `appExtensions` / local-credentials interaction — the one genuine unknown

For a CNG project eas-cli enumerates targets from `extra.eas.build.experimental.ios.appExtensions`
(§1.1). Both plugins write it. `credentialsSource: "local"` then means eas-cli takes the
credentials for each enumerated target from `credentials.json` rather than its own service
(`SetUpTargetBuildCredentialsFromCredentialsJson.ts` operates per target, keyed on
`app.projectName`, and does no bundle-id or entitlement validation of its own).

**Inferred, not verified:** that eas-cli also rewrites `CODE_SIGN_STYLE` to `Manual` on the
watch target. This matters because `@bacons/apple-targets` hardcodes
`CODE_SIGN_STYLE: "Automatic"` for the watch target (`configuration-list.js:279`), and
"Automatic signing is disabled and unable to generate a profile" is the literal text of the
eas-cli #2578 failure. If eas-cli does not fix it up, we need a follow-on config plugin to force
`CODE_SIGN_STYLE = Manual` + `PROVISIONING_PROFILE_SPECIFIER` on that target. **Assume one day
of work here and budget for it.**

### 2.4 Version-mismatch gate — must be cleared, verified from source

Apple rejects an upload whose watch app version does not match the host's:
`90379` for `CFBundleVersion`, `90473` for `CFBundleShortVersionString`.

- **`CFBundleVersion`: handled in 5.0.0.** `createWatchAppConfigurationList` sets
  `CURRENT_PROJECT_VERSION: currentProjectVersion`, and `configuration-list.js:464` seeds it
  from `process.env.EAS_BUILD_IOS_BUILD_NUMBER`. This is the [#148](https://github.com/EvanBacon/expo-apple-targets/issues/148)
  fix, and it *is* in the published tarball. Caveat: a commenter on #148 (2026-03-13) reports
  "I'm still seeing this error in EAS after updating."
- **`CFBundleShortVersionString`: NOT handled.** `MARKETING_VERSION: "1.0"` is hardcoded in
  `createWatchAppConfigurationList` (line 289) — and our app is at **1.18.0**. With
  `GENERATE_INFOPLIST_FILE: "YES"` the watch Info.plist will be built from that literal.
  Expected result: **ITMS-90473 on the first TestFlight upload.** This is the exact issue
  AlekseyP18 raised on #147.

Fix is small (patch `MARKETING_VERSION` to the host's version via a follow-on config plugin, or
`patch-package`), but it *will* happen and it happens at the most expensive point — after a
20-minute cloud build. Test it in Phase 1, not Phase 4. Note also that our `eas.json` uses
`autoIncrement: "buildNumber"`, which is precisely the setup that makes this bite.

---

## 3. Watch ↔ phone: the bridge

**WatchConnectivity is the only channel. App Groups do not work here** — an App Group is shared
between an app and its extensions *on one device*; the watch is a separate device. Both plugins
default watch targets into App Groups, which is a convenience for the watch app's own storage,
not a phone link.

### 3.1 `react-native-watch-connectivity` — the only real library

- **Latest `2.0.0`, published 2026-03-20** (jumping from `1.1.0` in Sep 2022 — the
  `Feat/modernise scaffold` rewrite). MIT. 787 stars, 32 open issues, last push 2026-07-03.
  Repo moved to `watch-connectivity/react-native-watch-connectivity`.
  ([npm](https://www.npmjs.com/package/react-native-watch-connectivity) ·
  [GitHub](https://github.com/watch-connectivity/react-native-watch-connectivity))
- **New Architecture ready.** Its `package.json` declares
  `codegenConfig: { name: "WatchConnectivitySpec", type: "modules" }` and it is built against
  `react-native@0.84.0`. README states "React Native 0.76+", "iOS 13.4+". We are on RN 0.86 —
  the most recent combination in the ecosystem, and untested together, but the codegen shape is
  correct.
- **No Expo config plugin**, and none needed: it's a standard RN module with a podspec, so Expo
  autolinking picks it up on prebuild. README: "This library has been successfully used in Expo
  apps (Bare Workflow with EAS Build)." 52k downloads/month.
- **It covers the phone side only.** README, verbatim:
  > "This library does not allow you to write your Apple Watch apps in React Native but rather
  > allows your RN iOS app to communicate with a watch app written in Obj-C/Swift."
- JS surface: `sendMessage` (with reply callback), `useReachability()`, `usePaired()`,
  `useInstalled()`, plus `transferUserInfo` / `updateApplicationContext` / file transfer on the
  native side.

### 3.2 What we write ourselves

Three pieces, none of which the plugins or the library provide:

1. **Watch side, Swift:** a `WCSessionDelegate`, activated *after* the delegate is set
   (the plugin's own skill doc: "Activating a `WCSession` without setting a delegate first is a
   programming error"), plus the entire SwiftUI UI.
2. **Phone side:** `react-native-watch-connectivity` wiring into our existing Zustand store.
3. **A message protocol**, constrained by the API: `sendMessage` / `transferUserInfo` payloads
   are `[String: Any]` restricted to **property-list types only** — no JSON blobs, no binary
   unless you go through `transferFile`.

And the behavioural constraint that shapes the whole product: `sendMessage` works **only when
both apps are foregrounded and `isReachable == true`**. For a live-coaching app that means the
watch must be actively on-screen (or holding a workout session) during a conversation, with
`updateApplicationContext`/`transferUserInfo` as the queued fallback. Watch background budget is
roughly **one refresh task per hour** for a docked app.

### 3.3 Not an option

Running React Native *on* watchOS. `@bacons/apple-targets` has an `exportJs` flag but its own
docs scope it: "Intended for App Clips and Share Extensions." `expo-targets` supports RN
`entry` targets for appex processes, not watchOS. The watch app is SwiftUI or nothing.

---

## 4. What the simulator can actually prove

**Current state: nothing — no watchOS runtime is installed.** `xcrun simctl list runtimes`
shows only `iOS 26.3`. Step zero is
`xcodebuild -downloadPlatform watchOS` (optionally `-architectureVariant arm64` to halve the
download on Apple silicon;
[Apple docs](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components.md)).
The watchOS 26.2 SDK itself *is* present, and Series 9/10/11 device types are registered — only
the runtime image is missing.

**Provable on the simulator:**

- The target builds. `SDKROOT=watchos`, `TARGETED_DEVICE_FAMILY=4` compile cleanly.
- The whole SwiftUI UI, on every watch size (41/42/45/46 mm), in light and dark, including
  Xcode Previews.
- The pbxproj surgery survives repeat prebuilds — this is the check for #201/#206, and it's the
  cheapest high-value test we have.
- A WCSession round trip, **with caveats**: pairing requires booting the watch simulator target
  first, then the iOS app; `isReachable` only goes true with both apps foregrounded; and
  Apple's own forum threads record WatchConnectivity errors that "do not happen when using
  physical devices." Treat a green simulator round trip as *necessary, not sufficient*.

**Not provable without an Apple Watch:**

- `Embed Watch Content` — `expo-targets`' `limits.md` classes this as a **device-only hard stop**.
- Real installation flow (watch app appearing via the iPhone Watch app), complication /
  Smart Stack placement (needs a user-added complication), Digital Crown, real haptics,
  heart rate, accelerometer, gyroscope (the plugin's own doc: "unavailable in the Simulator"),
  background refresh budget, battery cost.
- Anything about whether a live nudge actually lands on a wrist mid-conversation — which is the
  entire point of putting MindShift on a watch.

**And the App Store leg is unprovable until upload.** ITMS-90379 / ITMS-90473 only surface when
App Store Connect validates a real archive, so "does it ship" costs one full EAS production
build + TestFlight upload per attempt.

---

## 5. Effort, in phases

| Phase | Work | Estimate |
| --- | --- | --- |
| **P0 — Spike / go-no-go** | `xcodebuild -downloadPlatform watchOS`. Throwaway branch. Add plugin + `targets/watch/` with the stock template. `npx expo prebuild -p ios --clean`, then prebuild **again** without `--clean` (the #201 check). Confirm the watch target and `appExtensions` appear. Build for the watch simulator locally. Launch the *phone* app on device-less sim and confirm no dyld regression (#194). Falls back to `expo-targets@1.9.2` if prebuild breaks. | **0.5–1 day** |
| **P1 — Signing, end to end** | Extend `ios_credentials_bootstrap.py` (§2.2, ~60 lines). Rewrite `credentials.json` multi-target. One `eas build -p ios --profile production` **and** a TestFlight upload. Expect to fight ITMS-90473 (§2.4) and possibly `CODE_SIGN_STYLE` (§2.3). Budget for 2–3 cloud build round trips at ~20 min each. | **1–2 days** |
| **P2 — Skeleton + link** | SwiftUI watch app shell. `WCSessionDelegate` on both sides. `react-native-watch-connectivity@2.0.0` on RN 0.86 (untested pairing — allow slack). Define the plist-safe message protocol. First round trip in paired sims. | **2–4 days** |
| **P3 — The actual feature** | Whatever the wrist shows — heat dial, nudge haptic, session start/stop. Reachability fallbacks, queued `transferUserInfo`, watch-side state, complication if wanted. **This is a second app's worth of product work, in a language and framework not otherwise used in this repo.** | **1–3 weeks** |
| **P4 — Store** | watchOS platform on the App Store Connect record, watch-sized screenshots, review. | **0.5–1 day** |

**Mechanism proven: ~2 days. Something shippable: ~3–5 weeks.** P3 is the number that matters
and the number with the widest error bars.

---

## 6. Biggest risk

**We are building a second, native, unverifiable app.**

Three things compound:

1. **No code sharing.** The watch app is Swift/SwiftUI. None of the heat rubric, nudge policy,
   diarizer or hysteresis work in this repo runs on it. Every wrist feature is a reimplementation
   that then has to stay in sync with the phone across a plist-typed message protocol.
2. **No verification path.** The repo's standing rule is "every behavior proven by file replay +
   report + test gate; the owner is not the manual tester." A watch app cannot meet that. The
   simulator can prove the target builds and the UI renders; it cannot prove the thing the
   feature exists for. And there is no Apple Watch and no iPhone to fall back on.
3. **Plugin risk sits on a 2-month-idle dependency with no SDK 57 claim**, carrying a known
   RN 0.83+ framework-embed regression (#194), a known incremental-prebuild crash (#201), a
   watch target with no `Frameworks` build phase (#171), and a hardcoded `MARKETING_VERSION`
   that will reject our first upload. All individually small; collectively they mean the
   build plumbing owns a slice of every future SDK upgrade.

If this goes forward, gate it on P0+P1 clearing in two days. If P0 fails on both plugins, the
honest next question is not "commit `ios/`" — it is whether a watch app is worth acquiring
hardware for, because without a watch, P3 ships blind.

---

## 7. Verified vs inferred

**Verified — read from files in this repo, the published 5.0.0 tarball, npm's registry, the
GitHub API, `xcodebuild`/`simctl` on this machine, or quoted directly from Expo/Apple docs:**

- Every row in §0, including the `MindShift` Xcode target name, the gitignored `ios/`, the
  missing watchOS simulator runtime, and the live ASC bundle-id/profile listing.
- `@bacons/apple-targets@5.0.0` version, date, peer range, dependency on
  `@expo/prebuild-config ~55.0.6`, download count, star/issue counts, idle since 2026-07-17.
- The `watch` target type definition, its emitted build settings (`SDKROOT`,
  `TARGETED_DEVICE_FAMILY`, `WATCHOS_DEPLOYMENT_TARGET 9.4`, `WKCompanionAppBundleIdentifier`,
  `CODE_SIGN_STYLE: "Automatic"`, `MARKETING_VERSION: "1.0"`, `CURRENT_PROJECT_VERSION` from
  `EAS_BUILD_IOS_BUILD_NUMBER`) — read out of `build/configuration-list.js`.
- `withEASTargets` / `getEASCredentialsForXcodeProject` writing
  `extra.eas.build.experimental.ios.appExtensions` — read out of `build/with-eas-credentials.js`.
- eas-cli resolving CNG targets from that same config key
  (`packages/eas-cli/src/project/ios/target.ts`).
- The multi-target `credentials.json` schema, quoted verbatim from Expo's docs, keyed by
  Xcode target name.
- eas-cli #2578: EAS-managed credentials could not do a watch target; local multi-target
  credentials could.
- `expo-targets@1.9.2` version/date, 0 open issues, "Tested on Expo SDK 57", watch examples,
  its `withEASCredentials.ts`, and its own device-only watch limits.
- `react-native-watch-connectivity@2.0.0` version/date, `codegenConfig` (new arch), RN 0.76+ /
  iOS 13.4+ floor, no Expo plugin, and the README line that it does not let you write the watch
  app in RN.
- ITMS-90379/90473 semantics and that `MARKETING_VERSION`/`CURRENT_PROJECT_VERSION` are the fix.

**Inferred — flagged, and each cheap to falsify in P0/P1:**

- `profileType: IOS_APP_STORE` is correct for a `.watchkitapp` bundle id (strongly implied by
  Apple's portal text; falsified by a single POST).
- eas-cli overrides `CODE_SIGN_STYLE` to Manual for targets it has local credentials for. If not,
  add a config plugin (§2.3).
- `@bacons/apple-targets@5.0.0` works on Expo SDK 57 / RN 0.86 at all. **This is the single
  biggest untested assumption in the document** and the whole reason P0 exists.
- #194's dyld failure does not apply to a watch target (separate watchOS binary, no embedded RN).
- The generated target's name will be the directory name — must be read back from the pbxproj
  before writing `credentials.json`.
- P3's 1–3 week range. Pure judgement, no evidence behind it.

---

## Sources

- [@bacons/apple-targets on npm](https://www.npmjs.com/package/@bacons/apple-targets) · [EvanBacon/expo-apple-targets](https://github.com/EvanBacon/expo-apple-targets) · [apple-targets README](https://github.com/EvanBacon/expo-apple-targets/blob/main/packages/apple-targets/README.md) · [watch skill doc](https://github.com/EvanBacon/expo-apple-targets/blob/main/skills/apple-targets/watch.md)
- Issues: [#147](https://github.com/EvanBacon/expo-apple-targets/issues/147) · [#148](https://github.com/EvanBacon/expo-apple-targets/issues/148) · [#171](https://github.com/EvanBacon/expo-apple-targets/issues/171) · [#194](https://github.com/EvanBacon/expo-apple-targets/issues/194) · [#196](https://github.com/EvanBacon/expo-apple-targets/issues/196) · [#201](https://github.com/EvanBacon/expo-apple-targets/issues/201) · [#209](https://github.com/EvanBacon/expo-apple-targets/issues/209)
- [expo-targets on npm](https://www.npmjs.com/package/expo-targets) · [csark0812/expo-targets](https://github.com/csark0812/expo-targets) · [its limits.md](https://github.com/csark0812/expo-targets/blob/main/docs/limits.md)
- [react-native-watch-connectivity on npm](https://www.npmjs.com/package/react-native-watch-connectivity) · [watch-connectivity/react-native-watch-connectivity](https://github.com/watch-connectivity/react-native-watch-connectivity)
- [Expo — Local credentials (credentials.json, multi-target)](https://docs.expo.dev/app-signing/local-credentials/) · [Expo — iOS app extensions on EAS Build](https://docs.expo.dev/build-reference/app-extensions/)
- [eas-cli #2578 — provisioning profile on an Apple Watch target](https://github.com/expo/eas-cli/issues/2578) · [eas-cli #795 — building an app with a watch companion](https://github.com/expo/eas-cli/issues/795) · [eas-cli src/project/ios/target.ts](https://github.com/expo/eas-cli/blob/main/packages/eas-cli/src/project/ios/target.ts)
- [Apple — Create a provisioning profile (iOS profiles cover watchOS)](https://developer.apple.com/help/account/provisioning-profiles/create-a-development-provisioning-profile/) · [Apple — Downloading additional Xcode components](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components.md) · [Apple forum — CFBundleShortVersionString / watch version mismatch](https://developer.apple.com/forums/thread/702394) · [Apple forum — watchOS sim sendMessage limitations](https://developer.apple.com/forums/thread/125484)
