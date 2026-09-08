# "Hey Google, start my journal" — the remaining native step

Status: the OTA-able half **shipped** (deep links); the Assistant half **rides the next EAS build**.

## What works today (OTA, no native build needed)

- `mindshift://journal/start` → opens Live Coach, selects Journal mode, and starts it,
  honoring the existing gates (enrolled owner voiceprint; mic permission — a failed gate lands
  on the screen with its gate message/mic banner visible instead of a silent failure).
- `mindshift://journal/stop` → stops a running journal session (and only a *journal* session).
- Parsing: `apps/mobile/src/nav/journalLink.ts`; routing: `App.tsx` (`pendingJournal`, same
  wait-behind-auth pattern as call invites); execution: `LiveCoachScreen.tsx` (`journalAction`
  prop).
- Scheme intent filter: **already present** in the shipped native app. Expo prebuild generates
  the `VIEW` + `BROWSABLE`/`DEFAULT` intent filter for `mindshift://` from the top-level
  `"scheme": "mindshift"` in `apps/mobile/app.json` — proven on-device by the existing
  `mindshift://call/<code>` invite links. No extra `android.intentFilters` entry is needed
  (adding one for the same scheme would just duplicate the generated filter). Sanity check:
  `adb shell am start -a android.intent.action.VIEW -d "mindshift://journal/start"`.

## What cannot OTA: App Actions registration

Assistant only invokes an app when the APK's **manifest/resources** declare the capability —
resources can't change over the air, so this must ship in a native (EAS) build, together with
the pending mic foreground-service fix.

Exact remaining step (no native *module* — config only):

1. **BII**: use `actions.intent.OPEN_APP_FEATURE` (feature name "journal" / "my journal").
   There is no journaling-specific BII; OPEN_APP_FEATURE maps a spoken feature name to a deep
   link, which is exactly our shape.
2. **Where it's declared**: since App Actions moved onto the shortcuts framework, the
   declaration is a `res/xml/shortcuts.xml` with a `<capability
   android:name="actions.intent.OPEN_APP_FEATURE">` whose intent template points at
   `mindshift://journal/start`, plus
   `<meta-data android:name="android.app.shortcuts" android:resource="@xml/shortcuts"/>` on the
   MainActivity in `AndroidManifest.xml`.
3. **How, under Expo**: there is no official Expo plugin for App Actions. Write a small local
   config plugin — `apps/mobile/plugins/withAppActions.js`, same pattern as the existing
   `plugins/withOrtGradle9.js` — that (a) copies `shortcuts.xml` into
   `android/app/src/main/res/xml/` via `withDangerousMod`, and (b) adds the `meta-data` element
   via `withAndroidManifest`. Register it in `app.json`'s `plugins` array. (The community
   `expo-app-actions`-style packages are unmaintained; the ~40-line local plugin is the
   dependable route.)
4. **Play requirement**: App Actions only trigger for apps distributed through Play (internal
   track is fine). Test before rollout with the "Google Assistant plugin" App Actions test tool
   in Android Studio, or by uploading to the internal track.
5. **Ship it with**: the next EAS build, alongside the mic foreground-service fix — both are
   native-manifest changes and neither can OTA.

Until that build lands, the links themselves already work from anywhere that can fire an
Android intent (Assistant routines "Open URL", Tasker, notification actions, adb).
