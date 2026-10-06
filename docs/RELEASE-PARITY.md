# Keeping Android and iOS in step

One codebase, two binaries, one over-the-air (OTA) channel.

| Layer | How it stays equal |
|---|---|
| JavaScript (screens, copy, logic) | One `scripts/ota_publish.sh` run publishes to both platforms. Nothing to do. |
| Native binary (modules, plugins, permissions, the watchOS target) | `apps/mobile/__tests__/nativeParity.test.ts` fails when native inputs change without an `expo.version` bump. `scripts/release_both.sh` builds both platforms from one commit. |
| Update channel | Store builds listen on `production`; internal APKs on `preview`. Publish to `production`; publish to `preview` only for a phone that has an internal build. |

## When you change something native

1. Bump `expo.version` in `apps/mobile/app.json` (the runtime policy is `appVersion`, so an OTA can never reach a binary from another version).
2. `cd apps/mobile && node scripts/nativeFingerprint.js --update "why"` and commit `native-fingerprint.json` with the bump.
3. `scripts/release_both.sh` — refuses to run until the parity test passes, then queues Android and iOS in one EAS run.
4. Submit each store as the playbooks describe (`~/.claude/playbooks/`).

## Current markers (2026-09-25)

expo.version 1.18.0 · Android versionCode 40 (Play, live) · iOS buildNumber 5 (TestFlight) · both built from the same native set, recorded in `apps/mobile/native-fingerprint.json`.
