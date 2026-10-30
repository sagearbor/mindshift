# Ported from Gauge

`apps/watch/` (this Gradle project: `wearApp/`, `shared/`, plus the root Gradle wiring
`build.gradle.kts`, `settings.gradle.kts`, `gradle.properties`, `gradle/`, `gradlew`,
`gradlew.bat`) was imported from `sagearbor/gauge` @ commit `2157433` on 2026-08-15,
as part of Phase 1 of the MindShift/Gauge unification ("one repo, one engine").

## Decision: clean copy, no history import

This is a clean copy of the source tree at that commit — Gauge's git history was
**not** rewritten into this repo. The archived `sagearbor/gauge` repository remains
the permanent historical record for this code; consult it directly for blame/history
older than this import.

## What was left behind, and why

- **`androidApp/`** — Gauge's companion phone app. MindShift's `apps/mobile/` is the
  surviving phone client going forward; `androidApp/` is redundant and was not ported.
- **`webApp/`** — Gauge's web client. No equivalent surviving product in the unified
  repo's Phase 1 scope; not ported.
- **`server/`** — Gauge's backend. MindShift's `server/` is the surviving backend;
  Gauge's server logic is being ported piecemeal into MindShift's flat module layout
  in later Phase 1 tasks (see `docs/plans/2026-08-15-phase1-one-repo-one-engine.md`),
  not copied wholesale here.
- **Vendored `server/engine/`** — Gauge vendored its own copy of the analysis engine
  under `server/engine/`. Per the global constraints for this unification, that
  vendored engine is never copied; ported server code is adapted to import this
  repo's existing flat modules (`speaker_id`, `whisper_transcriber`, etc.) instead.

Only the Wear OS app (`wearApp/`) and the Kotlin Multiplatform module it depends on
(`shared/`) — i.e. the parts with no MindShift equivalent — were imported.

## Keystore and applicationId facts

**Superseded 2026-10-30 ("rebrand watch"):** the watch left its own Play listing and
now ships INSIDE the MindShift listing — Google's single-listing model for phone +
Wear OS.

- `applicationId` is now `com.sagearbor.mindshift.app` (debug keeps the `.debug`
  suffix). Kotlin namespaces stay `app.gauge.*` — invisible to users.
- versionCode = `1_000_000 + n` (first: 1_000_019) so it can never collide with the
  phone's Expo-incremented codes in the shared listing; versionName follows the phone.
- Release signing uses the **MindShift listing's upload key** (Expo EAS-managed for
  the phone). `apps/watch/keystore.properties` (gitignored) is a symlink to
  `~/.config/mindshift/keystore.properties`, which points at
  `~/.config/mindshift/mindshift-upload.jks`. Never copy either into the repo.
- The old upload key `~/.config/gauge/gauge-upload.jks` belongs to the retired
  `com.sagearbor.gauge.wear` listing only (being unpublished, never deleted). Keep it
  backed up; do not use it for MindShift builds.

## Build

`apps/watch/` is a standalone Gradle build (`rootProject.name = "mindshift-watch"`,
modules `:shared` + `:wearApp`, no `:androidApp`). See root `.gitignore` for the
build-output and local-config paths excluded from version control.
