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

- `apps/watch/keystore.properties` (gitignored, recreated locally by each developer/
  agent from the same source) points at `~/.config/gauge/gauge-upload.jks` — the
  wear package's permanent Play upload key. This is **deliberately unchanged**: the
  signing identity must stay stable for Play Store upload continuity, independent of
  which repo the source code lives in.
- The wear app's `applicationId`, `com.sagearbor.gauge.wear`, is likewise **permanent**
  and is not renamed by this import. Display branding (app name, icon, in-app copy)
  will be updated to MindShift branding in Phase 3; the package id itself stays as-is.

## Build

`apps/watch/` is a standalone Gradle build (`rootProject.name = "mindshift-watch"`,
modules `:shared` + `:wearApp`, no `:androidApp`). See root `.gitignore` for the
build-output and local-config paths excluded from version control.
