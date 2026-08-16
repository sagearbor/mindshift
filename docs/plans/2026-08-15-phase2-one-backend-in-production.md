# Phase 2 — One Backend in Production: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** The unified service (Phase 1's merged app) serving BOTH domains live on Cloud Run `mindshift-api`; watch vNext (vc10) on Play internal pointing at it; verified via telemetry; `gauge-api` enters its grace week.

**Owner decisions received 2026-08-15 evening:** Play upload GO. Telemetry stays open for testing (token-gating deferred; owner briefed on production patterns). Owner is granting the `play-publisher@arborfam-hub` service account access to the `com.sagearbor.gauge.wear` listing (removes browser flow permanently).

**Master context:** docs/plans/2026-08-15-unification-mindshift-absorbs-gauge.md §Phase 2 + the Phase 1 plan's "Pre-deploy hardening" hand-forward notes (binding here).

## Global Constraints
- Branch `feat/unification-phase2`; per-task commits; PR with merge commit.
- Gradle FOREGROUND, Bash timeout 600000, JAVA_HOME=/Users/sophie.arborbot/jdk17/Contents/Home. If a long run auto-backgrounds at the tool ceiling anyway: check its output artifacts (JUnit XML/exit code) on the next tool call — never end a turn waiting on a notification.
- Gates every task: full `python3 -m pytest -q` (exit code), `ruff check server tests`; watch gate when apps/watch changes.
- Deploys ONLY via `scripts/deploy_cloudrun.sh` (env forwarding is load-bearing); OTA only via `scripts/ota_publish.sh` (not needed this phase); Play upload ONLY via `scripts/play_publish.py` after the owner's SA grant (NO browser flow).
- Honest degradation; no store-visible action beyond the already-approved internal-track watch upload.

### Task H1: Rate-limit + cap the ported unauthenticated surface
**Files:** Modify `server/watch/app.py` (thread a rate-limit dependency), `server/watch/routers/{telemetry.py,pairing.py,rest.py}`; Test `server/tests/watch/test_rate_limits.py` (new)
- Attach main.py's `_rate_limit` (via a dependency parameter threaded from `build_watch_routers()`, defaulting to a no-op for `create_watch_test_app` unless supplied) to: `POST /telemetry`, `GET /telemetry`, `POST /me/pair/start`, `GET /me/pair/status`. Cap `POST /enroll` body at 5 MB (read gauge's semantics: raw WAV baseline clip; mirror voice.py's `MAX_DIRECT_ENROLL_BYTES` pattern with 413 over-cap).
- [ ] Tests first (rate-limited route returns 429 when the injected limiter trips; enroll 413 over cap; unauthenticated behavior otherwise unchanged) → RED → implement → GREEN → full gates → commit `feat(watch-api): rate-limit unauthenticated watch routes + enroll body cap`.

### Task H2: Firestore stores off the event loop
**Files:** Modify `server/watch/{store.py,pairing_store.py,telemetry_store.py}`
- Wrap every synchronous google-cloud SDK call inside the async Firestore-store methods in `asyncio.to_thread`, exactly matching `blobs.py`'s existing pattern. Memory stores untouched. No behavior change; existing ported tests are the net.
- [ ] Implement → full gates (memory-store tests prove interfaces intact) → commit `perf(watch-api): Firestore SDK calls via asyncio.to_thread (shared event loop safety)`.

### Task H3: Lazy diarizer — no torch at import time
**Files:** Modify `server/watch/app.py`; Test extend `server/tests/watch/test_prod_wiring.py`
- Replace the eager `speaker_id.is_available()` / `EmbeddingDiarizationService` construction in `build_watch_routers()` with a lazy proxy resolved on first use (mirror the embedder's documented lazy pattern). Test: importing `main` must not import torch (`assert "torch" not in sys.modules` after a fresh-subprocess import — write it as a subprocess check so prior test imports don't pollute).
- [ ] Test first → RED (torch currently imports... note: torch absent in CI, so assert on `speechbrain`/import-attempt instead — implementer verifies what actually gets imported and pins that) → implement → GREEN → full gates → commit `perf(watch-api): lazy diarizer — keep torch out of cold-start imports`.

### Task H4: Minors M6+M7
**Files:** `server/watch/services.py` (import MINDSHIFT_MODEL default from a shared constant instead of duplicating — put the literal in ONE place both main.py and services.py read; smallest honest change), `server/tests/watch/test_auth_routes.py` (StubVerifier import path made consistent). M8 (firestore dep in base reqs) recorded as accepted — needed in the prod image; note in commit body.
- [ ] Implement → full gates → commit `chore(watch-api): de-duplicate model default + tidy test import path`.

### Task D1: Firestore data migration — episodes → live_sessions
**Files:** Create `scripts/migrate_episodes_to_live_sessions.py`
- Copies every doc from collection `episodes` to `live_sessions` (idempotent: skip existing ids; dry-run default, `--execute` flag; prints counts). Uses ADC like the stores. Tiny dataset (personal testing). Include `--verify` mode comparing counts. NOT run in CI; run manually at deploy time (step D3).
- [ ] Script + a unit test for its transform logic with the memory pattern (no real Firestore in tests) → gates → commit `feat(migration): episodes→live_sessions Firestore copy script (idempotent, dry-run default)`.

### Task D2: PR + merge (hardening + migration land on main before any deploy)
- [ ] Full gates incl. watch (untouched → cached result ok) → whole-branch review (sonnet — small branch) → PR `feat: Phase 2 pre-deploy hardening + migration tooling` → merge (merge commit).

### Task D3: Deploy the unified service + verify  (owner-approved; controller-driven, no subagent)
- [ ] `git checkout main && git pull`; run `scripts/migrate_episodes_to_live_sessions.py` dry-run → review counts → `--execute` → `--verify`.
- [ ] Deploy via `scripts/deploy_cloudrun.sh` (reads .env; forwards MINDSHIFT_* incl. the four watch vars). Watch memory: service is 2Gi/4cpu; gauge ran 4Gi — after deploy, exercise a WS ingest and watch memory; bump to 4Gi if evidence demands (record decision).
- [ ] Verify: `GET /health` 200 on the run.app URL; watch-domain routes respond (`/me/pair/start` mints a code; `GET /telemetry` returns events after a test POST); existing MindShift routes unaffected (recordings list on owner's account via the app still works — ask owner for a 30-second phone check OR verify via web app).
- [ ] `gauge-api` NOT touched — grace window starts only after the watch vNext is verified (D5).

### Task D4: Watch vNext (vc10) — point at the unified service
**Files:** Modify `apps/watch/wearApp/build.gradle.kts` (GAUGE_API_BASE → `https://mindshift-api-<hash>.us-central1.run.app` — read the ACTUAL deployed URL from `gcloud run services describe mindshift-api`; versionCode 10, versionName 0.4.0), plus the queued pairing cold-start fast-follow: `DevicePairingClient`/`PairingPoller` timeout 10s→30s + one retry (+ tests).
- [ ] TDD on the pairing-client changes → full watch gate (FOREGROUND, timeout 600000) → `./gradlew :wearApp:bundleRelease` (keystore.properties present; verify signature `keytool -printcert -jarfile` SHA1 starts 81:C2:D2:98) → commit `feat(watch): v0.4.0 (vc10) — unified backend URL + pairing cold-start tolerance` → PR or direct per branch state.

### Task D5: Publish watch vNext to Play internal via API + verify end-to-end
**Files:** Modify `scripts/play_publish.py` only if needed (it's package-parameterized already — likely just invocation docs).
- [ ] `python3 scripts/play_publish.py --aab apps/watch/wearApp/build/outputs/bundle/release/wearApp-release.aab --package com.sagearbor.gauge.wear --track internal --service-account ~/.config/play/fitrival-sa.json --notes "0.4.0: unified MindShift backend"` (SA grant done by owner). Expected benign warnings: deobfuscation + native symbols.
- [ ] Owner updates the watch app from Play internal; verify via telemetry (not assumption): app-start beacon with 0.4.0, pairing round-trip (fresh code → claim in Gauge phone app or web → status), one live session end-to-end saved as `live_session` on the unified service.
- [ ] THEN note grace-window start for gauge-api (decommission after ~a week — Phase 4 boundary; keep-warm decision for unified service raised to owner with $ figure once real latencies observed).

## Phase-3 fast-start note (owner request 2026-08-15): first Phase 3 deliverable = "Set up your watch" screen in the Expo app (Play deep-link remote-install button + 6-digit pairing-code entry), OTA-shipped.
