# Phase 1 — One Repo, One Engine: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import Gauge's Wear OS app + KMP module into `apps/watch/` and port Gauge's backend (routes, stores, auth, tests) into a `server/watch/` package wired to THIS repo's analysis engine — ending with one repo whose combined suites are all green, nothing deployed.

**Architecture:** Two independent tracks. Track A copies `wearApp/` + `shared/` + Gradle scaffolding into `apps/watch/` as a standalone Gradle build (no `:androidApp`), gated by a new CI job. Track B ports Gauge's already-factored backend (closure-factory routers + Protocol stores with memory/Firestore impls) into `server/watch/`, renames the colliding "episode" concept to **live_session**, chains watch device-token auth after this repo's Firebase verification, and points the analysis pipeline at this repo's engine modules (`speaker_id` v2, `whisper_transcriber` shared-cache) instead of Gauge's vendored copies.

**Tech Stack:** Python/FastAPI/pytest (backend), Kotlin/Compose/KMP + Gradle 8.9/AGP 8.5.2/Kotlin 1.9.23 (watch), GitHub Actions CI.

**Master context:** `docs/plans/2026-08-15-unification-mindshift-absorbs-gauge.md` (locked decisions — do not relitigate). Source repo: `/Users/sophie.arborbot/PROJECTS/github_repos/gauge` at commit `2157433` (referred to below as `$GAUGE`).

## Global Constraints

- Branch: `feat/unification-phase1`; per-task commits; PR merged with a **merge commit** (NOT squash — preserve task history).
- **Gradle ALWAYS foreground with `timeout: 600000`** (cold 10–20 min on this Mac). Never background gradle/pytest/npm.
- `JAVA_HOME=/Users/sophie.arborbot/jdk17/Contents/Home`; Android SDK `~/Library/Android/sdk`.
- No adb. No Play/store actions in this phase. Nothing deploys in this phase.
- All existing suites must stay green: `python3 -m pytest -q` (repo root), `npm test`, `ruff check server tests`, plus (new) the watch Gradle gate.
- Honest-degradation doctrine: no mocks/stubs fabricating success; missing credentials/deps → explicit 503s (follow `server/routers/voice.py`).
- Env vars are `MINDSHIFT_*`-prefixed. Rename map (Gauge → here):
  `GAUGE_FIRESTORE_PROJECT→MINDSHIFT_FIRESTORE_PROJECT`, `GAUGE_CAPTURE_BUCKET→MINDSHIFT_CAPTURE_BUCKET`, `GAUGE_ALLOW_LEGACY_ACCOUNT→MINDSHIFT_ALLOW_LEGACY_ACCOUNT`, `GAUGE_STT→MINDSHIFT_WATCH_STT`, `GAUGE_MODEL→` (drop — use this repo's existing LLM config via `llm_client`), `GAUGE_FIREBASE_PROJECT→` (drop — reuse `FIREBASE_PROJECT_ID` in `server/auth.py`), `GAUGE_ALLOWED_ORIGINS→` (drop — main.py CORS governs).
- **Rename map ("episode" collision — LOCKED):** Gauge `Episode` → `LiveSession`; HTTP `/episodes*` → `/live-sessions*`; WS `/ws/episode/{id}` → `/ws/live-session/{id}`; WS frame `"episode_saved"` → `"live_session_saved"` and its `episode_id` field → `live_session_id`; `EpisodeStore` → `LiveSessionStore`; Firestore collection `episodes` → `live_sessions` (data migration is a Phase 2 deploy step, NOT here); `post_episode.py` → `post_session.py`; `analyze_episode()` → `analyze_live_session()`; `MAX_EPISODE_PCM_BYTES` → `MAX_LIVE_SESSION_PCM_BYTES`. Everything else (Capture, Account, Group, Pairing, telemetry, all other collection names) keeps its name so existing Firestore data reads without migration.
- Ported module docstrings keep Gauge's content but each ported file gets a first line: `# Ported from gauge@2157433 <original path>; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md`.
- Gauge's vendored `server/engine/` is NEVER copied. Ported code imports THIS repo's flat modules (`import speaker_id`, `import whisper_transcriber`, …) — `server/` is already on `sys.path` for tests via existing conftest.

## File Structure (end state)

```
apps/watch/                     # Track A — standalone Gradle build
  settings.gradle.kts           # NEW: shared + wearApp only
  build.gradle.kts              # copied (plugins apply false; google-services line REMOVED)
  gradle.properties             # copied verbatim (3 lines)
  gradle/{wrapper/*, libs.versions.toml}
  gradlew, gradlew.bat
  wearApp/                      # copied verbatim from $GAUGE/wearApp (minus build/)
  shared/                       # copied verbatim from $GAUGE/shared (minus build/)
  local.properties              # recreated, gitignored
  keystore.properties           # recreated, gitignored
  PORTED_FROM_GAUGE.md          # provenance note
server/watch/                   # Track B — ported backend package
  __init__.py
  config.py                     # Settings w/ MINDSHIFT_* envs
  models.py                     # wire/storage models (Episode→LiveSession)
  store.py                      # LiveSessionStore Protocol + Memory + Firestore
  pairing_store.py, telemetry_store.py, blobs.py
  auth.py                       # Principal/verifiers/ChainedTokenVerifier/deps
  vectors.py, nudge_policy.py, aggregates.py
  diarize.py                    # adapted to this repo's speaker_id v2
  post_session.py               # was post_episode.py; imports this repo's engine
  capture_fixtures.py
  routers/{__init__.py, rest.py, groups.py, captures.py, pairing.py, telemetry.py, ws.py, live_sessions.py}
  testing.py                    # create_watch_test_app(...) — replaces Gauge's create_app in tests
server/tests/watch/             # ported tests (~380; engine tests NOT ported)
  __init__.py, conftest.py, test_*.py
```

---

### Task A1: Import the watch into `apps/watch/` and prove a standalone build

**Files:**
- Create: `apps/watch/settings.gradle.kts`, `apps/watch/PORTED_FROM_GAUGE.md`, `apps/watch/local.properties`, `apps/watch/keystore.properties`
- Copy from `$GAUGE`: `wearApp/`, `shared/`, `build.gradle.kts`, `gradle.properties`, `gradle/` (wrapper + `libs.versions.toml`), `gradlew`, `gradlew.bat`
- Modify: root `.gitignore`

**Interfaces:**
- Produces: a Gradle build at `apps/watch/` where `./gradlew :wearApp:assembleDebug` succeeds. Task A2 and the CI job consume it.

- [ ] **Step 1: Copy the survivors (exclude build outputs and androidApp entirely)**

```bash
mkdir -p apps/watch
rsync -a --exclude 'build/' --exclude '.gradle/' "$GAUGE/wearApp/" apps/watch/wearApp/
rsync -a --exclude 'build/' --exclude '.gradle/' "$GAUGE/shared/"  apps/watch/shared/
cp "$GAUGE/build.gradle.kts" "$GAUGE/gradle.properties" "$GAUGE/gradlew" "$GAUGE/gradlew.bat" apps/watch/
rsync -a "$GAUGE/gradle/" apps/watch/gradle/
chmod +x apps/watch/gradlew
```

- [ ] **Step 2: Write the new `apps/watch/settings.gradle.kts`** (no `:androidApp`):

```kotlin
pluginManagement {
    repositories { gradlePluginPortal(); google(); mavenCentral() }
}
dependencyResolutionManagement {
    repositories { google(); mavenCentral() }
}
rootProject.name = "mindshift-watch"
include(":shared")
include(":wearApp")
```

- [ ] **Step 3: Remove the androidApp-only google-services plugin line** from `apps/watch/build.gradle.kts` (wearApp has no Firebase). Delete only that one `alias(...google.services...) apply false` line; leave the rest verbatim.

- [ ] **Step 4: Recreate gitignored local files**

`apps/watch/local.properties`: `sdk.dir=/Users/sophie.arborbot/Library/Android/sdk`
`apps/watch/keystore.properties`: copy content from `$GAUGE/keystore.properties` verbatim (points at `~/.config/gauge/gauge-upload.jks` — the wear package's permanent Play upload key; deliberately unchanged).

- [ ] **Step 5: Extend root `.gitignore`** with:

```
apps/watch/.gradle/
apps/watch/**/build/
apps/watch/local.properties
apps/watch/keystore.properties
```

- [ ] **Step 6: Write `apps/watch/PORTED_FROM_GAUGE.md`** — state: imported from `sagearbor/gauge` @ `2157433` on 2026-08-15 as a clean copy (decision: no history import; the archived Gauge repo keeps history); list what was left behind (androidApp, webApp, server, vendored engine) and why; note the keystore/applicationId facts (package `com.sagearbor.gauge.wear` is permanent; display branding changes in Phase 3).

- [ ] **Step 7: Verify module graph then build — FOREGROUND, timeout 600000**

```bash
cd apps/watch && JAVA_HOME=/Users/sophie.arborbot/jdk17/Contents/Home ./gradlew -q projects
JAVA_HOME=/Users/sophie.arborbot/jdk17/Contents/Home ./gradlew :wearApp:assembleDebug
```
Expected: projects lists `:shared` + `:wearApp` only; assembleDebug BUILD SUCCESSFUL. If unresolved-reference errors appear in untouched files: `rm -rf apps/watch/{wearApp,shared}/build` and rerun (known stale-cache trap).

- [ ] **Step 8: Commit** — `git add apps/watch .gitignore && git commit -m "feat(watch): import Gauge wearApp + shared KMP module as standalone apps/watch build"`

### Task A2: Watch test/lint gates green + CI job + gradle hook

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `.claude/hooks/block-background-gradle.sh` (copy from `$GAUGE/.claude/hooks/`), merge its `PreToolUse` Bash entry into this repo's `.claude/settings.json` (do NOT port block-monitor.sh — this repo uses Monitor legitimately)

**Interfaces:**
- Consumes: Task A1's build.
- Produces: `watch` CI job; local gates for later watch tasks.

- [ ] **Step 1: Run the full watch gate locally — FOREGROUND, timeout 600000**

```bash
cd apps/watch && JAVA_HOME=/Users/sophie.arborbot/jdk17/Contents/Home ./gradlew :shared:jvmTest :wearApp:testDebugUnitTest :wearApp:lintDebug
```
Expected: BUILD SUCCESSFUL; shared 98 tests, wearApp 342 tests, 0 lint errors. Fix nothing silently — any failure is investigated (likely stale cache or missing local.properties).

- [ ] **Step 2: Add the CI job** to `.github/workflows/ci.yml` (path-filtered so mobile/backend PRs aren't slowed; also runs when the workflow itself changes):

```yaml
  watch:
    name: Watch (gradle)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - id: changed
        uses: dorny/paths-filter@v3
        with:
          filters: |
            watch:
              - 'apps/watch/**'
              - '.github/workflows/ci.yml'
      - if: steps.changed.outputs.watch == 'true'
        uses: actions/setup-java@v4
        with: { distribution: temurin, java-version: "17" }
      - if: steps.changed.outputs.watch == 'true'
        uses: gradle/actions/setup-gradle@v4
      - if: steps.changed.outputs.watch == 'true'
        working-directory: apps/watch
        run: ./gradlew :shared:jvmTest :wearApp:testDebugUnitTest :wearApp:assembleDebug :wearApp:lintDebug
```

- [ ] **Step 3: Port the hook.** Copy `block-background-gradle.sh` into `.claude/hooks/`, `chmod +x`, and add its `PreToolUse` Bash matcher entry to `.claude/settings.json` (create the hooks key if absent, preserving any existing entries).

- [ ] **Step 4: Commit** — `git commit -m "ci(watch): gradle test/assemble/lint job + backgrounded-gradle guard hook"`

---

### Task B1: `server/watch/` skeleton — config + models + test scaffolding

**Files:**
- Create: `server/watch/__init__.py` (empty), `server/watch/config.py`, `server/watch/models.py`, `server/tests/watch/__init__.py` (empty), `server/tests/watch/conftest.py`, `server/tests/watch/test_models.py`
- Modify: `requirements.txt` (add `google-cloud-firestore>=2.16,<3` with a comment: watch-domain entities; lazy-imported, unset env → in-memory)

**Interfaces:**
- Produces: `server.watch.models` — all pydantic models with `LiveSession` (was Episode; same fields incl. `pcm_b64: str = Field(default="", exclude=True)`, `status` literal unchanged), `Capture`, `Account`, `Group`, `Pairing`, `VectorEvent`, `NudgeEvent`, `ConsentRecord`, `TelemetryBatch`…; `server.watch.config.Settings` reading the MINDSHIFT_* envs (preserve Gauge's default VALUES, only names change). Every later B task imports these.

- [ ] **Step 1: Port the test first.** Copy `$GAUGE/server/tests/test_models.py` → `server/tests/watch/test_models.py`; apply the rename map (Episode→LiveSession) and fix imports to `from watch import models` (the existing root conftest puts `server/` on sys.path, so `watch` is importable as a package). Write `server/tests/watch/conftest.py`:

```python
import os
os.environ.setdefault("MINDSHIFT_WATCH_STT", "none")  # WS tests must never invoke real Whisper
```

- [ ] **Step 2: Run to verify it fails** — `python3 -m pytest server/tests/watch/test_models.py -q` → ImportError (no module `watch.models`).
- [ ] **Step 3: Port `models.py` + `config.py`** from `$GAUGE/server/{models,config}.py` applying the rename + env maps. Drop `GAUGE_MODEL`/`GAUGE_FIREBASE_PROJECT`/`GAUGE_ALLOWED_ORIGINS` fields from Settings entirely.
- [ ] **Step 4: Run to verify pass** — same command → all pass.
- [ ] **Step 5: Full-suite check** — `python3 -m pytest -q` (existing tests unaffected) and `ruff check server tests`.
- [ ] **Step 6: Commit** — `git commit -m "feat(watch-api): port models + config (episode→live_session rename, MINDSHIFT_* envs)"`

### Task B2: Stores — live-session, pairing, telemetry, blobs

**Files:**
- Create: `server/watch/store.py`, `server/watch/pairing_store.py`, `server/watch/telemetry_store.py`, `server/watch/blobs.py`
- Test: `server/tests/watch/{test_store.py,test_pairing_store.py,test_telemetry_store.py,test_blobs.py}` (ported)

**Interfaces:**
- Consumes: `watch.models`.
- Produces: `LiveSessionStore` Protocol (~30 methods, renamed from EpisodeStore; atomic-mutator seams `update_group_atomically` etc. kept), `MemoryLiveSessionStore`, `FirestoreLiveSessionStore` (collection `"live_sessions"`), `get_store()`; same trio for pairing (`hash_secret`), telemetry, blobs (`GcsBlobStore` on `MINDSHIFT_CAPTURE_BUCKET`). Lazy `google.cloud.*` imports preserved (tests assert non-import).

- [ ] **Step 1: Port the four test files** (rename map + `from watch import store` etc.). Run: fail with ImportError.
- [ ] **Step 2: Port the four modules** applying rename+env maps. Keep Gauge's known wart (store raises HTTPException inside mutators) — parity beats refactor here; note it in the module docstring as inherited.
- [ ] **Step 3: Run** `python3 -m pytest server/tests/watch/ -q` → pass. Verify the lazy-import assertions still hold (they prove Firestore isn't touched).
- [ ] **Step 4: Full suite + ruff; commit** — `"feat(watch-api): port stores (live_session/pairing/telemetry/blobs) with memory+Firestore impls"`

### Task B3: Auth — principal ladder + Firebase + device tokens + chained verifier

**Files:**
- Create: `server/watch/auth.py`
- Test: `server/tests/watch/{test_auth.py,test_auth_routes.py}` (ported; `StubVerifier` comes with test_auth.py and later tasks import it from there, exactly as in Gauge)

**Interfaces:**
- Consumes: `watch.pairing_store` (`hash_secret`, device-token lookups), this repo's `auth.init_firebase` + `FIREBASE_PROJECT_ID`.
- Produces: `Principal`, `resolve_principal`, `resolve_ws_principal`, `FirebaseTokenVerifier` (adapted: uses `server/auth.py`'s `init_firebase()` then `fb_auth.verify_id_token(token)` keeping the `{"sub","email"}` return shape), `DeviceTokenVerifier` (401 on clean miss / **503 on store exception** — locked cross-repo contract: the watch clears its token on 401), `ChainedTokenVerifier` (Firebase first; only InvalidToken advances the chain), `make_auth_dependency`, `require_full_auth`, `ensure_account`, `get_full_verifier`. Legacy mode gated on `MINDSHIFT_ALLOW_LEGACY_ACCOUNT`.

- [ ] **Step 1: Port tests; run; fail.**  
- [ ] **Step 2: Port + adapt `auth.py`** per the Produces block. MindShift's existing `get_current_uid` and every existing route are untouched.
- [ ] **Step 3: Run watch tests → pass; full suite + ruff.**
- [ ] **Step 4: Commit** — `"feat(watch-api): port auth — Firebase+device-token chained verification"`

### Task B4: Pure engines — vectors, nudge policy, aggregates

**Files:**
- Create: `server/watch/{vectors.py,nudge_policy.py,aggregates.py}`
- Test: `server/tests/watch/{test_vectors.py,test_nudge_policy.py,test_aggregates.py,test_group_standing.py,test_me_standing.py}` — the last two are router tests; port ONLY their pure-math cases here if separable, else defer whole files to B5/B6 (implementer judgment; do not port a failing router test into this task).

**Interfaces:**
- Consumes: `watch.models` (`VectorEvent`, `NudgeEvent`).
- Produces: `VectorEngine` (numpy DSP over 1 s PCM windows; baseline-relative ONLY — the bias guard is non-negotiable), `NudgePolicy` (two-channel hysteresis/cooldown), `member_standing`/`group_standing`/calm score. These are the mirror-contract counterparts of `apps/watch/shared/.../NudgeStateMachine.kt` — keep thresholds/semantics identical and keep the cross-reference KDoc/docstrings pointing at the NEW paths.

- [ ] Steps: port tests → fail → port modules (near-verbatim; only import paths change) → pass → full suite + ruff → commit `"feat(watch-api): port vector engine, nudge policy, aggregates (bias guard intact)"`

### Task B5: REST router — me/standing/claim/lookup/live-session reads/labels/share/settings/enroll

**Files:**
- Create: `server/watch/routers/__init__.py`, `server/watch/routers/rest.py`, `server/watch/testing.py`
- Test: `server/tests/watch/{test_rest_api.py,test_claim_legacy.py,test_accounts_lookup.py,test_me_standing.py,test_enroll_voice.py,test_speaker_profile.py}` (+ any deferred from B4)

**Interfaces:**
- Consumes: stores, auth, aggregates; this repo's `speaker_id` (v2) for `/enroll` voiceprints.
- Produces: `make_rest_router(store, auth_dep, strict_auth_dep, embedder=None) -> APIRouter` (Gauge's closure-factory signature preserved); paths renamed: `GET /live-sessions`, `GET/DELETE /live-sessions/{id}`, `POST /live-sessions/{id}/labels`, `POST /live-sessions/{id}/share`; unchanged: `/me`, `/me/standing`, `/me/claim-legacy`, `/accounts/lookup`, `/settings/vectors`, `/enroll`, `/enroll/voice`. **Also produces `server/watch/testing.py`:**

```python
"""Test-only app assembly — replaces Gauge's create_app() in ported tests."""
from fastapi import FastAPI

def create_watch_test_app(*, store=None, pairing_store=None, telemetry_store=None,
                          blobs=None, verifier=None, full_verifier=None,
                          transcriber=None, llm=None, diarizer=None, embedder=None,
                          allow_legacy=False) -> FastAPI:
    ...  # builds ONLY the watch routers onto a bare FastAPI app, memory stores by default
```
Ported test files swap `create_app(...)` → `create_watch_test_app(...)` with the same keyword style; each router task extends `testing.py` to mount its router.

- [ ] Steps: port tests (rename map; `/enroll` cases adapt to speaker_id v2 — read `server/speaker_id.py` FIRST and use its real v2 API; if v2 signatures make a Gauge test meaningless, rewrite the test to assert the v2-equivalent behavior, never delete silently) → fail → port `rest.py` + write `testing.py` → pass → full suite + ruff → commit `"feat(watch-api): port REST router (live-session reads, standing, claim, settings, enroll on speaker_id v2)"`

### Task B6: Groups router

**Files:** Create `server/watch/routers/groups.py`; Test `server/tests/watch/{test_groups_api.py,test_group_standing.py}`
**Interfaces:** Consumes stores/auth/aggregates; Produces `make_groups_router(store, full_auth_dep) -> APIRouter` — paths unchanged (`/groups*`), all-consent 409 semantics preserved.
- [ ] Steps: port tests → fail → port router + mount in `testing.py` → pass → full suite + ruff → commit `"feat(watch-api): port groups/couples router"`

### Task B7: Captures router + fixtures helper

**Files:** Create `server/watch/routers/captures.py`, `server/watch/capture_fixtures.py`; Test `server/tests/watch/{test_captures_api.py,test_capture_fixtures.py,test_export_capture_fixtures.py}`
**Interfaces:** Consumes stores/auth/blobs; Produces `make_captures_router(store, blobs, full_auth_dep) -> APIRouter` — paths unchanged (`/captures*`); gzip `inflate_capped` (413 bomb / 422 bad-gzip) and caps (`MAX_CAPTURE_BYTES=20_000_000`, `MAX_CAPTURE_SECONDS=900`, `MAX_LABELS_BYTES=100_000`) preserved.
- [ ] Steps: port tests → fail → port → mount → pass → full suite + ruff → commit `"feat(watch-api): port captures router (retro-capture upload/download/labels)"`

### Task B8: Pairing router

**Files:** Create `server/watch/routers/pairing.py`; Test `server/tests/watch/test_pairing_api.py`
**Interfaces:** Consumes pairing_store/auth; Produces `make_pairing_router(pairing_store, full_auth_dep) -> APIRouter` — `/me/pair/start|status|claim` unchanged incl. `MAX_TOKEN_READS=5`, `PAIRING_TTL_MINUTES=10`, claim lockout (15 attempts / 24 h sliding → 429), always-200 status.
- [ ] Steps: port tests → fail → port → mount → pass → full suite + ruff → commit `"feat(watch-api): port device-pairing router (short-code flow, token minting)"`

### Task B9: Telemetry router

**Files:** Create `server/watch/routers/telemetry.py`; Test `server/tests/watch/test_telemetry_api.py`
**Interfaces:** Consumes telemetry_store; Produces `make_telemetry_router(telemetry_store) -> APIRouter` — `POST/GET /telemetry`, deliberately unauthenticated (the no-adb debug channel). Add a module-docstring note: **inherited risk, token-gating GET is a queued owner decision** (do not change behavior here).
- [ ] Steps: port tests → fail → port → mount → pass → full suite + ruff → commit `"feat(watch-api): port telemetry router (device beacon channel)"`

### Task B10: Analysis pipeline on THIS engine — post_session + diarize adapter  ⚠ riskiest task; review hard

**Files:**
- Create: `server/watch/post_session.py` (from `$GAUGE/server/post_episode.py`), `server/watch/diarize.py` (from `$GAUGE/server/diarize.py`)
- Test: `server/tests/watch/{test_post_session.py,test_post_session_diarization.py,test_diarize.py}` (renamed from test_post_episode*)

**Interfaces:**
- Consumes: THIS repo's `whisper_transcriber` (shared-model cache), `speaker_id` (v2 per-sample profiles), `llm_client`/`llm_cache`; `watch.models`, `watch.vectors`.
- Produces: `analyze_live_session(session, store, transcriber, llm, diarizer) -> LiveSession` (status transitions `captured→analyzed|transcription_unavailable` preserved; re-analysis idempotency via `TRANSCRIPT_EVENT_DETAIL_PREFIX` filter preserved); `TranscriptionService` Protocol + `WhisperTranscriptionService` (delegating to this repo's module) + `NullTranscriptionService`; `EmbeddingDiarizationService` adapted to speaker_id **v2** + `NullDiarizationService`.
- **Rules:** every `from engine import X` / `engine.X` → this repo's flat module `X`. Where Gauge's code used speaker_id v1 APIs, adapt to v2 by reading `server/speaker_id.py` — do NOT vendor v1 shims. `MINDSHIFT_WATCH_STT=none` (conftest) must keep the suite Whisper-free.
- [ ] Steps: port tests → fail → port+adapt modules → pass → full suite + ruff → commit `"feat(watch-api): live-session analysis pipeline on canonical engine (speaker_id v2, shared whisper)"`

### Task B11: WS ingest — `/ws/live-session/{id}`

**Files:** Create `server/watch/routers/ws.py` + `server/watch/routers/live_sessions.py` (the `POST /live-sessions/{id}/analyze` route from Gauge's main.py); Test `server/tests/watch/{test_ws_ingest.py,test_main.py→test_watch_wiring.py}`
**Interfaces:**
- Consumes: auth (`resolve_ws_principal`), store, `VectorEngine`, `NudgePolicy`, `post_session.analyze_live_session`.
- Produces: `make_ws_router(...) -> APIRouter` with `WEBSOCKET /ws/live-session/{id}?account=&token=`; protocol per Gauge with renames: final frame `{"type":"live_session_saved","live_session_id":...,"status":...}`; binary 32000-byte 1 s PCM16 windows; `{"type":"hr"}`, `{"type":"end"}`, error frames unchanged; close 1008 pre-accept on auth failure; abrupt-disconnect `finally:` persists `not_analyzed`; `MAX_LIVE_SESSION_PCM_BYTES=57_600_000` cap. `make_live_sessions_router(store, auth_dep, ...)` for the analyze route at `POST /live-sessions/{id}/analyze`.
- [ ] Steps: port tests (rename frames/paths) → fail → port → mount in `testing.py` → pass → full suite + ruff → commit `"feat(watch-api): live-session WS ingest + re-analyze route"`

### Task B12: Production wiring — main.py include + env assembly + `/health`

**Files:**
- Create: `server/watch/app.py` — `build_watch_routers() -> list[APIRouter]`: assembles prod deps from `watch.config.Settings` env (get_store/get_pairing_store/get_telemetry_store/get_blob_store/get_full_verifier; transcriber+llm+diarizer from this repo's modules per `MINDSHIFT_WATCH_STT`)
- Modify: `server/main.py` — one block only (follow the voice.py include pattern):

```python
# Watch domain (ported from Gauge — docs/plans/2026-08-15-unification-*.md):
from watch.app import build_watch_routers
for _r in build_watch_routers():
    app.include_router(_r)
```
plus add a `GET /health` alias next to the existing `/healthz` (Cloud Run's frontend intercepts the literal `/healthz` on run.app URLs — Gauge learned this the hard way; the watch pings `/health`).
- Test: `server/tests/watch/test_prod_wiring.py` (NEW — not from Gauge):

```python
def test_watch_routes_mounted_on_main_app():
    import main
    paths = {r.path for r in main.app.routes}
    assert "/me/pair/claim" in paths and "/telemetry" in paths and "/health" in paths
    assert "/live-sessions" in paths and "/analyze" in paths  # both domains coexist

def test_no_route_collisions():
    import main
    from collections import Counter
    dupes = {p: c for p, c in Counter(
        (r.path, ",".join(sorted(getattr(r, "methods", []) or ["WS"]))) for r in main.app.routes
    ).items() if c > 1}
    assert not dupes, f"colliding routes: {dupes}"
```
- [ ] Steps: write the two tests → fail (routes absent) → implement `app.py` + main.py block + `/health` → pass → **entire combined suite** `python3 -m pytest -q` + `npm test` + `ruff` → commit `"feat(watch-api): mount watch domain on the unified app (+/health alias)"`

### Task B13: OpenAPI contract seed — TS types

**Files:** Create `scripts/gen_api_types.sh`, `apps/mobile/src/api/generated/openapi.d.ts` (generated, committed); Modify `package.json` (devDep `openapi-typescript`, script `"gen:api"`)
**Interfaces:** Consumes the unified app's OpenAPI (`python3 -c "import json,main;print(json.dumps(main.app.openapi()))"`); Produces committed TS types for new/ported endpoints — mobile adopts incrementally starting Phase 3; **Kotlin generation deliberately deferred** to the Phase 2/3 watch release cycle (watch's hand-written `WireModels.kt` keeps the mirror contract until then — record this in the script header).
- [ ] Steps: write `gen_api_types.sh` (emit spec → npx openapi-typescript → output path) → run → commit generated file + script → full CI-equivalent gates → commit `"feat(api-contract): OpenAPI emission + generated TS types (seed)"`

### Task A3: Watch client speaks the renamed protocol

**Files:** Modify `apps/watch/wearApp/src/main/kotlin/app/gauge/wear/net/EpisodeWsClient.kt`, `service/SentinelService.kt` (WS path builder at ~:290), `shared/src/commonMain/kotlin/app/gauge/shared/WireModels.kt` (saved-frame model), plus their tests.
**Interfaces:** Consumes B11's protocol. Produces: watch builds `wss://…/ws/live-session/{id}`, parses `"live_session_saved"` / `live_session_id`. `GAUGE_API_BASE` value does NOT change here (URL flip is Phase 2). All other watch calls (`/me/standing`, `/groups*`, `/captures*`, `/me/claim-legacy`, `/telemetry`) are unchanged by design — verify by grep, not assumption.
- [ ] Steps: update tests to expect new path/frame → run gate (fail) → apply renames → full watch gate (`:shared:jvmTest :wearApp:testDebugUnitTest :wearApp:lintDebug`, FOREGROUND timeout 600000) → commit `"feat(watch): speak live-session protocol (renamed WS path + saved frame)"`

### Task F1: Whole-branch final review + PR

- [ ] **Step 1:** Full gates one last time: `python3 -m pytest -q` && `npm test` && `ruff check server tests` && watch gradle gate (foreground, timeout 600000).
- [ ] **Step 2:** Dispatch an adversarial whole-branch review (superpowers:subagent-driven-development final-review stage) over `git diff main...feat/unification-phase1` with special attention to: the rename map's completeness (`grep -ri "episode" server/watch/ apps/watch/` — every survivor must be justified), the 401-vs-503 device-token contract, the speaker_id v2 adaptation, bias-guard preservation in vectors, and no behavior change to any pre-existing MindShift route.
- [ ] **Step 3:** Open PR titled `feat: Phase 1 unification — one repo, one engine (absorb Gauge watch + backend)`; body links the master plan; **merge with a merge commit (not squash)** after review passes.
- [ ] **Step 4:** Update owner's hlist + memory per global conventions.

## Phase 2 hand-forward notes (recorded here so they aren't lost; NOT Phase 1 work)
- Firestore data migration: copy `episodes` collection docs → `live_sessions` (tiny personal dataset) before the watch vNext release; all other collections read as-is.
- Watch vNext: flip `GAUGE_API_BASE` buildConfig to the unified service URL; bump versionCode 10; `bundleRelease` with upload keystore; Play internal (browser flow — confirm device with owner first, or extend `scripts/play_publish.py`).
- Verify 2Gi/4CPU fits WS-ingest+engine; Gauge ran 4Gi.
- Pairing cold-start fast-follow: client timeout 10 s→30 s + one retry.
- Owner decision queue: keep-warm min-instances $ figure; token-gating `GET /telemetry`.
- Phase 3 "Pair a watch" screen (owner-requested 2026-08-15): add an **"Install watch app" button** that deep-links to the watch app's Play listing (`https://play.google.com/store/apps/details?id=com.sagearbor.gauge.wear`) — phone Play then remote-installs to the watch (standard Wear OS 3+ flow; no on-watch store browsing). Note: while on internal testing, testers need the opt-in link first.
