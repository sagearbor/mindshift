# AGENTS.md

## Purpose
Guide AI agents contributing to MindShift (see [PRD.md](PRD.md) for the product spec).

> **ACTIVE MAJOR EFFORT (2026-08-15):** MindShift is absorbing the sibling product Gauge
> (its Wear OS watch app, backend routes, and users) — one repo, one backend, one brand.
> Read [docs/plans/2026-08-15-unification-mindshift-absorbs-gauge.md](docs/plans/2026-08-15-unification-mindshift-absorbs-gauge.md)
> before substantial work; it carries locked decisions and transplanted operational
> knowledge (watch builds, Play uploads, telemetry debugging).

> **FRESH CLONE / NEW MACHINE (2026-08-24):** if this is a newly-cloned checkout
> (e.g. after a machine switch), read
> [docs/plans/2026-08-24-mac-transition-and-poker6-status.md](docs/plans/2026-08-24-mac-transition-and-poker6-status.md)
> and, for the realtime/on-device work that followed (poker6 is resolved there):
> [docs/plans/2026-08-24-realtime-three-tracks-handoff.md](docs/plans/2026-08-24-realtime-three-tracks-handoff.md)
> first — it lists what `git clone` doesn't bring over (`.env`, Python venvs,
> auth for `gcloud`/`gh`/`eas`) and the current status of the in-progress
> poker6 6-speaker diarization investigation (what's shipped, what's research-only,
> and the recommended next step).

## Layout
- `apps/mobile/` — the active Expo (React Native + Web) app. UI in `src/components`,
  screens in `src/screens`, state in `src/store` (Zustand), API/WS clients in `src/api`
  and `src/hooks`.
- `server/` — FastAPI backend + model-agnostic `LLMClient`. Tests in `server/tests`.
- `tests/` — top-level integration/contract tests (run together with `server/` via `pytest`).

## Coding Rules
- Dual stack: TypeScript (frontend) + Python (backend).
- Real LLM calls go through `server/llm_client.py`; cache via `server/llm_cache.py`.
- Don't introduce mocks/stubs that fabricate success. Gate external services on
  credentials and report unavailability explicitly rather than returning fake data.
- Keep commit messages faithful to what actually landed.
- Avoid storing user data or secrets.

## Testing
- `pytest` — backend (runs `server/` + `tests/` from the repo root).
- `npm test` — frontend Jest (jest-expo), delegated to `apps/mobile`.
- TDD: write the failing test first.

## Deploying
- Mobile OTA updates: always `./scripts/ota_publish.sh "message"`, never raw
  `eas update` — the raw command bundles JS without the production
  `EXPO_PUBLIC_*` env baked in, which shipped a broken build once (client
  fell back to `localhost:8000`). See the script's header comment.
  `OTA_DRY_RUN=1` proves the plumbing without publishing; `OTA_CHANNEL=preview`
  targets the internal-distribution (`eas build --profile preview`) builds.
- Runtime version = `expo.version` (`runtimeVersion.policy: appVersion`).
  Bump `expo.version` (and `android.versionCode`) whenever a native module
  is added/removed, BEFORE the next OTA — otherwise the update can land JS
  on a binary that lacks the native code. (1.17.0 = first runtime with the
  on-device modules; the 1.16.0 Play build stays on its last 1.16.0 OTA.)

## Secrets in process listings
- `scripts/deploy_cloudrun.sh` (and any `gcloud run deploy --set-env-vars`
  call) passes `ANTHROPIC_API_KEY`/`DEEPGRAM_API_KEY` as command-line
  arguments — `ps aux`/`ps -ef` show these in full while a deploy is
  running. When checking whether a deploy is still running, use `ps -o
  pid,etime,stat -p <pid>` or `pgrep -f <pattern>` instead — never a bare
  `ps aux`/`ps aux | grep`. (2026-08-19: this leaked both keys into a
  session transcript.)

## Exploratory / scratch work (`tmp/`)
- Before any bulk cleanup (`rm -f`, `rm -rf`, overwriting a working dir) in a
  scratch investigation directory holding real generated artifacts
  (downloads, API call outputs, scored results — anything costly or
  impossible to exactly regenerate): `git init` a throwaway LOCAL-ONLY repo
  in that directory first and commit after each milestone. Never push it
  anywhere. This is cheap insurance — an agent's overly-broad `rm -f`
  destroyed ~4 hours of real transcription-comparison data on 2026-08-18
  with no way to recover it.

## If you hit low disk space mid-task
- STOP and report the blocker. Do not delete anything you did not create
  in your own session's scratch area (your own worktree, your own `tmp/`
  subdirectory) to work around it — not a shared cache
  (`~/Library/Caches/...`), not "old log files," not anything else on this
  shared machine, even if it looks obviously safe/regenerable. Get
  explicit confirmation first. (2026-08-22: an agent `rm -rf`'d a shared
  Playwright browser cache under disk pressure with no direction naming
  that target — exactly the unauthorized-bulk-cleanup pattern the rule
  above already warns about, just triggered by a different kind of
  pressure.)
