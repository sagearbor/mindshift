# MindShift Unification — Absorbing Gauge (Master Handoff)

**Status:** Approved by owner 2026-08-15. This document is the complete context a fresh
Claude session in THIS repo needs to execute the unification. It supersedes any older
"keep them separate / sync the engine" guidance found elsewhere.

**Owner's decisive constraint (their words):** *"I'm horrible at maintaining two things."*
Solo founder; any plan relying on cross-repo sync discipline is rejected. Structure must
make drift impossible, not merely discouraged.

---

## 1. The decisions — all LOCKED, do not relitigate

| Decision | Ruling |
|---|---|
| Merge direction | **MindShift absorbs Gauge.** One repo (this one), one backend, one phone app. |
| Brand | **MindShift** everywhere. New icon direction: MindShift identity, not a gauge/dial. Play *package names never change* (they're permanent) — only display names, icons, copy. |
| Phone + web codebase | **`apps/mobile/` (Expo/React Native/TS) — permanently.** It serves Android + iOS + web (react-native-web). Gauge's Kotlin phone app and React webApp both retire after their unique flows are ported. |
| Watch apps | **Thin native shells, by design — but "thin" means thin PRODUCT surface, not a dumb terminal.** Wear OS = Kotlin/Compose (exists, moves here). Future Apple Watch = SwiftUI (nothing else can render on watchOS). The latency-critical loop (mic → on-watch detection → nudge state machine → haptic) runs ENTIRELY on the wrist with no network round-trip, works offline, and MUST STAY on-device — that loop lives in `shared/` + the sentinel service. What the watch does NOT carry: charts, analysis UI, account management, transcription, diarization, LLM. Server-derived nudge vectors are advisory enrichment (async, channel B), never load-bearing for the core buzz. **Never "simplify" by moving trigger detection server-side** — round-trips, cold starts, and dead zones would kill the product's defining feature. |
| Cross-watch logic sharing | Gauge's `shared/` KMP module (sentinel state machine, nudge escalation, wire models) is the watch-domain logic core. It compiles to Kotlin/Native watchOS targets later — same brains, SwiftUI skin. Keep it healthy; never fold its logic into UI code. |
| API contract | Generate client types from the FastAPI OpenAPI spec → TypeScript (mobile) + Kotlin (watch). The contract, not a shared UI framework, is what keeps clients in lockstep. |
| iOS sequencing | iPhone (Expo config + EAS) soon after merge settles; Apple Watch is a separate later project. Owner knows zero iOS — EAS handles signing/builds, which is why the Expo path was chosen. |
| Datastores | Coexistence is fine: Firestore (Gauge's entities) + SQLite-ephemeral/GCS (MindShift's) behind routers in one service. Consolidate gradually later; do NOT force it during the merge. |

Rationale in depth (stack comparison, engine-fork measurements, rejected options):
`../gauge/tmp/gauge-vs-mindshift-merge-report.html` (v2). Key facts that de-risk everything:
**both products already share Firebase project `arborfam-hub`, the same auth providers, and
therefore the same uids. There is no user migration anywhere in this plan.**

---

## 2. What Gauge is (context for a session that has never seen it)

Gauge is the sibling product: an always-on **Wear OS tone/behavior coach** — watch mic
sentinel detects the wearer's own escalation live, delivers haptic nudges, streams episodes
to its backend. Repo: `../gauge` — a single clean checkout on `main` (all branches merged
and deleted, local and remote, on 2026-08-15; no worktrees remain; working tree clean).

**Live inventory (2026-08-15):**

| Asset | Version / location | Fate |
|---|---|---|
| Wear OS app | 0.3.1 (vc9), package `com.sagearbor.gauge.wear`, Play internal testing | **SURVIVES** → moves to `apps/watch/` |
| `shared/` KMP module | sentinel state machine, wire models, nudge logic (98 tests) | **SURVIVES** → moves with watch |
| Kotlin phone app | 0.2.4 (vc8), `com.sagearbor.gauge.phone`, Play internal testing | RETIRES (Phase 3) |
| React webApp | 0.3.4 at `gauge-dashboard.web.app` (~300 tests) | RETIRES (Phase 3) |
| Backend | Cloud Run `gauge-api` (project `arborfam-hub`, us-central1), FastAPI + **Firestore**; 475+ tests | Routes/models/tests **SURVIVE** → port into `server/` here; service retires (Phase 2) |
| Vendored engine copy | `gauge/server/engine/` (8 files, forked from this repo 2026-07-31, zero commits since) | DELETED — this repo's engine is canonical |
| Telemetry channel | `GET/POST https://gauge-api-664594784582.us-central1.run.app/telemetry` — devices self-report crashes/errors; THE tool for on-device debugging (no adb) | **SURVIVES** → port route + keep the pattern |

**Gauge domain features the unified server must keep serving** (all built, reviewed, tested):
- Device pairing by short code: watch `POST /me/pair/start` → 6-char code → user enters it in a client → `POST /me/pair/claim` → watch polls `GET /me/pair/status` → device token minted. Codes TTL ~10 min.
- Episodes (watch capture units) + WS ingest of live audio windows; retro-capture uploads; claim-legacy (`POST /me/claim-legacy` moves pre-sign-in episodes onto a uid).
- Couples: groups/pairs, invites, standing (`GET /me/standing`, `GET /groups/{id}/standing`).
- Per-vector nudge settings (haptics on/off, sensitivity, channel per vector e.g. `hr_spike`, `airtime`).
- **Naming collision:** Gauge "episode" = a capture unit; MindShift "episode" = silence-gap
  segment within one recording (`server/episodes.py`). Resolve at the door: rename Gauge's
  concept during the port (suggestion: `captures` or `live_sessions`). Do not let both
  meanings coexist under one name.

**Nudge signal architecture today (verified in code 2026-08-15 — preserve this shape):**

| Signal | On-watch (offline, channel A) | Cloud live (WS episode, channel B) |
|---|---|---|
| Yelling (dB over the wearer's OWN enrolled baseline; +6/+10/+14 → L1/2/3) | ✅ THE core nudge — `shared/NudgeStateMachine`, a documented mirror of `server/nudge_policy.py` (same thresholds/semantics, cross-referenced tests). Tiny, deliberate duplication — keep the mirror contract. | ✅ `yelling` vector (`server/vectors.py` VectorEngine) |
| Aggressive tone (F0 pitch ≥1.3× own baseline) | deliberately deferred | ✅ `aggressive_tone` vector — live |
| HR spike | meter only | ✅ `hr_spike` vector |
| Interrupting / airtime | — | built, partially wired (needs live diarization turns — **MindShift's `diarize_local` is the missing piece**) |
| Words / semantic anger | — | — (post-hoc only; MindShift's transcription+LLM pipeline is the future source of live semantic vectors down channel B) |

Bias guard (non-negotiable, from Gauge's spec): every vector measures the wearer against
their OWN enrollment baseline or an episode running median — never an absolute threshold.
The engine reports physics, never judgments. **Owner's product intent:** nudging on
tone/anger including WORDS is the point of the product — the semantic live-vector phase
(MindShift engine behind the watch's WS, semantic events on channel B, few-seconds lag,
acoustic reflex stays instant) is the marquee post-merge feature. Plan it after Phase 3.

**NORTH STAR (owner, 2026-08-15): migrate nudge intelligence ONTO the watch over time —
eventually including on-device transcription and MindShift's voice/speaker analysis. The
on-wrist-intelligence watch is the intended differentiator and moat** (privacy: audio never
leaves the wrist — the eventual PHI/HIPAA story; latency; offline). This does NOT
contradict "thin shell" — the shell stays thin on product surface while vector producers
migrate down as watch hardware allows. The seam is the VectorEvent contract: the cloud
engine is the reference implementation; capabilities move on-device one vector at a time:
1. today — loudness-vs-own-baseline (shipped);
2. near-term — **on-watch speaker-ID** (ECAPA embeddings are small/cheap; "only nudge on
   the wearer's own voice" = highest-value single step, MindShift voiceprints on the wrist);
3. next — audio-direct tone/emotion classifiers (tiny SER models, NPU-friendly — "sounding
   angry" without transcription);
4. then — burst transcription around detected escalation + small distilled text
   classifiers (contempt, absolutes, escalation language);
5. eventually — continuous on-device ASR when watch silicon supports it (Apple already
   ships on-device dictation on Apple Watch; the trajectory is real).
Architectural commitments that keep this cheap: keep nudge policy + vector contract
device-agnostic (already true — watch mirrors server), and package on-device models in
portable formats (ONNX → LiteRT for Wear OS, → CoreML for watchOS) so each rung lands on
both wrists. Continuous full transcription on today's Wear OS hardware is a battery
non-starter — do not attempt rung 5 before the silicon exists; climb in order.

**Engine fork state (measured 2026-08-15):** `dynamics/prosody/word_metrics/llm_client/llm_cache`
byte-identical to this repo. Gauge's `speaker_id` is stuck at profile v1 (this repo is v2),
its `whisper_transcriber` lacks the shared-model cache, its `audio_pipeline` is a trimmed
copy, and it has **no `diarize_local`** (it has an older separate `server/diarize.py`).
Nothing in Gauge's copy is newer than this repo. Deleting it loses nothing.

---

## 3. The four phases

Each phase ends shippable. Run each as its own planned, reviewed effort (see §5 process).

### Phase 1 — One repo, one engine (~2–3 days)
1. Import the watch: `../gauge`'s `wearApp/` + `shared/` + Gradle wrapper/config into
   `apps/watch/` (history-preserving import if practical — `git subtree add` or
   filter-repo — else a clean copy with a provenance note). It builds standalone with its
   own `settings.gradle.kts`; do not entangle it with the Node/Python toolchains.
2. Port Gauge's backend: its routers (pairing, telemetry, episodes→renamed, claim, couples,
   nudge settings), Firestore access layer, and **its tests** into `server/` here as proper
   APIRouter modules (follow `server/routers/voice.py` as the pattern — do NOT graft onto
   the `main.py` monolith; this is also the start of factoring it).
3. Auth: both sides already verify Firebase ID tokens against `arborfam-hub`. Gauge adds a
   second credential type: **watch device tokens** (minted by pairing). Port its verifier so
   both `Authorization` forms work on the ported routes.
4. Delete nothing in Gauge yet — Gauge repo stays untouched until Phase 4.
5. Point the watch's audio path at THIS engine (`diarize_local`, `speaker_id` v2): the watch
   WS ingest handler being ported should call current modules, not the vendored copies.
6. CI: extend `.github/workflows/ci.yml` with a watch job (JDK 17 + Gradle
   `:wearApp:testDebugUnitTest :wearApp:assembleDebug :wearApp:lintDebug`). Combined
   backend suite = MindShift's tests + Gauge's ported tests, all green.
7. Contract generation seed: emit OpenAPI from FastAPI; generate TS types into
   `apps/mobile/src/api/generated/` and Kotlin wire types for the watch. Adopt
   incrementally — new/ported endpoints first.

### Phase 2 — One backend in production (~1–2 days)
1. Deploy the unified service (existing `scripts/deploy_cloudrun.sh`, service
   `mindshift-api`). Memory note: Gauge ran 4Gi; this service runs 2Gi/4CPU — verify the
   ported WS-ingest+engine path fits, bump if needed.
2. Watch release vNext: `GAUGE_API_BASE` → the unified service URL. Bump versionCode, build
   `bundleRelease`, upload to Play internal testing (see §4 for the exact release mechanics
   and traps).
3. Verify via telemetry (not assumption): watch app-start beacons, pairing round-trip
   (fresh code → claim → status), one live episode end-to-end.
4. Grace window for `gauge-api` (a week), then decommission. Note: it scales to zero and
   cold starts eat first requests — a known failure source (watch pairing timeouts). During
   the port consider the queued fast-follow: pairing client timeout 10s→30s + one retry.
5. Optional carry-over decision the owner deferred: keep-warm (min-instances or scheduled
   ping) for the unified service — raise it with the owner once, with the ~$ figure.

### Phase 3 — One phone app, one web (~3–5 days)
1. Port into the Expo app: **Pair a watch** (6-char code entry → claim endpoint; the web
   dashboard's Pair page is the reference), **Claim watch history** (button + returned
   counts), **nudge settings** (per-vector toggles/sensitivity, currently in the Kotlin
   phone app's Settings). Match this repo's screen/store/component conventions.
2. Retire: Gauge Kotlin phone app (halt releases; it was internal-testing only — nothing
   public is lost) and the React webApp (`gauge-dashboard.web.app` → redirect to the
   unified web app; Firebase Hosting config lives in `../gauge/webApp`).
3. Note for morale: the notorious Gauge Google-sign-in saga (Credential Manager
   self-cancel on the owner's Pixel 10 — never solved, config fully verified innocent)
   **dies with the Kotlin app**. This repo's Expo Google sign-in works on that same phone.
4. Branding pass: MindShift display name + new icon on the watch app (package unchanged),
   store listings, splash. The owner wants a non-dial icon.

### Phase 4 — Archive & tidy (~half day)
1. Archive `../gauge` (read-only): final commit pointing here, GitHub archive flag.
2. Migrate keepers into `docs/` here: Gauge's competitive brief
   (`../gauge/tmp/gauge-competitive-brief.html` — market/IP/clinical strategy), relevant
   specs, and this plan's completion notes.
3. Update the owner's Claude auto-memory (both repos' memory dirs) to point here.
4. Recommended while the toolbox is open: continue factoring `server/main.py` (5,198
   lines) into routers; add the missing `LICENSE`; delete the stale "auth is deferred"
   comment in `scripts/deploy_cloudrun.sh`.

---

## 4. Operational knowledge transplant (hard-won in Gauge — do not relearn)

**Building the watch (applies once it lives in `apps/watch/`):**
- `JAVA_HOME=/Users/sophie.arborbot/jdk17/Contents/Home`; Android SDK
  `~/Library/Android/sdk`; `local.properties` gitignored with `sdk.dir`.
- **Gradle ALWAYS foreground with explicit long timeout (600000 ms).** Background builds
  hang agent sessions (notifications never arrive; Gauge committed hooks that hard-block
  backgrounding — consider porting `.claude/hooks/` from `../gauge`). Cold builds
  10–20 min on the owner's old Mac, warm 5–11 min.
- Kotlin-DSL trap: `java.*` shadowed inside `android {}` — import at file top.
- Stale incremental-cache trap: unresolved-reference errors in untouched files →
  `rm -rf <module>/build` and rerun.
- Release signing: Play REJECTS debug-signed AABs on all tracks. Upload keystore
  `~/.config/gauge/gauge-upload.jks`, creds in gitignored `keystore.properties` (copy it
  into this repo root or the watch dir — it's gitignored in Gauge; keep it that way).
  Verify signatures with `keytool -printcert -jarfile <aab>` (upload-key SHA1 starts
  `81:C2:D2:98`).
- **No adb.** Installs go through Play internal testing; on-device diagnostics come from
  the telemetry channel. Design every watch/phone failure path to beacon.

**Play Console (browser automation):**
- Uploads are done via the Claude-in-Chrome bridge. **The bridge routes by claude.ai
  account, not machine** — the owner has a personal Mac and a WORK MacBook; always
  `list_connected_browsers` and confirm the device with the owner before driving anything.
  Extension should be signed into the personal account only on the personal Mac.
- Two apps both displayed as "Gauge" (wear + phone) — always verify the **package-name
  subtitle** before uploading. Post-merge the phone listing to use is
  `com.sagearbor.mindshift.app`; the wear listing keeps `com.sagearbor.gauge.wear`.
- Expected benign warnings on every upload: missing deobfuscation file + native debug
  symbols. Release-notes field has a char cap.
- Better future: Play Developer Publishing API service account (this repo already has
  `scripts/play_publish.py` for the mobile app!) — extend it to the watch AAB and the
  browser dance disappears. Owner was receptive; needs their ~5 min of Console clicking
  to grant the service account access to the wear listing.

**MindShift-side rules (from AGENTS.md + practice — binding):**
- Honest degradation doctrine: no mocks/stubs fabricating success; gate on credentials;
  report unavailability explicitly. Gauge followed the same doctrine; keep it absolute.
- OTA: never run `eas update` raw — use `scripts/ota_publish.sh` (bakes production env;
  a raw update once shipped a bundle pointing at localhost).
- Record `versionCode` bumps in `app.json`+commit (the `chore(release): record versionCode`
  pattern) to prevent Play collisions.
- `pytest` from repo root runs `server/` + `tests/`. In the GAUGE repo `pytest -q` prints
  no summary and looks hung (it isn't — check exit code); be patient with the merged suite.
- Deepgram is pinned `nova-2` for prerecorded (nova-3 merges speakers). Local path:
  faster-whisper + ECAPA `diarize_local` — the crown jewel; never regress it. Eval harness:
  `scripts/diarize_eval.py` + ground-truth rubric.
- Backend env is `MINDSHIFT_*`-prefixed; deploy script reads `.env` because
  `--set-env-vars` replaces the entire set.

**Process (how the owner works — follow it):**
- superpowers workflow: brainstorm → written plan (`docs/plans/`) → subagent-driven
  development (fresh implementer per task, adversarial review gate per task, whole-branch
  final review before merge). In Gauge this caught ~25+ real bugs including a
  password-in-saved-instance-state security issue and a telemetry-blindness gap — final
  reviews pay for themselves. TDD always; feature branch per plan; merge `--no-ff`.
- Deliverables the owner reads → HTML under `tmp/` (mobile-first, light+dark), sent to
  their phone; repo docs → Markdown. Session wrapups → `tmp/wrapups/*.yaml` per their
  global conventions.
- The owner tests on-device promptly and reports with screenshots — build telemetry/beacons
  first so their reports come with server-side evidence.

---

## 5. Inherited open items (park in the backlog, don't lose)

1. **$65 provisional patent before anything PUBLIC** (owner's hard IP rule; internal
   testing is fine). Now doubly relevant: Google sign-in for unverified internal-testing
   apps was flaky, and a public-track release requires the filing first. The filing should
   describe the unified system: live wearable sensing + retrospective conversation analysis.
2. HIPAA/PHI posture: PRD §10 question still open; storage stays opt-in,
   derivatives-only; the fully-local STT+diarization path is the PHI story's backbone.
3. Watch fast-follows queued in Gauge: pairing-client cold-start tolerance (timeout+retry),
   OkHttp call cancellation on screen dismiss, EncryptedSharedPreferences for the device
   token, pairing Retry button.
4. Keep-warm decision for the unified backend (cold starts ate diagnostics twice).
5. `LICENSE` file (neither repo has one).
6. Wife's device: owner's spouse tests on a Pixel 7a (phone) — couples flows need a second
   real account in testing; her Samsung-watch question from Gauge planning is moot until
   the merged platform stabilizes.

## 6. Where to start

Read this file, then: `AGENTS.md`, `PRD.md`, `../gauge/CLAUDE.md`,
`../gauge/docs/SESSION-WRAPUP-2026-08-05.md` (Gauge's last full state snapshot), and
skim `../gauge/tmp/gauge-vs-mindshift-merge-report.html` for the decision rationale.
Then brainstorm + write the Phase 1 plan in `docs/plans/` and execute it with the standard
pipeline. Confirm with the owner before each phase's Play/store-visible step, and send
them an hlist-style HTML status page as phases progress (they are often on their phone).
