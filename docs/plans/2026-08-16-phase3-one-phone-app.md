# Phase 3 — One Phone App, One Web: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** The MindShift Expo app becomes the only phone client (watch setup, nudge settings, claim-history ported in); the Gauge Kotlin phone app and React web dashboard retire; the watch gets MindShift branding.

**Context:** Phase 2 complete 2026-08-16 — unified backend live (rev 00042+), watch v0.4.0 verified end-to-end on it. Owner's priority order (explicit request): **Slice 1 = "Set up your watch" first**, OTA-shipped, so their phone→Play→watch→pairing test happens ASAP.

**Master context:** docs/plans/2026-08-15-unification-mindshift-absorbs-gauge.md §Phase 3.

## Global Constraints
- Branch per slice (`feat/watch-setup-screen` first); PR + CI green + merge each slice; OTA only via `scripts/ota_publish.sh`.
- Mobile conventions: screens in `apps/mobile/src/screens`, API clients in `src/api`, state in `src/store` (Zustand), tests in `apps/mobile/__tests__` (jest-expo). Follow neighboring screens' style. TDD.
- Gates: `npm test` + `npx tsc --noEmit` (from apps/mobile) + full `python3 -m pytest -q` when server files change (they shouldn't) — Bash timeout 600000, judge by exit code; never end a turn waiting on a background notification.
- Auth: all pairing calls use the signed-in Firebase user's token (`Authorization: Bearer` — follow how existing API calls in src/api attach it). The claim endpoint is `POST /me/pair/claim` (full auth) on the unified backend; the app's normal `EXPO_PUBLIC_API_URL` already points there.
- Honest degradation: claim failures surface the server's message; no fake success.

### Task P3-1: "Set up your watch" screen (Slice 1 — ship first)
**Files:** Create `apps/mobile/src/screens/WatchSetupScreen.tsx`, `apps/mobile/src/api/watchPairing.ts`; Modify navigation/settings entry point (find where Settings screens register — mirror how the voice-profile screen was added in PR #98); Tests `apps/mobile/__tests__/WatchSetupScreen.test.tsx`, `watchPairing.test.ts`.
- Screen contents, top to bottom:
  1. **"Install the watch app" button** → `Linking.openURL("https://play.google.com/store/apps/details?id=com.sagearbor.gauge.wear")` — Play handles remote install to the watch (device picker). Subtext: "Opens Google Play — choose your watch when prompted." (Internal-testing note: testers must be opted in; the owner is.)
  2. **"Pair your watch" section**: explains the watch shows a 6-character code (watch → Sign in). Uppercase 6-char input (auto-caps, no autocorrect) + "Pair" button → `POST /me/pair/claim {code}` with auth → success shows "Watch paired ✓" + the watch will finish sign-in itself within ~10 s (it polls); failure shows the server's detail (bad/expired code → friendly retry copy; 429 lockout → its message).
  3. Small status line after success; no polling needed phone-side (the WATCH polls status).
- API client: `claimWatchPairing(code): Promise<{ok, detail?}>` following src/api conventions (base URL, auth header, error normalization).
- [ ] Tests first (client: success/401/404-expired/429 mapping; screen: renders, disabled-until-6-chars, calls client, renders success + error states) → RED → implement → GREEN → gates → commit `feat(mobile): Set up your watch — Play install link + pairing-code claim`.

### Task P3-2: OTA ship Slice 1 + owner test
- [ ] PR → CI → merge → `scripts/ota_publish.sh` (bakes prod env) → tell owner: update flow is open app twice (OTA applies on second launch) → owner runs the real phone→Play→watch→pair test. Their report closes the slice.

### Task P3-3: Port nudge settings (per-vector toggles/sensitivity)
- Reference UI: gauge Kotlin app's Settings (read-only reference `/Users/sophie.arborbot/PROJECTS/github_repos/gauge/androidApp`); endpoints already live: `GET/PUT /settings/vectors` (auth). New screen or section following app conventions; TDD.

### Task P3-4: Claim watch history button
- `POST /me/claim-legacy` (strict auth) + returned counts UI (reference: web dashboard History page). Note: owner's watch currently runs as legacy `default` account — claiming moves those live sessions onto their uid; surface counts clearly.

### Task P3-4b: Rotating hero art — WIPE REVEAL (owner request 2026-08-16, refined)
- The six owner-curated hero images live in `assets/brand/hero/`. NOT a crossfade: a slow **wipe reveal** — a vertical edge travels left→right (~6–8s hold, ~2.5s wipe), unveiling the next image beneath it like a before/after comparison slider (no fly-in; the new image is stationary under the moving blind). **Animated edge effects riding the wipe line, rotating per transition:** burning embers, ice crackle, bubbles (owner's list; extendable). Implementation: CSS clip-path/canvas for the wipe + a narrow particle/effect strip anchored to the edge. Web landing first (site redeploy), mobile home header optional later (OTA). Respect prefers-reduced-motion (fall back to plain crossfade or static). Default order: all six shuffled.

### Task P3-5: Retirements + branding
- Gauge web dashboard (`gauge-dashboard.web.app`) → static redirect page to the MindShift web app (Firebase Hosting config in `../gauge/webApp`) — owner confirmation before flipping anything public-facing.
- Gauge Kotlin phone app: halt releases (no action needed beyond never publishing again); note in gauge repo README at Phase 4 archive time.
- Watch branding: display name "MindShift", new non-dial icon (owner wants non-dial; generate options for owner pick), splash/copy pass. Package name NEVER changes. vc11 release via wear:internal when ready.
- Phase 4 reminder: gauge-api decommission after ~2026-08-23 if telemetry shows no stragglers.

### Task P3-6: State-aware Advanced rows (owner field feedback 2026-08-17)
- "Set up your watch" must reflect reality: server-derived paired state (extend GET /me or a light endpoint to expose has-paired-watch via the account's device tokens; add the small pairing_store lookup if missing) → row shows "✓ Paired to this account" + remains tappable for re-pair/another watch. "Train my voice" → when enrolled, retitle to "Add more voice training" (profile state already fetched by the screen). No new prominent CTAs for done things.

### Task P3-7: Onboarding walkthrough (owner request 2026-08-17)
- First-launch tutorial: a few swipeable, skippable cards explaining Live Coach, Analyze a conversation, the watch app + nudges, Growth chart. Persist seen-state. "Show tutorial" row in Advanced re-runs it. OTA-shippable.

### Task P3-8: Watch daily-calm surface — red→green + line chart (owner request 2026-08-17)
- Wrist-glanceable "how am I doing today": extend the existing complication (RANGED_VALUE) to publish today's calm score (0–100 → red→green tint where the face supports it) and the existing tile to render a small today-line-chart. Data: /me/standing (watch already calls it; daily window). Cache last value for offline glance; honest empty state when no sessions today. Builds on Gauge v0.2.4 sparkline groundwork; needs a watch release (vc12) when done.

### Task P3-9: Navigation redesign exploration (owner request 2026-08-17 — DESIGN FIRST, owner reviews before any code)
- Owner dislikes current nav: home's "⋯" goes to one Advanced page; expects hamburger with settings/about/logout etc.; open to a consistent navbar; wondering where a hero fits on mobile. Deliverable = design proposal (HTML mockups under tmp/, sent to owner): 2–3 nav architectures compared honestly (bottom tabs + top bar w/ overflow; hamburger drawer; hybrid), full menu inventory (settings, about/version, tutorial, watch setup, voice profile, therapist dashboard, logout), where the hero lives on mobile in each, migration notes from current screens. Owner picks; implementation is a follow-up task.
### P3-4b addendum (owner decision: "both"): hero ALSO on the web sign-in/landing page with proper layout (banner above the title block, no overlap), same component, tests updated; mobile placement deferred to the P3-9 outcome.

### P3-9 RESOLVED — owner's architecture decisions (2026-08-17)
Chosen: hamburger (top-left, ALWAYS the complete destination catalog — the safety net that makes customization possible) + selfie avatar (top-right; onboarding prompts a profile selfie, retakeable in Settings; shows logged-in identity + account/logout) + configurable bottom bar (0–5 user-chosen slots; default = proposal's tabs) + home area of up to 4 user-arranged boxes, EVERY box icon+label (owner: bare colored boxes look horrible — users learn icons). Settings gains "Home screen design": drag-to-arrange for boxes + tab slots. Default home leans trend line/history when daily actions live in the tab bar — but owner will iterate once it's modifiable. Implementation order: Phase 0 (Advanced→Settings regroup + About/logout/tutorial rows) FIRST, then the customization system (P3-10: layout store + edit mode + icon set + avatar capture) as its own planned effort. Nav stays hand-rolled (react-navigation would break OTA); Android hardware-back handling gets fixed as part of the nav work.
