# P3-10 — Customizable Home & Navigation: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Owner's architecture decisions are LOCKED (see "P3-9 RESOLVED" in docs/plans/2026-08-16-phase3-one-phone-app.md); do not relitigate. Implementers work in ISOLATED WORKTREES.

**Goal:** The owner's chosen navigation: hamburger (complete catalog) + selfie avatar (top-right) + configurable bottom bar (0–5 slots) + home area of up to 4 icon boxes, all arranged by the user via Settings → "Home screen design". OTA-shippable (hand-rolled nav preserved — react-navigation is BANNED here, it breaks the OTA loop). Android hardware-back handled.

**Constraints:** react-native-svg 15.15.4 available for icons (HeatChart precedent). expo-camera available for selfie. All state via the app's existing storage pattern. Owner UX directive: best practices on merit; icons never bare colored boxes. TDD; gates = npm test + tsc; worktree isolation per implementer; never background/wait on notifications.

### Task N1: Layout store + destination registry (the foundation)
- `src/nav/destinations.ts`: registry of every destination: id, title, icon id, screen mapping (from the existing Screen union): coach, analyze, recordings, growth, watchSetup, voiceProfile (section anchor), therapistDashboard, settings, tutorial. Primary-eligible flag (can appear in tabs/boxes).
- `src/store/layoutStore.ts` (zustand + persistence like existing stores): `{tabSlots: DestId[] (0–5), homeBoxes: DestId[] (0–4)}` + defaults (tabs: coach, analyze, growth; boxes: recordings + growth-trend emphasis per owner's "home could be trend/history"), setters with validation (no dupes, caps), reset-to-default. Pure logic fully unit-tested.
### Task N2: Icon set (SVG, brand-derived)
- `src/components/icons/`: simple stroke icons via react-native-svg for every destination (mic/coach, waveform/analyze, list/recordings, trendline/growth, watch, voice, clipboard/dashboard, gear/settings, book/tutorial, hamburger, back) — consistent 2px stroke, currentColor, sized via prop. Tested renders (no snapshot explosion — one render-all test).
### Task N3: Chrome — top bar (hamburger + avatar) + bottom tab bar + Android back
- `src/components/AppChrome.tsx` (or per codebase convention): top bar with hamburger (opens full-screen/drawer catalog listing ALL destinations w/ icons; always complete regardless of customization) + avatar button (photo if set, else initial/silhouette) opening a small menu: account email, Settings, Log out. Bottom bar renders layoutStore.tabSlots (hidden entirely when 0). Wire into App.tsx's existing hand-rolled Screen switch — chrome shows on primary screens; pushed detail screens keep their back affordances.
- Android hardware back: BackHandler — pushed screens pop to home; home double-back exits (toast "Back again to exit" pattern). Web: no regression (document nothing new; URL routing stays a later project).
### Task N4: Home boxes
- HomeScreen main area renders layoutStore.homeBoxes as icon+label cards (grid 1–4, generous targets); when boxes exclude a daily action it remains reachable via tabs/hamburger. Keep the hero header (web) and existing home content harmony — if growth-trend box is present, it may render a mini trend preview (reuse existing chart data hook if cheap; else icon card now, preview follow-up).
### Task N5: Settings → "Home screen design" editor
- New Settings row + screen: two sections (Bottom bar 0–5; Home boxes 0–4) with add/remove from the full registry and reorder (up/down buttons acceptable v1 — dragging is polish; owner asked "drag to change" so attempt drag via existing gesture responder if time-safe, else buttons + ledger the drag follow-up honestly). Live preview strip. Reset to default. Persists via layoutStore.
### Task N6: Selfie avatar
- Avatar capture: expo-camera front-facing selfie flow from the avatar menu ("Set profile photo") + Settings row; stored locally (file + uri in prefs); circular-cropped display in top bar. Server upload = ledgered follow-up (cross-device later). Honest empty state (initial letter) until set. Permission denial → honest message.
### Task N7: Whole-branch review + OTA
- Final review (opus) over the whole feature branch; then merge + single OTA with everything else from tonight; morning hlist.
