# Changelog

Versions are numbered here. **App** = Android/EAS `version (versionCode)`.
**Backend** = Cloud Run revision of `mindshift-api` (project `arborfam-hub`,
`us-central1`). Newest first.

## App 1.5.0 (versionCode 9) — 2026-07-11 · Backend rev 00012
Dynamics v2 (Opus-reviewed, zero critical/major):
- **Absolute heat scale**: one anchored rubric (calm 0-15 → abusive 95+) shared
  by analyzer AND simulator — scores comparable across sessions/people
  (prerequisite for longitudinal averages).
- **Report cards** (owner decision — overt, comparable): per person, score /100
  + headline + "did well" + one concrete "work on".
- **What-If simulation**: `POST /analyze/counterfactual` — tap a turn, see the
  rewritten line + a dashed simulated heat trajectory overlaid on the real
  lines (same thermometer, honest disclaimer, one overlay at a time).

## App 1.4.0 (versionCode 8) — 2026-07-11 · Backend rev 00011
**Conversation Dynamics — "the impartial third chair"** (PR #19):
- `POST /analyze`: one batch LLM pass per transcript — heat 0-100 per turn,
  Gottman Four Horsemen + repair/validation markers, trigger phrases, requests
  & outcomes, strengths-first narrative. Pure-Python stats: talk share,
  interruptions (from live-session timestamps; honestly null for pasted text),
  spikes, repairs-accepted, coupling (LOCF + lag-0/1 Pearson, hand-verified
  lead direction), de-escalation who-first + follow-rate. N speakers (2-10).
- Client: multi-line SVG heat chart (spike dots, tap/scrub to read the turn),
  per-speaker stat cards, insights, "The third chair" narrative, ethics footer.
  Entry: Session tab → "Analyze dynamics" (chains off the live Review handoff
  or any pasted transcript).

## App 1.3.0 (versionCode 7) — 2026-07-10 · Backend revs 00009–00010
**Side-aware coaching — the coach knows who you are** (auto-published to Play
internal; web redeployed same day):
- **Identity**: "you speak first" convention (first voice = you) + a
  "You: Speaker A ⇄" tap-to-swap chip. Reset per session — diarization labels
  aren't stable across sessions (an Opus adversarial-review catch: a
  carried-over swap would have inverted coaching on the next session).
- **Your turns** → one ≤6-word delivery nudge ("ease up" / "be firmer, don't
  back down"); direction follows the empathy dial — the assertive bucket
  never softens. Fine delivery → silence.
- **Their turns** → full suggestions (unchanged).
- **UX**: suggestion history feed (newest first, older faded), amber nudge
  banners, header-overlap fix, idle explainer card, and post-session
  **"Review this conversation →"** handoff into the Session review tab —
  live coaching and async review are one loop now.
- **Deploy hardening**: the deploy script now defaults
  `MINDSHIFT_ALLOWED_ORIGINS` — a redeploy had silently wiped the manually
  added allowlist (--set-env-vars replaces everything), re-breaking the
  browser/iPhone mic; caught by post-deploy probe, root-caused, made
  impossible to recur.

## App 1.2.0 (versionCode 6) — 2026-07-10 · Backend revs 00007–00008
Suggestion **timing** overhaul (published to Play internal via the automated
`play_publish.py` flow; web redeployed same day):
- **Instant transcripts**: new `transcript` WS event per finalized utterance —
  the words no longer wait seconds behind LLM+TTS.
- **Latest-wins queue**: stale pending turns are superseded, killing the
  "suggestions keep coming after I stop talking" backlog flood. At most one
  final suggestion after you stop, about the last thing said.
- **Interject slider** (new, orthogonal to empathy): the LLM scores each
  moment's importance 0-100; the slider sets the voicing threshold
  (Every turn → Most turns → Key moments → Critical only). Below threshold:
  earbud stays silent, card renders dimmed. Empathy slider unchanged (style,
  bidirectional).
- rev 00008: `MINDSHIFT_ALLOWED_ORIGINS` finally set — **iPhone/browser live
  mic unblocked** (web-app origin allowed over WS; foreign origins still 403;
  verified by live handshake test).

## App 1.1.1 (versionCode 5) — 2026-07-06 · Backend rev 00006
- Fixed the two WS gates that rejected the native app (same-origin Origin +
  non-UUID session ids) — first build where live audio actually worked.
- Async **“Review a conversation”**: paste/type a transcript in the Session
  tab → Load → coached suggestions. No audio needed (train-friendly).
- Web: fixed blank screen (canonical `index.js` entry + `@expo/metro-runtime`;
  Google-auth hook isolated so an unconfigured client id can’t crash the app).

## App 1.1.0 (versionCode 4) — 2026-07-06 — ⚠️ never shipped
Built before the Google-auth-hook crash fix; would crash at the login screen.
Superseded by 1.1.1.

## App 1.0.0 (versionCode 1) — 2026-07-06
First release build. Real icon + adaptive icon, `RECORD_AUDIO`, points at the
live Cloud Run backend. Two artifacts built on EAS: preview **APK** (sideload)
and production **.aab** (Play Store internal testing).

## Backend rev 00003 — 2026-07-06
- Added the **"speaks in your voice"** feature (v1): optional per-`(relationship,
  participant)` few-shot voice profile injected into the coaching prompt;
  byte-identical behavior when no profile is set. GET/PUT edit endpoints.

## Backend rev 00002 — 2026-07-06
- Switched the coaching model to **`claude-haiku-4-5-20251001`** (Haiku 4.5).
  Verified end-to-end: `/respond` returns real suggestions + tone scores.

## Backend rev 00001 — 2026-07-05
- First Cloud Run deploy — the FastAPI backend (live audio WebSocket + Deepgram
  STT + Claude coaching + Aura/expo-speech TTS) went from "runs on the Mac" to a
  public HTTPS/WSS service.

---

### Also shipped this cycle (not yet a numbered release)
- **Web microphone capture** (`getUserMedia` + AudioWorklet) so live coaching
  works in browsers incl. iPhone Safari — merged; hosting + real-browser test
  in progress.
- **Play Store prep**: privacy + data-deletion pages (live), store assets,
  runbook.
- Production hardening earlier in the project: CI, security (WS origin allowlist,
  rate limiting, UUID validation), reliability (auto-reconnect, non-blocking LLM
  calls, atomic writes), observability, and honest credential gating.

### In progress
- **Accounts / auth** (Firebase Auth: email + Google) — so personalization and
  history are per-user and the backend is no longer open. Spec → build → deploy.
