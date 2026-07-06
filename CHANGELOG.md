# Changelog

Versions are numbered here. **App** = Android/EAS `version (versionCode)`.
**Backend** = Cloud Run revision of `mindshift-api` (project `arborfam-hub`,
`us-central1`). Newest first.

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
