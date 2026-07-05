# agent.md — mindshift

## Overview
App that helps users see situations from multiple perspectives using AI-driven persona roles. Users describe a situation, choose roles (e.g. "Husband"/"Wife"), and get AI-generated responses from each perspective.

## Tech Stack
- Expo SDK 57 (React Native + Web) — shared frontend (`expo-audio` for mic capture; `expo-av` is gone in SDK 57)
- FastAPI (Python) — backend API
- Zustand — state management
- Tailwind + shadcn/ui — UI
- Jest (TypeScript) + Pytest (Python) — testing
- Layout: `apps/mobile/` (active app), `server/`, `tests/`

## Status (2026-07-04, branch sophieArborBot_m2-realtime)
- M0–M1 done & tested: /respond, /score, /session CRUD + multi-turn, /session export (text+PDF),
  multi-vendor LLMClient, on-disk LLM response cache, relationship graph model.
- M2 done: live Deepgram streaming transcription + diarization (nova-3 over the raw
  `wss://api.deepgram.com/v1/listen` protocol via `websockets` — deliberately no deepgram-sdk),
  real timestamps/speakers/confidence in `TranscriptSegment`s. Credential-gated honestly:
  no `DEEPGRAM_API_KEY` or any Deepgram failure → `transcription_unavailable`; nothing fabricated.
  Tests run against a local fake Deepgram server (`server/tests/test_deepgram_live.py`) — no keys needed.
- Free voice path (no paid keys): `STT_PROVIDER=whisper` runs local faster-whisper
  (`server/whisper_transcriber.py`) as a drop-in transcriber — background-worker design
  (audio queue → transcribe off the receive loop), shared module-level model cache, honest
  gating (package absent → `transcription_unavailable`, never faked). Optional dep in
  `requirements-whisper.txt` only; base install stays light. Deepgram remains the default.
  Env loads from `.env` (see `env.example`) via python-dotenv.
- M3 partial: coaching suggestions are SPOKEN on-device for free via `expo-speech` (earpiece
  mode; most-recent-wins; honest degradation if a platform has no TTS). Server-side Deepgram
  Aura (`aura-2-thalia-en`, base64 mp3 in suggestion events) also available when the key is
  set — but the app uses free expo-speech, not the Aura audio, so no key is needed to hear
  suggestions.
- Mobile on Expo SDK 57 (`expo-av` removed): live mic streaming works on iOS/Android incl.
  Expo Go — `expo-audio` PCM → `src/utils/audio.ts` (downmix/resample/int16) → 100 ms binary
  WS frames (raw PCM int16 LE, 16 kHz mono) to `/ws/session/{id}`. Web mic capture is honestly
  unavailable (error banner) but web sessions still run (config/suggestions flow, no audio).
  Audio sender survives reconnects (≤5 s buffered). Graceful stop handshake: client flushes
  remaining PCM, sends `{"type":"stop"}`; server finalizes Deepgram, delivers all pending
  suggestion events, replies `{"type":"session_complete"}`, closes 1000 — so the final
  utterance is never lost (4 s client-side drain timeout).
- Frontend (apps/mobile): SessionScreen, LiveCoachScreen, TherapistDashboard, SessionDetail + components.
- Active app is `apps/mobile`; the old `apps/web/` + `packages/*` scaffold was removed.
  npm workspaces uses the root lockfile only (no `apps/mobile/package-lock.json`).

## How to Run
```bash
npm install
npm run dev:web      # web frontend
npm run dev:mobile   # Expo mobile
cd server && uvicorn main:app --reload  # backend
```

## Open Tasks
- Web mic capture (expo-audio 57 has no web recording backend)
- Deepgram auto-reconnect mid-session (a dead Deepgram socket surfaces
  `transcription_unavailable` rather than silently reconnecting)
- M4: audio tone detection (pleasantness scoring)
- M5: efficacy data collection
- M6: smartwatch haptics + per-person memory
- TTS_API_KEY / ELEVENLABS_API_KEY are recognized but unimplemented
- Auth (explicitly deferred)
- See PRD.md §9 milestones

## Branch
- Main: `main`
- Active working branch: `sophieArborBot_m2-realtime`

## Test Strategy
```bash
npm test          # Jest frontend tests
cd server && pytest  # Python backend tests
```
- TDD approach recommended from README
- Test role-switching logic and LLM response formatting
