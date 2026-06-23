# agent.md — mindshift

## Overview
App that helps users see situations from multiple perspectives using AI-driven persona roles. Users describe a situation, choose roles (e.g. "Husband"/"Wife"), and get AI-generated responses from each perspective.

## Tech Stack
- Expo (React Native + Web) — shared frontend
- FastAPI (Python) — backend/mock LLM API
- Zustand — state management
- Tailwind + shadcn/ui — UI
- Jest (TypeScript) + Pytest (Python) — testing
- Layout: `apps/mobile/` (active app), `server/`, `tests/`

## Status (2026-06-23, branch sophieArborBot_m2-realtime)
- M0–M1 done & tested: /respond, /score, /session CRUD + multi-turn, /session export (text+PDF),
  multi-vendor LLMClient, on-disk LLM response cache, relationship graph model.
- M2 in progress: WebSocket audio pipeline wired. Transcription (Deepgram) and TTS are
  credential-gated — they report `transcription_unavailable` rather than fabricating data;
  live integrations still to be built.
- Frontend (apps/mobile): SessionScreen, LiveCoachScreen, TherapistDashboard, SessionDetail + components.
- Active app is `apps/mobile`; the old `apps/web/` + `packages/*` scaffold was removed.

## How to Run
```bash
npm install
npm run dev:web      # web frontend
npm run dev:mobile   # Expo mobile
cd server && uvicorn main:app --reload  # backend
```

## Open Tasks
- Live Deepgram transcription + diarization (replace credential-gated stubs)
- Earpiece TTS integration
- Align `apps/mobile/src/hooks/useAudioStream.ts` with the server WebSocket protocol
- Auth (deferred for now)
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
