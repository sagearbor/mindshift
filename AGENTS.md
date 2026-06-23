# AGENTS.md

## Purpose
Guide AI agents contributing to MindShift (see [PRD.md](PRD.md) for the product spec).

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
