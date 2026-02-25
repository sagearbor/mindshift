# agent.md — mindshift

## Overview
App that helps users see situations from multiple perspectives using AI-driven persona roles. Users describe a situation, choose roles (e.g. "Husband"/"Wife"), and get AI-generated responses from each perspective.

## Tech Stack
- Expo (React Native + Web) — shared frontend
- FastAPI (Python) — backend/mock LLM API
- Zustand — state management
- Tailwind + shadcn/ui — UI
- Jest (TypeScript) + Pytest (Python) — testing
- Monorepo: `apps/`, `packages/`, `server/`, `tests/`

## Status (last commits)
- Initial setup merged into main via PR #1
- Very early stage — base repo structure only
- No significant feature work yet

## How to Run
```bash
npm install
npm run dev:web      # web frontend
npm run dev:mobile   # Expo mobile
cd server && uvicorn main:app --reload  # backend
```

## Open Tasks
- Core persona/role feature implementation
- LLM API integration (real, not mock)
- UI for situation input + role selection
- See `developer_checklist.yaml`

## Branch
- Main: `main`
- Sophie working branch: `sophieArborBot_firstBranch`

## Test Strategy
```bash
npm test          # Jest frontend tests
cd server && pytest  # Python backend tests
```
- TDD approach recommended from README
- Test role-switching logic and LLM response formatting
