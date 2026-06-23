# MindShift

MindShift is an AI-powered empathy coach that helps people understand how their words and tone land — and suggests better responses, calibrated to an empathy slider. Therapist-first go-to-market. See [PRD.md](PRD.md) for the full product spec.

## 🧩 Architecture

- **Frontend:** Expo (React Native + Web) — the active app is **`apps/mobile`** (it serves iOS, Android, and Web). Zustand for state.
- **Backend:** FastAPI (Python) + SQLite, with a model-agnostic `LLMClient` (Claude/OpenAI/Gemini/Mistral).
- **Tests:** Pytest (backend), Jest via jest-expo (frontend).

The source of truth is `apps/mobile/src` (frontend) and `server/` (backend).

## 🚀 Quickstart

### Backend (FastAPI)

```bash
python3 -m pip install -r requirements.txt
cd server && uvicorn main:app --reload   # http://localhost:8000
```

Configuration (env vars):

```bash
MINDSHIFT_MODEL=claude-3-haiku-20240307   # default LLM (see PRD §12 for provider rules)
ANTHROPIC_API_KEY=...                      # required for real LLM calls (tests mock it)
MINDSHIFT_DB_PATH=mindshift.db             # SQLite path

# Optional — real-time audio (M2). Without these, the WebSocket pipeline
# reports `transcription_unavailable` instead of fabricating transcripts.
DEEPGRAM_API_KEY=...                       # real-time transcription (integration pending)
TTS_API_KEY=... | ELEVENLABS_API_KEY=...   # earpiece TTS (integration pending)
```

### Frontend (Expo)

```bash
npm install            # installs the apps/mobile workspace
npm run dev:web        # expo start --web
npm run dev:mobile     # expo start (Expo Go / simulator)
```

## 🧪 Testing

```bash
pytest                 # backend — runs server/ + tests/ from the repo root
npm test               # frontend — jest-expo (delegates to apps/mobile)
```

## 📦 Deployment

Web build via Expo web export. Mobile via Expo Go or EAS build.

---

© 2025 MindShift
