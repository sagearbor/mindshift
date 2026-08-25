# MindShift

MindShift is an AI-powered empathy coach that helps people understand how their words and tone land — and suggests better responses, calibrated to an empathy slider. Therapist-first go-to-market. See [PRD.md](PRD.md) for the full product spec.

## 🧩 Architecture

- **Frontend:** Expo SDK 57 (React Native + Web) — the active app is **`apps/mobile`** (it serves iOS, Android, and Web). Zustand for state. Live mic streaming works on iOS/Android (including Expo Go) via `expo-audio`; web mic capture is not yet available (the UI shows an error banner rather than pretending).
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

# Optional — real-time audio (M2/M3). Without DEEPGRAM_API_KEY the app still
# runs; the WebSocket pipeline reports `transcription_unavailable` instead of
# fabricating transcripts.
DEEPGRAM_API_KEY=...                       # live streaming STT (nova-3, diarized) + Aura TTS
STT_PROVIDER=deepgram                      # deepgram (default) | whisper (free, local, offline)
WHISPER_MODEL=base                         # tiny|base|small|medium — only when STT_PROVIDER=whisper
TTS_API_KEY=... | ELEVENLABS_API_KEY=...   # recognized but not yet implemented
```

Env vars load from a `.env` at repo root (see `env.example`) if present; real
shell env always wins.

**Free voice path (no paid keys):** set `STT_PROVIDER=whisper` and
`pip install -r requirements-whisper.txt` for free on-device transcription via
faster-whisper (near-real-time, private — audio never leaves the machine;
slightly laggier than Deepgram's true streaming). On the mobile side, coaching
suggestions are spoken with on-device `expo-speech` (free, no key) — the Deepgram
key is optional throughout. Deepgram stays the default for the fastest, lowest-
latency streaming during dev.

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

## 💸 Cost guardrails

Every coached turn is an LLM call and every live minute can be a Deepgram
minute, so per-account spend is counted and capped.

```bash
# what each account has spent, priced (owner allowlist: MINDSHIFT_ADMIN_UIDS)
python scripts/usage_report.py --id-token "$TOKEN" --since 2026-08-01
```

Soft daily caps (`MINDSHIFT_DAILY_*` in `env.example`) **degrade rather than
break**: past the cap the live socket stops calling the cloud but keeps the
transcript and the phone's on-device loop running, and sends one
`quota_notice` frame saying what stopped and when it resets. The measured
cost of a coached session, the per-therapist monthly estimate, and the three
biggest levers to cut it are in
[docs/plans/2026-08-25-cost-model.md](docs/plans/2026-08-25-cost-model.md).

## 📦 Deployment

Web build via Expo web export. Mobile via Expo Go or EAS build.

---

© 2025 MindShift
