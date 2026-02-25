# API Key Setup Guide
_For Sage — do this when you're at your computer_

---

## 1. Anthropic API Key (for backend /respond + /score endpoints)

**Get it:**
1. Go to https://console.anthropic.com → **API Keys** → **Create Key**
2. Name it `mindshift-dev`
3. Copy the key (starts with `sk-ant-api03-...`) — you only see it once

**Where to put it:**
Create a file at `mindshift/server/.env`:
```
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

Then install python-dotenv and load it (already in requirements.txt).

**Which model to use:**

| Task | Model | Cost | Notes |
|------|-------|------|-------|
| Empathy suggestions (`/respond`) | `claude-3-haiku-20240307` | ~$0.00025/req | Already configured ✅ Fast + cheap |
| Tone scoring (`/score`) | `claude-3-haiku-20240307` | ~$0.00025/req | Already configured ✅ |
| If you want cheaper | `claude-3-5-haiku-20241022` | Slightly cheaper | Drop-in replacement |

**Run live tone tests after:**
```bash
cd mindshift
ANTHROPIC_API_KEY=sk-ant-... python3 -m pytest tests/ -v
```

---

## 2. OpenAI API Key (for audio transcription + optional tone tests)

**Get it:**
1. Go to https://platform.openai.com → **API Keys** → **Create new secret key**
2. Name it `mindshift-dev`
3. Copy the key (starts with `sk-...`)

**Where to put it:**
Add to `mindshift/server/.env`:
```
OPENAI_API_KEY=sk-your-key-here
```

**Which model to use:**

| Task | Model | Cost | Notes |
|------|-------|------|-------|
| Audio transcription (V2) | `whisper-1` | $0.006/min | Best quality, easy API |
| Fast chat suggestions | `gpt-4o-mini` | ~$0.0002/req | Good GPT-4 quality, very cheap |
| Realtime audio (V2 earpiece) | Use Deepgram instead | See below | OpenAI realtime is expensive |

---

## 3. Deepgram API Key (for real-time earpiece transcription — V2)

**Why Deepgram over Whisper for live mode:**
Whisper has ~2–5s latency. Deepgram Nova-2 is <300ms. For earpiece coaching, you need Deepgram.

**Get it:**
1. Go to https://console.deepgram.com → **Create API Key**
2. Free tier: $200 credit (enough for months of testing)

**Where to put it:**
```
DEEPGRAM_API_KEY=your-key-here
```

**Model:**
```
nova-2-general   # Best accuracy + speed balance
nova-2-conversational  # Better for natural dialogue
```

---

## 4. Hume AI API Key (for audio tone/emotion detection — V3 Pro)

**Get it:**
1. Go to https://platform.hume.ai → **API Keys** → **Create Key**
2. Free tier available

**Where to put it:**
```
HUME_API_KEY=your-key-here
```

**Why Hume:** Detects 53 emotional dimensions from voice. Best-in-class for tone detection. Way better than doing it with an LLM.

---

## 5. Final .env file (all together)

`mindshift/server/.env`:
```
# Required for MVP
ANTHROPIC_API_KEY=sk-ant-api03-...

# Required for V2 (real-time earpiece)
OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=...

# Required for V3 (tone detection)
HUME_API_KEY=...

# Optional
MINDSHIFT_DB_PATH=mindshift.db
```

---

## 6. Cost estimate (MVP testing)

| Usage | Monthly cost |
|-------|-------------|
| 1000 empathy suggestions (haiku) | ~$0.25 |
| 1000 tone scores (haiku) | ~$0.25 |
| 10 hours transcription (whisper) | ~$3.60 |
| Realtime earpiece (Deepgram) | ~$0.45/hr |
| **Total for active testing** | **< $10/month** |

---

## 7. Test it's working

```bash
cd mindshift

# Install deps
pip install -r server/requirements.txt

# Start server
ANTHROPIC_API_KEY=sk-ant-... python3 -m uvicorn server.main:app --reload

# In another terminal — test a request
curl -X POST http://localhost:8000/respond \
  -H "Content-Type: application/json" \
  -d '{
    "transcript_turn": "I feel like you never listen to me",
    "role": "Husband",
    "empathy_slider": 75,
    "context": ""
  }'
```

Expected response:
```json
{
  "suggestions": [
    "I hear you — it sounds like feeling heard matters a lot to you. Can you tell me more about when you've felt that way?",
    "You're right, and I want to do better. What would listening look like to you?",
    "That must be really frustrating. I'm here now — I'm listening."
  ],
  "tone_score": {
    "warmth": 45,
    "defensiveness": 20,
    "sarcasm": 5,
    "constructiveness": 55,
    "overall": 56
  }
}
```
