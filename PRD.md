# MindShift — Product Requirements Document

**Version:** 0.2  
**Owner:** Sage Arbor  
**AI Author:** Sophie  
**Last Updated:** 2026-02-25  
**Status:** Active

---

## 1. Vision

MindShift is an AI-powered empathy coach that helps people understand how others perceive their words and tone — in real time. Core differentiators vs. "just using ChatGPT":

1. **Real-time** — coaching happens *during* a conversation, not after
2. **Tone-aware** — detects *how* something is said, not just *what*
3. **Memory** — knows your relationship history and recurring patterns
4. **Measurable** — quantifies behavior over time so users and therapists can track improvement

**Primary use case:** A husband wears an earpiece. MindShift listens to his wife, transcribes her words (and eventually analyzes her tone), and whispers suggestions calibrated to his empathy slider setting.

**Go-to-market:** Therapists first. Deploy with 1–2 couples therapists, collect efficacy data, iterate.

---

## 2. Target Users

| User | Use Case | Priority |
|------|----------|----------|
| Couples therapists | Live session tool + outcome tracking | MVP |
| Therapy patients (couples) | Homework, self-coaching | MVP |
| Self-guided couples | Daily behavior awareness | V2 |
| Researchers | Labeled dialogue dataset collection | V3 |

---

## 3. Platform

**Single codebase, all platforms** — Expo (React Native + Web)  
- iOS, Android, Web from one stack  
- Backend: FastAPI (Python)  
- State: Zustand  
- UI: Tailwind + shadcn/ui  
- Tests: Jest (frontend) + Pytest (backend)

No stack migration needed — already correct.

---

## 4. Core Concepts

### Empathy Slider (0–100)
| Value | Mode | Behavior |
|-------|------|----------|
| 0–20 | Assertive | Challenge, push back, set boundaries |
| 21–50 | Balanced | Acknowledge feelings + redirect constructively |
| 51–80 | Empathetic | Validate, reflect, minimize judgment |
| 81–100 | Full empathy | Pure validation, no redirection |

Controls LLM prompt weighting for all suggestions.

### Roles
- Predefined: Husband/Wife, Parent/Child, Manager/Employee, Therapist/Patient, Friend/Friend
- Custom: user-defined role names
- Each role has a perspective prompt template + memory context

### Pleasantness Score
- Numeric 0–100 per speaker per utterance
- Aggregated to daily/weekly view
- Dimensions: warmth, tone, constructiveness, defensiveness, sarcasm
- Powers the daily "How did I do today?" review

---

## 5. Feature Tiers

### Tier 0 — MVP (Text, Therapist Pilot)
- [ ] Text input: type or paste transcript turns
- [ ] Role selector: assign names + roles to each speaker
- [ ] Empathy slider: 0–100
- [ ] AI generates 2–3 suggested responses per turn from the listener's perspective
- [ ] Text-based tone score per turn (LLM sentiment: warmth, defensiveness, sarcasm)
- [ ] Session log: timestamped conversation + scores saved locally
- [ ] Export: PDF/text session summary for therapist notes
- [ ] Therapist mode: therapist-facing dashboard showing patient sessions
- [ ] Latency target: <2s for text response

### Tier 1 — V2: Real-Time Earpiece Mode
- [ ] Live mic input → streaming transcription (Whisper / Deepgram)
- [ ] Speaker diarization: identify who is speaking
- [ ] LLM suggestion generation on each speaker turn
- [ ] TTS output → Bluetooth earpiece (system audio / AirPods)
- [ ] Push-to-suggest: only fires when the coached person is silent
- [ ] Visual fallback: text suggestions on screen (no earpiece required)
- [ ] Latency target: <800ms end-to-end (mic → earpiece)
- [ ] "Coach mode" toggle: on/off during conversation

### Tier 2 — V3: Tone & Nuance Detection (Pro)
- [ ] Audio tone analysis beyond words: detect frustration, sarcasm, defensiveness, sadness
- [ ] Tone indicators in UI ("⚠️ defensive tone detected")
- [ ] Suggestions adjusted for detected emotional state (not just words)
- [ ] Pleasantness score derived from both text + audio features
- [ ] Daily audio collection → end-of-day pleasantness graph per speaker
- [ ] Smartwatch integration: haptic feedback (vibration frequency ∝ tone score degradation)

**Tone detection candidates to research (Agent task):**
- Hume AI API (expression measurement)
- GPT-4o audio (multimodal)
- SpeechBrain (open source, HuggingFace)
- wav2vec2-based emotion classifiers (HuggingFace)
- pyAudioAnalysis (open source audio feature extraction)
- openSMILE (open source, research-grade)

### Tier 3 — V4: Per-Person Memory
- [ ] Voice fingerprinting: identify speakers automatically across sessions
- [ ] Per-person profile: communication patterns, recurring triggers, improvement trends
- [ ] Session history: timeline of conversations with scores
- [ ] Relationship graph: how tone varies by topic, time of day, stress level
- [ ] "Insight" summaries: AI-generated weekly relationship health report
- [ ] Privacy controls: local-only mode, encrypted storage option

---

## 6. Pleasantness Score System

### Collection
- Always-on (opt-in) ambient audio recording
- Speaker-separated via diarization
- Scored per utterance in near real-time

### Scoring Dimensions (0–100 each, weighted average)
| Dimension | Weight | Description |
|-----------|--------|-------------|
| Warmth | 30% | Kindness, affection, positive regard |
| Constructiveness | 25% | Solution-focused vs. blame |
| Calmness | 20% | Absence of escalation signals |
| Respect | 15% | Non-contemptuous, non-dismissive |
| Engagement | 10% | Active listening signals |

### User-Facing Features
- Daily timeline: sparkline of pleasantness score by hour
- Worst moments flagged (opt-in review)
- Comparative view: "You vs. partner over time"
- Private by default: each person sees only their own score unless they share

### Smartwatch Haptics (V3+)
- Score > 70: no haptic
- Score 50–70: single soft pulse every 2 min
- Score 30–50: double pulse every 1 min
- Score < 30: continuous escalating pattern ("you're in the red")

---

## 7. Testing Strategy (Fully Agentic)

### Philosophy
Design all tests to be runnable by an AI agent with zero human input. Sage reviews edge cases only.

### Tier 0 — Text-Only Tests
- Synthetic conversation transcripts (agent-generated)
- Ground truth: expected empathy suggestions at slider values 0/50/100
- Validate: does the output match expected tone/stance?
- Tools: Pytest, LLM-as-judge scoring

### Tier 1 — Audio Transcription Tests  
- Input: text transcript (no audio needed initially)
- Simulate real-time turn-by-turn delivery
- Validate suggestion quality + latency
- Test diarization accuracy on synthetic multi-speaker transcripts

### Tier 2 — Tone Detection Tests
- **Datasets to use (agent research task):**
  - IEMOCAP (emotional speech, 12h, labeled)
  - MELD (Friends TV dialogue, emotion labels)
  - MSP-Podcast (naturalistic speech, emotion + sentiment)
  - RAVDESS (acted emotional speech, good for model training)
  - CMU-MOSI (sentiment + emotion, multimodal)
- **Test format:** audio file + transcript → model assigns tone scores → compare to ground truth labels
- **Agent validation loop:** agent runs model, compares to dataset labels, reports accuracy + confusion matrix
- **Human vetting:** Sage reviews 20–30 edge cases to calibrate subjective dimensions

### Tier 3 — Memory + Identity Tests
- Synthetic user profiles with known patterns
- Validate: does memory retrieval affect suggestions correctly?
- Voice fingerprint accuracy across simulated sessions

---

## 8. Technical Architecture

```
Mobile/Web (Expo)
    │
    ├── Mic capture → streaming audio chunks
    ├── Empathy slider state (Zustand)
    ├── Role configuration
    └── UI: suggestions, score, timeline

Backend (FastAPI)
    │
    ├── /transcribe   → Whisper / Deepgram (streaming)
    ├── /respond      → LLM suggestion generation
    ├── /score        → Tone/pleasantness scoring
    ├── /session      → Save/load session logs
    └── /profile      → Per-person memory store

LLM Layer
    ├── Anthropic Claude (suggestions, tone analysis from text)
    └── GPT-4o audio (tone from audio — Pro tier)

Tone Detection (V3+)
    ├── Hume AI API (primary candidate)
    ├── SpeechBrain / wav2vec2 (open source fallback)
    └── pyAudioAnalysis (lightweight features)

Storage
    ├── Local SQLite (MVP)
    └── Postgres + S3 (V2+)
```

---

## 9. Milestones

| Milestone | Target | Deliverable |
|-----------|--------|-------------|
| M0 | Week 1–2 | Text MVP: roles + slider + AI responses |
| M1 | Week 3–4 | Tone scoring from text + session export |
| M2 | Week 5–6 | Real-time transcription + suggestions |
| M3 | Week 7–8 | Earpiece TTS output + diarization |
| M4 | Week 9–10 | Audio tone detection + pleasantness score |
| M5 | Week 11–12 | Therapist dashboard + efficacy data collection |
| M6 | TBD | Smartwatch haptics + per-person memory |

---

## 10. Open Questions

- [ ] Which therapist(s) to pilot with? (Sage to identify)
- [ ] Privacy/HIPAA considerations for session recordings?
- [ ] Earpiece UX: AirPods TTS vs. bone conduction?
- [ ] Smartwatch platform: Apple Watch vs. WearOS vs. both?
- [ ] Data ownership: local-only MVP or cloud sync from day 1?
- [ ] Voice fingerprinting: on-device (privacy) vs. cloud API?

---

## 11. Non-Goals (for MVP)

- Not a couples counseling replacement
- Not HIPAA-certified (out of scope for pilot)
- Not a general-purpose chatbot
- Not multi-language (English only for MVP)

---

## 12. Multi-Vendor Model Support

### Design Principle
The backend must be **model-agnostic**. Swap providers without touching business logic. All LLM calls go through a single `llm_client.py` abstraction layer.

### API Format Detection

Different providers use incompatible APIs. Detection must be automatic based on model name:

| Model Pattern | API Format | Notes |
|--------------|-----------|-------|
| `claude-*` | Anthropic Messages API | `anthropic` SDK |
| `gpt-4o*`, `gpt-4-*`, `gpt-3.5-*` | OpenAI Chat Completions | `openai` SDK, `/chat/completions` |
| `gpt-5*`, `o1*`, `o3*`, `o4*` | OpenAI Responses API | `openai` SDK, `/responses` — different input/output format |
| `gemini-*` | Google Generative AI | `google-generativeai` SDK |
| `mistral-*` | Mistral Chat Completions | Chat completions compatible |

### Temperature Rules (Critical)
Many bugs come from setting temperature on models that don't support it:

| Model | Temperature | Reasoning |
|-------|------------|-----------|
| `claude-*` | 0.0–1.0 ✅ | Normal |
| `gpt-4o*`, `gpt-4-*` | 0.0–2.0 ✅ | Normal |
| `gpt-5*` (non-reasoning) | 0.0–2.0 ✅ | Normal |
| `gpt-5*` with `reasoning` | **Must be 1.0** ⚠️ | Hardcoded — do not pass other values |
| `o1*`, `o3*`, `o4*` | **Omit entirely** ⚠️ | These models reject temperature param |
| `gemini-*` | 0.0–1.0 ✅ | Normal |

### Responses API vs Chat Completions (OpenAI)

**Chat Completions** (older, most models):
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "..."}],
    temperature=0.7
)
text = response.choices[0].message.content
```

**Responses API** (GPT-5+, future models):
```python
response = client.responses.create(
    model="gpt-5",
    input="...",          # Different field name
    temperature=1.0       # Or omit for reasoning variants
)
text = response.output_text  # Different output field
```

### `llm_client.py` — Required Interface

```python
class LLMClient:
    def __init__(self, model: str, api_key: str): ...

    def complete(self, 
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        # Detects provider + API format from model name
        # Applies temperature override rules automatically
        # Returns plain text string always
        ...
    
    @staticmethod
    def is_reasoning_model(model: str) -> bool:
        return model.startswith(("o1", "o3", "o4")) or \
               ("gpt-5" in model and "reasoning" in model)
    
    @staticmethod
    def uses_responses_api(model: str) -> bool:
        return model.startswith(("gpt-5", "o1", "o3", "o4"))
```

### Environment Config
```
# mindshift/server/.env
MINDSHIFT_MODEL=claude-3-haiku-20240307   # default — cheap + fast
# Override anytime:
# MINDSHIFT_MODEL=gpt-4o-mini
# MINDSHIFT_MODEL=gpt-5
# MINDSHIFT_MODEL=gemini-2.0-flash
```

### Agent Task (spawn when ready)
Implement `server/llm_client.py` with full provider detection, temperature rules, and tests covering all model pattern branches.
