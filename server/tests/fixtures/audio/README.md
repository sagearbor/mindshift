# Audio test fixtures — the three-rung ladder

Synthesized 2026-07-12 (no human recordings). One scripted two-person argument
(calm open → escalation → shouted spike → cold contempt → sad → scared →
repair → calm close), three ways. Regenerate with scripts/make_test_recording*.py.

| File | Engine | Deepgram diarization (nova-3, measured) | Use for |
|---|---|---|---|
| test_recording.wav | Deepgram Aura-2 + mathematical gain/tempo modulation | **1 speaker (WRONG — merges everyone)**: robotic sameness + resample pitch-shift breaks voice identity | Prosody-METER ground truth only (meta carries expected energy/rate labels; the modulated turns are physically known) |
| test_recording_openai.wav | OpenAI gpt-4o-mini-tts-2025-12-15, acted via instructions | **2 speakers, clean** | The clean end-to-end pipeline case |
| test_recording_gptaudio.wav | OpenAI gpt-audio-1.5 (voice-actor prompt) | 2 speakers + 2 turns misattributed to a phantom Speaker C | Realism STRESS test — extreme acted shifts fool clustering the way real fights do. Owner-rated the most human-sounding of the three. |

Lessons encoded here:
- Physics modulation validates the measurement layer; acted speech validates
  listeners/diarizers. Naive tempo resampling shifts pitch and destroys voice
  identity — never use the physics fixture to test diarization.
- gpt-audio-1.5 as a LISTENER (scripts/audio_tone_probe.py): correctly hears
  anger/sadness/calm arcs (shout = arousal peak) but confuses fear with
  sadness and returned unparseable JSON ~30% of the time (probe records those
  as honest errors).
