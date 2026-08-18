# Audio test fixtures — the three-rung ladder

Synthesized 2026-07-12 (no human recordings). One scripted two-person argument
(calm open → escalation → shouted spike → cold contempt → sad → scared →
repair → calm close), three ways. Regenerate with scripts/make_test_recording*.py.

| File | Engine | Deepgram diarization (nova-3, measured 2026-07-12) | diarize_local (ECAPA, measured 2026-08-17) | Use for |
|---|---|---|---|---|
| test_recording.wav | Deepgram Aura-2 + mathematical gain/tempo modulation | **1 speaker (WRONG — merges everyone)**: robotic sameness + resample pitch-shift breaks voice identity | 2 speakers, agreement 0.7-1.0 ceiling (`test_diarize_local_live.py`) | Prosody-METER ground truth only (meta carries expected energy/rate labels; the modulated turns are physically known) |
| test_recording_openai.wav | OpenAI gpt-4o-mini-tts-2025-12-15, acted via instructions | **2 speakers, clean** | **2 speakers, 10/10 = 100% exact per-turn accuracy** | The clean end-to-end pipeline case |
| test_recording_gptaudio.wav | OpenAI gpt-audio-1.5 (voice-actor prompt) | 2 speakers + 2 turns misattributed to a phantom Speaker C | **2 speakers, 10/10 = 100% exact per-turn accuracy** — the old phantom-C regression is gone | Realism STRESS test — extreme acted shifts fool clustering the way real fights do. Owner-rated the most human-sounding of the three. |

The nova-3 column is the VENDOR diarizer (Deepgram) and was not re-measured
here — it is a different system from `diarize_local`, the local ECAPA
fallback that exists BECAUSE nova-3 regressed (see `diarize_local.py`'s
module docstring). The `diarize_local` column is fresh as of 2026-08-17
(`server/tests/test_diarize_regression_ladder.py`, added that day): the
gptaudio fixture's 2026-07-12 "phantom Speaker C" claim predates
2026-08-14/15's N-way k-detection, anchor recalibration (0.15→0.20) and
word-level rapid-exchange splitting, and does not reproduce on the current
pipeline — ground-truth-verified first (script/meta diff + independent local
Whisper transcription + duration check, see
`.superpowers/sdd/2026-08-17-diarization-regression/report.md`), not assumed.

Lessons encoded here:
- Physics modulation validates the measurement layer; acted speech validates
  listeners/diarizers. Naive tempo resampling shifts pitch and destroys voice
  identity — never use the physics fixture to test diarization.
- gpt-audio-1.5 as a LISTENER (scripts/audio_tone_probe.py): correctly hears
  anger/sadness/calm arcs (shout = arousal peak) but confuses fear with
  sadness and returned unparseable JSON ~30% of the time (probe records those
  as honest errors).
- `diarize_local` requires 16 kHz input (`speaker_id.TARGET_SR`); the OpenAI
  TTS fixtures are natively 24 kHz, so a caller must resample first (the
  regression tests use `audio_ingest.decode_to_pcm_16k`, the same path
  `routers/voice.py` uses for voice enrollment). This surfaced that
  `main.py`'s `/analyze/upload` cross-check instead calls plain
  `decode_to_pcm` (native rate preserved) before invoking `diarize_local`,
  so on a real upload whose native WAV rate isn't already 16 kHz the ECAPA
  cross-check silently no-ops (caught by the broad `except Exception` a few
  lines later) — flagged in the report as a follow-up, not fixed here.
