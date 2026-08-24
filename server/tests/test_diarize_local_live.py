"""Live-gated test: local ECAPA diarization separates the TTS fixture voices
from a FULLY COLLAPSED starting point (every turn labeled "Speaker A") —
the "must discover the count from scratch" scenario that
test_diarize_regression_ladder.py doesn't cover (its tests start from the
fixture's own already-correct alternating labels).

The positive counterpart to test_audio_upload_live's nova-3 xfail: Deepgram's
nova-3 (2025-07-31.0) can no longer diarize these synthetic voices, so OUR
clustering must — that is the whole point of diarize_local. Needs the optional
voice deps (torch + speechbrain; first run downloads the pinned ECAPA
checkpoint) and the fixture. No Deepgram key needed: ground-truth turn
boundaries come from the fixture's metadata, not a live transcription.

REPOINTED 2026-08-24 from tmp/test_recording.wav to the checked-in
gptaudio.wav fixture: tmp/test_recording.wav is server/tests/fixtures/audio/
README.md's "Prosody-METER ground truth only" fixture — physically
gain/tempo-modulated from ONE flat neutral TTS voice (Deepgram Aura, which
"cannot act" per scripts/make_test_recording.py), NOT real acted speech, and
the README explicitly says "never use the physics fixture to test
diarization". This test was doing exactly that. gptaudio.wav (OpenAI
gpt-audio-1.5, a real voice-actor prompt) is the fixture this failure mode
was ORIGINALLY calibrated against (see diarize_local.py's
NEW_VOICE_ANCHOR_COSINE comment) and is real, checked-in, and listenable.
"""

import json
from pathlib import Path

import pytest

import audio_ingest
import diarize_local
import speaker_id

_AUDIO_DIR = Path(__file__).resolve().parent / "fixtures" / "audio"
_WAV = _AUDIO_DIR / "test_recording_gptaudio.wav"
_META = _AUDIO_DIR / "test_recording_gptaudio_meta.json"

pytestmark = [
    pytest.mark.skipif(
        not speaker_id.is_available(),
        reason="voice deps (torch + speechbrain) not installed",
    ),
    pytest.mark.skipif(
        not (_WAV.exists() and _META.exists()),
        reason="gptaudio fixture missing",
    ),
]


def test_ecapa_clustering_separates_tts_voices():
    # Decode via the SAME production path real uploads use for ECAPA
    # (audio_ingest.decode_to_pcm_16k -> ffmpeg, forced mono 16 kHz) — the
    # fixture is 24 kHz native (OpenAI TTS), and speaker_id.embed_pcm hard-
    # requires speaker_id.TARGET_SR.
    pcm, sr = audio_ingest.decode_to_pcm_16k(_WAV.read_bytes(), _WAV.name)
    assert sr == speaker_id.TARGET_SR

    meta = json.loads(_META.read_text())
    gap = meta["silence_gap_sec"]
    turns, t = [], 0.0
    for m in meta["turns"]:
        turns.append({
            "speaker": m["speaker"], "text": m["text"],
            "start_time": round(t, 4), "end_time": round(t + m["duration_sec"], 4),
        })
        t += m["duration_sec"] + gap
    truth = [t["speaker"] for t in turns]

    # Simulate the nova-3 failure shape: every turn attributed to ONE speaker.
    collapsed = [dict(t, speaker="Speaker A") for t in turns]

    got = diarize_local.diarize_turns(pcm, sr, collapsed)

    assert got is not None, "local diarization returned nothing on a clean fixture"
    assert got["num_speakers"] == 2, (
        f"expected 2 voices, heard {got['num_speakers']} "
        f"(pooled cosine {got['pooled_cosine']:.3f})"
    )
    labels = [t["speaker"] for t in got["turns"]]
    agreement = diarize_local.partition_agreement(truth, labels)
    # This fixture's utterance-level attribution measures 1.0 (exact) from
    # a collapsed start as of the 2026-08-24 threshold recalibration — see
    # test_diarize_regression_ladder.py's non-collapsed version of this same
    # fixture, also pinned at 1.0. Floor set at the measured ceiling; if this
    # regresses, investigate rather than lower the bar.
    assert agreement == 1.0, (
        f"clustering diverged from ground truth (agreement {agreement:.2f}): "
        f"{list(zip(truth, labels))}"
    )
