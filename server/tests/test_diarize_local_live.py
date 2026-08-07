"""Live-gated test: local ECAPA diarization separates the TTS fixture voices.

The positive counterpart to test_audio_upload_live's nova-3 xfail: Deepgram's
nova-3 (2025-07-31.0) can no longer diarize these synthetic voices, so OUR
clustering must — that is the whole point of diarize_local. Needs the optional
voice deps (torch + speechbrain; first run downloads the pinned ECAPA
checkpoint) and the generated fixture — skipped honestly when either is absent.
No Deepgram key needed: ground-truth turn boundaries come from the fixture's
metadata, not a live transcription.
"""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

import diarize_local
import speaker_id

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_WAV = _REPO_ROOT / "tmp" / "test_recording.wav"
_META = _REPO_ROOT / "tmp" / "test_recording_meta.json"

pytestmark = [
    pytest.mark.skipif(
        not speaker_id.is_available(),
        reason="voice deps (torch + speechbrain) not installed",
    ),
    pytest.mark.skipif(
        not (_WAV.exists() and _META.exists()),
        reason="TTS fixture missing — generate with scripts/make_test_recording.py",
    ),
]


def test_ecapa_clustering_separates_tts_voices():
    with wave.open(str(_WAV), "rb") as wf:
        sr = wf.getframerate()
        pcm = (
            np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            .astype(np.float32) / 32768.0
        )
    meta = json.loads(_META.read_text())
    truth = [t["speaker"] for t in meta["turns"]]

    # Simulate the nova-3 failure shape: every turn attributed to ONE speaker.
    collapsed = [dict(t, speaker="Speaker A") for t in meta["turns"]]

    got = diarize_local.diarize_turns(pcm, sr, collapsed)

    assert got is not None, "local diarization returned nothing on a clean fixture"
    assert got["num_speakers"] == 2, (
        f"expected 2 voices, heard {got['num_speakers']} "
        f"(pooled cosine {got['pooled_cosine']:.3f})"
    )
    labels = [t["speaker"] for t in got["turns"]]
    agreement = diarize_local.partition_agreement(truth, labels)
    # Utterance-level attribution has a measured ceiling on this fixture
    # (calibration 2026-08-06: 0.75 — the Aura TTS voices embed noisily at
    # single-utterance granularity). The split itself must be found; the floor
    # guards against regressions without demanding the impossible.
    assert agreement >= 0.7, (
        f"clustering diverged from ground truth (agreement {agreement:.2f}): "
        f"{list(zip(truth, labels))}"
    )
