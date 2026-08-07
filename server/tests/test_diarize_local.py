"""Unit tests for diarize_local — vendor-independent speaker diarization.

All tests here run WITHOUT torch/speechbrain: the math is pure numpy, and the
orchestrator accepts an injectable ``embed_fn(pcm_slice, sr)`` so tests drive
it with deterministic fake embeddings. The fake encodes a "voice" into the
audio samples themselves (each voice is a constant fill value; the embedding is
a unit vector whose angle tracks the slice's mean), so POOLED audio embeds to
the blend of its voices — mirroring how real pooled ECAPA embeddings behave.
(A separate live test exercises the real model when the voice deps exist.)

Empirical grounding (calibration on the real photos_share recording + the TTS
fixture, 2026-08-06): per-utterance embeddings do NOT separate cleanly (noisy,
same-speaker cosine can dip below cross-speaker), so the algorithm forces a
2-way split, refines it against POOLED cluster embeddings (those are reliable:
same-voice ≈0.73, different-voice ≈0.19), and only accepts the split when the
pooled centroids are clearly two different voices.
"""

from __future__ import annotations

import numpy as np
import pytest

import diarize_local
import speaker_id
from speaker_id import SpeakerIdUnavailable


SR = 16000


def _turn(start: float, end: float, speaker: str = "Speaker A", text: str = "hi") -> dict:
    return {"speaker": speaker, "text": text, "start_time": start, "end_time": end}


def _voiced_pcm(turns: list[dict], fills: list[float], seconds: float) -> np.ndarray:
    """PCM where each turn's samples are a constant "voice" fill value."""
    pcm = np.zeros(int(seconds * SR), dtype=np.float32)
    for t, fill in zip(turns, fills):
        pcm[int(t["start_time"] * SR):int(t["end_time"] * SR)] = fill
    return pcm


def _mean_angle_embed(pcm_slice: np.ndarray, sr: int) -> np.ndarray:
    """Fake ECAPA: unit vector whose angle tracks the slice's mean value.

    Distinct fills → orthogonal-ish vectors; pooled same-fill audio → the same
    vector; pooled MIXED audio → a blend between the two voices.
    """
    m = float(np.clip(np.mean(pcm_slice), -1.0, 1.0))
    a = (m + 1.0) * (np.pi / 2)  # mean -1 → 0 rad; mean +1 → pi
    return np.array([np.cos(a), np.sin(a)], dtype=np.float32)


VOICE_A = 0.5    # embeds at 3π/4
VOICE_B = -0.5   # embeds at π/4 — cosine(A, B) = 0 → clearly different
VOICE_A_TWIN = 0.4  # embeds close to VOICE_A — cosine ≈ 0.99 → same voice


# ---------------------------------------------------------------------------
# partition_agreement — pairwise (Rand) agreement between two labelings
# ---------------------------------------------------------------------------

class TestPartitionAgreement:
    def test_identical_partitions_agree_fully(self):
        assert diarize_local.partition_agreement(["A", "A", "B"], ["A", "A", "B"]) == 1.0

    def test_label_names_do_not_matter(self):
        assert diarize_local.partition_agreement(["A", "A", "B"], ["x", "x", "y"]) == 1.0

    def test_disagreement_is_fractional(self):
        got = diarize_local.partition_agreement(["A", "B", "B"], ["A", "A", "B"])
        assert got == pytest.approx(1 / 3)

    def test_fewer_than_two_items_is_full_agreement(self):
        assert diarize_local.partition_agreement(["A"], ["B"]) == 1.0
        assert diarize_local.partition_agreement([], []) == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            diarize_local.partition_agreement(["A"], ["A", "B"])


# ---------------------------------------------------------------------------
# diarize_turns — force-2 split + pooled refinement + validation
# ---------------------------------------------------------------------------

class TestDiarizeTurns:
    def test_splits_alternating_voices_collapsed_to_one_speaker(self):
        """The nova-3 failure shape: two voices, transcript said one speaker."""
        turns = [
            _turn(0.0, 2.0), _turn(2.0, 4.0), _turn(4.0, 6.0), _turn(6.0, 8.0),
        ]
        pcm = _voiced_pcm(turns, [VOICE_A, VOICE_B, VOICE_A, VOICE_B], 8.0)
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None
        assert [t["speaker"] for t in got["turns"]] == [
            "Speaker A", "Speaker B", "Speaker A", "Speaker B",
        ]
        assert got["num_speakers"] == 2

    def test_single_voice_is_never_split(self):
        """A genuine monologue must NOT be forced into two phantom speakers."""
        turns = [_turn(0.0, 2.0), _turn(2.0, 4.0), _turn(4.0, 6.0), _turn(6.0, 8.0)]
        pcm = _voiced_pcm(turns, [VOICE_A] * 4, 8.0)
        assert diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed) is None

    def test_similar_voices_rejected_not_guessed(self):
        """Two barely-different voices → pooled centroids too similar → None."""
        turns = [_turn(0.0, 2.0), _turn(2.0, 4.0), _turn(4.0, 6.0), _turn(6.0, 8.0)]
        pcm = _voiced_pcm(turns, [VOICE_A, VOICE_A_TWIN, VOICE_A, VOICE_A_TWIN], 8.0)
        assert diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed) is None

    def test_tiny_second_cluster_rejected(self):
        """A "second voice" with under ~3s of speech is too little evidence."""
        turns = [_turn(0.0, 3.0), _turn(3.0, 6.0), _turn(6.0, 7.2)]
        pcm = _voiced_pcm(turns, [VOICE_A, VOICE_A, VOICE_B], 8.0)
        assert diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed) is None

    def test_preserves_text_and_times(self):
        turns = [
            _turn(0.0, 2.0, text="one"), _turn(2.0, 4.0, text="two"),
            _turn(4.0, 6.0, text="three"), _turn(6.0, 8.0, text="four"),
        ]
        pcm = _voiced_pcm(turns, [VOICE_A, VOICE_B, VOICE_A, VOICE_B], 8.0)
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert [(t["text"], t["start_time"], t["end_time"]) for t in got["turns"]] == [
            ("one", 0.0, 2.0), ("two", 2.0, 4.0), ("three", 4.0, 6.0), ("four", 6.0, 8.0),
        ]

    def test_speaker_names_assigned_by_first_appearance(self):
        turns = [_turn(0.0, 2.0), _turn(2.0, 4.0), _turn(4.0, 6.0), _turn(6.0, 8.0)]
        # Voice B speaks FIRST → it must be named "Speaker A".
        pcm = _voiced_pcm(turns, [VOICE_B, VOICE_A, VOICE_B, VOICE_A], 8.0)
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert [t["speaker"] for t in got["turns"]] == [
            "Speaker A", "Speaker B", "Speaker A", "Speaker B",
        ]

    def test_short_turn_inherits_nearest_embedded_neighbor(self):
        turns = [
            _turn(0.0, 2.0), _turn(2.0, 4.0), _turn(4.0, 6.0), _turn(6.0, 8.0),
            _turn(8.1, 8.4),  # 0.3s — below min_seconds, nearest is the 6–8 turn
        ]
        pcm = _voiced_pcm(
            turns, [VOICE_A, VOICE_B, VOICE_A, VOICE_B, VOICE_A], 9.0,
        )
        got = diarize_local.diarize_turns(
            pcm, SR, turns, embed_fn=_mean_angle_embed, min_seconds=1.0,
        )
        assert [t["speaker"] for t in got["turns"]] == [
            "Speaker A", "Speaker B", "Speaker A", "Speaker B", "Speaker B",
        ]

    def test_returns_none_when_too_few_embeddable_turns(self):
        turns = [_turn(0.0, 2.0), _turn(2.0, 2.2)]
        pcm = _voiced_pcm(turns, [VOICE_A, VOICE_B], 3.0)
        assert diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed) is None

    def test_returns_none_when_model_unavailable(self):
        def unavailable(pcm_slice, sr):
            raise SpeakerIdUnavailable("no torch here")

        turns = [_turn(0.0, 2.0), _turn(2.0, 4.0)]
        pcm = _voiced_pcm(turns, [VOICE_A, VOICE_B], 4.0)
        assert diarize_local.diarize_turns(pcm, SR, turns, embed_fn=unavailable) is None

    @pytest.mark.skipif(
        speaker_id.is_available(), reason="voice deps installed — real model would load"
    )
    def test_default_embedder_degrades_to_none_without_voice_deps(self):
        """No torch/speechbrain → honest None, never an ImportError crash."""
        turns = [_turn(0.0, 2.0), _turn(2.0, 4.0)]
        assert diarize_local.diarize_turns(_voiced_pcm(turns, [0.1, 0.2], 4.0), SR, turns) is None

    def test_reports_diagnostics(self):
        turns = [
            _turn(0.0, 2.0), _turn(2.0, 4.0), _turn(4.0, 6.0), _turn(6.0, 8.0),
            _turn(8.1, 8.4),
        ]
        pcm = _voiced_pcm(
            turns, [VOICE_A, VOICE_B, VOICE_A, VOICE_B, VOICE_A], 9.0,
        )
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got["source"] == "local-ecapa"
        assert got["segments_total"] == 5
        assert got["segments_embedded"] == 4
        # Two clearly different fake voices → pooled centroids near-orthogonal.
        assert got["pooled_cosine"] < diarize_local.MAX_POOLED_COSINE
        # Input said ONE speaker everywhere; we split → agreement < 1.
        assert 0.0 <= got["agreement_with_input"] < 1.0
