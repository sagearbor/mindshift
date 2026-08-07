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


# ---------------------------------------------------------------------------
# Word-level speaker-change splitting — a transcriber can weld a speaker
# handoff into ONE utterance; per-word timings let us split it at the change.
# ---------------------------------------------------------------------------

def _words(start: float, end: float, texts: list[str]) -> list[dict]:
    """Evenly spaced word timings across [start, end] for the given texts."""
    n = len(texts)
    step = (end - start) / n
    return [
        {"word": w, "start_time": start + i * step, "end_time": start + (i + 1) * step}
        for i, w in enumerate(texts)
    ]


# A mixed utterance 3.0–11.0s: voice A until 7.0, voice B after — with one
# word per second so 7.0 is exactly a word boundary.
_MIXED_WORDS = _words(3.0, 11.0, ["wA1", "wA2", "wA3", "wA4", "wB1", "wB2", "wB3", "wB4"])


def _mixed_scenario_pcm(seconds: float = 14.0) -> np.ndarray:
    """Voice A fills 0–7s, voice B fills 7–14s."""
    pcm = np.zeros(int(seconds * SR), dtype=np.float32)
    pcm[: int(7.0 * SR)] = VOICE_A
    pcm[int(7.0 * SR):] = VOICE_B
    return pcm


def _fake_centroid(fill: float) -> np.ndarray:
    """Pooled-cluster centroid of a fake voice: embed 1s of its fill value."""
    return _mean_angle_embed(np.full(SR, fill, dtype=np.float32), SR)


_CENTROIDS_AB = (_fake_centroid(VOICE_A), _fake_centroid(VOICE_B))


class TestSustainedFlip:
    """Pure math: a change point is a SUSTAINED run of opposite-sign margins.

    Calibration on the real recording (2026-08-07): single flip candidates and
    tiny margins appear at the EDGES of genuine changes and nowhere else in
    pure utterances — so evidence requires >= SPLIT_SUSTAIN consecutive
    candidates whose weaker margin still clears the floor.
    """

    def test_lone_flip_candidate_is_rejected(self):
        got = diarize_local._sustained_flip(
            [1.0, 2.0, 3.0],
            [+0.5, -0.5, +0.5],   # left margins
            [+0.5, +0.5, +0.5],   # right margins — only b=2.0 flips
            0.15,
        )
        assert got is None

    def test_sustained_flip_returns_time_of_strongest_candidate(self):
        got = diarize_local._sustained_flip(
            [1.0, 2.0, 3.0, 4.0],
            [-0.4, -0.5, -0.3, +0.4],
            [+0.3, +0.45, +0.2, +0.4],  # flips at 1,2,3; strongest q at 2.0
            0.15,
        )
        assert got == 2.0

    def test_weak_margins_do_not_count(self):
        """Opposite signs with a margin under the floor are noise, not voice."""
        got = diarize_local._sustained_flip(
            [1.0, 2.0, 3.0],
            [-0.05, -0.08, -0.06],
            [+0.5, +0.5, +0.5],
            0.15,
        )
        assert got is None

    def test_no_flip_returns_none(self):
        got = diarize_local._sustained_flip(
            [1.0, 2.0, 3.0], [+0.5, +0.6, +0.5], [+0.5, +0.5, +0.6], 0.15,
        )
        assert got is None


class TestFindChangePoint:
    def test_finds_change_inside_mixed_utterance(self):
        pcm = _mixed_scenario_pcm()
        got = diarize_local.find_change_point(
            pcm, SR, 3.0, 11.0, _mean_angle_embed, _CENTROIDS_AB,
        )
        assert got is not None
        assert abs(got - 7.0) <= 2 * diarize_local.SPLIT_HOP_SECONDS

    def test_monologue_has_no_change_point(self):
        pcm = np.full(int(14.0 * SR), VOICE_A, dtype=np.float32)
        assert diarize_local.find_change_point(
            pcm, SR, 3.0, 11.0, _mean_angle_embed, _CENTROIDS_AB,
        ) is None

    def test_similar_voices_have_no_change_point(self):
        """Twin voices (cosine ≈0.99) are not evidence of a speaker change."""
        pcm = np.zeros(int(14.0 * SR), dtype=np.float32)
        pcm[: int(7.0 * SR)] = VOICE_A
        pcm[int(7.0 * SR):] = VOICE_A_TWIN
        centroids = (_fake_centroid(VOICE_A), _fake_centroid(VOICE_A_TWIN))
        assert diarize_local.find_change_point(
            pcm, SR, 3.0, 11.0, _mean_angle_embed, centroids,
        ) is None


class TestSplitTurnAtWordBoundary:
    def test_splits_at_nearest_word_boundary_and_divides_text(self):
        turn = dict(_turn(3.0, 11.0, text="ignored"), words=_MIXED_WORDS)
        got = diarize_local.split_turn_at_word_boundary(turn, 6.8)
        assert got is not None
        left, right = got
        assert (left["start_time"], left["end_time"]) == (3.0, 7.0)
        assert (right["start_time"], right["end_time"]) == (7.0, 11.0)
        assert left["text"] == "wA1 wA2 wA3 wA4"
        assert right["text"] == "wB1 wB2 wB3 wB4"
        assert left["speaker"] == right["speaker"] == turn["speaker"]
        # Pieces are plain turns — internal word plumbing is not propagated.
        assert "words" not in left and "words" not in right

    def test_refuses_split_leaving_a_sliver_piece(self):
        """Snapping must not create a piece shorter than MIN_SECONDS."""
        words = [
            {"word": "a", "start_time": 0.0, "end_time": 0.7},
            {"word": "b", "start_time": 0.7, "end_time": 8.0},
        ]
        turn = dict(_turn(0.0, 8.0), words=words)
        assert diarize_local.split_turn_at_word_boundary(turn, 1.5) is None

    def test_no_words_means_no_split(self):
        assert diarize_local.split_turn_at_word_boundary(_turn(0.0, 8.0), 4.0) is None


class TestSplitLongUtterances:
    def test_mixed_long_utterance_is_split(self):
        turns = [dict(_turn(3.0, 11.0), words=_MIXED_WORDS)]
        pcm = _mixed_scenario_pcm()
        got, stats = diarize_local.split_long_utterances(
            pcm, SR, turns, _mean_angle_embed, _CENTROIDS_AB,
        )
        assert len(got) == 2
        assert stats["split"] == 1 and stats["scanned"] == 1

    def test_long_utterance_without_words_is_skipped_and_logged(self, caplog):
        turns = [_turn(3.0, 11.0)]
        pcm = _mixed_scenario_pcm()
        with caplog.at_level("INFO", logger="diarize_local"):
            got, stats = diarize_local.split_long_utterances(
                pcm, SR, turns, _mean_angle_embed, _CENTROIDS_AB,
            )
        assert got == turns
        assert stats["skipped_no_words"] == 1
        assert any("word timings" in r.message for r in caplog.records)

    def test_short_utterance_is_not_scanned(self):
        """Compute bound: only utterances over the length floor get windowed."""
        turns = [dict(_turn(3.0, 7.5), words=_words(3.0, 7.5, ["a", "b", "c", "d"]))]
        pcm = _mixed_scenario_pcm()
        calls = []

        def counting_embed(pcm_slice, sr):
            calls.append(1)
            return _mean_angle_embed(pcm_slice, sr)

        got, stats = diarize_local.split_long_utterances(
            pcm, SR, turns, counting_embed, _CENTROIDS_AB,
        )
        assert got == turns
        assert stats["skipped_short"] == 1
        assert calls == []

    def test_monologue_long_utterance_is_not_split(self):
        turns = [dict(_turn(3.0, 11.0), words=_MIXED_WORDS)]
        pcm = np.full(int(14.0 * SR), VOICE_A, dtype=np.float32)
        got, stats = diarize_local.split_long_utterances(
            pcm, SR, turns, _mean_angle_embed, _CENTROIDS_AB,
        )
        assert got == turns
        assert stats["scanned"] == 1 and stats["split"] == 0


class TestDiarizeTurnsWithWordSplitting:
    def test_mixed_utterance_split_and_pieces_attributed_to_both_voices(self):
        """The welded-handoff shape: A(0–3), [A then B](3–11), B(11–14)."""
        turns = [
            _turn(0.0, 3.0, text="intro"),
            dict(_turn(3.0, 11.0, text="welded"), words=_MIXED_WORDS),
            _turn(11.0, 14.0, text="outro"),
        ]
        pcm = _mixed_scenario_pcm()
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None
        assert got["split_utterances"] == 1
        out = got["turns"]
        assert [(t["start_time"], t["end_time"]) for t in out] == [
            (0.0, 3.0), (3.0, 7.0), (7.0, 11.0), (11.0, 14.0),
        ]
        assert [t["text"] for t in out] == [
            "intro", "wA1 wA2 wA3 wA4", "wB1 wB2 wB3 wB4", "outro",
        ]
        # The two pieces land in DIFFERENT clusters.
        assert [t["speaker"] for t in out] == [
            "Speaker A", "Speaker A", "Speaker B", "Speaker B",
        ]
        assert all("words" not in t for t in out)

    def test_validated_split_gate_still_applies_after_splitting(self):
        """A found-and-made split still faces MIN_CLUSTER_SECONDS → None.

        Voice B totals 1.2s (own turn) + 1.5s (split piece) = 2.7s — a real
        change point is detected and the utterance IS split, but the second
        voice remains under MIN_CLUSTER_SECONDS, so the whole relabeling is
        rejected exactly as an unsplit weak cluster would be.
        """
        welded_words = _words(
            7.2, 13.5, ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "b1", "b2"],
        )
        turns = [
            _turn(0.0, 6.0),
            _turn(6.0, 7.2),
            dict(_turn(7.2, 13.5), words=welded_words),
        ]
        pcm = np.zeros(int(14.0 * SR), dtype=np.float32)
        pcm[: int(6.0 * SR)] = VOICE_A
        pcm[int(6.0 * SR):int(7.2 * SR)] = VOICE_B
        pcm[int(7.2 * SR):int(12.0 * SR)] = VOICE_A
        pcm[int(12.0 * SR):] = VOICE_B  # welded tail: 1.5s of voice B
        assert diarize_local.diarize_turns(
            pcm, SR, turns, embed_fn=_mean_angle_embed,
        ) is None

    def test_no_words_no_split_diagnostic_is_zero(self):
        """Turns without word timings run the pre-existing flow untouched."""
        turns = [
            _turn(0.0, 2.0), _turn(2.0, 4.0), _turn(4.0, 6.0), _turn(6.0, 8.0),
        ]
        pcm = _voiced_pcm(turns, [VOICE_A, VOICE_B, VOICE_A, VOICE_B], 8.0)
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None
        assert got["split_utterances"] == 0
        assert len(got["turns"]) == 4
