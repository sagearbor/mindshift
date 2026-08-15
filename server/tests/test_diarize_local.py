"""Unit tests for diarize_local — vendor-independent speaker diarization.

All tests here run WITHOUT torch/speechbrain: the math is pure numpy, and the
orchestrator accepts an injectable ``embed_fn(pcm_slice, sr)`` so tests drive
it with deterministic fake embeddings. The fake encodes a "voice" into the
audio samples themselves (each voice is a constant fill value; the embedding is
a unit vector whose angle tracks the slice's mean), so POOLED audio embeds to
the blend of its voices — mirroring how real pooled ECAPA embeddings behave.
(A separate live test exercises the real model when the voice deps exist.)

Empirical grounding (calibration on the real photos_share recording + the TTS
fixture 2026-08-06, N-way extension on the real 3-person recording
2026-08-14): per-utterance embeddings do NOT separate cleanly (noisy,
same-speaker cosine can dip below cross-speaker), so the algorithm merges to
each candidate k = 2..MAX_SPEAKERS_LOCAL, refines against POOLED cluster
embeddings (those are reliable: same-voice ≈0.73, different-voice ≈0.19), and
accepts the LARGEST k whose every centroid pair is clearly a different voice
and whose every cluster carries enough pooled speech.
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

# cosine(A, A_WEAK) ≈ 0.40 — passes the 2-voice gate (≤ MAX_POOLED_COSINE)
# but is NOT strongly separated (> STRONG_SEPARATION_COSINE), so a cluster of
# this voice needs the full MIN_CLUSTER_SECONDS of speech.
VOICE_A_WEAK = -0.238

# Three mutually distinct voices for the N-way tests: fills -1 / 0 / +1 embed
# at 0 / π/2 / π, so pairwise cosines are 0, 0, -1 — every pair clears both
# the accept gate AND the strong-separation bar.
VOICE_P = -1.0
VOICE_Q = 0.0
VOICE_R = 1.0


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
        # k-selection diagnostics: every k tried is reported with its verdict.
        ks = [e["k"] for e in got["k_evaluated"]]
        assert ks == sorted(ks)
        assert set(ks) <= {2, 3, 4}
        assert 2 in ks
        by_k = {e["k"]: e for e in got["k_evaluated"]}
        assert by_k[2]["ok"] is True
        # Two-voice input: every k above 2 must have been tried and REJECTED.
        assert all(not by_k[k]["ok"] for k in ks if k > 2)


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
        turns = [dict(_turn(3.0, 5.8), words=_words(3.0, 5.8, ["a", "b", "c", "d"]))]
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
        """A found-and-made split still faces the per-cluster evidence floor.

        The second voice totals 1.2s (own turn) + 1.5s (split piece) = 2.7s —
        a real change point is detected and the utterance IS split, but the
        voice is only MODERATELY separated (cosine ≈0.40 to voice A: past the
        accept gate, NOT past the strong-separation bar), so 2.7s stays under
        the full MIN_CLUSTER_SECONDS it needs and the whole relabeling is
        rejected exactly as an unsplit weak cluster would be.
        """
        welded_words = _words(
            7.5, 13.5, ["a1", "a2", "a3", "a4", "a5", "a6", "b1", "b2"],
        )  # step 0.75 → a word boundary lands exactly on the 12.0s change
        turns = [
            _turn(0.0, 6.0),
            _turn(6.0, 7.2),
            dict(_turn(7.5, 13.5), words=welded_words),
        ]
        pcm = np.zeros(int(14.0 * SR), dtype=np.float32)
        pcm[: int(6.0 * SR)] = VOICE_A
        pcm[int(6.0 * SR):int(7.2 * SR)] = VOICE_A_WEAK
        pcm[int(7.5 * SR):int(12.0 * SR)] = VOICE_A
        pcm[int(12.0 * SR):] = VOICE_A_WEAK  # welded tail: 1.5s
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


# ---------------------------------------------------------------------------
# Per-word rapid-exchange splitting — the sustained-flip scan needs
# SPLIT_SUSTAIN consecutive 1.5s windows per side, so a rapid multi-voice
# exchange (a ~1s interjection, a fast A/B/A volley) welded into ONE
# utterance can never satisfy it (measured live 2026-08-14: correct 3-voice
# k-selection but 89%/5%/5% attribution, "0 utterance(s) split"). The word
# pass labels EACH word against ALL k pooled centroids and splits at
# smoothed label-run boundaries.
# ---------------------------------------------------------------------------


class TestWordLabeling:
    def test_labels_words_by_nearest_centroid_with_margins(self):
        turn = dict(_turn(3.0, 11.0), words=_MIXED_WORDS)
        labels, margins = diarize_local._label_words(
            _mixed_scenario_pcm(), SR, turn, _mean_angle_embed,
            list(_CENTROIDS_AB),
        )
        assert labels == [0, 0, 0, 0, 1, 1, 1, 1]
        assert all(m > diarize_local.WORD_MIN_MARGIN for m in margins)

    def test_ambiguous_word_has_low_margin(self):
        """A word whose window favors neither centroid scores near-zero."""
        words = _words(3.0, 5.0, ["x", "y"])
        turn = dict(_turn(3.0, 5.0), words=words)
        pcm = np.zeros(int(6.0 * SR), dtype=np.float32)  # fill 0 = neither voice
        _, margins = diarize_local._label_words(
            pcm, SR, turn, _mean_angle_embed, list(_CENTROIDS_AB),
        )
        assert all(m < diarize_local.WORD_MIN_MARGIN for m in margins)


class TestSmoothWordLabels:
    def test_ambiguous_words_inherit_nearest_confident_label(self):
        got = diarize_local._smooth_word_labels(
            [0, 0, 1, 0, 1, 1], [0.5, 0.5, 0.02, 0.03, 0.5, 0.5],
            min_margin=0.1,
        )
        assert got == [0, 0, 0, 1, 1, 1]

    def test_tie_goes_to_the_earlier_confident_word(self):
        got = diarize_local._smooth_word_labels(
            [0, 1, 1], [0.5, 0.02, 0.5], min_margin=0.1,
        )
        assert got == [0, 0, 1]

    def test_no_confident_word_returns_none(self):
        assert diarize_local._smooth_word_labels(
            [0, 1, 0], [0.02, 0.03, 0.04], min_margin=0.1,
        ) is None


class TestCollapseWordRuns:
    def test_single_flipped_word_merges_into_larger_neighbor(self):
        words = _words(3.0, 11.0, list("abcdefgh"))
        got = diarize_local._collapse_word_runs(
            words, [0, 0, 0, 0, 1, 0, 0, 0], 3.0, 11.0,
        )
        assert got == [0] * 8

    def test_sub_second_run_merges_even_with_enough_words(self):
        # Words f/g/h span 10.2–11.0: a 2-word voice-1 run whose piece would
        # last 0.8s < MIN_SECONDS — too little signal to attribute.
        words = [
            {"word": w, "start_time": s, "end_time": e}
            for w, s, e in [
                ("a", 3.0, 4.2), ("b", 4.2, 5.4), ("c", 5.4, 6.6),
                ("d", 6.6, 7.8), ("e", 7.8, 9.0), ("f", 9.0, 10.2),
                ("g", 10.2, 10.6), ("h", 10.6, 11.0),
            ]
        ]
        got = diarize_local._collapse_word_runs(
            words, [0, 0, 0, 0, 0, 0, 1, 1], 3.0, 11.0,
        )
        assert got == [0] * 8

    def test_trustworthy_runs_survive(self):
        words = _words(3.0, 11.0, list("abcdefgh"))
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        got = diarize_local._collapse_word_runs(words, labels, 3.0, 11.0)
        assert got == labels


class TestSplitTurnAtWordRuns:
    def test_multiway_split_divides_text_and_times(self):
        turn = dict(_turn(3.0, 13.0), words=_words(3.0, 13.0, [
            "a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2", "e1", "e2",
        ]))
        got = diarize_local.split_turn_at_word_runs(
            turn, [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        )
        assert got is not None
        assert [(p["start_time"], p["end_time"]) for p in got] == [
            (3.0, 5.0), (5.0, 7.0), (7.0, 9.0), (9.0, 11.0), (11.0, 13.0),
        ]
        assert [p["text"] for p in got] == [
            "a1 a2", "b1 b2", "c1 c2", "d1 d2", "e1 e2",
        ]
        assert all("words" not in p for p in got)

    def test_single_run_is_no_split(self):
        turn = dict(_turn(3.0, 11.0), words=_MIXED_WORDS)
        assert diarize_local.split_turn_at_word_runs(turn, [0] * 8) is None


class TestRapidExchangeSplitting:
    def test_rapid_alternation_yields_multiway_split(self):
        """A fast A/B volley welded into ONE utterance splits at every turn."""
        texts = ["a1", "a2", "b1", "b2", "a3", "a4", "b3", "b4", "a5", "a6"]
        turns = [dict(_turn(3.0, 13.0), words=_words(3.0, 13.0, texts))]
        fills = [VOICE_A, VOICE_A, VOICE_B, VOICE_B, VOICE_A,
                 VOICE_A, VOICE_B, VOICE_B, VOICE_A, VOICE_A]
        pcm = np.full(int(14.0 * SR), VOICE_A, dtype=np.float32)
        for w, f in zip(turns[0]["words"], fills):
            pcm[int(w["start_time"] * SR):int(w["end_time"] * SR)] = f
        got, stats = diarize_local.split_long_utterances(
            pcm, SR, turns, _mean_angle_embed, _CENTROIDS_AB,
        )
        assert stats == {
            "scanned": 1, "split": 1, "skipped_short": 0, "skipped_no_words": 0,
        }
        assert [(p["start_time"], p["end_time"]) for p in got] == [
            (3.0, 5.0), (5.0, 7.0), (7.0, 9.0), (9.0, 11.0), (11.0, 13.0),
        ]
        assert [p["text"] for p in got] == [
            "a1 a2", "b1 b2", "a3 a4", "b3 b4", "a5 a6",
        ]

    def test_confident_lone_flip_is_smoothed_not_split_and_no_fallback(self):
        """One flipped word is noise; the word verdict is not overruled.

        The audio really does contain 1s of the other voice, and the
        sustained-flip scan WOULD split here (its 1.5s right window sees the
        interjection) — but a piece this short cannot be attributed honestly,
        so the confident per-word verdict (smoothed away) must stand.
        """
        texts = ["a1", "a2", "a3", "a4", "b1", "a5", "a6", "a7"]
        turns = [dict(_turn(3.0, 11.0), words=_words(3.0, 11.0, texts))]
        pcm = np.full(int(14.0 * SR), VOICE_A, dtype=np.float32)
        pcm[int(7.0 * SR):int(8.0 * SR)] = VOICE_B
        got, stats = diarize_local.split_long_utterances(
            pcm, SR, turns, _mean_angle_embed, _CENTROIDS_AB,
        )
        assert got == turns
        assert stats["scanned"] == 1 and stats["split"] == 0

    def test_ambiguous_middle_words_inherit_surrounding_voices(self):
        """Words that favor neither centroid split with their neighbors."""
        texts = ["a1", "a2", "a3", "a4", "x1", "x2", "b1", "b2", "b3", "b4"]
        turns = [dict(_turn(3.0, 13.0), words=_words(3.0, 13.0, texts))]
        pcm = np.zeros(int(14.0 * SR), dtype=np.float32)
        pcm[: int(7.0 * SR)] = VOICE_A          # a-words (3–7) pure A
        pcm[int(7.0 * SR):int(9.0 * SR)] = 0.0  # x-words: neither voice
        pcm[int(9.0 * SR):] = VOICE_B           # b-words (9–13) pure B
        got, stats = diarize_local.split_long_utterances(
            pcm, SR, turns, _mean_angle_embed, _CENTROIDS_AB,
        )
        assert stats["split"] == 1
        assert [(p["start_time"], p["end_time"]) for p in got] == [
            (3.0, 8.0), (8.0, 13.0),
        ]
        assert [p["text"] for p in got] == ["a1 a2 a3 a4 x1", "x2 b1 b2 b3 b4"]

    def test_word_pass_scans_utterances_the_sustained_scan_skipped(self):
        """A 4s welded utterance (under SPLIT_MIN_UTTERANCE_SECONDS) splits."""
        texts = ["a1", "a2", "a3", "a4", "a5", "a6", "b1", "b2"]
        turns = [dict(_turn(2.0, 6.0), words=_words(2.0, 6.0, texts))]
        pcm = np.full(int(9.0 * SR), VOICE_A, dtype=np.float32)
        pcm[int(5.0 * SR):] = VOICE_B
        got, stats = diarize_local.split_long_utterances(
            pcm, SR, turns, _mean_angle_embed, _CENTROIDS_AB,
        )
        assert stats["scanned"] == 1 and stats["split"] == 1
        assert [(p["start_time"], p["end_time"]) for p in got] == [
            (2.0, 5.0), (5.0, 6.0),
        ]
        assert [p["text"] for p in got] == ["a1 a2 a3 a4 a5 a6", "b1 b2"]

    def test_inconclusive_word_pass_falls_back_to_sustained_flip(self):
        """No confident word anywhere → the original 1.5s-window scan runs."""
        ambiguous = np.array([0.0, 1.0], dtype=np.float32)

        def coarse_embed(pcm_slice, sr):
            # Sub-1.2s windows (the word pass') carry no signal in this fake;
            # the fallback's 1.5s windows embed normally.
            if pcm_slice.size < int(1.2 * sr):
                return ambiguous
            return _mean_angle_embed(pcm_slice, sr)

        turns = [dict(_turn(3.0, 11.0), words=_MIXED_WORDS)]
        got, stats = diarize_local.split_long_utterances(
            _mixed_scenario_pcm(), SR, turns, coarse_embed, _CENTROIDS_AB,
        )
        assert stats["scanned"] == 1 and stats["split"] == 1
        assert [(p["start_time"], p["end_time"]) for p in got] == [
            (3.0, 7.0), (7.0, 11.0),
        ]


class TestDiarizeTurnsRapidExchange:
    def test_one_second_interjection_is_split_and_attributed(self):
        """The Duolingo shape: a ~1s second voice inside a 4s utterance."""
        welded_words = _words(2.0, 6.0, ["q1", "q2", "q3", "q4", "q5", "q6",
                                         "i1", "i2"])
        turns = [
            _turn(0.0, 2.0, text="dad intro"),
            dict(_turn(2.0, 6.0, text="welded"), words=welded_words),
            _turn(6.0, 9.0, text="kid more"),
        ]
        pcm = np.full(int(9.0 * SR), VOICE_A, dtype=np.float32)
        pcm[int(5.0 * SR):] = VOICE_B
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None
        assert got["num_speakers"] == 2
        assert got["split_utterances"] == 1
        assert [(t["text"], t["speaker"]) for t in got["turns"]] == [
            ("dad intro", "Speaker A"),
            ("q1 q2 q3 q4 q5 q6", "Speaker A"),
            ("i1 i2", "Speaker B"),
            ("kid more", "Speaker B"),
        ]
        assert [(t["start_time"], t["end_time"]) for t in got["turns"]] == [
            (0.0, 2.0), (2.0, 5.0), (5.0, 6.0), (6.0, 9.0),
        ]

    def test_three_voice_welded_exchange_attributed_to_all_three(self):
        """The maggiano's shape: a rapid 3-voice exchange in ONE utterance.

        The word pass scores against ALL THREE pooled centroids from the
        first k-selection — a 2-way margin contrast is structurally blind to
        the third voice.
        """
        welded_words = _words(9.0, 18.0, [
            "p1", "p2", "p3", "q1", "q2", "q3", "r1", "r2", "r3",
        ])
        turns = [
            _turn(0.0, 3.0, text="pp"), _turn(3.0, 6.0, text="qq"),
            _turn(6.0, 9.0, text="rr"),
            dict(_turn(9.0, 18.0, text="welded"), words=welded_words),
        ]
        pcm = np.zeros(int(18.0 * SR), dtype=np.float32)
        for lo, hi, fill in [
            (0, 3, VOICE_P), (3, 6, VOICE_Q), (6, 9, VOICE_R),
            (9, 12, VOICE_P), (12, 15, VOICE_Q), (15, 18, VOICE_R),
        ]:
            pcm[int(lo * SR):int(hi * SR)] = fill
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None
        assert got["num_speakers"] == 3
        assert got["split_utterances"] == 1
        assert [t["speaker"] for t in got["turns"]] == [
            "Speaker A", "Speaker B", "Speaker C",
            "Speaker A", "Speaker B", "Speaker C",
        ]
        assert [(t["start_time"], t["end_time"]) for t in got["turns"][3:]] == [
            (9.0, 12.0), (12.0, 15.0), (15.0, 18.0),
        ]
        # Pieces may not mint NEW speakers: the post-split re-selection is
        # capped at round 1's validated k (3 here), so k=4 is never tried
        # even though six embeddable segments now exist.
        assert [e["k"] for e in got["k_evaluated"]] == [2, 3]
        # Attribution is balanced — no 89%-style single-speaker pile-up.
        share: dict[str, float] = {}
        for t in got["turns"]:
            share[t["speaker"]] = share.get(t["speaker"], 0.0) + (
                t["end_time"] - t["start_time"]
            )
        assert max(share.values()) / sum(share.values()) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# N-way k-selection — evaluate k = 2..MAX_SPEAKERS_LOCAL, choose the LARGEST
# k whose every centroid pair and every cluster's evidence floor validates.
# Driven by the real maggiano's 3-person recording (2026-08-14): the transcript
# heard ONE voice, forcing k=2 buried a real third voice that k=3 separates
# cleanly — but only if a VERY distinct cluster may ride the relaxed
# MIN_CLUSTER_SECONDS_STRONG floor instead of the full MIN_CLUSTER_SECONDS.
# ---------------------------------------------------------------------------

# A third voice between VOICE_P and VOICE_R: cosine 0.40 to P, -0.40 to R —
# past the accept gate (≤ MAX_POOLED_COSINE) but NOT strongly separated
# (> STRONG_SEPARATION_COSINE), so its cluster needs the full 3.0s floor.
VOICE_W_MID = -0.262


def _by_k(got: dict) -> dict[int, dict]:
    return {e["k"]: e for e in got["k_evaluated"]}


class TestKSelection:
    def test_three_voices_collapsed_to_one_speaker_yield_three(self):
        """The maggiano's failure shape: 3 voices, transcript said ONE."""
        turns = [
            _turn(0.0, 2.0), _turn(2.0, 4.0), _turn(4.0, 6.0), _turn(6.0, 8.0),
            _turn(8.0, 9.7), _turn(9.7, 11.4),  # third voice: 3.4s total
        ]
        pcm = _voiced_pcm(
            turns, [VOICE_P, VOICE_R, VOICE_P, VOICE_R, VOICE_Q, VOICE_Q], 12.0,
        )
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None
        assert got["num_speakers"] == 3
        assert [t["speaker"] for t in got["turns"]] == [
            "Speaker A", "Speaker B", "Speaker A", "Speaker B",
            "Speaker C", "Speaker C",
        ]
        by_k = _by_k(got)
        assert by_k[3]["ok"] is True
        # The k=4 split of these three voices is spurious → tried and rejected.
        assert by_k[4]["ok"] is False

    def test_two_voices_are_never_upgraded_to_three(self):
        """k-selection must not invent a third speaker in a 2-voice recording."""
        turns = [
            _turn(0.0, 2.0), _turn(2.0, 4.0), _turn(4.0, 6.0), _turn(6.0, 8.0),
        ]
        pcm = _voiced_pcm(turns, [VOICE_P, VOICE_R, VOICE_P, VOICE_R], 8.0)
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None
        assert got["num_speakers"] == 2
        by_k = _by_k(got)
        assert by_k[2]["ok"] is True
        assert not by_k[3]["ok"] and not by_k[4]["ok"]

    def test_tiny_strongly_distinct_third_voice_accepted(self):
        """1.6s of a VERY distinct voice clears the relaxed strong floor."""
        turns = [
            _turn(0.0, 2.0), _turn(2.0, 4.0), _turn(4.0, 6.0), _turn(6.0, 8.0),
            _turn(8.0, 9.6),  # 1.6s ≥ MIN_CLUSTER_SECONDS_STRONG
        ]
        pcm = _voiced_pcm(
            turns, [VOICE_P, VOICE_R, VOICE_P, VOICE_R, VOICE_Q], 10.0,
        )
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None
        assert got["num_speakers"] == 3
        assert [t["speaker"] for t in got["turns"]] == [
            "Speaker A", "Speaker B", "Speaker A", "Speaker B", "Speaker C",
        ]

    def test_tiny_third_voice_below_strong_floor_rejected(self):
        """Even a VERY distinct voice needs MIN_CLUSTER_SECONDS_STRONG of speech."""
        turns = [
            _turn(0.0, 2.0), _turn(2.0, 4.0), _turn(4.0, 6.0), _turn(6.0, 8.0),
            _turn(8.0, 9.2),  # 1.2s < MIN_CLUSTER_SECONDS_STRONG
        ]
        pcm = _voiced_pcm(
            turns, [VOICE_P, VOICE_R, VOICE_P, VOICE_R, VOICE_Q], 10.0,
        )
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None
        assert got["num_speakers"] == 2
        assert _by_k(got)[3]["ok"] is False

    def test_weak_third_voice_needs_the_full_floor(self):
        """A moderately-separated voice (cos ≈0.40) gets NO floor relaxation.

        2.0s of it passes the accept gate at k=3 but is neither ≥
        MIN_CLUSTER_SECONDS nor strongly separated → k=3 rejected, the
        recording honestly stays a validated 2-way split.
        """
        turns = [
            _turn(0.0, 2.0), _turn(2.0, 4.0), _turn(4.0, 6.0), _turn(6.0, 8.0),
            _turn(8.0, 10.0),  # 2.0s of the weak middle voice
        ]
        pcm = _voiced_pcm(
            turns, [VOICE_P, VOICE_R, VOICE_P, VOICE_R, VOICE_W_MID], 11.0,
        )
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None
        assert got["num_speakers"] == 2
        assert _by_k(got)[3]["ok"] is False

    def test_same_voice_registers_do_not_become_a_third_speaker(self):
        """The couple-file regression shape (live calibration 2026-08-14).

        One real voice heard in two registers (calm vs shouting) can form two
        clusters that pass the 0.45 gate with PLENTY of speech each — on the
        real couple recording the two registers measured pooled cosine 0.359
        with 6+s per cluster, so no seconds floor can reject them. What does:
        the pair a k→k+1 split CREATES must be VERY clearly two voices
        (≤ STRONG_SEPARATION_COSINE); one-voice-two-registers (here 0.35)
        fails that bar and the recording honestly stays 2 speakers.
        """
        # V1 register a (fill -1, 0 rad), V1 register b (cos 0.35 to a),
        # V2 (fill +1, π) — registers 8s each, far more than any floor.
        reg_b = -0.2277  # angle arccos(0.35) ≈ 69.5° → cos(a, b) = 0.35
        turns = [
            _turn(0.0, 4.0), _turn(4.0, 8.0),      # V1 register a
            _turn(8.0, 12.0), _turn(12.0, 16.0),   # V1 register b
            _turn(16.0, 20.0), _turn(20.0, 24.0),  # V2
        ]
        pcm = _voiced_pcm(
            turns, [VOICE_P, VOICE_P, reg_b, reg_b, VOICE_R, VOICE_R], 25.0,
        )
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None
        assert got["num_speakers"] == 2
        assert _by_k(got)[3]["ok"] is False

    def test_unanchored_split_does_not_become_a_third_speaker(self):
        """The TTS-fixture regression shape (live calibration 2026-08-14).

        Noisy same-voice utterances can split into two clusters whose pair
        cosine (0.28 measured on the fixture: 0.277) slips UNDER the 0.30
        marginal-split bar with 5+s per half — nearly identical to the real
        third voice's 0.267. What separates them is the ANCHOR: a genuine
        new voice is wildly unlike an existing cluster (the real child vs
        her father: -0.017), while BOTH halves of a phantom split sit
        moderately far from everything (fixture: 0.216 / 0.238). A marginal
        split whose halves both exceed NEW_VOICE_ANCHOR_COSINE against every
        non-sibling cluster is rejected.
        """
        # Three directions with a controlled Gram matrix: registers r1, r2 of
        # one voice at cos 0.28 to each other, both at cos 0.20 to voice V —
        # all pairs pass the 0.45 gate, the split pair passes 0.30, but
        # neither half anchors (0.20 > 0.15).
        gram = np.array([
            [1.0, 0.28, 0.20],
            [0.28, 1.0, 0.20],
            [0.20, 0.20, 1.0],
        ])
        vecs = np.linalg.cholesky(gram)  # rows: unit vectors with that Gram
        fills = (-1.0, 0.0, 1.0)

        def blend_embed(pcm_slice: np.ndarray, sr: int) -> np.ndarray:
            w = np.array([
                float(np.mean(np.abs(pcm_slice - f) < 0.05)) for f in fills
            ])
            v = w @ vecs
            return (v / np.linalg.norm(v)).astype(np.float32)

        turns = [
            _turn(0.0, 4.0), _turn(4.0, 8.0),      # register r1 (fill -1)
            _turn(8.0, 12.0), _turn(12.0, 16.0),   # register r2 (fill 0)
            _turn(16.0, 20.0), _turn(20.0, 24.0),  # voice V (fill +1)
        ]
        pcm = _voiced_pcm(turns, [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0], 25.0)
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=blend_embed)
        assert got is not None
        assert got["num_speakers"] == 2
        assert _by_k(got)[3]["ok"] is False

    def test_select_k_respects_max_k(self):
        """The post-split re-selection cap: candidate ks stop at max_k."""
        turns = [_turn(2.0 * i, 2.0 * (i + 1)) for i in range(6)]
        fills = [VOICE_P, VOICE_R, VOICE_Q, VOICE_P, VOICE_R, VOICE_Q]
        pcm = _voiced_pcm(turns, fills, 12.0)
        embedded = diarize_local._embed_turns(
            pcm, SR, turns, _mean_angle_embed, 1.0,
        )
        order, embs = embedded
        k_eval, chosen = diarize_local._select_k(
            pcm, SR, turns, _mean_angle_embed, order, embs,
            diarize_local.MAX_POOLED_COSINE, max_k=2,
        )
        assert [e["k"] for e in k_eval] == [2]
        # Three genuinely distinct voices forced through a 2-way lens: the
        # merged pair's pooled centroid stays separable from the third, so
        # k=2 validates — but k=3 was never evaluated.
        assert chosen is not None and len(chosen[1]) == 2

    def test_k_is_capped_by_embeddable_turns(self):
        turns = [_turn(0.0, 3.5), _turn(3.5, 7.0), _turn(7.0, 10.5)]
        pcm = _voiced_pcm(turns, [VOICE_P, VOICE_R, VOICE_Q], 11.0)
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None
        assert got["num_speakers"] == 3
        assert [e["k"] for e in got["k_evaluated"]] == [2, 3]


# Four mutually orthogonal voices need more room than the 2-D angle fake
# offers: each voice is a basis direction, a slice embeds to the
# duration-weighted blend of the voices it contains (pooled behavior matches
# the mean-angle fake's).
_FILLS_4 = (-1.0, -1 / 3, 1 / 3, 1.0)


def _basis_blend_embed(pcm_slice: np.ndarray, sr: int) -> np.ndarray:
    v = np.array(
        [float(np.mean(np.abs(pcm_slice - f) < 0.05)) for f in _FILLS_4],
        dtype=np.float32,
    )
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return v / n


class TestKSelectionFourVoices:
    def test_four_genuine_voices_yield_four_speakers(self):
        turns = [_turn(2.0 * i, 2.0 * (i + 1)) for i in range(8)]
        fills = [_FILLS_4[i % 4] for i in range(8)]
        pcm = _voiced_pcm(turns, fills, 16.0)
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_basis_blend_embed)
        assert got is not None
        assert got["num_speakers"] == 4
        assert [t["speaker"] for t in got["turns"]] == [
            "Speaker A", "Speaker B", "Speaker C", "Speaker D",
        ] * 2
        # MAX_SPEAKERS_LOCAL caps evaluation at k=4 even with 8 turns.
        assert [e["k"] for e in got["k_evaluated"]] == [2, 3, 4]
        assert diarize_local.MAX_SPEAKERS_LOCAL == 4
