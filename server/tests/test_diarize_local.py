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
        """Compute bound: only utterances at or over the length floor
        (SCAN_MIN_UTTERANCE_SECONDS, 2.0 since round 2) get windowed."""
        turns = [dict(_turn(3.0, 4.8), words=_words(3.0, 4.8, ["a", "b", "c", "d"]))]
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
        labels, margins, bests = diarize_local._label_words(
            _mixed_scenario_pcm(), SR, turn, _mean_angle_embed,
            list(_CENTROIDS_AB),
        )
        assert labels == [0, 0, 0, 0, 1, 1, 1, 1]
        assert all(m > diarize_local.WORD_MIN_MARGIN for m in margins)
        # Every word is claimed by a voice: best cosine ~1 to its centroid.
        assert all(b > 0.9 for b in bests)

    def test_ambiguous_word_has_low_margin(self):
        """A word whose window favors neither centroid scores near-zero."""
        words = _words(3.0, 5.0, ["x", "y"])
        turn = dict(_turn(3.0, 5.0), words=words)
        pcm = np.zeros(int(6.0 * SR), dtype=np.float32)  # fill 0 = neither voice
        _, margins, _ = diarize_local._label_words(
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
            "window_boundaries": 0, "confirmed_short_pieces": [],
            "unknown_pieces": [],
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
        # capped at round 1's validated k (3 here) OR the window pass's
        # eigengap count, whichever is higher (2026-08-29). On this fake the
        # windows straddling a voice change embed as blends the eigengap
        # counts as a 4th "voice", so k=4 IS tried — by both routes — and
        # rejected by validation; k=5/6 stay untried.
        ks = [(e["k"], e["route"], e["ok"]) for e in got["k_evaluated"]]
        assert ks[:2] == [(2, "linkage", True), (3, "linkage", True)]
        assert {k for k, _, _ in ks} <= {2, 3, 4}
        assert all(not ok for k, _, ok in ks if k == 4)
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
        """A same-voice-in-two-registers split must NOT become a phantom
        third speaker (general regression shape; live calibration
        2026-08-14, thresholds recalibrated 2026-08-24 — see
        diarize_local.py's STRONG_SEPARATION_COSINE / NEW_VOICE_ANCHOR_COSINE
        comments for the real-fixture data behind both numbers).

        Noisy same-voice utterances can split into two clusters whose pair
        cosine slips UNDER the marginal-split bar (STRONG_SEPARATION_COSINE)
        the same way a genuine new voice's split does. What separates them is
        the ANCHOR: a genuine new voice is wildly unlike an existing cluster,
        while BOTH halves of a phantom split sit moderately far from
        everything. A marginal split whose halves both exceed
        NEW_VOICE_ANCHOR_COSINE against every non-sibling cluster is
        rejected. This test picks cosine values that sit just inside the
        "should be rejected" side of both current bars (0.28 < 0.32 marginal
        bar; 0.26 > 0.24 anchor bar) rather than pinning to any one
        historical recording's exact measurements, so it stays meaningful
        across future recalibrations without silently going stale.
        """
        # Three directions with a controlled Gram matrix: registers r1, r2 of
        # one voice at cos 0.28 to each other (under the 0.32 marginal bar),
        # both at cos 0.26 to voice V (over the 0.24 anchor bar) — all pairs
        # pass the 0.45 gate, the split pair passes the marginal bar, but
        # neither half anchors.
        gram = np.array([
            [1.0, 0.28, 0.26],
            [0.28, 1.0, 0.26],
            [0.26, 0.26, 1.0],
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

    @pytest.mark.parametrize(
        "split_cos, expect_third",
        [
            # The 3-person family recording's genuine third voice: marginal
            # split 0.325 on its 7-utterance transcript (2026-08-27). Under
            # the old 0.32 bar it was rejected and the son merged into the
            # owner — the reason the bar moved to 0.33.
            (0.325, True),
            # family_real's one-voice-two-registers split (the son, calm vs
            # shouting): 0.337. The bar must stay UNDER this or the owner's
            # own calibration fixture grows a phantom third speaker.
            (0.337, False),
        ],
    )
    def test_strong_separation_bar_brackets_the_real_measurements(
        self, split_cos, expect_third,
    ):
        """STRONG_SEPARATION_COSINE sits between the closest genuine and
        spurious marginal splits measured on real audio (0.325 vs 0.337).
        Both halves anchor clearly (cos 0.05 to the third voice, well under
        NEW_VOICE_ANCHOR_COSINE) and every cluster has plenty of speech, so
        the marginal-pair bar is the ONLY rule deciding k=3 here."""
        gram = np.array([
            [1.0, split_cos, 0.05],
            [split_cos, 1.0, 0.05],
            [0.05, 0.05, 1.0],
        ])
        vecs = np.linalg.cholesky(gram)
        fills = (-1.0, 0.0, 1.0)

        def blend_embed(pcm_slice: np.ndarray, sr: int) -> np.ndarray:
            w = np.array([
                float(np.mean(np.abs(pcm_slice - f) < 0.05)) for f in fills
            ])
            v = w @ vecs
            return (v / np.linalg.norm(v)).astype(np.float32)

        turns = [
            _turn(0.0, 4.0), _turn(4.0, 8.0),      # voice 1 (fill -1)
            _turn(8.0, 12.0), _turn(12.0, 16.0),   # voice 2 / register (fill 0)
            _turn(16.0, 20.0), _turn(20.0, 24.0),  # voice 3 (fill +1)
        ]
        pcm = _voiced_pcm(turns, [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0], 25.0)
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=blend_embed)
        assert got is not None
        assert (got["num_speakers"] == 3) is expect_third, _by_k(got)[3]
        assert _by_k(got)[3]["ok"] is expect_third

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
        # MAX_SPEAKERS_LOCAL (raised 4 -> 6 on 2026-08-21) caps evaluation at
        # k=6 with 8 turns; k=5/6 are evaluated but correctly rejected,
        # leaving the genuine k=4 split as the chosen result.
        assert [e["k"] for e in got["k_evaluated"]] == [2, 3, 4, 5, 6]
        assert diarize_local.MAX_SPEAKERS_LOCAL == 6


# ---------------------------------------------------------------------------
# 2026-08-29: the transcript-free WINDOW PASS (voice-separation bake-off,
# docs/research/2026-08-29-voice-separation/): spectral fallback partition
# in k-selection, the eigengap count as a lower bound that never raises k
# past validation, window-pass boundary proposals inside a welded utterance
# WITHOUT word timings, and the noise-floor-relative speech gate.
# ---------------------------------------------------------------------------


def _gram_embed(fills: tuple[float, ...], gram) -> "callable":
    """Fake embedder whose voices are unit vectors with the given Gram
    matrix (via Cholesky); a slice embeds to the fill-fraction-weighted
    blend of the voices it contains, so pooled audio behaves like pooled
    ECAPA."""
    vecs = np.linalg.cholesky(np.asarray(gram, dtype=np.float64))

    def embed(pcm_slice: np.ndarray, sr: int) -> np.ndarray:
        w = np.array([float(np.mean(np.abs(pcm_slice - f) < 0.05)) for f in fills])
        v = w @ vecs
        n = float(np.linalg.norm(v))
        return (v / n if n else vecs[0]).astype(np.float32)

    return embed


class _StubWindowPass:
    """Stands in for diarize_local._WindowPass in _select_k tests: a fixed
    eigengap count and fixed pooled spectral centroids per k."""

    def __init__(self, k_eigengap: int, centroids: dict[int, dict[int, np.ndarray]]):
        self.k_eigengap = k_eigengap
        self._centroids = centroids

    def pooled_centroids(self, k: int):
        return self._centroids.get(k)


class TestSpectralRoute:
    def test_spectral_partition_rescues_k_where_linkage_peels_a_sliver(self):
        """The maggiano3 shape: three real voices, but average linkage's
        3-way split carves a 1 s outlier off instead of separating the two
        similar adults, and the duration floor rightly rejects that sliver.
        The spectral centroids from the window pass seed a DIFFERENT 3-way
        partition of the same turns; it passes the same validation."""
        # A and B adults at cosine 0.30 (past the accept gate, under the
        # marginal bar); C orthogonal to both; D a 1.0 s outlier at 0.30 to C.
        gram = [[1, .3, 0, 0], [.3, 1, 0, 0], [0, 0, 1, .3], [0, 0, .3, 1]]
        fills = (-1.0, -1 / 3, 1 / 3, 1.0)
        embed = _gram_embed(fills, gram)
        turns = [
            _turn(0.0, 2.0), _turn(2.0, 4.0), _turn(4.0, 6.0),      # A
            _turn(6.0, 8.0), _turn(8.0, 10.0), _turn(10.0, 12.0),   # B
            _turn(12.0, 14.0), _turn(14.0, 16.0),                   # C
            _turn(16.0, 17.0),                                      # D sliver
        ]
        pcm = _voiced_pcm(
            turns, [fills[0]] * 3 + [fills[1]] * 3 + [fills[2]] * 2 + [fills[3]], 18.0,
        )
        order, embs = diarize_local._embed_turns(pcm, SR, turns, embed, 1.0)

        def unit(i: int) -> np.ndarray:
            return embed(np.full(SR, fills[i], dtype=np.float32), SR)

        stub = _StubWindowPass(3, {3: {0: unit(0), 1: unit(1), 2: unit(2)}})
        k_eval, chosen = diarize_local._select_k(
            pcm, SR, turns, embed, order, embs, diarize_local.MAX_POOLED_COSINE,
            window_pass=stub,
        )
        by = {(e["k"], e["route"]): e for e in k_eval}
        assert by[(3, "linkage")]["ok"] is False
        assert by[(3, "linkage")]["failed"] == "duration_floor"
        assert by[(3, "spectral")]["ok"] is True
        assert chosen is not None and len(chosen[1]) == 3
        assert chosen[2]["route"] == "spectral"
        # Without the window pass the same audio honestly stops at k=2.
        _, chosen_plain = diarize_local._select_k(
            pcm, SR, turns, embed, order, embs, diarize_local.MAX_POOLED_COSINE,
        )
        assert chosen_plain is not None and len(chosen_plain[1]) == 2

    def test_eigengap_lower_bound_never_raises_k_past_validation(self):
        """An eigengap count of 3 on a 2-voice recording only makes k=3 get
        TRIED via the spectral route; the partition still fails the same
        validation and k stays at 2."""
        turns = [_turn(2.0 * i, 2.0 * (i + 1)) for i in range(6)]
        fills = [VOICE_A, VOICE_A, VOICE_A_TWIN, VOICE_A_TWIN, VOICE_B, VOICE_B]
        pcm = _voiced_pcm(turns, fills, 12.0)
        order, embs = diarize_local._embed_turns(pcm, SR, turns, _mean_angle_embed, 1.0)
        stub = _StubWindowPass(3, {3: {
            0: _fake_centroid(VOICE_A), 1: _fake_centroid(VOICE_A_TWIN),
            2: _fake_centroid(VOICE_B),
        }})
        k_eval, chosen = diarize_local._select_k(
            pcm, SR, turns, _mean_angle_embed, order, embs,
            diarize_local.MAX_POOLED_COSINE, window_pass=stub,
        )
        assert chosen is not None and len(chosen[1]) == 2
        assert chosen[2]["route"] == "linkage"
        spectral = [e for e in k_eval if e["route"] == "spectral"]
        assert [e["k"] for e in spectral] == [3]
        assert spectral[0]["ok"] is False
        assert spectral[0]["failed"] == "centroids"
        # No spectral attempt above the eigengap count.
        assert all(e["route"] == "linkage" for e in k_eval if e["k"] > 3)


class TestWindowBoundaryProposals:
    def test_proposal_splits_no_words_utterance_and_divides_text_by_duration(self):
        turns = [_turn(3.0, 11.0, text="a b c d")]
        got, stats = diarize_local.split_long_utterances(
            _mixed_scenario_pcm(), SR, turns, _mean_angle_embed, _CENTROIDS_AB,
            proposals={0: [7.0]},
        )
        assert [(p["start_time"], p["end_time"], p["text"]) for p in got] == [
            (3.0, 7.0, "a b"), (7.0, 11.0, "c d"),
        ]
        assert stats["split"] == 1 and stats["window_boundaries"] == 1
        assert stats["skipped_no_words"] == 0

    def test_proposal_near_a_word_cut_is_the_same_cut(self):
        """A window proposal within BOUNDARY_DEDUPE_SECONDS of the per-word
        cut is deduplicated; the word-aligned cut wins."""
        turns = [dict(_turn(3.0, 11.0), words=_MIXED_WORDS)]
        got, stats = diarize_local.split_long_utterances(
            _mixed_scenario_pcm(), SR, turns, _mean_angle_embed, _CENTROIDS_AB,
            proposals={0: [6.8]},
        )
        assert [(p["start_time"], p["end_time"]) for p in got] == [(3.0, 7.0), (7.0, 11.0)]
        assert stats["split"] == 1 and stats["window_boundaries"] == 0

    def test_window_pass_splits_welded_three_voice_utterance_without_words(self):
        """End to end through the real window pass (fake embedders): a
        9 s utterance welding three voices, NO word timings — the per-word
        pass cannot touch it, the window pass proposes both cuts, the
        pieces are re-embedded and attributed to all three voices."""
        fills = (-1.0, 1 / 3, 1.0)  # three orthogonal, non-silent voices
        embed = _gram_embed(fills, np.eye(3))

        def embed_batch(chunks, sr):
            return [embed(c, sr) for c in chunks]

        turns = [
            _turn(0.0, 3.0, text="pp"), _turn(3.0, 6.0, text="qq"),
            _turn(6.0, 9.0, text="rr"), _turn(9.0, 18.0, text="w1 w2 w3"),
        ]
        pcm = np.zeros(int(18.0 * SR), dtype=np.float32)
        for lo, hi, f in [(0, 3, fills[0]), (3, 6, fills[1]), (6, 9, fills[2]),
                          (9, 12, fills[0]), (12, 15, fills[1]), (15, 18, fills[2])]:
            pcm[int(lo * SR):int(hi * SR)] = f
        got = diarize_local.diarize_turns(
            pcm, SR, turns, embed_fn=embed, embed_batch_fn=embed_batch,
        )
        assert got is not None
        assert got["num_speakers"] == 3
        assert got["split_utterances"] == 1
        assert got["window_pass"]["k_eigengap"] == 3
        assert got["window_pass"]["window_boundaries_used"] == 2
        assert [t["speaker"] for t in got["turns"]] == [
            "Speaker A", "Speaker B", "Speaker C",
        ] * 2
        pieces = got["turns"][3:]
        assert abs(pieces[0]["end_time"] - 12.0) <= 0.5
        assert abs(pieces[1]["end_time"] - 15.0) <= 0.5
        assert [p["text"] for p in pieces] == ["w1", "w2", "w3"]
        assert all("words" not in t for t in got["turns"])

    def test_pure_utterances_get_no_proposals(self):
        """Two clean alternating voices: the window pass must not cut a
        single-voice utterance (the regression ladder pins turn counts)."""
        turns = [_turn(4.0 * i, 4.0 * (i + 1)) for i in range(4)]
        pcm = _voiced_pcm(turns, [VOICE_A, VOICE_B, VOICE_A, VOICE_B], 16.0)

        def embed_batch(chunks, sr):
            return [_mean_angle_embed(c, sr) for c in chunks]

        got = diarize_local.diarize_turns(
            pcm, SR, turns, embed_fn=_mean_angle_embed, embed_batch_fn=embed_batch,
        )
        assert got is not None and got["num_speakers"] == 2
        assert got["split_utterances"] == 0
        assert got["window_pass"]["proposed_boundaries"] == 0
        assert len(got["turns"]) == 4


class TestNoiseFloorSpeechGate:
    def test_relative_gate_keeps_quiet_speaker_and_rejects_silence(self):
        rng = np.random.default_rng(0)
        room = (0.001 * rng.standard_normal(SR * 6)).astype(np.float32)  # floor RMS 0.001
        pcm = room.copy()
        pcm[SR * 2:SR * 4] += (0.005 * rng.standard_normal(SR * 2)).astype(np.float32)
        # The enrollment gate (absolute 0.01) hears nothing in a 0.005 RMS voice...
        assert speaker_id.speech_seconds(pcm, SR) == 0.0
        # ...the diarizer's noise-floor-relative gate keeps the 2 s: gate =
        # max(0.003, 1.5 x p10 ~ 0.0015) = 0.003 < 0.005.
        quiet = speaker_id.speech_seconds(pcm, SR, rms_threshold=speaker_id.SPEECH_RMS_FLOOR)
        assert quiet == pytest.approx(2.0, abs=0.15)
        mask, gate, frame_s = speaker_id.speech_mask(pcm, SR)
        assert speaker_id.SPEECH_RMS_FLOOR <= gate < 0.005
        assert float(mask.sum()) * frame_s == pytest.approx(2.0, abs=0.15)
        # Silence is still silence.
        silence = np.zeros(SR * 4, dtype=np.float32)
        assert speaker_id.speech_seconds(
            silence, SR, rms_threshold=speaker_id.SPEECH_RMS_FLOOR,
        ) == 0.0
        assert speaker_id.speech_mask(silence, SR)[0].sum() == 0

    def test_relative_gate_rejects_steady_room_tone_above_the_floor(self):
        """Room tone at RMS 0.008 everywhere sits above the absolute floor;
        the relative term (1.5 x p10 = 0.012) rejects all of it."""
        rng = np.random.default_rng(1)
        tone = (0.008 * rng.standard_normal(SR * 6)).astype(np.float32)
        assert speaker_id.speech_seconds(
            tone, SR, rms_threshold=speaker_id.SPEECH_RMS_FLOOR,
        ) == 0.0
        # The purely absolute gate would have counted every second.
        assert speaker_id.speech_seconds(
            tone, SR, rms_threshold=speaker_id.SPEECH_RMS_FLOOR, floor_mult=0.0,
        ) == pytest.approx(6.0, abs=0.1)

    def test_gate_is_capped_for_a_clip_with_no_quiet_frames(self):
        """A sustained tone has p10 at speech level; the cap keeps the gate
        under it instead of gating the whole clip out."""
        t = np.arange(SR * 4, dtype=np.float32) / SR
        tone = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        assert speaker_id.speech_seconds(tone, SR) == pytest.approx(4.0, abs=0.1)
        _, gate, _ = speaker_id.speech_mask(tone, SR)
        assert gate == speaker_id.SPEECH_RMS_GATE_CEILING


# ---------------------------------------------------------------------------
# Round 2 (2026-08-29): lower scan floors (SCAN_MIN_UTTERANCE_SECONDS 2.0),
# both-source-confirmed pieces down to CONFIRMED_PIECE_MIN_SECONDS (0.8),
# speech the transcript never covered becoming "(untranscribed)" turns.
# ---------------------------------------------------------------------------

# Words a-f cover 3.0-9.0 (one per second); g/h cover 9.0-9.85 — a 2-word
# run whose piece lasts 0.85 s: over CONFIRMED_PIECE_MIN_SECONDS, under
# MIN_SECONDS.
_SHORT_TAIL_WORDS = [
    {"word": w, "start_time": s, "end_time": e}
    for w, s, e in [
        ("a", 3.0, 4.0), ("b", 4.0, 5.0), ("c", 5.0, 6.0), ("d", 6.0, 7.0),
        ("e", 7.0, 8.0), ("f", 8.0, 9.0), ("g", 9.0, 9.4), ("h", 9.4, 9.85),
    ]
]
_SHORT_TAIL_LABELS = [0, 0, 0, 0, 0, 0, 1, 1]


def _short_tail_pcm(tail_end: float = 9.85) -> np.ndarray:
    """Voice A fills 3.0-9.0, voice B fills 9.0-``tail_end``; silence around."""
    pcm = np.zeros(int(12.0 * SR), dtype=np.float32)
    pcm[int(3.0 * SR):int(9.0 * SR)] = VOICE_A
    pcm[int(9.0 * SR):int(tail_end * SR)] = VOICE_B
    return pcm


class TestConfirmedShortPieces:
    def test_single_source_short_run_still_merges(self):
        """The per-word pass alone (no second instrument, or one that
        disagrees) keeps the MIN_SECONDS floor: a 0.85 s run merges."""
        assert diarize_local._collapse_word_runs(
            _SHORT_TAIL_WORDS, _SHORT_TAIL_LABELS, 3.0, 9.85,
        ) == [0] * 8
        assert diarize_local._collapse_word_runs(
            _SHORT_TAIL_WORDS, _SHORT_TAIL_LABELS, 3.0, 9.85,
            min_seconds_confirmed=diarize_local.CONFIRMED_PIECE_MIN_SECONDS,
            confirm=lambda p0, p1, q0, q1: False,
        ) == [0] * 8

    def test_both_source_confirmed_short_run_survives(self):
        calls = []

        def confirm(p0, p1, q0, q1):
            calls.append((p0, p1, q0, q1))
            return True

        confirmed: set = set()
        got = diarize_local._collapse_word_runs(
            _SHORT_TAIL_WORDS, _SHORT_TAIL_LABELS, 3.0, 9.85,
            min_seconds_confirmed=diarize_local.CONFIRMED_PIECE_MIN_SECONDS,
            confirm=confirm, confirmed_out=confirmed,
        )
        assert got == _SHORT_TAIL_LABELS
        # Asked once, about the piece and the neighbour it would merge into.
        assert calls == [(9.0, 9.85, 3.0, 9.0)]
        assert confirmed == {(9.0, 9.85)}

    def test_piece_under_the_confirmed_floor_merges_without_asking(self):
        words = [dict(w) for w in _SHORT_TAIL_WORDS]
        words[6]["end_time"], words[7]["start_time"], words[7]["end_time"] = 9.3, 9.3, 9.7
        calls = []
        got = diarize_local._collapse_word_runs(
            words, _SHORT_TAIL_LABELS, 3.0, 9.7,
            min_seconds_confirmed=diarize_local.CONFIRMED_PIECE_MIN_SECONDS,
            confirm=lambda *a: calls.append(a) or True,
        )
        assert got == [0] * 8
        assert calls == []

    def test_enforce_min_pieces_honours_the_confirmed_floor_only(self):
        kw = dict(min_seconds_confirmed=diarize_local.CONFIRMED_PIECE_MIN_SECONDS)
        assert diarize_local._enforce_min_pieces(
            3.0, 9.85, [9.0], confirmed={(9.0, 9.85)}, **kw,
        ) == [9.0]
        # The same 0.85 s piece from a cut NOBODY confirmed (a window
        # proposal, or an unconfirmed word cut) is still a sliver.
        assert diarize_local._enforce_min_pieces(3.0, 9.85, [9.0], confirmed=set(), **kw) == []
        assert diarize_local._enforce_min_pieces(3.0, 9.85, [9.0]) == []

    def test_split_pass_keeps_confirmed_short_piece_and_reports_it(self):
        turns = [dict(_turn(3.0, 9.85, text="welded"), words=_SHORT_TAIL_WORDS)]
        got, stats = diarize_local.split_long_utterances(
            _short_tail_pcm(), SR, turns, _mean_angle_embed, _CENTROIDS_AB,
            confirm=lambda p0, p1, q0, q1: True,
        )
        assert [(p["start_time"], p["end_time"], p["text"]) for p in got] == [
            (3.0, 9.0, "a b c d e f"), (9.0, 9.85, "g h"),
        ]
        assert stats["split"] == 1
        assert stats["confirmed_short_pieces"] == [(9.0, 9.85)]
        # One source only → no split, nothing confirmed.
        got, stats = diarize_local.split_long_utterances(
            _short_tail_pcm(), SR, turns, _mean_angle_embed, _CENTROIDS_AB,
        )
        assert len(got) == 1 and stats["split"] == 0
        assert stats["confirmed_short_pieces"] == []

    def test_two_second_utterance_is_scanned_and_split(self):
        """SCAN_MIN_UTTERANCE_SECONDS (2.0, was 3.0): a 2.2 s weld of two
        1.1 s voices — each piece over MIN_SECONDS — is now split."""
        words = [
            {"word": w, "start_time": s, "end_time": e}
            for w, s, e in [("a", 3.0, 3.55), ("b", 3.55, 4.1), ("c", 4.1, 4.65), ("d", 4.65, 5.2)]
        ]
        turns = [dict(_turn(3.0, 5.2), words=words)]
        pcm = np.zeros(int(8.0 * SR), dtype=np.float32)
        pcm[int(3.0 * SR):int(4.1 * SR)] = VOICE_A
        pcm[int(4.1 * SR):int(5.2 * SR)] = VOICE_B
        got, stats = diarize_local.split_long_utterances(
            pcm, SR, turns, _mean_angle_embed, _CENTROIDS_AB,
        )
        assert stats["scanned"] == 1 and stats["split"] == 1
        assert [(p["start_time"], p["end_time"]) for p in got] == [(3.0, 4.1), (4.1, 5.2)]

    def test_window_pass_confirms_two_voices_against_spectral_centroids(self):
        """The second source: both pieces embedded against the pooled
        spectral centroids must land on DIFFERENT centroids with margin ≥
        WORD_MIN_MARGIN each."""
        pcm = _short_tail_pcm()

        def embed_batch(chunks, sr):
            return [_mean_angle_embed(c, sr) for c in chunks]

        wp = diarize_local._WindowPass(pcm, SR, embed_batch, _mean_angle_embed)
        wp.k_eigengap = 2
        wp._centroids_at[2] = {0: _fake_centroid(VOICE_A), 1: _fake_centroid(VOICE_B)}
        assert wp.confirms_two_voices(9.0, 9.85, 3.0, 9.0) is True
        assert wp.confirms_two_voices(3.0, 6.0, 6.0, 9.0) is False   # same voice
        wp._centroids_at[2] = None                                    # no spectral pass
        assert wp.confirms_two_voices(9.0, 9.85, 3.0, 9.0) is False
        wp.k_eigengap = None
        assert wp.confirms_two_voices(9.0, 9.85, 3.0, 9.0) is False

    def test_confirmed_piece_attributed_by_its_own_voice_end_to_end(self, monkeypatch):
        """Through diarize_turns: a 6.85 s weld whose second voice lasts
        0.85 s. The per-word run hears voice B there and the window pass's
        verdict (stubbed — the 2-D fake embedder's spectral clusters are
        boundary blends, see the test above for the real verdict) agrees,
        so the piece survives under MIN_SECONDS and — never embedded for
        clustering — takes the centroid its own embedding is nearest to
        (B), not the neighbour it was cut from (A)."""
        asked = []

        def stub_confirm(self, p0, p1, q0, q1, *, min_margin=None):
            asked.append((p0, p1, q0, q1))
            return True

        monkeypatch.setattr(diarize_local._WindowPass, "confirms_two_voices", stub_confirm)
        words = [
            {"word": w, "start_time": 9.0 + i, "end_time": 10.0 + i}
            for i, w in enumerate(["a", "b", "c", "d", "e", "f"])
        ] + [
            {"word": "g", "start_time": 15.0, "end_time": 15.4},
            {"word": "h", "start_time": 15.4, "end_time": 15.85},
        ]
        turns = [
            _turn(0.0, 3.0, text="aa"), _turn(3.0, 6.0, text="bb"),
            _turn(6.0, 9.0, text="aa"),
            dict(_turn(9.0, 15.85, text="welded"), words=words),
        ]
        pcm = np.zeros(int(17.0 * SR), dtype=np.float32)
        for lo, hi, fill in [(0, 3, VOICE_A), (3, 6, VOICE_B), (6, 9, VOICE_A),
                             (9, 15, VOICE_A), (15, 15.85, VOICE_B)]:
            pcm[int(lo * SR):int(hi * SR)] = fill
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None and got["num_speakers"] == 2
        assert asked == [(15.0, 15.85, 9.0, 15.0)]
        assert got["split_utterances"] == 1
        assert got["confirmed_short_pieces"] == 1
        assert got["short_turn_attribution"] == {"self": 1, "neighbour": 0}
        assert [(t["start_time"], t["end_time"], t["speaker"]) for t in got["turns"]] == [
            (0.0, 3.0, "Speaker A"), (3.0, 6.0, "Speaker B"), (6.0, 9.0, "Speaker A"),
            (9.0, 15.0, "Speaker A"), (15.0, 15.85, "Speaker B"),
        ]
        # The 0.85 s piece was NOT part of the clustering set.
        assert got["segments_embedded"] == 4


class TestUncoveredSpeech:
    FRAME = 0.03

    def _mask(self, seconds: float, speech: list[tuple[float, float]]) -> np.ndarray:
        n = int(round(seconds / self.FRAME))
        mask = np.zeros(n, dtype=bool)
        for s, e in speech:
            mask[int(round(s / self.FRAME)):int(round(e / self.FRAME))] = True
        return mask

    def test_runs_outside_every_turn(self):
        mask = self._mask(10.0, [(0.0, 10.0)])
        turns = [_turn(0.0, 3.0), _turn(4.0, 6.0)]
        got = diarize_local._uncovered_speech(mask, self.FRAME, turns, 10.0)
        assert len(got) == 2
        assert got[0] == pytest.approx((3.0, 4.0), abs=0.05)
        assert got[1] == pytest.approx((6.0, 10.0), abs=0.05)
        # Nothing overlaps a turn.
        for s, e in got:
            for t in turns:
                assert diarize_local._overlap_seconds(s, e, t["start_time"], t["end_time"]) == 0.0

    def test_short_run_ignored_and_holes_bridged(self):
        turns = [_turn(0.0, 1.0)]
        # 0.3 s of speech: under UNCOVERED_MIN_SECONDS.
        assert diarize_local._uncovered_speech(
            self._mask(6.0, [(3.0, 3.3)]), self.FRAME, turns, 6.0,
        ) == []
        # Two bursts with a 0.09 s hole (≤ UNCOVERED_BRIDGE_SECONDS) → one run.
        got = diarize_local._uncovered_speech(
            self._mask(6.0, [(3.0, 3.3), (3.39, 3.7)]), self.FRAME, turns, 6.0,
        )
        assert len(got) == 1 and got[0] == pytest.approx((3.0, 3.7), abs=0.05)
        # A 0.3 s hole is NOT bridged → two sub-floor bursts, nothing kept.
        assert diarize_local._uncovered_speech(
            self._mask(6.0, [(3.0, 3.3), (3.6, 3.9)]), self.FRAME, turns, 6.0,
        ) == []

    def test_turn_dicts_overlap_guard_and_cap(self):
        turns = [_turn(0.0, 1.05, speaker="Speaker Q"), _turn(6.9, 7.3, speaker="Speaker R")]
        candidates = [(1.0, 1.5), (3.0, 5.0), (7.0, 7.6), (8.0, 8.5)]
        got = diarize_local._uncovered_turn_dicts(candidates, turns, cap=10)
        # (7.0, 7.6) overlaps the 6.9-7.3 turn by 0.3 s > 0.1 → dropped;
        # (1.0, 1.5) overlaps 0.05 s → allowed.
        assert [(t["start_time"], t["end_time"]) for t in got] == [(1.0, 1.5), (3.0, 5.0), (8.0, 8.5)]
        assert all(t["text"] == diarize_local.UNTRANSCRIBED_TEXT for t in got)
        assert got[0]["speaker"] == "Speaker Q" and got[2]["speaker"] == "Speaker R"
        # The cap keeps the LONGEST runs, in time order.
        got = diarize_local._uncovered_turn_dicts(candidates, turns, cap=2)
        assert [(t["start_time"], t["end_time"]) for t in got] == [(1.0, 1.5), (3.0, 5.0)]
        assert diarize_local._uncovered_turn_dicts(candidates, turns, cap=0) == []

    def test_insert_chronological_maps_old_indices(self):
        turns = [_turn(0.0, 1.0, text="t0"), _turn(2.0, 3.0, text="t1")]
        extras = [_turn(1.2, 1.6, text="x0"), _turn(3.5, 4.0, text="x1")]
        merged, index_map = diarize_local._insert_chronological(turns, extras)
        assert [t["text"] for t in merged] == ["t0", "x0", "t1", "x1"]
        assert index_map == [0, 2]

    def _four_turns(self):
        turns = [_turn(3.0 * i, 3.0 * (i + 1)) for i in range(4)]
        return turns, [VOICE_A, VOICE_B, VOICE_A, VOICE_B]

    def test_uncovered_speech_becomes_a_turn_labelled_by_its_own_voice(self):
        turns, fills = self._four_turns()
        pcm = _voiced_pcm(turns, fills, 18.0)
        pcm[int(13.0 * SR):int(13.7 * SR)] = VOICE_B   # 0.7 s, voice B, no utterance
        pcm[int(15.0 * SR):int(16.2 * SR)] = VOICE_A   # 1.2 s, voice A, no utterance
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None and got["num_speakers"] == 2
        assert got["uncovered_turns"] == 2
        assert len(got["turns"]) == 6
        extra = [t for t in got["turns"] if t["text"] == diarize_local.UNTRANSCRIBED_TEXT]
        assert [(t["start_time"], t["end_time"], t["speaker"]) for t in extra] == [
            (pytest.approx(13.0, abs=0.05), pytest.approx(13.7, abs=0.05), "Speaker B"),
            (pytest.approx(15.0, abs=0.05), pytest.approx(16.2, abs=0.05), "Speaker A"),
        ]
        # Chronological, and never part of the clustering set (even the
        # 1.2 s one): the transcript's own utterances decide the partition.
        assert [t["start_time"] for t in got["turns"]] == sorted(t["start_time"] for t in got["turns"])
        assert got["segments_embedded"] == 4
        assert got["short_turn_attribution"] == {"self": 2, "neighbour": 0}

    def test_uncovered_turns_are_capped_longest_first(self):
        turns, fills = self._four_turns()
        pcm = _voiced_pcm(turns, fills, 30.0)
        # Five uncovered bursts of 0.5 .. 0.9 s; the cap for 4 transcript
        # turns is int(0.2 * 4) + 3 = 3 → the three longest survive.
        bursts = [(13.0, 13.5), (15.0, 15.6), (17.0, 17.7), (19.0, 19.8), (21.0, 21.9)]
        for s, e in bursts:
            pcm[int(s * SR):int(e * SR)] = VOICE_A
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None
        assert got["uncovered_turns"] == 3
        extra = [t for t in got["turns"] if t["text"] == diarize_local.UNTRANSCRIBED_TEXT]
        assert [round(t["start_time"]) for t in extra] == [17, 19, 21]

    def test_no_uncovered_turn_when_utterances_cover_all_speech(self):
        turns, fills = self._four_turns()
        got = diarize_local.diarize_turns(
            _voiced_pcm(turns, fills, 12.0), SR, turns, embed_fn=_mean_angle_embed,
        )
        assert got is not None
        assert got["uncovered_turns"] == 0 and len(got["turns"]) == 4


# ---------------------------------------------------------------------------
# "Unknown" speaker for unclaimed speech (2026-08-30, flag
# MINDSHIFT_DIARIZE_UNKNOWN) — pure math with the fake embedder.
#
# Geometry: the fake embeds a slice's mean m at angle (m+1)*pi/2, so fills
# -1 / 0 / +1 sit at 0 / pi/2 / pi. With centroids at 0 (voice P) and pi/2
# (voice Q), a fill of +1 (angle pi) has cosine -1 / 0 to them — claimed by
# NOBODY under UNCLAIMED_COSINE — while P and Q words are claimed at margin 1.
# ---------------------------------------------------------------------------

_UNCLAIMED = 1.0  # angle pi: cosine -1 to voice P, 0 to voice Q
_CENT_P = _mean_angle_embed(np.full(8, VOICE_P, dtype=np.float32), SR)
_CENT_Q = _mean_angle_embed(np.full(8, VOICE_Q, dtype=np.float32), SR)


def _stub_window_pass(monkeypatch) -> None:
    """Silence the transcript-free window pass: no eigengap, no spectral
    centroids, no proposals (the 2-D fake's boundary-blend windows are not a
    meaningful second claimant — the spectral claimant is covered by
    ``extra_centroids`` in TestUnknownSpeaker directly)."""
    monkeypatch.setattr(diarize_local._WindowPass, "run_global", lambda self: None)


class TestUnknownSpeaker:
    def test_flag_is_read_at_call_time_and_defaults_off(self, monkeypatch):
        monkeypatch.delenv(diarize_local.UNKNOWN_FLAG_ENV, raising=False)
        assert diarize_local.unknown_enabled() is diarize_local.UNKNOWN_DEFAULT
        # The measured verdict (see the module docstring, 2026-08-30): OFF.
        assert diarize_local.UNKNOWN_DEFAULT is False
        for raw, want in [("1", True), ("on", True), ("YES", True), ("0", False),
                          ("off", False), ("", diarize_local.UNKNOWN_DEFAULT)]:
            monkeypatch.setenv(diarize_local.UNKNOWN_FLAG_ENV, raw)
            assert diarize_local.unknown_enabled() is want, raw

    def test_unknown_label_is_the_shared_constant(self):
        assert diarize_local.UNKNOWN_SPEAKER == speaker_id.UNKNOWN_SPEAKER == "Unknown"

    # -- per-word smoothing --------------------------------------------------

    def test_unclaimed_words_take_unknown_and_neither_seed_nor_inherit(self):
        got = diarize_local._smooth_word_labels(
            [0, 0, 1, 1, 1], [0.5, 0.5, 0.02, 0.02, 0.02],
            min_margin=0.1, unclaimed=[False, False, True, True, False],
        )
        # The unclaimed pair is UNKNOWN; the trailing AMBIGUOUS word inherits
        # the nearest confident CLAIMED word (index 1 -> 0), never the
        # Unknown words next to it.
        assert got == [0, 0, diarize_local.UNKNOWN_LABEL, diarize_local.UNKNOWN_LABEL, 0]

    def test_no_confident_claimed_word_is_still_inconclusive(self):
        # Two "confident" words that are both unclaimed: nothing to say.
        assert diarize_local._smooth_word_labels(
            [0, 1], [0.5, 0.5], min_margin=0.1, unclaimed=[True, True],
        ) is None

    def test_without_unclaimed_mask_behaviour_is_unchanged(self):
        assert diarize_local._smooth_word_labels(
            [0, 0, 1, 0, 1, 1], [0.5, 0.5, 0.02, 0.03, 0.5, 0.5], min_margin=0.1,
        ) == [0, 0, 0, 1, 1, 1]

    # -- run collapse --------------------------------------------------------

    def test_unclaimed_run_survives_with_enough_words_and_seconds(self):
        words = _words(3.0, 11.0, list("abcdefgh"))
        u = diarize_local.UNKNOWN_LABEL
        labels = [0, 0, 0, u, u, 0, 0, 0]
        out: set = set()
        got = diarize_local._collapse_word_runs(
            words, labels, 3.0, 11.0,
            min_seconds_unknown=diarize_local.UNKNOWN_MIN_SECONDS, unknown_out=out,
        )
        assert got == labels
        assert out == {(6.0, 8.0)}

    def test_unclaimed_run_merges_when_the_rule_is_off_or_too_short(self):
        words = _words(3.0, 11.0, list("abcdefgh"))
        u = diarize_local.UNKNOWN_LABEL
        # Rule off (no floor given): an unclaimed run is just a bad run.
        assert diarize_local._collapse_word_runs(
            words, [0, 0, 0, u, u, 0, 0, 0], 3.0, 11.0,
        ) == [0] * 8
        # A lone unclaimed word never stands (WORD_MIN_RUN).
        out: set = set()
        assert diarize_local._collapse_word_runs(
            words, [0, 0, 0, u, 0, 0, 0, 0], 3.0, 11.0,
            min_seconds_unknown=diarize_local.UNKNOWN_MIN_SECONDS, unknown_out=out,
        ) == [0] * 8
        assert out == set()
        # Two unclaimed words under UNKNOWN_MIN_SECONDS merge too.
        short = [dict(w) for w in words]
        short[3]["start_time"], short[3]["end_time"] = 6.0, 6.3
        short[4]["start_time"], short[4]["end_time"] = 6.3, 6.6
        short[5]["start_time"] = 6.6
        assert diarize_local._collapse_word_runs(
            short, [0, 0, 0, u, u, 0, 0, 0], 3.0, 11.0,
            min_seconds_unknown=diarize_local.UNKNOWN_MIN_SECONDS,
        ) == [0] * 8

    def test_short_claimed_run_beside_unknown_keeps_a_real_voice(self):
        words = _words(3.0, 11.0, list("abcdefgh"))
        u = diarize_local.UNKNOWN_LABEL
        # Runs: P(3) U(2) Q(1 word — untrustworthy) P(2). The lone Q word has
        # an Unknown neighbour on the left and a claimed one on the right:
        # it joins the claimed one, never the Unknown stretch.
        got = diarize_local._collapse_word_runs(
            words, [0, 0, 0, u, u, 1, 0, 0], 3.0, 11.0,
            min_seconds_unknown=diarize_local.UNKNOWN_MIN_SECONDS,
        )
        assert got == [0, 0, 0, u, u, 0, 0, 0]

    def test_merge_target_prefers_the_claimed_neighbour(self):
        pieces = [(0.0, 3.0), (3.0, 5.0), (5.0, 5.4), (5.4, 6.0)]
        u = diarize_local.UNKNOWN_LABEL
        assert diarize_local._merge_target(2, pieces, [0, u, 1, 0]) == 3
        assert diarize_local._merge_target(2, pieces, [0, 0, 1, u]) == 1
        # Both neighbours unclaimed: the plain (longer-piece) rule — a sliver
        # inside a stretch nobody claims is part of it; relabelling it to a
        # non-adjacent run could never terminate the collapse.
        assert diarize_local._merge_target(2, pieces, [0, u, 1, u]) == 1
        assert diarize_local._merge_target(2, pieces) == 1

    def test_collapse_terminates_with_a_sliver_between_two_unknown_runs(self):
        words = _words(3.0, 11.0, list("abcdefgh"))
        u = diarize_local.UNKNOWN_LABEL
        got = diarize_local._collapse_word_runs(
            words, [0, 0, u, u, 1, u, u, 0], 3.0, 11.0,
            min_seconds_unknown=diarize_local.UNKNOWN_MIN_SECONDS,
        )
        # The lone claimed word joined the unclaimed stretch; the leading /
        # trailing claimed runs (2 words, 2 s and 1 word) resolve as before.
        assert got[4] == u and len(set(got)) == 2

    # -- per-word scoring ----------------------------------------------------

    def test_extra_centroids_raise_the_best_cosine_but_never_label(self):
        words = _words(2.0, 4.0, ["x", "y"])
        turn = dict(_turn(2.0, 4.0), words=words)
        pcm = np.full(int(5.0 * SR), _UNCLAIMED, dtype=np.float32)
        labels, margins, bests = diarize_local._label_words(
            pcm, SR, turn, _mean_angle_embed, [_CENT_P, _CENT_Q],
        )
        assert labels == [1, 1]  # "least unlike": Q at cosine 0 vs P at -1
        assert all(abs(b) < 1e-6 for b in bests)
        own = _mean_angle_embed(np.full(8, _UNCLAIMED, dtype=np.float32), SR)
        labels2, _, bests2 = diarize_local._label_words(
            pcm, SR, turn, _mean_angle_embed, [_CENT_P, _CENT_Q],
            extra_centroids=[own],
        )
        assert labels2 == [1, 1]
        assert all(b > 0.99 for b in bests2)

    def test_enforce_min_pieces_honours_a_per_span_floor(self):
        assert diarize_local._enforce_min_pieces(
            3.0, 9.85, [9.0], floors={(9.0, 9.85): diarize_local.UNKNOWN_MIN_SECONDS},
        ) == [9.0]
        assert diarize_local._enforce_min_pieces(3.0, 9.85, [9.0], floors={}) == []

    # -- the split pass ------------------------------------------------------

    def _welded(self):
        texts = ["p1", "p2", "p3", "p4", "u1", "u2", "p5", "p6", "p7", "p8"]
        turns = [dict(_turn(3.0, 13.0, text="welded"), words=_words(3.0, 13.0, texts))]
        pcm = np.full(int(14.0 * SR), VOICE_P, dtype=np.float32)
        pcm[int(7.0 * SR):int(9.0 * SR)] = _UNCLAIMED
        return turns, pcm

    def test_split_pass_cuts_an_unclaimed_run_into_an_unknown_piece(self):
        turns, pcm = self._welded()
        got, stats = diarize_local.split_long_utterances(
            pcm, SR, turns, _mean_angle_embed, (_CENT_P, _CENT_Q),
            unclaimed_centroids=[], unknown=True,
        )
        assert stats["split"] == 1
        assert stats["unknown_pieces"] == [(7.0, 9.0)]
        assert [(t["start_time"], t["end_time"], t["text"]) for t in got] == [
            (3.0, 7.0, "p1 p2 p3 p4"), (7.0, 9.0, "u1 u2"), (9.0, 13.0, "p5 p6 p7 p8"),
        ]

    def test_split_pass_off_gives_the_run_to_the_least_unlike_voice(self):
        """The pre-2026-08-30 behaviour, kept when the flag is off: the
        unclaimed words are confidently labelled Q (cosine 0 beats -1), so
        the same piece is cut but reported as a claimed run."""
        turns, pcm = self._welded()
        got, stats = diarize_local.split_long_utterances(
            pcm, SR, turns, _mean_angle_embed, (_CENT_P, _CENT_Q),
        )
        assert stats["split"] == 1
        assert stats["unknown_pieces"] == []
        assert len(got) == 3

    def test_split_pass_leaves_a_claimed_run_alone(self):
        """A claimed second voice (Q) is a split, never Unknown."""
        texts = ["p1", "p2", "p3", "p4", "q1", "q2", "p5", "p6", "p7", "p8"]
        turns = [dict(_turn(3.0, 13.0, text="welded"), words=_words(3.0, 13.0, texts))]
        pcm = np.full(int(14.0 * SR), VOICE_P, dtype=np.float32)
        pcm[int(7.0 * SR):int(9.0 * SR)] = VOICE_Q
        got, stats = diarize_local.split_long_utterances(
            pcm, SR, turns, _mean_angle_embed, (_CENT_P, _CENT_Q),
            unclaimed_centroids=[], unknown=True,
        )
        assert stats["split"] == 1 and stats["unknown_pieces"] == []
        assert len(got) == 3

    # -- never seeds k -------------------------------------------------------

    def test_embed_turns_exclude_keeps_a_span_out_of_the_clustering_set(self):
        turns = [_turn(0.0, 2.0), _turn(2.0, 4.0), _turn(4.0, 6.0)]
        pcm = _voiced_pcm(turns, [VOICE_P, _UNCLAIMED, VOICE_Q], 6.0)
        order, _ = diarize_local._embed_turns(
            pcm, SR, turns, _mean_angle_embed, 1.0, exclude={(2.0, 4.0)},
        )
        assert order == [0, 2]

    # -- end to end ----------------------------------------------------------

    def _exchange(self):
        """P 0-3 | welded 3-8 (p-words 3-6, unclaimed 6-8) | "hm" 8-8.5 |
        Q 8.5-11.5 | P 11.5-14.5 | Q 14.5-17.5."""
        words = _words(3.0, 8.0, ["a1", "a2", "a3", "u1", "u2"])
        turns = [
            _turn(0.0, 3.0, text="p"),
            dict(_turn(3.0, 8.0, text="welded"), words=words),
            _turn(8.0, 8.5, text="hm"),
            _turn(8.5, 11.5, text="q"), _turn(11.5, 14.5, text="p"),
            _turn(14.5, 17.5, text="q"),
        ]
        pcm = np.zeros(int(18.0 * SR), dtype=np.float32)
        for lo, hi, fill in [(0, 6, VOICE_P), (6, 8, _UNCLAIMED), (8, 8.5, VOICE_P),
                             (8.5, 11.5, VOICE_Q), (11.5, 14.5, VOICE_P),
                             (14.5, 17.5, VOICE_Q)]:
            pcm[int(lo * SR):int(hi * SR)] = fill
        return turns, pcm

    def test_unclaimed_piece_is_unknown_never_a_cluster_never_inherited(self, monkeypatch):
        _stub_window_pass(monkeypatch)
        monkeypatch.setenv(diarize_local.UNKNOWN_FLAG_ENV, "1")
        turns, pcm = self._exchange()
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None
        assert got["num_speakers"] == 2  # Unknown is not a speaker
        assert got["unknown_turns"] == 1 and got["unknown_seconds"] == 2.0
        assert [(t["start_time"], t["end_time"], t["speaker"]) for t in got["turns"]] == [
            (0.0, 3.0, "Speaker A"), (3.0, 6.0, "Speaker A"),
            (6.0, 8.0, diarize_local.UNKNOWN_SPEAKER),
            # The 0.5 s "hm" sits nearest the Unknown piece by midpoint, yet
            # inherits the nearest CLAIMED embedded turn (Q) — never Unknown.
            (8.0, 8.5, "Speaker B"),
            (8.5, 11.5, "Speaker B"), (11.5, 14.5, "Speaker A"), (14.5, 17.5, "Speaker B"),
        ]
        # Never seeds k: the Unknown piece and the 0.5 s turn are the only
        # turns outside the clustering set.
        assert got["segments_total"] == 7 and got["segments_embedded"] == 5
        assert all(e["k"] <= 2 or not e["ok"] for e in got["k_evaluated"])

    def test_flag_off_gives_the_unclaimed_piece_to_the_least_unlike_voice(self, monkeypatch):
        _stub_window_pass(monkeypatch)
        monkeypatch.setenv(diarize_local.UNKNOWN_FLAG_ENV, "0")
        turns, pcm = self._exchange()
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None and got["num_speakers"] == 2
        assert got["unknown_turns"] == 0 and got["unknown_seconds"] == 0.0
        by_span = {(t["start_time"], t["end_time"]): t["speaker"] for t in got["turns"]}
        assert by_span[(6.0, 8.0)] == "Speaker B"
        assert diarize_local.UNKNOWN_SPEAKER not in by_span.values()

    def test_whole_turn_claimed_by_no_voice_is_unknown(self, monkeypatch):
        """An EMBEDDABLE turn the refinement parked in a cluster it does not
        resemble: 1.2 s of the unclaimed fill among 35 s of P and 30 s of a
        Q voice at angle 0.51 pi (fill 0.02 — a hair nearer the unclaimed
        fill than P is, so average linkage folds the 1.2 s into Q rather
        than pairing P with Q). k=3 would make it its own cluster but fails
        the strong duration floor (1.2 < 1.5 s); at k=2 it lands on Q's
        pooled centroid at cosine ~0.09 (-1 to P) — under UNCLAIMED_COSINE,
        so Unknown."""
        _stub_window_pass(monkeypatch)
        q_near = 0.02
        spec = [("p", 5.0, VOICE_P), ("q", 5.0, q_near)] * 6 + [
            ("??", 1.2, _UNCLAIMED), ("p", 5.0, VOICE_P),
        ]
        turns, fills, t = [], [], 0.0
        for text, dur, fill in spec:
            turns.append(_turn(t, t + dur, text=text))
            fills.append(fill)
            t += dur
        pcm = _voiced_pcm(turns, fills, t + 0.5)
        monkeypatch.setenv(diarize_local.UNKNOWN_FLAG_ENV, "1")
        got = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got is not None and got["num_speakers"] == 2
        assert got["turns"][12]["text"] == "??"
        assert got["turns"][12]["speaker"] == diarize_local.UNKNOWN_SPEAKER
        assert got["unknown_turns"] == 1 and got["unknown_seconds"] == 1.2
        assert got["segments_embedded"] == 14  # it WAS embedded; flagged after refinement
        assert [e["k"] for e in got["k_evaluated"] if e["ok"]] == [2]
        monkeypatch.setenv(diarize_local.UNKNOWN_FLAG_ENV, "0")
        got_off = diarize_local.diarize_turns(pcm, SR, turns, embed_fn=_mean_angle_embed)
        assert got_off["turns"][12]["speaker"] == "Speaker B"
        assert got_off["unknown_turns"] == 0


# ---------------------------------------------------------------------------
# Windows-first engine (2026-08-30) — transcript-free labelling, words regrouped
# ---------------------------------------------------------------------------

class TestWindowsFirst:
    _FILLS = (-1.0, 1 / 3, 1.0)  # three orthogonal, non-silent voices

    def _three_voice_pcm(self) -> np.ndarray:
        f = self._FILLS
        pcm = np.zeros(int(18.0 * SR), dtype=np.float32)
        for lo, hi, fill in [(0, 3, f[0]), (3, 6, f[1]), (6, 9, f[2]),
                             (9, 12, f[0]), (12, 15, f[1]), (15, 18, f[2])]:
            pcm[int(lo * SR):int(hi * SR)] = fill
        return pcm

    def _embedders(self):
        embed = _gram_embed(self._FILLS, np.eye(3))

        def embed_batch(chunks, sr):
            return [embed(c, sr) for c in chunks]
        return embed, embed_batch

    def test_labels_come_from_the_windows_and_words_follow_the_segments(self):
        """The same welded 9 s utterance the utterance engine splits via
        proposals: here the window timeline IS the labelling — three voices
        found by the eigengap, six segments at the voice changes, the welded
        utterance's words regrouped into three turns, the clean utterances
        untouched."""
        embed, embed_batch = self._embedders()
        turns = [
            _turn(0.0, 3.0, text="pp"), _turn(3.0, 6.0, text="qq"),
            _turn(6.0, 9.0, text="rr"), _turn(9.0, 18.0, text="w1 w2 w3"),
        ]
        got = diarize_local.diarize_windows_first(
            self._three_voice_pcm(), SR, turns, embed_fn=embed, embed_batch_fn=embed_batch,
        )
        assert got is not None
        assert got["source"] == diarize_local.SOURCE_WINDOWS
        assert got["num_speakers"] == 3
        assert got["split_utterances"] == 1
        assert got["uncovered_turns"] == 0
        assert got["k_evaluated"][0]["k_eigengap"] == 3
        assert got["k_evaluated"][0]["route"] == "spectral-windows"
        assert len(got["k_evaluated"][0]["eigenvalues"]) >= 4
        assert got["window_pass"]["k_eigengap"] == 3
        segs = got["segments"]
        assert [s["label"] for s in segs] == ["Speaker A", "Speaker B", "Speaker C"] * 2
        for s, edge in zip(segs, [0.0, 3.0, 6.0, 9.0, 12.0, 15.0]):
            assert abs(s["start"] - edge) <= 0.5, segs
        assert segs[-1]["end"] == pytest.approx(18.0, abs=1e-6)
        assert [t["speaker"] for t in got["turns"]] == [
            "Speaker A", "Speaker B", "Speaker C",
        ] * 2
        assert [t["text"] for t in got["turns"]] == ["pp", "qq", "rr", "w1", "w2", "w3"]
        assert all("words" not in t and "utterance" not in t for t in got["turns"])
        assert 0.0 <= got["agreement_with_input"] <= 1.0
        assert got["pooled_cosine"] <= 0.5

    def test_turn_breaks_at_the_utterance_boundary_even_without_a_voice_change(self):
        """Two utterances of ONE voice stay two turns (the transcriber's
        granularity is kept, as reanalyze-with-segments keeps it)."""
        embed, embed_batch = self._embedders()
        turns = [
            _turn(0.0, 1.5, text="a a"), _turn(1.5, 3.0, text="b b"),
            _turn(3.0, 6.0, text="qq"), _turn(6.0, 9.0, text="rr"),
            _turn(9.0, 18.0, text="w1 w2 w3"),
        ]
        got = diarize_local.diarize_windows_first(
            self._three_voice_pcm(), SR, turns, embed_fn=embed, embed_batch_fn=embed_batch,
        )
        assert got is not None
        assert [(t["speaker"], t["text"]) for t in got["turns"][:2]] == [
            ("Speaker A", "a a"), ("Speaker A", "b b"),
        ]

    def test_uncovered_speech_becomes_untranscribed_turns_labelled_by_the_timeline(self):
        """The transcript covers the first 9 s only: the other 9 s of speech
        become "(untranscribed)" turns carrying the SEGMENT's label (one per
        voice change), not a neighbour's."""
        embed, embed_batch = self._embedders()
        turns = [
            _turn(0.0, 3.0, text="pp"), _turn(3.0, 6.0, text="qq"),
            _turn(6.0, 9.0, text="rr"),
        ]
        got = diarize_local.diarize_windows_first(
            self._three_voice_pcm(), SR, turns, embed_fn=embed, embed_batch_fn=embed_batch,
        )
        assert got is not None
        assert got["num_speakers"] == 3
        extra = [t for t in got["turns"] if t["text"] == diarize_local.UNTRANSCRIBED_TEXT]
        assert got["uncovered_turns"] == len(extra) == 3
        assert [t["speaker"] for t in extra] == ["Speaker A", "Speaker B", "Speaker C"]
        for t, edge in zip(extra, [9.0, 12.0, 15.0]):
            assert abs(t["start_time"] - edge) <= 0.5, extra
        assert [t["text"] for t in got["turns"]] == [
            "pp", "qq", "rr", diarize_local.UNTRANSCRIBED_TEXT,
            diarize_local.UNTRANSCRIBED_TEXT, diarize_local.UNTRANSCRIBED_TEXT,
        ]

    def test_one_voice_returns_none_so_the_caller_falls_back(self):
        """Every window the same voice → the refined affinity is rank one,
        the eigengap says k=1 and the engine has nothing to say."""
        turns = [_turn(0.0, 4.0, text="a"), _turn(4.0, 8.0, text="b")]
        pcm = _voiced_pcm(turns, [VOICE_A, VOICE_A], 8.0)

        def embed_batch(chunks, sr):
            return [_mean_angle_embed(c, sr) for c in chunks]

        got = diarize_local.diarize_windows_first(
            pcm, SR, turns, embed_fn=_mean_angle_embed, embed_batch_fn=embed_batch,
        )
        assert got is None

    def test_too_little_speech_returns_none(self):
        turns = [_turn(0.0, 1.0, text="a")]
        pcm = _voiced_pcm(turns, [VOICE_A], 1.0)
        got = diarize_local.diarize_windows_first(
            pcm, SR, turns, embed_fn=_mean_angle_embed,
        )
        assert got is None

    def test_voice_model_unavailable_returns_none(self):
        def broken(chunks, sr):
            raise SpeakerIdUnavailable("no torch")

        turns = [_turn(0.0, 4.0, text="a"), _turn(4.0, 8.0, text="b")]
        pcm = _voiced_pcm(turns, [VOICE_A, VOICE_B], 8.0)
        got = diarize_local.diarize_windows_first(
            pcm, SR, turns, embed_fn=_mean_angle_embed, embed_batch_fn=broken,
        )
        assert got is None


class TestRegroupTranscriptBySegments:
    """The word regrouping shared by the windows engine and
    POST …/reanalyze-with-segments (main.py delegates here)."""

    def test_breaks_at_label_change_and_at_utterance_boundary(self):
        rows = [
            {"speaker": "X", "text": "one two three four", "start_time": 0.0, "end_time": 4.0},
            {"speaker": "X", "text": "five six", "start_time": 4.0, "end_time": 6.0},
        ]
        segs = [
            {"start": 0.0, "end": 2.0, "label": "A"},
            {"start": 2.0, "end": 6.0, "label": "B"},
        ]
        got = diarize_local.regroup_transcript_by_segments(rows, segs)
        assert [(t["speaker"], t["text"]) for t in got] == [
            ("A", "one two"), ("B", "three four"), ("B", "five six"),
        ]
        assert all("utterance" not in t for t in got)

    def test_accepts_objects_with_attributes_and_unsorted_segments(self):
        class Seg:
            def __init__(self, s, e, lab):
                self.start, self.end, self.label = s, e, lab

        rows = [{"speaker": "X", "text": "a b", "start_time": 0.0, "end_time": 2.0}]
        got = diarize_local.regroup_transcript_by_segments(
            rows, [Seg(1.0, 2.0, " B "), Seg(0.0, 1.0, "A")],
        )
        assert [(t["speaker"], t["text"]) for t in got] == [("A", "a"), ("B", "b")]

    def test_no_words_yields_empty(self):
        assert diarize_local.regroup_transcript_by_segments(
            [{"speaker": "X", "text": "  ", "start_time": 0.0, "end_time": 1.0}],
            [{"start": 0.0, "end": 1.0, "label": "A"}],
        ) == []
