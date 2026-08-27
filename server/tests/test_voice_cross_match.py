"""Cross-recording ("contrast") voice matching — speaker_id.identify_from_embeddings.

Pure tests over small synthetic vectors, plus two real-audio checks over the
checked-in fixtures. Calibration data behind the constants lives in
speaker_id.py next to CROSS_MATCH_THRESHOLD.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import speaker_id

_FIX = Path(__file__).resolve().parent / "fixtures" / "audio"
_FAMILY = _FIX / "test_recording_family_real.wav"
_POKER = _FIX / "test_recording_poker6_real.wav"


def _unit(*xs: float) -> np.ndarray:
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def _at_cosine(c: float) -> np.ndarray:
    """A 2-d unit vector at cosine ``c`` to (1, 0)."""
    return _unit(c, float(np.sqrt(max(0.0, 1.0 - c * c))))


SELF = _unit(1.0, 0.0)
TWO_SETTINGS = {"self": {"display_name": "You", "is_self": True, "settings": 2}}
ONE_SETTING = {"self": {"display_name": "You", "is_self": True, "settings": 1}}


# ---------------------------------------------------------------------------
# Absolute path unchanged
# ---------------------------------------------------------------------------

def test_absolute_match_reports_basis_and_embedding():
    rep = speaker_id.identify_from_embeddings(
        {"Speaker A": _at_cosine(0.80), "Speaker B": _at_cosine(0.10)},
        {"self": SELF}, people=ONE_SETTING,
    )
    assert rep["matched"] == {"Speaker A": "self"}
    a = rep["speakers"]["Speaker A"]
    assert a["match_basis"] == "absolute" and a["is_you"] is True
    assert rep["speakers"]["Speaker B"]["match_basis"] is None
    assert rep["matched_speaker"] == "Speaker A"
    assert rep["cross_match_threshold"] == speaker_id.CROSS_MATCH_THRESHOLD
    # The embedding rides along (unit norm) for audio-free re-matching.
    e = np.asarray(a["embedding"], dtype=np.float32)
    assert np.isclose(np.linalg.norm(e), 1.0)
    assert np.allclose(e, _at_cosine(0.80), atol=1e-6)


# ---------------------------------------------------------------------------
# Contrast path — all four conditions
# ---------------------------------------------------------------------------

def test_contrast_match_when_print_pools_two_settings_and_speaker_stands_out():
    # The real poker-night shape: owner 0.42, runner-up 0.19 (gap 0.23).
    rep = speaker_id.identify_from_embeddings(
        {"P6": _at_cosine(0.42), "P2": _at_cosine(0.19), "P1": _at_cosine(0.12)},
        {"self": SELF}, people=TWO_SETTINGS,
    )
    assert rep["matched"] == {"P6": "self"}
    assert rep["speakers"]["P6"]["match_basis"] == "contrast"
    assert rep["speakers"]["P6"]["is_you"] is True


def test_contrast_needs_a_multi_setting_print():
    # Same scores, but the print came from ONE recording: honest miss
    # (single-setting cross-recording scores overlap different-people scores).
    rep = speaker_id.identify_from_embeddings(
        {"P6": _at_cosine(0.42), "P2": _at_cosine(0.19)},
        {"self": SELF}, people=ONE_SETTING,
    )
    assert rep["matched"] == {}
    # Unknown settings counts as one.
    rep = speaker_id.identify_from_embeddings(
        {"P6": _at_cosine(0.42), "P2": _at_cosine(0.19)}, {"self": SELF},
    )
    assert rep["matched"] == {}


def test_contrast_needs_a_second_speaker_to_contrast_against():
    rep = speaker_id.identify_from_embeddings(
        {"Only": _at_cosine(0.55)}, {"self": SELF}, people=TWO_SETTINGS,
    )
    assert rep["matched"] == {}


def test_contrast_needs_the_margin():
    # 0.45 vs 0.35 — a 0.10 gap is inside different-people noise; no match.
    rep = speaker_id.identify_from_embeddings(
        {"A": _at_cosine(0.45), "B": _at_cosine(0.35)},
        {"self": SELF}, people=TWO_SETTINGS,
    )
    assert rep["matched"] == {}
    # Exactly the margin clears it.
    rep = speaker_id.identify_from_embeddings(
        {"A": _at_cosine(0.45), "B": _at_cosine(0.30)},
        {"self": SELF}, people=TWO_SETTINGS,
    )
    assert rep["matched"] == {"A": "self"}


def test_contrast_needs_the_floor():
    rep = speaker_id.identify_from_embeddings(
        {"A": _at_cosine(0.39), "B": _at_cosine(0.05)},
        {"self": SELF}, people=TWO_SETTINGS,
    )
    assert rep["matched"] == {}


def test_one_to_one_still_holds_across_bases():
    # Two speakers both contrast-eligible for the same person: only the
    # stronger one is labeled (a person is one voice).
    rep = speaker_id.identify_from_embeddings(
        {"A": _at_cosine(0.50), "B": _at_cosine(0.45), "C": _at_cosine(0.05)},
        {"self": SELF}, people=TWO_SETTINGS,
    )
    # A beats B by only 0.05 → A is NOT contrast-eligible; neither is B.
    assert rep["matched"] == {}


# ---------------------------------------------------------------------------
# stored_speaker_embeddings — the audio-free re-match reader
# ---------------------------------------------------------------------------

def test_stored_speaker_embeddings_reads_only_well_formed_vectors():
    good = (np.arange(speaker_id.EMBEDDING_DIM, dtype=np.float32) + 1.0).tolist()
    identity = {"speakers": {
        "Speaker A": {"embedding": good},
        "Speaker B": {"embedding": [1.0, 2.0]},            # wrong dim
        "Speaker C": {"scores": {"self": 0.1}},             # legacy, no vector
        "Speaker D": {"embedding": [float("nan")] * speaker_id.EMBEDDING_DIM},
        7: {"embedding": good},                             # non-string key
    }}
    out = speaker_id.stored_speaker_embeddings(identity)
    assert set(out) == {"Speaker A"}
    assert np.isclose(np.linalg.norm(out["Speaker A"]), 1.0)
    assert speaker_id.stored_speaker_embeddings(None) == {}
    assert speaker_id.stored_speaker_embeddings({"matched_speaker": "Speaker A"}) == {}


def test_roundtrip_report_embeddings_rescore_identically():
    rep = speaker_id.identify_from_embeddings(
        {"A": _unit(*([1.0] + [0.0] * (speaker_id.EMBEDDING_DIM - 1))),
         "B": _unit(*([0.0, 1.0] + [0.0] * (speaker_id.EMBEDDING_DIM - 2)))},
        {"self": _unit(*([1.0] + [0.0] * (speaker_id.EMBEDDING_DIM - 1)))},
    )
    again = speaker_id.identify_from_embeddings(
        speaker_id.stored_speaker_embeddings(rep),
        {"self": _unit(*([1.0] + [0.0] * (speaker_id.EMBEDDING_DIM - 1)))},
    )
    assert again["matched"] == rep["matched"] == {"A": "self"}
    assert again["speakers"]["A"]["scores"] == rep["speakers"]["A"]["scores"]


# ---------------------------------------------------------------------------
# Real audio — the contrast path must not mint a false "You"
# ---------------------------------------------------------------------------

_needs_voice = pytest.mark.skipif(
    not speaker_id.is_available(), reason="voice deps (torch + speechbrain) not installed",
)


def _pcm(path: Path):
    import audio_ingest
    return audio_ingest.decode_to_pcm_16k(path.read_bytes(), path.name)


@_needs_voice
@pytest.mark.skipif(not (_FAMILY.exists() and _POKER.exists()), reason="real fixtures missing")
def test_single_setting_print_never_false_matches_poker_night():
    """Measured 2026-08-27: a print enrolled from the family clip alone scores
    the owner at poker night 0.24 and the five other men 0.01-0.22 — an honest
    MISS under both bars (single setting → no contrast path), and no false
    "You" on anyone else. Pinned so a relaxed rule can never turn the five
    strangers into the owner."""
    fp, fsr = _pcm(_FAMILY)
    meta = json.loads(_FAMILY.with_name(_FAMILY.stem + "_meta.json").read_text())
    owner_turns = [t for t in meta["turns"] if t["speaker"] == "Sage"]
    print_vec = speaker_id.embed_speaker(fp, fsr, owner_turns, "Sage")
    assert print_vec is not None
    pp, psr = _pcm(_POKER)
    players = {f"P{i + 1}": speaker_id.embed_pcm(
        np.ascontiguousarray(pp[int((5 * i + 0.3) * psr):int((5 * i + 4.8) * psr)]), psr,
    ) for i in range(6)}
    for people in (ONE_SETTING, TWO_SETTINGS):
        rep = speaker_id.identify_from_embeddings(players, {"self": print_vec}, people=people)
        scores = {k: v["scores"]["self"] for k, v in rep["speakers"].items()}
        others = {k: v for k, v in scores.items() if k != "P6"}
        assert max(others.values()) < speaker_id.CROSS_MATCH_THRESHOLD, scores
        assert not any(sp != "P6" for sp in rep["matched"]), (rep["matched"], scores)
