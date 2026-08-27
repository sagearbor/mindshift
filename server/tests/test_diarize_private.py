"""Opt-in regression against the owner's PRIVATE 3-person family recording.

The recording ("maggiano's to go or not 3 person", 2026-08-14: owner + wife +
son, 42s, restaurant) is NOT checked in — it is private family audio. It is
the fourth real calibration point behind ``STRONG_SEPARATION_COSINE`` (see
diarize_local.py, "RECALIBRATED 0.32 -> 0.33 (2026-08-27)"). To run this
locally, drop the recording under ``tmp/private_fixtures/maggiano3/`` (or
point ``MINDSHIFT_PRIVATE_FIXTURES`` at a directory holding ``maggiano3/``):

    audio.m4a             the stored derivative (GCS recordings/<uid>/<id>/audio.m4a)
    transcript_7utt.json  the Deepgram transcript variant with word timings
                          (7 utterances, 2 speakers) that exposed the merge

Ground truth is the owner's report ("my son was merged into my speaker")
plus in-recording reference-voice affinity — approximate interval labels
below, used only for a time-weighted owner-turn PURITY check, never exact
per-turn accuracy. Skipped whenever the private files are absent.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import pytest

import diarize_local

_ROOT = Path(__file__).resolve().parents[2]
_PRIVATE = Path(
    os.getenv("MINDSHIFT_PRIVATE_FIXTURES") or (_ROOT / "tmp" / "private_fixtures")
)
_AUDIO = _PRIVATE / "maggiano3" / "audio.m4a"
_TRANSCRIPT = _PRIVATE / "maggiano3" / "transcript_7utt.json"

try:  # voice deps are optional
    import speaker_id

    _VOICE_OK = speaker_id.is_available()
except Exception:  # noqa: BLE001
    _VOICE_OK = False

# Best-estimate speaker intervals (seconds): owner / wife / son.
_GT = [
    (0.8, 2.9, "owner"), (2.9, 4.98, "son"), (5.44, 6.5, "son"),
    (6.6, 8.02, "owner"), (8.88, 9.8, "owner"), (9.8, 10.74, "wife"),
    (11.04, 12.97, "wife"), (13.36, 17.61, "owner"), (18.39, 21.9, "son"),
    (21.9, 23.6, "owner"), (23.9, 25.1, "wife"), (25.2, 29.2, "son"),
    (29.2, 33.7, "wife"), (34.0, 36.2, "son"), (36.2, 36.8, "owner"),
    (36.8, 37.5, "son"), (38.0, 40.8, "owner"),
]


def _overlap(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))


def owner_purity(turns: list[dict]) -> tuple[float, int]:
    """Best-permutation time-weighted mapping; returns (owner-cluster purity,
    number of predicted speakers)."""
    labels = sorted({t["speaker"] for t in turns})
    people = ["owner", "wife", "son"]
    conf = {
        (lab, p): sum(
            _overlap(t["start_time"], t["end_time"], a, b)
            for t in turns if t["speaker"] == lab
            for a, b, who in _GT if who == p
        )
        for lab in labels for p in people
    }
    best, best_map = -1.0, {}
    for perm in itertools.permutations(people, min(len(labels), 3)):
        m = dict(zip(labels, perm))
        s = sum(conf[(lab, m[lab])] for lab in m)
        if s > best:
            best, best_map = s, m
    owner_label = next(lab for lab, p in best_map.items() if p == "owner")
    total = sum(conf[(owner_label, p)] for p in people)
    return (conf[(owner_label, "owner")] / total if total else 0.0), len(labels)


@pytest.mark.skipif(not _VOICE_OK, reason="voice deps (torch + speechbrain) not installed")
@pytest.mark.skipif(
    not (_AUDIO.exists() and _TRANSCRIPT.exists()),
    reason="private maggiano3 fixture absent (tmp/private_fixtures/maggiano3/)",
)
def test_maggiano3_owner_and_son_are_not_one_speaker():
    """Measured 2026-08-27 at STRONG_SEPARATION_COSINE=0.33: k=3, owner-turn
    purity 0.79, welded first utterance split owner/son at the word level.
    At the old 0.32 bar the genuine third voice's marginal split (0.325) was
    rejected: k=2, purity 0.52, son folded into the owner."""
    import audio_ingest

    data = _AUDIO.read_bytes()
    pcm, sr = audio_ingest.decode_to_pcm_16k(data, "audio.m4a")
    raw = json.loads(_TRANSCRIPT.read_text())
    got = diarize_local.diarize_turns(pcm, sr, [dict(t) for t in raw])
    assert got is not None, "maggiano3: local diarization returned nothing"
    purity, k = owner_purity(got["turns"])
    detail = "\n".join(
        f"  {t['speaker']:9s} {t['start_time']:6.2f}-{t['end_time']:6.2f} {t['text'][:70]!r}"
        for t in got["turns"]
    )
    assert k == 3, f"maggiano3: heard {k} speakers, expected 3\n{detail}"
    assert purity >= 0.75, (
        f"maggiano3: owner-turn purity {purity:.2f} < 0.75 — the son is being "
        f"merged into the owner again\n{detail}"
    )
    first = got["turns"][0]
    assert "Duolingo" not in first["text"], (
        "maggiano3: the welded first utterance (owner question + son's answer) "
        f"was not split: {first['text']!r}"
    )
