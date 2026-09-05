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
    rubric.json           the OWNER'S per-second listen-through (dad/mom/asher,
                          overlap segments list both) — the ground truth

Measured against that rubric (2026-08-29): the shipped pipeline scores
frame accuracy 0.64 on this transcript with dad-cluster purity 0.76 — it
finds three voices but mixes them (dad's "Hey hey hey, settle down" lands
on mom; "What do you think, mom?" on the son). This test pins only the
narrow thing the 0.33 recalibration fixed (k=3 found, the welded first
utterance split, dad-cluster purity not collapsing back to ~0.5); the
research bake-off in docs/research/2026-08-29-voice-separation/ is where
the real fix is being worked out. Skipped whenever the private files are absent.
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

_RUBRIC = _PRIVATE / "maggiano3" / "rubric.json"


def _gt() -> list[tuple[float, float, tuple[str, ...]]]:
    """The OWNER'S per-second listen-through rubric (v2, 2026-08-15; ±1 s
    boundary slop; overlap segments list every speaker — either credited).
    Labels: dad / mom / asher. Kept as the single source of truth — the
    research scorer (docs/research/2026-08-29-voice-separation/score.py)
    reads the same file."""
    rub = json.loads(_RUBRIC.read_text())
    out = []
    for seg in rub["segments"]:
        sp = seg["speaker"]
        out.append((float(seg["start"]), float(seg["end"]), tuple(sp) if isinstance(sp, list) else (sp,)))
    return out


def _overlap(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))


def owner_purity(turns: list[dict]) -> tuple[float, int]:
    """Best one-to-one time-weighted mapping of predicted labels onto the
    rubric's speakers; returns (purity of the cluster mapped to dad, number
    of predicted speakers). An overlap second credits either speaker."""
    gt = _gt()
    labels = sorted({t["speaker"] for t in turns})
    people = ["dad", "mom", "asher"]

    def credit(lab: str, person: str) -> float:
        return sum(
            _overlap(t["start_time"], t["end_time"], a, b)
            for t in turns if t["speaker"] == lab
            for a, b, who in gt if person in who
        )

    best, best_map = -1.0, {}
    for perm in itertools.permutations(people, min(len(labels), 3)):
        m = dict(zip(labels, perm))
        sc = sum(credit(lab, m[lab]) for lab in m)
        if sc > best:
            best, best_map = sc, m
    dad_label = next((lab for lab, p in best_map.items() if p == "dad"), None)
    if dad_label is None:
        return 0.0, len(labels)
    claimed = sum(
        _overlap(t["start_time"], t["end_time"], a, b)
        for t in turns if t["speaker"] == dad_label for a, b, _ in gt
    )
    return (credit(dad_label, "dad") / claimed if claimed else 0.0), len(labels)


@pytest.mark.skipif(not _VOICE_OK, reason="voice deps (torch + speechbrain) not installed")
@pytest.mark.skipif(
    not (_AUDIO.exists() and _TRANSCRIPT.exists() and _RUBRIC.exists()),
    reason="private maggiano3 fixture absent (tmp/private_fixtures/maggiano3/ incl. rubric.json)",
)
def test_maggiano3_owner_and_son_are_not_one_speaker():
    """Against the owner's rubric: k=3 and dad-cluster purity 0.76 at
    STRONG_SEPARATION_COSINE=0.33 (vs k=2 / ~0.5 at the old 0.32 bar, son
    folded into dad). Floor set at 0.70 — a regression, not a target."""
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
    assert purity >= 0.70, (
        f"maggiano3: dad-cluster purity {purity:.2f} < 0.70 — the son is being "
        f"merged into the owner again\n{detail}"
    )
    first = got["turns"][0]
    assert "Duolingo" not in first["text"], (
        "maggiano3: the welded first utterance (owner question + son's answer) "
        f"was not split: {first['text']!r}"
    )


@pytest.mark.skipif(not _VOICE_OK, reason="voice deps (torch + speechbrain) not installed")
@pytest.mark.skipif(
    not (_AUDIO.exists() and _TRANSCRIPT.exists() and _RUBRIC.exists()),
    reason="private maggiano3 fixture absent (tmp/private_fixtures/maggiano3/ incl. rubric.json)",
)
def test_windows_first_maggiano3_three_voices():
    """The windows engine (production default since 2026-08-30) on the same
    transcript. Measured 2026-08-30 against the rubric: eigengap k=3, frame
    accuracy 0.694 (7utt) / 0.681 (8utt) with dad-cluster purity 0.775 —
    the utterance engine's 0.702 / 0.671 at purity 0.80 / 0.79; on the
    rubric's own boundaries 0.865 vs 0.833, and the engine's raw segment
    timeline scores 0.761 (the bake-off's B number, reproduced). What the
    transcript's turns cannot reach is the 7-8 % of rubric speech that has
    no words and sits under the speech gate or in sub-0.4 s gaps. Pinned:
    k=3, purity ≥ 0.75, the welded first utterance split."""
    import audio_ingest

    data = _AUDIO.read_bytes()
    pcm, sr = audio_ingest.decode_to_pcm_16k(data, "audio.m4a")
    raw = json.loads(_TRANSCRIPT.read_text())
    got = diarize_local.diarize_windows_first(pcm, sr, [dict(t) for t in raw])
    assert got is not None, "maggiano3: the windows engine returned nothing"
    assert got["source"] == diarize_local.SOURCE_WINDOWS
    purity, k = owner_purity(got["turns"])
    detail = "\n".join(
        f"  {t['speaker']:9s} {t['start_time']:6.2f}-{t['end_time']:6.2f} {t['text'][:70]!r}"
        for t in got["turns"]
    )
    assert got["k_evaluated"][0]["k_eigengap"] == 3, f"eigengap {got['k_evaluated']}\n{detail}"
    assert k == 3, f"maggiano3 (windows): heard {k} speakers, expected 3\n{detail}"
    assert purity >= 0.75, (
        f"maggiano3 (windows): dad-cluster purity {purity:.2f} < 0.75\n{detail}"
    )
    first = got["turns"][0]
    assert "Duolingo" not in first["text"], (
        "maggiano3 (windows): the welded first utterance was not split: "
        f"{first['text']!r}"
    )
