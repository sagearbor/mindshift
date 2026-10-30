"""Overnight-tuning tools on synthetic data: post-hoc speak gates
(recreplay.regate), the 30 s clip slicer (recreplay.clips) and the judge
export context window (recreplay.judge_export)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from recreplay import clips, judge_export, regate  # noqa: E402


def _ln(at, utt, imp=80, uid=None, kind="response", source="server"):
    return {"source": source, "kind": kind, "at_s": at, "utterance_text": utt, "importance": imp, "fires": True,
            "speak": True, "turn_uid": uid, "text": "x"}


def test_gates_only_remove_server_fires():
    lines = [_ln(1.0, "yeah", uid="a"), _ln(5.0, "I think we should go now", 90, "b"),
             _ln(10.0, "this is a long enough turn", 40, "c"), _ln(20.0, "another long enough turn here", 85, "d"),
             {**_ln(21.0, None), "source": "phone", "kind": "haptic"}]
    facts = {"sent": {"a": {"is_self": True}, "b": {"is_self": True}, "c": {"is_self": True}, "d": {"is_self": None}},
             "laughs": [4.0]}
    keep = lambda g: [ln["at_s"] for ln in regate.apply_gate(lines, facts, g) if ln["fires"]]   # noqa: E731
    assert keep({}) == [1.0, 5.0, 10.0, 20.0, 21.0]
    assert keep({"min_words": 3}) == [5.0, 10.0, 20.0, 21.0]
    assert keep({"skip_bc": True}) == [5.0, 10.0, 20.0, 21.0]
    assert keep({"laugh_s": 2.0}) == [1.0, 10.0, 20.0, 21.0]
    assert keep({"importance": 60}) == [1.0, 5.0, 20.0, 21.0]
    assert keep({"importance": 60, "unknown_cap": 50}) == [1.0, 5.0, 21.0]
    assert keep({"min_gap_s": 6.0}) == [1.0, 10.0, 20.0, 21.0]
    assert keep({"max_per_5min": 2}) == [1.0, 5.0, 21.0]
    assert lines[0]["fires"] is True                       # input untouched


def test_pareto_front_collapses_ties_to_the_simplest_gate():
    rows = [{"gate": {**regate.DEFAULT_GATE}, "metrics": {"calm_fires_h": 300, "lift": -0.05}},
            {"gate": {**regate.DEFAULT_GATE, "importance": 80}, "metrics": {"calm_fires_h": 40, "lift": 0.05}},
            {"gate": {**regate.DEFAULT_GATE, "importance": 80, "skip_bc": True}, "metrics": {"calm_fires_h": 40, "lift": 0.05}},
            {"gate": {**regate.DEFAULT_GATE, "min_gap_s": 30.0}, "metrics": {"calm_fires_h": 90, "lift": 0.02}}]
    front = regate.pareto(rows)
    assert [r["gate"].get("importance") for r in front] == [80]           # dominates both others
    assert front[0]["gate"]["skip_bc"] is False


def test_clip_slicing_shifts_times_and_drops_outside():
    ann = {"segments": [{"start": 5.0, "end": 12.0, "speaker": "S1", "text": "a"},
                        {"start": 11.95, "end": 12.05, "speaker": "S2", "text": "b"},
                        {"start": 40.0, "end": 50.0, "speaker": "S2", "text": "c"}],
           "events": [{"t": 9.0, "type": "laughter"}, {"t": 60.0, "type": "laughter"}],
           "coach_moments": [{"t": 11.0, "ideal_nudge": "Let them finish first."}],
           "audio": {"duration_s": 300.0}, "source": {"window_s": [100.0, 400.0]}, "annotator": {"notes": "n"}}
    out = clips.slice_annotation(ann, 8.0, 38.0, keep_moments=True, clip_name="c", parent="p")
    assert [(s["start"], s["end"]) for s in out["segments"]] == [(0.0, 4.0)]   # 0.1 s sliver dropped
    assert out["events"] == [{"t": 1.0, "type": "laughter"}]
    assert out["coach_moments"][0]["t"] == 3.0
    assert out["source"]["window_s"] == [108.0, 138.0] and out["audio"]["duration_s"] == 30.0
    assert clips.slice_annotation(ann, 8.0, 38.0, keep_moments=False, clip_name="c", parent="p")["coach_moments"] == []
    raw = {"results": {"channels": [{"alternatives": [{"words": [
        {"word": "hi", "start": 7.9, "end": 8.2}, {"word": "there", "start": 9.0, "end": 9.4},
        {"word": "late", "start": 37.9, "end": 38.3}]}]}]}}
    words = clips.slice_deepgram(raw, 8.0, 38.0, parent="p")["results"]["channels"][0]["alternatives"][0]["words"]
    assert words == [{"word": "there", "start": 1.0, "end": 1.4}]


def test_clip_start_keeps_the_scoring_window_inside():
    assert clips.clip_start(50.0, 52.0, 300.0) == 35.0
    assert clips.clip_start(50.0, 62.0, 300.0) == 39.0          # anchor + 6 s + 1 s = 69 = start + 30
    assert clips.clip_start(5.0, 7.0, 300.0) == 0.0
    assert clips.clip_start(295.0, 297.0, 300.0) == 270.0


def test_judge_context_window():
    turns = [{"start": float(i), "end": i + 0.8, "text": str(i)} for i in range(12)]
    before, after = judge_export._context(turns, 8.5)
    assert [t["text"] for t in before] == ["3", "4", "5", "6", "7", "8"]
    assert [t["text"] for t in after] == ["9", "10"]


@pytest.mark.parametrize("notes,wid", [("who: I'm S4 (corpus speaker)", "S4"), ("who: the man", None)])
def test_clip_wearer_from_notes(notes, wid):
    assert clips._wearer_id(notes) == wid
