"""Gemini audio annotator (scripts/annotate_audio.py + scripts/annotator/):
cost ledger + hard cap, window planning and stitching, output sanitising,
validation scoring, and the open-audio licence filter. Pure functions, no
network, no audio: always runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from annotator import core, licence, scoring

SCHEMA = Path(__file__).parent / "fixtures" / "annotation" / "mindshift-annotation-v1.schema.json"
DOC = Path(__file__).resolve().parents[2] / "docs" / "recording-annotation-format.md"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def test_prompt_is_the_documented_prompt():
    doc = DOC.read_text()
    for line in core.PROMPT.splitlines():
        line = line.strip()
        if line.startswith("- ") and len(line) > 30:
            # each rule line of the prompt appears verbatim in the doc
            assert line[:60] in doc.replace("> ", ""), line
    assert '"format": "mindshift-annotation/v1"' in core.PROMPT
    assert "SCALES:" in core.PROMPT


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def test_cost_counts_audio_input_and_thinking_as_output():
    usage = {"audio_in": 1_000_000, "text_in": 0, "out": 0, "thoughts": 0}
    p = core.price_for("gemini-2.5-flash")
    assert core.estimate_cost("gemini-2.5-flash", usage) == pytest.approx(p["audio_in"])
    usage = {"audio_in": 0, "text_in": 0, "out": 500_000, "thoughts": 500_000}
    assert core.estimate_cost("gemini-2.5-flash", usage) == pytest.approx(p["out"])


def test_unknown_model_is_priced_at_the_most_expensive_known_rate():
    worst = max(v["out"] for v in core.PRICES.values())
    assert core.price_for("gemini-9-ultra")["out"] == worst


def test_ledger_hard_stops_at_cap(tmp_path):
    led = core.Ledger(tmp_path / "spend.json", cap_usd=1.0)
    led.record("m", "a.wav", 0.6, {})
    assert led.total == pytest.approx(0.6)
    led.check(0.3)  # fine
    with pytest.raises(core.BudgetExceeded):
        led.check(0.5)
    # persisted across instances (the cap spans the whole effort)
    led2 = core.Ledger(tmp_path / "spend.json", cap_usd=1.0)
    assert led2.total == pytest.approx(0.6)
    led2.record("m", "b.wav", 0.5, {})
    with pytest.raises(core.BudgetExceeded):
        led2.check(0.0)


# ---------------------------------------------------------------------------
# Windows + stitching
# ---------------------------------------------------------------------------

def test_plan_windows_short_file_is_one_window():
    assert core.plan_windows(300.0, window_s=600, overlap_s=30) == [(0.0, 300.0)]


def test_plan_windows_overlap_and_cover():
    w = core.plan_windows(1300.0, window_s=600, overlap_s=30)
    assert w[0] == (0.0, 600.0)
    assert w[-1][1] == 1300.0
    for (a0, a1), (b0, b1) in zip(w, w[1:]):
        assert b0 == pytest.approx(a1 - 30)


def _seg(s, e, spk, text="hello there"):
    return {"start": s, "end": e, "speaker": spk, "text": text,
            "vocal": {"emotion": "neutral", "intensity": 1, "volume": "normal", "pace": "normal"}}


def test_stitch_maps_speakers_by_overlap_region_and_cuts_at_midpoint():
    # window 0: 0-100, window 1: 80-180 (local times 0-100), overlap 80-100
    w0 = {"speakers": [{"id": "S1", "voice_description": "man"}, {"id": "S2", "voice_description": "woman"}],
          "segments": [_seg(10, 20, "S1"), _seg(82, 88, "S1", "the same words"), _seg(90, 98, "S2", "other words")],
          "events": [{"t": 95.0, "type": "question", "speakers": ["S2"]}],
          "coach_moments": [], "summary": {"overall_heat": 1, "peak_heat_t": 15.0, "tone_arc": "calm"}}
    # window 1 calls the woman S1 and the man S2, plus a new S3
    w1 = {"speakers": [{"id": "S1", "voice_description": "woman"}, {"id": "S2", "voice_description": "man"},
                       {"id": "S3", "voice_description": "child"}],
          "segments": [_seg(2, 8, "S2", "the same words"), _seg(10, 18, "S1", "other words"),
                       _seg(40, 50, "S3", "new kid")],
          "events": [{"t": 15.0, "type": "question", "speakers": ["S1"]}],
          "coach_moments": [{"t": 45.0, "for_speaker": "S3", "ideal_nudge": "x"}],
          "summary": {"overall_heat": 3, "peak_heat_t": 45.0, "tone_arc": "heated"}}
    out = core.stitch([w0, w1], [(0.0, 100.0), (80.0, 180.0)])
    ids = [s["id"] for s in out["speakers"]]
    assert ids == ["S1", "S2", "S3"]
    starts = [(s["start"], s["speaker"]) for s in out["segments"]]
    # cut at 90: window-0 segments before 90, window-1 segments from 90 on
    assert (10, "S1") in starts and (82, "S1") in starts
    assert (90, "S2") in starts            # window-1 "S1" (woman) -> global S2
    assert (120, "S3") in starts
    assert all(s["start"] != 82 or s["speaker"] == "S1" for s in out["segments"])
    assert len([s for s in out["segments"] if s["text"] == "other words"]) == 1
    assert out["summary"]["overall_heat"] == 3
    assert out["summary"]["peak_heat_t"] == pytest.approx(125.0)
    assert out["coach_moments"][0]["for_speaker"] == "S3"
    assert out["coach_moments"][0]["t"] == pytest.approx(125.0)
    assert len(out["events"]) == 1 and out["events"][0]["speakers"] == ["S2"]


def test_offset_shifts_all_times():
    obj = {"segments": [_seg(1, 2, "S1")], "events": [{"t": 1.0, "type": "question"}],
           "coach_moments": [{"t": 2.0, "for_speaker": "S1"}], "summary": {"peak_heat_t": 3.0}}
    o = core.offset(obj, 10.0)
    assert o["segments"][0]["start"] == 11 and o["events"][0]["t"] == 11
    assert o["coach_moments"][0]["t"] == 12 and o["summary"]["peak_heat_t"] == 13


# ---------------------------------------------------------------------------
# Sanitise + validate
# ---------------------------------------------------------------------------

def test_sanitize_makes_messy_output_schema_valid():
    jsonschema = pytest.importorskip("jsonschema")
    messy = {
        "speakers": [{"id": "S1", "approx_age": "Adult", "talk_share_pct": 140},
                     {"id": "speaker 2"}],
        "segments": [
            {"start": "1.25", "end": 3, "speaker": "S1", "text": "hi",
             "vocal": {"emotion": "Furious", "intensity": 2.6, "volume": "loud", "pace": "normal"},
             "text_emotion": "negative", "confidence": 1.4},
            {"start": 5, "end": 4, "speaker": "S2", "text": None},
        ],
        "events": [{"t": 2, "type": "shouting"}, {"t": 3, "type": "interruption"}],
        "coach_moments": [{"t": 1, "for_speaker": "S1", "kind": "tip", "priority": 0}],
        "summary": {"overall_heat": 5, "peak_heat_t": "2"},
    }
    fixes: list[str] = []
    out = core.sanitize(messy, model="gemini-x", duration_s=10.0, fixes=fixes)
    schema = json.loads(SCHEMA.read_text())
    errs = list(jsonschema.Draft202012Validator(schema).iter_errors(out))
    assert errs == [], [e.message for e in errs]
    assert out["format"] == "mindshift-annotation/v1"
    assert out["annotator"]["model"] == "gemini-x"
    assert out["audio"]["duration_s"] == 10.0
    assert out["segments"][0]["vocal"]["intensity"] == 3
    assert out["segments"][0]["vocal"]["emotion"] is None
    assert out["segments"][1]["end"] >= out["segments"][1]["start"]
    assert [e["type"] for e in out["events"]] == ["interruption"]
    assert out["summary"]["overall_heat"] == 3
    assert fixes


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_wer_basic():
    assert scoring.wer("the cat sat", "the cat sat") == 0.0
    assert scoring.wer("the cat sat on the mat", "the cat sat on mat") == pytest.approx(1 / 6)
    assert scoring.wer("Hello, World!", "hello world") == 0.0


def test_der_perfect_and_label_permutation():
    truth = {"A": [(0, 10)], "B": [(10, 20)]}
    hyp = {"S2": [(0, 10)], "S1": [(10, 20)]}
    r = scoring.der(truth, hyp, 20.0)
    assert r["der"] == pytest.approx(0.0, abs=1e-6)
    assert r["mapping"] == {"S2": "A", "S1": "B"}


def test_der_confusion_and_miss():
    truth = {"A": [(0, 10)], "B": [(10, 20)]}
    hyp = {"S1": [(0, 15)]}   # B's first 5 s confused, last 5 s missed
    r = scoring.der(truth, hyp, 20.0)
    assert r["confusion"] == pytest.approx(0.25, abs=0.01)
    assert r["miss"] == pytest.approx(0.25, abs=0.01)
    assert r["der"] == pytest.approx(0.5, abs=0.01)


def test_best_shift_finds_constant_offset():
    truth = {"A": [(5, 10), (20, 25)], "B": [(10, 20)]}
    hyp = {"S1": [(7, 12), (22, 27)], "S2": [(12, 22)]}
    assert scoring.best_shift(truth, hyp, 30.0, max_shift=5.0) == pytest.approx(-2.0, abs=0.11)


def test_turn_boundary_error():
    truth = {"A": [(0, 10)], "B": [(10, 20)], "C": [(20, 30)]}
    hyp = {"x": [(0, 11)], "y": [(11, 19)], "z": [(19, 30)]}
    r = scoring.boundary_error(truth, hyp)
    assert r["median_s"] == pytest.approx(1.0)


def test_spearman():
    assert scoring.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert scoring.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Licence filter (open web audio)
# ---------------------------------------------------------------------------

def test_licence_accepts_creative_commons():
    ok, kind, why = licence.decide({"license": "Creative Commons Attribution license (reuse allowed)",
                                    "channel": "Some Town TV", "uploader_id": "@sometown"})
    assert ok and kind.startswith("CC")


def test_licence_accepts_official_us_federal_channel():
    ok, kind, why = licence.decide({"license": None, "channel": "House Committee on Oversight and Accountability",
                                    "channel_url": "https://www.youtube.com/@OversightCommittee",
                                    "uploader_id": "@OversightCommittee",
                                    "channel_description": "Official channel. https://oversight.house.gov"})
    assert ok and kind == "US federal government work"
    # same name without a federal .gov link on the channel: not proven, so skipped
    ok, _, why = licence.decide({"license": None, "channel": "House Committee on Oversight and Accountability",
                                 "uploader_id": "@OversightCommittee", "channel_description": "fan reuploads"})
    assert not ok


def test_licence_rejects_standard_youtube_and_lookalikes():
    ok, _, why = licence.decide({"license": None, "channel": "Fox News", "uploader_id": "@FoxNews"})
    assert not ok and "licence" in why
    # a news channel re-uploading a hearing is NOT a federal work
    ok, _, why = licence.decide({"license": None, "channel": "PBS NewsHour", "uploader_id": "@PBSNewsHour",
                                 "title": "Senate hearing gets heated",
                                 "channel_description": "see also senate.gov"})
    assert not ok
    # state/city government is not a federal work
    ok, _, _ = licence.decide({"license": None, "channel": "California State Senate", "uploader_id": "@CASenate",
                               "channel_description": "https://www.senate.ca.gov"})
    assert not ok
    # "Standard YouTube License" explicit
    ok, _, _ = licence.decide({"license": "Standard YouTube License", "channel": "x"})
    assert not ok
