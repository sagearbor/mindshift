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


def test_ledger_is_shared_by_concurrent_runs(tmp_path):
    # two runs open the ledger at the same time; neither may drop the other's calls
    a = core.Ledger(tmp_path / "spend.json", cap_usd=10.0)
    b = core.Ledger(tmp_path / "spend.json", cap_usd=10.0)
    a.record("m", "x", 1.0, {})
    b.record("m", "y", 2.0, {})
    a.record("m", "z", 3.0, {})
    assert a.total == pytest.approx(6.0)
    assert b.total == pytest.approx(6.0)
    assert len(core.Ledger(tmp_path / "spend.json").calls) == 3


def test_a_lower_cap_written_into_the_ledger_file_wins_and_survives_writes(tmp_path):
    p = tmp_path / "spend.json"
    led = core.Ledger(p, cap_usd=25.0)
    led.record("m", "a", 1.0, {})
    d = json.loads(p.read_text())
    d["cap_usd"] = 1.0           # the owner freezes spending at what is spent
    p.write_text(json.dumps(d))
    with pytest.raises(core.BudgetExceeded):
        led.check(0.0)
    led.record("m", "late", 0.1, {})   # an in-flight call still gets recorded...
    assert json.loads(p.read_text())["cap_usd"] == 1.0   # ...without raising the cap back


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


def test_timestamps_covering_a_sliver_of_the_audio_are_flagged():
    obj = {"speakers": [{"id": "S1"}],
           "segments": [{"start": i / 100, "end": i / 100 + 0.01, "speaker": "S1", "text": "w"} for i in range(50)]}
    fixes: list[str] = []
    out = core.sanitize(obj, model="m", duration_s=300.0, fixes=fixes)
    assert any("TIMING UNTRUSTWORTHY" in f for f in fixes)
    assert "TIMING UNTRUSTWORTHY" in (out["annotator"]["notes"] or "")


def test_clock_style_timestamps_are_repaired():
    txt = '{"segments": [{"start": 1:52.4, "end": 1:53.0}], "summary": {"peak_heat_t": 0:09.5}, "events": [{"t": 1:02:03}]}'
    obj = json.loads(core.repair_text(txt))
    assert obj["segments"][0]["start"] == pytest.approx(112.4)
    assert obj["segments"][0]["end"] == pytest.approx(113.0)
    assert obj["summary"]["peak_heat_t"] == pytest.approx(9.5)
    assert obj["events"][0]["t"] == pytest.approx(3723.0)


def test_join_parts_drops_the_half_object_when_the_continuation_restarts_it():
    p1 = '{"segments": [\n {"start": 1.0, "text": "a"},\n {"start": 2.0, "end": 2.5, "text": "b", "is_'
    p2 = '{"start": 2.0, "end": 2.5, "text": "b", "is_backchannel": false},\n {"start": 3.0, "text": "c"}]}'
    obj = json.loads(core.join_parts([p1, p2]))
    assert [s["text"] for s in obj["segments"]] == ["a", "b", "c"]
    # a well-behaved continuation (picks up mid-token) is concatenated as is
    q1 = '{"segments": [{"start": 1.0, "text": "a"}, {"start": 2.0, "te'
    q2 = 'xt": "b"}]}'
    assert json.loads(core.join_parts([q1, q2]))["segments"][1]["text"] == "b"


def test_offset_shifts_all_times():
    obj = {"segments": [_seg(1, 2, "S1")], "events": [{"t": 1.0, "type": "question"}],
           "coach_moments": [{"t": 2.0, "for_speaker": "S1"}], "summary": {"peak_heat_t": 3.0}}
    o = core.offset(obj, 10.0)
    assert o["segments"][0]["start"] == 11 and o["events"][0]["t"] == 11
    assert o["coach_moments"][0]["t"] == 12 and o["summary"]["peak_heat_t"] == 13


def test_runaway_reply_is_detected():
    from annotator.gemini import degenerate
    assert degenerate('{"speakers": [{"voice_description": "' + "she is very loud. " * 120)
    assert degenerate('{"segments": [' + '{"start": 1.0, "text": "yes"}, ' * 200)
    ok = json.dumps({"segments": [{"start": i, "end": i + 1, "speaker": "S1", "text": f"words number {i} here"}
                                  for i in range(200)]})
    assert not degenerate(ok)


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


def test_best_affine_undoes_drift():
    truth = {"A": [(0, 10), (40, 50)], "B": [(10, 40), (50, 60)]}
    # the annotator's clock runs 25% fast
    hyp = {"x": [(0, 12.5), (50, 62.5)], "y": [(12.5, 50), (62.5, 75)]}
    scale, shift = scoring.best_affine(truth, hyp, 60.0)
    assert scale == pytest.approx(0.8, abs=0.021)
    assert abs(shift) <= 0.5
    warped = scoring.warp(hyp, scale, shift)
    assert scoring.der(truth, warped, 60.0)["der"] < 0.05


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
    # a musician who named his project "Oversight Committee" (real channel @oversightcommittee)
    ok, _, _ = licence.decide({"license": None, "channel": "Cj Hoeflich", "uploader_id": "@oversightcommittee",
                               "channel_description": "Oversight Committee is a new Mtl based musical project"})
    assert not ok


def test_cc_label_on_an_obvious_reupload_is_skipped():
    cc = "Creative Commons Attribution license (reuse allowed)"
    for meta in (
        {"license": cc, "channel": "Hope In Christ", "title": "Andrew Wilson DESTROYS Feminists in Heated Debate"},
        {"license": cc, "channel": "Bodycam Footage", "title": "Jealous Sister Sparks Chaos at Walmart"},
        {"license": cc, "channel": "Real Police Stories", "title": "How a Parking Lot Argument Triggered an Arrest"},
        {"license": cc, "channel": "Dronetek", "title": "LOL: Outraged Parents Melt Down at School Meeting"},
    ):
        ok, _, why = licence.decide(meta)
        assert not ok, meta
        assert "re-upload" in why
    ok, _, _ = licence.decide({"license": cc, "channel": "The Space Coast Rocket",
                               "title": "School Board Meeting gets heated between Trent and Jenkins"})
    assert ok


# ---------------------------------------------------------------------------
# Open-audio screening helpers
# ---------------------------------------------------------------------------

def test_hot_window_finds_the_loud_stretch():
    import numpy as np

    import fetch_open_audio as foa
    rng = np.random.default_rng(0)
    db = np.full(1200, -70.0)
    db[100:1100] = -30 + rng.normal(0, 1.5, 1000)      # ordinary speech
    db[700:900] = -18 + rng.normal(0, 1.5, 200)        # raised voices
    w = foa.hot_windows(db, 300, k=1)[0]
    assert w["start"] <= 700 and w["end"] >= 900
    assert w["raised_frac"] > 0.5


def test_measured_heat_needs_arousal_or_raised_voices():
    import fetch_open_audio as foa
    assert foa.measured_heated({"arousal_mean": 0.8, "raised_vs_own_baseline_pct": 0})[0]
    assert foa.measured_heated({"arousal_mean": 0.3, "raised_vs_own_baseline_pct": 15})[0]
    ok, why = foa.measured_heated({"arousal_mean": 0.34, "raised_vs_own_baseline_pct": 2})
    assert not ok and "NOT heated" in why
    assert not foa.measured_heated({})[0]


def test_overlap_and_wearer():
    import fetch_open_audio as foa
    ann = {"audio": {"duration_s": 60.0},
           "speakers": [{"id": "S1", "voice_description": "man"}, {"id": "S2", "voice_description": "woman"},
                        {"id": "S3", "voice_description": "chair"}],
           "segments": [{"start": 0, "end": 20, "speaker": "S1"}, {"start": 15, "end": 40, "speaker": "S2"},
                        {"start": 38, "end": 40, "speaker": "S1"}, {"start": 50, "end": 52, "speaker": "S3"}],
           "events": [{"t": 15, "type": "interruption", "speakers": ["S2"]}]}
    ov = foa.overlap_stats(ann)
    assert ov["overlap_pct"] == pytest.approx(100 * 7 / 42, abs=0.2)
    assert ov["speakers_active"] == 2
    spk, desc = foa.wearer(ann)
    assert spk == "S2" and desc == "woman"
