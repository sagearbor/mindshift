"""Recording-replay pipeline, input side: the owner's notes file, the audio
LLM's annotation (lenient parse + schema check), and re-timing annotation
segments against Deepgram's word timings. Pure functions, no audio, no
network: always runs (CI included)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recreplay import annotation as ann
from recreplay import notes as notes_mod

SCHEMA = Path(__file__).parent / "fixtures" / "annotation" / "mindshift-annotation-v1.schema.json"


# ---------------------------------------------------------------------------
# notes.txt
# ---------------------------------------------------------------------------

def test_notes_three_lines_and_moments():
    text = (
        "who: I'm the man with the low voice; other voice is my son (12)\n"
        "setting: dinner at home, son talking about his school day and a quiz\n"
        "phone: on the table, no earpiece\n"
        "moments (optional, one per line, mm:ss — what a good coach would have said):\n"
        "03:10 I interrupted him — \"let him finish\"\n"
        "1:02:03 - nice save\n"
        "  0:07   praise him\n"
    )
    n = notes_mod.parse_notes(text)
    assert n.who.startswith("I'm the man with the low voice")
    assert n.self_clause == "the man with the low voice"
    assert "son (12)" in n.others_clause
    assert n.setting.startswith("dinner at home")
    assert n.phone == "on the table, no earpiece"
    assert n.earpiece is False
    assert [m.t for m in n.moments] == [190.0, 3723.0, 7.0]
    assert n.moments[0].text == 'I interrupted him — "let him finish"'
    assert n.moments[1].text == "nice save"
    assert n.problems == []


def test_notes_lenient_keys_and_flags():
    text = (
        "- WHO = I'm S2\n"
        "Setting - car ride\n"
        "phone: in my pocket with earbuds\n"
        "relationship: child\n"
        "mode: room\n"
        "library: lib-1, lib-2\n"
        "random chatter line\n"
    )
    n = notes_mod.parse_notes(text)
    assert n.self_speaker_id == "S2"
    assert n.setting == "car ride"
    assert n.earpiece is True
    assert n.relationship == "child"
    assert n.mode == "room"
    assert n.library_item_ids == ["lib-1", "lib-2"]
    # an unparseable line is reported, never fatal
    assert any("random chatter" in p for p in n.problems)


def test_notes_missing_everything_is_reported_not_raised():
    n = notes_mod.parse_notes("")
    assert n.who is None and n.setting is None
    assert any("who" in p for p in n.problems)


def test_self_descriptor_keywords():
    n = notes_mod.parse_notes("who: I'm the man with the low voice; other voice is my son (12)")
    kw = notes_mod.descriptor_keywords(n.self_clause)
    assert {"male", "low", "adult"} <= kw
    other = notes_mod.descriptor_keywords(n.others_clause)
    assert "child" in other


# ---------------------------------------------------------------------------
# annotation: lenient load
# ---------------------------------------------------------------------------

GOOD = {
    "format": "mindshift-annotation/v1",
    "annotator": {"model": "gemini-2.5-pro", "notes": ""},
    "audio": {"duration_s": 20.0, "quality": "ok", "environment": "kitchen"},
    "speakers": [
        {"id": "S1", "voice_description": "adult male, low voice", "approx_age": "adult", "talk_share_pct": 60},
        {"id": "S2", "voice_description": "boy, high voice", "approx_age": "child", "talk_share_pct": 40},
    ],
    "segments": [
        {"start": 1.0, "end": 3.0, "speaker": "S1", "text": "how was school today",
         "vocal": {"emotion": "calm", "intensity": 0, "volume": "normal", "pace": "normal"},
         "text_emotion": "questioning", "confidence": 0.9},
        {"start": 3.5, "end": 6.0, "speaker": "S2", "text": "fine I guess we had a quiz",
         "vocal": {"emotion": "tired", "intensity": 1, "volume": "quiet", "pace": "slow"},
         "text_emotion": "neutral", "confidence": 0.8},
    ],
    "events": [{"t": 3.4, "type": "question", "speakers": ["S1"], "note": "opener"}],
    "coach_moments": [{"t": 6.2, "for_speaker": "S1", "what_happened": "short answer",
                       "ideal_nudge": "ask about the quiz", "kind": "encouragement", "priority": 1}],
    "summary": {"tone_arc": "flat", "overall_heat": 0, "peak_heat_t": None},
}


def test_load_clean_annotation_has_no_problems():
    a = ann.parse_annotation_text(json.dumps(GOOD))
    assert a.ok
    assert a.problems == []
    assert [s.id for s in a.speakers] == ["S1", "S2"]
    assert a.segments[1].vocal_emotion == "tired"
    assert a.coach_moments[0].ideal_nudge == "ask about the quiz"
    assert a.model == "gemini-2.5-pro"


def test_strips_fences_and_prose():
    raw = "Sure! Here is the JSON:\n```json\n" + json.dumps(GOOD) + "\n```\nHope that helps."
    a = ann.parse_annotation_text(raw)
    assert a.ok and len(a.segments) == 2
    assert any("fence" in p or "prose" in p for p in a.repairs)


def test_truncated_tail_is_repaired_and_reported():
    full = json.dumps(GOOD)
    cut = full[: full.index('"events"') + 30]   # cut mid-way through events
    a = ann.parse_annotation_text(cut)
    assert a.ok
    assert len(a.segments) == 2               # everything before the cut survives
    assert any("truncat" in r for r in a.repairs)


def test_part_files_concatenate():
    full = json.dumps(GOOD)
    k = len(full) // 2
    a = ann.parse_annotation_text(full[:k], continuations=[full[k:]])
    assert a.ok and a.problems == [] and len(a.coach_moments) == 1


def test_messy_values_coerced_and_reported():
    messy = json.loads(json.dumps(GOOD))
    messy["format"] = "mindshift-annotation/v2"
    messy["segments"][0]["start"] = "1.0"           # numeric string
    messy["segments"][1].pop("end")                 # missing key
    messy["segments"].append({"speaker": "S1"})     # no timing, no text -> dropped
    messy["coach_moments"][0]["t"] = "0:06.2"       # mm:ss string
    messy["speakers"][1]["approx_age"] = "kid"      # off-vocabulary
    a = ann.parse_annotation_text(json.dumps(messy))
    assert a.ok
    assert a.segments[0].start == 1.0
    assert a.segments[1].end is not None and a.segments[1].end > a.segments[1].start
    assert len(a.segments) == 2
    assert a.coach_moments[0].t == pytest.approx(6.2)
    text = " | ".join(a.problems)
    assert "format" in text and "end" in text and "dropped" in text and "approx_age" in text


def test_garbage_is_not_ok_but_does_not_raise():
    a = ann.parse_annotation_text("I'm sorry, I can't listen to audio.")
    assert not a.ok
    assert a.problems


def test_schema_file_validates_the_example():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text())
    jsonschema.validate(GOOD, schema)
    a = ann.parse_annotation_text(json.dumps(GOOD))
    assert ann.schema_problems(a.raw, schema) == []
    bad = json.loads(json.dumps(GOOD))
    bad["segments"][0]["vocal"]["emotion"] = "furious"
    assert ann.schema_problems(bad, schema)


def test_discover_annotation_files(tmp_path):
    (tmp_path / "x.annotation.json").write_text("{}")
    (tmp_path / "x.annotation.json.part2").write_text("")
    (tmp_path / "x.annotation.gemini.json").write_text("{}")
    (tmp_path / "x.notes.txt").write_text("")
    found = ann.discover(tmp_path, "x")
    assert [f.label for f in found] == ["default", "gemini"]
    assert found[0].continuations == [tmp_path / "x.annotation.json.part2"]


# ---------------------------------------------------------------------------
# alignment to Deepgram words
# ---------------------------------------------------------------------------

def _words(spec):
    """[(word, start, end, speaker)] -> Deepgram-ish word dicts."""
    return [{"word": w, "start": s, "end": e, "speaker": sp} for w, s, e, sp in spec]


def test_alignment_retimes_drifted_segments():
    words = _words([
        ("How", 2.0, 2.2, 0), ("was", 2.2, 2.4, 0), ("school", 2.4, 2.8, 0), ("today?", 2.8, 3.2, 0),
        ("Fine.", 4.6, 5.0, 1), ("I", 5.1, 5.2, 1), ("guess", 5.2, 5.5, 1), ("we", 5.6, 5.7, 1),
        ("had", 5.7, 5.9, 1), ("a", 5.9, 6.0, 1), ("quiz.", 6.0, 6.6, 1),
    ])
    a = ann.parse_annotation_text(json.dumps(GOOD))   # annotation is ~1 s early
    al = ann.align(a, words)
    s0, s1 = al.segments
    assert s0.start == pytest.approx(2.0) and s0.end == pytest.approx(3.2)
    assert s1.start == pytest.approx(4.6) and s1.end == pytest.approx(6.6)
    assert s0.match == pytest.approx(1.0) and s1.match == pytest.approx(1.0)
    assert al.quality["words_matched_pct"] == pytest.approx(100.0)
    assert al.quality["median_shift_s"] == pytest.approx(1.05, abs=0.1)
    # moments/events shift with the piecewise correction (segment-start
    # anchors; past the last one, that anchor's shift)
    assert al.coach_moments[0].t == pytest.approx(6.2 + 1.1, abs=0.05)
    assert al.events[0].t == pytest.approx(2.0 + (3.4 - 1.0) * (4.6 - 2.0) / (3.5 - 1.0), abs=0.05)
    # annotation speakers map onto Deepgram speakers by overlap
    assert al.speaker_map == {"S1": "Speaker A", "S2": "Speaker B"}


def test_alignment_unmatched_segment_keeps_shifted_time():
    words = _words([("How", 2.0, 2.2, 0), ("was", 2.2, 2.4, 0), ("school", 2.4, 2.8, 0), ("today?", 2.8, 3.2, 0)])
    a = ann.parse_annotation_text(json.dumps(GOOD))
    al = ann.align(a, words)
    s1 = al.segments[1]
    assert s1.match == 0.0
    assert s1.aligned is False
    assert s1.start == pytest.approx(3.5 + 1.0, abs=0.2)   # moved by the neighbours' shift
    assert al.quality["segments_aligned"] == 1


def test_schema_vocabularies_match_the_parser():
    """The JSON Schema and the lenient parser must agree on every closed
    vocabulary (no jsonschema dependency needed for this check)."""
    schema = json.loads(SCHEMA.read_text())
    seg = schema["properties"]["segments"]["items"]["properties"]
    assert set(seg["vocal"]["properties"]["emotion"]["enum"]) - {None} == set(ann.EMOTIONS)
    ev = schema["properties"]["events"]["items"]["properties"]["type"]["enum"]
    assert set(ev) == set(ann.EVENT_TYPES)
    ages = schema["properties"]["speakers"]["items"]["properties"]["approx_age"]["enum"]
    assert set(ann.AGES) <= set(ages)
    assert schema["properties"]["format"]["const"] == ann.FORMAT
