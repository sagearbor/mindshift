"""Corpus -> recording-replay inbox (scripts/corpus_to_inbox.py,
scripts/recreplay/corpora.py): human corpus transcripts become
``mindshift-annotation/v1`` ground truth, synthesized notes and a held-out
voiceprint. Pure functions on tiny inline samples: no audio, no network, no
corpus on disk, so this always runs (CI included)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from recreplay import annotation as ann
from recreplay import corpora
from recreplay import notes as notes_mod

SCHEMA = Path(__file__).parent / "fixtures" / "annotation" / "mindshift-annotation-v1.schema.json"

B = "\x15"

CHA = f"""@UTF8
@Begin
@Participants:\tJENN JENNIFER Speaker, LISB LISBETH Speaker, MANY MANY Speaker, ENV ENV Environment
@Comment:\tGuilt
@Comment:\tA lively family argument recorded at a vacation home.
\tDiscussion centers around a disagreement.
*LISB:\tYou never ⌈ listen to me ⌉ . {B}1000_3000{B}
*JENN:\t⌊ &{{l=F I do listen &}}l=F ⌋ . {B}2000_3500{B}
*JENN:\tI just think &=laugh it's unfair {B}3600_5000{B}
\tand you know it . {B}5000_6200{B}
*MANY:\t&=LAUGHTER . {B}6200_7000{B}
*LEAN:\t⌈ It's a li⌉⌈2ttle bit like s⌉2⌈3:in ⌉3 . {B}6500_6900{B}
*LISB:\tMhm . {B}7100_7400{B}
*JENN:\t&{{l=YELL Stop it &}}l=YELL XXX . {B}8000_9000{B}
@End
"""


# ---------------------------------------------------------------------------
# SBCSAE (CHAT)
# ---------------------------------------------------------------------------

def test_parse_cha_turns_text_and_marks():
    s = corpora.parse_cha_text(CHA, sid="SBC033")
    assert s.title == "Guilt"
    assert "family argument" in s.description
    sp = [t.speaker for t in s.turns]
    assert sp[:2] == ["LISB", "JENN"]
    t0, t1 = s.turns[0], s.turns[1]
    assert (t0.start, t0.end) == (1.0, 3.0)
    assert t0.text == "You never listen to me."
    assert t1.text == "I do listen."
    assert t1.loud == "raised"
    # continuation lines of one speaker merge into one turn; laughter -> event, not text
    t2 = s.turns[2]
    assert t2.speaker == "JENN" and (t2.start, t2.end) == (3.6, 6.2)
    assert "laugh" not in t2.text and t2.text.startswith("I just think it's unfair")
    assert any(e["type"] == "laughter" and e["speakers"] == ["JENN"] and e["t"] == pytest.approx(3.6) for e in s.events)
    # MANY laughter-only line is an event, not a turn
    assert not any(t.speaker == "MANY" for t in s.turns)
    assert any(e["type"] == "laughter" and e["speakers"] == ["MANY"] for e in s.events)
    assert any(t.text == "It's a little bit like sin." for t in s.turns)
    yell = s.turns[-1]
    assert yell.loud == "shouting" and "[inaudible]" in yell.text
    assert corpora.is_backchannel_text("Mhm.") and not corpora.is_backchannel_text("You never listen")
    assert next(t for t in s.turns if t.text == "Mhm.").backchannel


# ---------------------------------------------------------------------------
# AMI (NXT words + dialogue acts)
# ---------------------------------------------------------------------------

WORDS = """<?xml version="1.0"?><nite:root xmlns:nite="http://nite.sourceforge.net/">
<w nite:id="M.A.words0" starttime="1.0" endtime="1.4">Hi</w>
<w nite:id="M.A.words1" starttime="1.4" endtime="1.4" punc="true">,</w>
<w nite:id="M.A.words2" starttime="1.5" endtime="2.0">I&#39;m</w>
<w nite:id="M.A.words3" starttime="2.0" endtime="2.5">David</w>
<vocalsound nite:id="M.A.words4" starttime="3.0" endtime="4.0" type="laugh"/>
<w nite:id="M.A.words5" starttime="5.0" endtime="5.3">Yeah</w>
</nite:root>"""
DACTS = """<?xml version="1.0"?><nite:root xmlns:nite="http://nite.sourceforge.net/">
<dact nite:id="d1"><nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_4)"/>
<nite:child href="M.A.words.xml#id(M.A.words0)..id(M.A.words3)"/></dact>
<dact nite:id="d2"><nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_1)"/>
<nite:child href="M.A.words.xml#id(M.A.words5)"/></dact>
</nite:root>"""


def test_parse_ami_words_and_dialogue_acts():
    turns, events = corpora.parse_ami_speaker(WORDS, DACTS, speaker="MEE006")
    assert [t.text for t in turns] == ["Hi, I'm David", "Yeah"]
    assert (turns[0].start, turns[0].end) == (1.0, 2.5)
    assert turns[1].backchannel is True and turns[0].backchannel is False
    assert events == [{"t": 3.0, "type": "laughter", "speakers": ["MEE006"], "note": "corpus: vocalsound laugh"}]


# ---------------------------------------------------------------------------
# CHiME-6 (JSON utterances)
# ---------------------------------------------------------------------------

def test_parse_chime_utterances():
    utts = [
        {"start_time": "00:00:40.60", "end_time": "00:00:43.82", "words": "[laughs] It's the blue, I think.", "speaker": "P05", "location": "kitchen"},
        {"start_time": "00:01:10.11", "end_time": "00:01:12.97", "words": "[noise] Okay. [inaudible]", "speaker": "P08", "location": "kitchen"},
    ]
    turns, events, locations = corpora.parse_chime(utts)
    assert turns[0].text == "It's the blue, I think." and turns[0].start == pytest.approx(40.6)
    assert turns[1].text == "Okay. [inaudible]"
    assert events[0]["type"] == "laughter" and events[0]["speakers"] == ["P05"]
    assert locations == {"kitchen": 2}


# ---------------------------------------------------------------------------
# Interaction features, wearer choice, window choice
# ---------------------------------------------------------------------------

def T(sp, a, b, text="words here", **kw):
    return corpora.Turn(speaker=sp, start=a, end=b, text=text, **kw)


def test_overlap_and_interruptions():
    turns = [T("A", 0, 10), T("B", 8, 12), T("A", 13, 14, "mm", backchannel=True), T("B", 13, 20), T("A", 15, 18)]
    f = corpora.window_features(turns, 0, 30)
    # A&B overlap 8-10 (2 s) and 15-18 (3 s); the backchannel does not count
    assert f["overlap_s"] == pytest.approx(5.0)
    assert f["interruptions"] == {"B": 1, "A": 1}
    assert f["talk_s"]["A"] == pytest.approx(13.0)


def test_choose_wearer_heated_prefers_conflict_over_talk():
    turns = [T("A", 0, 60), T("B", 10, 14), T("B", 20, 24, loud="raised"), T("B", 30, 34), T("C", 40, 41)]
    w, why = corpora.choose_wearer(turns, 0, 70, heated=True, exclude=set())
    assert w == "B" and "interrupt" in why
    w2, why2 = corpora.choose_wearer(turns, 0, 70, heated=False, exclude=set())
    assert w2 == "A" and "most talk" in why2


def test_pick_window_finds_the_hot_stretch():
    turns = []
    t = 0.0
    while t < 1200:            # calm alternation, no overlap
        turns.append(T("A" if int(t / 10) % 2 == 0 else "B", t, t + 9.0))
        t += 10
    for k in range(20):        # a hot stretch around 600-700 s: overlapping, loud
        turns.append(T("A", 600 + 5 * k, 600 + 5 * k + 4, loud="raised"))
    win = corpora.pick_window(turns, 1200, length=180, heated=True)
    assert win[0] <= 600 and win[1] >= 690
    calm = corpora.pick_window(turns, 1200, length=180, heated=False, avoid=[win])
    assert calm[1] <= win[0] or calm[0] >= win[1]


# ---------------------------------------------------------------------------
# Coach moments from ground truth only
# ---------------------------------------------------------------------------

def test_derive_moments_rules():
    segs = [
        {"speaker": "S1", "start": 0.0, "end": 40.0, "text": "long story", "is_backchannel": False, "loud": None},
        {"speaker": "S2", "start": 20.0, "end": 21.0, "text": "but", "is_backchannel": False, "loud": None},   # cut off
        {"speaker": "S2", "start": 50.0, "end": 60.0, "text": "my turn", "is_backchannel": False, "loud": None},
        {"speaker": "S1", "start": 52.0, "end": 58.0, "text": "no listen", "is_backchannel": False, "loud": None},  # barges in
        {"speaker": "S1", "start": 90.0, "end": 92.0, "text": "stop it", "is_backchannel": False, "loud": "shouting"},
    ]
    ms = corpora.derive_moments(segs, wearer="S1")
    kinds = {m["rule"] for m in ms}
    assert {"talks-over", "raised-voice", "monologue-cuts-off"} <= kinds
    for m in ms:
        assert m["for_speaker"] == "S1" and m["kind"] == "warning" and len(m["ideal_nudge"].split()) <= 10
    assert next(m for m in ms if m["rule"] == "talks-over")["t"] == pytest.approx(52.0)
    # nothing for the speaker who was talked over
    assert not corpora.derive_moments(segs, wearer="S2", rules=("talks-over",))


def test_confer_moments_from_rated_spans():
    series = [100.0] * 10 + [450.0] * 8 + [200.0] * 5 + [700.0] * 6
    ms = corpora.confer_moments(series)
    assert [m["t"] for m in ms] == [10.0, 23.0]
    assert ms[0]["for_speaker"] is None and ms[1]["priority"] == 1
    assert corpora.heat_level(700) == 3 and corpora.heat_level(100) == 0


# ---------------------------------------------------------------------------
# The written annotation + notes
# ---------------------------------------------------------------------------

def _session():
    turns = [T("LISB", 100, 103, "You never listen to me."), T("JENN", 102, 104, "I do listen.", loud="raised"),
             T("JENN", 105, 108, "It's unfair."), T("LISB", 108.5, 109, "Mhm.", backchannel=True),
             T("ENV", 109, 110, "door")]
    return corpora.Session(corpus="SBCSAE", sid="SBC033", turns=turns,
                           events=[{"t": 106.0, "type": "laughter", "speakers": ["JENN"], "note": "corpus"}],
                           duration_s=400.0, setting="Guilt: a lively family argument",
                           speakers={"JENN": "corpus speaker JENN", "LISB": "corpus speaker LISB"},
                           licence="CC BY-ND 3.0", phone="corpus room recording", exclude={"ENV"})


def test_build_annotation_is_valid_v1_and_window_relative():
    s = _session()
    obj, idmap = corpora.build_annotation(s, (100.0, 110.0), wearer="JENN", group="heated", derivation="rules here")
    a = ann.parse_annotation_obj(obj)
    assert a.ok, a.problems
    assert obj["retime"] is False
    assert obj["annotator"]["model"].startswith("corpus-ground-truth")
    assert "rules here" in obj["annotator"]["notes"]
    assert idmap["LISB"] == "S1" and idmap["JENN"] == "S2"
    seg = a.segments[1]
    assert seg.speaker == "S2" and seg.start == pytest.approx(2.0) and seg.overlaps_with == ["S1"]
    assert seg.volume == "raised" and seg.vocal_emotion is None
    assert any(sg.is_backchannel for sg in a.segments)
    assert not any("door" in sg.text for sg in a.segments)          # environment is not a speaker
    assert a.events and a.events[0].type == "laughter" and a.events[0].t == pytest.approx(6.0)
    assert {sp.id for sp in a.speakers} == {"S1", "S2"}
    if SCHEMA.exists():
        assert ann.schema_problems(obj, json.loads(SCHEMA.read_text())) == []


def test_notes_name_the_wearer_and_parse_clean():
    s = _session()
    _, idmap = corpora.build_annotation(s, (100.0, 110.0), wearer="JENN", group="heated", derivation="")
    text = corpora.build_notes(s, (100.0, 110.0), wearer="JENN", idmap=idmap, why="most conflict", group="heated")
    assert "source: corpus ground truth" in text
    n = notes_mod.parse_notes(text)
    assert n.problems == []
    assert n.self_speaker_id == "S2"
    assert "JENN" in n.who
    assert n.setting.startswith("Guilt")


def test_heldout_enrollment_turns_never_touch_the_window():
    turns = [T("A", 10, 20), T("A", 150, 160), T("B", 155, 158), T("A", 300, 310), T("A", 500, 505)]
    held = corpora.heldout_turns(turns, "A", (140.0, 320.0), guard_s=30.0)
    assert [(t.start, t.end) for t in held] == [(10, 20), (500, 505)]
    # overlapped stretches are excluded (another voice would pollute the print)
    turns2 = [T("A", 0, 10), T("B", 5, 7)]
    solo = corpora.heldout_turns(turns2, "A", (100.0, 200.0))
    assert [(round(t.start, 2), round(t.end, 2)) for t in solo] == [(0.0, 5.0), (7.0, 10.0)]


def test_voiceprint_document_shape():
    emb = np.ones(192, dtype=np.float32)
    doc = corpora.voiceprint_document(emb, recording_id="corpus:SBCSAE:SBC033", speaker="JENN", seconds=42.0,
                                      window=(100.0, 400.0))
    assert len(doc["embedding"]) == 192 and doc["is_self"] is True
    assert doc["samples"][0]["recording_id"] == "corpus:SBCSAE:SBC033"
    assert "held-out" in doc["samples"][0]["note"]
