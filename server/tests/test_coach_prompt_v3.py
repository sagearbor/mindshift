"""Quiet coach prompt v3 (server/coach_prompt_v3.py, MINDSHIFT_COACH_PROMPT_V3).

Pins: the flag is OFF by default and then nothing changes; the exact prompt
bytes for each line type; the SILENT path; the unknown-wearer call is skipped;
and the post-filter on the known bad lines ("Let Speaker C finish", "Pete").
"""

from __future__ import annotations

import hashlib
import json

import pytest

import coach_prompt_v3 as v3
from main import app
from test_audio_pipeline import (  # noqa: F401 - local_first_env is a fixture
    LOCAL_SID,
    StoppableTranscriber,
    _inject,
    _turn_local,
    local_first_env,
    open_ws,
)
from test_coach_gate_pipeline import _finals

OTHER_HEATED = "You never listen, this is all your fault."
QUESTION = "What do you think we should do about the car?"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(v3.FLAG, raising=False)


def test_flag_is_off_by_default_and_read_at_call_time(monkeypatch):
    assert not v3.enabled()
    monkeypatch.setenv(v3.FLAG, "1")
    assert v3.enabled()
    monkeypatch.setenv(v3.FLAG, "0")
    assert not v3.enabled()


@pytest.mark.parametrize("triggers,identity,expected", [
    (("heat",), "self", "heat"),
    (("heat",), "other", "read_other"),
    (("floor",), "self", "floor"),
    (("question",), "other", "question"),
    (("library",), "other", "question"),
    (("heat", "floor"), "self", "heat"),          # precedence heat > floor
    (("floor",), "other", None),                   # a floor needs the wearer
    (("heat",), "unknown", None),                  # unknown wearer: no call at all
    ((), "self", None),
])
def test_line_type_per_trigger_and_identity(triggers, identity, expected):
    assert v3.line_type(triggers, identity) == expected


# The exact system prompt per line type (default slider 50, no hints). A change
# to any of these is a prompt change and must be a deliberate, re-measured one.
PROMPT_SHA = {
    "heat": "e9dd51d30f37d797",
    "floor": "f8a8c2ded599eb88",
    "question": "8080c9426d2db6c4",
    "read_other": "08791f5f081b3d52",
}


@pytest.mark.parametrize("ltype", v3.LINE_TYPES)
def test_system_prompt_bytes_are_pinned(ltype):
    sys_prompt = v3.system_prompt(ltype, 50)
    assert hashlib.sha256(sys_prompt.encode()).hexdigest()[:16] == PROMPT_SHA[ltype]
    # one line type per prompt: the other types' instructions are absent
    for other in v3.LINE_TYPES:
        if other != ltype:
            assert f"Line type {other.upper()}" not in sys_prompt
    assert "SILENT is the default" in sys_prompt
    assert '"line"' in sys_prompt and '"suggestions"' not in sys_prompt


def test_trigger_block_names_the_trigger_and_evidence():
    block = v3.trigger_block(("heat", "question"), {"heat": "all your fault", "question": "What now?"})
    assert block == ("Why the coach was called: heated words (all your fault); "
                     "a direct question to the wearer (What now?).")


@pytest.mark.parametrize("data,expected", [
    ({"line": "name the deadline", "type": "heat", "importance": 88}, ("name the deadline", "heat", 88)),
    ({"line": "", "type": "silent", "importance": 0}, ("", "silent", 0)),
    ({"line": "x y", "type": "silent", "importance": 90}, ("", "silent", 0)),
    ({"line": "ask about the budget", "type": "nonsense", "importance": 90}, ("", "silent", 0)),
    ({"line": "ask about the budget", "type": "floor"}, ("ask about the budget", "floor", 100)),
    ("not a dict", ("", "silent", 0)),
])
def test_parse(data, expected):
    assert v3.parse(data) == expected


CONTEXT = "I told Angie the budget is late. Angie said the deadline moved."


@pytest.mark.parametrize("line,ltype,reason", [
    ("Let Speaker C finish", "heat", "label"),
    ("Tell the wearer to breathe", "heat", "label"),
    ("Pete", "heat", "unseen_name"),
    ("Apologise to Pete about the budget", "heat", "unseen_name"),
    ("Great job staying calm", "heat", "praise"),
    ("Ask an open-ended question", "question", "engage"),
    ("Pause and let them finish", "heat", "listen_no_overlap"),
    ("One two three four five six seven eight nine ten eleven twelve thirteen", "heat", "too_long"),
    ("", "heat", "empty"),
])
def test_post_filter_drops_known_bad_lines(line, ltype, reason):
    assert v3.post_filter(line, ltype, CONTEXT) == ("", reason)


@pytest.mark.parametrize("line,ltype,overlap", [
    ("Name the deadline, drop the always", "heat", False),
    ("Ask Angie about the budget", "heat", False),
    ("Stop and let them answer on the deadline", "floor", False),   # a floor line is a listen cue by design
    ("Let them finish", "heat", True),                               # the wearer did talk over someone
    ("They sound guarded; ask what would help", "read_other", False),
    ("Take a breath. Let them speak.", "floor", False),               # 2nd sentence opener is not a name
    ("Pause. Ask what specifically scares them about her.", "read_other", True),
])
def test_post_filter_keeps_good_lines(line, ltype, overlap):
    assert v3.post_filter(line, ltype, CONTEXT, wearer_overlapped=overlap) == (line, None)


def test_wearer_overlap_needs_another_voice_over_the_turn():
    hist = {"turns": [
        {"who": "Other", "start": 0.0, "t": 5.0, "text": "x"},
        {"who": "You", "start": 5.0, "t": 6.0, "text": "y"},
    ]}
    assert v3.wearer_overlapped(hist, 4.0, 9.0)           # other until 5.0 > start 4.0 by 1 s
    assert not v3.wearer_overlapped(hist, 5.2, 9.0)
    assert not v3.wearer_overlapped(None, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Pipeline wiring (real WS path, scripted LLM double)
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Returns the next scripted JSON answer; records every (system, user)."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, **_) -> str:
        self.calls.append((system, user))
        ans = self.answers.pop(0) if self.answers else {"line": "", "type": "silent", "importance": 0}
        return json.dumps(ans)


@pytest.fixture
def quiet(monkeypatch, local_first_env):  # noqa: F811
    monkeypatch.setenv("MINDSHIFT_TRIGGER_GATE", "1")
    monkeypatch.setenv("MINDSHIFT_SPEAK_GATE", "0")
    yield


def _run(turns, llm, n_finals):
    client = _inject(StoppableTranscriber())
    app.state.llm_client = llm
    with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
        for t in turns:
            ws.send_text(json.dumps(t))
        return _finals(ws, n_finals)


def test_flag_off_keeps_the_three_cue_prompt(quiet):
    llm = ScriptedLLM([{"suggestions": ["a b", "c d", "e f"], "importance": 90}])
    [ev] = _run([_turn_local(text=OTHER_HEATED, start_time=0.0, end_time=3.0, is_self=False)], llm, 1)
    system, user = llm.calls[0]
    assert "exactly 3 short cues" in system and '"line"' not in system
    assert "Why the coach was called" not in user
    assert ev["suggestions"] == ["a b", "c d", "e f"]


def test_flag_on_sends_the_trigger_and_one_line(quiet, monkeypatch):
    monkeypatch.setenv(v3.FLAG, "1")
    llm = ScriptedLLM([{"line": "they sound guarded; ask what would help", "type": "read_other", "importance": 90}])
    [ev] = _run([_turn_local(text=OTHER_HEATED, start_time=0.0, end_time=3.0, is_self=False)], llm, 1)
    system, user = llm.calls[0]
    assert "Line type READ_OTHER" in system
    assert "Why the coach was called: heated words" in user
    assert ev["suggestions"] == ["they sound guarded; ask what would help"]


def test_silent_answer_sends_no_event(quiet, monkeypatch):
    monkeypatch.setenv(v3.FLAG, "1")
    llm = ScriptedLLM([
        {"line": "", "type": "silent", "importance": 0},
        {"line": "answer the car question first", "type": "question", "importance": 88},
    ])
    [ev] = _run([
        _turn_local(text=OTHER_HEATED, start_time=0.0, end_time=3.0, is_self=False),
        _turn_local(text=QUESTION, start_time=4.0, end_time=6.0, is_self=False),
    ], llm, 1)
    assert len(llm.calls) == 2
    assert ev["utterance_text"] == QUESTION


def test_post_filter_silences_an_invented_name(quiet, monkeypatch):
    monkeypatch.setenv(v3.FLAG, "1")
    llm = ScriptedLLM([
        {"line": "Pete", "type": "read_other", "importance": 95},
        {"line": "answer the car question first", "type": "question", "importance": 88},
    ])
    [ev] = _run([
        _turn_local(text=OTHER_HEATED, start_time=0.0, end_time=3.0, is_self=False),
        _turn_local(text=QUESTION, start_time=4.0, end_time=6.0, is_self=False),
    ], llm, 1)
    assert ev["utterance_text"] == QUESTION


def test_unknown_wearer_turn_is_not_sent_to_the_llm_under_v3(quiet, monkeypatch):
    # no is_self verdict and nobody confirmed -> identity "unknown"
    monkeypatch.setenv(v3.FLAG, "1")
    llm = ScriptedLLM([{"line": "answer the car question first", "type": "question", "importance": 88}])
    [ev] = _run([
        _turn_local(text=OTHER_HEATED, start_time=0.0, end_time=3.0, is_self=None),
        _turn_local(text=QUESTION, start_time=4.0, end_time=6.0, is_self=False),
    ], llm, 1)
    assert ev["utterance_text"] == QUESTION
    assert len(llm.calls) == 1            # only the question turn cost a call
