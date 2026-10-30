"""Quiet coach: the pre-LLM trigger gate (server/coach_trigger.py) wired into
the live WS pipeline. An untriggered turn never reaches the LLM; a turn with
heated words, a direct question, a phone-reported floor cue or a library
phrase does; MINDSHIFT_TRIGGER_GATE=0 restores one call per turn."""

import json

import pytest

from main import app
from test_audio_pipeline import (  # noqa: F401 - local_first_env is a fixture
    LOCAL_SID,
    StoppableTranscriber,
    _inject,
    _turn_local,
    local_first_env,
    open_ws,
)
from test_coach_gate_pipeline import CountingLLM, _finals


class BothLLM(CountingLLM):
    """Answers a suggestion prompt AND a self-turn nudge prompt."""

    def __init__(self):
        super().__init__(["Take a breath first."], importance=90)
        self.response = json.dumps({"suggestions": ["Take a breath first."], "nudge": "Check in with them.",
                                    "importance": 90})

CALM = "We could look at the budget for next quarter on Monday."
HEATED = "Shut up, this is all your fault and you know it."
QUESTION = "What do you think we should do about the car?"


@pytest.fixture
def trigger_on(monkeypatch, local_first_env):  # noqa: F811
    monkeypatch.setenv("MINDSHIFT_TRIGGER_GATE", "1")
    monkeypatch.setenv("MINDSHIFT_SPEAK_GATE", "0")
    yield


def _run(turns, n_finals):
    llm = BothLLM()
    client = _inject(StoppableTranscriber())
    app.state.llm_client = llm
    with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
        for t in turns:
            ws.send_text(json.dumps(t))
        finals = _finals(ws, n_finals)
    return llm, finals


def test_untriggered_turn_never_reaches_the_llm(trigger_on):
    llm, [ev] = _run([
        _turn_local(text=CALM, start_time=0.0, end_time=3.0, is_self=False),
        _turn_local(text=HEATED, start_time=4.0, end_time=6.0, is_self=False),
    ], 1)
    assert ev["utterance_text"] == HEATED
    assert len(llm.users) == 1  # the calm turn cost nothing


def test_direct_question_from_the_other_person_triggers(trigger_on):
    llm, [ev] = _run([
        _turn_local(text=CALM, start_time=0.0, end_time=3.0, is_self=False),
        _turn_local(text=QUESTION, start_time=4.0, end_time=6.0, is_self=False),
    ], 1)
    assert ev["utterance_text"] == QUESTION
    assert len(llm.users) == 1


def test_phone_floor_trigger_is_honoured(trigger_on):
    llm, [ev] = _run([
        _turn_local(text=CALM, start_time=0.0, end_time=3.0, is_self=True),
        _turn_local(text=CALM + " Again.", start_time=4.0, end_time=7.0, is_self=True, triggers=["floor"]),
    ], 1)
    assert ev["utterance_text"] == CALM + " Again."
    assert len(llm.users) == 1


def test_server_floor_clock_triggers_on_a_long_wearer_run(trigger_on):
    # 0-50 s of the wearer, in two turns: the second ends at 50 s of floor.
    llm, [ev] = _run([
        _turn_local(text=CALM, start_time=0.0, end_time=25.0, is_self=True),
        _turn_local(text=CALM + " Then.", start_time=25.5, end_time=50.0, is_self=True),
    ], 1)
    assert ev["utterance_text"] == CALM + " Then."
    assert len(llm.users) == 1


def test_gate_off_coaches_every_turn(trigger_on, monkeypatch):
    monkeypatch.setenv("MINDSHIFT_TRIGGER_GATE", "0")
    llm, [ev] = _run([_turn_local(text=CALM, start_time=0.0, end_time=3.0, is_self=False)], 1)
    assert ev["utterance_text"] == CALM
    assert len(llm.users) == 1


def test_turn_local_triggers_field_validates():
    from models.audio import TurnLocalEvent

    ev = TurnLocalEvent(**_turn_local(triggers=["floor", "heat"]))
    assert ev.triggers == ["floor", "heat"]
    assert TurnLocalEvent(**_turn_local()).triggers is None
    with pytest.raises(ValueError):
        TurnLocalEvent(**_turn_local(triggers=["nonsense"]))
