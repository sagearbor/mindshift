"""coach_gate wired into the live WS pipeline (turn_local path): label
scrub on the wire, fragment skip before the LLM, and the speak gate
(shown-not-spoken, min gap, unknown-wearer cap)."""

import json

import pytest

from main import app
from test_audio_pipeline import (  # noqa: F401 — local_first_env is a fixture
    LOCAL_SID,
    StoppableTranscriber,
    _inject,
    _turn_local,
    local_first_env,
    open_ws,
)


class CountingLLM:
    """``complete()``-only double: one fixed JSON answer, every call's user
    prompt recorded."""

    def __init__(self, suggestions, importance=90):
        self.response = json.dumps({"suggestions": suggestions, "importance": importance})
        self.users: list[str] = []

    def complete(self, system: str, user: str, **_) -> str:
        self.users.append(user)
        return self.response


@pytest.fixture
def gates_on(monkeypatch, local_first_env):  # noqa: F811
    monkeypatch.setenv("MINDSHIFT_TURN_SKIP", "1")
    monkeypatch.setenv("MINDSHIFT_SPEAK_GATE", "1")
    monkeypatch.setenv("MINDSHIFT_SPEAK_MIN_IMPORTANCE", "75")
    monkeypatch.setenv("MINDSHIFT_SPEAK_MIN_GAP_S", "30")
    monkeypatch.setenv("MINDSHIFT_SPEAK_UNKNOWN_CAP", "70")
    yield


def _finals(ws, n):
    out = []
    while len(out) < n:
        msg = json.loads(ws.receive_text())
        if msg.get("type") == "suggestion" and not msg.get("partial"):
            out.append(msg)
    return out


LONG = "You never listen to anything I say to you."


def test_label_never_reaches_the_wire(gates_on):
    llm = CountingLLM(["Let Speaker E finish.", "Ask Speaker B why.", "Breathe."])
    client = _inject(StoppableTranscriber())
    app.state.llm_client = llm
    with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
        ws.send_text(json.dumps(_turn_local(text=LONG, is_self=False)))
        [ev] = _finals(ws, 1)
    assert ev["suggestions"] == ["Let them finish.", "Ask them why.", "Breathe."]
    assert "Speaker" not in json.dumps(ev["suggestions"])


def test_fragment_turn_is_not_sent_to_the_llm(gates_on):
    llm = CountingLLM(["Say what you heard."])
    client = _inject(StoppableTranscriber())
    app.state.llm_client = llm
    with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
        ws.send_text(json.dumps(_turn_local(text="Yeah.", start_time=0.0, end_time=0.5, is_self=False)))
        ws.send_text(json.dumps(_turn_local(text=LONG, start_time=1.0, end_time=3.0, is_self=False)))
        [ev] = _finals(ws, 1)
    assert ev["utterance_text"] == LONG
    assert len(llm.users) == 1 and "Yeah." not in llm.users[0].split("\n")[0]


def test_speak_gate_min_gap_shows_but_does_not_speak(gates_on):
    llm = CountingLLM(["Say what you heard."], importance=90)
    client = _inject(StoppableTranscriber())
    app.state.llm_client = llm
    with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
        ws.send_text(json.dumps(_turn_local(text=LONG, start_time=0.0, end_time=2.0, is_self=False)))
        first = _finals(ws, 1)[0]
        ws.send_text(json.dumps(_turn_local(text=LONG + " Ever.", start_time=4.0, end_time=6.0, is_self=False)))
        second = _finals(ws, 1)[0]
        ws.send_text(json.dumps(_turn_local(text=LONG + " Never.", start_time=40.0, end_time=42.0, is_self=False)))
        third = _finals(ws, 1)[0]
    assert first["speak"] is True
    assert second["speak"] is False and second["suggestions"] == ["Say what you heard."]  # shown, silent
    assert second["audio_b64"] is None
    assert third["speak"] is True


def test_speak_gate_importance_and_unknown_cap(gates_on):
    client = _inject(StoppableTranscriber())
    app.state.llm_client = CountingLLM(["Slow down."], importance=60)
    with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
        ws.send_text(json.dumps(_turn_local(text=LONG, is_self=False)))
        low = _finals(ws, 1)[0]
    assert low["speak"] is False and low["importance"] == 60
    app.state.llm_client = CountingLLM(["Slow down."], importance=90)
    with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
        ws.send_text(json.dumps(_turn_local(text=LONG)))  # no is_self -> wearer UNKNOWN
        unk = _finals(ws, 1)[0]
    assert unk["speak"] is False  # capped at 70 < 75
    assert unk["importance"] == 90  # the model's score is reported unchanged


def test_gate_off_restores_interject_only(monkeypatch, gates_on):
    monkeypatch.setenv("MINDSHIFT_SPEAK_GATE", "0")
    client = _inject(StoppableTranscriber())
    app.state.llm_client = CountingLLM(["Slow down."], importance=10)
    with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
        ws.send_text(json.dumps(_turn_local(text=LONG)))
        ev = _finals(ws, 1)[0]
    assert ev["speak"] is True  # interject 0 -> spoken, the legacy behaviour
