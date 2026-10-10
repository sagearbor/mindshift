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


# --- speak verifier + unknown-wearer neutral cues ---------------------------------

def _client_with(llm):
    """_inject() installs its own LLM double; put ours back after it."""
    client = _inject(StoppableTranscriber())
    app.state.llm_client = llm
    return client


class RoutingLLM(CountingLLM):
    """Coach answers like CountingLLM; the speak verifier's calls (its
    system prompt is one of speak_verifier.VARIANTS) get ``verdict``."""

    def __init__(self, suggestions, importance=90, verdict='{"speak": false, "reason": "calm"}'):
        super().__init__(suggestions, importance)
        self.verdict = verdict
        self.verifier_users: list[str] = []

    def complete(self, system: str, user: str, **kw) -> str:
        import speak_verifier
        if system in speak_verifier.VARIANTS.values():
            self.verifier_users.append(user)
            return self.verdict
        return super().complete(system, user, **kw)


def test_verifier_off_by_default_is_never_called(gates_on):
    llm = RoutingLLM(["Say what you heard."])
    app.state.llm_client = llm
    with open_ws(_client_with(app.state.llm_client), f"/ws/session/{LOCAL_SID}") as ws:
        ws.send_text(json.dumps(_turn_local(text=LONG, is_self=False)))
        [ev] = _finals(ws, 1)
    assert ev["speak"] is True and llm.verifier_users == []


def test_verifier_no_silences_but_still_shows(monkeypatch, gates_on):
    monkeypatch.setenv("MINDSHIFT_SPEAK_VERIFIER", "1")
    llm = RoutingLLM(["Say what you heard."])
    app.state.llm_client = llm
    with open_ws(_client_with(app.state.llm_client), f"/ws/session/{LOCAL_SID}") as ws:
        ws.send_text(json.dumps(_turn_local(text=LONG, is_self=False)))
        [ev] = _finals(ws, 1)
    assert ev["speak"] is False and ev["suggestions"] == ["Say what you heard."]
    assert len(llm.verifier_users) == 1
    assert "Say what you heard." in llm.verifier_users[0] and LONG in llm.verifier_users[0]


def test_verifier_yes_speaks_and_runs_only_after_the_gate(monkeypatch, gates_on):
    monkeypatch.setenv("MINDSHIFT_SPEAK_VERIFIER", "1")
    llm = RoutingLLM(["Say what you heard."], verdict='{"speak": true, "reason": "real moment"}')
    app.state.llm_client = llm
    with open_ws(_client_with(app.state.llm_client), f"/ws/session/{LOCAL_SID}") as ws:
        ws.send_text(json.dumps(_turn_local(text=LONG, start_time=0.0, end_time=2.0, is_self=False)))
        first = _finals(ws, 1)[0]
        # inside the min gap: the gate already says no, so no verifier call
        ws.send_text(json.dumps(_turn_local(text=LONG + " Ever.", start_time=4.0, end_time=6.0, is_self=False)))
        second = _finals(ws, 1)[0]
    assert first["speak"] is True and second["speak"] is False
    assert len(llm.verifier_users) == 1


def test_verifier_garbage_fails_closed(monkeypatch, gates_on):
    monkeypatch.setenv("MINDSHIFT_SPEAK_VERIFIER", "1")
    app.state.llm_client = RoutingLLM(["Say what you heard."], verdict="I think yes!")
    with open_ws(_client_with(app.state.llm_client), f"/ws/session/{LOCAL_SID}") as ws:
        ws.send_text(json.dumps(_turn_local(text=LONG, is_self=False)))
        [ev] = _finals(ws, 1)
    assert ev["speak"] is False


HELD = "You are obliged to apply the law, you are obliged, and that is the whole point here."
CLAIM = "Can I answer? Of course, I'm listening to you."


@pytest.mark.parametrize("flag,want", [("1", True), ("0", False)])
def test_unknown_wearer_neutral_cue_after_floor_claim(monkeypatch, gates_on, flag, want):
    monkeypatch.setenv("MINDSHIFT_SPEAK_UNKNOWN_NEUTRAL", flag)
    app.state.llm_client = CountingLLM(["Pause. Let them finish."], importance=72)
    with open_ws(_client_with(app.state.llm_client), f"/ws/session/{LOCAL_SID}") as ws:
        # no is_self anywhere -> wearer UNKNOWN (capped at 70 < 75)
        ws.send_text(json.dumps(_turn_local(speaker="Speaker A", text=HELD, start_time=0.0, end_time=30.0)))
        _finals(ws, 1)
        ws.send_text(json.dumps(_turn_local(speaker="Speaker B", text=CLAIM, start_time=30.0, end_time=33.0)))
        ev = _finals(ws, 1)[0]
    assert ev["utterance_text"] == CLAIM
    assert ev["speak"] is want


def test_unknown_wearer_neutral_needs_evidence(monkeypatch, gates_on):
    monkeypatch.setenv("MINDSHIFT_SPEAK_UNKNOWN_NEUTRAL", "1")
    app.state.llm_client = CountingLLM(["Pause. Let them finish."], importance=72)
    with open_ws(_client_with(app.state.llm_client), f"/ws/session/{LOCAL_SID}") as ws:
        ws.send_text(json.dumps(_turn_local(speaker="Speaker A", text=LONG, start_time=0.0, end_time=3.0)))
        ev = _finals(ws, 1)[0]
    assert ev["speak"] is False
