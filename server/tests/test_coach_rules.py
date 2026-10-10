"""--coach rules: the no-LLM coach (server/coach_rules.py) — deterministic
triggers only, so a replay can run fully offline and at $0, and the fast
tier has a prototype. Floor-time rule from tmp/recordings/exp_FT (best live
cell: wearer floor >= 45 s, >= 30 s between fires); lexical triggers are a
hook (none ship by default)."""

import json

import pytest

import coach_rules
from main import app
from test_audio_pipeline import (  # noqa: F401 — local_first_env is a fixture
    LOCAL_SID,
    StoppableTranscriber,
    _inject,
    _turn_local,
    local_first_env,
    open_ws,
)


def _coach(**kw):
    return coach_rules.RulesCoach(**kw)


# --- floor rule ---------------------------------------------------------------

def test_wearer_monologue_fires_once_floor_reaches_45s():
    c = _coach()
    assert c.observe_turn("a", 0.0, 20.0, is_self=True) is None
    assert c.observe_turn("b", 21.0, 40.0, is_self=True) is None
    fire = c.observe_turn("c", 41.0, 50.0, is_self=True)
    assert fire is not None
    assert fire.rule == "floor" and fire.kind == "nudge"
    assert fire.line == coach_rules.FLOOR_LINE
    assert fire.importance >= 80
    assert fire.floor_s == pytest.approx(50.3)


def test_other_speakers_real_turn_breaks_the_floor():
    c = _coach()
    c.observe_turn("a", 0.0, 30.0, is_self=True)
    c.observe_turn("other", 30.5, 33.0, is_self=False)  # 2.5 s, not a backchannel
    assert c.observe_turn("b", 33.5, 60.0, is_self=True) is None  # floor 26.8 s


def test_backchannel_does_not_break_the_floor():
    c = _coach()
    c.observe_turn("a", 0.0, 30.0, is_self=True)
    c.observe_turn("mm", 30.2, 30.8, is_self=False)
    assert c.observe_turn("b", 31.0, 46.0, is_self=True) is not None


def test_wearer_pause_over_3s_breaks_the_floor():
    c = _coach()
    c.observe_turn("a", 0.0, 30.0, is_self=True)
    assert c.observe_turn("b", 34.0, 60.0, is_self=True) is None  # new run from 34.0


def test_min_gap_between_fires():
    c = _coach()
    assert c.observe_turn("a", 0.0, 46.0, is_self=True) is not None   # t = 46.3
    assert c.observe_turn("b", 47.0, 60.0, is_self=True) is None      # 14 s later
    assert c.observe_turn("c", 61.0, 77.0, is_self=True) is not None  # t = 77.3, 31 s later


def test_other_turns_never_fire_the_floor_rule():
    c = _coach()
    assert c.observe_turn("long other", 0.0, 90.0, is_self=False) is None


def test_unknown_identity_counts_as_not_the_wearer():
    c = _coach()
    assert c.observe_turn("a", 0.0, 90.0, is_self=None) is None


def test_stats_count_turns_and_fires():
    c = _coach()
    c.observe_turn("a", 0.0, 46.0, is_self=True)
    c.observe_turn("b", 50.0, 52.0, is_self=False)
    assert c.stats == {"turns": 2, "fires": 1, "by_rule": {"floor": 1}}


# --- lexical hook ---------------------------------------------------------------

def test_no_lexical_triggers_ship_by_default():
    assert coach_rules.RulesCoach().lexicon == ()


def test_lexical_trigger_fires_on_matching_turn():
    trig = coach_rules.LexicalTrigger(name="let_me_finish", pattern=r"\blet me finish\b",
                                      line="Let them finish.", importance=85, applies_to="other")
    c = _coach(lexicon=(trig,))
    assert c.observe_turn("Can I just", 0.0, 1.0, is_self=False) is None
    fire = c.observe_turn("Will you LET ME FINISH please", 2.0, 4.0, is_self=False)
    assert fire.rule == "lexical:let_me_finish" and fire.line == "Let them finish."
    assert fire.kind == "response" and fire.importance == 85
    assert c.observe_turn("let me finish", 5.0, 6.0, is_self=True) is None  # applies_to other only


def test_lexicon_from_env_json(tmp_path, monkeypatch):
    f = tmp_path / "lex.json"
    f.write_text(json.dumps([{"name": "always", "pattern": "you always", "line": "Name one time.",
                              "applies_to": "any"}]))
    monkeypatch.setenv("MINDSHIFT_COACH_RULES_LEXICON", str(f))
    c = coach_rules.from_env()
    assert [t.name for t in c.lexicon] == ["always"]
    assert c.observe_turn("You always do this", 0.0, 2.0, is_self=True).line == "Name one time."


def test_enabled_flag(monkeypatch):
    monkeypatch.delenv("MINDSHIFT_COACH_RULES", raising=False)
    assert coach_rules.enabled() is False
    monkeypatch.setenv("MINDSHIFT_COACH_RULES", "1")
    assert coach_rules.enabled() is True


# --- wired into the live pipeline (turn_local path) ------------------------------

class ExplodingLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, **_):
        self.calls += 1
        raise AssertionError("rules mode must never call the LLM")


def _drain(ws):
    seen = []
    while True:
        msg = json.loads(ws.receive_text())
        seen.append(msg)
        if msg.get("type") == "session_complete":
            return seen


def test_rules_mode_coaches_without_the_llm(monkeypatch, local_first_env):  # noqa: F811
    monkeypatch.setenv("MINDSHIFT_COACH_RULES", "1")
    coach_rules.reset_stats()
    llm = ExplodingLLM()
    client = _inject(StoppableTranscriber())
    app.state.llm_client = llm
    with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
        ws.send_text(json.dumps(_turn_local(text="So the thing is", start_time=0.0, end_time=20.0, is_self=True)))
        ws.send_text(json.dumps(_turn_local(text="and another thing", start_time=21.0, end_time=40.0,
                                            is_self=True)))
        ws.send_text(json.dumps(_turn_local(text="and finally", start_time=41.0, end_time=50.0, is_self=True)))
        ws.send_text(json.dumps(_turn_local(text="Okay.", speaker="Speaker B", start_time=51.0, end_time=52.0,
                                            is_self=False)))
        ws.send_text(json.dumps({"type": "stop"}))
        seen = _drain(ws)
    sugg = [m for m in seen if m.get("type") == "suggestion"]
    assert len(sugg) == 1, sugg
    [ev] = sugg
    assert ev["suggestion_source"] == "rules"
    assert ev["kind"] == "nudge" and ev["suggestions"] == [coach_rules.FLOOR_LINE]
    assert ev["speak"] is True and ev["utterance_text"] == "and finally"
    assert not [m for m in seen if m.get("type") == "suggestion_error"]
    assert llm.calls == 0
    snap = coach_rules.stats_snapshot()
    assert snap["turns"] == 4 and snap["fires"] == 1


def test_rules_fire_respects_the_interject_slider(monkeypatch, local_first_env):  # noqa: F811
    monkeypatch.setenv("MINDSHIFT_COACH_RULES", "1")
    client = _inject(StoppableTranscriber())
    app.state.llm_client = ExplodingLLM()
    with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
        ws.send_text(json.dumps({"type": "config", "interject_level": 100}))
        ws.send_text(json.dumps(_turn_local(text="long", start_time=0.0, end_time=46.0, is_self=True)))
        ws.send_text(json.dumps({"type": "stop"}))
        seen = _drain(ws)
    [ev] = [m for m in seen if m.get("type") == "suggestion"]
    assert ev["speak"] is False  # shown, not spoken: importance 90 < interject 100
