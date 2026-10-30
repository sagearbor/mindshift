"""Quiet coach pre-LLM trigger gate (server/coach_trigger.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import coach_trigger as ct

REPO = Path(__file__).resolve().parents[2]
MOBILE = REPO / "apps" / "mobile"
VECTORS = MOBILE / "__tests__" / "fixtures" / "coachTrigger.vectors.json"
HEAT_VECTORS = MOBILE / "__tests__" / "fixtures" / "heatLexicon.vectors.json"
PHONE_LEXICON = MOBILE / "src" / "live" / "heatLexicon.lexicon.json"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("MINDSHIFT_TRIGGER_GATE", "MINDSHIFT_TRIGGER_HEAT", "MINDSHIFT_TRIGGER_FLOOR",
              "MINDSHIFT_TRIGGER_QUESTION", "MINDSHIFT_TRIGGER_LIBRARY", "MINDSHIFT_TRIGGER_HEAT_MIN",
              "MINDSHIFT_TRIGGER_FLOOR_S"):
        monkeypatch.delenv(k, raising=False)


def test_gate_on_by_default_and_kill_switch(monkeypatch):
    assert ct.enabled()
    monkeypatch.setenv("MINDSHIFT_TRIGGER_GATE", "0")
    assert not ct.enabled()


@pytest.mark.skipif(not PHONE_LEXICON.exists(), reason="mobile sources not in this checkout")
def test_server_lexicon_is_the_phone_lexicon():
    assert json.loads(ct.LEXICON_PATH.read_text()) == json.loads(PHONE_LEXICON.read_text())


@pytest.mark.skipif(not HEAT_VECTORS.exists(), reason="mobile fixtures not in this checkout")
def test_heat_score_matches_the_phone_vectors():
    vectors = json.loads(HEAT_VECTORS.read_text())
    assert len(vectors) > 100
    for v in vectors:
        score, cats = ct.heat_score(v["text"])
        assert score == pytest.approx(v["score"]), v["text"]
        assert cats == sorted(set(v["cats"])), v["text"]


@pytest.mark.skipif(not VECTORS.exists(), reason="mobile fixtures not in this checkout")
def test_direct_question_vectors_shared_with_the_phone():
    for v in json.loads(VECTORS.read_text())["question"]:
        assert ct.is_direct_question(v["text"]) is v["q"], v["text"]


def test_untriggered_calm_turn_does_not_fire():
    v = ct.evaluate("we should look at the budget for next quarter", is_wearer=False)
    assert not v.fires and v.triggers == ()


def test_heat_trigger():
    v = ct.evaluate("shut up, it's your fault", is_wearer=False)
    assert "heat" in v.triggers and v.heat >= 1.5


def test_question_is_only_a_trigger_from_someone_else():
    q = "what do you think we should do about it?"
    assert ct.evaluate(q, is_wearer=False).triggers == ("question",)
    assert ct.evaluate(q, is_wearer=True).triggers == ()


def test_floor_trigger_from_server_clock_or_phone():
    assert ct.evaluate("and then we drove on", is_wearer=True, wearer_floor_s=50).triggers == ("floor",)
    assert ct.evaluate("and then we drove on", is_wearer=True, wearer_floor_s=30).triggers == ()
    # Once per run: only the turn whose span crosses the mark fires.
    assert ct.evaluate("x y z", is_wearer=True, wearer_floor_s=50, wearer_floor_before_s=40).triggers == ("floor",)
    assert ct.evaluate("x y z", is_wearer=True, wearer_floor_s=70, wearer_floor_before_s=50).triggers == ()
    # The other person's long floor is not the wearer's.
    assert ct.evaluate("and then we drove on", is_wearer=False, wearer_floor_s=50).triggers == ()
    assert ct.evaluate("and then we drove on", is_wearer=True, phone_triggers=["floor"]).triggers == ("floor",)
    # Unknown values from a newer phone are ignored.
    assert ct.evaluate("and then we drove on", is_wearer=True, phone_triggers=["mystery"]).triggers == ()


def test_library_phrase_trigger():
    phrases = ct.library_phrases("<item title='Lease'>The late fee is $40 after the fifth. Payment plan options.</item>")
    assert "late fee" in phrases and "payment plan" in phrases
    assert ct.evaluate("so what about the late fee", is_wearer=True, library=phrases).triggers == ("library",)
    assert ct.evaluate("so what about dinner", is_wearer=True, library=phrases).triggers == ()
    assert not ct.library_hit("late fee", None)


def test_each_trigger_has_its_own_flag(monkeypatch):
    monkeypatch.setenv("MINDSHIFT_TRIGGER_HEAT", "0")
    assert ct.evaluate("shut up", is_wearer=False).triggers == ()
    monkeypatch.setenv("MINDSHIFT_TRIGGER_QUESTION", "0")
    assert ct.evaluate("what do you think?", is_wearer=False).triggers == ()
    monkeypatch.setenv("MINDSHIFT_TRIGGER_FLOOR", "0")
    assert ct.evaluate("x y z", is_wearer=True, wearer_floor_s=99).triggers == ()
    monkeypatch.setenv("MINDSHIFT_TRIGGER_LIBRARY", "0")
    assert ct.evaluate("late fee", is_wearer=True, library=frozenset({"late fee"})).triggers == ()


def test_thresholds_are_env_tunable(monkeypatch):
    monkeypatch.setenv("MINDSHIFT_TRIGGER_FLOOR_S", "20")
    assert ct.evaluate("x y z", is_wearer=True, wearer_floor_s=25).triggers == ("floor",)
    monkeypatch.setenv("MINDSHIFT_TRIGGER_HEAT_MIN", "99")
    assert ct.evaluate("shut up", is_wearer=False).triggers == ()
