"""speak_verifier: strict parse, fail-closed on error/timeout, deterministic prompt."""

import asyncio
import time

import pytest

import speak_verifier as sv
from speak_verifier import Turn


class FixedLLM:
    def __init__(self, raw="", delay=0.0, exc=None):
        self.raw, self.delay, self.exc = raw, delay, exc
        self.calls = []

    def complete(self, system, user, **kw):
        self.calls.append((system, user, kw))
        if self.delay:
            time.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.raw


TURNS = [
    Turn("wearer", "Speaker A", 0.0, 50.0, "You are obliged to apply the law, obliged."),
    Turn("other", "Speaker B", 50.0, 52.0, "Can I answer? Of course, I'm listening."),
]


def test_default_off(monkeypatch):
    monkeypatch.delenv("MINDSHIFT_SPEAK_VERIFIER", raising=False)
    assert sv.enabled() is False
    monkeypatch.setenv("MINDSHIFT_SPEAK_VERIFIER", "1")
    assert sv.enabled() is True


def test_variant_and_timeout_env(monkeypatch):
    monkeypatch.setenv("MINDSHIFT_SPEAK_VERIFIER_PROMPT", "v3")
    assert sv.variant() == "v3"
    monkeypatch.setenv("MINDSHIFT_SPEAK_VERIFIER_PROMPT", "nope")
    assert sv.variant() == sv.DEFAULT_VARIANT
    monkeypatch.setenv("MINDSHIFT_SPEAK_VERIFIER_TIMEOUT_S", "0.5")
    assert sv.timeout_s() == 0.5


@pytest.mark.parametrize("raw,which,want", [
    ('{"speak": true, "reason": "x"}', "v1", True),
    ('{"speak": "true", "reason": "x"}', "v1", False),  # only an explicit boolean yes
    ('sure! {"speak": false}', "v1", False),
    ("not json", "v1", False),
    ('{"evidence": "can I answer", "speak": true}', "v2", True),
    ('{"evidence": "", "speak": true}', "v2", False),  # a yes needs a quote
    ('{"moment": "floor_request", "fits": 3}', "v3", True),
    ('{"moment": "floor_request", "fits": 2}', "v3", False),
    ('{"moment": "none", "fits": 3}', "v3", False),
])
def test_parse_is_strict(raw, which, want):
    assert sv.parse(raw, which)[0] is want


def test_user_prompt_is_deterministic_and_marks_handoffs():
    u1 = sv.build_user(TURNS, "Let him answer.", setting="TV debate", relationship="other")
    u2 = sv.build_user(TURNS, "Let him answer.", setting="TV debate", relationship="other")
    assert u1 == u2
    assert "WEARER" in u1 and "OTHER (Speaker B)" in u1
    assert "starts the instant the previous turn stops" in u1
    assert 'Candidate line' in u1 and "Let him answer." in u1
    unk = sv.build_user([Turn("unknown", "Speaker A", 0, 1, "hi there you")], "Pause.", wearer_unknown=True)
    assert "not identified" in unk and "UNIDENTIFIED (Speaker A)" in unk


def test_overlap_is_marked():
    turns = [Turn("other", "S1", 0.0, 5.0, "I was saying that we"), Turn("wearer", "S2", 3.0, 6.0, "No no no listen")]
    assert "overlaps the previous turn by 2.0s" in sv.build_user(turns, "Let them finish.")


def test_only_last_six_turns():
    turns = [Turn("other", "S1", i, i + 0.5, f"turn number {i}") for i in range(10)]
    u = sv.build_user(turns, "x")
    assert "turn number 3" not in u and "turn number 4" in u and "turn number 9" in u


def test_verify_yes_and_errors_fail_closed():
    v = asyncio.run(sv.verify(FixedLLM('{"evidence":"can I answer","speak":true,"reason":"floor"}'), "u", "v2", 1.0))
    assert v.speak is True and v.error is None
    v = asyncio.run(sv.verify(FixedLLM(exc=RuntimeError("boom")), "u", "v1", 1.0))
    assert v.speak is False and v.error == "RuntimeError"


def test_verify_timeout_fails_closed():
    v = asyncio.run(sv.verify(FixedLLM('{"speak": true}', delay=0.5), "u", "v1", 0.05))
    assert v.speak is False and v.error == "timeout"


def test_call_uses_variant_system_and_small_budget():
    llm = FixedLLM('{"speak": false}')
    sv.verify_sync(llm, "u", "v1")
    system, user, kw = llm.calls[0]
    assert system == sv.VARIANTS["v1"] and user == "u"
    assert kw["max_tokens"] <= 120 and kw["temperature"] == 0.0
