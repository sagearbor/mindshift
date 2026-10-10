"""Unit tests for server/coach_gate.py (label scrub, turn skip, speak gate)."""

import pytest

import coach_gate as cg


@pytest.fixture(autouse=True)
def _gates_on(monkeypatch):
    # server/conftest.py turns the skip + speak gate OFF for the legacy
    # suites; these tests pin the shipped defaults.
    for k in ("MINDSHIFT_TURN_SKIP", "MINDSHIFT_SPEAK_GATE", "MINDSHIFT_LABEL_SCRUB",
              "MINDSHIFT_SPEAK_MIN_IMPORTANCE", "MINDSHIFT_SPEAK_MIN_GAP_S",
              "MINDSHIFT_SPEAK_MIN_WORDS", "MINDSHIFT_SPEAK_UNKNOWN_CAP",
              "MINDSHIFT_TURN_SKIP_MAX_WORDS", "MINDSHIFT_LAUGH_SKIP_S"):
        monkeypatch.delenv(k, raising=False)


# --- 1. label scrub ---------------------------------------------------------

@pytest.mark.parametrize("raw, want", [
    ("Let Speaker E finish.", "Let them finish."),
    ("Ask Speaker B what they need.", "Ask them what they need."),
    ("Acknowledge Speaker C's point first.", "Acknowledge their point first."),
    ("Speaker D sounds hurt — slow down.", "The other person sounds hurt — slow down."),
    ("Pause, because Speaker A is still talking.", "Pause, because the other person is still talking."),
    ("Let SPEAKER_01 finish.", "Let them finish."),
    ("Let speaker e finish.", "Let them finish."),
    ("Give S2 a moment.", "Give them a moment."),
    ("Let Speaker A and Speaker B sort it out.", "Let both of them sort it out."),
    ("Speakers A and B are talking over each other.", "Both of them are talking over each other."),
    ("Let Speaker AA finish.", "Let them finish."),
])
def test_scrub_rewrites_labels(raw, want):
    out = cg.scrub_labels(raw)
    assert out == want
    assert not cg.has_label(out)


def test_scrub_leaves_clean_lines_alone():
    for line in ("Let them finish.", "The speakers are loud.", "Take a breath.", "Say what you heard."):
        assert cg.scrub_labels(line) == line


def test_scrub_maps_the_wearers_own_label_to_you():
    assert cg.scrub_labels("Speaker A, slow down.", {"Speaker A"}) == "You, slow down."
    assert cg.scrub_labels("Lower Speaker A's voice.", {"Speaker A"}) == "Lower your voice."


def test_scrub_lines_never_emits_a_label():
    lines = ["Let Speaker E finish.", "Breathe.", "Ask Speaker F why."]
    out = cg.scrub_lines(lines)
    assert out == ["Let them finish.", "Breathe.", "Ask them why."]
    assert not any(cg.has_label(x) for x in out)


def test_scrub_flag_off_passes_through(monkeypatch):
    monkeypatch.setenv("MINDSHIFT_LABEL_SCRUB", "0")
    assert cg.scrub_lines(["Let Speaker E finish."]) == ["Let Speaker E finish."]


# --- 2. turn skip -------------------------------------------------------------

@pytest.mark.parametrize("text, reason", [
    ("Yeah.", "fragment"),
    ("Okay, sure.", "fragment"),
    ("Yeah yeah okay right.", "backchannel"),
    ("Mhm, yeah, I know.", None) ,
    ("haha", "laughter"),
    ("[laughter] oh my god", "laughter"),
    ("You never listen to me.", None),
    ("Shut up.", "fragment"),
])
def test_skip_reason(text, reason):
    assert cg.skip_reason(text, 0.0, 1.0) == reason


def test_greek_words_are_counted():
    assert cg.skip_reason("Σαφής η κυβέρνησή σας ταύτισε", 0.0, 2.0) is None
    assert cg.skip_reason("Σαφής", 0.0, 0.5) == "fragment"


def test_after_laughter_window():
    recent = [(0.0, 2.0, "that is so funny hahaha")]
    assert cg.skip_reason("And then what did he say to you", 3.0, 5.0, recent) == "after_laughter"
    assert cg.skip_reason("And then what did he say to you", 9.0, 11.0, recent) is None


def test_tone_laughter_flag_and_strong_negative_exemption():
    assert cg.skip_reason("You really did that again", 0, 1, tone_context={"laughter": True}) == "laughter"
    assert cg.skip_reason("Shut up!", 0, 1, tone_context={"text_tone": {"frustration": 0.8}}) is None


def test_skip_kill_switch(monkeypatch):
    monkeypatch.setenv("MINDSHIFT_TURN_SKIP", "0")
    assert cg.skip_reason("Yeah.", 0, 1) is None


# --- 3. speak gate --------------------------------------------------------------

def _gate(**kw):
    base = dict(enabled=True, min_importance=75, min_gap_s=30.0, min_words=4, unknown_cap=70)
    base.update(kw)
    return cg.SpeakGate(**base)


SUBST = "You never listen to anything I say."


def test_gate_importance_threshold():
    g = _gate()
    assert g.passes(80, wearer_unknown=False, turn_text=SUBST, turn_duration_s=2, now_s=100, last_spoken_s=None)[0]
    assert g.passes(74, wearer_unknown=False, turn_text=SUBST, turn_duration_s=2, now_s=100, last_spoken_s=None) == (False, "importance")


def test_gate_min_gap():
    g = _gate()
    assert g.passes(90, wearer_unknown=False, turn_text=SUBST, turn_duration_s=2, now_s=100, last_spoken_s=80) == (False, "gap")
    assert g.passes(90, wearer_unknown=False, turn_text=SUBST, turn_duration_s=2, now_s=111, last_spoken_s=80)[0]


def test_gate_requires_substantive_turn():
    g = _gate()
    assert g.passes(90, wearer_unknown=False, turn_text="No way, really?", turn_duration_s=1, now_s=100,
                    last_spoken_s=None) == (False, "not_substantive")


def test_gate_caps_unknown_wearer():
    g = _gate()
    assert g.passes(90, wearer_unknown=True, turn_text=SUBST, turn_duration_s=2, now_s=100, last_spoken_s=None) == (False, "unknown_cap")
    g2 = _gate(unknown_cap=85)
    assert g2.passes(90, wearer_unknown=True, turn_text=SUBST, turn_duration_s=2, now_s=100, last_spoken_s=None)[0]


def test_gate_disabled_always_passes():
    g = _gate(enabled=False)
    assert g.passes(0, wearer_unknown=True, turn_text="ok", turn_duration_s=0.2, now_s=1, last_spoken_s=0.5)[0]


def test_gate_from_env(monkeypatch):
    monkeypatch.setenv("MINDSHIFT_SPEAK_MIN_IMPORTANCE", "60")
    monkeypatch.setenv("MINDSHIFT_SPEAK_MIN_GAP_S", "12")
    monkeypatch.setenv("MINDSHIFT_SPEAK_GATE", "0")
    g = cg.SpeakGate.from_env()
    assert (g.enabled, g.min_importance, g.min_gap_s) == (False, 60, 12.0)
