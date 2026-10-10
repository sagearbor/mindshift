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


def test_after_laughter_window_is_opt_in(monkeypatch):
    recent0 = [(0.0, 2.0, "that is so funny hahaha")]
    assert cg.skip_reason("And then what did he say to you", 3.0, 5.0, recent0) is None  # default off
    monkeypatch.setenv("MINDSHIFT_LAUGH_SKIP_S", "4")
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


# --- dark prompt flag: importance anchors -------------------------------------

def test_importance_anchors_dark_by_default(monkeypatch):
    import main
    monkeypatch.delenv("MINDSHIFT_IMPORTANCE_PROMPT", raising=False)
    prompts = [main.empathy_system_prompt(50, live=True), main.unknown_wearer_prompt(50),
               main.self_feedback_prompt(50)]
    assert not any(main.IMPORTANCE_ANCHORS in p for p in prompts)
    monkeypatch.setenv("MINDSHIFT_IMPORTANCE_PROMPT", "anchored")
    on = [main.empathy_system_prompt(50, live=True), main.unknown_wearer_prompt(50),
          main.self_feedback_prompt(50)]
    assert all(main.IMPORTANCE_ANCHORS in p for p in on)
    for before, after in zip(prompts, on):
        assert after.replace(" " + main.IMPORTANCE_ANCHORS, "") == before


# --- 3b. unknown-wearer neutral cues (mid-argument switch-on) -------------------

ARGUE = [  # (speaker, start, end, text)
    ("Speaker A", 0.0, 50.0, "You are obliged to apply the law, you are obliged, that is the point."),
    ("Speaker B", 50.0, 52.0, "Can I answer? Of course, I'm listening."),
]
GREEK = [
    ("Speaker A", 24.6, 76.0, "Εάν ήσασταν βουλευτής θα μπορούσατε να το είχατε καταψηφίσει."),
    ("Speaker B", 77.3, 79.4, "Μπορώ να απαντήσω και ρε πρετε. Βεβαίως, ακούω."),
]


@pytest.mark.parametrize("line,want", [
    ("Pause. Let them finish.", True),
    ("Let him respond fully before you speak again.", True),
    ("Pause.", True),
    ("Give them room to answer.", True),
    ("Let them complete their response fully.", True),
    ("What's one thing you enjoy here?", False),
    ("You're spiraling. Take a breath and ask what they meant by that.", False),
])
def test_neutral_cue(line, want):
    assert cg.neutral_cue(line) is want


def test_interruption_evidence_needs_a_floor_claim_after_someone_else_held_it():
    assert cg.interruption_evidence(ARGUE)
    assert cg.interruption_evidence(GREEK)
    calm = [("Speaker A", 0.0, 50.0, ARGUE[0][3]), ("Speaker B", 51.0, 54.0, "I see what you mean about that law.")]
    assert not cg.interruption_evidence(calm)
    # a floor claim with nobody holding/overlapping before it is not enough
    assert not cg.interruption_evidence([("Speaker B", 0.0, 2.0, "Can I answer? Of course.")])
    # the same speaker asking after their own turn is not a claim against anyone
    assert not cg.interruption_evidence([("Speaker B", 0.0, 30.0, ARGUE[0][3]), ("Speaker B", 30.0, 32.0, "Can I answer?")])


def test_interruption_evidence_sees_a_monologue_the_phone_chunked():
    # E-020 confer_20111212_seq11: the phone cuts a 76 s monologue into ~8 s
    # turns; the floor claim follows 1.5 s after the last chunk.
    chunks = [("Speaker W", s, e, "Εάν ήσασταν βουλευτής θα μπορούσατε να το είχατε καταψηφίσει.")
              for s, e in [(0.06, 37.8), (38.2, 46.4), (46.9, 57.6), (58.2, 66.3), (66.6, 71.9), (72.6, 76.1)]]
    turns = chunks + [("Speaker A", 77.6, 79.4, "Μπορώ να απαντήσω και ρε πρετε. Βεβαίως, ακούω."),
                      ("Unknown", 80.0, 80.7, "Μπορώ να απαντήσω και ρε πρετε. Βεβαίως, ακούω. Πρώτα")]
    assert cg.interruption_evidence(turns)
    assert cg.interruption_evidence(turns[:-1])
    short = [("Speaker W", 70.0, 73.0, "and that is what the law says, clearly"),
             ("Speaker W", 73.5, 76.1, "so you have to apply it"),
             ("Speaker A", 77.6, 79.4, "Can I answer?")]
    assert not cg.interruption_evidence(short)  # 6 s of talk, 1.5 s gap: no hold, no cut-in


def test_interruption_evidence_overlap_plus_claim():
    turns = [("S1", 0.0, 6.0, "So what I was going to say is that the budget"),
             ("S2", 4.0, 7.0, "No no that's not what happened at all"),
             ("S1", 7.0, 9.0, "Let me finish, please, let me finish.")]
    assert cg.interruption_evidence(turns)


def test_interruption_evidence_ignores_stale_turns():
    old = [("Speaker A", 0.0, 50.0, ARGUE[0][3]), ("Speaker B", 50.0, 52.0, "Can I answer?"),
           ("Speaker A", 90.0, 95.0, "So anyway the weather has been lovely this week.")]
    assert not cg.interruption_evidence(old)


def test_gate_unknown_neutral_cue_bypasses_cap(monkeypatch):
    g = _gate()
    kw = dict(wearer_unknown=True, turn_text=SUBST, turn_duration_s=2, now_s=100, last_spoken_s=None)
    assert g.passes(72, **kw)[0] is False
    monkeypatch.setenv("MINDSHIFT_SPEAK_UNKNOWN_NEUTRAL", "1")
    monkeypatch.setenv("MINDSHIFT_SPEAK_NEUTRAL_MIN_IMPORTANCE", "60")
    assert g.passes(72, neutral_ok=True, **kw) == (True, "pass_neutral")
    assert g.passes(55, neutral_ok=True, **kw)[0] is False      # below the neutral floor
    assert g.passes(72, neutral_ok=False, **kw)[0] is False     # no evidence -> capped as before
    assert g.passes(72, neutral_ok=True, **{**kw, "last_spoken_s": 90})[1] == "gap"  # gap still applies
    monkeypatch.setenv("MINDSHIFT_SPEAK_UNKNOWN_NEUTRAL", "0")
    assert g.passes(72, neutral_ok=True, **kw)[0] is False
