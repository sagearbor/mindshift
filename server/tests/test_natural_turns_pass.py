"""main._natural_turns_pass — the conservative NaturalTurn post-pass over the
final turn list (merge same-speaker turns across short pauses only when
nothing but backchannels sits between; tag backchannels). See
NATURAL_TURNS_ENV in main.py."""
import pytest

import main
from main import AnalyzeTurn


@pytest.fixture(autouse=True)
def _natural_turns_on(monkeypatch):
    monkeypatch.setenv(main.NATURAL_TURNS_ENV, "1")


def T(speaker, text, start, end):
    return AnalyzeTurn(speaker=speaker, text=text, start_time=start, end_time=end)


def test_same_speaker_short_pause_merges():
    turns = [T("A", "we should talk about", 0.0, 2.0), T("A", "the thing from yesterday", 3.2, 5.0), T("B", "okay tell me", 5.5, 7.0),
             T("A", "it was not fair to me", 8.0, 10.0), T("B", "i hear that", 11.0, 12.0), T("A", "thank you", 13.0, 14.0)]
    out, note = main._natural_turns_pass(turns)
    assert [t.text for t in out][:2] == ["we should talk about the thing from yesterday", "okay tell me"]
    assert len(out) == 5
    assert out[0].end_time == 5.0 and out[0].kind == "primary"
    assert "1 same-speaker turn(s) merged" in note


def test_backchannel_between_does_not_break_merge_and_is_tagged():
    turns = [T("A", "so i told him the plan was off", 0.0, 3.0), T("B", "mhm", 1.0, 1.4), T("A", "and he took it well", 3.8, 6.0),
             T("B", "did he really though", 7.0, 8.5), T("A", "he said so at least", 9.0, 10.5), T("B", "fair enough then", 11.0, 12.0)]
    out, note = main._natural_turns_pass(turns)
    assert ("B", "backchannel") in [(t.speaker, t.kind) for t in out]
    merged = [t for t in out if t.speaker == "A"]
    assert len(merged) == 2 and merged[0].text == "so i told him the plan was off and he took it well"
    assert "backchannel" in note


def test_a_real_reply_between_breaks_the_merge():
    turns = [T("A", "so i told him", 0.0, 2.0), T("B", "wait what did you say to him", 2.2, 4.0), T("A", "that the plan was off", 4.5, 6.0)]
    out, _ = main._natural_turns_pass(turns)
    assert [t.speaker for t in out] == ["A", "B", "A"]
    assert all(t.kind == "primary" for t in out)


def test_long_pause_does_not_merge():
    turns = [T("A", "first thought", 0.0, 2.0), T("A", "second thought later", 4.0, 6.0)]
    out, note = main._natural_turns_pass(turns)
    assert len(out) == 2 and note is None


def test_untimed_turns_pass_through():
    turns = [AnalyzeTurn(speaker="A", text="hello"), AnalyzeTurn(speaker="A", text="again")]
    out, note = main._natural_turns_pass(turns)
    assert out is turns and note is None


def test_env_off_disables(monkeypatch):
    monkeypatch.setenv(main.NATURAL_TURNS_ENV, "0")
    turns = [T("A", "one", 0.0, 1.0), T("A", "two", 1.2, 2.0)]
    out, note = main._natural_turns_pass(turns)
    assert out is turns and note is None


def test_merge_never_drops_below_the_analysis_minimum():
    # A rapid five-sentence monologue would merge to ONE turn — below
    # ANALYZE_MIN_TURNS. The pass keeps the tags and skips the merge instead.
    turns = [T("A", f"sentence number {i} here", i * 1.0, i * 1.0 + 0.8) for i in range(5)]
    out, note = main._natural_turns_pass(turns)
    assert len(out) == 5
    assert all(t.kind == "primary" for t in out)
    assert note is None
