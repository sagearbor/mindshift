"""Recording-replay scoring on a hand-built run bundle: coach lines and
their real-time latency, moment hits, unmatched lines, identity accuracy
over time, and the invented-fact / put-words-in-mouth checks."""

from __future__ import annotations

import pytest

from recreplay import score


def _turn(uid, speaker, is_self, text, start, end, sent):
    return {"type": "turn_local", "turn_uid": uid, "speaker": speaker, "is_self": is_self, "text": text,
            "start_time": start, "end_time": end, "sent_at_audio_s": sent,
            "speaker_match_score": 0.7 if is_self else None}


def _sugg(at, utt, kind, lines, *, speak=True, partial=False, speaker="X"):
    return {"at_s": at, "event": {"type": "suggestion", "kind": kind, "utterance_text": utt, "speaker": speaker,
                                  "suggestions": lines, "speak": speak, "partial": partial, "importance": 80}}


def _bundle():
    turns = [
        _turn("u0", "You", True, "Why don't you want to go to dinner with mom?", 1.0, 4.0, 4.8),
        _turn("u1", "Speaker B", False, "Because I want to do my homework.", 4.5, 7.0, 7.9),
        _turn("u2", "You", None, "Italian food just has carbs anyway.", 12.0, 16.0, 16.9),
        _turn("u3", "You", True, "Hey hey settle down.", 22.0, 24.0, 24.8),
    ]
    events = [
        {"at_s": 0.0, "event": {"type": "config_ack"}},
        _sugg(5.9, turns[0]["text"], "nudge", ["soften your voice"]),
        _sugg(8.6, turns[1]["text"], "response", ["Ask what homework he has."], partial=True, speak=False),
        _sugg(9.1, turns[1]["text"], "response", ["Ask what homework he has.", "I finished my taxes in Denver yesterday."]),
        {"at_s": 13.0, "event": {"type": "speaker_identity", "speaker": "You", "is_self": True, "person_id": "self", "score": 0.7}},
        _sugg(18.2, turns[2]["text"], "response", ["Tell him he should say sorry to mom."], speak=False),
        _sugg(26.0, turns[3]["text"], "nudge", ["lower your voice"]),
        {"at_s": 27.0, "event": {"type": "suggestion_error", "utterance_text": "x", "reason": "llm_parse_error"}},
    ]
    sent = [{"at_s": t["sent_at_audio_s"], "turn": {k: v for k, v in t.items() if k != "sent_at_audio_s"}} for t in turns]
    truth = [
        {"start": 1.0, "end": 4.0, "speaker": "S1", "text": "why don't you want to go to dinner with mom", "aligned": True},
        {"start": 4.5, "end": 7.0, "speaker": "S2", "text": "because I want to do my homework", "aligned": True},
        {"start": 12.0, "end": 16.0, "speaker": "S1", "text": "italian food just has carbs anyway", "aligned": True,
         "vocal_intensity": 2},
        {"start": 22.0, "end": 24.0, "speaker": "S1", "text": "hey hey settle down", "aligned": True},
    ]
    return {
        "audio": {"duration_s": 30.0},
        "settings": {"speed": 1.0, "session_context": "dinner at home with my son and wife"},
        "notes": {"moments": [{"t": 17.0, "text": "skip the jab", "source": "owner"},
                              {"t": 3.0, "text": "ask, don't argue", "source": "owner"}]},
        "identity": {"wearer_label": "Speaker A", "wearer_ann_id": "S1"},
        "annotations": [{"label": "default", "ok": True, "aligned": {
            "segments": truth,
            "coach_moments": [{"t": 24.5, "for_speaker": "S1", "ideal_nudge": "lower your voice", "kind": "warning", "priority": 1},
                              {"t": 6.0, "for_speaker": "S2", "ideal_nudge": "say more", "kind": "warning", "priority": 2}],
        }}],
        "primary_annotation": "default",
        "stt": {"words": [], "turns": []},
        "phone": {"sent": turns, "haptics": [{"level": 2, "atSec": 16.9, "code": "H", "atMs": 16900}],
                  "nudges": [], "positives": []},
        "phone_ceiling": None,
        "server": {"events": events, "sent": sent, "session_complete": {"type": "session_complete"}, "llm": {}},
    }


def test_coach_lines_are_timed_against_the_turn_they_answer():
    s = score.score(_bundle())
    lines = s["lines"]
    srv = [ln for ln in lines if ln["source"] == "server"]
    assert [ln["kind"] for ln in srv] == ["nudge", "response", "response", "nudge"]
    first = srv[0]
    assert first["turn_end_s"] == 4.0
    assert first["latency_ms"] == pytest.approx(1900, abs=1)          # 5.9 - 4.0
    assert first["from_sent_ms"] == pytest.approx(1100, abs=1)        # 5.9 - 4.8
    resp = srv[1]
    assert resp["first_words_ms"] == pytest.approx(1600, abs=1)       # the partial at 8.6 - 7.0
    assert resp["latency_ms"] == pytest.approx(2100, abs=1)
    assert srv[2]["speak"] is False and srv[2]["fires"] is False
    phone = [ln for ln in lines if ln["source"] == "phone"]
    assert phone and phone[0]["kind"] == "haptic" and phone[0]["latency_ms"] == pytest.approx(900, abs=1)
    lat = s["latency"]
    assert lat["n"] == 4 and lat["p50_ms"] == pytest.approx(2100, abs=150)
    assert s["errors"] == [{"at_s": 27.0, "reason": "llm_parse_error", "utterance_text": "x"}]


def test_moments_hit_and_unmatched_lines():
    s = score.score(_bundle(), moment_window_s=6.0)
    m = {(x["source"], x["t"]): x for x in s["moments"]["items"]}
    assert m[("owner", 17.0)]["hit"] is True                # the phone haptic at 16.9 (and nothing else) is near
    assert m[("owner", 3.0)]["hit"] is True                 # the nudge at 5.9
    assert m[("annotation", 24.5)]["hit"] is True           # the nudge at 26.0
    # the S2 moment is for someone else: listed, not scored
    assert all(x["t"] != 6.0 for x in s["moments"]["items"])
    assert s["moments"]["hits"] == 3 and s["moments"]["total"] == 3
    # 9.1 (response) is > 6 s from every moment
    assert [round(u["at_s"], 1) for u in s["moments"]["unmatched_lines"]] == [9.1]


def test_identity_accuracy_and_first_confirmation():
    s = score.score(_bundle())
    ph = s["identity"]["phone"]
    assert ph["turns"] == 4 and ph["decided"] == 3 and ph["correct"] == 3
    assert ph["accuracy"] == pytest.approx(1.0)
    assert ph["wearer_turns"] == 3 and ph["wearer_recall"] == pytest.approx(2 / 3)
    assert ph["first_confirmed_s"] == 4.8
    srv = s["identity"]["server"]
    assert srv["first_confirmed_s"] == 13.0 and srv["events"] == 1 and srv["correct"] == 1
    # coaching direction: a nudge (wearer's own turn) / response (someone else's) vs truth
    d = s["identity"]["coaching_direction"]
    assert d["correct"] == 3 and d["total"] == 4     # u2 was the owner but got a "response"
    assert s["identity"]["timeline"][0]["t"] == 4.8


def test_violations_flag_invented_facts_and_scripting_the_other_person():
    s = score.score(_bundle())
    v = s["violations"]
    kinds = {x["kind"] for x in v}
    assert "invented-fact" in kinds          # Denver / taxes / yesterday: nowhere in the conversation
    assert "words-in-mouth" in kinds         # "Tell him he should say sorry"
    texts = " ".join(x["text"] for x in v)
    assert "Denver" in texts and "should say sorry" in texts
    assert not any("soften your voice" in x["text"] for x in v)


def test_empty_bundle_scores_without_crashing():
    s = score.score({"audio": {"duration_s": 5.0}, "settings": {}, "notes": {"moments": []}, "identity": {},
                     "annotations": [], "stt": {"words": [], "turns": []}, "phone": None, "server": None})
    assert s["latency"]["n"] == 0 and s["moments"]["total"] == 0
    assert s["identity"]["phone"]["turns"] == 0


def test_percentile_helper():
    assert score.percentile([], 50) is None
    assert score.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert score.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90) == pytest.approx(9.1)
