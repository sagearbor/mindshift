"""Heads-up cues in Room mode (server/room_mode.py): proactive library
passages on a topic mention, and the gated contradiction 'Check'.

Text fixtures only, $0: the LLM is a scripted fake that records its calls
(the gate decides WHETHER a call happens; the fake's verdict is scripted
per test, never a claim about model quality).
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import room_heads_up
import room_mode

LIB = """<wearer_library>
<library_item id="plan1" title="Claude working rules" kind="note">
# Permissions
Dangerous permission bypass should be asked for, not skipped. Never run with the skip flag by default.

# Data folder
Claude should read from data/ but not write there. Outputs go to tmp/.

# Meetings
Standup is at 9:30 every weekday in the blue room.
</library_item>
<library_item id="plan2" title="FY27 business plan" kind="note">
# Profit plan
Our profit target for next year is 18% net margin, up from 12% this year. Pricing changes ship in March.

# Renewals
Enterprise renewals are due in Q4 and must be signed 60 days before the term ends.

# Office
The kitchen is restocked on Fridays.
</library_item>
</wearer_library>"""

CTX = SimpleNamespace(text=LIB, mode="full", item_ids=["plan1", "plan2"])

CALM = [
    "How was your weekend?",
    "Let's grab lunch after this.",
    "Yeah I think that sounds good to me.",
    "Can everyone see my screen okay?",
    "Sorry, I was on mute for a second.",
    "I will send the notes over later today.",
    "That is a great point, thanks for raising it.",
    "Okay, let's move on to the next item.",
    "The weather has been really nice lately.",
    "Does anyone have questions before we wrap up?",
]


class FakeLLM:
    def __init__(self, reply: str = '{"contradicts": false, "confidence": 0.1, "reason": ""}'):
        self.reply, self.calls = reply, []

    def complete(self, *, system, user, max_tokens):
        self.calls.append((system, user))
        return self.reply


YES = json.dumps({"contradicts": True, "confidence": 0.95, "reason": "The note says to ask first."})


def drive(state, utterances, llm, *, start=100.0, step=40.0):
    sent = []

    async def send(f):
        sent.append(f)

    async def get_library(q):
        return CTX

    async def go():
        for i, u in enumerate(utterances):
            await room_mode.handle_room_turn(
                state, text=u, speaker="A", t=start + i * step, conversation=[],
                get_library=get_library, llm=llm, send=send)

    asyncio.run(go())
    return sent


def test_off_by_default_no_new_frames():
    st = room_mode.RoomState()
    assert st.heads_up is False
    llm = FakeLLM(YES)
    out = drive(st, ["we can use dangerously skip permissions", "our profit next year"], llm)
    assert not [f for f in out if f["type"] in ("room_headsup", "room_check")]
    assert llm.calls == []


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("ROOM_HEADS_UP_ENABLED", "0")
    st = room_mode.RoomState(heads_up=True)
    llm = FakeLLM(YES)
    out = drive(st, ["what about profit next year", "we are allowed to write to the data folder"], llm)
    assert not [f for f in out if f["type"] in ("room_headsup", "room_check")]
    assert llm.calls == []


def test_profit_headsup_has_real_title_source_snippet_and_full_passage():
    st = room_mode.RoomState(heads_up=True)
    out = drive(st, ["so what are we doing about profit next year"], FakeLLM())
    [card] = [f for f in out if f["type"] == "room_headsup"]
    assert card["title"] == "Profit plan"
    assert card["source_title"] == "FY27 business plan"
    assert card["source_item_id"] == "plan2"
    assert "profit" in card["headline"].lower()
    assert "18% net margin" in card["snippet"]
    assert len(card["snippet"]) <= 260
    assert "Pricing changes ship in March." in card["passage"]


def test_synonym_matches():
    st = room_mode.RoomState(heads_up=True)
    out = drive(st, ["our margins look thin"], FakeLLM())
    assert [f["title"] for f in out if f["type"] == "room_headsup"] == ["Profit plan"]


def test_title_never_one_word_without_heading():
    lib = SimpleNamespace(mode="full", item_ids=["x"], text=(
        '<library_item id="x" title="Notes" kind="note">\n'
        "Data retention is ninety days for all customer exports, no exceptions.\n"
        "</library_item>"))
    st = room_mode.RoomState(heads_up=True)
    sent = []

    async def send(f):
        sent.append(f)

    async def gl(q):
        return lib

    asyncio.run(room_mode.handle_room_turn(
        st, text="what is our retention policy", speaker=None, t=1.0, conversation=[],
        get_library=gl, llm=FakeLLM(), send=send))
    [c] = sent
    assert len(c["title"].split()) >= 2


def test_calm_talk_yields_nothing_and_no_llm():
    st = room_mode.RoomState(heads_up=True)
    llm = FakeLLM(YES)
    out = drive(st, CALM * 6, llm, step=10.0)  # 60 utterances ~ 10 min
    assert out == []
    assert llm.calls == []


def test_owner_cases_trigger_checks():
    st = room_mode.RoomState(heads_up=True)
    llm = FakeLLM(YES)
    out = drive(st, ["we can use dangerously skip permissions"], llm)
    checks = [f for f in out if f["type"] == "room_check"]
    assert len(llm.calls) == 1
    assert len(checks) == 1
    assert "asked for, not skipped" in checks[0]["passage"]
    assert checks[0]["statement"] == "we can use dangerously skip permissions"

    st2 = room_mode.RoomState(heads_up=True)
    llm2 = FakeLLM(YES)
    out2 = drive(st2, ["we are allowed to write to the data folder"], llm2)
    assert len(llm2.calls) == 1
    assert [f["source_title"] for f in out2 if f["type"] == "room_check"] == ["Claude working rules"]


def test_no_check_card_unless_confident_contradiction():
    for reply in ('{"contradicts": false, "confidence": 0.9}',
                  '{"contradicts": true, "confidence": 0.4}',
                  "not json at all"):
        st = room_mode.RoomState(heads_up=True)
        llm = FakeLLM(reply)
        out = drive(st, ["we are allowed to write to the data folder"], llm)
        assert len(llm.calls) == 1
        assert not [f for f in out if f["type"] == "room_check"]


def test_llm_error_shows_nothing():
    class Boom(FakeLLM):
        def complete(self, **kw):
            raise RuntimeError("down")

    st = room_mode.RoomState(heads_up=True)
    out = drive(st, ["we are allowed to write to the data folder"], Boom())
    assert not [f for f in out if f["type"] == "room_check"]


def test_topic_mention_without_checkable_claim_makes_no_llm_call():
    st = room_mode.RoomState(heads_up=True)
    llm = FakeLLM(YES)
    drive(st, ["thinking about profit"], llm)
    assert llm.calls == []


def test_per_minute_and_session_caps(monkeypatch):
    monkeypatch.setattr(room_heads_up, "CHECKS_PER_MIN", 2)
    monkeypatch.setattr(room_heads_up, "CHECKS_PER_SESSION", 3)
    utts = [f"we are allowed to write to the data folder {w}" for w in
            ("now", "today", "again", "please", "okay")]
    st = room_mode.RoomState(heads_up=True)
    llm = FakeLLM()
    drive(st, utts, llm, start=0.0, step=5.0)  # 5 in 20s
    assert len(llm.calls) == 2  # per-minute cap
    st = room_mode.RoomState(heads_up=True)
    llm = FakeLLM()
    drive(st, utts, llm, start=0.0, step=40.0)
    assert len(llm.calls) == 3  # session cap


def test_headsup_dedupe_and_min_gap():
    st = room_mode.RoomState(heads_up=True)
    out = drive(st, ["profit next year", "profit again", "renewal dates?"], FakeLLM(), start=0, step=5.0)
    assert [f["title"] for f in out if f["type"] == "room_headsup"] == ["Profit plan"]
    st = room_mode.RoomState(heads_up=True)
    out = drive(st, ["profit next year", "profit again", "renewal dates?"], FakeLLM(), start=0, step=60.0)
    assert [f["title"] for f in out if f["type"] == "room_headsup"] == ["Profit plan", "Renewals"]


def test_precision_report_on_fixtures():
    cases = [  # (utterance, expected passage title or None)
        ("we can use dangerously skip permissions", "Permissions"),
        ("we are allowed to write to the data folder", "Data folder"),
        ("what are we doing about profit next year", "Profit plan"),
        ("the renewal is coming up", "Renewals"),
    ] + [(u, None) for u in CALM]
    hits = tp = fp = 0
    for u, want in cases:
        out = drive(room_mode.RoomState(heads_up=True), [u], FakeLLM())
        got = [f["title"] for f in out if f["type"] == "room_headsup"]
        if got:
            hits += 1
            tp += want is not None and got[0] == want
            fp += want is None or got[0] != want
    print(f"heads-up precision {tp}/{hits}, false positives {fp}")
    assert fp == 0 and tp == 4


def test_config_frame_toggles_heads_up():
    import audio_pipeline

    ctx = audio_pipeline.SessionContext(session_id="s", uid="u")
    ack: dict = {}
    asyncio.run(audio_pipeline._apply_config(ctx, {"heads_up_cues": True, "mode": "room"}, ack))
    assert ctx.room is not None and ctx.room.heads_up is True
    assert ack["heads_up_cues"] is True
    asyncio.run(audio_pipeline._apply_config(ctx, {"heads_up_cues": False}, ack))
    assert ctx.room.heads_up is False and ack["heads_up_cues"] is False
    # Toggle before the mode: remembered when room mode starts.
    ctx2 = audio_pipeline.SessionContext(session_id="s2", uid="u")
    asyncio.run(audio_pipeline._apply_config(ctx2, {"heads_up_cues": True}, {}))
    asyncio.run(audio_pipeline._apply_config(ctx2, {"mode": "room"}, {}))
    assert ctx2.room.heads_up is True
    # Non-bool ignored; default stays OFF.
    ctx3 = audio_pipeline.SessionContext(session_id="s3", uid="u")
    asyncio.run(audio_pipeline._apply_config(ctx3, {"heads_up_cues": "yes", "mode": "room"}, {}))
    assert ctx3.room.heads_up is False
