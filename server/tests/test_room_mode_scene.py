"""Room mode end to end over the live WebSocket — a scripted meeting.

The phone opens a session with ``mode: "room"`` and an "Acme account" note
plus a pipeline note selected. A meeting is replayed through the real WS
handler (scripted transcriber, one finalized utterance per audio frame):

1. small talk                                -> transcript only
2. "...Acme ... their Q3 order..."           -> a room_card carrying the Acme Q3
                                                line, verbatim, with its source
3. "MindShift, what did Acme buy last quarter?"
                                             -> a spoken room_answer drawn from
                                                the library (120 seats / $48,000)
4. "Hey MindShift, what's Acme's churn rate?" -> the library has no churn figure:
                                                the answer ADMITS it (known=false)

No coaching happens in room mode: not one ``suggestion`` frame, and the LLM
is called exactly twice (the two addressed questions) — never for the
mentions.

The answers come from a RECORDED REAL MODEL: tests/fixtures/room_mode/
answers.json holds the model's actual reply to the exact prompt this code
builds (item ids normalised). If the prompt changes, the lookup misses and
the test says how to re-record (``ROOM_RECORD=1`` with ANTHROPIC_API_KEY set
runs the real call through server/llm_client.py and rewrites the fixture).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import library
from audio_pipeline import TranscriptSegment
from library.blobs import MemoryLibraryBlobs
from library.service import LibraryService
from library.store import MemoryLibraryStore
from library_fakes import FakeEmbedder
from main import app

ME = "test-user"
SID = "c0c0c0c0-0000-4000-8000-0000000000aa"
FIXTURE = Path(__file__).parent / "fixtures" / "room_mode" / "answers.json"

ACME_NOTE = (
    "Acme Corp account notes.\n"
    "Q3 2026: Acme bought 120 seats of the Pro plan for $48,000.\n"
    "Q2 2026: Acme renewed support for $6,000.\n"
    "Champion: Dana Ortiz, VP Operations.\n"
)
PIPELINE_NOTE = "Globex is evaluating the Starter plan; decision expected in November."

MEETING = [
    # (speaker index, text, start, end)
    (0, "Morning everyone, let's get started with the weekly review.", 0.0, 3.0),
    (1, "Sure. The big one this week is Acme, their Q3 order finally closed.", 4.0, 8.0),
    (2, "MindShift, what did Acme buy last quarter?", 30.0, 33.0),
    (1, "Hey MindShift, what's Acme's churn rate?", 60.0, 63.0),
]


class ScriptedTranscriber:
    """One finalized segment per audio frame, in order (like Deepgram's
    finals); nothing left to flush at stop."""

    def __init__(self, script):
        self._segments = [TranscriptSegment(text, s, e, speaker=spk) for spk, text, s, e in script]

    async def connect(self):
        pass

    @property
    def is_connected(self):
        return True

    async def stream(self, audio_bytes):
        return [self._segments.pop(0)] if self._segments else []

    async def finish(self):
        return []

    async def close(self):
        pass


def _normalize(text: str, ids: dict[str, str]) -> str:
    for real, alias in ids.items():
        text = text.replace(real, alias)
    return text


def _key(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n---\n{user}".encode()).hexdigest()


class RecordedLLM:
    """Replays the real model's reply for the exact (normalised) prompt.
    ``ROOM_RECORD=1`` calls the real model instead and saves the reply."""

    def __init__(self, ids: dict[str, str]):
        self.ids = ids
        self.calls: list[dict] = []
        self.data = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else {"answers": {}}
        self.record = os.getenv("ROOM_RECORD") == "1"
        self.model = self.data.get("model", "recorded")

    def complete(self, *, system, user, max_tokens=512, temperature=0.7):
        s, u = _normalize(str(system), self.ids), _normalize(user, self.ids)
        key = _key(s, u)
        self.calls.append({"system": s, "user": u})
        if self.record:
            from llm_client import LLMClient
            from main import MINDSHIFT_MODEL

            client = LLMClient(model=MINDSHIFT_MODEL)
            raw = client.complete(system=system, user=user, max_tokens=max_tokens)
            self.data["model"] = MINDSHIFT_MODEL
            self.data["answers"][key] = {
                "question": u.rsplit("\n", 1)[-1], "response": _normalize(raw, self.ids),
            }
            FIXTURE.parent.mkdir(parents=True, exist_ok=True)
            FIXTURE.write_text(json.dumps(self.data, indent=2) + "\n")
            return raw
        entry = self.data["answers"].get(key)
        if entry is None:
            raise AssertionError(
                "No recorded model reply for this room prompt (the prompt changed?). "
                "Re-record: ROOM_RECORD=1 ANTHROPIC_API_KEY=... pytest "
                "server/tests/test_room_mode_scene.py"
            )
        reply = entry["response"]
        for real, alias in self.ids.items():
            reply = reply.replace(alias, real)
        return reply


@pytest.fixture
def svc():
    s = LibraryService(MemoryLibraryStore(), MemoryLibraryBlobs(), FakeEmbedder())
    library.set_service(s)
    yield s
    library.set_service(None)


def _run_meeting(svc, script=MEETING, select=True):
    from test_audio_pipeline import FakeTTS, _clear_overrides, open_ws

    acme = asyncio.run(svc.create_note(ME, "Acme account", ACME_NOTE))
    pipe = asyncio.run(svc.create_note(ME, "Pipeline", PIPELINE_NOTE))
    llm = RecordedLLM({acme.id: "ITEM-ACME", pipe.id: "ITEM-PIPELINE"})
    _clear_overrides()
    app.state.llm_client = llm
    app.state.transcriber_factory = lambda: ScriptedTranscriber(script)
    app.state.tts_client = FakeTTS()
    frames: list[dict] = []
    try:
        client = TestClient(app)
        with open_ws(client, f"/ws/session/{SID}") as ws:
            cfg = {"type": "config", "mode": "room"}
            if select:
                cfg["library_item_ids"] = [acme.id, pipe.id]
            ws.send_text(json.dumps(cfg))
            ack = json.loads(ws.receive_text())
            for _ in script:
                ws.send_bytes(b"\x00" * 3200)
            ws.send_text(json.dumps({"type": "stop"}))
            while True:
                msg = json.loads(ws.receive_text())
                frames.append(msg)
                if msg.get("type") == "session_complete":
                    break
    finally:
        _clear_overrides()
    return ack, frames, llm, acme, pipe


@pytest.mark.skipif(not FIXTURE.exists() and os.getenv("ROOM_RECORD") != "1",
                    reason="recorded room answers missing")
class TestRoomMeeting:
    def test_scene(self, svc):
        ack, frames, llm, acme, _pipe = _run_meeting(svc)
        assert ack["type"] == "config_ack" and ack["mode"] == "room"
        types = [f["type"] for f in frames]
        assert types.count("transcript") == len(MEETING)
        # Room mode never coaches.
        assert "suggestion" not in types and "suggestion_error" not in types

        cards = [f for f in frames if f["type"] == "room_card"]
        assert len(cards) == 1
        card = cards[0]
        assert card["fact"] == "Q3 2026: Acme bought 120 seats of the Pro plan for $48,000."
        assert card["source_item_id"] == acme.id
        assert card["source_title"] == "Acme account"
        assert card["title"] == "Acme · Q3"
        assert card["t"] == 8.0

        answers = [f for f in frames if f["type"] == "room_answer"]
        assert [a["question"] for a in answers] == [
            "what did Acme buy last quarter?", "what's Acme's churn rate?",
        ]
        bought, churn = answers
        # Grounded in the library, spoken, short.
        assert bought["known"] is True and bought["speak"] is True
        assert "120" in bought["text"] and re.search(r"48,?000", bought["text"])
        assert acme.id in bought["source_item_ids"]
        assert len(re.findall(r"[.!?](\s|$)", bought["text"])) <= 2
        # Not in the library or the conversation: admitted, not invented.
        assert churn["known"] is False and churn["speak"] is True
        assert re.search(r"don'?t|do not|not", churn["text"], re.I)
        assert not re.search(r"\d+(\.\d+)?\s*%", churn["text"])

        # Exactly two LLM calls: the addressed questions. The mention cost none.
        assert len(llm.calls) == 2
        first = llm.calls[0]
        assert "Q3 2026: Acme bought 120 seats" in first["system"]
        assert "Speaker B: Sure. The big one this week is Acme" in first["user"]
        # Order: the card precedes the first answer on the wire.
        assert types.index("room_card") < types.index("room_answer")


class TestRoomWithoutLibrary:
    def test_mentions_make_no_cards_and_no_llm_calls(self, svc):
        script = MEETING[:2]
        ack, frames, llm, *_ = _run_meeting(svc, script, select=False)
        assert ack["mode"] == "room"
        assert [f["type"] for f in frames] == ["transcript", "transcript", "session_complete"]
        assert llm.calls == []

    def test_wake_alone_then_question(self, svc, monkeypatch):
        import room_mode

        sent = []

        class Echo:
            model = "echo"
            calls = []

            def complete(self, *, system, user, max_tokens=512, temperature=0.7):
                self.calls.append(user)
                return '{"answer": "I don\'t have that in the library or in this conversation.", "known": false}'

        from test_audio_pipeline import FakeTTS, _clear_overrides, open_ws

        script = [(0, "Hey MindShift.", 0.0, 1.0), (0, "What is our ARR?", 2.0, 3.0)]
        _clear_overrides()
        echo = Echo()
        app.state.llm_client = echo
        app.state.transcriber_factory = lambda: ScriptedTranscriber(script)
        app.state.tts_client = FakeTTS()
        try:
            with open_ws(TestClient(app), f"/ws/session/{SID}") as ws:
                ws.send_text(json.dumps({"type": "config", "mode": "room"}))
                json.loads(ws.receive_text())
                for _ in script:
                    ws.send_bytes(b"\x00" * 3200)
                ws.send_text(json.dumps({"type": "stop"}))
                while True:
                    msg = json.loads(ws.receive_text())
                    sent.append(msg)
                    if msg["type"] == "session_complete":
                        break
        finally:
            _clear_overrides()
        types = [m["type"] for m in sent]
        assert "room_listening" in types
        answer = next(m for m in sent if m["type"] == "room_answer")
        assert answer["question"] == "What is our ARR?" and answer["known"] is False
        assert len(echo.calls) == 1
        assert room_mode.WAKE_FOLLOWUP_S >= 2.0

    def test_llm_failure_is_reported_never_fabricated(self, svc):
        from test_audio_pipeline import FakeTTS, _clear_overrides, open_ws

        class Broken:
            model = "broken"

            def complete(self, **_kw):
                raise RuntimeError("no key")

        script = [(0, "MindShift, what's on the agenda?", 0.0, 2.0)]
        _clear_overrides()
        app.state.llm_client = Broken()
        app.state.transcriber_factory = lambda: ScriptedTranscriber(script)
        app.state.tts_client = FakeTTS()
        sent = []
        try:
            with open_ws(TestClient(app), f"/ws/session/{SID}") as ws:
                ws.send_text(json.dumps({"type": "config", "mode": "room"}))
                json.loads(ws.receive_text())
                ws.send_bytes(b"\x00" * 3200)
                ws.send_text(json.dumps({"type": "stop"}))
                while True:
                    msg = json.loads(ws.receive_text())
                    sent.append(msg)
                    if msg["type"] == "session_complete":
                        break
        finally:
            _clear_overrides()
        err = next(m for m in sent if m["type"] == "room_answer_error")
        assert err == {"type": "room_answer_error", "question": "what's on the agenda?",
                       "reason": "RuntimeError", "t": 2.0}
        assert not any(m["type"] == "room_answer" for m in sent)


class TestModeConfig:
    def test_mode_can_be_switched_back_to_coaching(self, svc):
        from test_audio_pipeline import MOCK_LLM_JSON, FakeTranscriber, FakeTTS, _clear_overrides, open_ws
        from unittest.mock import MagicMock

        _clear_overrides()
        llm = MagicMock()
        llm.complete.return_value = MOCK_LLM_JSON
        app.state.llm_client = llm
        app.state.transcriber_factory = lambda: FakeTranscriber()
        app.state.tts_client = FakeTTS()
        try:
            with open_ws(TestClient(app), f"/ws/session/{SID}") as ws:
                ws.send_text(json.dumps({"type": "config", "mode": "room"}))
                assert json.loads(ws.receive_text())["mode"] == "room"
                ws.send_text(json.dumps({"type": "config", "mode": "earpiece"}))
                assert json.loads(ws.receive_text())["mode"] == "earpiece"
                ws.send_text(json.dumps({"type": "config", "mode": "bogus"}))
                assert json.loads(ws.receive_text())["mode"] == "earpiece"
        finally:
            _clear_overrides()


class TestSessionRecord:
    def test_live_session_record_accepts_room_mode(self):
        from pydantic import ValidationError

        from routers.sessions import LiveSessionIn

        body = {
            "session_id": SID, "started_at": "2026-10-07T10:00:00Z",
            "ended_at": "2026-10-07T10:30:00Z", "turns": [],
        }
        assert LiveSessionIn(**body, mode="room").mode == "room"
        with pytest.raises(ValidationError):
            LiveSessionIn(**body, mode="boardroom")
