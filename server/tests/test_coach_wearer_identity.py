"""Who is the wearer? (owner requirement C, 2026-10-07).

The incident this replays: the owner wore an earpiece in live-coach mode at
a family dinner. His son spoke first (about his school day and a quiz), so
Deepgram called the son "Speaker A". The phone's on-device loop produced no
turns, so the server's cloud suggestions were voiced — and the server
decided who "self" was by LABEL ALONE: the phone's default
``self_speaker="Speaker A"`` made the son the wearer, and the coach scripted
first-person lines in his voice ("I think I did well on the quiz").

The rules these tests pin:

* a ``self_speaker`` label with no confirmation is NOT the wearer — it is
  "unknown";
* while the wearer is unknown, every turn gets the speaker-neutral prompt
  (no first-person scripted replies, no attributing statements, no "you
  said"), and no turn is coached as the wearer's own;
* the wearer becomes known by: the phone's voiceprint verdict on a turn
  (``turn_local.is_self``), the user tapping a label as themselves
  (``speaker_label`` with ``is_self``), the phone asserting its
  ``self_speaker`` is confirmed (config ``wearer_known: true``), or the
  SERVER's own voiceprint match against the user's enrolled print;
* once known, personalized coaching resumes.

Real prompt construction throughout; the LLM is a recording double. No real
LLM call is made (no credentials in the suite), so the assertions are on the
exact prompts the model would receive.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

import audio_pipeline
from audio_pipeline import (
    WEARER_OTHER,
    WEARER_SELF,
    WEARER_UNKNOWN,
    SessionContext,
    TranscriptSegment,
    _resolve_wearer,
    apply_speaker_label,
)
from main import (
    COACH_GROUND_RULES,
    COACH_UNKNOWN_WEARER_RULES,
    app,
    empathy_system_prompt,
    self_feedback_prompt,
    unknown_wearer_prompt,
)
from models.audio import Utterance

RELATIONSHIP_WORDS = ("husband", "wife", "spouse", "son", "daughter", "marriage")


def _utt(text: str, speaker: str, start: float = 0.0, end: float = 2.0) -> Utterance:
    return Utterance(session_id="s", speaker=speaker, text=text, start_time=start, end_time=end)


# ---------------------------------------------------------------------------
# The unknown-wearer prompt itself
# ---------------------------------------------------------------------------

class TestUnknownWearerPrompt:
    @pytest.mark.parametrize("slider", [0, 50, 100])
    def test_carries_the_neutral_rules_and_the_core_job(self, slider):
        p = unknown_wearer_prompt(slider)
        assert COACH_UNKNOWN_WEARER_RULES in p
        assert COACH_GROUND_RULES in p
        low = COACH_UNKNOWN_WEARER_RULES.lower()
        assert "not yet identified" in low
        assert "never write a first-person line" in low
        assert "never attribute" in low
        assert '"you said"' in low
        assert '"suggestions"' in p and '"importance"' in p

    def test_known_wearer_prompts_do_not_carry_the_unknown_rules(self):
        assert COACH_UNKNOWN_WEARER_RULES not in empathy_system_prompt(50, live=True)
        assert COACH_UNKNOWN_WEARER_RULES not in self_feedback_prompt(50)


# ---------------------------------------------------------------------------
# Identity resolution (pure)
# ---------------------------------------------------------------------------

class TestResolveWearer:
    def test_self_speaker_label_alone_is_unknown(self):
        """The 'Speaker A = self' shortcut is gone."""
        ctx = SessionContext(session_id="s")
        ctx.self_speaker = "Speaker A"
        assert _resolve_wearer(ctx, _utt("hi", "Speaker A"), None) == WEARER_UNKNOWN
        assert _resolve_wearer(ctx, _utt("hi", "Speaker B"), None) == WEARER_UNKNOWN

    def test_wearer_known_config_confirms_self_speaker(self):
        ctx = SessionContext(session_id="s")
        asyncio.run(audio_pipeline._apply_config(
            ctx, {"self_speaker": "Speaker B", "wearer_known": True},
        ))
        assert _resolve_wearer(ctx, _utt("hi", "Speaker B"), None) == WEARER_SELF
        assert _resolve_wearer(ctx, _utt("hi", "Speaker A"), None) == WEARER_OTHER
        asyncio.run(audio_pipeline._apply_config(ctx, {"wearer_known": False}))
        assert _resolve_wearer(ctx, _utt("hi", "Speaker B"), None) == WEARER_UNKNOWN

    def test_wearer_known_rejects_non_bool(self):
        ctx = SessionContext(session_id="s")
        asyncio.run(audio_pipeline._apply_config(
            ctx, {"self_speaker": "Speaker A", "wearer_known": "yes"},
        ))
        assert ctx.wearer_known is False

    def test_turn_verdict_is_remembered_for_later_label_only_turns(self):
        ctx = SessionContext(session_id="s")
        assert _resolve_wearer(ctx, _utt("hi", "Speaker B"), True) == WEARER_SELF
        # A later turn from the same label without a verdict: still self.
        assert _resolve_wearer(ctx, _utt("more", "Speaker B"), None) == WEARER_SELF
        # The wearer is known, so any other label is someone else.
        assert _resolve_wearer(ctx, _utt("hey", "Speaker A"), None) == WEARER_OTHER

    def test_user_tapping_a_label_as_self_confirms_it(self):
        ctx = SessionContext(session_id="s")
        apply_speaker_label(ctx, {"speaker": "Speaker B", "display_name": "Me", "is_self": True})
        assert _resolve_wearer(ctx, _utt("hi", "Speaker B"), None) == WEARER_SELF
        assert _resolve_wearer(ctx, _utt("hi", "Speaker A"), None) == WEARER_OTHER


# ---------------------------------------------------------------------------
# Server-side voiceprint confirmation (Deepgram path — no phone turns)
# ---------------------------------------------------------------------------

class _Store:
    async def list_voiceprints(self, uid):
        return [{"person_id": "self", "is_self": True, "display_name": "Me",
                 "embedding": [0.1, 0.2, 0.3]}]


class TestServerVoiceprint:
    def _ctx(self) -> SessionContext:
        ctx = SessionContext(session_id="s", uid="u1")
        ctx.pcm.append(b"\x01\x00" * 16000 * 3)  # 3 s of audio in the ring
        return ctx

    def _stub(self, monkeypatch, verdicts: dict[str, dict]):
        stub = MagicMock()
        stub.is_available = lambda: True
        stub.MIN_MATCH_SECONDS = 1.0
        monkeypatch.setattr(audio_pipeline, "speaker_id", stub)
        monkeypatch.setattr(audio_pipeline, "SLICE_GRACE_S", 0.0)
        monkeypatch.setattr(
            audio_pipeline, "_identify_turn_person",
            lambda pcm, sr, speaker, docs: verdicts.get(speaker),
        )

    def test_server_match_confirms_the_wearer(self, monkeypatch):
        self._stub(monkeypatch, {"Speaker B": {
            "matched_person_id": "self", "is_self": True, "display_name": "Me",
            "scores": {"self": 0.81},
        }})
        ctx = self._ctx()
        sent: list[dict] = []

        async def send(p):
            sent.append(p)

        asyncio.run(audio_pipeline._confirm_wearer_by_voiceprint(
            ctx, "Speaker B", 0.0, 2.0, send, _Store(),
        ))
        assert _resolve_wearer(ctx, _utt("hi", "Speaker B"), None) == WEARER_SELF
        assert _resolve_wearer(ctx, _utt("hi", "Speaker A"), None) == WEARER_OTHER
        assert sent and sent[0]["type"] == "speaker_identity" and sent[0]["is_self"] is True

    def test_no_match_leaves_the_wearer_unknown(self, monkeypatch):
        self._stub(monkeypatch, {"Speaker A": {
            "matched_person_id": None, "is_self": False, "scores": {"self": 0.31},
        }})
        ctx = self._ctx()

        async def send(p):
            pass

        asyncio.run(audio_pipeline._confirm_wearer_by_voiceprint(
            ctx, "Speaker A", 0.0, 2.0, send, _Store(),
        ))
        assert _resolve_wearer(ctx, _utt("hi", "Speaker A"), None) == WEARER_UNKNOWN

    def test_no_enrolled_print_is_a_clean_skip(self, monkeypatch):
        self._stub(monkeypatch, {})
        ctx = self._ctx()

        class Empty:
            async def list_voiceprints(self, uid):
                return []

        async def send(p):
            raise AssertionError("nothing to send")

        asyncio.run(audio_pipeline._confirm_wearer_by_voiceprint(
            ctx, "Speaker A", 0.0, 2.0, send, Empty(),
        ))
        assert _resolve_wearer(ctx, _utt("hi", "Speaker A"), None) == WEARER_UNKNOWN


# ---------------------------------------------------------------------------
# The dinner, replayed through the real WebSocket pipeline
# ---------------------------------------------------------------------------

# Son speaks first (→ "Speaker A"), wearer asks questions (→ "Speaker B").
DINNER = [
    (0, "We had a quiz in math today.", 0.0, 2.5),
    (1, "Oh yeah? How did it go?", 3.0, 4.5),
    (0, "I think I did okay, the fractions part was hard.", 5.0, 8.0),
    (1, "Which part of the fractions was hardest?", 8.5, 10.5),
    (0, "Dividing them. And then at recess we played tag.", 11.0, 14.0),
]


class _OneSegmentPerFrame:
    def __init__(self, segments):
        self._segments = list(segments)

    async def connect(self):
        pass

    async def stream(self, audio_bytes):
        return [self._segments.pop(0)] if self._segments else []

    async def close(self):
        pass


def _neutral_llm(system: str, user: str, **_) -> str:
    return json.dumps({"suggestions": ["slow down", "ask an open question", "let them finish"],
                       "importance": 60})


class TestDinnerReplay:
    @pytest.fixture
    def env(self, monkeypatch):
        from fastapi.testclient import TestClient

        from tests.test_audio_pipeline import FakeTTS, _clear_overrides, open_ws

        monkeypatch.setattr(audio_pipeline, "watch_relay", None)
        monkeypatch.setattr(audio_pipeline, "speaker_id", None, raising=False)
        _clear_overrides()
        segments = [TranscriptSegment(text=t, start_time=s, end_time=e, speaker=spk)
                    for spk, t, s, e in DINNER]
        app.state.transcriber_factory = lambda: _OneSegmentPerFrame(segments)
        app.state.tts_client = FakeTTS()
        llm = MagicMock()
        llm.model = "fake"
        del llm.stream_complete
        llm.complete = MagicMock(side_effect=_neutral_llm)
        app.state.llm_client = llm
        try:
            yield TestClient(app), llm, open_ws
        finally:
            _clear_overrides()

    def _drive(self, ws, n):
        events = []
        for _ in range(n):
            ws.send_bytes(b"\x00" * 3200)
            while True:
                msg = json.loads(ws.receive_text())
                events.append(msg)
                if msg.get("type") in ("suggestion", "suggestion_error"):
                    break
        return events

    def test_unconfirmed_speaker_a_is_never_coached_as_the_wearer(self, env):
        client, llm, open_ws = env
        with open_ws(client, "/ws/session/5d1a7c00-0000-4000-8000-00000000d1e1") as ws:
            # Exactly what the phone sends today: the "Speaker A" default.
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 50,
                                     "self_speaker": "Speaker A"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            events = self._drive(ws, len(DINNER))

        suggestions = [e for e in events if e["type"] == "suggestion"]
        assert len(suggestions) == len(DINNER)
        # No turn was treated as the wearer's own (no delivery nudges).
        assert all(e.get("kind") != "nudge" for e in suggestions)

        calls = llm.complete.call_args_list
        assert len(calls) == len(DINNER)
        for c in calls:
            system, user = c.kwargs["system"], c.kwargs["user"]
            assert system == unknown_wearer_prompt(50)
            assert COACH_UNKNOWN_WEARER_RULES in system
            low = system.lower()
            for word in RELATIONSHIP_WORDS:
                assert f" {word}" not in low
            assert "role in this conversation" not in low
            # The history never calls an unconfirmed label "You".
            assert "- You:" not in user
            assert "not yet identified" in user

    def test_coaching_personalizes_once_the_wearer_is_confirmed(self, env):
        client, llm, open_ws = env
        with open_ws(client, "/ws/session/5d1a7c00-0000-4000-8000-00000000d1e2") as ws:
            ws.send_text(json.dumps({"type": "config", "self_speaker": "Speaker A"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            self._drive(ws, 2)  # son, then wearer — both unknown
            # Mid-stream: the wearer taps "Speaker B" as themselves.
            ws.send_text(json.dumps({"type": "speaker_label", "speaker": "Speaker B",
                                     "display_name": "Me", "is_self": True}))
            while json.loads(ws.receive_text())["type"] != "speaker_label_ack":
                pass
            llm.complete.side_effect = lambda system, user, **_: (
                json.dumps({"nudge": "", "importance": 0})
                if "nudge" in system and "Produce ONE nudge" in system
                else _neutral_llm(system, user)
            )
            ws.send_bytes(b"\x00" * 3200)   # son (other) → suggestion
            while json.loads(ws.receive_text())["type"] != "suggestion":
                pass
            ws.send_bytes(b"\x00" * 3200)   # wearer (self) → nudge prompt (silent)
            ws.send_text(json.dumps({"type": "stop"}))   # drains the worker
            while json.loads(ws.receive_text())["type"] != "session_complete":
                pass

        systems = [c.kwargs["system"] for c in llm.complete.call_args_list]
        assert systems[0] == unknown_wearer_prompt(50)
        assert systems[1] == unknown_wearer_prompt(50)
        assert systems[2] == empathy_system_prompt(50, live=True)
        assert systems[3] == self_feedback_prompt(50)
        # The history now names the confirmed wearer "You" — and only them.
        last_user = llm.complete.call_args_list[3].kwargs["user"]
        assert '- You: "Oh yeah? How did it go?"' in last_user
        assert '- Speaker A: "We had a quiz in math today."' in last_user

