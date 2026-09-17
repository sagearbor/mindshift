"""Nudge-quality fixes in audio_pipeline (docs/research/2026-08-30-nudge-quality).

Two failure modes measured offline on the scene fixtures and the owner's
real live sessions, and the policy/prompt changes that address them:

1. The coach saw ONE transcript turn and nothing else, so it re-issued the
   same frame turn after turn and could not tell a fragment from a new
   thought -> ``_history_for_prompt`` / ``_render_history`` put the recent
   exchange and the coach's own recent lines into the user turn, with
   per-kind guidance; ``_turn_prompt`` stays byte-identical without it.
2. Nudges fired on calm / repairing self turns (praise at importance 15-25,
   "drop the apology" on a sincere apology) and lines were re-issued ->
   ``_gate_nudge`` (importance floor + repeat cooldown) and
   ``_gate_suggestions`` (demote a repeated first line).

Pure-function tests plus one WebSocket run through the real worker, so the
wiring in ``process_segment`` (history reaches the LLM on the second turn;
a repeated nudge is silenced) is covered too.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

import audio_pipeline
from audio_pipeline import (
    SessionContext,
    _coaching_overlap,
    _gate_nudge,
    _gate_suggestions,
    _history_for_prompt,
    _is_repeat_coaching,
    _remember_coaching,
    _remember_utterance,
    _render_history,
    _turn_prompt,
)
from models.audio import Utterance


def _utt(text: str, speaker: str = "Speaker A", start: float = 0.0, end: float = 2.0) -> Utterance:
    return Utterance(session_id="s", speaker=speaker, text=text, start_time=start, end_time=end)


@pytest.fixture
def ctx() -> SessionContext:
    return SessionContext(session_id="s")


# ---------------------------------------------------------------------------
# Prompt context
# ---------------------------------------------------------------------------

class TestHistoryForPrompt:
    def test_prompt_is_byte_identical_without_history(self):
        u = _utt("hi")
        assert _turn_prompt(u) == 'Transcript turn: "hi"'
        assert _turn_prompt(u, None, "Mom") == 'Transcript turn from Mom: "hi"'
        assert _turn_prompt(u, None, None, None) == 'Transcript turn: "hi"'

    def test_off_switch_returns_none(self, ctx, monkeypatch):
        monkeypatch.setattr(audio_pipeline, "COACH_CONTEXT", False)
        assert _history_for_prompt(ctx, _utt("hi"), is_self=True) is None

    def test_history_names_self_turns_you_and_others_by_display_name(self, ctx):
        a1 = _utt("You never listen.", "Speaker A", 0, 2)
        b1 = _utt("I do listen.", "Speaker B", 2, 4)
        a2 = _utt("Then prove it.", "Speaker A", 4, 6)
        ctx.speaker_labels["Speaker B"] = {"display_name": "Mom", "is_self": False}
        for u in (a1, b1, a2):
            _remember_utterance(ctx, u)
        # The phone called Speaker A self on an EARLIER frame; this frame
        # (a2) carries is_self too — both routes label A's turns "You".
        _history_for_prompt(ctx, a1, is_self=True)
        h = _history_for_prompt(ctx, a2, is_self=True)
        assert h["self"] is True
        assert [(t["who"], t["text"]) for t in h["turns"]] == [
            ("You", "You never listen."), ("Mom", "I do listen."),
        ]
        prompt = _turn_prompt(a2, None, None, h)
        assert prompt.startswith('Transcript turn: "Then prove it."')
        assert '- You: "You never listen."' in prompt
        assert '- Mom: "I do listen."' in prompt
        assert "Nudge ONLY if something about HOW" in prompt
        assert "never praise" in prompt

    def test_history_is_bounded_and_interleaves_the_coach_lines_by_time(self, ctx, monkeypatch):
        monkeypatch.setattr(audio_pipeline, "COACH_CONTEXT_TURNS", 2)
        turns = [_utt(f"turn {i}", "Speaker B", i * 2, i * 2 + 1) for i in range(5)]
        for u in turns[:4]:
            _remember_utterance(ctx, u)
        _remember_coaching(ctx, turns[1], "I hear you.", "response")
        _remember_coaching(ctx, turns[3], "Tell me more.", "response")
        _remember_utterance(ctx, turns[4])
        h = _history_for_prompt(ctx, turns[4], is_self=False)
        assert [t["text"] for t in h["turns"]] == ["turn 2", "turn 3"]
        # Only coach lines from the window on (plus always the latest one).
        assert [c["text"] for c in h["coach"]] == ["Tell me more."]
        block = _render_history(h)
        lines = block.splitlines()
        assert lines[0] == "Recent exchange before this turn (oldest first):"
        assert lines[1:4] == [
            '- Speaker B: "turn 2"',
            '- Speaker B: "turn 3"',
            '- (coach suggested the user say: "Tell me more.")',
        ]
        assert lines[-1].startswith("Each suggestion is something the user can say verbatim")

    def test_empty_history_still_carries_the_guidance(self, ctx):
        u = _utt("hi", "Speaker B")
        _remember_utterance(ctx, u)
        h = _history_for_prompt(ctx, u, is_self=None)
        assert h == {"turns": [], "coach": [], "self": False}
        assert _render_history(h).startswith("Each suggestion is something the user can say verbatim")

    def test_whispered_nudges_render_as_such(self, ctx):
        u = _utt("Fine.", "Speaker A", 0, 1)
        _remember_utterance(ctx, u)
        _remember_coaching(ctx, u, "ease up", "nudge")
        h = _history_for_prompt(ctx, _utt("Whatever.", "Speaker A", 2, 3), is_self=True)
        assert '- (coach whispered to the user: "ease up")' in _render_history(h)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

class TestRepeatGates:
    def test_overlap_is_bigram_jaccard(self):
        assert _coaching_overlap("ease up", "Ease up!") == 1.0
        assert _coaching_overlap("ease up", "let them finish") == 0.0
        assert _coaching_overlap("", "ease up") == 0.0
        assert _coaching_overlap("breathe", "breathe") == 1.0
        assert _coaching_overlap(
            "Please be specific. I need clear directions to find you.",
            "Please be specific, I need clear directions to find you now.",
        ) >= 0.5
        assert _coaching_overlap("I hear you, that sounds hard.", "I hear you — where do we meet?") < 0.5

    def test_repeat_within_cooldown_only(self, ctx, monkeypatch):
        monkeypatch.setattr(audio_pipeline, "COACH_REPEAT_COOLDOWN_S", 45.0)
        _remember_coaching(ctx, _utt("x", end=10.0), "ease up", "nudge")
        assert _is_repeat_coaching(ctx, "ease up", 20.0)
        assert _is_repeat_coaching(ctx, "ease up", 55.0)
        assert not _is_repeat_coaching(ctx, "ease up", 55.1)
        assert not _is_repeat_coaching(ctx, "ease up", 20.0, kind="response")
        assert not _is_repeat_coaching(ctx, "let them finish", 20.0)

    def test_gate_nudge_importance_floor_and_repeat(self, ctx, monkeypatch):
        monkeypatch.setattr(audio_pipeline, "NUDGE_MIN_IMPORTANCE", 40)
        u = _utt("You're right, I'm sorry.", end=30.0)
        # Praise at importance 25 (the scene fixtures' false positives).
        assert _gate_nudge(ctx, u, "good — hold that tone", 25) == ("", 0)
        assert _gate_nudge(ctx, u, "", 90) == ("", 0)
        assert _gate_nudge(ctx, u, "ease up", 72) == ("ease up", 72)
        _remember_coaching(ctx, u, "ease up", "nudge")
        later = _utt("I said ease off!", end=40.0)
        assert _gate_nudge(ctx, later, "Ease up.", 80) == ("", 0)
        assert _gate_nudge(ctx, later, "lower your volume", 80) == ("lower your volume", 80)

    def test_gate_suggestions_demotes_a_repeated_first_line(self, ctx):
        u1 = _utt("Where are you?", "Speaker B", end=4.5)
        _remember_coaching(ctx, u1, "Please be specific. I need clear directions to find you.", "response")
        u2 = _utt("I'm right here.", "Speaker B", end=7.7)
        suggestions = [
            "Please be specific — I need clear directions to find you.",
            "Which corner are you on?",
            "Stay there, I'm walking over.",
        ]
        assert _gate_suggestions(ctx, u2, suggestions) == [
            "Which corner are you on?",
            "Please be specific — I need clear directions to find you.",
            "Stay there, I'm walking over.",
        ]
        # Nothing to promote: unchanged. A single line: unchanged.
        assert _gate_suggestions(ctx, u2, ["Please be specific. I need clear directions to find you."]) == [
            "Please be specific. I need clear directions to find you.",
        ]
        assert _gate_suggestions(ctx, u2, ["Which corner are you on?", "x"]) == ["Which corner are you on?", "x"]

    def test_coaching_log_is_bounded(self, ctx, monkeypatch):
        monkeypatch.setattr(audio_pipeline, "COACHING_LOG_MAX", 10)
        for i in range(25):
            _remember_coaching(ctx, _utt("x", end=float(i)), f"line {i}", "response")
        assert len(ctx.coaching_log) <= 10
        assert ctx.coaching_log[-1]["text"] == "line 24"


# ---------------------------------------------------------------------------
# Through the real worker (WebSocket, local-first turn_local frames)
# ---------------------------------------------------------------------------

def _turn_local(text: str, *, start: float, end: float, is_self: bool | None, speaker: str = "Speaker A") -> dict:
    return {
        "type": "turn_local", "session_id": "sess-ctx", "speaker": speaker, "is_self": is_self,
        "speaker_person_id": "self" if is_self else None, "text": text,
        "start_time": start, "end_time": end, "transcript_source": "on-device",
        "prosody": None, "text_tone": None, "suggestion": None, "suggestion_source": None,
        "tts_source": "on-device",
    }


class TestThroughTheWorker:
    """The same harness shape as test_audio_pipeline.TestTurnLocal: the app's
    WS endpoint with the transcriber/TTS/LLM injected on app.state."""

    @pytest.fixture
    def env(self, monkeypatch):
        from main import app
        from tests.test_audio_pipeline import FakeTTS, StoppableTranscriber, _inject, open_ws

        monkeypatch.setattr(audio_pipeline, "watch_relay", None)
        monkeypatch.setattr(audio_pipeline, "tone_id", None, raising=False)
        monkeypatch.setattr(audio_pipeline, "speaker_id", None, raising=False)
        monkeypatch.setattr(audio_pipeline, "SLICE_GRACE_S", 0.0)
        client = _inject(StoppableTranscriber())   # installs its own LLM double…
        llm = MagicMock()
        llm.model = "fake"
        llm.complete = MagicMock()
        # No stream_complete: the plain complete() path (prompt bytes identical).
        del llm.stream_complete
        app.state.llm_client = llm                  # …replaced by ours AFTER
        app.state.tts_client = FakeTTS()
        yield client, llm, open_ws

    def test_second_turn_prompt_carries_history_and_a_repeated_nudge_is_silenced(self, env):
        client, llm, open_ws = env
        sid = "sess-ctx"

        def answer(system: str, user: str, **_) -> str:
            # Routed by the turn text (not call order: latest-wins may
            # reorder), so the model "repeats itself" on the second turn.
            if user.startswith('Transcript turn: "Well?"'):
                return json.dumps({"nudge": "let them finish", "importance": 75})
            return json.dumps({"nudge": "ease up", "importance": 80})

        llm.complete.side_effect = answer
        with open_ws(client, f"/ws/session/{sid}") as ws:
            ws.send_text(json.dumps(_turn_local("You never listen to me.", start=0.0, end=2.0, is_self=True)))
            first = json.loads(ws.receive_text())
            assert first["type"] == "suggestion" and first["kind"] == "nudge"
            assert first["suggestions"] == ["ease up"]
            # Second self turn, 5 s later: the model repeats itself -> silence
            # (no event). The third turn's different nudge proves the socket
            # was simply quiet, not stuck. The pause lets the (instant) fake
            # finish turn 2 before turn 3 arrives, so latest-wins never
            # supersedes it.
            ws.send_text(json.dumps(_turn_local("I SAID you never listen!", start=5.0, end=7.0, is_self=True)))
            time.sleep(0.5)
            ws.send_text(json.dumps(_turn_local("Well?", start=9.0, end=10.0, is_self=True)))
            third = json.loads(ws.receive_text())
            assert third["suggestions"] == ["let them finish"]
            assert third["utterance_text"] == "Well?"

        prompts = [c.kwargs["user"] for c in llm.complete.call_args_list]
        assert len(prompts) == 3
        assert prompts[0].startswith('Transcript turn: "You never listen to me."')
        assert "Recent exchange" not in prompts[0]           # nothing before the first turn
        assert "Nudge ONLY if something about HOW" in prompts[0]
        assert prompts[1].startswith('Transcript turn: "I SAID you never listen!"')
        assert '- You: "You never listen to me."' in prompts[1]
        assert '- (coach whispered to the user: "ease up")' in prompts[1]
        assert '- You: "I SAID you never listen!"' in prompts[2]

    def test_context_off_keeps_the_single_turn_prompt(self, env, monkeypatch):
        monkeypatch.setattr(audio_pipeline, "COACH_CONTEXT", False)
        client, llm, open_ws = env
        llm.complete.return_value = json.dumps({"suggestions": ["I hear you.", "Go on.", "Tell me."], "importance": 50})
        with open_ws(client, "/ws/session/sess-ctx") as ws:
            ws.send_text(json.dumps(_turn_local("Hi.", start=0.0, end=1.0, is_self=False, speaker="Speaker B")))
            json.loads(ws.receive_text())
            ws.send_text(json.dumps(_turn_local("Still there?", start=2.0, end=3.0, is_self=False, speaker="Speaker B")))
            json.loads(ws.receive_text())
        prompts = [c.kwargs["user"] for c in llm.complete.call_args_list]
        assert prompts == ['Transcript turn: "Hi."', 'Transcript turn: "Still there?"']
