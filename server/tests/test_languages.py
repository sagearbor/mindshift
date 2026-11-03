"""Coach / conversation language settings (server/languages.py + wiring)."""
import asyncio
import json
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import pytest

import audio_pipeline
import languages
import room_mode
from audio_pipeline import (
    DEEPGRAM_LIVE_PARAMS,
    SessionContext,
    _apply_config,
    _generate_nudge,
    _generate_suggestions,
    live_params,
)
from models.audio import Utterance


def run(coro):
    return asyncio.run(coro)


def test_validate_language_allowlist():
    assert languages.validate_language("es-ES") == "es-ES"
    for bad in ("auto", "xx-XX", "", None, 5, ["es-ES"]):
        assert languages.validate_language(bad) is None


def test_instruction_only_when_set():
    assert languages.language_instruction(None) is None
    assert languages.language_instruction("nope") is None
    assert languages.language_instruction("de-DE") == "Write the suggestion in German."
    assert languages.with_language("SYS", None) == "SYS"
    assert languages.with_language("SYS", "de-DE") == "SYS\n\nWrite the suggestion in German."


def test_apply_config_validates_and_acks():
    ctx = SessionContext(session_id="s")
    ack: dict = {}
    run(_apply_config(ctx, {"coach_language": "fr-FR", "conversation_language": "ja-JP"}, ack))
    assert (ctx.coach_language, ctx.conversation_language) == ("fr-FR", "ja-JP")
    assert ack["coach_language"] == "fr-FR"
    # Unknown values are ignored (-> auto); absent keys leave the setting alone.
    run(_apply_config(ctx, {"coach_language": "klingon"}, {}))
    assert ctx.coach_language is None and ctx.conversation_language == "ja-JP"
    run(_apply_config(ctx, {"interject_level": 40}, {}))
    assert ctx.conversation_language == "ja-JP"


def test_defaults_are_auto():
    ctx = SessionContext(session_id="s")
    assert ctx.coach_language is None and ctx.conversation_language is None


def test_deepgram_params_default_unchanged_and_language_added():
    assert live_params(None) == DEEPGRAM_LIVE_PARAMS
    assert "language" not in live_params(None)
    assert live_params("es-ES")["language"] == "es"
    assert live_params("zh-CN")["model"] == "nova-3"
    assert live_params("unknown") == DEEPGRAM_LIVE_PARAMS  # falls back, logged


def test_transcriber_url_carries_language():
    t = audio_pipeline.DeepgramTranscriber(language="fr-FR")
    q = parse_qs(urlparse(f"{t._base_url}?" + __import__("urllib.parse").parse.urlencode(live_params(t.language))).query)
    assert q["language"] == ["fr"]


class _LLM:
    def __init__(self):
        self.systems = []

    def complete(self, *, system, user, max_tokens=512, temperature=0.7):
        self.systems.append(system)
        return json.dumps({"suggestions": ["a", "b", "c"], "nudge": "ease up", "importance": 50})


def _utt():
    return Utterance(session_id="s", speaker="Speaker A", text="hi", start_time=0.0, end_time=1.0)


def test_suggestion_prompt_default_identical_and_language_appended():
    llm = _LLM()
    run(_generate_suggestions(llm, _utt(), 50, "Husband"))
    run(_generate_suggestions(llm, _utt(), 50, "Husband", coach_language="es-ES"))
    default, spanish = llm.systems
    assert spanish == default + "\n\nWrite the suggestion in Spanish."


def test_nudge_prompt_default_identical_and_language_appended():
    llm = _LLM()
    run(_generate_nudge(llm, _utt(), 50, "Husband"))
    run(_generate_nudge(llm, _utt(), 50, "Husband", coach_language="it-IT"))
    default, italian = llm.systems
    assert italian == default + "\n\nWrite the suggestion in Italian."


def test_room_answer_prompt_default_identical_and_language_appended():
    class L:
        systems = []

        def complete(self, *, system, user, max_tokens=512, temperature=0.7):
            self.systems.append(system)
            return json.dumps({"answer": "ok", "known": False, "source_item_ids": []})

    llm = L()
    for lang in (None, "pt-BR"):
        run(room_mode.answer_question(llm, "q?", asker=None, library_ctx=None,
                                      conversation=[], t=1.0, coach_language=lang))
    default, pt = llm.systems
    assert pt == default + "\n\nWrite the answer in Portuguese."
