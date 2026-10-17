"""Live Coach ROOM mode — the pure parts (server/room_mode.py).

* wake-phrase detection at the start of an utterance, tolerant of STT
  spellings ("Mind shift", "mind-shift", "Hey MindShift,");
* info cards from the library: deterministic, no LLM, verbatim fact + source,
  rate-limited and de-duplicated;
* the answer prompt (library as data, conversation, admit not-knowing) and
  the answer parser.

The library block is built by the REAL library service (in-memory store), so
the parser is tested against exactly what build_library_context renders.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import library
import room_mode
import room_prompts
from library.blobs import MemoryLibraryBlobs
from library.service import LibraryService
from library.store import MemoryLibraryStore
from library_fakes import FakeEmbedder
from llm_client import CachedPrefixPrompt

ME = "test-user"

ACME_NOTE = (
    "Acme Corp account notes.\n"
    "Q3 2026: Acme bought 120 seats of the Pro plan for $48,000.\n"
    "Q2 2026: Acme renewed support for $6,000.\n"
    "Champion: Dana Ortiz, VP Operations.\n"
)
GLOBEX_NOTE = "Globex is evaluating the Starter plan; decision expected in November."


@pytest.fixture
def svc():
    s = LibraryService(MemoryLibraryStore(), MemoryLibraryBlobs(), FakeEmbedder())
    library.set_service(s)
    yield s
    library.set_service(None)


def run(coro):
    return asyncio.run(coro)


def block(svc, *notes):
    items = [run(svc.create_note(ME, title, text)) for title, text in notes]
    ctx = run(library.build_library_context(ME, [i.id for i in items], "", 6000))
    assert ctx.mode == "full"
    return ctx, items


# ---------------------------------------------------------------------------
# wake phrase
# ---------------------------------------------------------------------------

class TestWakePhrase:
    @pytest.mark.parametrize("text, question", [
        ("MindShift, what did Acme buy last quarter?", "what did Acme buy last quarter?"),
        ("Hey MindShift what's our churn?", "what's our churn?"),
        ("Mind shift, who is the champion at Acme?", "who is the champion at Acme?"),
        ("mind-shift: how many seats?", "how many seats?"),
        ("Hey, Mind Shift. When does Globex decide?", "When does Globex decide?"),
        ("Okay mindshift how many seats did they buy", "how many seats did they buy"),
        ("Mine shift, what's the renewal?", "what's the renewal?"),
        ("  MINDSHIFT ... summarize Acme", "summarize Acme"),
    ])
    def test_addressed(self, text, question):
        assert room_mode.detect_address(text) == question

    @pytest.mark.parametrize("text", ["Hey MindShift.", "MindShift?", "mind shift"])
    def test_wake_alone_is_addressed_without_a_question(self, text):
        assert room_mode.detect_address(text) == ""

    @pytest.mark.parametrize("text", [
        "We should ask MindShift about that later.",
        "Our mindset shifted after Q3.",
        "The mind shifts when you sleep.",
        "Acme bought 120 seats.",
        "",
    ])
    def test_not_addressed(self, text):
        assert room_mode.detect_address(text) is None

    def test_phrase_is_a_constant(self):
        assert room_mode.WAKE_PHRASE == "MindShift"


# ---------------------------------------------------------------------------
# library block -> facts
# ---------------------------------------------------------------------------

class TestLibraryFacts:
    def test_parses_the_real_rendered_block(self, svc):
        ctx, (acme,) = block(svc, ("Acme account", ACME_NOTE))
        facts = room_mode.facts_from_library(ctx.text)
        assert all(f.item_id == acme.id for f in facts)
        assert all(f.item_title == "Acme account" for f in facts)
        lines = [f.text for f in facts]
        assert "Q3 2026: Acme bought 120 seats of the Pro plan for $48,000." in lines
        # No markup from the wrapper leaks into a fact.
        assert not any("<" in line or "library_item" in line for line in lines)

    def test_retrieved_excerpts_parse_too(self):
        text = (
            "<wearer_library>\npreamble\n"
            '<library_item id="i1" title="Deals &amp; more" kind="note">\n'
            '<excerpt n="2">\nGlobex signed in May.\n</excerpt>\n'
            "</library_item>\n</wearer_library>\n"
        )
        facts = room_mode.facts_from_library(text)
        assert [(f.item_id, f.item_title, f.text) for f in facts] == [
            ("i1", "Deals & more", "Globex signed in May."),
        ]

    def test_empty(self):
        assert room_mode.facts_from_library("") == []


# ---------------------------------------------------------------------------
# info cards
# ---------------------------------------------------------------------------

class TestCards:
    def test_mention_produces_the_right_fact_and_source(self, svc):
        ctx, (acme, _g) = block(svc, ("Acme account", ACME_NOTE), ("Pipeline", GLOBEX_NOTE))
        state = room_mode.RoomState()
        card = state.card_for("So how did Acme do in Q3?", ctx, t=10.0)
        assert card is not None
        assert card["type"] == "room_card"
        assert card["fact"] == "Q3 2026: Acme bought 120 seats of the Pro plan for $48,000."
        assert card["source_item_id"] == acme.id
        assert card["source_title"] == "Acme account"
        assert card["title"] == "Acme · Q3"
        assert card["t"] == 10.0
        assert card["id"]

    def test_spoken_quarter_forms_match(self, svc):
        ctx, _ = block(svc, ("Acme account", ACME_NOTE))
        state = room_mode.RoomState()
        card = state.card_for("acme's third quarter was strong", ctx, t=1.0)
        assert card is not None and card["fact"].startswith("Q3 2026")

    def test_no_card_without_a_library_anchor(self, svc):
        ctx, _ = block(svc, ("Acme account", ACME_NOTE))
        state = room_mode.RoomState()
        # A period code alone, or ordinary words, never make a card.
        assert state.card_for("What happened in Q3 for us?", ctx, t=1.0) is None
        assert state.card_for("The plan is to buy lunch.", ctx, t=2.0) is None
        assert state.card_for("Let's talk about Initech.", ctx, t=3.0) is None

    def test_dedupe_and_rate_limit(self, svc):
        ctx, _ = block(svc, ("Acme account", ACME_NOTE), ("Pipeline", GLOBEX_NOTE))
        state = room_mode.RoomState()
        first = state.card_for("Acme in Q3", ctx, t=10.0)
        assert first is not None
        # Same fact again, much later: deduped for the session.
        assert state.card_for("Acme Q3 again", ctx, t=500.0) is None
        # A different fact too soon after the last card: rate-limited.
        assert state.card_for("And Globex?", ctx, t=10.0 + room_mode.CARD_MIN_GAP_S / 2) is None
        # ...and allowed once the gap has passed.
        later = state.card_for("And Globex?", ctx, t=10.0 + room_mode.CARD_MIN_GAP_S + 1)
        assert later is not None and "Globex" in later["fact"]

    def test_no_library_no_card(self):
        assert room_mode.RoomState().card_for("Acme in Q3", None, t=1.0) is None

    def test_long_fact_is_clipped(self, svc):
        long_line = "Initech " + "renewal detail " * 40
        ctx, _ = block(svc, ("Initech", long_line))
        card = room_mode.RoomState().card_for("Initech update", ctx, t=1.0)
        assert card is not None
        assert len(card["fact"]) <= room_mode.CARD_FACT_MAX_CHARS
        assert card["fact"].endswith("…")

    def test_likely_mention_prefilter(self):
        state = room_mode.RoomState()
        assert not state.likely_mention("we should get lunch soon")
        assert state.likely_mention("we should call Acme soon")
        state.known_terms.add("acme")
        assert state.likely_mention("we should call acme soon")


# ---------------------------------------------------------------------------
# answers
# ---------------------------------------------------------------------------

class FakeLLM:
    """Records the prompt; returns a scripted reply. Only for prompt/parse
    plumbing — the answer QUALITY is proven against a recorded real model in
    test_room_mode_scene.py."""

    model = "fake"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, *, system, user, max_tokens=512, temperature=0.7):
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        return self.reply


class TestAnswer:
    def test_prompt_has_library_as_cached_prefix_rules_and_conversation(self, svc):
        ctx, (acme,) = block(svc, ("Acme account", ACME_NOTE))
        llm = FakeLLM(json.dumps({
            "answer": "Acme bought 120 Pro seats for $48,000 in Q3.",
            "known": True, "source_item_ids": [acme.id, "made-up"],
        }))
        frame = run(room_mode.answer_question(
            llm, "what did Acme buy last quarter?", asker="Speaker B",
            library_ctx=ctx, conversation=["Speaker A: Let's review Acme."], t=42.0,
        ))
        call = llm.calls[0]
        assert isinstance(call["system"], CachedPrefixPrompt)
        assert call["system"].startswith("<wearer_library>")
        assert room_prompts.ROOM_LIBRARY_RULES in call["system"]
        assert room_prompts.NOT_KNOWN_LINE in call["system"]
        assert "Speaker A: Let's review Acme." in call["user"]
        assert "what did Acme buy last quarter?" in call["user"]
        assert frame == {
            "type": "room_answer",
            "question": "what did Acme buy last quarter?",
            "text": "Acme bought 120 Pro seats for $48,000 in Q3.",
            "known": True,
            "source_item_ids": [acme.id],  # ids not in the library dropped
            "t": 42.0,
            "speak": True,
        }

    def test_no_library_means_no_library_rules(self):
        llm = FakeLLM('{"answer": "' + room_prompts.NOT_KNOWN_LINE + '", "known": false}')
        frame = run(room_mode.answer_question(
            llm, "what's our ARR?", asker=None, library_ctx=None, conversation=[], t=1.0,
        ))
        assert room_prompts.ROOM_LIBRARY_RULES not in llm.calls[0]["system"]
        assert "<wearer_library>" not in llm.calls[0]["system"] + llm.calls[0]["user"]
        assert frame["known"] is False
        assert frame["text"] == room_prompts.NOT_KNOWN_LINE

    def test_retrieved_library_rides_with_the_turn(self):
        from library.models import LibraryContext
        lib = LibraryContext(
            text='<wearer_library>\n<library_item id="x" title="t" kind="note">\nEXCERPT\n</library_item>\n</wearer_library>\n',
            mode="retrieved", item_ids=["x"],
        )
        llm = FakeLLM('{"answer": "ok", "known": true}')
        run(room_mode.answer_question(llm, "q", asker=None, library_ctx=lib, conversation=[], t=1.0))
        assert "EXCERPT" in llm.calls[0]["user"]
        assert "EXCERPT" not in llm.calls[0]["system"]

    @pytest.mark.parametrize("raw, text, known", [
        ('```json\n{"answer": "Two hundred.", "known": true}\n```', "Two hundred.", True),
        ('{"answer": "I don\'t have that in the library.", "known": "maybe"}',
         "I don't have that in the library.", False),
        ('{"answer": "One. Two. Three. Four."}', "One. Two.", True),
    ])
    def test_parse(self, raw, text, known):
        out = room_mode.parse_answer(raw)
        assert out["text"] == text and out["known"] is known

    @pytest.mark.parametrize("raw", ["", "not json", '{"answer": ""}', '{"known": true}', "[]"])
    def test_unparseable_raises_unavailable(self, raw):
        with pytest.raises(room_mode.RoomAnswerUnavailable):
            room_mode.parse_answer(raw)
