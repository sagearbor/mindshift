"""Computed table FACTS in live coaching and /respond.

* Ingest stores each spreadsheet's typed form as ``table.json`` beside its
  text in the blob tier (deleted with the item; skipped, never truncated,
  when over the size cap).
* ``LibraryService.build_facts`` answers the latest turn from the selected
  tables only (uid-scoped: another user's table never answers).
* ``LiveLibrary.for_turn`` adds the ``<library_facts>`` block within the SAME
  300 ms budget; a late lookup never blocks the turn and serves the next
  turn while its question is still current; a failing one is ignored.
* The block rides with the turn (user message): a full-mode library block
  stays a byte-stable cached system prefix. LIBRARY_RULES says to use exact
  numbers only from FACTS / the library.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import library
from library import live as live_mod
from library import models
from library.blobs import MemoryLibraryBlobs
from library.models import LibraryContext, LibraryFacts
from library.service import LibraryService
from library.store import MemoryLibraryStore
from library_fakes import FakeEmbedder
from llm_client import CachedPrefixPrompt
from main import LIBRARY_RULES, app

ME, OTHER = "test-user", "someone-else"
CSV = (Path(__file__).parent / "fixtures" / "sales_2026.csv").read_bytes()
QUESTION = "What was Q3 revenue in the Northeast?"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def svc():
    s = LibraryService(MemoryLibraryStore(), MemoryLibraryBlobs(), FakeEmbedder())
    library.set_service(s)
    yield s
    library.set_service(None)


def _upload(svc, uid=ME, data=CSV, name="Sales 2026.csv"):
    async def go():
        rec = await svc.create_document(uid, name, "text/csv", data)
        await svc.drain()
        return rec
    return run(go())


# ---------------------------------------------------------------------------
# ingest + service
# ---------------------------------------------------------------------------

class TestService:
    def test_table_json_is_stored_beside_the_text_and_deleted_with_it(self, svc):
        rec = _upload(svc)
        assert rec.table_key and rec.table_key.endswith("/table.json")
        assert rec.table_key.startswith(f"library/{ME}/{rec.id}/")
        doc = json.loads(svc.blobs._data[rec.table_key])
        assert [c["name"] for c in doc["sheets"][0]["columns"]][:2] == ["Region", "Quarter"]
        run(svc.delete_item(ME, rec.id))
        assert rec.table_key not in svc.blobs._data

    def test_notes_and_documents_have_no_table(self, svc):
        note = run(svc.create_note(ME, "n", "Revenue in Q3 was great."))
        assert note.table_key is None

    def test_oversized_table_form_is_skipped_not_truncated(self, svc, monkeypatch):
        from library import service as service_mod

        monkeypatch.setattr(service_mod, "MAX_TABLE_JSON_BYTES", 100)
        rec = _upload(svc)
        assert rec.table_key is None
        assert run(svc.get_item(ME, rec.id)).status == "ready"  # still saved, as text
        assert run(svc.build_facts(ME, [rec.id], QUESTION)).text == ""

    def test_build_facts_answers_from_the_selected_table(self, svc):
        rec = _upload(svc)
        facts = run(svc.build_facts(ME, [rec.id], "hello\n" + QUESTION))
        assert "$225,500" in facts.text and "rows 5, 6, 7" in facts.text
        assert facts.item_ids == [rec.id] and facts.has_tables is True

    def test_no_question_no_facts(self, svc):
        rec = _upload(svc)
        facts = run(svc.build_facts(ME, [rec.id], "Nice to meet you."))
        assert facts.text == "" and facts.has_tables is True

    def test_selection_without_tables_says_so(self, svc):
        note = run(svc.create_note(ME, "n", "text"))
        assert run(svc.build_facts(ME, [note.id], QUESTION)).has_tables is False

    def test_user_isolation(self, svc):
        theirs = _upload(svc, uid=OTHER)
        facts = run(svc.build_facts(ME, [theirs.id], QUESTION))
        assert facts.text == "" and facts.item_ids == []


# ---------------------------------------------------------------------------
# LiveLibrary
# ---------------------------------------------------------------------------

class TestLiveLibrary:
    def test_facts_ride_beside_a_byte_stable_full_block(self, svc):
        rec = _upload(svc)

        async def go():
            lib = live_mod.LiveLibrary(ME, [rec.id])
            a = await lib.for_turn("Hi there.")
            b = await lib.for_turn("Hi there.\n" + QUESTION)
            c = await lib.for_turn("Hi there.\n" + QUESTION + "\nOkay, thanks.\nAnything else?")
            lib.close()
            return a, b, c

        a, b, c = run(go())
        assert a.mode == "full" and a.facts == ""
        assert b.mode == "full" and b.text == a.text  # the cached prefix is untouched
        assert "$225,500" in b.facts and "<library_facts>" in b.facts
        assert "<library_facts>" not in b.text
        assert c.facts == ""  # the question is no longer current

    def test_facts_only_when_the_library_text_is_unavailable(self, monkeypatch):
        # Too big for full mode and no embedder: no excerpts, but exact facts
        # still reach the turn (in the user message).
        s = LibraryService(MemoryLibraryStore(), MemoryLibraryBlobs(), None)
        library.set_service(s)
        try:
            rec = _upload(s)
            monkeypatch.setattr(live_mod, "FULL_BUDGET_TOKENS", 50)

            async def go():
                lib = live_mod.LiveLibrary(ME, [rec.id], warm=False)
                res = await lib.for_turn(QUESTION)
                lib.close()
                return res

            res = run(go())
        finally:
            library.set_service(None)
        assert res is not None and res.mode == "retrieved"
        assert res.text.startswith("<library_facts>") and "$225,500" in res.text

    def test_slow_facts_never_block_and_serve_the_next_turn(self, svc, monkeypatch):
        rec = _upload(svc)
        calls = []

        async def slow_facts(uid, ids, recent):
            calls.append(recent)
            if not recent.strip():
                return LibraryFacts(text="", has_tables=True)
            await asyncio.sleep(0.15)
            return LibraryFacts(text="<library_facts>\nLATE-FACT\n</library_facts>\n", item_ids=ids, has_tables=True)

        monkeypatch.setattr(live_mod, "build_library_facts", slow_facts)
        monkeypatch.setattr(live_mod, "TURN_TIMEOUT_S", 0.05)

        async def go():
            lib = live_mod.LiveLibrary(ME, [rec.id])
            await asyncio.sleep(0.06)  # let the warm-up finish
            t0 = asyncio.get_running_loop().time()
            first = await lib.for_turn(QUESTION)
            elapsed = asyncio.get_running_loop().time() - t0
            await asyncio.sleep(0.2)
            second = await lib.for_turn(QUESTION + "\nRight.")
            lib.close()
            return first, elapsed, second

        first, elapsed, second = run(go())
        assert elapsed < 0.12  # one shared budget, not 300 ms + 300 ms
        assert first is not None and first.facts == ""
        assert "LATE-FACT" in second.facts

    def test_failing_facts_lookup_still_coaches(self, svc, monkeypatch):
        rec = _upload(svc)

        async def boom(*_a, **_k):
            raise RuntimeError("blob store down")

        monkeypatch.setattr(live_mod, "build_library_facts", boom)

        async def go():
            lib = live_mod.LiveLibrary(ME, [rec.id])
            res = await lib.for_turn(QUESTION)
            lib.close()
            return res

        res = run(go())
        assert res is not None and res.mode == "full" and res.facts == ""

    def test_selection_without_tables_stops_asking(self, svc, monkeypatch):
        note = run(svc.create_note(ME, "Pricing", "Starter is 49 a month."))
        calls = []

        async def facts(uid, ids, recent):
            calls.append(recent)
            return LibraryFacts(text="", has_tables=False)

        monkeypatch.setattr(live_mod, "build_library_facts", facts)

        async def go():
            lib = live_mod.LiveLibrary(ME, [note.id])
            await asyncio.sleep(0.01)  # config → first turn: the warm-up has run
            for _ in range(3):
                await lib.for_turn(QUESTION)
            lib.close()

        run(go())
        assert len(calls) == 1  # the warm-up learned there is no table


# ---------------------------------------------------------------------------
# prompt placement
# ---------------------------------------------------------------------------

class TestPrompt:
    def test_rules_require_exact_numbers_from_facts(self):
        assert "<library_facts>" in LIBRARY_RULES
        assert "never estimate" in LIBRARY_RULES

    def test_ws_turn_gets_facts_in_the_user_message(self, svc):
        from test_audio_pipeline import (
            KNOWN_WEARER_ELSEWHERE, MOCK_LLM_JSON, FakeTranscriber, FakeTTS,
            _clear_overrides, open_ws, recv_skipping_transcripts,
        )

        rec = _upload(svc)
        _clear_overrides()
        llm = MagicMock()
        llm.complete.return_value = MOCK_LLM_JSON
        app.state.llm_client = llm
        app.state.transcriber_factory = lambda: FakeTranscriber(QUESTION)
        app.state.tts_client = FakeTTS()
        try:
            client = TestClient(app)
            with open_ws(client, "/ws/session/c0c0c0c0-0000-4000-8000-00000000f4c7") as ws:
                ws.send_text(json.dumps({
                    "type": "config", "library_item_ids": [rec.id], **KNOWN_WEARER_ELSEWHERE,
                }))
                json.loads(ws.receive_text())
                ws.send_bytes(b"\x00" * 50)
                assert recv_skipping_transcripts(ws)["type"] == "suggestion"
                kw = llm.complete.call_args.kwargs
        finally:
            _clear_overrides()
        system, user = kw["system"], kw["user"]
        assert isinstance(system, CachedPrefixPrompt)
        assert "<library_facts>" not in str(system.cache_prefix)
        assert "<library_facts>" in user and "$225,500" in user
        assert user.index("</library_facts>") < user.index(QUESTION)
        assert LIBRARY_RULES in str(system)

    def test_respond_gets_facts(self, svc):
        from test_audio_pipeline import MOCK_LLM_JSON

        mine = _upload(svc)
        theirs = _upload(svc, uid=OTHER, name="Their Sales.csv")
        llm = MagicMock()
        llm.complete.return_value = MOCK_LLM_JSON
        app.state.llm_client = llm
        client = TestClient(app)
        res = client.post("/respond", json={
            "transcript_turn": QUESTION, "empathy_slider": 50,
            "library_item_ids": [mine.id, theirs.id],
        })
        assert res.status_code == 200, res.text
        user = llm.complete.call_args.kwargs["user"]
        assert user.count("$225,500") == 1 and '"Sales 2026"' in user
        assert "Their Sales" not in user


def test_limits_are_unchanged():
    assert models.MAX_TEXT_CHARS == 2 * 1024 * 1024
    assert isinstance(LibraryContext(text="", mode="full").facts, str)
