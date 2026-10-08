"""REST surface of the coach knowledge library — /library/items.

Runs the real router + LibraryService against the in-memory store/blobs and
the deterministic test embedder. Auth is the suite's standard ``X-Test-Uid``
override of ``auth.get_current_uid``.
"""

import pytest
from httpx import ASGITransport, AsyncClient

import library
from library import models
from library.blobs import MemoryLibraryBlobs
from library.service import LibraryService
from library.store import MemoryLibraryStore
from library_fakes import FakeEmbedder
from main import app

pytestmark = pytest.mark.anyio

A = {"X-Test-Uid": "user-a"}
B = {"X-Test-Uid": "user-b"}


@pytest.fixture
def svc():
    s = LibraryService(MemoryLibraryStore(), MemoryLibraryBlobs(), FakeEmbedder())
    library.set_service(s)
    yield s
    library.set_service(None)


@pytest.fixture
async def client(svc):
    import main

    main._rate_limiter.reset()  # the per-IP budget is process-wide
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def _note(client, headers=A, title="Pitch", text="We save teams 10 hours a week."):
    r = await client.post("/library/items", json={"kind": "note", "title": title, "text": text}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# create / read / update / delete
# ---------------------------------------------------------------------------

async def test_create_note_returns_library_item_shape(client, svc):
    item = await _note(client)
    assert set(item) >= {"id", "title", "kind", "chars", "status", "created_at"}
    assert item["kind"] == "note"
    assert item["title"] == "Pitch"
    assert item["chars"] == len("We save teams 10 hours a week.")
    assert item["status"] in {"processing", "ready"}
    await svc.drain()
    r = await client.get(f"/library/items/{item['id']}", headers=A)
    assert r.json()["status"] == "ready"


async def test_upload_document_and_table(client, svc):
    r = await client.post(
        "/library/items", headers=A,
        files={"file": ("raise-notes.md", b"# Ask\nI led the migration.", "text/markdown")},
    )
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["kind"] == "document" and doc["title"] == "raise-notes"

    r = await client.post(
        "/library/items", headers=A, data={"title": "Price list"},
        files={"file": ("prices.csv", b"Plan,Price\nStarter,49\n", "text/csv")},
    )
    assert r.status_code == 201, r.text
    table = r.json()
    assert table["kind"] == "table" and table["title"] == "Price list"
    await svc.drain()

    # Original file kept byte-for-byte in the blob tier.
    assert svc.blobs._data[f"library/user-a/{table['id']}/original.csv"] == b"Plan,Price\nStarter,49\n"

    r = await client.get(f"/library/items/{table['id']}", headers=A)
    assert r.status_code == 200
    detail = r.json()
    assert "Plan: Starter; Price: 49" in detail["preview"]
    assert detail["filename"] == "prices.csv"
    assert detail["preview_truncated"] is False


async def test_list_is_oldest_first_and_reports_capabilities(client, svc):
    first = await _note(client, title="One")
    second = await _note(client, title="Two")
    r = await client.get("/library/items", headers=A)
    assert r.status_code == 200
    body = r.json()
    assert [i["id"] for i in body["items"]] == [first["id"], second["id"]]
    assert body["retrieval_available"] is True
    assert body["limits"] == {
        "max_file_bytes": models.MAX_FILE_BYTES,
        "max_text_chars": models.MAX_TEXT_CHARS,
        "max_items": models.MAX_ITEMS_PER_USER,
    }


async def test_preview_is_capped_but_flagged(client):
    item = await _note(client, text="x" * (models.PREVIEW_CHARS + 50))
    r = await client.get(f"/library/items/{item['id']}", headers=A)
    body = r.json()
    assert len(body["preview"]) == models.PREVIEW_CHARS
    assert body["preview_truncated"] is True
    assert body.get("text") is None  # full text only on request


async def test_full_text_is_opt_in_and_owner_only(client):
    long_text = "Line of a long note. " * 500  # ~10,500 chars, past the preview
    item = await _note(client, text=long_text)
    iid = item["id"]
    r = await client.get(f"/library/items/{iid}?full=1", headers=A)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == long_text.strip()
    assert len(body["preview"]) == models.PREVIEW_CHARS and body["preview_truncated"] is True
    # Another user gets the same 404 with or without ?full.
    assert (await client.get(f"/library/items/{iid}?full=1", headers=B)).status_code == 404
    # full=0 behaves like no flag.
    assert (await client.get(f"/library/items/{iid}?full=0", headers=A)).json().get("text") is None


async def test_patch_renames_and_edits_note_text(client, svc):
    item = await _note(client)
    r = await client.patch(f"/library/items/{item['id']}", headers=A, json={"title": "New", "text": "Fresh text"})
    assert r.status_code == 200, r.text
    assert r.json()["title"] == "New" and r.json()["chars"] == len("Fresh text")
    await svc.drain()
    detail = (await client.get(f"/library/items/{item['id']}", headers=A)).json()
    assert detail["preview"] == "Fresh text" and detail["status"] == "ready"


async def test_patch_text_of_a_document_is_refused(client):
    r = await client.post("/library/items", headers=A, files={"file": ("a.txt", b"abc", "text/plain")})
    r = await client.patch(f"/library/items/{r.json()['id']}", headers=A, json={"text": "nope"})
    assert r.status_code == 422


async def test_delete_removes_blob_doc_and_vectors(client, svc):
    r = await client.post("/library/items", headers=A, files={"file": ("a.txt", b"some text here", "text/plain")})
    item_id = r.json()["id"]
    await svc.drain()
    assert any(c.item_id == item_id for c in svc.store._chunks.values())

    r = await client.delete(f"/library/items/{item_id}", headers=A)
    assert r.status_code == 200 and r.json() == {"deleted": True, "id": item_id}
    assert not any(c.item_id == item_id for c in svc.store._chunks.values())
    assert not any(item_id in k for k in svc.blobs._data)
    assert (await client.get(f"/library/items/{item_id}", headers=A)).status_code == 404
    assert (await client.delete(f"/library/items/{item_id}", headers=A)).status_code == 404


# ---------------------------------------------------------------------------
# per-user isolation
# ---------------------------------------------------------------------------

async def test_user_b_cannot_read_list_patch_or_delete_user_a_items(client, svc):
    item = await _note(client, headers=A, text="A's private pitch")
    iid = item["id"]
    assert (await client.get("/library/items", headers=B)).json()["items"] == []
    assert (await client.get(f"/library/items/{iid}", headers=B)).status_code == 404
    assert (await client.patch(f"/library/items/{iid}", headers=B, json={"title": "pwned"})).status_code == 404
    assert (await client.delete(f"/library/items/{iid}", headers=B)).status_code == 404
    # Still intact for A.
    r = await client.get(f"/library/items/{iid}", headers=A)
    assert r.status_code == 200 and r.json()["title"] == "Pitch"


async def test_requires_auth(client):
    import auth

    override = app.dependency_overrides.pop(auth.get_current_uid)
    try:
        assert (await client.get("/library/items")).status_code == 401
        r = await client.post("/library/items", json={"kind": "note", "title": "t", "text": "x"})
        assert r.status_code == 401
    finally:
        app.dependency_overrides[auth.get_current_uid] = override


async def test_malformed_item_id_is_404_not_a_path_trick(client):
    assert (await client.get("/library/items/..%2F..%2Fx", headers=A)).status_code == 404
    assert (await client.get("/library/items/not-a-uuid", headers=A)).status_code == 404


# ---------------------------------------------------------------------------
# limits — clear errors, nothing stored, nothing truncated
# ---------------------------------------------------------------------------

async def test_file_over_10mb_is_413(client, svc):
    big = b"a" * (models.MAX_FILE_BYTES + 1)
    r = await client.post("/library/items", headers=A, files={"file": ("big.txt", big, "text/plain")})
    assert r.status_code == 413
    assert "10 MB" in r.json()["detail"]
    assert svc.store._items == {} and svc.blobs._data == {}


async def test_extracted_text_over_2mb_is_413_not_truncated(client, svc):
    text = ("word " * (models.MAX_TEXT_CHARS // 5 + 10)).encode()
    assert len(text) < models.MAX_FILE_BYTES
    r = await client.post("/library/items", headers=A, files={"file": ("long.txt", text, "text/plain")})
    assert r.status_code == 413
    assert "limit" in r.json()["detail"]
    assert svc.store._items == {}

    r = await client.post("/library/items", headers=A, json={"kind": "note", "title": "t", "text": "x" * (models.MAX_TEXT_CHARS + 1)})
    assert r.status_code == 413


async def test_fifty_item_cap_is_422(client, svc, monkeypatch):
    monkeypatch.setattr("library.service.MAX_ITEMS_PER_USER", 3)
    for i in range(3):
        await _note(client, title=f"n{i}")
    r = await client.post("/library/items", headers=A, json={"kind": "note", "title": "four", "text": "x"})
    assert r.status_code == 422
    assert "full" in r.json()["detail"]
    # Another user is unaffected by A's cap.
    await _note(client, headers=B)


@pytest.mark.parametrize("payload", [
    {"kind": "note", "title": "", "text": "x"},
    {"kind": "note", "title": "t", "text": "   "},
    {"kind": "document", "title": "t", "text": "x"},
    {"title": "t"},
])
async def test_bad_note_payloads_are_422(client, payload):
    r = await client.post("/library/items", headers=A, json=payload)
    assert r.status_code == 422


async def test_unsupported_and_unreadable_files_are_422(client, svc):
    r = await client.post("/library/items", headers=A, files={"file": ("x.png", b"\x89PNG", "image/png")})
    assert r.status_code == 422 and "unsupported" in r.json()["detail"]
    r = await client.post("/library/items", headers=A, files={"file": ("x.pdf", b"%PDF-junk", "application/pdf")})
    assert r.status_code == 422
    r = await client.post("/library/items", headers=A, files={"file": ("x.txt", b"", "text/plain")})
    assert r.status_code == 422
    assert svc.store._items == {}


async def test_no_embedder_items_still_save_and_list_says_retrieval_unavailable(client):
    s = LibraryService(MemoryLibraryStore(), MemoryLibraryBlobs(), None, embedder_reason="no creds")
    library.set_service(s)
    item = await _note(client)
    assert item["status"] == "ready"
    body = (await client.get("/library/items", headers=A)).json()
    assert body["retrieval_available"] is False
    assert body["items"][0]["indexed"] is False
