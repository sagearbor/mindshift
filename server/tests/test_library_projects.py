"""Library projects (folders): nesting, depth cap, cycle rejection, moves,
safe delete, per-user isolation, backwards compatibility (no project = root)."""

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

    main._rate_limiter.reset()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def _proj(client, name="P", parent_id=None, headers=A):
    r = await client.post("/library/projects", json={"name": name, "parent_id": parent_id}, headers=headers)
    return r


async def _ok_proj(client, name="P", parent_id=None, headers=A):
    r = await _proj(client, name, parent_id, headers)
    assert r.status_code == 201, r.text
    return r.json()


async def _note(client, title="N", project_id=None, headers=A):
    body = {"kind": "note", "title": title, "text": "hello world"}
    if project_id is not None:
        body["project_id"] = project_id
    r = await client.post("/library/items", json=body, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


async def _list(client, headers=A):
    return (await client.get("/library/items", headers=headers)).json()


async def test_create_and_list_projects_and_old_items_are_root(client):
    legacy = await _note(client, "legacy")
    assert legacy["project_id"] is None
    p = await _ok_proj(client, "Clients")
    child = await _ok_proj(client, "Acme", p["id"])
    assert p["parent_id"] is None and child["parent_id"] == p["id"]
    data = await _list(client)
    assert {x["id"] for x in data["projects"]} == {p["id"], child["id"]}


async def test_rename_and_unknown_parent_and_empty_name(client):
    p = await _ok_proj(client, "A")
    r = await client.patch(f"/library/projects/{p['id']}", json={"name": "B"}, headers=A)
    assert r.status_code == 200 and r.json()["name"] == "B"
    assert (await _proj(client, "x", "11111111-1111-4111-8111-111111111111")).status_code == 404
    assert (await _proj(client, "   ")).status_code == 422


async def test_depth_cap_is_10_and_deeper_is_rejected(client):
    parent = None
    ids = []
    for i in range(models.MAX_PROJECT_DEPTH):
        p = await _ok_proj(client, f"L{i}", parent)
        ids.append(p["id"])
        parent = p["id"]
    r = await _proj(client, "too deep", parent)
    assert r.status_code == 422
    assert "10" in r.json()["detail"]


async def test_move_project_rejects_cycles_and_depth_overflow(client):
    a = await _ok_proj(client, "a")
    b = await _ok_proj(client, "b", a["id"])
    c = await _ok_proj(client, "c", b["id"])
    # into itself / into own descendant
    for target in (a["id"], b["id"], c["id"]):
        r = await client.patch(f"/library/projects/{a['id']}", json={"parent_id": target}, headers=A)
        assert r.status_code == 422, (target, r.text)
        assert "cycle" in r.json()["detail"].lower() or "itself" in r.json()["detail"].lower() \
            or "inside" in r.json()["detail"].lower()
    # valid move to root and back
    r = await client.patch(f"/library/projects/{c['id']}", json={"parent_id": None}, headers=A)
    assert r.status_code == 200 and r.json()["parent_id"] is None

    # depth overflow on move: a 10-deep chain, moving a 2-deep subtree under its bottom
    parent = None
    for i in range(models.MAX_PROJECT_DEPTH):
        parent = (await _ok_proj(client, f"d{i}", parent))["id"]
    r = await client.patch(f"/library/projects/{a['id']}", json={"parent_id": parent}, headers=A)
    assert r.status_code == 422 and "10" in r.json()["detail"]


async def test_move_item_to_project_and_back_to_root(client):
    p = await _ok_proj(client, "P")
    item = await _note(client, "n")
    r = await client.patch(f"/library/items/{item['id']}", json={"project_id": p["id"]}, headers=A)
    assert r.status_code == 200 and r.json()["project_id"] == p["id"]
    r = await client.patch(f"/library/items/{item['id']}", json={"project_id": None}, headers=A)
    assert r.status_code == 200 and r.json()["project_id"] is None
    # title-only patch leaves the project alone
    await client.patch(f"/library/items/{item['id']}", json={"project_id": p["id"]}, headers=A)
    r = await client.patch(f"/library/items/{item['id']}", json={"title": "z"}, headers=A)
    assert r.json()["project_id"] == p["id"]
    r = await client.patch(
        f"/library/items/{item['id']}",
        json={"project_id": "11111111-1111-4111-8111-111111111111"}, headers=A,
    )
    assert r.status_code == 404


async def test_upload_with_project_id(client, svc):
    p = await _ok_proj(client, "P")
    r = await client.post(
        "/library/items", headers=A, data={"project_id": p["id"], "title": "Brief"},
        files={"file": ("x.md", b"# Ask\nhello", "text/markdown")},
    )
    assert r.status_code == 201, r.text
    assert r.json()["project_id"] == p["id"] and r.json()["title"] == "Brief"
    r = await client.post(
        "/library/items", headers=A, data={"project_id": "11111111-1111-4111-8111-111111111111"},
        files={"file": ("x.md", b"# Ask\nhello", "text/markdown")},
    )
    assert r.status_code == 404


async def test_delete_nonempty_project_is_refused_unless_move_to_parent(client):
    top = await _ok_proj(client, "top")
    mid = await _ok_proj(client, "mid", top["id"])
    sub = await _ok_proj(client, "sub", mid["id"])
    it = await _note(client, "in-mid", mid["id"])
    r = await client.delete(f"/library/projects/{mid['id']}", headers=A)
    assert r.status_code == 409
    assert "move_to_parent" in r.json()["detail"]
    data = await _list(client)
    assert len(data["projects"]) == 3  # nothing deleted

    r = await client.delete(f"/library/projects/{mid['id']}?contents=move_to_parent", headers=A)
    assert r.status_code == 200 and r.json()["deleted"] is True
    data = await _list(client)
    byid = {p["id"]: p for p in data["projects"]}
    assert mid["id"] not in byid
    assert byid[sub["id"]]["parent_id"] == top["id"]
    assert [i for i in data["items"] if i["id"] == it["id"]][0]["project_id"] == top["id"]


async def test_delete_empty_project_and_root_level_move_to_parent(client):
    e = await _ok_proj(client, "empty")
    assert (await client.delete(f"/library/projects/{e['id']}", headers=A)).status_code == 200
    top = await _ok_proj(client, "top")
    it = await _note(client, "x", top["id"])
    r = await client.delete(f"/library/projects/{top['id']}?contents=move_to_parent", headers=A)
    assert r.status_code == 200
    data = await _list(client)
    assert data["projects"] == []
    assert data["items"][0]["id"] == it["id"] and data["items"][0]["project_id"] is None
    assert (await client.delete(f"/library/projects/{e['id']}", headers=A)).status_code == 404


async def test_projects_are_per_user(client):
    p = await _ok_proj(client, "mine")
    assert (await _list(client, B))["projects"] == []
    assert (await client.patch(f"/library/projects/{p['id']}", json={"name": "x"}, headers=B)).status_code == 404
    assert (await client.delete(f"/library/projects/{p['id']}", headers=B)).status_code == 404
    assert (await _proj(client, "child", p["id"], B)).status_code == 404
    item_b = await _note(client, "b", headers=B)
    r = await client.patch(f"/library/items/{item_b['id']}", json={"project_id": p["id"]}, headers=B)
    assert r.status_code == 404


async def test_delete_all_for_user_removes_projects(svc, client):
    await _ok_proj(client, "mine")
    await svc.delete_all_for_user("user-a")
    assert await svc.store.list_projects("user-a") == []
