"""``GET|HEAD /models/ecapa.onnx`` (routers/models.py) — torch-free.

The endpoint hands the phone the ONNX export of the pinned ECAPA speaker
embedder so on-device speaker-ID can run in the server's embedding space.
These tests never load torch: the exporter is a module attribute
(``models._export_onnx``) replaced with a fake that writes bytes, and
``speaker_id.is_available`` is monkeypatched per case. Covered:

* auth — the real dependency rejects a missing bearer token (401);
* the honest 503 when there is no file and no voice deps (the reason names
  both facts; nothing is written);
* the generated-once path: the first request exports (under the lock), the
  second serves the same bytes with NO second export; a pre-existing file
  is served without ever consulting the deps (a torch-less image that ships
  the file is a supported deploy);
* the caching contract — ``ETag`` = the quoted revision, ``Cache-Control``
  a day, ``Content-Length`` the file size, ``If-None-Match`` -> 304 (also
  weak / list / ``*`` forms, and BEFORE any export), a mismatch -> 200;
* ``HEAD`` — same headers, empty body, and it too triggers/reuses the export;
* an exporter that raises, or that produces a too-small file -> 503 and no
  usable file left behind; ``MINDSHIFT_ECAPA_ONNX_PATH`` is honored.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

import speaker_id
from auth import get_current_uid
from main import app, init_db
from routers import models

pytestmark = pytest.mark.anyio

H = {"X-Test-Uid": "u1"}
FAKE_MODEL = b"ONNX" + bytes(range(256)) * 8  # 2052 bytes: over MIN_ONNX_BYTES


@pytest.fixture
async def client():
    await init_db()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the cache (and thus the default export path) at a fresh tmp dir
    and make sure no override path leaks in from the environment."""
    monkeypatch.setenv("MINDSHIFT_ECAPA_CACHE", str(tmp_path))
    monkeypatch.delenv("MINDSHIFT_ECAPA_ONNX_PATH", raising=False)
    return tmp_path


def _default_path(cache_dir: Path) -> Path:
    return cache_dir / f"ecapa_{speaker_id.ECAPA_REVISION}.onnx"


def _fake_exporter(calls: list, payload: bytes = FAKE_MODEL):
    def _export(path: Path) -> Path:
        calls.append(Path(path))
        Path(path).write_bytes(payload)
        return Path(path)
    return _export


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------

async def test_requires_auth(client, cache_dir, monkeypatch):
    monkeypatch.delitem(app.dependency_overrides, get_current_uid)
    res = await client.get("/models/ecapa.onnx")
    assert res.status_code == 401
    res = await client.head("/models/ecapa.onnx")
    assert res.status_code == 401
    # Even a revalidation is gated: no anonymous probing of the revision.
    res = await client.get("/models/ecapa.onnx", headers={"If-None-Match": models.ecapa_etag()})
    assert res.status_code == 401


# ---------------------------------------------------------------------------
# 503 — absent and not producible
# ---------------------------------------------------------------------------

async def test_503_when_no_file_and_no_voice_deps(client, cache_dir, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: False)
    calls: list = []
    monkeypatch.setattr(models, "_export_onnx", _fake_exporter(calls))
    res = await client.get("/models/ecapa.onnx", headers=H)
    assert res.status_code == 503
    detail = res.json()["detail"]
    assert "torch" in detail and str(_default_path(cache_dir)) in detail
    assert calls == []  # never tried to export without the deps
    assert not _default_path(cache_dir).exists()  # and nothing fabricated
    # HEAD can't carry a body: the reason rides in a header.
    res = await client.head("/models/ecapa.onnx", headers=H)
    assert res.status_code == 503
    assert "torch" in res.headers["x-model-unavailable"]


async def test_503_when_export_raises_and_nothing_left_behind(client, cache_dir, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)

    def _boom(path: Path) -> Path:
        raise RuntimeError("tracing blew up")

    monkeypatch.setattr(models, "_export_onnx", _boom)
    res = await client.get("/models/ecapa.onnx", headers=H)
    assert res.status_code == 503
    assert "tracing blew up" in res.json()["detail"]
    assert not _default_path(cache_dir).exists()


async def test_503_when_export_yields_unusable_file(client, cache_dir, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    calls: list = []
    monkeypatch.setattr(models, "_export_onnx", _fake_exporter(calls, payload=b"tiny"))
    res = await client.get("/models/ecapa.onnx", headers=H)
    assert res.status_code == 503
    assert "no usable file" in res.json()["detail"]


async def test_503_maps_speaker_id_unavailable_reason(client, cache_dir, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)

    def _unavailable(path: Path) -> Path:
        raise speaker_id.SpeakerIdUnavailable("could not load speaker-embedding model")

    monkeypatch.setattr(models, "_export_onnx", _unavailable)
    res = await client.get("/models/ecapa.onnx", headers=H)
    assert res.status_code == 503
    assert res.json()["detail"] == "could not load speaker-embedding model"


# ---------------------------------------------------------------------------
# generated once, then served
# ---------------------------------------------------------------------------

async def test_generates_once_then_serves_with_cache_headers(client, cache_dir, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    calls: list = []
    monkeypatch.setattr(models, "_export_onnx", _fake_exporter(calls))

    first = await client.get("/models/ecapa.onnx", headers=H)
    assert first.status_code == 200, first.text
    assert first.content == FAKE_MODEL
    assert calls == [_default_path(cache_dir)]
    assert first.headers["etag"] == f'"{speaker_id.ECAPA_REVISION}"'
    assert first.headers["cache-control"] == "private, max-age=86400"
    assert int(first.headers["content-length"]) == len(FAKE_MODEL)
    assert first.headers["content-type"] == "application/octet-stream"

    second = await client.get("/models/ecapa.onnx", headers=H)
    assert second.status_code == 200
    assert second.content == FAKE_MODEL
    assert len(calls) == 1  # served from disk, not re-exported


async def test_preexisting_file_is_served_without_voice_deps(client, cache_dir, monkeypatch):
    """A torch-less image that ships the export (or an operator pre-warm) is
    a supported deploy: the file is served, the deps are never consulted."""
    _default_path(cache_dir).write_bytes(FAKE_MODEL)
    monkeypatch.setattr(speaker_id, "is_available", lambda: False)
    monkeypatch.setattr(models, "_export_onnx", _fake_exporter([]))
    res = await client.get("/models/ecapa.onnx", headers=H)
    assert res.status_code == 200
    assert res.content == FAKE_MODEL


async def test_env_override_path_is_honored(client, cache_dir, tmp_path, monkeypatch):
    custom = tmp_path / "elsewhere" / "model.onnx"
    custom.parent.mkdir()
    custom.write_bytes(FAKE_MODEL + b"custom")
    monkeypatch.setenv("MINDSHIFT_ECAPA_ONNX_PATH", str(custom))
    monkeypatch.setattr(speaker_id, "is_available", lambda: False)
    res = await client.get("/models/ecapa.onnx", headers=H)
    assert res.status_code == 200
    assert res.content == FAKE_MODEL + b"custom"
    assert not _default_path(cache_dir).exists()


async def test_ensure_ecapa_onnx_is_serialized_under_the_lock(cache_dir, monkeypatch):
    """Two callers racing on a cold cache run ONE export: the second waits on
    the lock, then finds the file. Driven synchronously (the lock is a
    threading lock; the router calls this via asyncio.to_thread)."""
    import threading

    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    calls: list = []
    started = threading.Event()
    release = threading.Event()

    def _slow_export(path: Path) -> Path:
        calls.append(path)
        started.set()
        release.wait(timeout=5)
        Path(path).write_bytes(FAKE_MODEL)
        return Path(path)

    monkeypatch.setattr(models, "_export_onnx", _slow_export)
    results: list = []
    t1 = threading.Thread(target=lambda: results.append(models.ensure_ecapa_onnx()))
    t1.start()
    assert started.wait(timeout=5)
    t2 = threading.Thread(target=lambda: results.append(models.ensure_ecapa_onnx()))
    t2.start()
    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert results == [_default_path(cache_dir)] * 2
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# revalidation: If-None-Match -> 304, HEAD
# ---------------------------------------------------------------------------

async def test_if_none_match_304_without_touching_the_export(client, cache_dir, monkeypatch):
    """The phone's launch re-check answers from the revision alone: no file
    read, no export, even on a cold cache with no deps."""
    monkeypatch.setattr(speaker_id, "is_available", lambda: False)
    calls: list = []
    monkeypatch.setattr(models, "_export_onnx", _fake_exporter(calls))
    etag = models.ecapa_etag()
    for header in (etag, f"W/{etag}", f'"stale", {etag}', "*"):
        res = await client.get("/models/ecapa.onnx", headers={**H, "If-None-Match": header})
        assert res.status_code == 304, header
        assert res.headers["etag"] == etag
        assert res.headers["cache-control"] == "private, max-age=86400"
        assert res.content == b""
        res = await client.head("/models/ecapa.onnx", headers={**H, "If-None-Match": header})
        assert res.status_code == 304, header
    assert calls == [] and not _default_path(cache_dir).exists()


async def test_if_none_match_mismatch_serves_the_file(client, cache_dir, monkeypatch):
    _default_path(cache_dir).write_bytes(FAKE_MODEL)
    monkeypatch.setattr(speaker_id, "is_available", lambda: False)
    res = await client.get("/models/ecapa.onnx", headers={**H, "If-None-Match": '"olderrevision"'})
    assert res.status_code == 200
    assert res.content == FAKE_MODEL
    assert res.headers["etag"] == models.ecapa_etag()


async def test_head_has_headers_no_body_and_reuses_export(client, cache_dir, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    calls: list = []
    monkeypatch.setattr(models, "_export_onnx", _fake_exporter(calls))
    res = await client.head("/models/ecapa.onnx", headers=H)
    assert res.status_code == 200
    assert res.content == b""
    assert res.headers["etag"] == models.ecapa_etag()
    assert int(res.headers["content-length"]) == len(FAKE_MODEL)
    assert res.headers["cache-control"] == "private, max-age=86400"
    assert len(calls) == 1  # HEAD on a cold cache produces the file once…
    res = await client.get("/models/ecapa.onnx", headers=H)
    assert res.status_code == 200 and res.content == FAKE_MODEL
    assert len(calls) == 1  # …and the GET reuses it


def test_if_none_match_matching_rules():
    etag = '"abc"'
    assert models.if_none_match_matches('"abc"', etag)
    assert models.if_none_match_matches('W/"abc"', etag)
    assert models.if_none_match_matches('"x", "abc"', etag)
    assert models.if_none_match_matches("*", etag)
    assert not models.if_none_match_matches('"abd"', etag)
    assert not models.if_none_match_matches("abc", etag)  # unquoted is not a tag
    assert not models.if_none_match_matches(None, etag)
    assert not models.if_none_match_matches("", etag)


def test_openapi_documents_the_route():
    spec = app.openapi()
    path = spec["paths"]["/models/ecapa.onnx"]
    assert "get" in path and "head" in path
    assert "503" in path["get"]["responses"] and "304" in path["get"]["responses"]


# ---------------------------------------------------------------------------
# review 2026-08-24: concurrent cold requests must not each occupy a worker
# thread for the whole export
# ---------------------------------------------------------------------------

async def test_concurrent_cold_requests_use_one_thread_and_one_export(
    client, cache_dir, monkeypatch,
):
    """Two phones hit a cold cache at once. The threading lock alone made
    the SECOND request park a default-executor worker on it for the whole
    export (tens of seconds in production) — and that executor is shared
    with the realtime WS token verification and every LLM/model call in the
    process. The handler now serializes on an event-loop lock first, so at
    most ONE request is ever inside ``ensure_ecapa_onnx`` (i.e. holds a
    thread); the other waits on the loop and then finds the file."""
    import asyncio
    import threading

    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    started = threading.Event()
    release = threading.Event()
    export_calls: list = []

    def _slow_export(path: Path) -> Path:
        export_calls.append(path)
        started.set()
        assert release.wait(timeout=5), "test never released the export"
        Path(path).write_bytes(FAKE_MODEL)
        return Path(path)

    monkeypatch.setattr(models, "_export_onnx", _slow_export)

    inflight = 0
    max_inflight = 0
    real_ensure = models.ensure_ecapa_onnx

    def _counting_ensure():
        nonlocal inflight, max_inflight
        inflight += 1
        max_inflight = max(max_inflight, inflight)
        try:
            return real_ensure()
        finally:
            inflight -= 1

    monkeypatch.setattr(models, "ensure_ecapa_onnx", _counting_ensure)

    async def _release_once_both_requests_are_in():
        # Wait until the first request is inside the export, then give the
        # second request time to reach the lock, then let the export finish.
        await asyncio.to_thread(started.wait, 5)
        await asyncio.sleep(0.05)
        release.set()

    first, second, _ = await asyncio.gather(
        client.get("/models/ecapa.onnx", headers=H),
        client.get("/models/ecapa.onnx", headers=H),
        _release_once_both_requests_are_in(),
    )
    assert first.status_code == 200 and second.status_code == 200
    assert first.content == FAKE_MODEL and second.content == FAKE_MODEL
    assert len(export_calls) == 1
    assert max_inflight == 1  # never two worker threads on the export
