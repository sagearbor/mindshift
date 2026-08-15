# Ported from gauge@2157433 server/tests/test_captures_api.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B7): server.main.create_app -> watch.testing.create_watch_test_app
# (keyword-only assembly); server.store.MemoryEpisodeStore -> watch.store.
# MemoryLiveSessionStore; server.blobs.MemoryBlobStore -> watch.blobs.
# MemoryBlobStore; server.captures_api.MAX_CAPTURE_BYTES/MAX_LABELS_BYTES ->
# watch.routers.captures's equivalents. The captures router already required
# a NON-LEGACY (full-auth) principal on every route in gauge (the I2/I3
# controller ruling) -- testing.py's `create_watch_test_app` always builds
# both auth dependencies and passes strict_auth_dep to make_captures_router,
# same B6 precedent as test_groups_api.py, so no behavior changed here.
import asyncio
import gzip

from fastapi.testclient import TestClient

from server.tests.watch.test_auth import StubVerifier
from watch.blobs import MemoryBlobStore
from watch.store import MemoryLiveSessionStore
from watch.testing import create_watch_test_app

TOKENS = {
    "alice-token": {"sub": "alice", "email": "alice@example.com"},
    "bob-token": {"sub": "bob", "email": "bob@example.com"},
}
A = {"Authorization": "Bearer alice-token"}
B = {"Authorization": "Bearer bob-token"}
PCM = b"\x01\x02" * 32000            # 4 s of PCM16 @ 16 kHz


class _RaisingDeleteBlobStore(MemoryBlobStore):
    """Wave B Task 8 (gauge): simulates a blob-storage delete failure (e.g. a
    transient GCS error) so DELETE /captures/{id}'s blob-first-never-orphan-
    audio ordering can be proven -- the capture's metadata must survive
    exactly this failure mode."""

    async def delete(self, key: str) -> None:
        raise RuntimeError("simulated blob delete failure")


def _client(blobs=None, blob_delete_raises=False):
    store = MemoryLiveSessionStore()
    if blobs is None:
        blobs = _RaisingDeleteBlobStore() if blob_delete_raises else MemoryBlobStore()
    # allow_legacy=True: mirrors gauge's GAUGE_ALLOW_LEGACY_ACCOUNT=true test
    # posture (and test_groups_api.py's B6 precedent) -- the captures router
    # requires full-auth regardless (see require_full_auth in watch/auth.py),
    # so this only matters for
    # test_captures_reject_legacy_account_param_even_though_flag_is_on below,
    # which needs the legacy `?account=` path to actually resolve to a
    # (rejected) legacy Principal instead of failing earlier as "no auth at
    # all".
    app = create_watch_test_app(
        store=store, verifier=StubVerifier(TOKENS), blobs=blobs, allow_legacy=True
    )
    return store, blobs, TestClient(app)


def _meta(duration=240.0, attested=True):
    return {"captured_at": "2026-08-02T09:00:00Z", "duration_s": duration,
            "trigger": "volume", "device": "pixel-watch-1", "attested": attested}


def test_create_capture_records_self_attested_consent():
    _, _, client = _client()
    body = client.post("/captures", headers=A, json=_meta()).json()
    assert body["account_id"] == "alice" and body["status"] == "awaiting_audio"
    assert body["duration_s"] == 240.0 and body["trigger"] == "volume"
    assert body["received_at"] and body["audio_uri"] is None and body["audio_bytes"] is None
    consent = body["consents"][0]
    assert consent["kind"] == "capture" and consent["attested_by"] == "alice"
    assert consent["confirmed"] is False


def test_create_without_attestation_is_422():
    _, _, client = _client()
    assert client.post("/captures", headers=A, json=_meta(attested=False)).status_code == 422


def test_create_with_truthy_string_attestation_is_422():
    _, _, client = _client()
    meta = _meta()
    meta["attested"] = "yes"
    assert client.post("/captures", headers=A, json=meta).status_code == 422


def test_create_with_absurd_duration_is_422():
    _, _, client = _client()
    assert client.post("/captures", headers=A, json=_meta(duration=100000.0)).status_code == 422
    assert client.post("/captures", headers=A, json=_meta(duration=0.0)).status_code == 422


def test_upload_audio_stores_the_blob_and_flips_status():
    _, blobs, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    resp = client.put(f"/captures/{cid}/audio", headers=A, content=PCM)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "stored" and body["audio_bytes"] == len(PCM)
    assert body["audio_uri"] == f"memory://captures/alice/{cid}.pcm"
    assert body["upload_encoding"] is None
    assert asyncio.run(blobs.get(f"captures/alice/{cid}.pcm")) == PCM


def test_gzip_upload_is_stored_decompressed():
    _, blobs, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    resp = client.put(f"/captures/{cid}/audio", headers={**A, "Content-Encoding": "gzip"},
                      content=gzip.compress(PCM))
    assert resp.status_code == 200
    assert resp.json()["upload_encoding"] == "gzip"
    assert resp.json()["audio_bytes"] == len(PCM)
    assert asyncio.run(blobs.get(f"captures/alice/{cid}.pcm")) == PCM


def test_malformed_gzip_is_422():
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    assert client.put(f"/captures/{cid}/audio", headers={**A, "Content-Encoding": "gzip"},
                      content=b"not gzip at all").status_code == 422


def test_oversized_upload_is_413():
    from watch.routers.captures import MAX_CAPTURE_BYTES
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    assert client.put(f"/captures/{cid}/audio", headers=A,
                      content=b"\x00" * (MAX_CAPTURE_BYTES + 1)).status_code == 413


def test_gzip_bomb_is_413_not_an_oom():
    from watch.routers.captures import MAX_CAPTURE_BYTES
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    bomb = gzip.compress(b"\x00" * (MAX_CAPTURE_BYTES + 1024))    # tiny on the wire
    assert len(bomb) < 100_000
    assert client.put(f"/captures/{cid}/audio", headers={**A, "Content-Encoding": "gzip"},
                      content=bomb).status_code == 413


def test_empty_upload_is_422():
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    assert client.put(f"/captures/{cid}/audio", headers=A, content=b"").status_code == 422


def test_second_upload_is_409():
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    client.put(f"/captures/{cid}/audio", headers=A, content=PCM)
    assert client.put(f"/captures/{cid}/audio", headers=A, content=PCM).status_code == 409


def test_upload_without_blob_store_is_503_and_status_unchanged():
    s = MemoryLiveSessionStore()
    c = TestClient(create_watch_test_app(store=s, verifier=StubVerifier(TOKENS), blobs=None))
    cid = c.post("/captures", headers=A, json=_meta()).json()["id"]
    assert c.put(f"/captures/{cid}/audio", headers=A, content=PCM).status_code == 503
    assert asyncio.run(s.get_capture(cid)).status == "awaiting_audio"


def test_upload_by_non_owner_is_403():
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    assert client.put(f"/captures/{cid}/audio", headers=B, content=PCM).status_code == 403


def test_unknown_capture_is_404():
    _, _, client = _client()
    assert client.put("/captures/nope/audio", headers=A, content=PCM).status_code == 404
    assert client.get("/captures/nope", headers=A).status_code == 404


def test_list_is_account_scoped_and_newest_first():
    _, _, client = _client()
    for when in ("2026-08-01T00:00:00Z", "2026-08-03T00:00:00Z", "2026-08-02T00:00:00Z"):
        meta = _meta()
        meta["captured_at"] = when
        client.post("/captures", headers=A, json=meta)
    client.post("/captures", headers=B, json=_meta())
    got = client.get("/captures", headers=A).json()
    assert [c["captured_at"] for c in got] == [
        "2026-08-03T00:00:00Z", "2026-08-02T00:00:00Z", "2026-08-01T00:00:00Z"]
    assert len(client.get("/captures", headers=B).json()) == 1


def test_download_returns_the_exact_bytes():
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    client.put(f"/captures/{cid}/audio", headers=A, content=PCM)
    resp = client.get(f"/captures/{cid}/audio", headers=A)
    assert resp.status_code == 200
    assert resp.content == PCM
    assert resp.headers["content-type"].startswith("application/octet-stream")


def test_download_before_upload_is_409():
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    assert client.get(f"/captures/{cid}/audio", headers=A).status_code == 409


def test_download_when_blob_vanished_is_404_not_empty_audio():
    _, blobs, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    client.put(f"/captures/{cid}/audio", headers=A, content=PCM)
    asyncio.run(blobs.delete(f"captures/alice/{cid}.pcm"))
    assert client.get(f"/captures/{cid}/audio", headers=A).status_code == 404


def test_download_by_non_owner_is_403():
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    client.put(f"/captures/{cid}/audio", headers=A, content=PCM)
    assert client.get(f"/captures/{cid}/audio", headers=B).status_code == 403


def test_labels_are_stored_verbatim_and_stamped():
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    payload = {"speakers": ["self", "other-1"],
               "events": [{"vector": "interrupting", "level": 2, "t": 41.5}],
               "notes": "self cuts in twice near the end"}
    body = client.put(f"/captures/{cid}/labels", headers=A, json=payload).json()
    assert body["labels"] == payload and body["labels_updated_at"]
    assert client.get(f"/captures/{cid}", headers=A).json()["labels"] == payload


def test_labels_replace_rather_than_merge():
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    client.put(f"/captures/{cid}/labels", headers=A, json={"a": 1, "b": 2})
    assert client.put(f"/captures/{cid}/labels", headers=A, json={"a": 9}).json()["labels"] == {"a": 9}


def test_non_object_labels_are_422():
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    resp = client.put(f"/captures/{cid}/labels", headers=A, json=[1, 2, 3])
    assert resp.status_code == 422
    assert resp.json()["detail"] == "labels must be a json object"


def test_oversized_labels_are_422():
    from watch.routers.captures import MAX_LABELS_BYTES
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    huge = {"notes": "x" * (MAX_LABELS_BYTES + 100)}
    assert client.put(f"/captures/{cid}/labels", headers=A, json=huge).status_code == 422


def test_labels_by_non_owner_are_403():
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    assert client.put(f"/captures/{cid}/labels", headers=B, json={"a": 1}).status_code == 403


def test_captures_require_auth():
    _, _, client = _client()
    assert client.get("/captures").status_code == 401


def test_captures_reject_legacy_account_param_even_though_flag_is_on():
    """I2/I3 pinning test: before the fix (gauge), GAUGE_ALLOW_LEGACY_ACCOUNT=true
    let ANY caller reach the whole captures router by sending an unauthenticated
    `?account=<anyone>` -- no token, no proof of identity -- which would
    expose someone else's own-voice audio captures. `_client()` passes
    allow_legacy=True (same posture) so the query param actually resolves to
    a legacy Principal; the captures router still requires a full
    (non-legacy) principal on every route (require_full_auth), so that
    legacy principal is rejected with a 401, not let through as a 200."""
    _, _, client = _client()
    resp = client.get("/captures", params={"account": "alice"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "this endpoint requires sign-in"


# --------------------------------------------------------------- deletion (gauge Task 8) --

def test_delete_capture_removes_blob_then_doc():
    store, blobs, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    client.put(f"/captures/{cid}/audio", headers=A, content=PCM)
    assert client.delete(f"/captures/{cid}", headers=A).status_code == 204
    assert asyncio.run(store.get_capture(cid)) is None
    assert asyncio.run(blobs.get(f"captures/alice/{cid}.pcm")) is None


def test_delete_capture_metadata_only_when_no_audio_uploaded():
    store, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]     # status awaiting_audio
    assert client.delete(f"/captures/{cid}", headers=A).status_code == 204
    assert asyncio.run(store.get_capture(cid)) is None


def test_delete_capture_blob_failure_keeps_metadata():
    store, _, client = _client(blob_delete_raises=True)
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    client.put(f"/captures/{cid}/audio", headers=A, content=PCM)
    resp = client.delete(f"/captures/{cid}", headers=A)
    assert resp.status_code == 502
    assert asyncio.run(store.get_capture(cid)) is not None   # retryable, never orphaned audio


def test_delete_capture_without_blob_store_is_503_when_audio_present():
    # Same honest-degradation posture as upload/download: metadata says audio IS
    # stored, but this deployment has no blob store configured to delete it from
    # -- never silently drop the metadata while leaving orphaned audio unaccounted
    # for in some OTHER deployment that later gets a blob store configured.
    store = MemoryLiveSessionStore()
    blobs = MemoryBlobStore()
    client = TestClient(create_watch_test_app(store=store, verifier=StubVerifier(TOKENS), blobs=blobs))
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    client.put(f"/captures/{cid}/audio", headers=A, content=PCM)

    # Re-create the app with the SAME store but no blob store, simulating a
    # config where storage vanished between upload and delete.
    client_no_blobs = TestClient(
        create_watch_test_app(store=store, verifier=StubVerifier(TOKENS), blobs=None)
    )
    resp = client_no_blobs.delete(f"/captures/{cid}", headers=A)
    assert resp.status_code == 503
    assert asyncio.run(store.get_capture(cid)) is not None


def test_delete_capture_not_owner_is_403():
    # NOTE (gauge Task 8): the brief there described _get_owned_or_404 as
    # folding "not yours" into 404 for captures, but the actual, already-
    # tested contract (see test_upload_by_non_owner_is_403 /
    # test_download_by_non_owner_is_403 / test_labels_by_non_owner_are_403
    # above) is 403 -- _get_owned_or_404 raises 404 only when the capture
    # doesn't exist at all, 403 when it exists but belongs to someone else.
    # Preserved verbatim here.
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    client.put(f"/captures/{cid}/audio", headers=A, content=PCM)
    assert client.delete(f"/captures/{cid}", headers=B).status_code == 403


def test_delete_capture_unknown_is_404():
    _, _, client = _client()
    assert client.delete("/captures/nope", headers=A).status_code == 404


def test_delete_captures_require_auth():
    _, _, client = _client()
    assert client.delete("/captures/nope").status_code == 401
