# Ported from gauge@2157433 server/tests/test_blobs.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
import asyncio

from watch.blobs import MemoryBlobStore, get_blob_store


def test_put_get_roundtrip():
    b = MemoryBlobStore()
    uri = asyncio.run(b.put("captures/a/1.pcm", b"\x01\x02\x03"))
    assert uri == "memory://captures/a/1.pcm"
    assert asyncio.run(b.get("captures/a/1.pcm")) == b"\x01\x02\x03"


def test_missing_key_is_none_not_empty_bytes():
    assert asyncio.run(MemoryBlobStore().get("nope")) is None


def test_delete_removes():
    b = MemoryBlobStore()
    asyncio.run(b.put("k", b"x"))
    asyncio.run(b.delete("k"))
    assert asyncio.run(b.get("k")) is None
    asyncio.run(b.delete("k"))            # idempotent, never raises


def test_get_blob_store_is_none_without_bucket(monkeypatch):
    monkeypatch.delenv("MINDSHIFT_CAPTURE_BUCKET", raising=False)
    assert get_blob_store() is None


def test_get_blob_store_builds_gcs_when_bucket_set(monkeypatch):
    import sys
    from watch.blobs import GcsBlobStore
    monkeypatch.setenv("MINDSHIFT_CAPTURE_BUCKET", "arborfam-hub-gauge-captures")
    before = "google.cloud.storage" in sys.modules
    s = get_blob_store()
    assert isinstance(s, GcsBlobStore) and s.bucket == "arborfam-hub-gauge-captures"
    assert ("google.cloud.storage" in sys.modules) == before   # construction stays lazy


def test_gcs_delete_of_absent_key_is_idempotent_not_notfound(monkeypatch):
    """BlobStore.delete promises 'idempotent -- never raises when absent'
    (MemoryBlobStore honors it via dict.pop(key, None)). GcsBlobStore must
    honor the same contract even though the real SDK's Blob.delete() raises
    NotFound for an absent/already-deleted object (including the TOCTOU
    where the object vanishes between any check and the delete call) --
    _delete_sync must swallow that, not propagate it.

    Uses a fake `google.cloud.exceptions` module (injected into
    sys.modules) plus a fake bucket/blob, rather than the real SDK, so this
    test never depends on google-cloud-storage actually being installed --
    consistent with blobs.py's lazy-import design.
    """
    import sys
    import types
    from watch.blobs import GcsBlobStore

    class FakeNotFound(Exception):
        pass

    fake_exceptions = types.ModuleType("google.cloud.exceptions")
    fake_exceptions.NotFound = FakeNotFound
    monkeypatch.setitem(sys.modules, "google.cloud.exceptions", fake_exceptions)

    class FakeBlob:
        def delete(self):
            raise FakeNotFound("already gone")

    class FakeBucket:
        def blob(self, key):
            return FakeBlob()

    store = GcsBlobStore("some-bucket")
    monkeypatch.setattr(store, "_get_bucket", lambda: FakeBucket())

    asyncio.run(store.delete("nope"))   # must not raise
