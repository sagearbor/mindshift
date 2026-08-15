# Ported from gauge@2157433 server/blobs.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
"""Blob storage tier for capture audio (Task 14).

Companion to ``store.py``: captures' metadata lives in the live-session
store's ``captures`` collection, but the audio bytes themselves go through a
``BlobStore`` — in-memory for tests/default runtime, Cloud Storage in
production. ``google.cloud.storage`` is imported LAZILY inside
``GcsBlobStore._get_bucket``, exactly like ``FirestoreLiveSessionStore._get_db``
in ``store.py``, so the base test suite never needs the SDK installed.
"""

import asyncio
import os
from typing import Protocol


class BlobStore(Protocol):
    async def put(self, key: str, data: bytes,
                  content_type: str = "application/octet-stream") -> str:
        """Store ``data`` under ``key`` and return a uri identifying it."""
        ...

    async def get(self, key: str) -> bytes | None:
        """Retrieve the bytes stored under ``key``, or None if absent."""
        ...

    async def delete(self, key: str) -> None:
        """Remove ``key`` if present. Idempotent — never raises when absent."""
        ...


class MemoryBlobStore:
    """dict-backed BlobStore for tests and default runtime.

    uri scheme is ``memory://<key>``. ``get`` returns bytes copies so a
    caller mutating the returned bytes-like object (not possible for
    immutable ``bytes``, but kept honest for parity with the store.py
    copy-out convention) can never corrupt the stored value.
    """

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes,
                  content_type: str = "application/octet-stream") -> str:
        self._data[key] = bytes(data)
        return f"memory://{key}"

    async def get(self, key: str) -> bytes | None:
        data = self._data.get(key)
        return bytes(data) if data is not None else None

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)


class GcsBlobStore:
    """Cloud Storage-backed BlobStore.

    ``from google.cloud import storage`` is imported LAZILY inside
    ``_get_bucket()``, exactly like ``FirestoreLiveSessionStore._get_db`` in
    ``store.py``, so the base test suite never needs the SDK installed.
    """

    def __init__(self, bucket: str) -> None:
        self.bucket = bucket
        self._bucket_ref = None

    def _get_bucket(self):
        """Lazily import and initialize the Cloud Storage bucket handle."""
        if self._bucket_ref is None:
            from google.cloud import storage
            client = storage.Client()
            self._bucket_ref = client.bucket(self.bucket)
        return self._bucket_ref

    def _put_sync(self, key: str, data: bytes, content_type: str) -> str:
        bucket = self._get_bucket()
        blob = bucket.blob(key)
        blob.upload_from_string(data, content_type=content_type)
        return f"gs://{self.bucket}/{key}"

    def _get_sync(self, key: str) -> bytes | None:
        bucket = self._get_bucket()
        blob = bucket.blob(key)
        if not blob.exists():
            return None
        return blob.download_as_bytes()

    def _delete_sync(self, key: str) -> None:
        # No exists() pre-check: that would be TOCTOU (the object can vanish
        # between the check and .delete()) and BlobStore.delete promises
        # idempotency ("never raises when absent") the same way
        # MemoryBlobStore honors it via dict.pop(key, None). Catch GCS's
        # NotFound directly instead.
        from google.cloud.exceptions import NotFound
        bucket = self._get_bucket()
        blob = bucket.blob(key)
        try:
            blob.delete()
        except NotFound:
            pass

    async def put(self, key: str, data: bytes,
                  content_type: str = "application/octet-stream") -> str:
        # A 10 MB upload runs through the blocking SDK call via
        # asyncio.to_thread (same pattern as post_session.py's
        # analyze_live_session / rest_api.py's _maybe_enroll_voice) so it
        # never blocks the event loop.
        return await asyncio.to_thread(self._put_sync, key, data, content_type)

    async def get(self, key: str) -> bytes | None:
        return await asyncio.to_thread(self._get_sync, key)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._delete_sync, key)


def get_blob_store() -> "BlobStore | None":
    """GcsBlobStore when MINDSHIFT_CAPTURE_BUCKET is set, else None.

    None means capture audio upload is honestly unavailable (503), NOT
    silently dropped — the caller is responsible for surfacing that.
    """
    bucket = os.environ.get("MINDSHIFT_CAPTURE_BUCKET")
    if bucket:
        return GcsBlobStore(bucket)
    return None
