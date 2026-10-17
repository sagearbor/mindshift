"""Blob tier for the library: the original upload and its extracted text.

Production writes into the SAME bucket recordings use
(``MINDSHIFT_RECORDINGS_BUCKET``, ``arborfam-hub-mindshift-recordings`` in
prod) under its own prefix::

    library/{uid}/{item_id}/original{.ext}   the uploaded file, byte-for-byte
    library/{uid}/{item_id}/text.txt         the extracted (or note) text

Extracted text lives here rather than in Firestore because an item may carry
up to 2 MB of text and a Firestore document caps at 1 MiB. With the bucket
unset (keyless dev / CI) an in-memory store is used and a warning is logged.
"""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


def item_prefix(uid: str, item_id: str) -> str:
    return f"library/{uid}/{item_id}/"


def user_prefix(uid: str) -> str:
    return f"library/{uid}/"


class MemoryLibraryBlobs:
    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self._data[key] = bytes(data)

    async def get(self, key: str) -> bytes | None:
        return self._data.get(key)

    async def delete_prefix(self, prefix: str) -> int:
        keys = [k for k in self._data if k.startswith(prefix)]
        for k in keys:
            del self._data[k]
        return len(keys)


class GcsLibraryBlobs:
    def __init__(self, bucket_name: str) -> None:
        self.bucket_name = bucket_name
        self._bucket = None

    def _b(self):
        if self._bucket is None:
            from google.cloud import storage

            self._bucket = storage.Client().bucket(self.bucket_name)
        return self._bucket

    def _put_sync(self, key, data, content_type):
        self._b().blob(key).upload_from_string(data, content_type=content_type)

    def _get_sync(self, key):
        from google.cloud.exceptions import NotFound

        try:
            return self._b().blob(key).download_as_bytes()
        except NotFound:
            return None

    def _delete_prefix_sync(self, prefix):
        from google.cloud.exceptions import NotFound

        n = 0
        for blob in list(self._b().list_blobs(prefix=prefix)):
            try:
                blob.delete()
                n += 1
            except NotFound:
                pass
        return n

    async def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        await asyncio.to_thread(self._put_sync, key, data, content_type)

    async def get(self, key: str) -> bytes | None:
        return await asyncio.to_thread(self._get_sync, key)

    async def delete_prefix(self, prefix: str) -> int:
        parts = prefix.split("/")
        if len(parts) < 3 or parts[0] != "library" or not parts[1]:
            raise ValueError(f"refusing to delete outside a library user prefix: {prefix!r}")
        return await asyncio.to_thread(self._delete_prefix_sync, prefix)


def get_blobs():
    bucket = (os.getenv("MINDSHIFT_RECORDINGS_BUCKET") or "").strip()
    if bucket:
        return GcsLibraryBlobs(bucket)
    logger.warning(
        "MINDSHIFT_RECORDINGS_BUCKET unset — library files are kept IN MEMORY "
        "and lost on restart (dev/CI only)",
    )
    return MemoryLibraryBlobs()
