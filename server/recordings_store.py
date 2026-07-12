"""Consent-gated recording persistence for POST /analyze/upload — list, replay,
delete on Google Cloud Storage.

Design constraints (see the feature spec):

* Storage is OPT-IN and OFF by default. The bucket comes from the env var
  ``MINDSHIFT_RECORDINGS_BUCKET``; when it is unset/empty :func:`create_store`
  returns ``None`` and every recordings endpoint reports an honest 503 while
  ``/analyze/upload`` keeps its original process-and-discard behaviour.
* NO GCS signed URLs. Signing would need an extra IAM grant on the Cloud Run
  service account, so instead the backend streams media itself behind short
  lived HMAC tokens (minted in ``main.py``); this module only reads the bytes.
* Every blocking google-cloud-storage call runs inside ``asyncio.to_thread`` —
  the SDK is synchronous and must never sit on the event loop (house pattern,
  matching the Deepgram/ffmpeg calls in the upload path).

Object layout, per recording::

    recordings/{uid}/{recording_id}/original.{ext}   # raw uploaded bytes
    recordings/{uid}/{recording_id}/meta.json        # {id, created_at, ...}
    recordings/{uid}/{recording_id}/turns.json       # transcribed turns
    recordings/{uid}/{recording_id}/analysis.json    # full analysis response

``uid`` comes from the verified Firebase token (trusted); ``recording_id`` is a
server-minted uuid4 validated against ``UUID_PATTERN`` at the endpoint. Every
read/write/delete is scoped under ``recordings/{uid}/`` so one user can never
touch another's objects.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Streaming chunk size for the in-memory body iterator (the bytes are already
# downloaded off the event loop; this only slices what to hand the transport).
_STREAM_CHUNK = 64 * 1024

# Content types we map to a friendly file extension when the upload's filename
# carried none. Best-effort only — a wrong guess affects the stored object name,
# never correctness (the real Content-Type is read back from blob metadata).
_EXT_BY_CONTENT_TYPE = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/aac": "aac",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
    "audio/flac": "flac",
    "video/mp4": "mp4",
    "video/quicktime": "mov",
    "video/webm": "webm",
}


def create_store() -> "RecordingsStore | None":
    """Build a :class:`RecordingsStore`, or ``None`` when storage is disabled.

    Disabled means ``MINDSHIFT_RECORDINGS_BUCKET`` is unset/empty. Returning
    ``None`` is the load-bearing "storage not enabled" signal the endpoints turn
    into honest 503s. ``google.cloud.storage`` is imported lazily (like
    firebase_admin / imageio-ffmpeg elsewhere) so the module — and the whole
    test suite — imports cleanly without the package or any credentials.
    """
    bucket_name = (os.getenv("MINDSHIFT_RECORDINGS_BUCKET") or "").strip()
    if not bucket_name:
        logger.info(
            "MINDSHIFT_RECORDINGS_BUCKET unset — recording storage disabled",
        )
        return None
    from google.cloud import storage  # lazy: no dep/credentials at import time

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    logger.info("Recording storage enabled — bucket=%s", bucket_name)
    return RecordingsStore(bucket)


# ---------------------------------------------------------------------------
# Pure helpers (no I/O) — unit-testable and shared by the GCS store + fakes
# ---------------------------------------------------------------------------

def _ext_for(filename: str | None, content_type: str | None) -> str:
    """Pick a lowercase extension for ``original.{ext}`` from the filename, then
    the content type, defaulting to ``bin``. Purely cosmetic on the object name.
    """
    if filename:
        _, _, ext = filename.rpartition(".")
        if ext and "/" not in ext and len(ext) <= 8:
            return ext.lower()
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    return _EXT_BY_CONTENT_TYPE.get(ct, "bin")


def _media_type_for(content_type: str | None) -> str:
    """"video" for a video/* mime, else "audio" (audio is the default kind)."""
    return "video" if (content_type or "").lower().startswith("video/") else "audio"


def _parse_range(range_header: str | None, size: int) -> "tuple[int, int] | None":
    """Parse a single HTTP ``Range`` into inclusive ``(start, end)`` byte
    offsets, or ``None`` for absent/unsatisfiable/malformed input (caller then
    serves the full 200). Supports ``bytes=start-end``, ``bytes=start-`` and the
    suffix form ``bytes=-N`` — enough for media-element seeking.
    """
    if not range_header or size <= 0:
        return None
    header = range_header.strip()
    if not header.startswith("bytes="):
        return None
    # A single range only — take the first if a comma-list was sent.
    spec = header[len("bytes="):].split(",", 1)[0].strip()
    if "-" not in spec:
        return None
    start_s, _, end_s = spec.partition("-")
    try:
        if start_s == "":
            # Suffix range: the last N bytes.
            n = int(end_s)
            if n <= 0:
                return None
            start = max(0, size - n)
            end = size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
    except ValueError:
        return None
    if start < 0 or start > end or start >= size:
        return None
    return start, min(end, size - 1)


def plan_media_response(
    size: int, content_type: str, range_header: str | None,
) -> "tuple[int, int, int, dict[str, str]]":
    """Decide status + headers for a media response.

    Returns ``(start, end, status, headers)`` where ``start``/``end`` are the
    inclusive byte offsets to read. A satisfiable Range yields 206 with
    ``Content-Range``; otherwise a full 200. ``Accept-Ranges: bytes`` is always
    advertised so clients know seeking is supported. Pure — the GCS store and
    the test fake share it so their range math cannot drift.
    """
    rng = _parse_range(range_header, size)
    if rng is None:
        headers = {
            "Content-Type": content_type,
            "Content-Length": str(size),
            "Accept-Ranges": "bytes",
        }
        return 0, max(0, size - 1), 200, headers
    start, end = rng
    headers = {
        "Content-Type": content_type,
        "Content-Length": str(end - start + 1),
        "Accept-Ranges": "bytes",
        "Content-Range": f"bytes {start}-{end}/{size}",
    }
    return start, end, 206, headers


def _iter_bytes(data: bytes):
    """Yield ``data`` in transport-sized chunks (bytes already in memory)."""
    for i in range(0, len(data), _STREAM_CHUNK):
        yield data[i:i + _STREAM_CHUNK]


# ---------------------------------------------------------------------------
# GCS-backed store
# ---------------------------------------------------------------------------

class RecordingsStore:
    """Async facade over a single GCS bucket. Every method scopes objects under
    ``recordings/{uid}/`` and offloads the blocking SDK to a worker thread."""

    def __init__(self, bucket) -> None:
        self._bucket = bucket

    # -- prefixes ----------------------------------------------------------
    @staticmethod
    def _prefix(uid: str, recording_id: str | None = None) -> str:
        if recording_id is None:
            return f"recordings/{uid}/"
        return f"recordings/{uid}/{recording_id}/"

    # -- save --------------------------------------------------------------
    async def save_recording(
        self,
        uid: str,
        *,
        data: bytes,
        filename: str | None,
        content_type: str | None,
        duration_seconds: float | None,
        turns: list[dict],
        analysis: dict,
    ) -> str:
        """Persist one recording (original bytes + meta + turns + analysis) and
        return its new uuid4 id. All four objects are written in one worker
        thread; created_at is the server clock (ISO-8601 UTC)."""
        recording_id = str(uuid.uuid4())
        ext = _ext_for(filename, content_type)
        meta = {
            "id": recording_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "filename": filename or f"recording.{ext}",
            "media_type": _media_type_for(content_type),
            "duration_seconds": duration_seconds,
            "size_bytes": len(data),
        }
        await asyncio.to_thread(
            self._save_sync, uid, recording_id, ext, data, content_type,
            meta, turns, analysis,
        )
        return recording_id

    def _save_sync(
        self, uid, recording_id, ext, data, content_type, meta, turns, analysis,
    ) -> None:
        prefix = self._prefix(uid, recording_id)
        self._bucket.blob(prefix + f"original.{ext}").upload_from_string(
            data, content_type=content_type or "application/octet-stream",
        )
        self._bucket.blob(prefix + "meta.json").upload_from_string(
            json.dumps(meta), content_type="application/json",
        )
        self._bucket.blob(prefix + "turns.json").upload_from_string(
            json.dumps(turns), content_type="application/json",
        )
        self._bucket.blob(prefix + "analysis.json").upload_from_string(
            json.dumps(analysis), content_type="application/json",
        )

    # -- list --------------------------------------------------------------
    async def list_recordings(self, uid: str) -> list[dict]:
        """All of ``uid``'s recordings' meta (+ ``has_analysis``), newest first."""
        return await asyncio.to_thread(self._list_sync, uid)

    def _list_sync(self, uid: str) -> list[dict]:
        prefix = self._prefix(uid)
        # One list call; group blobs by recording id from the object name.
        by_id: dict[str, dict] = {}
        for blob in self._bucket.list_blobs(prefix=prefix):
            rel = blob.name[len(prefix):]
            recording_id, _, fname = rel.partition("/")
            if not recording_id or not fname:
                continue
            by_id.setdefault(recording_id, {})[fname] = blob
        out: list[dict] = []
        for files in by_id.values():
            meta_blob = files.get("meta.json")
            if meta_blob is None:
                continue  # incomplete/partial recording — skip honestly
            meta = json.loads(meta_blob.download_as_bytes())
            meta["has_analysis"] = "analysis.json" in files
            out.append(meta)
        out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return out

    # -- detail ------------------------------------------------------------
    async def get_recording(self, uid: str, recording_id: str) -> dict | None:
        """Meta + turns + analysis for one recording, or ``None`` (→ 404)."""
        return await asyncio.to_thread(self._get_sync, uid, recording_id)

    def _get_sync(self, uid: str, recording_id: str) -> dict | None:
        prefix = self._prefix(uid, recording_id)
        meta_blob = self._bucket.blob(prefix + "meta.json")
        if not meta_blob.exists():
            return None
        meta = json.loads(meta_blob.download_as_bytes())
        turns_blob = self._bucket.blob(prefix + "turns.json")
        turns = (
            json.loads(turns_blob.download_as_bytes())
            if turns_blob.exists() else []
        )
        analysis_blob = self._bucket.blob(prefix + "analysis.json")
        analysis = (
            json.loads(analysis_blob.download_as_bytes())
            if analysis_blob.exists() else None
        )
        return {**meta, "turns": turns, "analysis": analysis}

    async def recording_exists(self, uid: str, recording_id: str) -> bool:
        """Cheap ownership/existence check (meta.json present) for media_url."""
        return await asyncio.to_thread(self._exists_sync, uid, recording_id)

    def _exists_sync(self, uid: str, recording_id: str) -> bool:
        return self._bucket.blob(
            self._prefix(uid, recording_id) + "meta.json"
        ).exists()

    # -- delete ------------------------------------------------------------
    async def delete_recording(self, uid: str, recording_id: str) -> bool:
        """Delete every object for a recording. ``False`` when none existed
        (→ 404); ``True`` on a successful delete."""
        return await asyncio.to_thread(self._delete_sync, uid, recording_id)

    def _delete_sync(self, uid: str, recording_id: str) -> bool:
        prefix = self._prefix(uid, recording_id)
        blobs = list(self._bucket.list_blobs(prefix=prefix))
        if not blobs:
            return False
        for blob in blobs:
            blob.delete()
        return True

    # -- media stream ------------------------------------------------------
    async def open_media_stream(
        self, uid: str, recording_id: str, range_header: str | None,
    ):
        """Return ``(iterator, status, headers)`` for the stored original, or
        ``None`` when it is missing (→ 404).

        Honors a single ``Range`` via a GCS ranged download (206 +
        ``Content-Range``), else a full 200. The Content-Type is read back from
        the stored blob metadata. The download runs in a worker thread; the
        returned iterator only slices the already-fetched bytes.
        """
        return await asyncio.to_thread(
            self._open_media_stream_sync, uid, recording_id, range_header,
        )

    def _open_media_stream_sync(self, uid, recording_id, range_header):
        prefix = self._prefix(uid, recording_id)
        # The extension is unknown here, so locate the original.* object.
        candidates = list(self._bucket.list_blobs(prefix=prefix + "original."))
        if not candidates:
            return None
        blob = candidates[0]
        blob.reload()  # populate size + content_type
        size = blob.size or 0
        content_type = blob.content_type or "application/octet-stream"
        start, end, status, headers = plan_media_response(
            size, content_type, range_header,
        )
        if status == 206:
            # google-cloud-storage `end` is INCLUSIVE (the last byte), which is
            # exactly the HTTP Range semantics — no off-by-one adjustment.
            data = blob.download_as_bytes(start=start, end=end)
        else:
            data = blob.download_as_bytes()
        return _iter_bytes(data), status, headers
