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

Object layout, per recording (we store compressed DERIVATIVES, never the
original bytes — see :meth:`RecordingsStore.save_recording`)::

    recordings/{uid}/{recording_id}/audio.m4a        # AAC audio derivative
    recordings/{uid}/{recording_id}/video_360p.mp4   # 360p H.264 (video only)
    recordings/{uid}/{recording_id}/meta.json        # {id, created_at, source, ...}
    recordings/{uid}/{recording_id}/turns.json       # transcribed turns
    recordings/{uid}/{recording_id}/analysis.json    # full analysis response

In-progress chunked uploads live under a separate ``uploads/{uid}/{upload_id}/``
namespace (manifest.json + parts/) — see the upload-session methods.

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
        audio_m4a: bytes,
        video_360p: bytes | None,
        original_filename: str | None,
        original_content_type: str | None,
        original_bytes: int,
        duration_seconds: float | None,
        turns: list[dict],
        analysis: dict,
        source: dict | None = None,
        title: str | None = None,
    ) -> str:
        """Persist one recording's DERIVATIVES (never the original bytes) + meta
        + turns + analysis and return its new uuid4 id.

        We always store a compressed ``audio.m4a`` and, when the input carried a
        video stream, a small ``video_360p.mp4`` — a deliberate cost decision (a
        50-300MB phone original becomes a handful of MB). ``media_type`` is
        derived from what was actually STORED (video only when the 360p clip is
        present), and ``original_bytes``/``original_filename`` are kept for
        provenance. ``title`` is the user-facing display name; when absent/blank it
        falls back to the filename so every recording always has one. All objects
        are written in one worker thread; created_at is the server clock (ISO-8601
        UTC)."""
        recording_id = str(uuid.uuid4())
        stored_variants = ["audio.m4a"]
        if video_360p is not None:
            stored_variants.append("video_360p.mp4")
        stored_bytes = len(audio_m4a) + (len(video_360p) if video_360p else 0)
        filename = original_filename or "recording"
        meta = {
            "id": recording_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            # User-facing display name; defaults to the filename when not provided.
            "title": (title or "").strip() or filename,
            # media_type reflects the STORED artifact, not the upload's mime.
            "media_type": "video" if video_360p is not None else "audio",
            "duration_seconds": duration_seconds,
            "size_bytes": stored_bytes,
            "stored_variants": stored_variants,
            "original_bytes": original_bytes,
            "original_filename": original_filename,
            "original_content_type": original_content_type,
            # Provenance for a future replay feature (stream the user's own hosted
            # HD copy instead of our derivative). Metadata only — no playback here.
            "source": source or {
                "type": "upload", "url": None,
                "original_filename": original_filename,
            },
        }
        await asyncio.to_thread(
            self._save_sync, uid, recording_id, audio_m4a, video_360p,
            meta, turns, analysis,
        )
        return recording_id

    def _save_sync(
        self, uid, recording_id, audio_m4a, video_360p, meta, turns, analysis,
    ) -> None:
        prefix = self._prefix(uid, recording_id)
        self._bucket.blob(prefix + "audio.m4a").upload_from_string(
            audio_m4a, content_type="audio/mp4",
        )
        if video_360p is not None:
            self._bucket.blob(prefix + "video_360p.mp4").upload_from_string(
                video_360p, content_type="video/mp4",
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

    # -- update source -----------------------------------------------------
    async def update_source(
        self, uid: str, recording_id: str, source: dict,
    ) -> dict | None:
        """Replace an existing recording's ``source`` provenance, returning the
        stored source dict, or ``None`` when the recording does not exist for
        this uid (→ 404).

        Read-modify-write of meta.json only — the derivatives, turns, and
        analysis are untouched. Used to attach an HD source link to a recording
        after the fact (the user records in-app, then pastes the durable share
        link once the original has backed up to their cloud)."""
        return await asyncio.to_thread(
            self._update_source_sync, uid, recording_id, source,
        )

    def _update_source_sync(self, uid, recording_id, source) -> dict | None:
        blob = self._bucket.blob(
            self._prefix(uid, recording_id) + "meta.json"
        )
        if not blob.exists():
            return None
        meta = json.loads(blob.download_as_bytes())
        meta["source"] = source
        blob.upload_from_string(
            json.dumps(meta), content_type="application/json",
        )
        return source

    # -- update title -----------------------------------------------------
    async def update_title(
        self, uid: str, recording_id: str, title: str,
    ) -> dict | None:
        """Rename an existing recording, returning its updated meta, or ``None``
        when it does not exist for this uid (→ 404).

        Read-modify-write of meta.json only (derivatives/turns/analysis untouched)
        — the same shape as :meth:`update_source`. The caller has already
        stripped/validated ``title``."""
        return await asyncio.to_thread(
            self._update_title_sync, uid, recording_id, title,
        )

    def _update_title_sync(self, uid, recording_id, title) -> dict | None:
        blob = self._bucket.blob(
            self._prefix(uid, recording_id) + "meta.json"
        )
        if not blob.exists():
            return None
        meta = json.loads(blob.download_as_bytes())
        meta["title"] = title
        blob.upload_from_string(
            json.dumps(meta), content_type="application/json",
        )
        return meta

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
        # Serve the richest stored derivative: the 360p video when present, else
        # the audio. Both have fixed names now (no original.* to locate).
        blob = None
        for name in ("video_360p.mp4", "audio.m4a"):
            candidate = self._bucket.blob(prefix + name)
            if candidate.exists():
                blob = candidate
                break
        if blob is None:
            return None
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

    # -- chunked upload sessions ------------------------------------------
    # A large recording (phone video, 50-300MB) is streamed to the server in 8MB
    # parts because Cloud Run's ~32MB request limit forbids a single big body.
    # The session state lives entirely in GCS under a separate ``uploads/``
    # namespace (NOT ``recordings/``), scoped per uid so one user can never touch
    # another's in-progress upload::
    #
    #     uploads/{uid}/{upload_id}/manifest.json    # start()'s declared metadata
    #     uploads/{uid}/{upload_id}/parts/{i:05d}    # one object per 8MB chunk
    #     uploads/{uid}/{upload_id}/assembled        # transient compose target
    #
    # Everything under the prefix is deleted by :meth:`cleanup_upload` on
    # complete or abort.
    @staticmethod
    def _upload_prefix(uid: str, upload_id: str) -> str:
        return f"uploads/{uid}/{upload_id}/"

    async def write_upload_manifest(
        self, uid: str, upload_id: str, manifest: dict,
    ) -> None:
        """Write the session manifest (start())."""
        await asyncio.to_thread(
            self._write_upload_manifest_sync, uid, upload_id, manifest,
        )

    def _write_upload_manifest_sync(self, uid, upload_id, manifest) -> None:
        self._bucket.blob(
            self._upload_prefix(uid, upload_id) + "manifest.json"
        ).upload_from_string(
            json.dumps(manifest), content_type="application/json",
        )

    async def read_upload_manifest(
        self, uid: str, upload_id: str,
    ) -> dict | None:
        """The session manifest, or ``None`` when it does not exist for this uid
        (→ 404). uid-scoped: a foreign upload_id reads as absent."""
        return await asyncio.to_thread(
            self._read_upload_manifest_sync, uid, upload_id,
        )

    def _read_upload_manifest_sync(self, uid, upload_id) -> dict | None:
        blob = self._bucket.blob(
            self._upload_prefix(uid, upload_id) + "manifest.json"
        )
        if not blob.exists():
            return None
        return json.loads(blob.download_as_bytes())

    async def write_upload_part(
        self, uid: str, upload_id: str, index: int, data: bytes,
    ) -> None:
        """Store (or overwrite — idempotent) one part at ``parts/{index:05d}``."""
        await asyncio.to_thread(
            self._write_upload_part_sync, uid, upload_id, index, data,
        )

    def _write_upload_part_sync(self, uid, upload_id, index, data) -> None:
        self._bucket.blob(
            self._upload_prefix(uid, upload_id) + f"parts/{index:05d}"
        ).upload_from_string(data, content_type="application/octet-stream")

    async def get_upload_part_sizes(
        self, uid: str, upload_id: str,
    ) -> dict[int, int]:
        """Map ``{part_index: size_bytes}`` for every part currently present —
        the endpoint uses it to list missing indexes and verify the total."""
        return await asyncio.to_thread(
            self._get_upload_part_sizes_sync, uid, upload_id,
        )

    def _get_upload_part_sizes_sync(self, uid, upload_id) -> dict[int, int]:
        prefix = self._upload_prefix(uid, upload_id) + "parts/"
        sizes: dict[int, int] = {}
        for blob in self._bucket.list_blobs(prefix=prefix):
            name = blob.name[len(prefix):]
            if not name.isdigit():
                continue
            sizes[int(name)] = blob.size or 0
        return sizes

    async def assemble_upload(
        self, uid: str, upload_id: str, expected_chunks: int,
    ) -> bytes:
        """Reassemble all parts (in index order) into the original bytes."""
        return await asyncio.to_thread(
            self._assemble_upload_sync, uid, upload_id, expected_chunks,
        )

    def _assemble_upload_sync(self, uid, upload_id, expected_chunks) -> bytes:
        prefix = self._upload_prefix(uid, upload_id)
        part_blobs = [
            self._bucket.blob(prefix + f"parts/{i:05d}")
            for i in range(expected_chunks)
        ]
        # GCS `compose` concatenates up to 32 source objects server-side in ONE
        # operation — no download/re-upload of the intermediate bytes. With 8MB
        # chunks and the 200MB cap there are at most 25 parts, so compose always
        # applies here. The download-concat branch is a correctness fallback for
        # a hypothetical larger part count (e.g. a smaller chunk size).
        if expected_chunks <= 32:
            assembled = self._bucket.blob(prefix + "assembled")
            assembled.compose(part_blobs)
            return assembled.download_as_bytes()
        return b"".join(b.download_as_bytes() for b in part_blobs)

    async def cleanup_upload(self, uid: str, upload_id: str) -> None:
        """Delete every object for an upload session (parts + manifest +
        transient assembled blob). Best-effort per blob so one failed delete does
        not abort the rest."""
        await asyncio.to_thread(self._cleanup_upload_sync, uid, upload_id)

    def _cleanup_upload_sync(self, uid, upload_id) -> None:
        prefix = self._upload_prefix(uid, upload_id)
        for blob in self._bucket.list_blobs(prefix=prefix):
            try:
                blob.delete()
            except Exception:  # noqa: BLE001 — best-effort cleanup, per blob
                logger.warning("Failed to delete upload blob %s", blob.name)

    # -- async analysis jobs ----------------------------------------------
    # A submit-and-poll analysis job (POST /analyze/link/jobs or
    # /uploads/{id}/complete/jobs) runs as an in-process background task and
    # records its staged progress in a single JSON object here, under a ``jobs/``
    # namespace scoped per uid (so one user can never read another's job)::
    #
    #     jobs/{uid}/{job_id}/state.json   # {status, progress_note, result, ...}
    #
    # The state is rewritten in full between stages (the owning task is the sole
    # writer), so there is no read-modify-write race. Consistent with the upload
    # sessions above: jobs REQUIRE a bucket, so storage-disabled means the
    # job endpoints report an honest 503 while the old synchronous endpoints keep
    # working.
    @staticmethod
    def _job_prefix(uid: str, job_id: str) -> str:
        return f"jobs/{uid}/{job_id}/"

    async def write_job_state(
        self, uid: str, job_id: str, state: dict,
    ) -> None:
        """Write (or overwrite) a job's full state document."""
        await asyncio.to_thread(self._write_job_state_sync, uid, job_id, state)

    def _write_job_state_sync(self, uid, job_id, state) -> None:
        self._bucket.blob(
            self._job_prefix(uid, job_id) + "state.json"
        ).upload_from_string(
            json.dumps(state), content_type="application/json",
        )

    async def read_job_state(self, uid: str, job_id: str) -> dict | None:
        """A job's state, or ``None`` when it does not exist for this uid (→ 404).
        uid-scoped: a foreign job_id reads as absent."""
        return await asyncio.to_thread(self._read_job_state_sync, uid, job_id)

    def _read_job_state_sync(self, uid, job_id) -> dict | None:
        blob = self._bucket.blob(self._job_prefix(uid, job_id) + "state.json")
        if not blob.exists():
            return None
        return json.loads(blob.download_as_bytes())

    async def delete_job(self, uid: str, job_id: str) -> None:
        """Delete a job's state (lazy TTL cleanup on read). Best-effort."""
        await asyncio.to_thread(self._delete_job_sync, uid, job_id)

    def _delete_job_sync(self, uid, job_id) -> None:
        for blob in self._bucket.list_blobs(
            prefix=self._job_prefix(uid, job_id)
        ):
            try:
                blob.delete()
            except Exception:  # noqa: BLE001 — best-effort cleanup, per blob
                logger.warning("Failed to delete job blob %s", blob.name)
