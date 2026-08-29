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

import speaker_id  # pure profile-shape helpers (as_person) — no torch at import

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


# meta.json fields that describe a live session's ATTACHED audio (see
# RecordingsStore.attach_audio). A re-POST of the session (save_live_session)
# carries these over from the existing meta so the phone's "no audio" meta
# never disowns an audio.m4a that is still stored. duration_seconds is among
# them because attach_audio wrote the duration DECODED from the audio, which
# beats the transcript-derived estimate a re-POST would otherwise reinstate.
_ATTACHED_AUDIO_META_KEYS = (
    "media_type", "stored_variants", "size_bytes", "original_bytes",
    "storage_note", "audio_attached_at", "duration_seconds",
)


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
        storage_note: str | None = None,
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
            # Honest reason a derivative is absent (e.g. the video transcode timed
            # out, so only audio.m4a was stored). Carried in meta so the LIST view
            # — which reads meta.json alone — can explain a video link that landed
            # as an audio-only recording, instead of silently dropping the video.
            "storage_note": storage_note,
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

    # -- update manual speaker labels -------------------------------------
    async def update_manual_speaker_labels(
        self, uid: str, recording_id: str, labels: dict,
    ) -> dict | None:
        """Persist the recording's MANUAL speaker labels, returning the updated
        meta, or ``None`` when it does not exist for this uid (→ 404).

        Read-modify-write of meta.json ONLY — turns/analysis/derivatives are
        untouched. Storing manual labels in meta (not analysis.json) is the whole
        point: a re-analyze overwrites analysis.json but preserves meta, so a
        human's correction survives every re-run. An empty ``labels`` map removes
        the key entirely (a fully-cleared recording carries no manual-labels
        field). The caller has already validated/cleaned the map."""
        return await asyncio.to_thread(
            self._update_manual_speaker_labels_sync, uid, recording_id, labels,
        )

    def _update_manual_speaker_labels_sync(
        self, uid, recording_id, labels,
    ) -> dict | None:
        blob = self._bucket.blob(
            self._prefix(uid, recording_id) + "meta.json"
        )
        if not blob.exists():
            return None
        meta = json.loads(blob.download_as_bytes())
        if labels:
            meta["manual_speaker_labels"] = labels
        else:
            meta.pop("manual_speaker_labels", None)
        blob.upload_from_string(
            json.dumps(meta), content_type="application/json",
        )
        return meta

    async def update_manual_speaker_people(
        self, uid: str, recording_id: str, people: dict,
    ) -> dict | None:
        """Persist the recording's manual-person map (``{speaker: person_id}``
        — people labeling: which ENROLLED person a manually named speaker is),
        returning the updated meta, or ``None`` when the recording does not
        exist for this uid (→ 404). Same meta.json read-modify-write as
        ``update_manual_speaker_labels`` — stored beside the name map so a
        re-analyze never wipes it; an empty map removes the key."""
        return await asyncio.to_thread(
            self._update_manual_speaker_people_sync, uid, recording_id, people,
        )

    def _update_manual_speaker_people_sync(
        self, uid, recording_id, people,
    ) -> dict | None:
        blob = self._bucket.blob(
            self._prefix(uid, recording_id) + "meta.json"
        )
        if not blob.exists():
            return None
        meta = json.loads(blob.download_as_bytes())
        if people:
            meta["manual_speaker_people"] = people
        else:
            meta.pop("manual_speaker_people", None)
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
        # Tear down any share grants FIRST so no recipient's reverse index is left
        # dangling once the recording's own objects are gone (deleting the
        # recording must kill recipient access — spec §4). meta.json carries the
        # recipient list; best-effort per grant so one failure never blocks the
        # actual delete below.
        meta_blob = next(
            (b for b in blobs if b.name == prefix + "meta.json"), None,
        )
        if meta_blob is not None:
            try:
                meta = json.loads(meta_blob.download_as_bytes())
            except Exception:  # noqa: BLE001 — a corrupt meta must not block delete
                meta = {}
            for share in (meta.get("shares") or []):
                recipient_uid = share.get("uid")
                if not recipient_uid:
                    continue
                index_blob = self._bucket.blob(
                    self._share_blob_name(recipient_uid, uid, recording_id)
                )
                try:
                    if index_blob.exists():
                        index_blob.delete()
                except Exception:  # noqa: BLE001 — best-effort reverse-index cleanup
                    logger.warning(
                        "Failed to delete share index for recipient %s",
                        recipient_uid,
                    )
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

    # -- stored audio derivative bytes ------------------------------------
    async def get_audio_bytes(
        self, uid: str, recording_id: str,
    ) -> bytes | None:
        """Return the recording's ``audio.m4a`` derivative bytes, or ``None``
        when the recording (or its audio) is absent for this uid.

        Voice ENROLLMENT reads this: the stored AAC derivative is decoded back to
        PCM (audio_ingest.decode_to_pcm) and the enrolled speaker's turns are
        pooled + embedded. uid-scoped exactly like every other read here."""
        return await asyncio.to_thread(
            self._get_audio_bytes_sync, uid, recording_id,
        )

    def _get_audio_bytes_sync(self, uid, recording_id) -> bytes | None:
        blob = self._bucket.blob(self._prefix(uid, recording_id) + "audio.m4a")
        if not blob.exists():
            return None
        return blob.download_as_bytes()

    # -- re-analysis overwrite --------------------------------------------
    async def overwrite_analysis(
        self,
        uid: str,
        recording_id: str,
        *,
        turns: list[dict],
        analysis: dict,
        reanalyzed_at: str,
    ) -> dict | None:
        """Overwrite an existing recording's turns.json + analysis.json in place
        and stamp ``meta.reanalyzed_at`` — the persistence half of POST
        …/reanalyze. Returns the updated meta, or ``None`` when the recording
        does not exist for this uid (→ 404).

        Deliberately preserves everything else: the id, title, source, consent
        provenance, media_type, duration, and the stored audio/video derivatives
        are ALL untouched — re-analysis re-runs the pipeline over the SAME stored
        audio, it does not create a new recording."""
        return await asyncio.to_thread(
            self._overwrite_analysis_sync,
            uid, recording_id, turns, analysis, reanalyzed_at,
        )

    def _overwrite_analysis_sync(
        self, uid, recording_id, turns, analysis, reanalyzed_at,
    ) -> dict | None:
        prefix = self._prefix(uid, recording_id)
        meta_blob = self._bucket.blob(prefix + "meta.json")
        if not meta_blob.exists():
            return None
        meta = json.loads(meta_blob.download_as_bytes())
        meta["reanalyzed_at"] = reanalyzed_at
        meta_blob.upload_from_string(
            json.dumps(meta), content_type="application/json",
        )
        self._bucket.blob(prefix + "turns.json").upload_from_string(
            json.dumps(turns), content_type="application/json",
        )
        self._bucket.blob(prefix + "analysis.json").upload_from_string(
            json.dumps(analysis), content_type="application/json",
        )
        return meta

    # -- raw transcript (STT output, incl. word timings) --------------------
    # transcript.json is the transcriber's RAW utterances for the recording —
    # pre-diarization labels, per-word timings and all — saved at first
    # analysis so a re-analysis can skip STT yet still re-run every later
    # stage exactly as the first pass did (the local-diarization cross-check
    # needs the word timings to split a welded multi-voice utterance;
    # turns.json is the POST-diarization transcript without them). Absent on
    # recordings analyzed before this existed; re-analysis then transcribes
    # once more and writes it.
    async def save_transcript(
        self, uid: str, recording_id: str, transcript: list[dict],
    ) -> None:
        await asyncio.to_thread(
            self._save_transcript_sync, uid, recording_id, transcript,
        )

    def _save_transcript_sync(self, uid, recording_id, transcript) -> None:
        prefix = self._prefix(uid, recording_id)
        self._bucket.blob(prefix + "transcript.json").upload_from_string(
            json.dumps(transcript), content_type="application/json",
        )

    async def get_transcript(
        self, uid: str, recording_id: str,
    ) -> list[dict] | None:
        """The raw STT transcript, or ``None`` when none was stored."""
        return await asyncio.to_thread(self._get_transcript_sync, uid, recording_id)

    def _get_transcript_sync(self, uid, recording_id) -> list[dict] | None:
        blob = self._bucket.blob(self._prefix(uid, recording_id) + "transcript.json")
        if not blob.exists():
            return None
        data = json.loads(blob.download_as_bytes())
        return data if isinstance(data, list) else None

    # -- live sessions (Track 2) -------------------------------------------
    # A live coaching session lands WITHOUT audio (the phone did the
    # listening) — stored as meta + turns + analysis ONLY, under the same
    # recordings/{uid}/{id}/ prefix as an upload — unless the phone attaches
    # its mic recording afterwards (POST /sessions/{id}/audio → attach_audio
    # below writes audio.m4a and flips meta.media_type to "audio"). Every
    # list/detail/growth/share/delete path treats it exactly like a
    # recording; until audio is attached the media endpoints 404 honestly
    # (no audio.m4a object exists — see _open_media_stream_sync, which
    # returns None for a missing derivative).
    async def save_live_session(
        self,
        uid: str,
        recording_id: str,
        *,
        meta: dict,
        turns: list[dict],
        analysis: dict,
    ) -> dict:
        """Write (or REWRITE — ingest is idempotent on the caller-derived id)
        one live session's meta.json + turns.json + analysis.json.

        The caller mints ``recording_id`` deterministically from the session
        id so a re-POST of the same session lands on the same objects. When a
        meta.json already exists, the human-owned fields it carries —
        ``manual_speaker_labels`` / ``manual_speaker_people`` (a correction),
        ``shares`` (grants) and ``title`` when the new meta has none — are
        preserved: a phone re-sending its turns must never wipe what the user
        did afterwards. Likewise, when the existing meta says audio was
        ATTACHED (``media_type != "none"`` — see :meth:`attach_audio`), the
        audio-describing fields (``media_type``, ``stored_variants``,
        ``size_bytes``, ``original_bytes``, ``storage_note``,
        ``audio_attached_at``, and the decoded ``duration_seconds``) are
        carried over: the incoming meta always
        says "no audio", and a re-POST must not disown the audio.m4a that
        is still sitting next to it. Returns the meta actually written."""
        return await asyncio.to_thread(
            self._save_live_session_sync, uid, recording_id, meta, turns, analysis,
        )

    def _save_live_session_sync(self, uid, recording_id, meta, turns, analysis) -> dict:
        prefix = self._prefix(uid, recording_id)
        meta_blob = self._bucket.blob(prefix + "meta.json")
        written = dict(meta)
        if meta_blob.exists():
            existing = json.loads(meta_blob.download_as_bytes())
            for key in ("manual_speaker_labels", "manual_speaker_people", "shares"):
                if key in existing and key not in written:
                    written[key] = existing[key]
            if not written.get("title") and existing.get("title"):
                written["title"] = existing["title"]
            # Attached audio survives a re-POST (the phone's meta says "none";
            # the audio.m4a object is still there and must stay described).
            if existing.get("media_type") not in (None, "none"):
                for key in _ATTACHED_AUDIO_META_KEYS:
                    if key in existing:
                        written[key] = existing[key]
        meta_blob.upload_from_string(
            json.dumps(written), content_type="application/json",
        )
        self._bucket.blob(prefix + "turns.json").upload_from_string(
            json.dumps(turns), content_type="application/json",
        )
        self._bucket.blob(prefix + "analysis.json").upload_from_string(
            json.dumps(analysis), content_type="application/json",
        )
        return written

    async def attach_audio(
        self,
        uid: str,
        recording_id: str,
        *,
        audio_m4a: bytes,
        duration_seconds: float | None = None,
        original_bytes: int = 0,
    ) -> dict | None:
        """Attach the phone's mic recording to an already-stored live
        session (POST /sessions/{id}/audio): write its AAC derivative as
        ``audio.m4a`` under the recording prefix, then read-modify-write
        meta.json so the recording describes itself as an audio recording
        (``media_type: "audio"``, ``stored_variants``, sizes, the
        ``storage_note`` cleared, ``audio_attached_at`` stamped, and the
        decoded ``duration_seconds`` when given). Every "no audio" refusal
        downstream (media 404, reanalyze 422, voice enrollment) keys off
        media_type + the blob, so both flip together here.

        Idempotent: re-attaching overwrites audio.m4a (a retry after a
        dropped connection must not conflict). Returns the updated meta, or
        ``None`` when the recording does not exist for this uid (→ 404) —
        nothing is written in that case."""
        return await asyncio.to_thread(
            self._attach_audio_sync, uid, recording_id, audio_m4a,
            duration_seconds, original_bytes,
        )

    def _attach_audio_sync(
        self, uid, recording_id, audio_m4a, duration_seconds, original_bytes,
    ) -> dict | None:
        prefix = self._prefix(uid, recording_id)
        meta_blob = self._bucket.blob(prefix + "meta.json")
        if not meta_blob.exists():
            return None
        meta = json.loads(meta_blob.download_as_bytes())
        # Audio first, meta second: a crash between the two leaves an
        # undescribed blob (harmless — the next retry overwrites it) rather
        # than a meta that promises audio which is not there.
        self._bucket.blob(prefix + "audio.m4a").upload_from_string(
            audio_m4a, content_type="audio/mp4",
        )
        meta["media_type"] = "audio"
        meta["stored_variants"] = ["audio.m4a"]
        meta["size_bytes"] = len(audio_m4a)
        meta["original_bytes"] = original_bytes
        meta["storage_note"] = None
        if duration_seconds is not None:
            meta["duration_seconds"] = duration_seconds
        meta["audio_attached_at"] = datetime.now(timezone.utc).isoformat()
        meta_blob.upload_from_string(
            json.dumps(meta), content_type="application/json",
        )
        return meta

    async def update_analysis(
        self, uid: str, recording_id: str, analysis: dict,
    ) -> bool:
        """Replace analysis.json ONLY (turns + meta untouched, no
        ``reanalyzed_at`` stamp — this is not a re-analysis of stored audio
        but the post-ingest batch pass / a reflection landing on a live
        session). False when the recording does not exist for this uid."""
        return await asyncio.to_thread(
            self._update_analysis_sync, uid, recording_id, analysis,
        )

    def _update_analysis_sync(self, uid, recording_id, analysis) -> bool:
        prefix = self._prefix(uid, recording_id)
        if not self._bucket.blob(prefix + "meta.json").exists():
            return False
        self._bucket.blob(prefix + "analysis.json").upload_from_string(
            json.dumps(analysis), content_type="application/json",
        )
        return True

    # -- account-to-account sharing ---------------------------------------
    # A recording's OWNER can grant another account READ-ONLY access to it. The
    # grant is stored in TWO places so both directions are a cheap lookup:
    #
    #   1. On the owner's meta.json — ``shares: [{uid, email, created_at}]`` — so
    #      the owner's own list/detail can show who a recording is shared with
    #      (and revoke removes the entry). ``email`` is the RECIPIENT's email.
    #   2. A reverse-index object per grant, so the RECIPIENT's "shared with me"
    #      list is one prefix scan (never a full walk of every owner's bucket)::
    #
    #          shared/{recipient_uid}/{owner_uid}:{recording_id}.json
    #
    #      whose body carries ``{owner_uid, recording_id, owner_email, created_at}``
    #      — enough to render the recipient's row ("from linda@…") and to resolve
    #      the owning uid for a read WITHOUT trusting anything from the request.
    #
    # A recording_id is a uuid4 (no ``:``) so ``{owner_uid}:{recording_id}``
    # round-trips unambiguously on the recording-id suffix. Deleting a recording
    # or revoking a grant removes BOTH sides, so a stale grant can never outlive the
    # thing it points at.
    @staticmethod
    def _shares_prefix(recipient_uid: str) -> str:
        return f"shared/{recipient_uid}/"

    @staticmethod
    def _share_blob_name(
        recipient_uid: str, owner_uid: str, recording_id: str,
    ) -> str:
        return f"shared/{recipient_uid}/{owner_uid}:{recording_id}.json"

    async def add_share(
        self,
        owner_uid: str,
        recording_id: str,
        *,
        recipient_uid: str,
        recipient_email: str,
        owner_email: str | None,
    ) -> "list[dict] | None":
        """Grant ``recipient_uid`` read-only access to the owner's recording.

        Writes the reverse-index object AND appends ``{uid, email, created_at}`` to
        the owner meta.json ``shares`` list (idempotent — re-sharing to the same
        recipient refreshes the entry rather than duplicating it). Returns the
        updated shares list, or ``None`` when the recording does not exist for the
        owner (→ 404). The caller has already resolved + validated the recipient."""
        return await asyncio.to_thread(
            self._add_share_sync, owner_uid, recording_id,
            recipient_uid, recipient_email, owner_email,
        )

    def _add_share_sync(
        self, owner_uid, recording_id, recipient_uid, recipient_email, owner_email,
    ) -> "list[dict] | None":
        prefix = self._prefix(owner_uid, recording_id)
        meta_blob = self._bucket.blob(prefix + "meta.json")
        if not meta_blob.exists():
            return None
        meta = json.loads(meta_blob.download_as_bytes())
        created_at = datetime.now(timezone.utc).isoformat()
        shares = [
            s for s in (meta.get("shares") or [])
            if s.get("uid") != recipient_uid
        ]
        shares.append({
            "uid": recipient_uid,
            "email": recipient_email,
            "created_at": created_at,
        })
        meta["shares"] = shares
        meta_blob.upload_from_string(
            json.dumps(meta), content_type="application/json",
        )
        self._bucket.blob(
            self._share_blob_name(recipient_uid, owner_uid, recording_id)
        ).upload_from_string(
            json.dumps({
                "owner_uid": owner_uid,
                "recording_id": recording_id,
                "owner_email": owner_email,
                "created_at": created_at,
            }),
            content_type="application/json",
        )
        return shares

    async def remove_share(
        self, owner_uid: str, recording_id: str, recipient_uid: str,
    ) -> bool:
        """Revoke a recipient's access. Removes the meta.json ``shares`` entry AND
        the reverse-index object. Returns ``False`` when the recording does not
        exist for the owner (→ 404); ``True`` otherwise (idempotent — revoking a
        grant that was never present still succeeds, deleting nothing)."""
        return await asyncio.to_thread(
            self._remove_share_sync, owner_uid, recording_id, recipient_uid,
        )

    def _remove_share_sync(self, owner_uid, recording_id, recipient_uid) -> bool:
        prefix = self._prefix(owner_uid, recording_id)
        meta_blob = self._bucket.blob(prefix + "meta.json")
        if not meta_blob.exists():
            return False
        meta = json.loads(meta_blob.download_as_bytes())
        shares = [
            s for s in (meta.get("shares") or [])
            if s.get("uid") != recipient_uid
        ]
        meta["shares"] = shares
        meta_blob.upload_from_string(
            json.dumps(meta), content_type="application/json",
        )
        index_blob = self._bucket.blob(
            self._share_blob_name(recipient_uid, owner_uid, recording_id)
        )
        if index_blob.exists():
            index_blob.delete()
        return True

    async def find_share(
        self, recipient_uid: str, recording_id: str,
    ) -> "dict | None":
        """The reverse-index grant for ``recording_id`` shared with
        ``recipient_uid`` (``{owner_uid, owner_email, ...}``), or ``None`` when no
        such live grant exists.

        This is the per-request access check for every recipient read/write: it is
        a fresh GCS read, so a revoked grant (its index object deleted) is denied
        IMMEDIATELY on the next request. A recipient never supplies the owner uid —
        it is recovered here from the trusted index — so one user can never reach
        another's recording by guessing an id."""
        return await asyncio.to_thread(
            self._find_share_sync, recipient_uid, recording_id,
        )

    def _find_share_sync(self, recipient_uid, recording_id) -> "dict | None":
        # A recipient's grants are few; scan their own prefix and match the id
        # suffix. (Owner uid is unknown to the caller, so we cannot address the
        # object directly — but the scan is scoped to THIS recipient.)
        suffix = f":{recording_id}.json"
        prefix = self._shares_prefix(recipient_uid)
        for blob in self._bucket.list_blobs(prefix=prefix):
            if blob.name.endswith(suffix):
                return json.loads(blob.download_as_bytes())
        return None

    async def list_shared_with(self, recipient_uid: str) -> list[dict]:
        """Every recording shared WITH ``recipient_uid`` as summary metas (owner's
        meta + ``owner_email`` + ``shared: True``), newest-share first.

        One prefix scan of the recipient's reverse index, then a meta.json read per
        grant. A grant whose recording has since been deleted is skipped honestly
        (never a phantom row)."""
        return await asyncio.to_thread(self._list_shared_with_sync, recipient_uid)

    def _list_shared_with_sync(self, recipient_uid: str) -> list[dict]:
        prefix = self._shares_prefix(recipient_uid)
        grants = []
        for blob in self._bucket.list_blobs(prefix=prefix):
            if not blob.name.endswith(".json"):
                continue
            grants.append(json.loads(blob.download_as_bytes()))
        out: list[dict] = []
        for grant in grants:
            owner_uid = grant.get("owner_uid")
            recording_id = grant.get("recording_id")
            if not owner_uid or not recording_id:
                continue
            rec_prefix = self._prefix(owner_uid, recording_id)
            by_name = {}
            for blob in self._bucket.list_blobs(prefix=rec_prefix):
                by_name[blob.name[len(rec_prefix):]] = blob
            meta_blob = by_name.get("meta.json")
            if meta_blob is None:
                continue  # recording deleted since the grant — skip honestly
            meta = json.loads(meta_blob.download_as_bytes())
            meta["has_analysis"] = "analysis.json" in by_name
            meta["owner_email"] = grant.get("owner_email")
            meta["shared"] = True
            # The recipient must NEVER see who else the owner shared with.
            meta.pop("shares", None)
            meta["_shared_at"] = grant.get("created_at", "")
            out.append(meta)
        out.sort(key=lambda m: m.get("_shared_at", ""), reverse=True)
        for m in out:
            m.pop("_shared_at", None)
        return out

    # -- voiceprints -------------------------------------------------------
    # Enrolled voice signatures live in their OWN namespace, deliberately NOT
    # under ``recordings/`` — they must survive deleting the recording they were
    # enrolled from, and they are biometric data whose deletion is a first-class
    # user action (DELETE /voice/voiceprint, DELETE /voice/people/{id}), scoped
    # per uid. An account holds N named PEOPLE (Foundation B), one document each::
    #
    #     voiceprints/{uid}/{person_id}/profile.json   # {person_id, display_name,
    #                                                  #  is_self, embedding, ...}
    #
    # ``person_id`` is the reserved ``speaker_id.SELF_PERSON_ID`` ("self") for
    # the account owner's own voice and a client-chosen slug for anyone else
    # (validated against ``speaker_id.PERSON_ID_PATTERN`` at the endpoint — it
    # is a path segment, so it is never taken raw off the wire).
    #
    # LEGACY layout (before multi-person) was a single owner document::
    #
    #     voiceprints/{uid}/profile.json
    #
    # It keeps working through a READ-THROUGH SHIM rather than a migration:
    # ``read_voiceprint(uid, "self")`` and ``list_voiceprints`` fall back to the
    # legacy blob when no ``self/profile.json`` exists, viewing it as the self
    # person via ``speaker_id.as_person`` (pure). Chosen over copy-on-first-read
    # because (a) the house rule is that reads stay side-effect free (GET
    # /voice/profile already serves v1 docs through the v2 view without
    # rewriting them), (b) there is no migration state to track or get half
    # done, and (c) the legacy blob is retired naturally: the first WRITE of the
    # self person lands on the new path and then removes the legacy blob, so
    # there is never a moment with two live sources of truth for "self".
    #
    # Only the numeric signature + metadata is stored — never the user's audio.
    @staticmethod
    def _voiceprint_blob_name(uid: str, person_id: str = "self") -> str:
        return f"voiceprints/{uid}/{person_id}/profile.json"

    @staticmethod
    def _legacy_voiceprint_blob_name(uid: str) -> str:
        return f"voiceprints/{uid}/profile.json"

    @staticmethod
    def _voiceprints_prefix(uid: str) -> str:
        return f"voiceprints/{uid}/"

    def _read_json_blob(self, name: str) -> dict | None:
        blob = self._bucket.blob(name)
        if not blob.exists():
            return None
        return json.loads(blob.download_as_bytes())

    async def read_voiceprint(self, uid: str, person_id: str | None = None) -> dict | None:
        """One person's stored voiceprint document (the person view — always
        carries ``person_id``/``display_name``/``is_self``), or ``None`` when
        that person isn't enrolled. ``person_id`` defaults to the account
        owner ("self"), which is the only person the legacy single-document
        layout could hold — so it alone consults the legacy blob."""
        return await asyncio.to_thread(self._read_voiceprint_sync, uid, person_id)

    def _read_voiceprint_sync(self, uid, person_id=None) -> dict | None:
        pid = person_id or speaker_id.SELF_PERSON_ID
        doc = self._read_json_blob(self._voiceprint_blob_name(uid, pid))
        if doc is None and pid == speaker_id.SELF_PERSON_ID:
            doc = self._read_json_blob(self._legacy_voiceprint_blob_name(uid))
        if doc is None:
            return None
        return speaker_id.as_person(doc, person_id=pid)

    async def list_voiceprints(self, uid: str) -> list[dict]:
        """Every enrolled person's voiceprint document for ``uid`` (person
        views), the owner ("self") first, then partners by display name. The
        legacy owner blob counts as "self" only when no new-layout self
        document exists (the write path removes it, so both are never live)."""
        return await asyncio.to_thread(self._list_voiceprints_sync, uid)

    def _list_voiceprints_sync(self, uid) -> list[dict]:
        prefix = self._voiceprints_prefix(uid)
        legacy_name = self._legacy_voiceprint_blob_name(uid)
        by_person: dict[str, dict] = {}
        legacy_doc: dict | None = None
        for blob in self._bucket.list_blobs(prefix=prefix):
            if blob.name == legacy_name:
                legacy_doc = json.loads(blob.download_as_bytes())
                continue
            rest = blob.name[len(prefix):]
            parts = rest.split("/")
            if len(parts) != 2 or parts[1] != "profile.json" or not parts[0]:
                continue  # not a person document — ignore honestly
            by_person[parts[0]] = speaker_id.as_person(
                json.loads(blob.download_as_bytes()), person_id=parts[0],
            )
        if legacy_doc is not None and speaker_id.SELF_PERSON_ID not in by_person:
            by_person[speaker_id.SELF_PERSON_ID] = speaker_id.as_person(
                legacy_doc, person_id=speaker_id.SELF_PERSON_ID,
            )
        return sorted(
            by_person.values(),
            key=lambda p: (not p.get("is_self"), (p.get("display_name") or "").lower(), p["person_id"]),
        )

    async def write_voiceprint(self, uid: str, profile: dict) -> None:
        """Persist (overwrite) one person's voiceprint document, keyed by
        ``profile["person_id"]`` (absent → the owner, "self" — every
        pre-multi-person caller wrote the owner's print and still does).
        Writing the owner's document retires the legacy single-document blob
        (see the layout note above)."""
        await asyncio.to_thread(self._write_voiceprint_sync, uid, profile)

    def _write_voiceprint_sync(self, uid, profile) -> None:
        doc = speaker_id.as_person(profile)
        pid = doc["person_id"]
        self._bucket.blob(self._voiceprint_blob_name(uid, pid)).upload_from_string(
            json.dumps(doc), content_type="application/json",
        )
        if pid == speaker_id.SELF_PERSON_ID:
            legacy = self._bucket.blob(self._legacy_voiceprint_blob_name(uid))
            if legacy.exists():
                legacy.delete()

    async def delete_voiceprint(self, uid: str, person_id: str | None = None) -> bool:
        """Delete one person's voiceprint (default: the owner's). ``True`` when
        one existed and was removed, ``False`` when there was nothing to
        delete. Deletion is REAL — the biometric signature is gone, not
        tombstoned. Deleting the owner also removes the legacy blob if it is
        still the live copy."""
        return await asyncio.to_thread(self._delete_voiceprint_sync, uid, person_id)

    def _delete_voiceprint_sync(self, uid, person_id=None) -> bool:
        pid = person_id or speaker_id.SELF_PERSON_ID
        names = [self._voiceprint_blob_name(uid, pid)]
        if pid == speaker_id.SELF_PERSON_ID:
            names.append(self._legacy_voiceprint_blob_name(uid))
        deleted = False
        for name in names:
            blob = self._bucket.blob(name)
            if blob.exists():
                blob.delete()
                deleted = True
        return deleted

    # -- therapist link + therapist-private notes ---------------------------
    # A patient may name ONE therapist account (server/therapist_links.py,
    # routers/therapist.py). The link is stored twice so both sides are one
    # read: the patient's own document and a per-therapist reverse index::
    #
    #     therapist_links/{patient_uid}/link.json
    #     therapist_patients/{therapist_uid}/{patient_uid}.json
    #
    # Both bodies are the same ``link`` dict ({patient_uid, patient_email,
    # therapist_uid, therapist_email, status, auto_share, created_at,
    # accepted_at}). The link itself grants NOTHING — the per-episode share
    # grant (``add_share``) is still the only access mechanism; the link only
    # tells ingest whom to grant to. Notes a viewer keeps on an episode are
    # private to that viewer::
    #
    #     therapist_notes/{uid}/{episode_id}.json      # {text, updated_at}
    @staticmethod
    def _therapist_link_blob_name(patient_uid: str) -> str:
        return f"therapist_links/{patient_uid}/link.json"

    @staticmethod
    def _therapist_patients_prefix(therapist_uid: str) -> str:
        return f"therapist_patients/{therapist_uid}/"

    @staticmethod
    def _therapist_patient_blob_name(therapist_uid: str, patient_uid: str) -> str:
        return f"therapist_patients/{therapist_uid}/{patient_uid}.json"

    @staticmethod
    def _therapist_note_blob_name(uid: str, episode_id: str) -> str:
        return f"therapist_notes/{uid}/{episode_id}.json"

    @staticmethod
    def _therapist_notes_prefix(uid: str) -> str:
        return f"therapist_notes/{uid}/"

    async def read_therapist_link(self, patient_uid: str) -> dict | None:
        """The patient's therapist link, or ``None`` when none is set."""
        return await asyncio.to_thread(
            self._read_json_blob, self._therapist_link_blob_name(patient_uid),
        )

    async def write_therapist_link(self, patient_uid: str, link: dict) -> None:
        """Set (or replace) the patient's link. A previous link to a DIFFERENT
        therapist has its reverse-index entry removed so the old therapist's
        patient list no longer shows this patient."""
        await asyncio.to_thread(self._write_therapist_link_sync, patient_uid, link)

    def _write_therapist_link_sync(self, patient_uid: str, link: dict) -> None:
        previous = self._read_json_blob(self._therapist_link_blob_name(patient_uid))
        body = json.dumps(link)
        self._bucket.blob(self._therapist_link_blob_name(patient_uid)).upload_from_string(
            body, content_type="application/json",
        )
        self._bucket.blob(
            self._therapist_patient_blob_name(link["therapist_uid"], patient_uid)
        ).upload_from_string(body, content_type="application/json")
        old_t = (previous or {}).get("therapist_uid")
        if old_t and old_t != link["therapist_uid"]:
            stale = self._bucket.blob(self._therapist_patient_blob_name(old_t, patient_uid))
            if stale.exists():
                stale.delete()

    async def delete_therapist_link(self, patient_uid: str) -> bool:
        """Remove the link (both sides). ``True`` when one existed."""
        return await asyncio.to_thread(self._delete_therapist_link_sync, patient_uid)

    def _delete_therapist_link_sync(self, patient_uid: str) -> bool:
        name = self._therapist_link_blob_name(patient_uid)
        link = self._read_json_blob(name)
        if link is None:
            return False
        self._bucket.blob(name).delete()
        reverse = self._bucket.blob(
            self._therapist_patient_blob_name(link.get("therapist_uid", ""), patient_uid)
        )
        if reverse.exists():
            reverse.delete()
        return True

    async def list_therapist_patients(self, therapist_uid: str) -> list[dict]:
        """Every link naming ``therapist_uid`` as the therapist (one prefix
        scan of the reverse index), oldest first."""
        return await asyncio.to_thread(self._list_therapist_patients_sync, therapist_uid)

    def _list_therapist_patients_sync(self, therapist_uid: str) -> list[dict]:
        out: list[dict] = []
        for blob in self._bucket.list_blobs(prefix=self._therapist_patients_prefix(therapist_uid)):
            if not blob.name.endswith(".json"):
                continue
            try:
                out.append(json.loads(blob.download_as_bytes()))
            except (ValueError, TypeError):
                continue
        out.sort(key=lambda link: link.get("created_at") or "")
        return out

    async def read_therapist_note(self, uid: str, episode_id: str) -> dict | None:
        return await asyncio.to_thread(
            self._read_json_blob, self._therapist_note_blob_name(uid, episode_id),
        )

    async def write_therapist_note(self, uid: str, episode_id: str, note: dict) -> None:
        def _write() -> None:
            self._bucket.blob(self._therapist_note_blob_name(uid, episode_id)).upload_from_string(
                json.dumps(note), content_type="application/json",
            )
        await asyncio.to_thread(_write)

    async def delete_therapist_note(self, uid: str, episode_id: str) -> bool:
        def _delete() -> bool:
            blob = self._bucket.blob(self._therapist_note_blob_name(uid, episode_id))
            if not blob.exists():
                return False
            blob.delete()
            return True
        return await asyncio.to_thread(_delete)

    async def list_therapist_notes(self, uid: str) -> dict[str, dict]:
        """``{episode_id: note}`` for every note the viewer keeps — one prefix scan."""
        def _list() -> dict[str, dict]:
            prefix = self._therapist_notes_prefix(uid)
            out: dict[str, dict] = {}
            for blob in self._bucket.list_blobs(prefix=prefix):
                name = blob.name[len(prefix):]
                if not name.endswith(".json"):
                    continue
                try:
                    out[name[:-5]] = json.loads(blob.download_as_bytes())
                except (ValueError, TypeError):
                    continue
            return out
        return await asyncio.to_thread(_list)

    # -- account deletion (DELETE /me) -------------------------------------
    # The bulk reads/deletes ``account_deletion.delete_account_data`` composes.
    # Everything here is a WHOLE-NAMESPACE operation for ONE uid, so each takes
    # the uid and builds its own prefix from the same ``_*_prefix`` helpers the
    # rest of the class uses — a caller can never hand in a raw path, and no
    # method here can reach outside ``.../{uid}/``.

    def _delete_prefix_sync(self, prefix: str) -> int:
        """Delete every object under ``prefix``; return how many were removed.

        Refuses an empty or non-uid-scoped prefix outright (a guard against a
        future caller passing "" and wiping the bucket): every namespace in
        this store is ``<kind>/<uid>/…``, so a legal prefix always has at
        least two path segments and a trailing slash.
        """
        if not prefix.endswith("/") or prefix.count("/") < 2:
            raise ValueError(f"refusing to bulk-delete unscoped prefix {prefix!r}")
        removed = 0
        for blob in list(self._bucket.list_blobs(prefix=prefix)):
            blob.delete()
            removed += 1
        return removed

    async def list_received_share_grants(self, recipient_uid: str) -> list[dict]:
        """The RAW reverse-index grants issued TO ``recipient_uid``:
        ``[{owner_uid, recording_id, owner_email, created_at}, …]``.

        Unlike :meth:`list_shared_with` (which resolves each grant to the
        owner's meta for display, and silently skips grants whose recording is
        gone), this returns the index entries themselves — deletion needs the
        ``(owner_uid, recording_id)`` pairs so it can revoke each grant on the
        OWNER's side too, including entries whose recording already vanished."""
        def _list() -> list[dict]:
            out: list[dict] = []
            for blob in self._bucket.list_blobs(
                prefix=self._shares_prefix(recipient_uid)
            ):
                if not blob.name.endswith(".json"):
                    continue
                try:
                    out.append(json.loads(blob.download_as_bytes()))
                except (ValueError, TypeError):
                    continue
            return out
        return await asyncio.to_thread(_list)

    async def list_recording_share_recipients(self, uid: str) -> dict[str, list[str]]:
        """``{recording_id: [recipient_uid, …]}`` for every recording ``uid``
        owns that has live share grants on it.

        Read BEFORE the recordings are deleted: it is the only record of who
        could see each episode, and therefore of whose private notes about it
        must go with it (the shared-data rule — see
        ``account_deletion``'s module docstring)."""
        def _list() -> dict[str, list[str]]:
            out: dict[str, list[str]] = {}
            prefix = self._prefix(uid)
            for blob in self._bucket.list_blobs(prefix=prefix):
                rel = blob.name[len(prefix):]
                recording_id, _, fname = rel.partition("/")
                if fname != "meta.json" or not recording_id:
                    continue
                try:
                    meta = json.loads(blob.download_as_bytes())
                except (ValueError, TypeError):
                    continue
                recipients = [
                    s.get("uid") for s in (meta.get("shares") or []) if s.get("uid")
                ]
                if recipients:
                    out[recording_id] = recipients
            return out
        return await asyncio.to_thread(_list)

    async def delete_all_recordings(self, uid: str) -> int:
        """Delete every recording ``uid`` owns — audio, video, meta, turns,
        analysis — plus each recipient's reverse-index grant. Returns the
        number of RECORDINGS removed (not objects).

        Per recording this is :meth:`delete_recording`, so the grant teardown
        is the same code path a single delete uses; nothing about revocation
        can drift between the two."""
        ids = await asyncio.to_thread(self._list_recording_ids_sync, uid)
        deleted = 0
        for recording_id in ids:
            if await self.delete_recording(uid, recording_id):
                deleted += 1
        # Sweep any partial recording (no meta.json) the id scan above skipped,
        # so the namespace is provably empty afterwards rather than "empty of
        # the things we could name".
        await asyncio.to_thread(self._delete_prefix_sync, self._prefix(uid))
        return deleted

    def _list_recording_ids_sync(self, uid: str) -> list[str]:
        prefix = self._prefix(uid)
        ids: set[str] = set()
        for blob in self._bucket.list_blobs(prefix=prefix):
            recording_id, _, fname = blob.name[len(prefix):].partition("/")
            if recording_id and fname:
                ids.add(recording_id)
        return sorted(ids)

    async def delete_all_voiceprints(self, uid: str) -> int:
        """Delete every enrolled person's voiceprint for ``uid`` (self, every
        named partner, and the legacy single-document layout). Returns how many
        PEOPLE were removed."""
        people = await self.list_voiceprints(uid)
        await asyncio.to_thread(self._delete_prefix_sync, self._voiceprints_prefix(uid))
        return len(people)

    async def delete_all_uploads(self, uid: str) -> int:
        """Delete every in-progress chunked upload (manifest + parts) for
        ``uid``. Returns the number of OBJECTS removed — an abandoned upload
        has no stable id worth counting."""
        return await asyncio.to_thread(self._delete_prefix_sync, f"uploads/{uid}/")

    async def delete_all_jobs(self, uid: str) -> int:
        """Delete every stored analysis-job state doc for ``uid``. Returns the
        number of objects removed."""
        return await asyncio.to_thread(self._delete_prefix_sync, f"jobs/{uid}/")

    async def delete_all_therapist_notes(self, uid: str) -> int:
        """Delete every note ``uid`` wrote as a viewer. Returns how many."""
        notes = await self.list_therapist_notes(uid)
        await asyncio.to_thread(self._delete_prefix_sync, self._therapist_notes_prefix(uid))
        return len(notes)

    async def delete_therapist_patient_entry(
        self, therapist_uid: str, patient_uid: str,
    ) -> bool:
        """Remove ONE row of a therapist's reverse patient index. ``True`` when
        one existed. Used when the THERAPIST's account is deleted: each linked
        patient's own ``therapist_links/{patient}/link.json`` is removed by
        :meth:`delete_therapist_link`, and this clears the matching row."""
        def _delete() -> bool:
            blob = self._bucket.blob(
                self._therapist_patient_blob_name(therapist_uid, patient_uid)
            )
            if not blob.exists():
                return False
            blob.delete()
            return True
        return await asyncio.to_thread(_delete)
