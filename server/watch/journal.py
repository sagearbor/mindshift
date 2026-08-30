"""Journal-capture self-filtering (watch A/B journal mode, server half).

A *journal capture* is a retro-buffer snapshot the watch uploads automatically
every ~5 minutes while its Journal toggle is on, marked with labels
``{"journal": true, "interval_s": N}`` (the watch PUTs those labels BEFORE the
audio, so the audio-upload success path in ``watch/routers/captures.py`` can
see them and spawn :func:`process_journal_capture` here — same fire-and-forget
task pattern as ``watch/routers/ws.py``'s ``_spawn_live_session_analysis``).

Privacy contract (load-bearing, not aspirational):

* Raw journal audio is retained AT MOST :data:`JOURNAL_RETENTION_HOURS` (48 h):
  every processed journal capture is stamped ``labels["journal_expires_at"]``,
  and each new journal capture for an account triggers
  :func:`cleanup_expired_journal_captures`, which deletes expired journal
  captures blob-FIRST (metadata survives a failed blob delete so nothing is
  ever orphaned — same D4 ordering as ``DELETE /captures/{id}``).
* Only the wearer's OWN voice is ever meant to outlive that window: the
  filter runs ``watch/diarize.py``'s VAD + voiceprint speaker assignment
  against the account's enrolled voiceprint and records the "self" spans
  (padded ±:data:`SELF_SEGMENT_PAD_SECONDS`, merged) in the capture's labels.
* No enrolled voiceprint → NOTHING is guessed: the capture is labeled
  ``{"journal_status": "no_voiceprint"}`` and kept (until expiry) so it can be
  reprocessed once the wearer enrolls — honesty over fabrication, per the
  repo-wide degradation doctrine.

Ownership boundary (deliberate stop): appending the kept "self" audio to the
account's journal recording in the MAIN recordings store requires
``server/main.py``'s upload pipeline (``_analyze_recording_bytes``'s
transcode + analysis + ``recordings_store.save_recording``), which lives
outside the watch domain and takes a main.py hook to reach correctly. This
module therefore records the filter RESULT on the capture —
``labels["self_segments"]`` / ``labels["self_seconds"]`` /
``labels["journal_status"]`` — and stops there; the recording-creation half
reads those labels from a future main.py-side hook.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import numpy as np

from watch.blobs import BlobStore
from watch.diarize import DiarizationService
from watch.store import LiveSessionStore

# (account_id, wav_bytes, title, context) -> awaitable. Wired by server/main.py;
# turns filtered self audio into a stored, analyzed recording.
RecordingSink = "Callable[[str, bytes, str, str], Awaitable[None]]"


def _journal_title(captured_at_iso: str) -> str:
    """"Journal (watch) — 2026-08-30 14:05" from the device's captured_at
    (opaque ISO; fall back to the raw string's date-ish prefix on parse
    failure — a title must never break the pipeline)."""
    try:
        from datetime import datetime as _dt

        t = _dt.fromisoformat(captured_at_iso.replace("Z", "+00:00"))
        return f"Journal (watch) — {t.strftime('%Y-%m-%d %H:%M')}"
    except Exception:  # noqa: BLE001
        return f"Journal (watch) — {captured_at_iso[:16]}"

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # PCM16 mono

# Keep a little context around each matched "self" span so word onsets/offsets
# aren't clipped mid-phoneme by the VAD's frame boundaries.
SELF_SEGMENT_PAD_SECONDS = 0.5

# Below this much total "self" audio a journal snapshot has nothing worth
# keeping as a recording (a cough's worth of voice, not a journal entry).
MIN_SELF_SECONDS = 3.0

JOURNAL_RETENTION_HOURS = 48


def _now() -> datetime:
    return datetime.now(timezone.utc)


def is_journal_capture(labels: dict) -> bool:
    """True only for the exact boolean the watch writes — a truthy string
    ("yes") must not opt a capture into journal processing/expiry."""
    return labels.get("journal") is True


def merge_padded_self_segments(
    turns: list[tuple[str, float, float]],
    *,
    total_s: float,
    pad_s: float = SELF_SEGMENT_PAD_SECONDS,
) -> list[tuple[float, float]]:
    """The "self" spans from diarized ``turns``, padded ±``pad_s`` (clamped to
    ``[0, total_s]``) and merged where the padding makes them touch/overlap.

    Pure; returns spans ascending by start. Non-"self" turns (other-N, or
    anything else) contribute nothing — they are exactly what the journal
    throws away."""
    spans = sorted(
        (max(0.0, start - pad_s), min(total_s, end + pad_s))
        for label, start, end in turns
        if label == "self" and end > start
    )
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _expiry_for(cap, now: datetime) -> datetime | None:
    """When this journal capture's raw audio must be gone: its stamped
    ``journal_expires_at`` when parseable, else ``received_at`` + 48 h, else
    ``None`` (never delete on a guess)."""
    raw = cap.labels.get("journal_expires_at")
    if raw:
        try:
            return datetime.fromisoformat(raw)
        except (TypeError, ValueError):
            pass
    try:
        received = datetime.fromisoformat(cap.received_at)
    except (TypeError, ValueError):
        return None
    if received.tzinfo is None:
        received = received.replace(tzinfo=timezone.utc)
    return received + timedelta(hours=JOURNAL_RETENTION_HOURS)


async def cleanup_expired_journal_captures(
    account_id: str,
    store: LiveSessionStore,
    blobs: BlobStore | None,
    *,
    now: datetime | None = None,
    keep_id: str | None = None,
) -> int:
    """Delete this account's journal captures past their 48 h retention.

    Runs when a NEW journal capture arrives for the account (spawned from
    :func:`process_journal_capture`), so retention needs no separate cron.
    Blob-FIRST, exactly like ``DELETE /captures/{id}``: a failed (or
    unconfigured) blob delete leaves the metadata doc in place — the pointer
    to still-existing audio must never vanish first. Non-journal captures are
    NEVER touched, whatever their age. Returns how many captures were fully
    deleted.
    """
    from watch.routers.captures import capture_key  # lazy: captures.py imports this module

    now = now or _now()
    deleted = 0
    for cap in await store.list_captures(account_id):
        if cap.id == keep_id or not is_journal_capture(cap.labels):
            continue
        expires = _expiry_for(cap, now)
        if expires is None or expires > now:
            continue
        if cap.audio_uri is not None:
            if blobs is None:
                continue  # can't delete the audio → keep the metadata pointing at it
            try:
                await blobs.delete(capture_key(account_id, cap.id))
            except Exception:  # noqa: BLE001 — metadata must NOT vanish while audio survives
                logger.warning(
                    "journal cleanup: blob delete failed for capture %s — keeping metadata", cap.id
                )
                continue
        await store.delete_capture(cap.id)
        deleted += 1
    return deleted


async def process_journal_capture(
    capture_id: str,
    store: LiveSessionStore,
    blobs: BlobStore | None,
    diarizer: DiarizationService | None,
    recording_sink: "RecordingSink | None" = None,
) -> None:
    """Self-filter one freshly-uploaded journal capture and stamp its labels.

    Reads the raw PCM back from blob storage, diarizes it against the
    account's enrolled voiceprint, and writes the result INTO the capture's
    labels (full-document replace discipline: the capture is re-read here and
    only its labels are updated — see captures.py's CRITICAL SEAM note):

    * ``journal_expires_at`` — always stamped (the 48 h retention clock).
    * ``journal_status`` — ``"no_voiceprint"`` (kept, honestly unfiltered),
      ``"diarization_unavailable"`` (no diarizer/audio to filter with),
      ``"diarization_failed"``, ``"self_filtered"`` (≥ 3 s of self audio
      found), or ``"self_filtered_below_minimum"``.
    * ``self_segments`` / ``self_seconds`` — the kept spans, only when
      diarization actually ran.

    ``recording_sink`` (wired by server/main.py) receives the concatenated
    SELF audio as a 16 kHz WAV whenever the filter keeps >= MIN_SELF_SECONDS —
    main.py turns it into a "Journal (watch) — ..." recording with the same
    analysis an upload gets. None (tests, keyless dev) stops at the labels. Finishes with :func:`cleanup_expired_journal_captures`
    for the account. Never raises (fire-and-forget contract).
    """
    from watch.routers.captures import capture_key  # lazy: captures.py imports this module

    cap = await store.get_capture(capture_id)
    if cap is None or not is_journal_capture(cap.labels) or cap.status != "stored":
        return
    account = cap.account_id
    now = _now()

    labels = dict(cap.labels)
    labels["journal_expires_at"] = (now + timedelta(hours=JOURNAL_RETENTION_HOURS)).isoformat()

    pcm = await blobs.get(capture_key(account, cap.id)) if blobs is not None else None
    profile = await store.get_speaker_profile(account)

    if profile is None:
        # No honest basis to say which voice is the wearer — keep the capture
        # (until expiry) so enrolling later allows a reprocess; guess nothing.
        labels["journal_status"] = "no_voiceprint"
    elif pcm is None or diarizer is None:
        labels["journal_status"] = "diarization_unavailable"
    else:
        self_print = np.asarray(profile.embedding, dtype=np.float32)
        try:
            turns = await asyncio.to_thread(diarizer.diarize, pcm, SAMPLE_RATE, self_print)
        except Exception:  # noqa: BLE001 — a broken diarizer must never break the hook
            logger.exception("journal: diarization failed for capture %s", capture_id)
            turns = None
        if turns is None:
            labels["journal_status"] = "diarization_failed"
        else:
            total_s = len(pcm) / BYTES_PER_SECOND
            segments = merge_padded_self_segments(turns, total_s=total_s)
            self_seconds = round(sum(end - start for start, end in segments), 3)
            labels["self_segments"] = [[round(s, 3), round(e, 3)] for s, e in segments]
            labels["self_seconds"] = self_seconds
            labels["journal_status"] = (
                "self_filtered" if self_seconds >= MIN_SELF_SECONDS
                else "self_filtered_below_minimum"
            )
            if labels["journal_status"] == "self_filtered" and recording_sink is not None:
                try:
                    import audio_ingest

                    pieces = [
                        pcm[int(s0 * BYTES_PER_SECOND // 2) * 2:int(e0 * BYTES_PER_SECOND // 2) * 2]
                        for s0, e0 in segments
                    ]
                    ints = np.frombuffer(b"".join(pieces), dtype="<i2")
                    wav = audio_ingest.pcm_to_wav16(
                        ints.astype(np.float32) / 32768.0, SAMPLE_RATE,
                    )
                    await recording_sink(
                        account,
                        wav,
                        _journal_title(cap.captured_at),
                        f"Watch voice journal: {len(segments)} stretch(es) of the "
                        f"wearer's own voice, {self_seconds:.0f}s total, filtered from a "
                        f"{total_s:.0f}s capture. Other voices were discarded before storage.",
                    )
                    labels["journal_recording"] = "spawned"
                except Exception:  # noqa: BLE001 — the sink must never break the labels write
                    logger.exception("journal: recording sink failed for capture %s", capture_id)
                    labels["journal_recording"] = "failed"

    cap.labels = labels
    cap.labels_updated_at = now.isoformat()
    await store.put_capture(cap)

    await cleanup_expired_journal_captures(account, store, blobs, now=now, keep_id=cap.id)


# --- fire-and-forget spawn (mirrors watch/routers/ws.py's analysis spawn) ------------------

# asyncio.create_task holds only a WEAK reference — same documented gotcha
# ws.py guards against: keep a strong reference until each task completes.
_background_tasks: set[asyncio.Task] = set()


def _on_task_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Background journal processing task failed unexpectedly", exc_info=exc)


async def _run(capture_id: str, store, blobs, diarizer, recording_sink=None) -> None:
    try:
        await process_journal_capture(capture_id, store, blobs, diarizer, recording_sink)
    except Exception:  # noqa: BLE001 — background task; must log, never raise
        logger.exception("Journal processing failed for capture %s", capture_id)


def spawn_journal_processing(
    capture_id: str,
    store: LiveSessionStore,
    blobs: BlobStore | None,
    diarizer: DiarizationService | None,
    recording_sink: "RecordingSink | None" = None,
) -> None:
    """Fire-and-forget :func:`process_journal_capture`, keeping a strong task
    reference until completion. Called from the captures router's audio-upload
    success path for captures labeled ``journal: true``."""
    task = asyncio.create_task(_run(capture_id, store, blobs, diarizer, recording_sink))
    _background_tasks.add(task)
    task.add_done_callback(_on_task_done)
