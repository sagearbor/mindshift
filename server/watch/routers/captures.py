# Ported from gauge@2157433 server/captures_api.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B7):
# * Episode -> LiveSession per the locked "episode" rename map:
#   `EpisodeStore` -> `LiveSessionStore`. Capture keeps its name entirely —
#   it is explicitly OUT of the rename (LOCKED, see global-constraints.md)
#   — so `Capture`/`ConsentRecord(kind="capture")`/`capture_key`/
#   `inflate_capped`/every route path (`/captures*`) and every cap
#   (`MAX_CAPTURE_BYTES`, `MAX_CAPTURE_SECONDS`, `MAX_LABELS_BYTES`) are
#   verbatim.
# * `make_captures_router`'s closure-factory signature is now
#   `(store, blobs, full_auth_dep)` per the B7 task brief — gauge's was
#   `(store, auth, blobs=None)`. `blobs` is no longer optional-with-a-
#   default: the caller (watch/testing.py, and later the real app assembly)
#   always passes an explicit value (a real `BlobStore` or `None`), same
#   "no silent env-var fallback" posture B6's groups router adopted for its
#   two auth deps. `full_auth_dep` mirrors B5/B6's naming for the
#   `require_full_auth`-wrapped dependency — gauge's captures router already
#   required a non-legacy principal on every route (the I2/I3 controller
#   ruling), so this is the same behavior made explicit in the param name.
"""Captures API: create, upload audio, list, label, download retro-capture
clips (gauge Task 15).

A Capture is a short clip of the wearer's OWN audio, captured on-device
(e.g. triggered by a volume spike) and uploaded later on request — never a
recording of the other party. Consent enforcement lives here: creation
requires ``attested is True`` (mirrors ``rest.py``'s ``label_participant``)
and mints a ``ConsentRecord(kind="capture")`` server-side, never
client-supplied.

CRITICAL SEAM (flagged by gauge Task 14's reviewer): ``store.put_capture``
is a FULL-DOCUMENT replace, not a partial update. Every handler below that
mutates a capture reads the full ``Capture`` via ``_get_owned_or_404``,
mutates only the fields it owns, and writes the WHOLE object back —
otherwise a later write (e.g. a label update) would silently drop earlier
fields (e.g. ``consents``), which would be the worst bug this task could
ship.
"""

from __future__ import annotations

import json
import uuid
import zlib
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, StrictBool

from watch.auth import AuthDep, Principal
from watch.blobs import BlobStore
from watch.models import SELF_PARTICIPANT_ID, Capture, ConsentRecord
from watch.store import LiveSessionStore

# 2-5 min of PCM16 mono 16 kHz is 3.8-9.6 MB; the cap leaves headroom to ~10 min
# and bounds both the in-process buffer and a gzip bomb's inflated size.
MAX_CAPTURE_BYTES = 20_000_000
MAX_CAPTURE_SECONDS = 900.0
# Serialized ground-truth labels; keeps the Firestore capture doc far under 1 MiB.
MAX_LABELS_BYTES = 100_000
CAPTURE_SAMPLE_RATE = 16000

# Chunk size for incremental gzip inflation — arbitrary but small relative to
# MAX_CAPTURE_BYTES so a bomb's running total is checked often, not just once
# at the end.
_INFLATE_CHUNK = 65536


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CreateCaptureRequest(BaseModel):
    captured_at: str
    duration_s: float = Field(gt=0, le=MAX_CAPTURE_SECONDS)
    trigger: str = ""
    device: str | None = None
    sample_rate: int = CAPTURE_SAMPLE_RATE
    # StrictBool for the same reason LabelRequest uses it (rest.py's
    # label_participant): "yes"/1 must NOT coerce past a consent gate.
    attested: StrictBool


def capture_key(account_id: str, capture_id: str) -> str:
    """Object key: f"captures/{account_id}/{capture_id}.pcm" — always raw
    PCM16 at rest, whatever the transport encoding was."""
    return f"captures/{account_id}/{capture_id}.pcm"


def inflate_capped(raw: bytes, max_bytes: int) -> bytes:
    """gzip-inflate incrementally, raising HTTPException(413) the moment the
    running output exceeds max_bytes — a bomb is a rejected request, not an
    OOM. Malformed gzip -> HTTPException(422, "body is not valid gzip data")."""
    decompressor = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)  # gzip format
    out = bytearray()
    try:
        # No unconsumed_tail drain loop needed: decompress(..., max_bytes + 1)
        # caps THIS call's output at max_bytes + 1, so any call that leaves
        # unconsumed_tail behind has already pushed len(out) past max_bytes —
        # the check right below always fires first, making a drain loop
        # unreachable dead code.
        for start in range(0, len(raw), _INFLATE_CHUNK):
            chunk = decompressor.decompress(raw[start:start + _INFLATE_CHUNK], max_bytes + 1)
            out.extend(chunk)
            if len(out) > max_bytes:
                raise HTTPException(
                    status_code=413, detail="capture audio exceeds the size limit"
                )
        out.extend(decompressor.flush())
        if len(out) > max_bytes:
            raise HTTPException(status_code=413, detail="capture audio exceeds the size limit")
    except HTTPException:
        raise
    except zlib.error as exc:
        raise HTTPException(status_code=422, detail="body is not valid gzip data") from exc
    return bytes(out)


async def _get_owned_or_404(store: LiveSessionStore, capture_id: str, account: str) -> Capture:
    cap = await store.get_capture(capture_id)
    if cap is None:
        raise HTTPException(status_code=404, detail="capture not found")
    if cap.account_id != account:
        raise HTTPException(status_code=403, detail="only the capture owner may do this")
    return cap


def make_captures_router(
    store: LiveSessionStore, blobs: BlobStore | None, full_auth_dep: AuthDep
) -> APIRouter:
    router = APIRouter()

    @router.post("/captures", response_model=Capture)
    async def create_capture(
        body: CreateCaptureRequest, principal: Principal = Depends(full_auth_dep)
    ) -> Capture:
        if body.attested is not True:
            raise HTTPException(status_code=422, detail="attested must be true")
        account = principal.account_id
        now = _now_iso()
        capture = Capture(
            id=uuid.uuid4().hex,
            account_id=account,
            device=body.device,
            captured_at=body.captured_at,
            received_at=now,
            duration_s=body.duration_s,
            trigger=body.trigger,
            sample_rate=body.sample_rate,
            status="awaiting_audio",
            consents=[ConsentRecord(
                id=uuid.uuid4().hex,
                participant_id=SELF_PARTICIPANT_ID,
                kind="capture",
                attested_by=account,
                confirmed=False,
                ts=now,
            )],
        )
        await store.put_capture(capture)
        return capture

    @router.put("/captures/{capture_id}/audio", response_model=Capture)
    async def upload_audio(
        capture_id: str, request: Request, principal: Principal = Depends(full_auth_dep)
    ) -> Capture:
        account = principal.account_id
        cap = await _get_owned_or_404(store, capture_id, account)

        if cap.status == "stored":
            raise HTTPException(status_code=409, detail="capture audio is already stored")
        if blobs is None:
            raise HTTPException(status_code=503, detail="capture storage is not configured")

        raw = await request.body()
        if len(raw) == 0:
            raise HTTPException(status_code=422, detail="empty upload body")
        if len(raw) > MAX_CAPTURE_BYTES:
            raise HTTPException(
                status_code=413, detail="capture audio exceeds the size limit"
            )

        is_gzip = request.headers.get("content-encoding", "").lower() == "gzip"
        if is_gzip:
            pcm = inflate_capped(raw, MAX_CAPTURE_BYTES)
        else:
            pcm = raw

        uri = await blobs.put(capture_key(account, capture_id), pcm)

        cap.audio_uri = uri
        cap.audio_bytes = len(pcm)
        cap.upload_encoding = "gzip" if is_gzip else None
        cap.status = "stored"
        await store.put_capture(cap)
        return cap

    @router.get("/captures", response_model=list[Capture])
    async def list_captures(principal: Principal = Depends(full_auth_dep)) -> list[Capture]:
        return await store.list_captures(principal.account_id)

    @router.get("/captures/{capture_id}", response_model=Capture)
    async def get_capture(
        capture_id: str, principal: Principal = Depends(full_auth_dep)
    ) -> Capture:
        return await _get_owned_or_404(store, capture_id, principal.account_id)

    @router.get("/captures/{capture_id}/audio")
    async def download_audio(capture_id: str, principal: Principal = Depends(full_auth_dep)):
        account = principal.account_id
        cap = await _get_owned_or_404(store, capture_id, account)
        if cap.status != "stored":
            raise HTTPException(
                status_code=409, detail="capture audio has not been uploaded"
            )
        if blobs is None:
            raise HTTPException(status_code=503, detail="capture storage is not configured")

        pcm = await blobs.get(capture_key(account, capture_id))
        if pcm is None:
            raise HTTPException(
                status_code=404, detail="capture audio is missing from storage"
            )
        return Response(content=pcm, media_type="application/octet-stream")

    @router.delete("/captures/{capture_id}", status_code=204)
    async def delete_capture(
        capture_id: str, principal: Principal = Depends(full_auth_dep)
    ) -> None:
        # `full_auth_dep` here is already the full-auth dependency
        # watch/testing.py (and the real app assembly) passes to
        # make_captures_router — every captures route is full-auth by the
        # I2/I3 ruling (see the module docstring / test_captures_reject_
        # legacy_account_param... above), so this is owner-only AND
        # full-auth-only with no extra wrapping needed.
        #
        # Gauge Wave B Task 8 (D4): blob-first, never orphan audio. If a
        # Capture has audio_uri set, the stored blob is deleted BEFORE the
        # metadata doc — a failed blob delete (502) leaves the metadata in
        # place so the wearer can retry, rather than losing the pointer to
        # audio that still exists in storage. Only once the blob is
        # confirmed gone (or never existed) does the metadata doc itself
        # get removed.
        account = principal.account_id
        cap = await _get_owned_or_404(store, capture_id, account)
        if cap.audio_uri is not None:
            if blobs is None:
                raise HTTPException(status_code=503, detail=(
                    "capture storage isn't configured on this server — the stored audio "
                    "can't be deleted right now"))
            try:
                await blobs.delete(capture_key(account, capture_id))
            except Exception:  # noqa: BLE001 — metadata must NOT vanish while audio survives
                raise HTTPException(status_code=502, detail=(
                    "couldn't delete the stored audio — the capture was left in place; try again"))
        await store.delete_capture(capture_id)

    @router.put("/captures/{capture_id}/labels", response_model=Capture)
    async def put_labels(
        capture_id: str,
        principal: Principal = Depends(full_auth_dep),
        # Any (not dict): a `dict`-typed param lets FastAPI/pydantic reject a
        # non-object body itself, with its own generic validation-error
        # detail, before the isinstance check below ever runs — making the
        # brief-mandated "labels must be a json object" detail unreachable.
        # Accepting loosely and gating explicitly is what makes that message
        # reachable.
        labels: Any = Body(...),
    ) -> Capture:
        if not isinstance(labels, dict):
            raise HTTPException(status_code=422, detail="labels must be a json object")
        if len(json.dumps(labels).encode()) > MAX_LABELS_BYTES:
            raise HTTPException(status_code=422, detail="labels payload is too large")

        account = principal.account_id
        cap = await _get_owned_or_404(store, capture_id, account)
        cap.labels = labels
        cap.labels_updated_at = _now_iso()
        await store.put_capture(cap)
        return cap

    return router
