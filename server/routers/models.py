"""Model-asset router — the on-device pieces the phone can't bundle.

One endpoint today, ``GET|HEAD /models/ecapa.onnx``: the ONNX export of the
PINNED ECAPA-TDNN speaker embedder (``ecapa_onnx.py``, ~80 MB) that lets the
fast loop compute voiceprints on the phone in the SAME embedding space the
server enrolls them in (``speaker_id``). Without it the phone's realtime loop
(``apps/mobile/src/live/speakerId.ts`` + ``ortNative.ts``) is fully wired but
inert: "speaker-ID off". Why a runtime download and not an app asset: 80 MB
is far too much to ship in every OTA bundle, the file is a pure function of
the pinned revision (reproducible, not user data), and absence must degrade
to "speaker-ID off" rather than a broken build.

Where the file comes from, in order:

1. ``MINDSHIFT_ECAPA_ONNX_PATH`` when set — an operator who exported it
   ahead of time (e.g. baked into an image WITHOUT torch: the server can
   then serve it while being unable to produce it). Trusted to match
   ``speaker_id.ECAPA_REVISION``.
2. else ``<speaker_id.cache_dir()>/ecapa_<ECAPA_REVISION>.onnx`` — where
   ``scripts/export_ecapa_onnx.py`` writes by default (pre-warm), and
3. if that is absent and the voice deps are installed, the server GENERATES
   it once, in a worker thread under a process-wide lock (the export loads
   the checkpoint + traces the graph: tens of seconds; concurrent first
   requests all wait for the one export rather than each running their own).
   Written atomically (``.partial`` + rename) so a crash can never leave a
   truncated file that would be served as a model.
4. if that is impossible (no torch/speechbrain, or the export fails) ->
   honest **503** with the reason — never a fabricated or placeholder file.

Caching contract (what the phone relies on — ``ortNative.ts`` /
``modelDownload.ts``): ``ETag`` is the pinned REVISION (quoted, strong) —
the artifact is a pure function of it, so the tag is honest without hashing
80 MB on every request; ``Cache-Control: private, max-age=86400`` (one day:
long enough that a launch never re-downloads, short enough that a model bump
propagates); ``Content-Length`` so the phone can sanity-check the download;
``If-None-Match`` -> **304** — answered from the revision alone, BEFORE any
export, so a phone that already holds the current model never makes the
server produce (or read) the file just to say "unchanged"; ``HEAD`` carries
the same headers with no body, which is the phone's cheap launch re-check.

Authenticated like every other route (the verified Firebase uid) — the
weights are public research artifacts, but the endpoint can trigger a
tens-of-seconds export, so it stays behind sign-in and the per-IP limiter.

Kept out of main.py (own file, one ``include_router`` line) like the voice
router, and never imports main at module load (see ``_rate_limit``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse

import ecapa_onnx
import speaker_id
from auth import get_current_uid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])

ECAPA_ROUTE = "/ecapa.onnx"
CACHE_CONTROL = "private, max-age=86400"
MEDIA_TYPE = "application/octet-stream"

# A truncated/empty file is not a model. The real export is ~80 MB; anything
# under this is treated as absent (and regenerated when possible) rather
# than served. Small on purpose so tests can stand in a fake exporter
# without writing megabytes.
MIN_ONNX_BYTES = 1024

# Serializes the export across requests AND threads (asyncio.to_thread runs
# it off the loop; a threading lock is what actually holds there).
_export_lock = threading.Lock()

# Review 2026-08-24: the handler ALSO serializes on an event-loop lock before
# it ever calls to_thread. The threading lock alone let every concurrent
# cold request park a default-executor worker thread on it for the whole
# tens-of-seconds export — on a 2-vCPU Cloud Run instance that executor has
# ~6 workers, and they are shared with everything else the process runs off
# the loop (Firebase token verification for the realtime WS handshake, LLM
# calls, model passes). A handful of phones fetching the model at once could
# stall the whole server. Waiting on an asyncio.Lock costs no thread.
_export_async_lock: asyncio.Lock | None = None


def _get_export_async_lock() -> asyncio.Lock:
    # Created lazily on first use so it binds to the running loop (module
    # import happens before uvicorn's loop exists).
    global _export_async_lock
    if _export_async_lock is None:
        _export_async_lock = asyncio.Lock()
    return _export_async_lock


class ModelUnavailable(RuntimeError):
    """The model can't be served: absent AND not producible here. The router
    maps this to an honest 503 with the reason."""


def _export_onnx(path: Path) -> Path:
    """The production exporter — a module attribute so tests substitute a
    fake that writes bytes instead of loading torch."""
    return ecapa_onnx.export(path)


def _usable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= MIN_ONNX_BYTES
    except OSError:
        return False


def ensure_ecapa_onnx() -> Path:
    """Blocking: return the path of a usable export, generating it once when
    it is absent and the voice deps allow. Raises :class:`ModelUnavailable`
    otherwise (never returns a path to something that isn't a model)."""
    path = ecapa_onnx.configured_onnx_path()
    if _usable(path):
        return path
    if not speaker_id.is_available():
        raise ModelUnavailable(
            "ECAPA ONNX model is not available on this server: no exported "
            f"file at {path} and the voice deps (torch + speechbrain) needed "
            "to export it are not installed"
        )
    with _export_lock:
        # Another request may have finished the export while we waited.
        if _usable(path):
            return path
        logger.info(
            "Exporting ECAPA %s@%s to ONNX at %s (first request; one-time)",
            speaker_id.ECAPA_SOURCE, speaker_id.ECAPA_REVISION, path,
        )
        try:
            _export_onnx(path)
        except speaker_id.SpeakerIdUnavailable as exc:
            raise ModelUnavailable(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — surface the reason, don't crash the API
            logger.exception("ECAPA ONNX export failed")
            raise ModelUnavailable(f"ECAPA ONNX export failed: {exc}") from exc
        if not _usable(path):
            raise ModelUnavailable(
                f"ECAPA ONNX export produced no usable file at {path}"
            )
        logger.info(
            "Exported ECAPA ONNX model: %s (%d bytes)", path, path.stat().st_size,
        )
        return path


def ecapa_etag() -> str:
    """Strong, quoted ETag = the pinned revision (see the module docstring)."""
    return f'"{speaker_id.ECAPA_REVISION}"'


def if_none_match_matches(header: str | None, etag: str) -> bool:
    """RFC 7232 ``If-None-Match`` evaluation with weak comparison: ``*``
    matches anything; otherwise any comma-separated tag equal to ``etag``
    once a ``W/`` prefix is stripped (a phone may echo the tag either way)."""
    if not header:
        return False
    header = header.strip()
    if header == "*":
        return True
    wanted = etag[2:] if etag.startswith("W/") else etag
    for raw in header.split(","):
        tag = raw.strip()
        if tag.startswith("W/"):
            tag = tag[2:]
        if tag == wanted:
            return True
    return False


async def _rate_limit(request: Request) -> None:
    """Reuse main's per-IP limiter, imported lazily at request time — main.py
    includes this router at module load, so a top-level import would be
    circular (same pattern as routers/voice.py)."""
    import main

    await main._rate_limit(request)


# One handler, two registrations (distinct operation ids keep OpenAPI free
# of the duplicate-id warning a single multi-method route would raise).
_ROUTE_DOC = dict(
    summary="The ECAPA-TDNN speaker-embedding model as ONNX (on-device voice ID)",
    response_class=Response,
    responses={
        200: {
            "content": {MEDIA_TYPE: {}},
            "description": (
                "The ONNX graph: input `waveform` float32 [1, T] mono 16 kHz, "
                "output `embedding` float32 [1, 192] L2-normalized. `ETag` is "
                "the pinned model revision; `Cache-Control: private, max-age=86400`."
            ),
        },
        304: {"description": "`If-None-Match` matched the current revision."},
        503: {
            "description": (
                "No exported model and this server can't produce one (voice "
                "deps not installed, or the export failed) — the reason is in "
                "`detail` (and the `X-Model-Unavailable` header for HEAD)."
            ),
        },
    },
)


@router.head(ECAPA_ROUTE, operation_id="head_ecapa_onnx", **_ROUTE_DOC)
@router.get(ECAPA_ROUTE, operation_id="get_ecapa_onnx", **_ROUTE_DOC)
async def get_ecapa_onnx(
    request: Request,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
) -> Response:
    etag = ecapa_etag()
    headers = {"ETag": etag, "Cache-Control": CACHE_CONTROL}
    # Revalidation first, from the revision alone: the phone's launch
    # re-check must never trigger (or wait on) an export.
    if if_none_match_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=headers)
    try:
        path = ecapa_onnx.configured_onnx_path()
        if not _usable(path):
            # Cold cache: one request at a time may go to a worker thread
            # for the export; the rest wait here, on the loop, thread-free,
            # and find the file when it is their turn.
            async with _get_export_async_lock():
                path = await asyncio.to_thread(ensure_ecapa_onnx)
    except ModelUnavailable as exc:
        raise HTTPException(
            status_code=503, detail=str(exc), headers={"X-Model-Unavailable": str(exc)},
        )
    # FileResponse streams the file, sets Content-Length from stat, and
    # answers HEAD with headers only; our ETag/Cache-Control win over its
    # stat-derived defaults (it uses setdefault).
    return FileResponse(os.fspath(path), media_type=MEDIA_TYPE, headers=headers)
