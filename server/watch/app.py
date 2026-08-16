# Ported from gauge@2157433 server/main.py's `create_app`; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
"""Production app assembly for the watch domain (Task B12).

``build_watch_routers()`` is gauge's own ``create_app`` MINUS the FastAPI
``app`` object itself: MindShift's ``server/main.py`` already owns the app,
its CORS middleware, its lifespan, and its own ``/health``/``/healthz`` —
this module's only job is building every watch-domain dependency from env
(``watch.config.Settings``) exactly the way gauge's ``create_app`` did, and
handing back the list of routers for ``main.py``'s single include block to
mount.

Every store/pairing-store/telemetry-store/blob-store factory below gates on
its own ``MINDSHIFT_*`` env var (see each ``get_*()``'s own docstring) —
unset means an in-memory (stores) or ``None`` (blobs) fallback, which is
CORRECT for keyless local dev and CI, never a fabricated success (honest-
degradation doctrine).

``embedder`` is deliberately NOT resolved here (unlike ``diarizer`` below):
``watch/routers/rest.py``'s own ``_resolve_embedder`` already does the same
"injected wins, else fall back to ``speaker_id.embed_pcm`` when
``speaker_id.is_available()``" lazily, at request time — passing
``embedder=None`` into ``make_rest_router`` here reproduces gauge's own
production default (gauge's ``create_app()`` with no args also resolves to
``resolved_embedder = None`` and relies on ``rest_api._resolve_embedder``'s
own lazy fallback). ``diarizer`` has no equivalent per-request fallback in
``watch/routers/ws.py`` / ``watch/routers/live_sessions.py`` — both just use
whatever object they're given, so it must be built ONCE here and shared by
both routers, mirroring gauge's own eager
``EmbeddingDiarizationService(speaker_id.embed_pcm) if speaker_id.is_available()
else NullDiarizationService()`` resolution.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Request

import speaker_id
from watch.auth import get_full_verifier, make_auth_dependency, require_full_auth
from watch.blobs import get_blob_store
from watch.config import Settings
from watch.diarize import DiarizationService, EmbeddingDiarizationService, NullDiarizationService
from watch.pairing_store import get_pairing_store
from watch.routers.captures import make_captures_router
from watch.routers.groups import make_groups_router
from watch.routers.live_sessions import make_live_sessions_router
from watch.routers.pairing import make_pairing_router
from watch.routers.rest import make_rest_router
from watch.routers.telemetry import make_telemetry_router
from watch.routers.ws import make_ws_router
from watch.services import build_llm, build_transcriber
from watch.store import get_store
from watch.telemetry_store import get_telemetry_store

logger = logging.getLogger(__name__)


async def _rate_limit(request: Request) -> None:
    """Reuse main's per-IP rate limiter. Imported lazily at request time,
    exactly mirroring server/routers/voice.py's own ``_rate_limit`` wrapper
    (see its docstring for the circular-import reason): main.py imports
    ``build_watch_routers`` at module load, so a top-level `import main`
    here would be circular.

    Task H1 (final Phase-1 review finding): the watch domain's deliberately-
    unauthenticated routes (telemetry, pairing start/status) now sit on the
    same internet-facing app as every rate-limited main.py route — this
    wrapper is what lets ``make_telemetry_router``/``make_pairing_router``
    take a real limiter without importing main at module scope.
    """
    import main

    await main._rate_limit(request)


def build_watch_routers() -> list[APIRouter]:
    """Assemble every watch router against real, env-driven dependencies.

    Returns them as a flat list, in the same order gauge's ``create_app``
    mounted them, for ``server/main.py``'s include block to iterate over.
    """
    settings = Settings()

    # I2 (final whole-branch review 2026-08-15): the in-memory/None fallbacks
    # below are correct for keyless local dev and CI, but silent in prod —
    # flag them loudly so a misconfigured deploy is caught before it loses
    # data on the next restart, instead of discovered after.
    if not os.environ.get("MINDSHIFT_FIRESTORE_PROJECT"):
        logger.warning(
            "watch domain running on in-memory stores — data will not "
            "persist across restarts (set MINDSHIFT_FIRESTORE_PROJECT)"
        )
    if not os.environ.get("MINDSHIFT_CAPTURE_BUCKET"):
        logger.warning(
            "watch domain running with no capture blob store — capture "
            "audio uploads will 503 (set MINDSHIFT_CAPTURE_BUCKET)"
        )

    store = get_store()
    pairing_store = get_pairing_store()
    telemetry_store = get_telemetry_store()
    blobs = get_blob_store()

    transcriber = build_transcriber(settings)
    llm = build_llm()
    verifier = get_full_verifier(pairing_store)

    diarizer: DiarizationService = (
        EmbeddingDiarizationService(speaker_id.embed_pcm)
        if speaker_id.is_available()
        else NullDiarizationService()
    )

    auth_dep = make_auth_dependency(verifier, settings.allow_legacy_account, store)
    # I2/I3 controller ruling (watch/auth.py's require_full_auth docstring):
    # captures, groups, and pairing's /me/pair/claim must never be reachable
    # by an unauthenticated `?account=` legacy principal — every other
    # surface (live sessions, settings, enroll, /me, WS ingest, telemetry)
    # stays on the plain `auth_dep` the shipped watch and phone's legacy
    # account-override bridge depend on.
    strict_auth_dep = require_full_auth(auth_dep)

    return [
        make_rest_router(store, auth_dep, strict_auth_dep),
        make_groups_router(store, strict_auth_dep),
        make_captures_router(store, blobs, strict_auth_dep),
        make_pairing_router(pairing_store, strict_auth_dep, rate_limit_dep=_rate_limit),
        make_telemetry_router(telemetry_store, rate_limit_dep=_rate_limit),
        make_ws_router(
            store,
            transcriber,
            llm,
            verifier=verifier,
            # DEFAULT-INVERSION WARNING (B11 review round 1, carried forward
            # here): make_ws_router's own `allow_legacy` kwarg defaults to
            # False (correct, fail-closed, for the function's OWN
            # signature), but gauge's EFFECTIVE production default was True
            # via Settings().allow_legacy_account
            # (MINDSHIFT_ALLOW_LEGACY_ACCOUNT defaults to "true" — see
            # watch/config.py). Omitting this kwarg here would silently
            # fail-closed the default deployed config, 1008-closing every
            # legacy `?account=` connection from the currently-shipped
            # watch, while REST (auth_dep above) would still accept the
            # same callers.
            allow_legacy=settings.allow_legacy_account,
            stt=settings.stt,
            diarizer=diarizer,
        ),
        make_live_sessions_router(store, auth_dep, transcriber, llm, diarizer=diarizer),
    ]
