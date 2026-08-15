# Ported from gauge@2157433 server/main.py's `POST /episodes/{id}/analyze` route; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B11): gauge defined this endpoint inline in `create_app`
# (server/main.py), closed over `resolved_store`/`resolved_transcriber`/
# `resolved_llm`/`resolved_diarizer`. This repo's watch routers are each a
# standalone `make_*_router(...) -> APIRouter` factory (see rest.py, groups.py,
# ...), so the same closure shape is used here instead, in its own file per
# the plan's file structure (`server/watch/routers/live_sessions.py`).
# Episode -> LiveSession, `/episodes/{id}/analyze` -> `/live-sessions/{id}/analyze`,
# `analyze_episode` -> `analyze_live_session` per the locked rename map.
"""POST /live-sessions/{id}/analyze: owner-only, idempotent re-analysis.

Wires `watch/post_session.py`'s `analyze_live_session` (Task B10) behind
auth. Safe to call more than once on the same live session — e.g. after
enabling real STT, or to pick up a freshly-enrolled voiceprint for
diarization — since `analyze_live_session` replaces (never accumulates)
its own transcript-/diarization-derived events on each call.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from watch.auth import AuthDep, Principal
from watch.models import LiveSession
from watch.post_session import analyze_live_session
from watch.store import LiveSessionStore


def make_live_sessions_router(
    store: LiveSessionStore,
    auth_dep: AuthDep,
    transcriber=None,
    llm=None,
    diarizer=None,
) -> APIRouter:
    router = APIRouter()

    @router.post("/live-sessions/{live_session_id}/analyze", response_model=LiveSession)
    async def analyze(
        live_session_id: str, principal: Principal = Depends(auth_dep)
    ) -> LiveSession:
        existing = await store.get_live_session(live_session_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="live session not found")
        if existing.owner_account != principal.account_id:
            raise HTTPException(
                status_code=403, detail="only the live session owner may trigger analysis"
            )
        return await analyze_live_session(
            live_session_id, store, transcriber, llm, diarizer=diarizer
        )

    return router
