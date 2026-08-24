# Ported from gauge@2157433 server/ws_ingest.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B11):
# * Episode -> LiveSession per the locked "episode" rename map: WS path
#   `/ws/episode/{episode_id}` -> `/ws/live-session/{live_session_id}`; final
#   frame `{"type": "episode_saved", "episode_id": ...}` ->
#   `{"type": "live_session_saved", "live_session_id": ...}`. Every OTHER
#   frame (binary PCM windows, `hr`, `end`, `vector_event`, `nudge`, `error`)
#   is byte-for-byte unchanged. `EpisodeStore` -> `LiveSessionStore`,
#   `MAX_EPISODE_PCM_BYTES` -> `MAX_LIVE_SESSION_PCM_BYTES`,
#   `analyze_episode` -> `analyze_live_session`, `_on_episode_end` ->
#   `_on_live_session_end`, `_spawn_episode_analysis` ->
#   `_spawn_live_session_analysis`.
# * `make_ws_router`'s single `settings: Settings | None` param is split into
#   explicit `allow_legacy: bool` and `stt: str` kwargs instead. Every other
#   watch router (and `watch/testing.py`'s own assembly) takes explicit,
#   env-var-free knobs rather than an internally-constructed `Settings()` —
#   see `watch/testing.py`'s module docstring for why (a test's auth/config
#   posture must be visible at the call site). Reading `settings.stt`/
#   `settings.allow_legacy_account` here instead would let WS auth silently
#   diverge from the SAME `verifier`/`allow_legacy` every other router in a
#   given app assembly is built from. `stt` mirrors `watch/config.py`'s
#   `Settings.stt` naming/values ("whisper" | "none" | anything else) so a
#   real app assembly (Task B12) can pass `Settings().stt` straight through
#   with no translation.
"""WebSocket ingest: /ws/live-session/{live_session_id}?account={account_id}.

Wires the per-connection VectorEngine + NudgePolicy into a live stream:
binary frames are 1-second PCM16 windows, text frames are JSON control
messages (``hr`` / ``end``). Vector/nudge events are pushed to the client as
they occur; on ``end`` the full live session is persisted and the socket
closes.

Honest-degradation contract: unknown/malformed client messages get an
``error`` frame and the connection stays open — we never fabricate data or
silently drop a client for a bad message.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from watch.auth import TokenVerifier, resolve_ws_principal
from watch.models import (
    SELF_PARTICIPANT_ID,
    EnrollmentBaseline,
    LiveSession,
    NudgeEvent,
    Participant,
    VectorEvent,
)
from watch.nudge_policy import NudgePolicy
from watch.post_session import analyze_live_session
from watch.relay import LiveWatchSession, register_live_session, unregister_live_session
from watch.store import LiveSessionStore
from watch.vectors import VectorEngine

logger = logging.getLogger(__name__)

# Floor for windows with (near-)zero RMS, matching VectorEngine's silence
# handling — never emit -inf into a JSON-serializable series.
RMS_DB_FLOOR = -120.0

# Final-review Finding 1d (gauge): bound the in-RAM pcm_buffer so a very
# long-running live session can't grow it (and the base64 blob later derived
# from it) without limit. 30 minutes at the wire contract's 32000 bytes/s (1s
# PCM16 mono 16kHz windows, per the WS protocol table in README.md) =
# 57,600,000 bytes. Past the cap we stop EXTENDING the buffer but keep
# processing every incoming window through VectorEngine as normal — live
# nudges/vector_events never stop — we just no longer retain audio for
# post-session analysis beyond the cap. This is a distinct guard from
# MAX_FIRESTORE_PCM_B64 (server/watch/store.py): that one bounds what a doc
# can persist, this one bounds what the live handler ever buffers in the
# first place.
MAX_LIVE_SESSION_PCM_BYTES = 57_600_000


def _rms_dbfs(pcm: bytes) -> float:
    """``20*log10(rms/32768)`` for int16-scaled PCM, floored for silence.

    Kept in the WS handler (not the engine) per design: this is purely for
    the persisted ``series["rms_db"]`` telemetry, not a detection input.
    """
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return RMS_DB_FLOOR
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms <= 0.0:
        return RMS_DB_FLOOR
    return 20.0 * math.log10(rms / 32768.0)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _on_live_session_end(
    live_session: LiveSession, store: LiveSessionStore, transcriber, llm, pcm: bytes, diarizer=None
) -> None:
    """Fire-and-forget seam: run Task B10's post-session analysis pipeline.

    ``pcm`` (final-review Finding 1a, gauge): the raw audio captured for this
    live session, straight out of the WS handler's in-memory ``pcm_buffer`` —
    passed DIRECTLY to ``analyze_live_session`` rather than letting it re-read
    ``live_session.pcm_b64`` from the store. This is the primary live path, so
    it must never depend on the store having round-tripped the audio intact;
    ``server/watch/store.py``'s ``live_session_to_doc`` may have persisted
    ``pcm_b64=""`` for an oversized live session (Firestore's 1MiB doc limit).

    ``diarizer`` (Task B10): the configured ``DiarizationService`` (or
    ``None`` to skip diarization entirely) — threaded straight through to
    ``analyze_live_session``.

    Called via ``asyncio.create_task`` — this coroutine's own errors must
    never propagate anywhere: a broken/unavailable transcriber or LLM is
    already handled honestly *inside* ``analyze_live_session`` (it degrades
    the live session's status instead of raising), but anything else going
    wrong here (e.g. a store write failing) is caught and logged so it can
    never crash the WS handler or vanish silently.
    """
    try:
        await analyze_live_session(
            live_session.id, store, transcriber, llm, pcm=pcm, diarizer=diarizer
        )
    except Exception:  # noqa: BLE001 — background task; must log, never raise
        logger.exception("Post-session analysis failed for live session %s", live_session.id)


# asyncio.create_task() only holds a WEAK reference to the task it returns;
# if nothing else references it, the task can be garbage-collected mid-flight
# (a documented asyncio gotcha), silently killing analysis before
# _on_live_session_end's own try/except ever gets a chance to log anything.
# This module-level set holds a strong reference for the task's lifetime; the
# done-callback removes it (and defensively logs any exception that escaped
# _on_live_session_end, which shouldn't happen given its own try/except, but
# must never vanish silently if it somehow does).
_background_tasks: set[asyncio.Task] = set()


def _on_analysis_task_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Background post-session analysis task failed unexpectedly", exc_info=exc)


def _spawn_live_session_analysis(
    live_session: LiveSession, store: LiveSessionStore, transcriber, llm, pcm: bytes, diarizer=None
) -> None:
    """Fire-and-forget, but keep a strong reference until the task completes."""
    task = asyncio.create_task(
        _on_live_session_end(live_session, store, transcriber, llm, pcm, diarizer=diarizer)
    )
    _background_tasks.add(task)
    task.add_done_callback(_on_analysis_task_done)


def make_ws_router(
    store: LiveSessionStore,
    transcriber=None,
    llm=None,
    *,
    verifier: TokenVerifier | None = None,
    allow_legacy: bool = False,
    stt: str = "whisper",
    diarizer=None,
) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/live-session/{live_session_id}")
    async def ws_live_session(
        websocket: WebSocket,
        live_session_id: str,
        account: str | None = Query(None),
        token: str | None = Query(None),
    ) -> None:
        try:
            principal = resolve_ws_principal(token, account, verifier, allow_legacy)
        except HTTPException:
            await websocket.close(code=1008)
            return

        await websocket.accept()
        account = principal.account_id

        baseline: EnrollmentBaseline | None = await store.get_baseline(account)
        subs = await store.get_subscriptions(account)
        engine = VectorEngine(baseline)  # None baseline -> live-session-relative fallback
        policy = NudgePolicy(subs)

        vector_events: list[VectorEvent] = []
        nudge_events: list[NudgeEvent] = []
        rms_db_series: list[float] = []
        pcm_buffer = bytearray()
        pcm_buffer_capped = False  # Finding 1d: log the cap crossing only once
        started_at = _now_iso()
        captured = False

        async def emit(events: list[VectorEvent], t: float) -> None:
            vector_events.extend(events)
            for e in events:
                await websocket.send_json({"type": "vector_event", **e.model_dump()})
            nudges = policy.on_events(events, t)
            nudge_events.extend(nudges)
            for n in nudges:
                await websocket.send_json({"type": "nudge", **n.model_dump()})

        # Track 1 (2026-08-24): expose THIS connection's engine + emit to the
        # phone->watch relay (watch/relay.py) for as long as the socket is
        # open, so a hostile-tone self turn the PHONE heard can escalate the
        # same NudgePolicy the watch mic feeds. Registered here — after the
        # engine/policy exist and before the first frame — and unregistered
        # in the `finally` below, so the relay can never hold a socket that
        # has already gone away. The PCM/HR paths above/below are untouched.
        relay_session = LiveWatchSession(
            account_id=account,
            live_session_id=live_session_id,
            engine=engine,
            emit=emit,
            loop=asyncio.get_running_loop(),
        )
        register_live_session(relay_session)

        def build_live_session(status: str) -> LiveSession:
            return LiveSession(
                id=live_session_id,
                owner_account=account,
                started_at=started_at,
                ended_at=_now_iso(),
                status=status,
                participants=[
                    Participant(id=SELF_PARTICIPANT_ID, role="self", speaker_label="You", account_id=account)
                ],
                vector_events=vector_events,
                nudge_events=nudge_events,
                series={"rms_db": rms_db_series},
                pcm_b64=base64.b64encode(bytes(pcm_buffer)).decode("ascii"),
            )

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break

                raw_bytes = message.get("bytes")
                if raw_bytes is not None:
                    # Finding 1d: stop retaining audio past
                    # MAX_LIVE_SESSION_PCM_BYTES (30 min @ 32KB/s) —
                    # everything else (live vector/nudge detection below, the
                    # rms_db series) keeps running exactly as before; only
                    # post-session audio retention is capped.
                    if len(pcm_buffer) < MAX_LIVE_SESSION_PCM_BYTES:
                        pcm_buffer.extend(raw_bytes)
                    elif not pcm_buffer_capped:
                        pcm_buffer_capped = True
                        logger.warning(
                            "Live session %s pcm_buffer hit MAX_LIVE_SESSION_PCM_BYTES (%d) — "
                            "no longer retaining audio; live nudges continue",
                            live_session_id, MAX_LIVE_SESSION_PCM_BYTES,
                        )
                    rms_db_series.append(_rms_dbfs(raw_bytes))
                    # Capture the stream clock BEFORE push_pcm advances it,
                    # so the policy call uses the same t the events are
                    # stamped with (one clock, not a pre/post-increment mix).
                    t0 = engine.t
                    events = engine.push_pcm(raw_bytes, speaker="self")
                    await emit(events, t0)
                    continue

                text = message.get("text")
                if text is None:
                    continue

                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "detail": "malformed_json"})
                    continue

                if not isinstance(payload, dict) or "type" not in payload:
                    await websocket.send_json({"type": "error", "detail": "malformed_json"})
                    continue

                msg_type = payload["type"]

                if msg_type == "hr":
                    try:
                        bpm = float(payload["bpm"])
                        float(payload["t"])  # required by protocol, but see below —
                                             # the client's clock is advisory only.
                    except (KeyError, TypeError, ValueError):
                        await websocket.send_json({"type": "error", "detail": "malformed_json"})
                        continue
                    # Client-supplied "t" is jittery/untrusted; NudgePolicy
                    # hysteresis must share ONE clock with the PCM path, so we
                    # stamp and schedule HR on the server's stream clock
                    # (engine.t) instead. push_hr doesn't advance engine.t,
                    # so this is simply "now" on the same clock PCM windows use.
                    stream_t = engine.t
                    events = engine.push_hr(bpm, stream_t)
                    await emit(events, stream_t)
                elif msg_type == "end":
                    live_session = build_live_session("captured")
                    captured = True
                    await store.put_live_session(live_session)
                    if stt != "none":
                        # Fire-and-forget: live_session_saved must never delay
                        # for this, nor crash the handler if the
                        # transcriber/LLM misbehaves (see _on_live_session_end).
                        # _spawn_live_session_analysis holds a strong
                        # reference to the created task so it can't be GC'd
                        # mid-flight.
                        #
                        # Finding 1a: hand the in-memory pcm_buffer bytes
                        # DIRECTLY to analysis rather than letting it re-read
                        # live_session.pcm_b64 back out of the store — the
                        # primary live path must never depend on a store
                        # round-trip that Firestore's 1MiB doc limit can
                        # truncate away (see server/watch/store.py's
                        # MAX_FIRESTORE_PCM_B64).
                        _spawn_live_session_analysis(
                            live_session, store, transcriber, llm, bytes(pcm_buffer), diarizer=diarizer
                        )
                    await websocket.send_json({
                        "type": "live_session_saved",
                        "live_session_id": live_session_id,
                        "status": live_session.status,
                    })
                    await websocket.close()
                    return
                else:
                    await websocket.send_json({"type": "error", "detail": "unknown_type"})
        except WebSocketDisconnect:
            pass
        finally:
            # Track 1: first thing on the way out, before any store write —
            # a phone turn arriving during the save below must find no
            # session rather than a half-torn-down one.
            unregister_live_session(relay_session)
            if not captured:
                # Abrupt disconnect before a clean "end": persist what we
                # captured so far so data isn't lost, but never mislabel it
                # as "captured" — it hasn't gone through the normal close path.
                #
                # P4-1 investigation finding (gauge): a saved "not_analyzed"
                # live session with empty vector_events/nudge_events is NOT
                # evidence of a bug in this save path — build_live_session()
                # reads the SAME closures emit() already extended, so
                # anything genuinely detected before the disconnect is always
                # preserved. An empty save instead means the connection died
                # before a single PCM window arrived — worth flagging
                # distinctly so a repeat doesn't require a manual
                # Firestore/REST dig to tell apart "detected nothing" from
                # "received nothing".
                if not rms_db_series:
                    logger.warning(
                        "Live session %s (account=%s) saved not_analyzed with ZERO PCM windows "
                        "received — connection died before any audio arrived, not a "
                        "detection/persistence bug. Investigate the client-side connection "
                        "(open()/preamble send), not this save path.",
                        live_session_id, account,
                    )
                await store.put_live_session(build_live_session("not_analyzed"))

    return router
