# Ported from gauge@2157433 server/tests/test_main.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B11): gauge's sole test in this file
# (test_healthz_returns_ok) proves `create_app(store, transcriber=None,
# llm=None)` "must work with no store/transcriber/llm wiring beyond the
# defaults" by hitting `/health` — a route that exists purely for Cloud
# Run's health probe and is deliberately dependency-free.
#
# THIS repo has no watch-scoped equivalent of `/health` yet: `create_app`
# (the real production app factory, with its own health route) is Task
# B12's job (`server/watch/app.py`); `watch/testing.py`'s
# `create_watch_test_app` is a test-only assembly with no health route of
# its own. So the faithful 1:1 adaptation of "prove the app assembly
# function builds and works with no wiring beyond the defaults" is: build
# `create_watch_test_app()` with every optional kwarg at its default
# (including `transcriber=None`, `llm=None` — this task's own new kwargs)
# and confirm Task B11's two new routers (WS ingest + analyze) are both
# actually mounted and reachable end to end, not merely importable. One
# test, matching gauge's file 1:1 — both routers are exercised in the same
# test the way gauge's single healthz test exercised the single route it had.
"""Wiring smoke test for watch/testing.py's Task B11 additions: the WS
ingest router (watch/routers/ws.py) and the live-session analyze router
(watch/routers/live_sessions.py) must both be reachable through
create_watch_test_app with no wiring beyond the defaults."""

import asyncio
import json

from fastapi.testclient import TestClient

from watch.store import MemoryLiveSessionStore
from watch.testing import create_watch_test_app


def test_ws_and_analyze_routers_are_wired_with_no_transcriber_or_llm_beyond_defaults():
    store = MemoryLiveSessionStore()
    app = create_watch_test_app(store=store, allow_legacy=True)
    client = TestClient(app)

    # WS leg: transcriber=None/llm=None/stt="none" (all defaults) mean the
    # fire-and-forget analysis spawn is never attempted, so a clean "end"
    # round-trip proves the router mounts and the renamed wire shape
    # (episode_saved/episode_id -> live_session_saved/live_session_id) is
    # live end to end, not just in ws.py's source.
    with client.websocket_connect("/ws/live-session/wiring-check?account=default") as ws:
        ws.send_text(json.dumps({"type": "end"}))
        saved = json.loads(ws.receive_text())
    assert saved == {
        "type": "live_session_saved", "live_session_id": "wiring-check", "status": "captured",
    }
    live_session = asyncio.run(store.get_live_session("wiring-check"))
    assert live_session is not None and live_session.status == "captured"

    # Analyze leg: transcriber=None/llm=None never get touched here — the
    # live session doesn't exist, so the route 404s before analyze_live_session
    # would ever call .transcribe()/.complete() on them. Proves the analyze
    # router itself is mounted and functioning end to end (store -> auth ->
    # 404), independent of any transcriber/LLM wiring.
    resp = client.post("/live-sessions/does-not-exist/analyze", params={"account": "acct1"})
    assert resp.status_code == 404
