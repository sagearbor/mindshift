# Ported from gauge@2157433 server/tests/test_ws_ingest.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B11):
# * Episode -> LiveSession per the locked rename map: WS path
#   `/ws/episode/{id}` -> `/ws/live-session/{id}`; final frame
#   `episode_saved`/`episode_id` -> `live_session_saved`/`live_session_id`.
#   Every other frame (binary PCM, `hr`, `end`, `vector_event`, `nudge`,
#   `error`) is unchanged.
# * `server.main.create_app` -> `watch.testing.create_watch_test_app`
#   (keyword-only assembly); `server.store.MemoryEpisodeStore` ->
#   `watch.store.MemoryLiveSessionStore`.
# * gauge's `make_ws_router(..., settings=Settings(GAUGE_STT=...))` gate
#   became `create_watch_test_app(..., stt=...)` — see watch/testing.py's
#   and watch/routers/ws.py's own ADAPTED notes for why (explicit,
#   env-var-free kwarg matching every other router's construction here,
#   instead of a router-internal `Settings()` read from the process env).
#   `monkeypatch.setenv("GAUGE_STT", "whisper")` in the source test becomes
#   `stt="whisper"` passed directly to `create_watch_test_app`.
# * `server.tests.test_vectors.pcm` -> `server.tests.watch.test_vectors.pcm`
#   (this repo's watch-scoped vectors test module has the identical helper).
import asyncio
import base64
import json
import time

from fastapi.testclient import TestClient

import watch.routers.ws as ws_module
from server.tests.watch.test_vectors import pcm
from watch.store import MemoryLiveSessionStore
from watch.testing import create_watch_test_app


def test_ws_yelling_produces_nudge_and_saves_live_session():
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with client.websocket_connect("/ws/live-session/e1?account=alice") as ws:
        ws.send_bytes(pcm(0.02))                       # quiet
        ws.send_bytes(pcm(0.4))                        # loud → yelling (live-session-relative fallback)
        msgs = []
        ws.send_text(json.dumps({"type": "hr", "bpm": 120, "t": 2.0}))
        ws.send_text(json.dumps({"type": "end"}))
        while True:
            m = json.loads(ws.receive_text())
            msgs.append(m)
            if m["type"] == "live_session_saved":
                break
    kinds = {m["type"] for m in msgs}
    assert "vector_event" in kinds and "nudge" in kinds
    ls = asyncio.run(store.get_live_session("e1"))
    assert ls is not None and ls.status == "captured" and ls.owner_account == "alice"
    assert len(ls.series["rms_db"]) == 2 and ls.nudge_events


def test_ws_unknown_text_type_gets_error_and_stays_open():
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with client.websocket_connect("/ws/live-session/e2?account=bob") as ws:
        ws.send_text(json.dumps({"type": "bogus"}))
        err = json.loads(ws.receive_text())
        assert err == {"type": "error", "detail": "unknown_type"}

        # Connection must still be usable after the error — send "end" and
        # confirm we get a normal live_session_saved, not a closed/broken socket.
        ws.send_text(json.dumps({"type": "end"}))
        saved = json.loads(ws.receive_text())
        assert saved["type"] == "live_session_saved" and saved["status"] == "captured"

    ls = asyncio.run(store.get_live_session("e2"))
    assert ls is not None and ls.status == "captured"


def test_ws_malformed_json_gets_error_and_stays_open():
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with client.websocket_connect("/ws/live-session/e3?account=carol") as ws:
        ws.send_text("{not valid json")
        err = json.loads(ws.receive_text())
        assert err == {"type": "error", "detail": "malformed_json"}

        ws.send_text(json.dumps({"type": "end"}))
        saved = json.loads(ws.receive_text())
        assert saved["type"] == "live_session_saved"


def test_ws_abrupt_disconnect_saves_not_analyzed():
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with client.websocket_connect("/ws/live-session/e4?account=dave") as ws:
        ws.send_bytes(pcm(0.02))
        # No "end" sent — client just goes away (context manager closes the
        # socket), simulating a dropped connection mid-live-session.

    ls = asyncio.run(store.get_live_session("e4"))
    assert ls is not None
    assert ls.status == "not_analyzed"
    assert ls.owner_account == "dave"
    assert len(ls.series["rms_db"]) == 1


def test_ws_abrupt_disconnect_after_live_nudge_preserves_events():
    # Regression / P4-1 investigation (gauge): production reported a saved
    # episode with status="not_analyzed" and EMPTY vector_events/nudge_events
    # despite a live nudge having been felt on the device during the session.
    #
    # This test reproduces the closest server-observable analogue: PCM loud
    # enough to fire a live vector_event + nudge (matching "wearer felt a
    # haptic nudge"), immediately followed by an ABRUPT disconnect (no clean
    # "end" — the context manager just closes the socket). It passes today —
    # proving ws.py's accumulate-then-save closures (vector_events/
    # nudge_events, read by build_live_session() straight out of the
    # connection's own local lists) are NOT the source of the production bug:
    # whatever a connection has already detected and emitted survives into
    # the "not_analyzed" fallback save.
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with client.websocket_connect("/ws/live-session/e10?account=default") as ws:
        ws.send_bytes(pcm(0.02))  # quiet, seeds the running median
        ws.send_bytes(pcm(0.4))   # loud -> yelling level 3, live nudge fires
        vector_event = json.loads(ws.receive_text())
        nudge = json.loads(ws.receive_text())
        assert vector_event["type"] == "vector_event" and vector_event["vector"] == "yelling"
        assert nudge["type"] == "nudge"
        # No "end" sent — abrupt disconnect, exactly like the production episode.

    ls = asyncio.run(store.get_live_session("e10"))
    assert ls is not None
    assert ls.status == "not_analyzed"
    assert len(ls.vector_events) == 1 and ls.vector_events[0].vector == "yelling"
    assert len(ls.nudge_events) == 1


def test_ws_bogus_hr_client_timestamp_does_not_corrupt_nudge_cooldown():
    # Regression for: HR path must share ONE clock (the server stream clock,
    # engine.t) with the PCM path for NudgePolicy hysteresis. A client "t"
    # that's wildly off (clock skew/jitter/bad client) must neither force an
    # early de-escalation nor delay/prevent the on-time one — de-escalation
    # timing is purely a function of the number of 1-second PCM windows.
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with client.websocket_connect("/ws/live-session/e6?account=erin") as ws:
        ws.send_bytes(pcm(0.02))  # window 1 (t 0->1): quiet, seeds the running median, no events
        ws.send_bytes(pcm(0.4))   # window 2 (t 1->2): loud -> yelling level 3, channel A escalates 0->3
        vector_event = json.loads(ws.receive_text())
        nudge = json.loads(ws.receive_text())
        assert vector_event["type"] == "vector_event" and vector_event["vector"] == "yelling"
        assert nudge == {"type": "nudge", "channel": "A", "level": 3, "t": 1.0, "vectors": ["yelling"]}

        # Bogus client timestamp, sent immediately (stream clock is only ~2.0s
        # in). Under the old bug (policy driven by client t) this alone would
        # instantly satisfy "9999.0 - 1.0 > 20.0" and drop channel A early.
        ws.send_text(json.dumps({"type": "hr", "bpm": 70, "t": 9999.0}))

        # 20 more quiet windows (3..22): not enough stream-clock time has
        # passed yet (cooldown_s=20.0) for a drop — expect total silence on
        # the wire, proving the bogus HR t didn't leak into the policy clock.
        for _ in range(20):
            ws.send_bytes(pcm(0.02))

        # Window 23: stream clock now clears the cooldown boundary (t0=22.0,
        # 22.0 - 1.0 > 20.0) -> exactly one step-down, driven by window count.
        ws.send_bytes(pcm(0.02))
        drop = json.loads(ws.receive_text())
        assert drop == {"type": "nudge", "channel": "A", "level": 2, "t": 22.0, "vectors": []}

        ws.send_text(json.dumps({"type": "end"}))
        saved = json.loads(ws.receive_text())
        assert saved["type"] == "live_session_saved"


def test_ws_end_fires_real_analysis_pipeline_when_stt_enabled():
    # Every other WS test in this file runs under this app's `stt="none"`
    # default (see watch/testing.py's own ADAPTED note), so the
    # `if stt != "none": _spawn_live_session_analysis(...)` branch in
    # watch/routers/ws.py is never taken by any of them. This test passes
    # `stt="whisper"` explicitly (with FAKE transcriber/LLM — never a real
    # model) to exercise the actual asyncio.create_task wiring end to end.
    from server.tests.watch.test_post_session import TWO_SEGMENTS, FakeLLM, FakeTranscriber

    store = MemoryLiveSessionStore()
    # A deliberate 1s delay: if live_session_saved ever waited for analysis,
    # this round trip would take >= 1s. It must not.
    transcriber = FakeTranscriber(segments=TWO_SEGMENTS, delay=1.0)
    llm = FakeLLM(summary="Analyzed via fire-and-forget.")
    client = TestClient(create_watch_test_app(
        store=store, transcriber=transcriber, llm=llm, stt="whisper", allow_legacy=True,
    ))

    # No PCM frame sent here on purpose: Starlette's WebSocketTestSession
    # transport hands frames to the app via a zero-buffer rendezvous, so a
    # prior send_bytes() wouldn't return control to the test until the app
    # loops back to `await websocket.receive()` — which it only does once
    # it's done pushing that frame through VectorEngine's (real, pre-existing,
    # non-trivial) prosody processing. That's a confound unrelated to this
    # test's purpose and would corrupt the timing assertion below; skipping
    # it isolates exactly the "end" -> live_session_saved leg.
    # NOTE: the poll for "analyzed" below happens WHILE the websocket
    # connection is still open (before this `with` block exits). Starlette's
    # TestClient runs the ASGI app on a per-connection portal thread/loop that
    # gets torn down — cancelling any outstanding asyncio.create_task()s,
    # including our fire-and-forget one — as soon as the connection closes.
    # That's a property of this TEST TRANSPORT, not of production (a real
    # uvicorn server's event loop lives for the process's lifetime and isn't
    # torn down when one client disconnects) — verified directly: exiting
    # the `with` block before polling reliably leaves the task cancelled
    # mid-`await asyncio.to_thread(...)`, never reaching "analyzed". Polling
    # before exit sidesteps that test-only artifact.
    with client.websocket_connect("/ws/live-session/e7?account=frank") as ws:
        start = time.monotonic()
        ws.send_text(json.dumps({"type": "end"}))
        saved = json.loads(ws.receive_text())
        elapsed_to_saved = time.monotonic() - start

        assert saved == {"type": "live_session_saved", "live_session_id": "e7", "status": "captured"}
        assert elapsed_to_saved < 0.5, (
            f"live_session_saved must not wait for analysis (took {elapsed_to_saved:.2f}s)"
        )

        async def wait_for_analyzed():
            # Bounded poll (5s max) — well past the fake's 1s delay — for the
            # fire-and-forget task to finish and persist the analyzed live session.
            for _ in range(50):
                ls = await store.get_live_session("e7")
                if ls is not None and ls.status == "analyzed":
                    return ls
                await asyncio.sleep(0.1)
            return await store.get_live_session("e7")

        ls = asyncio.run(wait_for_analyzed())

    assert ls is not None
    assert ls.status == "analyzed"
    assert ls.summary == "Analyzed via fire-and-forget."


def test_ws_pcm_buffer_caps_at_max_bytes_but_keeps_processing_live_windows(monkeypatch, caplog):
    # Final-review Finding 1d (gauge): MAX_LIVE_SESSION_PCM_BYTES bounds the
    # in-RAM pcm_buffer for a very long live session. Monkeypatched down to
    # exactly one window (32000 bytes = 1s of PCM16/16kHz/mono) so the test
    # doesn't need to actually stream 30 minutes of audio to exercise the cap.
    monkeypatch.setattr(ws_module, "MAX_LIVE_SESSION_PCM_BYTES", 32000)

    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with client.websocket_connect("/ws/live-session/e9?account=gail") as ws:
        with caplog.at_level("WARNING"):
            ws.send_bytes(pcm(0.02))  # window 1 (quiet): fills the cap exactly
            ws.send_bytes(pcm(0.4))   # window 2 (loud): over the cap, but still
                                       # live-processed -> a real vector_event/nudge
            vector_event = json.loads(ws.receive_text())
            nudge = json.loads(ws.receive_text())
        assert vector_event["type"] == "vector_event" and vector_event["vector"] == "yelling"
        assert nudge["type"] == "nudge"

        ws.send_text(json.dumps({"type": "end"}))
        saved = json.loads(ws.receive_text())
        assert saved["type"] == "live_session_saved"

    ls = asyncio.run(store.get_live_session("e9"))
    assert ls is not None
    # Only window 1's audio was retained -- window 2 pushed past the cap.
    assert len(base64.b64decode(ls.pcm_b64)) == 32000
    # But BOTH windows still went through live vector/nudge detection.
    assert len(ls.series["rms_db"]) == 2

    assert any(
        "MAX_LIVE_SESSION_PCM_BYTES" in rec.message and "e9" in rec.message
        for rec in caplog.records
    )


def test_live_session_pcm_b64_excluded_from_wire():
    from watch.models import LiveSession, Participant

    ls = LiveSession(
        id="e5",
        owner_account="a1",
        started_at="2026-08-01T00:00:00Z",
        ended_at=None,
        status="captured",
        participants=[Participant(id="p1", role="self", speaker_label="You")],
        vector_events=[],
        nudge_events=[],
        pcm_b64="not-empty-audio-bytes",
    )
    assert ls.pcm_b64 == "not-empty-audio-bytes"
    dumped = json.loads(ls.model_dump_json())
    assert "pcm_b64" not in dumped
