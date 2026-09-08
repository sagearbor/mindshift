"""Journal-capture self-filtering (watch A/B journal mode) — server half.

Covers: the audio-upload hook fires (and only) for journal-labeled captures;
self segments computed against the enrolled voiceprint with a fake diarizer;
no voiceprint → labeled ``no_voiceprint`` and KEPT (honest, never guessed);
48h expiry cleanup deletes blob-first, exactly like ``DELETE /captures/{id}``.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi.testclient import TestClient

import speaker_id
from server.tests.watch.test_auth import StubVerifier
from watch import journal
from watch.blobs import MemoryBlobStore
from watch.journal import (
    cleanup_expired_journal_captures,
    merge_padded_self_segments,
    process_journal_capture,
)
from watch.models import Capture, SpeakerProfile
from watch.routers.captures import capture_key
from watch.store import MemoryLiveSessionStore
from watch.testing import create_watch_test_app

TOKENS = {"alice-token": {"sub": "alice", "email": "alice@example.com"}}
A = {"Authorization": "Bearer alice-token"}
PCM_8S = b"\x01\x02" * (16000 * 8)   # 8 s of PCM16 @ 16 kHz
ME = np.eye(192, dtype=np.float32)[0]


class ScriptedDiarizer:
    """Returns fixed turns; records the voiceprint it was handed (mirrors
    test_post_session_diarization.py's fake)."""

    def __init__(self, turns):
        self.turns = turns
        self.calls = 0
        self.seen_print = "unset"

    def diarize(self, pcm, sr, self_print):
        self.calls += 1
        self.seen_print = self_print
        return list(self.turns)


def _client(diarizer=None):
    store = MemoryLiveSessionStore()
    blobs = MemoryBlobStore()
    app = create_watch_test_app(
        store=store, verifier=StubVerifier(TOKENS), blobs=blobs, diarizer=diarizer
    )
    return store, blobs, TestClient(app)


def _meta():
    return {"captured_at": "2026-08-30T09:00:00Z", "duration_s": 300.0,
            "trigger": "journal", "device": "pixel-watch-1", "attested": True}


def _capture(cid="cap1", labels=None, received_at="2026-08-30T09:00:01+00:00",
             audio_uri="memory://captures/alice/cap1.pcm"):
    return Capture(
        id=cid, account_id="alice", captured_at="2026-08-30T09:00:00Z",
        received_at=received_at, duration_s=8.0, trigger="journal",
        status="stored", audio_uri=audio_uri,
        labels=labels if labels is not None else {"journal": True, "interval_s": 300},
    )


async def _seed(cap=None, *, profile=True, pcm=PCM_8S):
    store = MemoryLiveSessionStore()
    blobs = MemoryBlobStore()
    cap = cap or _capture()
    await store.put_capture(cap)
    await blobs.put(capture_key("alice", cap.id), pcm)
    if profile:
        await store.put_speaker_profile(SpeakerProfile(
            account_id="alice",
            **speaker_id.new_profile(ME, None, recording_id="r", speaker="self", now_iso="t")))
    return store, blobs, cap


# ------------------------------------------------------------------ the upload-success hook --

def test_journal_labeled_upload_spawns_processing(monkeypatch):
    spawned = []
    monkeypatch.setattr(
        journal, "spawn_journal_processing",
        lambda cid, store, blobs, diarizer, sink=None: spawned.append(cid),
    )
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    # The watch PUTs labels BEFORE audio precisely so the hook can see them here.
    client.put(f"/captures/{cid}/labels", headers=A, json={"journal": True, "interval_s": 300})
    assert client.put(f"/captures/{cid}/audio", headers=A, content=PCM_8S).status_code == 200
    assert spawned == [cid]


def test_non_journal_upload_never_spawns(monkeypatch):
    spawned = []
    monkeypatch.setattr(
        journal, "spawn_journal_processing",
        lambda cid, store, blobs, diarizer, sink=None: spawned.append(cid),
    )
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    assert client.put(f"/captures/{cid}/audio", headers=A, content=PCM_8S).status_code == 200
    assert spawned == []


def test_truthy_string_journal_label_does_not_spawn(monkeypatch):
    """Only the exact boolean opts in — "yes" must not enroll a capture into
    journal processing/48h expiry (same StrictBool spirit as `attested`)."""
    spawned = []
    monkeypatch.setattr(
        journal, "spawn_journal_processing",
        lambda cid, store, blobs, diarizer, sink=None: spawned.append(cid),
    )
    _, _, client = _client()
    cid = client.post("/captures", headers=A, json=_meta()).json()["id"]
    client.put(f"/captures/{cid}/labels", headers=A, json={"journal": "yes"})
    client.put(f"/captures/{cid}/audio", headers=A, content=PCM_8S)
    assert spawned == []


# ------------------------------------------------------------------ the self-filtering pass --

def test_self_segments_are_padded_merged_and_written_to_labels():
    async def run():
        store, blobs, cap = await _seed()
        # ±0.5s padding makes (1.0,2.0) and (2.5,4.0) touch: one merged (0.5,4.5) span, 4.0s.
        d = ScriptedDiarizer([("self", 1.0, 2.0), ("other-1", 4.5, 6.0), ("self", 2.5, 4.0)])
        await process_journal_capture(cap.id, store, blobs, d)
        got = await store.get_capture(cap.id)
        assert got.labels["self_segments"] == [[0.5, 4.5]]
        assert got.labels["self_seconds"] == 4.0
        assert got.labels["journal_status"] == "self_filtered"
        assert got.labels["journal"] is True                       # original labels preserved
        assert got.labels["journal_expires_at"]                    # 48h retention stamped
        assert got.labels_updated_at
        # Raw audio stays (until expiry) — filtering never deletes the fresh capture.
        assert await blobs.get(capture_key("alice", cap.id)) is not None
    asyncio.run(run())


def test_diarizer_receives_the_enrolled_voiceprint():
    async def run():
        store, blobs, cap = await _seed()
        d = ScriptedDiarizer([("self", 0.0, 5.0)])
        await process_journal_capture(cap.id, store, blobs, d)
        assert isinstance(d.seen_print, np.ndarray) and d.seen_print.shape == (192,)
    asyncio.run(run())


def test_below_minimum_self_audio_is_labeled_honestly():
    async def run():
        store, blobs, cap = await _seed()
        d = ScriptedDiarizer([("self", 1.0, 2.0)])   # padded: 1.5s < the 3s floor
        await process_journal_capture(cap.id, store, blobs, d)
        got = await store.get_capture(cap.id)
        assert got.labels["journal_status"] == "self_filtered_below_minimum"
        assert got.labels["self_seconds"] == 2.0
    asyncio.run(run())


def test_no_voiceprint_marks_no_voiceprint_and_keeps_the_capture():
    async def run():
        store, blobs, cap = await _seed(profile=False)
        d = ScriptedDiarizer([("self", 0.0, 5.0)])
        await process_journal_capture(cap.id, store, blobs, d)
        got = await store.get_capture(cap.id)
        assert got is not None                                     # kept, honestly unfiltered
        assert got.labels["journal_status"] == "no_voiceprint"
        assert "self_segments" not in got.labels                   # nothing guessed
        assert d.calls == 0                                        # never even attempted
        assert await blobs.get(capture_key("alice", cap.id)) is not None
    asyncio.run(run())


def test_diarizer_failure_marks_diarization_failed_never_fabricates():
    class Boom:
        def diarize(self, pcm, sr, self_print):
            raise RuntimeError("diarizer exploded")

    async def run():
        store, blobs, cap = await _seed()
        await process_journal_capture(cap.id, store, blobs, Boom())
        got = await store.get_capture(cap.id)
        assert got.labels["journal_status"] == "diarization_failed"
        assert "self_segments" not in got.labels
    asyncio.run(run())


def test_non_journal_capture_is_left_completely_untouched():
    async def run():
        store, blobs, cap = await _seed(_capture(labels={"notes": "manual clip"}))
        d = ScriptedDiarizer([("self", 0.0, 5.0)])
        await process_journal_capture(cap.id, store, blobs, d)
        got = await store.get_capture(cap.id)
        assert got.labels == {"notes": "manual clip"}
        assert d.calls == 0
    asyncio.run(run())


def test_merge_padded_self_segments_pads_clamps_and_merges():
    turns = [("self", 0.2, 1.0), ("other-1", 1.2, 3.0), ("self", 1.4, 2.0), ("self", 5.0, 7.9)]
    segs = merge_padded_self_segments(turns, total_s=8.0)
    # (0.2,1.0)->( -0.3..1.5 clamped 0.0..1.5) merges with (0.9..2.5); (4.5..8.4 clamped ..8.0).
    assert segs == [(0.0, 2.5), (4.5, 8.0)]
    assert merge_padded_self_segments([("other-1", 0.0, 5.0)], total_s=8.0) == []


# ------------------------------------------------------------------ 48h retention cleanup --

def _expired_iso(hours_ago=1.0):
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def test_cleanup_deletes_expired_journal_captures_blob_first():
    async def run():
        old = _capture(cid="old", audio_uri="memory://captures/alice/old.pcm",
                       labels={"journal": True, "journal_expires_at": _expired_iso()})
        store, blobs, _ = await _seed(old)
        deleted = await cleanup_expired_journal_captures("alice", store, blobs)
        assert deleted == 1
        assert await store.get_capture("old") is None
        assert await blobs.get(capture_key("alice", "old")) is None
    asyncio.run(run())


def test_cleanup_blob_failure_keeps_the_metadata():
    class RaisingDeleteBlobStore(MemoryBlobStore):
        async def delete(self, key: str) -> None:
            raise RuntimeError("simulated blob delete failure")

    async def run():
        store = MemoryLiveSessionStore()
        blobs = RaisingDeleteBlobStore()
        old = _capture(cid="old", audio_uri="memory://captures/alice/old.pcm",
                       labels={"journal": True, "journal_expires_at": _expired_iso()})
        await store.put_capture(old)
        await blobs.put(capture_key("alice", "old"), PCM_8S)
        deleted = await cleanup_expired_journal_captures("alice", store, blobs)
        assert deleted == 0
        # Metadata survives a failed blob delete — audio is never orphaned (D4 ordering).
        assert await store.get_capture("old") is not None
    asyncio.run(run())


def test_cleanup_never_touches_unexpired_or_non_journal_captures():
    async def run():
        store = MemoryLiveSessionStore()
        blobs = MemoryBlobStore()
        fresh = _capture(cid="fresh", audio_uri=None,
                         labels={"journal": True,
                                 "journal_expires_at": (datetime.now(timezone.utc)
                                                        + timedelta(hours=47)).isoformat()})
        manual = _capture(cid="manual", audio_uri=None,
                          received_at="2020-01-01T00:00:00+00:00",   # ancient, but NOT journal
                          labels={"notes": "manual retro clip"})
        await store.put_capture(fresh)
        await store.put_capture(manual)
        assert await cleanup_expired_journal_captures("alice", store, blobs) == 0
        assert await store.get_capture("fresh") is not None
        assert await store.get_capture("manual") is not None
    asyncio.run(run())


def test_cleanup_falls_back_to_received_at_when_no_expiry_stamp():
    async def run():
        store = MemoryLiveSessionStore()
        blobs = MemoryBlobStore()
        # Uploaded 49h ago but never processed (no journal_expires_at stamp).
        stale = _capture(cid="stale", audio_uri=None,
                         received_at=(datetime.now(timezone.utc)
                                      - timedelta(hours=49)).isoformat(),
                         labels={"journal": True})
        await store.put_capture(stale)
        assert await cleanup_expired_journal_captures("alice", store, blobs) == 1
        assert await store.get_capture("stale") is None
    asyncio.run(run())


def test_processing_a_new_capture_cleans_up_the_account_expired_ones():
    async def run():
        store, blobs, fresh = await _seed()
        old = _capture(cid="old", audio_uri="memory://captures/alice/old.pcm",
                       labels={"journal": True, "journal_expires_at": _expired_iso()})
        await store.put_capture(old)
        await blobs.put(capture_key("alice", "old"), PCM_8S)
        await process_journal_capture(fresh.id, store, blobs, ScriptedDiarizer([("self", 0.0, 5.0)]))
        assert await store.get_capture("old") is None              # expired sibling swept
        assert await store.get_capture(fresh.id) is not None       # the new one kept
    asyncio.run(run())


# ------------------------------------------------------------------ the recording sink --


def _sink_recorder():
    calls = []

    async def sink(account_id, wav_bytes, title, context):
        calls.append({"account": account_id, "wav": wav_bytes, "title": title, "context": context})

    return sink, calls


def test_self_filtered_capture_feeds_the_recording_sink():
    async def run():
        store, blobs, cap = await _seed()
        d = ScriptedDiarizer([("self", 1.0, 3.0), ("self", 4.0, 6.5)])  # padded+merged >= 3 s
        sink, calls = _sink_recorder()
        await process_journal_capture(cap.id, store, blobs, d, sink)
        got = await store.get_capture(cap.id)
        assert got.labels["journal_status"] == "self_filtered"
        assert got.labels["journal_recording"] == "spawned"
        assert len(calls) == 1
        call = calls[0]
        assert call["account"] == "alice"
        assert call["title"].startswith("Journal (watch) — 2026-08-30")
        assert "own voice" in call["context"]
        import io
        import wave

        with wave.open(io.BytesIO(call["wav"]), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            kept = sum(e - s for s, e in got.labels["self_segments"])
            assert abs(wf.getnframes() / 16000 - kept) < 0.05
    asyncio.run(run())


def test_sink_not_called_below_minimum_or_without_voiceprint():
    async def run():
        sink, calls = _sink_recorder()
        store, blobs, cap = await _seed()
        await process_journal_capture(cap.id, store, blobs,
                                      ScriptedDiarizer([("self", 1.0, 2.0)]), sink)
        store2, blobs2, cap2 = await _seed(profile=False)
        await process_journal_capture(cap2.id, store2, blobs2,
                                      ScriptedDiarizer([("self", 0.0, 6.0)]), sink)
        assert calls == []
    asyncio.run(run())


def test_sink_failure_marks_labels_and_never_raises():
    async def run():
        store, blobs, cap = await _seed()

        async def bad_sink(*_a):
            raise RuntimeError("boom")

        await process_journal_capture(cap.id, store, blobs,
                                      ScriptedDiarizer([("self", 0.0, 6.0)]), bad_sink)
        got = await store.get_capture(cap.id)
        assert got.labels["journal_status"] == "self_filtered"
        assert got.labels["journal_recording"] == "failed"
    asyncio.run(run())
