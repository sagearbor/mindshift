"""Tests for consent-gated recording persistence + list/replay/delete.

GCS is never touched: a small in-memory :class:`FakeRecordingsStore` is injected
via ``app.state.recordings_store`` (the repo's DI style, mirroring
``app.state.llm_client``). The fake reuses the REAL
``recordings_store.plan_media_response`` so the Range/Content-Range math is
exercised for real, not re-implemented. Deepgram and the LLM are mocked exactly
as in test_analyze_upload, so the upload path is deterministic and key-free.
"""

import io
import json
import uuid
import wave
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import audio_ingest
import main
import recordings_store
from main import app, init_db

SR = 16000

# Deterministic stand-in for the ffmpeg audio derivative (build_derivatives is
# patched in the `store` fixture so the storage tests never invoke real ffmpeg
# and never depend on its output bytes). Long enough to slice a Range from.
FAKE_AUDIO_M4A = b"FAKE-M4A-AUDIO-DERIVATIVE-" * 20  # 500 bytes


# ---------------------------------------------------------------------------
# In-memory fake store (async interface identical to RecordingsStore)
# ---------------------------------------------------------------------------

class FakeRecordingsStore:
    def __init__(self, fail_on_save: bool = False):
        # {uid: {recording_id: {meta, turns, analysis, data, content_type}}}
        self._by_uid: dict[str, dict[str, dict]] = {}
        self.save_calls: list[dict] = []
        self._fail_on_save = fail_on_save

    async def save_recording(
        self, uid, *, audio_m4a, video_360p, original_filename,
        original_content_type, original_bytes, duration_seconds, turns,
        analysis, source=None, title=None,
    ):
        if self._fail_on_save:
            raise RuntimeError("simulated GCS outage")
        recording_id = str(uuid.uuid4())
        stored_variants = ["audio.m4a"]
        if video_360p is not None:
            stored_variants.append("video_360p.mp4")
        media_type = "video" if video_360p is not None else "audio"
        # open_media_stream serves the richest stored derivative.
        if video_360p is not None:
            media_bytes, media_ct = video_360p, "video/mp4"
        else:
            media_bytes, media_ct = audio_m4a, "audio/mp4"
        filename = original_filename or "recording"
        meta = {
            "id": recording_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            "title": (title or "").strip() or filename,
            "media_type": media_type,
            "duration_seconds": duration_seconds,
            "size_bytes": len(audio_m4a) + (len(video_360p) if video_360p else 0),
            "stored_variants": stored_variants,
            "original_bytes": original_bytes,
            "original_filename": original_filename,
            "original_content_type": original_content_type,
            "source": source or {
                "type": "upload", "url": None,
                "original_filename": original_filename,
            },
        }
        self._by_uid.setdefault(uid, {})[recording_id] = {
            "meta": meta, "turns": turns, "analysis": analysis,
            "data": media_bytes, "content_type": media_ct,
        }
        self.save_calls.append(
            {"uid": uid, "recording_id": recording_id, "audio_m4a": audio_m4a,
             "video_360p": video_360p, "turns": turns, "analysis": analysis,
             "source": source}
        )
        return recording_id

    async def list_recordings(self, uid):
        recs = self._by_uid.get(uid, {})
        out = [
            {**r["meta"], "has_analysis": r["analysis"] is not None}
            for r in recs.values()
        ]
        out.sort(key=lambda m: m["created_at"], reverse=True)
        return out

    async def get_recording(self, uid, recording_id):
        r = self._by_uid.get(uid, {}).get(recording_id)
        if r is None:
            return None
        return {**r["meta"], "turns": r["turns"], "analysis": r["analysis"]}

    async def recording_exists(self, uid, recording_id):
        return recording_id in self._by_uid.get(uid, {})

    async def update_source(self, uid, recording_id, source):
        r = self._by_uid.get(uid, {}).get(recording_id)
        if r is None:
            return None
        r["meta"]["source"] = source
        return source

    async def update_title(self, uid, recording_id, title):
        r = self._by_uid.get(uid, {}).get(recording_id)
        if r is None:
            return None
        r["meta"]["title"] = title
        return r["meta"]

    async def delete_recording(self, uid, recording_id):
        return self._by_uid.get(uid, {}).pop(recording_id, None) is not None

    async def open_media_stream(self, uid, recording_id, range_header):
        r = self._by_uid.get(uid, {}).get(recording_id)
        if r is None:
            return None
        payload = r["data"]
        start, end, status, headers = recordings_store.plan_media_response(
            len(payload), r["content_type"], range_header,
        )
        body = payload[start:end + 1]
        return recordings_store._iter_bytes(body), status, headers


# ---------------------------------------------------------------------------
# Fixtures — audio + mocks (mirrors test_analyze_upload)
# ---------------------------------------------------------------------------

def _wav_bytes(pcm: np.ndarray, sr: int = SR) -> bytes:
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _sine(freq: float, seconds: float, amp: float) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


_AMPS = [0.1, 0.2, 0.5, 0.15, 0.3, 0.08]
FIXTURE_WAV = _wav_bytes(
    np.concatenate([_sine(180.0, 1.0, a) for a in _AMPS]).astype(np.float32)
)

MOCK_TURNS = [
    {"speaker": "Speaker A", "text": "Hey, can we talk about the schedule?",
     "start_time": 0.0, "end_time": 1.0},
    {"speaker": "Speaker B", "text": "Sure, what about it.",
     "start_time": 1.0, "end_time": 2.0},
    {"speaker": "Speaker A", "text": "You never stick to what we agree.",
     "start_time": 2.0, "end_time": 3.0},
    {"speaker": "Speaker B", "text": "That is not fair and you know it.",
     "start_time": 3.0, "end_time": 4.0},
    {"speaker": "Speaker A", "text": "Okay. I hear you. Let me try again.",
     "start_time": 4.0, "end_time": 5.0},
    {"speaker": "Speaker B", "text": "Thanks. I appreciate that.",
     "start_time": 5.0, "end_time": 6.0},
]
_SPEAKERS = ["Speaker A", "Speaker B"]


def _analyze_llm_json(n_turns: int) -> str:
    return json.dumps({
        "per_turn": [
            {"heat": 20 + i * 3, "markers": [], "trigger_phrase": None}
            for i in range(n_turns)
        ],
        "requests": [],
        "narrative": "You both keep showing up and trying to reconnect.",
        "report_cards": {
            sp: {
                "score": 70,
                "headline": f"{sp} stayed engaged",
                "did_well": "Kept trying to reconnect.",
                "work_on": "Pause before answering criticism.",
            }
            for sp in _SPEAKERS
        },
    })


def _mock_llm(payload: str) -> MagicMock:
    m = MagicMock()
    m.complete.return_value = payload
    return m


@pytest.fixture
async def client():
    await init_db()
    main._rate_limiter.reset()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def store(monkeypatch):
    """Inject a fake store + patch build_derivatives to deterministic audio-only
    bytes (the storage tests use audio WAV → no real ffmpeg, no ffmpeg-output
    dependency). Torn back down so other modules see storage disabled."""
    monkeypatch.setattr(
        main, "build_derivatives",
        lambda data, **kw: audio_ingest.Derivatives(
            audio_m4a=FAKE_AUDIO_M4A, video_360p=None, has_video=False,
            video_note=None,
        ),
    )
    fake = FakeRecordingsStore()
    app.state.recordings_store = fake
    yield fake
    del app.state.recordings_store


def _upload(client, *, data=None, uid="test-user", **form):
    return client.post(
        "/analyze/upload",
        files={"file": ("clip.wav", data or FIXTURE_WAV, "audio/wav")},
        data=form,
        headers={"X-Test-Uid": uid},
    )


def _patched_upload():
    return (
        patch("main.transcribe_prerecorded", return_value=MOCK_TURNS),
        patch("main.get_llm_client",
              return_value=_mock_llm(_analyze_llm_json(len(MOCK_TURNS)))),
    )


# ---------------------------------------------------------------------------
# /analyze/upload persistence gate
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_upload_with_consent_stores(client, store):
    p1, p2 = _patched_upload()
    with p1, p2:
        resp = await _upload(client, consent="true", store="true")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["stored"] is True
    assert data["recording_id"]
    assert data["storage_note"] is None
    # The fake actually received the write — of the DERIVATIVE, not the original.
    assert len(store.save_calls) == 1
    call = store.save_calls[0]
    assert call["uid"] == "test-user"
    assert call["audio_m4a"] == FAKE_AUDIO_M4A  # compressed audio, not raw WAV
    assert call["video_360p"] is None           # WAV input → audio-only
    # Provenance: an upload has type "upload" and no url.
    assert call["source"] == {
        "type": "upload", "url": None, "original_filename": "clip.wav",
    }
    assert [t["text"] for t in call["turns"]] == [t["text"] for t in MOCK_TURNS]
    # Analysis is unchanged by storage.
    assert len(data["per_turn"]) == len(MOCK_TURNS)
    assert set(data["report_cards"]) == {"Speaker A", "Speaker B"}


@pytest.mark.anyio
async def test_upload_consent_false_does_not_store(client, store):
    p1, p2 = _patched_upload()
    with p1, p2:
        resp = await _upload(client, consent="false", store="true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["stored"] is False
    assert data["recording_id"] is None
    assert data["storage_note"] == "consent not given"
    assert store.save_calls == []  # nothing written


@pytest.mark.anyio
async def test_upload_storage_disabled_note(client):
    # No store fixture → app.state has no recordings_store → disabled.
    p1, p2 = _patched_upload()
    with p1, p2:
        resp = await _upload(client, consent="true", store="true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["stored"] is False
    assert data["storage_note"] == "storage not enabled"


@pytest.mark.anyio
async def test_upload_storage_failure_degrades_but_analysis_ok(client, monkeypatch):
    monkeypatch.setattr(
        main, "build_derivatives",
        lambda data, **kw: audio_ingest.Derivatives(
            audio_m4a=FAKE_AUDIO_M4A, video_360p=None, has_video=False,
            video_note=None,
        ),
    )
    app.state.recordings_store = FakeRecordingsStore(fail_on_save=True)
    try:
        p1, p2 = _patched_upload()
        with p1, p2:
            resp = await _upload(client, consent="true", store="true")
    finally:
        del app.state.recordings_store
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["stored"] is False
    assert data["recording_id"] is None
    assert data["storage_note"] == "storage failed: RuntimeError"
    # Analysis still fully returned.
    assert len(data["per_turn"]) == len(MOCK_TURNS)
    assert len(data["turns"]) == len(MOCK_TURNS)


# ---------------------------------------------------------------------------
# Derivatives — we store compressed audio (+ 360p video), never the original
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_video_input_stores_both_derivatives(client, store, monkeypatch):
    """A video input yields BOTH an audio.m4a and a video_360p.mp4; media_type
    reflects the stored video, and the media stream serves the 360p clip."""
    monkeypatch.setattr(
        main, "build_derivatives",
        lambda data, **kw: audio_ingest.Derivatives(
            audio_m4a=FAKE_AUDIO_M4A, video_360p=b"FAKE-360P-VIDEO",
            has_video=True, video_note=None,
        ),
    )
    p1, p2 = _patched_upload()
    with p1, p2:
        resp = await _upload(client, consent="true", store="true")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["stored"] is True
    assert data["storage_note"] is None
    call = store.save_calls[0]
    assert call["audio_m4a"] == FAKE_AUDIO_M4A
    assert call["video_360p"] == b"FAKE-360P-VIDEO"

    rid = data["recording_id"]
    detail = await client.get(
        f"/recordings/{rid}", headers={"X-Test-Uid": "test-user"},
    )
    assert detail.json()["media_type"] == "video"
    # Media stream serves the 360p video derivative.
    mu = await client.get(
        f"/recordings/{rid}/media_url", headers={"X-Test-Uid": "test-user"},
    )
    tk = parse_qs(urlparse(mu.json()["url"]).query)["tk"][0]
    media = await client.get(f"/recordings/{rid}/media?tk={tk}")
    assert media.headers["content-type"].startswith("video/mp4")
    assert media.content == b"FAKE-360P-VIDEO"


@pytest.mark.anyio
async def test_video_transcode_failure_degrades_to_audio_only(client, store, monkeypatch):
    """When the 360p transcode fails, the recording is still stored (audio-only)
    with an honest note — analysis + audio replay still work."""
    monkeypatch.setattr(
        main, "build_derivatives",
        lambda data, **kw: audio_ingest.Derivatives(
            audio_m4a=FAKE_AUDIO_M4A, video_360p=None, has_video=True,
            video_note="video replay unavailable: libx264 not available",
        ),
    )
    p1, p2 = _patched_upload()
    with p1, p2:
        resp = await _upload(client, consent="true", store="true")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["stored"] is True
    assert data["recording_id"]
    assert data["storage_note"] == "video replay unavailable: libx264 not available"
    call = store.save_calls[0]
    assert call["audio_m4a"] == FAKE_AUDIO_M4A
    assert call["video_360p"] is None  # audio-only degrade


@pytest.mark.anyio
async def test_audio_transcode_failure_is_honest_storage_failure(client, store, monkeypatch):
    """When even the AUDIO derivative can't be produced, storage fails honestly
    (stored=false + note) — the analysis still returns."""
    def _boom(data, **kw):
        raise audio_ingest.TranscodeError("ffmpeg unavailable")

    monkeypatch.setattr(main, "build_derivatives", _boom)
    p1, p2 = _patched_upload()
    with p1, p2:
        resp = await _upload(client, consent="true", store="true")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["stored"] is False
    assert data["recording_id"] is None
    assert data["storage_note"] == "storage failed: TranscodeError"
    assert store.save_calls == []  # nothing persisted
    # Analysis still fully returned.
    assert len(data["per_turn"]) == len(MOCK_TURNS)


def test_build_derivatives_real_ffmpeg_audio_only():
    """Integration: build_derivatives on a real WAV produces a non-empty m4a and
    (no video track) no 360p. Skips if the bundled ffmpeg is unavailable."""
    try:
        derivs = audio_ingest.build_derivatives(FIXTURE_WAV)
    except audio_ingest.TranscodeError as exc:  # pragma: no cover — env-dependent
        pytest.skip(f"ffmpeg unavailable: {exc}")
    assert derivs.has_video is False
    assert derivs.video_360p is None
    assert derivs.video_note is None
    assert len(derivs.audio_m4a) > 0
    # An MP4/m4a container carries an "ftyp" box near the start.
    assert b"ftyp" in derivs.audio_m4a[:64]


# ---------------------------------------------------------------------------
# list / detail / delete
# ---------------------------------------------------------------------------

async def _store_one(client, uid="test-user"):
    p1, p2 = _patched_upload()
    with p1, p2:
        resp = await _upload(client, consent="true", store="true", uid=uid)
    assert resp.status_code == 200, resp.text
    return resp.json()["recording_id"]


@pytest.mark.anyio
async def test_list_and_detail_happy_path(client, store):
    rid = await _store_one(client)

    lst = await client.get("/recordings", headers={"X-Test-Uid": "test-user"})
    assert lst.status_code == 200
    recs = lst.json()["recordings"]
    assert len(recs) == 1
    row = recs[0]
    assert row["id"] == rid
    assert row["media_type"] == "audio"
    assert row["has_analysis"] is True
    assert row["source_type"] == "upload"
    assert set(row) == {
        "id", "created_at", "filename", "title", "media_type",
        "duration_seconds", "has_analysis", "source_type",
    }
    # No explicit title on store → falls back to the filename.
    assert row["title"] == row["filename"]

    detail = await client.get(
        f"/recordings/{rid}", headers={"X-Test-Uid": "test-user"},
    )
    assert detail.status_code == 200
    d = detail.json()
    assert d["id"] == rid
    assert [t["text"] for t in d["turns"]] == [t["text"] for t in MOCK_TURNS]
    assert d["analysis"]["narrative"]
    assert d["source"] == {
        "type": "upload", "url": None, "original_filename": "clip.wav",
    }
    assert set(d) == {
        "id", "created_at", "filename", "title", "media_type",
        "duration_seconds", "turns", "analysis", "source",
    }
    assert d["title"] == d["filename"]


@pytest.mark.anyio
async def test_delete_happy_path(client, store):
    rid = await _store_one(client)
    resp = await client.delete(
        f"/recordings/{rid}", headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 204
    # Gone now → 404.
    detail = await client.get(
        f"/recordings/{rid}", headers={"X-Test-Uid": "test-user"},
    )
    assert detail.status_code == 404


@pytest.mark.anyio
async def test_detail_and_delete_404_for_unknown(client, store):
    missing = str(uuid.uuid4())
    assert (await client.get(
        f"/recordings/{missing}", headers={"X-Test-Uid": "test-user"},
    )).status_code == 404
    assert (await client.delete(
        f"/recordings/{missing}", headers={"X-Test-Uid": "test-user"},
    )).status_code == 404


@pytest.mark.anyio
async def test_cross_uid_isolation(client, store):
    rid = await _store_one(client, uid="user-a")
    # user-b sees an empty list and cannot read or delete user-a's recording.
    lst_b = await client.get("/recordings", headers={"X-Test-Uid": "user-b"})
    assert lst_b.json()["recordings"] == []
    assert (await client.get(
        f"/recordings/{rid}", headers={"X-Test-Uid": "user-b"},
    )).status_code == 404
    assert (await client.delete(
        f"/recordings/{rid}", headers={"X-Test-Uid": "user-b"},
    )).status_code == 404
    # user-a still has it (b's delete was a no-op).
    assert (await client.get(
        f"/recordings/{rid}", headers={"X-Test-Uid": "user-a"},
    )).status_code == 200


# ---------------------------------------------------------------------------
# Storage-disabled → 503 on every recordings endpoint
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_recordings_endpoints_503_when_disabled(client):
    uid = {"X-Test-Uid": "test-user"}
    rid = str(uuid.uuid4())
    assert (await client.get("/recordings", headers=uid)).status_code == 503
    assert (await client.get(f"/recordings/{rid}", headers=uid)).status_code == 503
    assert (await client.delete(f"/recordings/{rid}", headers=uid)).status_code == 503
    assert (await client.get(
        f"/recordings/{rid}/media_url", headers=uid,
    )).status_code == 503


# ---------------------------------------------------------------------------
# media_url + media streaming
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_media_url_and_full_stream(client, store):
    rid = await _store_one(client)
    mu = await client.get(
        f"/recordings/{rid}/media_url", headers={"X-Test-Uid": "test-user"},
    )
    assert mu.status_code == 200
    body = mu.json()
    assert body["expires_in"] == 900
    parsed = urlparse(body["url"])
    assert parsed.path == f"/recordings/{rid}/media"
    tk = parse_qs(parsed.query)["tk"][0]
    # Token validates for this recording.
    assert main._verify_media_token(tk, rid) == "test-user"

    # Full fetch (no Range) → 200 + correct content type + full derivative bytes.
    media = await client.get(f"/recordings/{rid}/media?tk={tk}")
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("audio/mp4")
    assert media.headers["accept-ranges"] == "bytes"
    assert media.content == FAKE_AUDIO_M4A


@pytest.mark.anyio
async def test_media_range_request_206(client, store):
    rid = await _store_one(client)
    tk = main._make_media_token("test-user", rid, _future())
    media = await client.get(
        f"/recordings/{rid}/media?tk={tk}",
        headers={"Range": "bytes=0-99"},
    )
    assert media.status_code == 206
    assert media.headers["content-range"] == f"bytes 0-99/{len(FAKE_AUDIO_M4A)}"
    assert media.headers["content-length"] == "100"
    assert media.content == FAKE_AUDIO_M4A[:100]


@pytest.mark.anyio
async def test_media_rejects_garbage_and_expired_tokens(client, store):
    rid = await _store_one(client)
    # Garbage token.
    assert (await client.get(
        f"/recordings/{rid}/media?tk=not-a-real-token",
    )).status_code == 403
    # Expired token (past expiry, otherwise well-formed + correctly signed).
    expired = main._make_media_token("test-user", rid, int(_now()) - 10)
    assert (await client.get(
        f"/recordings/{rid}/media?tk={expired}",
    )).status_code == 403


@pytest.mark.anyio
async def test_media_token_bound_to_recording_and_uid(client, store):
    # Two recordings under two users.
    rid_a = await _store_one(client, uid="user-a")
    rid_b = await _store_one(client, uid="user-b")
    # A valid token for user-a's recording...
    tk_a = main._make_media_token("user-a", rid_a, int(_now()) + 900)
    # ...cannot be replayed against user-b's recording (signature covers the id).
    assert (await client.get(
        f"/recordings/{rid_b}/media?tk={tk_a}",
    )).status_code == 403
    # It does work for its own recording.
    ok = await client.get(f"/recordings/{rid_a}/media?tk={tk_a}")
    assert ok.status_code == 200


def _now() -> float:
    import time
    return time.time()


def _future() -> int:
    return int(_now()) + 900


# ---------------------------------------------------------------------------
# source_url — HD replay from a link-sourced recording's original source
# ---------------------------------------------------------------------------

def _seed_link_recording(store, uid="test-user", url="https://photos.app.goo.gl/abc"):
    """Inject a link-sourced recording straight into the fake store (bypassing the
    upload path) so the source_url endpoint has a `source.type == "link"` row with
    a durable url to re-resolve."""
    rid = str(uuid.uuid4())
    meta = {
        "id": rid,
        "created_at": "2026-07-01T10:00:00Z",
        "filename": "linked.mp4",
        "media_type": "video",
        "duration_seconds": 12,
        "size_bytes": len(FAKE_AUDIO_M4A),
        "stored_variants": ["audio.m4a", "video_360p.mp4"],
        "original_bytes": 999,
        "original_filename": "linked.mp4",
        "original_content_type": "video/mp4",
        "source": {"type": "link", "url": url, "original_filename": "linked.mp4"},
    }
    store._by_uid.setdefault(uid, {})[rid] = {
        "meta": meta, "turns": [], "analysis": None,
        "data": FAKE_AUDIO_M4A, "content_type": "video/mp4",
    }
    return rid


@pytest.mark.anyio
async def test_source_url_resolves_link_source(client, store, monkeypatch):
    url = "https://photos.app.goo.gl/abc"
    rid = _seed_link_recording(store, url=url)
    seen = {}

    def _fake_resolve(u):
        seen["url"] = u
        return "https://lh3.googleusercontent.com/pw/XYZ=dv", "video/mp4"

    monkeypatch.setattr(main.link_fetch, "resolve_media_url", _fake_resolve)
    resp = await client.get(
        f"/recordings/{rid}/source_url", headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The DURABLE stored url was handed to the resolver (not a derivative).
    assert seen["url"] == url
    assert body["url"] == "https://lh3.googleusercontent.com/pw/XYZ=dv"
    assert body["content_type"] == "video/mp4"
    assert "expire" in body["expires_hint"].lower()


@pytest.mark.anyio
async def test_source_url_404_for_upload_source(client, store):
    # A normal upload has source.type "upload" — no remote source to stream.
    rid = await _store_one(client)
    resp = await client.get(
        f"/recordings/{rid}/source_url", headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "no remote source for this recording"


@pytest.mark.anyio
async def test_source_url_resolution_failure_is_honest(client, store, monkeypatch):
    rid = _seed_link_recording(store)

    # A revoked/unparseable link → the resolver's LinkError status passes through.
    def _raise_link_error(u):
        raise main.link_fetch.LinkError(422, "link revoked")

    monkeypatch.setattr(main.link_fetch, "resolve_media_url", _raise_link_error)
    resp = await client.get(
        f"/recordings/{rid}/source_url", headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"] == "link revoked"

    # An UNEXPECTED upstream failure → an honest 502 (client falls back).
    def _boom(u):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(main.link_fetch, "resolve_media_url", _boom)
    resp = await client.get(
        f"/recordings/{rid}/source_url", headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_source_url_404_for_unknown_recording(client, store):
    rid = str(uuid.uuid4())
    resp = await client.get(
        f"/recordings/{rid}/source_url", headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_source_url_cross_uid_isolation(client, store, monkeypatch):
    rid = _seed_link_recording(store, uid="user-a")
    monkeypatch.setattr(
        main.link_fetch, "resolve_media_url",
        lambda u: ("https://cdn/x=dv", "video/mp4"),
    )
    # user-b cannot resolve user-a's link source — reads as a plain 404.
    resp = await client.get(
        f"/recordings/{rid}/source_url", headers={"X-Test-Uid": "user-b"},
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_source_url_503_when_storage_disabled(client):
    rid = str(uuid.uuid4())
    resp = await client.get(
        f"/recordings/{rid}/source_url", headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# PATCH source — attach an HD source link to an existing recording after the fact
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_patch_source_attaches_link(client, store, monkeypatch):
    """Happy path: an upload recording gets a durable share link attached; the
    link is validated by RESOLVING (not downloading), the returned source is the
    link shape, and the fake store persists it verbatim."""
    rid = await _store_one(client)
    url = "https://photos.app.goo.gl/newHD"
    seen = {}

    def _fake_resolve(u):
        seen["url"] = u
        return "https://lh3.googleusercontent.com/pw/XYZ=dv", "video/mp4"

    monkeypatch.setattr(main.link_fetch, "resolve_media_url", _fake_resolve)
    resp = await client.patch(
        f"/recordings/{rid}/source",
        json={"url": url},
        headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 200, resp.text
    # The ORIGINAL pasted url was handed to the resolver for validation.
    assert seen["url"] == url
    # Response is the link-shaped source; original_filename preserved (clip.wav).
    assert resp.json() == {
        "type": "link", "url": url, "original_filename": "clip.wav",
    }
    # The fake store actually persisted the new source onto meta.
    persisted = (await store.get_recording("test-user", rid))["source"]
    assert persisted == {
        "type": "link", "url": url, "original_filename": "clip.wav",
    }
    # And the detail read now reflects the link provenance.
    detail = await client.get(
        f"/recordings/{rid}", headers={"X-Test-Uid": "test-user"},
    )
    assert detail.json()["source"]["type"] == "link"
    assert detail.json()["source"]["url"] == url


@pytest.mark.anyio
async def test_patch_source_unresolvable_link_422_leaves_meta_unchanged(
    client, store, monkeypatch,
):
    """An unusable link surfaces the resolver's LinkError as 422 with its detail
    passed through, and meta.json's source is UNCHANGED (still the upload)."""
    rid = await _store_one(client)
    before = (await store.get_recording("test-user", rid))["source"]

    def _raise(u):
        raise main.link_fetch.LinkError(
            422, "that Google Photos link contains multiple items — share a "
            "single video instead",
        )

    monkeypatch.setattr(main.link_fetch, "resolve_media_url", _raise)
    resp = await client.patch(
        f"/recordings/{rid}/source",
        json={"url": "https://photos.app.goo.gl/album"},
        headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 422
    assert "multiple items" in resp.json()["detail"]
    # Meta was never touched — still the original upload source.
    after = (await store.get_recording("test-user", rid))["source"]
    assert after == before
    assert after["type"] == "upload"


@pytest.mark.anyio
async def test_patch_source_404_for_unknown_recording(client, store, monkeypatch):
    # Resolver must never be called for a recording that doesn't exist.
    called = {"n": 0}

    def _resolve(u):
        called["n"] += 1
        return "https://cdn/x=dv", "video/mp4"

    monkeypatch.setattr(main.link_fetch, "resolve_media_url", _resolve)
    rid = str(uuid.uuid4())
    resp = await client.patch(
        f"/recordings/{rid}/source",
        json={"url": "https://photos.app.goo.gl/x"},
        headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 404
    assert called["n"] == 0


@pytest.mark.anyio
async def test_patch_source_cross_uid_404(client, store, monkeypatch):
    """user-b cannot attach a link to user-a's recording — reads as a plain 404
    and user-a's source is untouched."""
    rid = _seed_link_recording(store, uid="user-a", url="https://old/x")
    monkeypatch.setattr(
        main.link_fetch, "resolve_media_url",
        lambda u: ("https://cdn/x=dv", "video/mp4"),
    )
    resp = await client.patch(
        f"/recordings/{rid}/source",
        json={"url": "https://photos.app.goo.gl/hijack"},
        headers={"X-Test-Uid": "user-b"},
    )
    assert resp.status_code == 404
    # user-a's original source is unchanged.
    src = (await store.get_recording("user-a", rid))["source"]
    assert src["url"] == "https://old/x"


@pytest.mark.anyio
async def test_patch_source_503_when_storage_disabled(client):
    rid = str(uuid.uuid4())
    resp = await client.patch(
        f"/recordings/{rid}/source",
        json={"url": "https://photos.app.goo.gl/x"},
        headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 503


@pytest.mark.anyio
async def test_patch_source_requires_auth_401(client, store, monkeypatch):
    from auth import get_current_uid
    monkeypatch.delitem(app.dependency_overrides, get_current_uid)
    rid = str(uuid.uuid4())
    resp = await client.patch(
        f"/recordings/{rid}/source",
        json={"url": "https://photos.app.goo.gl/x"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Recording title — set on submit (default = filename) + PATCH /recordings/{id}
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_upload_title_persisted_and_returned(client, store):
    p1, p2 = _patched_upload()
    with p1, p2:
        resp = await _upload(
            client, consent="true", store="true", title="Kitchen argument",
        )
    assert resp.status_code == 200, resp.text
    rid = resp.json()["recording_id"]
    # The chosen title flows to list + detail (not the filename fallback).
    lst = await client.get("/recordings", headers={"X-Test-Uid": "test-user"})
    row = lst.json()["recordings"][0]
    assert row["title"] == "Kitchen argument"
    assert row["filename"] == "clip.wav"
    detail = await client.get(
        f"/recordings/{rid}", headers={"X-Test-Uid": "test-user"},
    )
    assert detail.json()["title"] == "Kitchen argument"


@pytest.mark.anyio
async def test_patch_title_renames_recording(client, store):
    rid = await _store_one(client)
    resp = await client.patch(
        f"/recordings/{rid}",
        json={"title": "  Renamed talk  "},  # surrounding whitespace is stripped
        headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"id": rid, "title": "Renamed talk"}
    # Persisted + reflected on the detail read.
    detail = await client.get(
        f"/recordings/{rid}", headers={"X-Test-Uid": "test-user"},
    )
    assert detail.json()["title"] == "Renamed talk"


@pytest.mark.anyio
async def test_patch_title_blank_is_422(client, store):
    rid = await _store_one(client)
    resp = await client.patch(
        f"/recordings/{rid}",
        json={"title": "   "},
        headers={"X-Test-Uid": "test-user"},
    )
    # Whitespace-only is rejected (min_length=1 after the model, guarded again in
    # the endpoint) — never a silent no-op rename.
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_patch_title_404_for_unknown_recording(client, store):
    rid = str(uuid.uuid4())
    resp = await client.patch(
        f"/recordings/{rid}",
        json={"title": "whatever"},
        headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_patch_title_cross_uid_404(client, store):
    """user-b cannot rename user-a's recording — plain 404, title untouched."""
    rid = await _store_one(client, uid="user-a")
    resp = await client.patch(
        f"/recordings/{rid}",
        json={"title": "hijacked"},
        headers={"X-Test-Uid": "user-b"},
    )
    assert resp.status_code == 404
    meta = await store.get_recording("user-a", rid)
    assert meta["title"] == meta["filename"]


@pytest.mark.anyio
async def test_patch_title_503_when_storage_disabled(client):
    rid = str(uuid.uuid4())
    resp = await client.patch(
        f"/recordings/{rid}",
        json={"title": "x"},
        headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 503
