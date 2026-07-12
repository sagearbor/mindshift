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

import main
import recordings_store
from main import app, init_db

SR = 16000


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
        self, uid, *, data, filename, content_type, duration_seconds, turns,
        analysis,
    ):
        if self._fail_on_save:
            raise RuntimeError("simulated GCS outage")
        recording_id = str(uuid.uuid4())
        meta = {
            "id": recording_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "filename": filename or "recording.bin",
            "media_type": recordings_store._media_type_for(content_type),
            "duration_seconds": duration_seconds,
            "size_bytes": len(data),
        }
        self._by_uid.setdefault(uid, {})[recording_id] = {
            "meta": meta, "turns": turns, "analysis": analysis,
            "data": data, "content_type": content_type or "application/octet-stream",
        }
        self.save_calls.append(
            {"uid": uid, "recording_id": recording_id, "data": data,
             "turns": turns, "analysis": analysis}
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
def store():
    """Inject a fake store, tearing it back down so other modules see disabled."""
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
    # The fake actually received the write.
    assert len(store.save_calls) == 1
    call = store.save_calls[0]
    assert call["uid"] == "test-user"
    assert call["data"] == FIXTURE_WAV
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
async def test_upload_storage_failure_degrades_but_analysis_ok(client):
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
    assert set(row) == {
        "id", "created_at", "filename", "media_type", "duration_seconds",
        "has_analysis",
    }

    detail = await client.get(
        f"/recordings/{rid}", headers={"X-Test-Uid": "test-user"},
    )
    assert detail.status_code == 200
    d = detail.json()
    assert d["id"] == rid
    assert [t["text"] for t in d["turns"]] == [t["text"] for t in MOCK_TURNS]
    assert d["analysis"]["narrative"]
    assert set(d) == {
        "id", "created_at", "filename", "media_type", "duration_seconds",
        "turns", "analysis",
    }


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

    # Full fetch (no Range) → 200 + correct content type + full bytes.
    media = await client.get(f"/recordings/{rid}/media?tk={tk}")
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("audio/wav")
    assert media.headers["accept-ranges"] == "bytes"
    assert media.content == FIXTURE_WAV


@pytest.mark.anyio
async def test_media_range_request_206(client, store):
    rid = await _store_one(client)
    tk = main._make_media_token("test-user", rid, _future())
    media = await client.get(
        f"/recordings/{rid}/media?tk={tk}",
        headers={"Range": "bytes=0-99"},
    )
    assert media.status_code == 206
    assert media.headers["content-range"] == f"bytes 0-99/{len(FIXTURE_WAV)}"
    assert media.headers["content-length"] == "100"
    assert media.content == FIXTURE_WAV[:100]


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
