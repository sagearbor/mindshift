"""Endpoint tests for the outcome-engine mood check (Workstream 4 —
docs/plans/2026-09-04-naturalturn-conversation-quality.md): CANDOR's single
outcome item ("positive vs negative feelings right now", 1-9) taken once
before and once after a live session.

``mood_before`` rides in on the existing ``POST /sessions/live`` body (the
phone already has it at stop). ``mood_after`` is answered a beat later, once
the episode already exists, so it has its own endpoint:
``PATCH /sessions/live/{episode_id}/mood``.

GCS/LLM are never touched: reuses test_sessions_live's FakeLiveStore (the
same in-memory fake this router's other suites inject at
``app.state.recordings_store``), extended with ``update_mood`` and the
``mood_after``-survives-a-re-POST preservation real recordings_store.py's
``_save_live_session_sync`` implements.
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from main import app, init_db
from routers import sessions as sessions_router
from tests.test_sessions_live import FakeLiveStore, _body

pytestmark = pytest.mark.anyio


class FakeMoodStore(FakeLiveStore):
    """FakeLiveStore + update_mood, with mood_after preserved across a
    re-POST — mirrors recordings_store.RecordingsStore._save_live_session_sync
    (mood_after is set by a separate PATCH, never by the POST body, so a
    later re-POST's meta never carries it)."""

    async def save_live_session(self, uid, recording_id, *, meta, turns, analysis):
        self.save_calls += 1
        slot = self._by_uid.setdefault(uid, {})
        written = dict(meta)
        existing = slot.get(recording_id)
        if existing:
            old = existing["meta"]
            for key in ("manual_speaker_labels", "shares", "mood_after"):
                if key in old and key not in written:
                    written[key] = old[key]
        slot[recording_id] = {"meta": written, "turns": turns, "analysis": analysis}
        return written

    async def update_mood(self, uid, recording_id, *, mood_after):
        r = self._by_uid.get(uid, {}).get(recording_id)
        if r is None:
            return None
        r["meta"]["mood_after"] = mood_after
        return r["meta"]

    def add_upload_recording(self, uid, recording_id):
        """A stored UPLOAD (not a live session) — mood can't be PATCHed
        onto it (moods are a live-session concept only)."""
        self._by_uid.setdefault(uid, {})[recording_id] = {
            "meta": {
                "id": recording_id, "created_at": "2026-08-20T10:00:00+00:00",
                "filename": "clip.m4a", "title": "clip", "media_type": "audio",
                "source": {"type": "upload", "url": None, "original_filename": "clip.m4a"},
            },
            "turns": [], "analysis": None,
        }


@pytest.fixture
async def client():
    await init_db()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def store():
    fake = FakeMoodStore()
    app.state.recordings_store = fake
    sessions_router._REFLECT_LOCKS.clear()
    sessions_router._REFLECT_LOCK_USERS.clear()
    yield fake
    del app.state.recordings_store


async def _ingest(client, **overrides):
    body = _body(analyze=False, reflect=False, **overrides)
    res = await client.post("/sessions/live", json=body)
    assert res.status_code == 201, res.text
    return res.json()["episode_id"]


class TestMoodBefore:
    async def test_stored_and_readable(self, client, store):
        rid = await _ingest(client, mood_before=3)
        detail = (await client.get(f"/recordings/{rid}")).json()
        assert detail["mood_before"] == 3
        assert detail["mood_after"] is None

    async def test_absent_when_not_sent(self, client, store):
        rid = await _ingest(client)
        detail = (await client.get(f"/recordings/{rid}")).json()
        assert detail["mood_before"] is None

    async def test_out_of_range_is_422(self, client, store):
        for bad in (0, 10, -1):
            res = await client.post("/sessions/live", json=_body(mood_before=bad))
            assert res.status_code == 422, bad
        assert store.save_calls == 0

    async def test_repost_can_update_mood_before(self, client, store):
        rid = await _ingest(client, mood_before=4)
        rid2 = await _ingest(client, mood_before=7)
        assert rid2 == rid
        detail = (await client.get(f"/recordings/{rid}")).json()
        assert detail["mood_before"] == 7


class TestMoodAfterPatch:
    async def test_patches_the_episode(self, client, store):
        rid = await _ingest(client, mood_before=3)
        res = await client.patch(f"/sessions/live/{rid}/mood", json={"mood_after": 6})
        assert res.status_code == 200, res.text
        assert res.json() == {"episode_id": rid, "mood_after": 6}
        detail = (await client.get(f"/recordings/{rid}")).json()
        assert detail["mood_before"] == 3 and detail["mood_after"] == 6

    async def test_survives_a_repost_of_the_same_session(self, client, store):
        """Review case: the phone re-POSTs (e.g. a retry) AFTER the AFTER
        check has already been PATCHed in — the retry's body never carries
        mood_after, so it must not be wiped."""
        rid = await _ingest(client, mood_before=3)
        await client.patch(f"/sessions/live/{rid}/mood", json={"mood_after": 8})
        rid2 = await _ingest(client, mood_before=3)
        assert rid2 == rid
        detail = (await client.get(f"/recordings/{rid}")).json()
        assert detail["mood_after"] == 8

    async def test_out_of_range_is_422(self, client, store):
        rid = await _ingest(client)
        for bad in (0, 10, -3):
            res = await client.patch(f"/sessions/live/{rid}/mood", json={"mood_after": bad})
            assert res.status_code == 422, bad

    async def test_missing_episode_is_404(self, client, store):
        res = await client.patch(
            f"/sessions/live/{uuid.uuid4()}/mood", json={"mood_after": 5},
        )
        assert res.status_code == 404

    async def test_foreign_episode_is_404(self, client, store):
        rid = await _ingest(client)
        res = await client.patch(
            f"/sessions/live/{rid}/mood", json={"mood_after": 5},
            headers={"X-Test-Uid": "user-b"},
        )
        assert res.status_code == 404

    async def test_upload_recording_is_404_not_a_live_episode(self, client, store):
        rid = str(uuid.uuid4())
        store.add_upload_recording("test-user", rid)
        res = await client.patch(f"/sessions/live/{rid}/mood", json={"mood_after": 5})
        assert res.status_code == 404

    async def test_storage_disabled_is_503(self, client):
        res = await client.patch(
            f"/sessions/live/{uuid.uuid4()}/mood", json={"mood_after": 5},
        )
        assert res.status_code == 503

    async def test_requires_auth(self, client, store):
        from auth import get_current_uid

        rid = await _ingest(client)
        saved = app.dependency_overrides.pop(get_current_uid)
        try:
            res = await client.patch(f"/sessions/live/{rid}/mood", json={"mood_after": 5})
            assert res.status_code == 401
        finally:
            app.dependency_overrides[get_current_uid] = saved
