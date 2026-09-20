"""Voice profile v2 — per-sample storage, delete-recompute, v1 migration.

v2 stores each enrollment's INDIVIDUAL embedding (``samples: [{id, embedding,
recording_id, speaker, at}]``) with the blended voiceprint recomputed as the
L2-normalized mean of the samples. A v1 profile migrates by treating its blend
as ONE legacy sample (deletable whole). Deleting the last sample leaves the
same state as "forget my voice" — no profile at all.

Torch-free like test_voice_enrollment.py: pure math tested directly, the router
exercised with a fake store + mocked embedder.
"""

import uuid

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import speaker_id
from main import app, init_db

pytestmark = pytest.mark.anyio

E_X = np.array([1.0, 0.0], dtype=np.float32)
E_Y = np.array([0.0, 1.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Pure math — as_v2 / new_profile / remove_sample / blend
# ---------------------------------------------------------------------------

def _v1_profile(embedding=E_X, count=3) -> dict:
    return {
        "version": 1,
        "embedding": [float(x) for x in speaker_id.l2_normalize(embedding)],
        "dim": int(np.asarray(embedding).size),
        "enroll_count": count,
        "model": "m@rev",
        "created_at": "t0",
        "updated_at": "t1",
        "sources": [{"recording_id": "r1", "speaker": "Speaker A", "at": "t1"}],
    }


def test_as_v2_none_passthrough():
    assert speaker_id.as_v2(None) is None


def test_as_v2_migrates_v1_blend_to_one_legacy_sample():
    v2 = speaker_id.as_v2(_v1_profile(count=3))
    assert v2["version"] == 2
    assert len(v2["samples"]) == 1
    legacy = v2["samples"][0]
    # Deterministic id — the client can delete it even before the profile has
    # ever been rewritten as v2.
    assert legacy["id"] == speaker_id.LEGACY_SAMPLE_ID
    assert legacy["recording_id"] is None
    assert "pre-v2 blend of 3 enrollments" in legacy["note"]
    # The blend carries over unchanged; count now counts SAMPLES.
    assert np.allclose(v2["embedding"], _v1_profile()["embedding"])
    assert v2["enroll_count"] == 1
    assert v2["created_at"] == "t0"


def test_as_v2_passes_v2_through_unchanged():
    p = speaker_id.new_profile(
        E_X, None, recording_id="r1", speaker="Speaker A", now_iso="t0",
    )
    assert speaker_id.as_v2(p) is p


def test_new_profile_v2_first_enrollment_stores_sample():
    p = speaker_id.new_profile(
        E_X, None, recording_id="r1", speaker="Speaker A", now_iso="t0",
        sample_id="s1",
    )
    assert p["version"] == 2
    assert p["enroll_count"] == 1
    assert len(p["samples"]) == 1
    s = p["samples"][0]
    assert s == {
        "id": "s1",
        "embedding": [1.0, 0.0],
        "recording_id": "r1",
        "speaker": "Speaker A",
        "at": "t0",
    }
    assert np.allclose(p["embedding"], [1.0, 0.0])


def test_new_profile_v2_blend_is_normalized_mean_of_samples():
    p1 = speaker_id.new_profile(
        E_X, None, recording_id="r1", speaker="Speaker A", now_iso="t0",
        sample_id="s1",
    )
    p2 = speaker_id.new_profile(
        E_Y, p1, recording_id="r2", speaker="Speaker B", now_iso="t1",
        sample_id="s2",
    )
    assert p2["enroll_count"] == 2
    assert [s["id"] for s in p2["samples"]] == ["s1", "s2"]
    v = np.asarray(p2["embedding"], dtype=np.float32)
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert v[0] == pytest.approx(v[1], abs=1e-6)  # 45° between the two
    assert p2["created_at"] == "t0"
    assert p2["updated_at"] == "t1"


def test_new_profile_generates_sample_ids_when_not_given():
    p = speaker_id.new_profile(
        E_X, None, recording_id="r1", speaker="Speaker A", now_iso="t0",
    )
    q = speaker_id.new_profile(
        E_Y, p, recording_id="r2", speaker="Speaker B", now_iso="t1",
    )
    ids = [s["id"] for s in q["samples"]]
    assert all(isinstance(i, str) and i for i in ids)
    assert len(set(ids)) == 2


def test_new_profile_onto_v1_migrates_then_appends():
    p = speaker_id.new_profile(
        E_Y, _v1_profile(count=4), recording_id="r9", speaker="Speaker B",
        now_iso="t2", sample_id="s-new",
    )
    assert p["version"] == 2
    assert p["enroll_count"] == 2
    assert [s["id"] for s in p["samples"]] == [
        speaker_id.LEGACY_SAMPLE_ID, "s-new",
    ]
    # Blend = normalized mean of (legacy blend, new sample).
    v = np.asarray(p["embedding"], dtype=np.float32)
    assert v[0] == pytest.approx(v[1], abs=1e-6)


def test_remove_sample_recomputes_blend():
    p1 = speaker_id.new_profile(
        E_X, None, recording_id="r1", speaker="Speaker A", now_iso="t0",
        sample_id="s1",
    )
    p2 = speaker_id.new_profile(
        E_Y, p1, recording_id="r2", speaker="Speaker B", now_iso="t1",
        sample_id="s2",
    )
    out = speaker_id.remove_sample(p2, "s2", now_iso="t2")
    assert out is not None
    assert out["enroll_count"] == 1
    assert [s["id"] for s in out["samples"]] == ["s1"]
    # Blend snapped back to s1 alone.
    assert np.allclose(out["embedding"], [1.0, 0.0])
    assert out["updated_at"] == "t2"


def test_remove_last_sample_returns_none_empty_profile():
    p1 = speaker_id.new_profile(
        E_X, None, recording_id="r1", speaker="Speaker A", now_iso="t0",
        sample_id="s1",
    )
    assert speaker_id.remove_sample(p1, "s1", now_iso="t2") is None


def test_remove_unknown_sample_raises():
    p1 = speaker_id.new_profile(
        E_X, None, recording_id="r1", speaker="Speaker A", now_iso="t0",
        sample_id="s1",
    )
    with pytest.raises(KeyError):
        speaker_id.remove_sample(p1, "nope", now_iso="t2")


def test_remove_legacy_sample_from_v1_profile():
    # A v1 profile's whole blend is ONE deletable legacy sample.
    assert (
        speaker_id.remove_sample(
            _v1_profile(), speaker_id.LEGACY_SAMPLE_ID, now_iso="t2",
        )
        is None
    )


# ---------------------------------------------------------------------------
# Router — profile detail + per-sample delete
# ---------------------------------------------------------------------------

class FakeVoiceStore:
    def __init__(self):
        self._recordings: dict[tuple, dict] = {}
        self._voiceprints: dict[str, dict] = {}

    def add_recording(self, uid, rid, turns, audio=b"AUDIO"):
        self._recordings[(uid, rid)] = {"turns": turns, "audio": audio}

    async def get_recording(self, uid, recording_id):
        r = self._recordings.get((uid, recording_id))
        if r is None:
            return None
        return {"id": recording_id, "turns": r["turns"], "analysis": None}

    async def get_audio_bytes(self, uid, recording_id):
        r = self._recordings.get((uid, recording_id))
        return None if r is None else r["audio"]

    # Multi-person voiceprints (Foundation B): the OWNER's profile stays at
    # ``_voiceprints[uid]`` (the tests above inspect it there — it is also the
    # legacy single-document shape the real store reads through as "self");
    # named partners live under ``_partners[(uid, person_id)]``.
    async def read_voiceprint(self, uid, person_id=None):
        pid = person_id or speaker_id.SELF_PERSON_ID
        if pid == speaker_id.SELF_PERSON_ID:
            doc = self._voiceprints.get(uid)
        else:
            doc = getattr(self, "_partners", {}).get((uid, pid))
        return speaker_id.as_person(doc, person_id=pid)

    async def list_voiceprints(self, uid):
        out = []
        if uid in self._voiceprints:
            out.append(speaker_id.as_person(self._voiceprints[uid]))
        for (u, pid), doc in getattr(self, "_partners", {}).items():
            if u == uid:
                out.append(speaker_id.as_person(doc, person_id=pid))
        return out

    async def write_voiceprint(self, uid, profile):
        doc = speaker_id.as_person(profile)
        if doc["person_id"] == speaker_id.SELF_PERSON_ID:
            self._voiceprints[uid] = doc
        else:
            if not hasattr(self, "_partners"):
                self._partners = {}
            self._partners[(uid, doc["person_id"])] = doc

    async def delete_voiceprint(self, uid, person_id=None):
        pid = person_id or speaker_id.SELF_PERSON_ID
        if pid == speaker_id.SELF_PERSON_ID:
            return self._voiceprints.pop(uid, None) is not None
        return getattr(self, "_partners", {}).pop((uid, pid), None) is not None


@pytest.fixture
async def client():
    await init_db()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def voice_store():
    fake = FakeVoiceStore()
    app.state.recordings_store = fake
    yield fake
    del app.state.recordings_store


TURNS = [
    {"speaker": "Speaker A", "text": "hi", "start_time": 0.0, "end_time": 3.0},
    {"speaker": "Speaker B", "text": "yo", "start_time": 3.0, "end_time": 6.0},
]


def _enroll_ready(monkeypatch, embedding):
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    import routers.voice as voice_router
    monkeypatch.setattr(
        voice_router, "decode_to_pcm",
        lambda data, name: (np.zeros(16000 * 5, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(speaker_id, "embed_speaker", lambda *a, **k: embedding)


async def _enroll(client, voice_store, monkeypatch, uid="u1", embedding=None):
    rid = str(uuid.uuid4())
    voice_store.add_recording(uid, rid, TURNS)
    _enroll_ready(
        monkeypatch, embedding if embedding is not None else E_X,
    )
    res = await client.post(
        "/voice/enroll", json={"recording_id": rid, "speaker": "Speaker A"},
        headers={"X-Test-Uid": uid},
    )
    assert res.status_code == 200, res.text
    return rid


async def test_profile_detail_lists_samples_with_provenance(
    client, voice_store, monkeypatch,
):
    rid = await _enroll(client, voice_store, monkeypatch)
    res = await client.get("/voice/profile", headers={"X-Test-Uid": "u1"})
    body = res.json()
    assert body["enrolled"] is True
    assert body["enroll_count"] == 1
    assert len(body["samples"]) == 1
    s = body["samples"][0]
    assert s["recording_id"] == rid
    assert s["speaker"] == "Speaker A"
    assert s["at"]
    assert s["id"]
    # The raw signature NEVER leaves the server.
    assert "embedding" not in s


async def test_profile_detail_migrated_v1_shows_legacy_sample(
    client, voice_store, monkeypatch,
):
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    voice_store._voiceprints["u1"] = _v1_profile(count=2)
    res = await client.get("/voice/profile", headers={"X-Test-Uid": "u1"})
    body = res.json()
    assert body["enrolled"] is True
    assert body["enroll_count"] == 1
    s = body["samples"][0]
    assert s["id"] == speaker_id.LEGACY_SAMPLE_ID
    assert s["recording_id"] is None
    assert "pre-v2 blend of 2 enrollments" in s["note"]
    # A read never rewrites the stored doc (GET stays side-effect free).
    assert voice_store._voiceprints["u1"]["version"] == 1


async def test_delete_sample_recomputes_and_persists(
    client, voice_store, monkeypatch,
):
    await _enroll(client, voice_store, monkeypatch, embedding=E_X)
    await _enroll(client, voice_store, monkeypatch, embedding=E_Y)
    prof = await client.get("/voice/profile", headers={"X-Test-Uid": "u1"})
    samples = prof.json()["samples"]
    assert len(samples) == 2

    res = await client.request(
        "DELETE", f"/voice/samples/{samples[1]['id']}",
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["deleted"] is True
    assert body["enrolled"] is True
    assert body["enroll_count"] == 1
    # The stored blend snapped back to the remaining sample.
    stored = voice_store._voiceprints["u1"]
    assert np.allclose(stored["embedding"], [1.0, 0.0])


async def test_delete_last_sample_forgets_the_profile(
    client, voice_store, monkeypatch,
):
    await _enroll(client, voice_store, monkeypatch)
    prof = await client.get("/voice/profile", headers={"X-Test-Uid": "u1"})
    sid = prof.json()["samples"][0]["id"]
    res = await client.request(
        "DELETE", f"/voice/samples/{sid}", headers={"X-Test-Uid": "u1"},
    )
    body = res.json()
    assert body["deleted"] is True
    assert body["enrolled"] is False
    assert body["enroll_count"] == 0
    # Same state as "forget my voice": nothing stored at all.
    assert "u1" not in voice_store._voiceprints
    prof2 = await client.get("/voice/profile", headers={"X-Test-Uid": "u1"})
    assert prof2.json()["enrolled"] is False


async def test_delete_legacy_sample_from_v1_profile(
    client, voice_store, monkeypatch,
):
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    voice_store._voiceprints["u1"] = _v1_profile()
    res = await client.request(
        "DELETE", f"/voice/samples/{speaker_id.LEGACY_SAMPLE_ID}",
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 200
    assert res.json()["enrolled"] is False
    assert "u1" not in voice_store._voiceprints


async def test_delete_unknown_sample_404(client, voice_store, monkeypatch):
    await _enroll(client, voice_store, monkeypatch)
    res = await client.request(
        "DELETE", "/voice/samples/not-a-sample", headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 404


async def test_delete_sample_without_profile_404(client, voice_store):
    res = await client.request(
        "DELETE", "/voice/samples/whatever", headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 404


async def test_delete_sample_storage_disabled_503(client):
    res = await client.request(
        "DELETE", "/voice/samples/whatever", headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 503


async def test_delete_sample_is_uid_scoped(client, voice_store, monkeypatch):
    await _enroll(client, voice_store, monkeypatch, uid="u1")
    prof = await client.get("/voice/profile", headers={"X-Test-Uid": "u1"})
    sid = prof.json()["samples"][0]["id"]
    # u2 cannot delete u1's sample — u2 simply has no profile → 404.
    res = await client.request(
        "DELETE", f"/voice/samples/{sid}", headers={"X-Test-Uid": "u2"},
    )
    assert res.status_code == 404
    assert "u1" in voice_store._voiceprints


async def test_enrollment_after_migration_keeps_legacy_sample(
    client, voice_store, monkeypatch,
):
    # Enrolling on top of a v1 profile persists the v2 form: legacy blend
    # sample + the new individual sample.
    voice_store._voiceprints["u1"] = _v1_profile(count=5)
    rid = str(uuid.uuid4())
    voice_store.add_recording("u1", rid, TURNS)
    _enroll_ready(monkeypatch, E_Y)
    res = await client.post(
        "/voice/enroll", json={"recording_id": rid, "speaker": "Speaker A"},
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 200
    assert res.json()["enroll_count"] == 2
    stored = voice_store._voiceprints["u1"]
    assert stored["version"] == 2
    ids = [s["id"] for s in stored["samples"]]
    assert ids[0] == speaker_id.LEGACY_SAMPLE_ID
    assert len(ids) == 2


# ---------------------------------------------------------------------------
# Per-RECORDING blending (2026-08-27) — breadth of settings, not tap count
# ---------------------------------------------------------------------------

def test_blend_is_per_recording_not_per_sample():
    """Three samples from the same recording form ONE centroid; a lone sample
    from another recording weighs the same as all three together (45°), not
    1:3. The owner's real print had 3 of 5 samples from one restaurant clip."""
    p = None
    for i in range(3):
        p = speaker_id.new_profile(
            E_X, p, recording_id="r1", speaker="Speaker A", now_iso=f"t{i}",
            sample_id=f"s{i}",
        )
    p = speaker_id.new_profile(
        E_Y, p, recording_id="r2", speaker="Speaker B", now_iso="t3",
        sample_id="s3",
    )
    v = np.asarray(p["embedding"], dtype=np.float32)
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert v[0] == pytest.approx(v[1], abs=1e-6)  # 45°: r1 and r2 weigh equally
    assert p["enroll_count"] == 4
    assert speaker_id.profile_settings(p) == 2


def test_guided_samples_are_each_their_own_setting():
    a = speaker_id.new_profile(
        E_X, None, recording_id=None, speaker=None, now_iso="t0",
        sample_id="g1", note="guided enrollment",
    )
    b = speaker_id.new_profile(
        E_Y, a, recording_id=None, speaker=None, now_iso="t1",
        sample_id="g2", note="guided enrollment",
    )
    assert speaker_id.profile_settings(b) == 2
    v = np.asarray(b["embedding"], dtype=np.float32)
    assert v[0] == pytest.approx(v[1], abs=1e-6)


def test_profile_settings_edge_cases():
    assert speaker_id.profile_settings(None) == 0
    assert speaker_id.profile_settings({}) == 0
    # A v1 print (blend only, no samples) is one setting.
    assert speaker_id.profile_settings({"embedding": [1.0, 0.0]}) == 1
