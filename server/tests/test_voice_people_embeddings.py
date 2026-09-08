"""``GET /voice/people?include_embeddings=true`` — the on-device opt-in.

Mirrors test_voice_people.py's fake store + auth harness (torch-free).
Covers, per the contract in routers/voice.py::list_voice_people:

* the DEFAULT response is byte-for-byte what it was: no ``embedding`` key
  at all (not ``null``) — "the raw signature never leaves the server" stays
  the default, and ``GET /voice/profile`` never grows one either;
* with the flag, every enrolled person of the CALLER carries ``embedding``
  (L2-normalized, ``dim`` long, the same vector the server matches with —
  a stored un-normalized blend is normalized on the way out), plus ``dim``
  and ``model`` — the owner first, partners after;
* scope: another account sees only its own people (an empty list here),
  never the caller's prints; storage disabled / nobody enrolled -> empty,
  never a 503;
* a legacy v1 (single-document) owner print is served through the same
  view with its embedding;
* the OpenAPI schema advertises the query parameter and the optional field
  (what the generated mobile types are built from).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import speaker_id
from main import app, init_db

pytestmark = pytest.mark.anyio


class FakeVoiceStore:
    """The voiceprint slice of test_voice_people.py's fake store (pytest's
    importlib mode keeps test modules un-importable by name, so the three
    methods this file needs are restated rather than imported)."""

    def __init__(self):
        self._people: dict[tuple, dict] = {}  # (uid, person_id) -> profile

    async def read_voiceprint(self, uid, person_id=None):
        pid = person_id or speaker_id.SELF_PERSON_ID
        return speaker_id.as_person(self._people.get((uid, pid)), person_id=pid)

    async def list_voiceprints(self, uid):
        return [speaker_id.as_person(p, person_id=pid) for (u, pid), p in self._people.items() if u == uid]

    async def write_voiceprint(self, uid, profile):
        doc = speaker_id.as_person(profile)
        self._people[(uid, doc["person_id"])] = doc


def _doc(embedding, **extra):
    return {"version": 2, "embedding": list(map(float, embedding)), "dim": len(embedding),
            "enroll_count": 1, "samples": [{"id": "s1", "embedding": list(map(float, embedding))}],
            **extra}

H = {"X-Test-Uid": "u1"}
E_SELF = np.array([1.0, 0.0, 0.0], dtype=np.float32)
E_ALEX_RAW = np.array([0.0, 3.0, 4.0], dtype=np.float32)  # norm 5: NOT unit


@pytest.fixture
async def client():
    await init_db()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
def voice_store():
    fake = FakeVoiceStore()
    app.state.recordings_store = fake
    yield fake
    del app.state.recordings_store


async def _enroll_two(voice_store):
    await voice_store.write_voiceprint("u1", _doc(E_SELF, model="ecapa@rev"))
    await voice_store.write_voiceprint(
        "u1", _doc(E_ALEX_RAW, person_id="alex", display_name="Alex", model="ecapa@rev"),
    )


async def test_default_response_never_carries_an_embedding(client, voice_store, monkeypatch):
    await _enroll_two(voice_store)
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    res = await client.get("/voice/people", headers=H)
    assert res.status_code == 200
    rows = res.json()["people"]
    assert len(rows) == 2
    assert "embedding" not in res.text  # the key is ABSENT, not null
    # An explicit false is the same as absent.
    res = await client.get("/voice/people", params={"include_embeddings": "false"}, headers=H)
    assert "embedding" not in res.text
    # /profile has no such switch at all.
    res = await client.get("/voice/profile", headers=H)
    assert res.status_code == 200 and "embedding" not in res.text
    res = await client.get("/voice/profile", params={"include_embeddings": "true"}, headers=H)
    assert res.status_code == 200 and "embedding" not in res.text


async def test_include_embeddings_serves_normalized_prints_with_dim_and_model(client, voice_store, monkeypatch):
    await _enroll_two(voice_store)
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    res = await client.get("/voice/people", params={"include_embeddings": "true"}, headers=H)
    assert res.status_code == 200, res.text
    rows = res.json()["people"]
    assert [(p["person_id"], p["is_self"]) for p in rows] == [("self", True), ("alex", False)]
    me, alex = rows
    assert me["embedding"] == pytest.approx([1.0, 0.0, 0.0])
    assert me["dim"] == 3 and me["model"] == "ecapa@rev"
    # The stored blend was (0, 3, 4): served L2-normalized — the vector the
    # server's cosine effectively matches with.
    assert alex["embedding"] == pytest.approx([0.0, 0.6, 0.8])
    assert len(alex["embedding"]) == alex["dim"] == 3
    assert alex["display_name"] == "Alex" and alex["model"] == "ecapa@rev"
    # Everything else in the per-person shape is unchanged by the flag.
    assert me["enrolled"] is True and me["enroll_count"] == 1
    assert [s["id"] for s in me["samples"]] == ["s1"]
    # Samples still never carry their own embeddings.
    assert "embedding" not in json.dumps(me["samples"])


async def test_scope_is_the_callers_own_account(client, voice_store, monkeypatch):
    await _enroll_two(voice_store)
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    other = await client.get(
        "/voice/people", params={"include_embeddings": "true"}, headers={"X-Test-Uid": "u2"},
    )
    assert other.status_code == 200
    assert other.json()["people"] == []
    assert "embedding" not in other.text


async def test_storage_disabled_or_unenrolled_is_empty_not_503(client, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: False)
    res = await client.get("/voice/people", params={"include_embeddings": "true"}, headers=H)
    assert res.status_code == 200
    assert res.json() == {"available": False, "storage_enabled": False, "people": []}


async def test_legacy_v1_owner_print_is_served_with_its_embedding(client, voice_store, monkeypatch):
    # A pre-v2 single-document profile: no samples list, embedding at top level.
    voice_store._people[("u1", "self")] = {
        "embedding": [0.0, 0.0, 2.0], "dim": 3, "enroll_count": 3,
        "model": "ecapa@old", "updated_at": "2026-01-01T00:00:00+00:00",
    }
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    res = await client.get("/voice/people", params={"include_embeddings": "true"}, headers=H)
    assert res.status_code == 200
    (me,) = res.json()["people"]
    assert me["person_id"] == "self" and me["is_self"] is True
    assert me["embedding"] == pytest.approx([0.0, 0.0, 1.0])
    assert me["model"] == "ecapa@old"
    assert me["samples"][0]["id"] == "legacy-blend"


def test_openapi_advertises_the_switch_and_the_optional_field():
    spec = app.openapi()
    params = spec["paths"]["/voice/people"]["get"]["parameters"]
    names = {p["name"]: p for p in params}
    assert "include_embeddings" in names
    assert names["include_embeddings"]["schema"]["default"] is False
    props = spec["components"]["schemas"]["VoiceProfileResponse"]["properties"]
    assert "embedding" in props
    assert "embedding" not in spec["components"]["schemas"]["VoiceProfileResponse"].get("required", [])
    # `settings` (distinct recordings pooled) is part of the per-person shape
    # — what the phone's contrast match is gated on.
    assert props["settings"]["type"] == "integer"


# ---------------------------------------------------------------------------
# settings + the current per-recording blend (the phone's contrast match
# needs BOTH to agree with the server's own matcher)
# ---------------------------------------------------------------------------

def _two_recording_doc():
    """Three samples: two from recording r1 (which would outvote a per-sample
    mean), one from r2. Stored top-level `embedding` is deliberately STALE
    (an older blend rule) so the test can see which vector is served."""
    return {
        "version": 2, "dim": 3, "enroll_count": 3, "model": "ecapa@rev",
        "embedding": [0.0, 0.0, 1.0],
        "samples": [
            {"id": "a", "recording_id": "r1", "embedding": [1.0, 0.0, 0.0]},
            {"id": "b", "recording_id": "r1", "embedding": [1.0, 0.0, 0.0]},
            {"id": "c", "recording_id": "r2", "embedding": [0.0, 1.0, 0.0]},
        ],
    }


async def test_settings_counts_distinct_recordings_with_and_without_embeddings(client, voice_store, monkeypatch):
    await voice_store.write_voiceprint("u1", _two_recording_doc())
    await voice_store.write_voiceprint(
        "u1", _doc(E_ALEX_RAW, person_id="alex", display_name="Alex", model="ecapa@rev"),
    )
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    for params in ({}, {"include_embeddings": "true"}):
        res = await client.get("/voice/people", params=params, headers=H)
        assert res.status_code == 200, res.text
        me, alex = res.json()["people"]
        # 3 samples, 2 recordings: enroll_count 3 but settings 2.
        assert (me["enroll_count"], me["settings"]) == (3, 2)
        # One sample with no recording is its own setting.
        assert (alex["enroll_count"], alex["settings"]) == (1, 1)
    # /profile serves the same number; an unenrolled person has none.
    res = await client.get("/voice/profile", headers=H)
    assert res.json()["settings"] == 2
    res = await client.get("/voice/profile", params={"person_id": "nobody"}, headers=H)
    assert res.status_code == 200 and res.json()["settings"] == 0


async def test_include_embeddings_serves_the_current_per_recording_blend(client, voice_store, monkeypatch):
    doc = _two_recording_doc()
    await voice_store.write_voiceprint("u1", doc)
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    res = await client.get("/voice/people", params={"include_embeddings": "true"}, headers=H)
    assert res.status_code == 200, res.text
    (me,) = res.json()["people"]
    # One centroid PER RECORDING, then the mean: (1,0,0) and (0,1,0) ->
    # (1,1,0)/sqrt2 — NOT the stale stored blend (0,0,1) and NOT the
    # per-sample mean (2,1,0)/sqrt5 that r1's two taps would produce.
    assert me["embedding"] == pytest.approx([2 ** -0.5, 2 ** -0.5, 0.0], abs=1e-6)
    # Byte-for-byte the vector main's matcher loads for the same document.
    served = speaker_id.l2_normalize(speaker_id.current_blend(doc))
    assert me["embedding"] == pytest.approx(served.tolist(), abs=1e-6)
    assert np.allclose(served, speaker_id.blend_samples(doc["samples"]))


def test_current_blend_falls_back_to_the_stored_vector():
    # v1 (no samples) -> the stored blend; malformed samples -> the stored
    # blend; no vector at all -> None.
    assert np.allclose(speaker_id.current_blend({"embedding": [0.0, 3.0, 4.0]}), [0.0, 3.0, 4.0])
    assert np.allclose(
        speaker_id.current_blend({"embedding": [0.0, 3.0, 4.0], "samples": [{"id": "x"}]}), [0.0, 3.0, 4.0],
    )
    assert speaker_id.current_blend({"samples": [{"id": "x", "embedding": [1.0]}]}) is None
    assert speaker_id.current_blend(None) is None
