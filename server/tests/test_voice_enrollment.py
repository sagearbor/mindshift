"""Voice enrollment ("This is me") + auto-label ("You") — torch-free tests.

The heavy ECAPA embedder is NEVER loaded here: the pure vector math is tested
directly, and every path that would call the model monkeypatches
``speaker_id.embed_speaker`` / ``speaker_id.embed_pcm`` / ``speaker_id.is_available``
with deterministic doubles. So this whole file (and the base suite) stays green
WITHOUT torch/speechbrain installed. The real embedder is validated separately
by the empirical script in the PR (tmp/, not committed).

Coverage: pure math (cosine/normalize/running-mean/pool/new_profile);
identify_speakers labeling (match, below-threshold no-label, at most one "You");
and the router (enroll → store → match → label; delete actually deletes;
absent-dep 503; storage-disabled 503; missing recording/speaker 404/422).
"""

import uuid

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import speaker_id
from main import app, init_db

# Async tests here run under anyio (sync tests are unaffected by the marker),
# matching the rest of the server suite (see test_recordings.py).
pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Pure vector math — no torch
# ---------------------------------------------------------------------------

def test_l2_normalize_unit_and_zero():
    v = np.array([3.0, 4.0], dtype=np.float32)
    n = speaker_id.l2_normalize(v)
    assert np.isclose(np.linalg.norm(n), 1.0)
    # A zero vector is returned as-is (cosine against it is 0 — honest no-signal).
    z = speaker_id.l2_normalize(np.zeros(3, dtype=np.float32))
    assert np.allclose(z, 0.0)


def test_cosine_identity_orthogonal_opposite():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert speaker_id.cosine(a, a) == pytest.approx(1.0)
    assert speaker_id.cosine(a, b) == pytest.approx(0.0)
    assert speaker_id.cosine(a, -a) == pytest.approx(-1.0)
    # Mismatched shapes / empty → 0.0, never a crash.
    assert speaker_id.cosine(a, np.zeros(0, dtype=np.float32)) == 0.0


def test_running_mean_embedding_averages_and_normalizes():
    e0 = np.array([1.0, 0.0], dtype=np.float32)
    # First enrollment: just the (normalized) new vector.
    m1 = speaker_id.running_mean_embedding(None, 0, e0 * 5.0)
    assert np.allclose(m1, e0)
    # Second enrollment folds toward the new direction; result stays unit-norm.
    e1 = np.array([0.0, 1.0], dtype=np.float32)
    m2 = speaker_id.running_mean_embedding(m1, 1, e1)
    assert np.isclose(np.linalg.norm(m2), 1.0)
    # Equal weight of two orthogonal unit vectors → 45°.
    assert m2[0] == pytest.approx(m2[1], abs=1e-6)


def test_pool_speaker_pcm_slices_only_that_speaker():
    sr = 16000
    pcm = np.arange(sr * 4, dtype=np.float32)  # 4 seconds, ramp
    turns = [
        {"speaker": "A", "start_time": 0.0, "end_time": 1.0},
        {"speaker": "B", "start_time": 1.0, "end_time": 2.0},
        {"speaker": "A", "start_time": 2.0, "end_time": 3.0},
    ]
    pooled = speaker_id.pool_speaker_pcm(pcm, sr, turns, "A")
    # Two 1s A-turns → 2s of audio.
    assert pooled.size == sr * 2
    # First sample is the start of the first A turn (index 0).
    assert pooled[0] == 0.0
    # None of B's samples (sr..2sr) leaked in.
    assert (sr + 5) not in set(pooled.tolist())


def test_pool_respects_max_seconds_cap():
    sr = 16000
    pcm = np.zeros(sr * 200, dtype=np.float32)
    turns = [{"speaker": "A", "start_time": 0.0, "end_time": 200.0}]
    pooled = speaker_id.pool_speaker_pcm(pcm, sr, turns, "A", max_seconds=60.0)
    assert pooled.size == sr * 60


def test_new_profile_first_and_second_enrollment():
    e0 = np.array([1.0, 0.0], dtype=np.float32)
    p1 = speaker_id.new_profile(
        e0, None, recording_id="r1", speaker="Speaker A", now_iso="t0",
    )
    assert p1["enroll_count"] == 1
    assert p1["dim"] == 2
    # v2: each enrollment is stored as an individual sample with provenance
    # (the deeper v2 behavior is covered in test_voice_profile_v2.py).
    assert len(p1["samples"]) == 1
    assert p1["samples"][0]["recording_id"] == "r1"
    e1 = np.array([0.0, 1.0], dtype=np.float32)
    p2 = speaker_id.new_profile(
        e1, p1, recording_id="r2", speaker="Speaker A", now_iso="t1",
    )
    assert p2["enroll_count"] == 2
    assert p2["created_at"] == "t0"  # preserved across enrollments
    assert p2["updated_at"] == "t1"
    # The stored embedding is the running mean of the two, unit-norm.
    v = np.array(p2["embedding"], dtype=np.float32)
    assert np.isclose(np.linalg.norm(v), 1.0)


# ---------------------------------------------------------------------------
# identify_speakers — labeling logic with a mocked embedder
# ---------------------------------------------------------------------------

def _fake_embed_by_speaker(mapping):
    """Return an embed_speaker double that maps speaker label → fixed vector."""
    def _embed(pcm, sr, turns, speaker, **kw):
        return mapping.get(speaker)
    return _embed


def test_identify_labels_best_above_threshold(monkeypatch):
    e_you = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e_other = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(
        speaker_id, "embed_speaker",
        _fake_embed_by_speaker({"Speaker A": e_you, "Speaker B": e_other}),
    )
    turns = [
        {"speaker": "Speaker A", "start_time": 0.0, "end_time": 2.0},
        {"speaker": "Speaker B", "start_time": 2.0, "end_time": 4.0},
    ]
    report = speaker_id.identify_speakers(
        np.zeros(10, dtype=np.float32), 16000, turns, e_you, threshold=0.5,
    )
    assert report["matched_speaker"] == "Speaker A"
    assert report["speakers"]["Speaker A"]["is_you"] is True
    assert report["speakers"]["Speaker A"]["score"] == pytest.approx(1.0)
    assert report["speakers"]["Speaker B"]["is_you"] is False
    assert report["speakers"]["Speaker B"]["score"] == pytest.approx(0.0)


def test_identify_no_label_below_threshold(monkeypatch):
    # Best speaker is only 0.4 similar — below the 0.5 floor → no "You".
    e_you = np.array([1.0, 0.0], dtype=np.float32)
    weak = np.array([0.4, np.sqrt(1 - 0.16)], dtype=np.float32)
    monkeypatch.setattr(
        speaker_id, "embed_speaker", _fake_embed_by_speaker({"Speaker A": weak}),
    )
    turns = [{"speaker": "Speaker A", "start_time": 0.0, "end_time": 2.0}]
    report = speaker_id.identify_speakers(
        np.zeros(10, dtype=np.float32), 16000, turns, e_you, threshold=0.5,
    )
    assert report["matched_speaker"] is None
    assert report["speakers"]["Speaker A"]["is_you"] is False
    # The near-miss score is still recorded for debugging.
    assert report["speakers"]["Speaker A"]["score"] == pytest.approx(0.4, abs=1e-3)


def test_identify_skips_speaker_with_too_little_audio(monkeypatch):
    # embed_speaker returns None (too little pooled speech) → speaker omitted.
    monkeypatch.setattr(
        speaker_id, "embed_speaker", _fake_embed_by_speaker({"Speaker A": None}),
    )
    turns = [{"speaker": "Speaker A", "start_time": 0.0, "end_time": 0.2}]
    report = speaker_id.identify_speakers(
        np.zeros(10, dtype=np.float32), 16000, turns,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    assert report["matched_speaker"] is None
    assert report["speakers"] == {}


# ---------------------------------------------------------------------------
# Router — in-memory fake store, mocked embedder
# ---------------------------------------------------------------------------

class FakeVoiceStore:
    """Minimal async store exposing only what the voice router + pipeline use.

    Extended (growth-voice-catchup) with ``list_recordings`` + an
    ``analysis``/``created_at`` carrying ``get_recording`` and a real
    ``overwrite_analysis`` — enough to exercise Part A's relabel-and-persist
    end-to-end through GET /growth and GET /recordings/{id}, without pulling
    in the full FakeGrowthStore from test_growth.py."""

    def __init__(self):
        self._recordings: dict[str, dict] = {}  # (uid, rid) → recording dict
        self._voiceprints: dict[str, dict] = {}  # uid → profile

    def add_recording(
        self, uid, rid, turns, audio=b"AUDIO", analysis=None,
        created_at="2026-08-01T00:00:00+00:00",
    ):
        self._recordings[(uid, rid)] = {
            "turns": turns, "audio": audio, "analysis": analysis,
            "created_at": created_at,
        }

    async def get_recording(self, uid, recording_id):
        r = self._recordings.get((uid, recording_id))
        if r is None:
            return None
        return {
            "id": recording_id,
            "created_at": r["created_at"],
            "filename": "rec.m4a",
            "title": None,
            "media_type": "audio",
            "duration_seconds": 60.0,
            "turns": r["turns"],
            "analysis": r["analysis"],
        }

    async def list_recordings(self, uid):
        out = [
            {
                "id": rid,
                "created_at": r["created_at"],
                "has_analysis": r["analysis"] is not None,
            }
            for (u, rid), r in self._recordings.items()
            if u == uid
        ]
        out.sort(key=lambda m: m["created_at"], reverse=True)
        return out

    async def get_audio_bytes(self, uid, recording_id):
        r = self._recordings.get((uid, recording_id))
        return None if r is None else r["audio"]

    async def read_voiceprint(self, uid):
        return self._voiceprints.get(uid)

    async def write_voiceprint(self, uid, profile):
        self._voiceprints[uid] = profile

    async def delete_voiceprint(self, uid):
        return self._voiceprints.pop(uid, None) is not None

    async def overwrite_analysis(self, uid, recording_id, *, turns, analysis, reanalyzed_at):
        r = self._recordings.get((uid, recording_id))
        if r is None:
            return None
        r["turns"] = turns
        r["analysis"] = analysis
        r["reanalyzed_at"] = reanalyzed_at
        return {"id": recording_id, "reanalyzed_at": reanalyzed_at}


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


def _rid():
    return str(uuid.uuid4())


def _enroll_ready(monkeypatch, embedding):
    """Wire the router's torch-touching calls to deterministic doubles."""
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    # decode_to_pcm is imported into the router's namespace.
    import routers.voice as voice_router
    monkeypatch.setattr(
        voice_router, "decode_to_pcm",
        lambda data, name: (np.zeros(16000 * 5, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(
        speaker_id, "embed_speaker", lambda *a, **k: embedding,
    )


TURNS = [
    {"speaker": "Speaker A", "text": "hi", "start_time": 0.0, "end_time": 3.0},
    {"speaker": "Speaker B", "text": "yo", "start_time": 3.0, "end_time": 6.0},
]


async def test_profile_reports_availability(client, voice_store, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: False)
    res = await client.get("/voice/profile", headers={"X-Test-Uid": "u1"})
    assert res.status_code == 200
    body = res.json()
    assert body["available"] is False
    assert body["storage_enabled"] is True
    assert body["enrolled"] is False
    assert body["enroll_count"] == 0


async def test_enroll_then_match_labels_you(client, voice_store, monkeypatch):
    rid = _rid()
    voice_store.add_recording("u1", rid, TURNS)
    e_you = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    _enroll_ready(monkeypatch, e_you)

    res = await client.post(
        "/voice/enroll", json={"recording_id": rid, "speaker": "Speaker A"},
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["enrolled"] is True
    assert body["enroll_count"] == 1
    assert "not your audio" in body["stored"]

    # Profile now reports enrolled.
    prof = await client.get("/voice/profile", headers={"X-Test-Uid": "u1"})
    assert prof.json()["enrolled"] is True
    assert prof.json()["enroll_count"] == 1

    # The stored voiceprint, matched against a fresh analysis, labels A "You".
    monkeypatch.setattr(
        speaker_id, "embed_speaker",
        _fake_embed_by_speaker({
            "Speaker A": e_you,
            "Speaker B": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }),
    )
    stored = voice_store._voiceprints["u1"]
    report = speaker_id.identify_speakers(
        np.zeros(10, dtype=np.float32), 16000, TURNS,
        np.array(stored["embedding"], dtype=np.float32), threshold=0.5,
    )
    assert report["matched_speaker"] == "Speaker A"


async def test_second_enrollment_increments_count(client, voice_store, monkeypatch):
    rid = _rid()
    voice_store.add_recording("u1", rid, TURNS)
    _enroll_ready(monkeypatch, np.array([1.0, 0.0], dtype=np.float32))
    for expected in (1, 2):
        res = await client.post(
            "/voice/enroll", json={"recording_id": rid, "speaker": "Speaker A"},
            headers={"X-Test-Uid": "u1"},
        )
        assert res.status_code == 200
        assert res.json()["enroll_count"] == expected


async def test_enroll_unavailable_dep_503(client, voice_store, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: False)
    rid = _rid()
    voice_store.add_recording("u1", rid, TURNS)
    res = await client.post(
        "/voice/enroll", json={"recording_id": rid, "speaker": "Speaker A"},
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 503
    assert "not available" in res.json()["detail"]


async def test_enroll_storage_disabled_503(client, monkeypatch):
    # No app.state.recordings_store → storage disabled.
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    res = await client.post(
        "/voice/enroll", json={"recording_id": _rid(), "speaker": "Speaker A"},
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 503


async def test_enroll_missing_recording_404(client, voice_store, monkeypatch):
    _enroll_ready(monkeypatch, np.array([1.0, 0.0], dtype=np.float32))
    res = await client.post(
        "/voice/enroll", json={"recording_id": _rid(), "speaker": "Speaker A"},
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 404


async def test_enroll_speaker_not_in_recording_422(client, voice_store, monkeypatch):
    rid = _rid()
    voice_store.add_recording("u1", rid, TURNS)
    _enroll_ready(monkeypatch, np.array([1.0, 0.0], dtype=np.float32))
    res = await client.post(
        "/voice/enroll", json={"recording_id": rid, "speaker": "Speaker Z"},
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 422


async def test_enroll_too_little_audio_422(client, voice_store, monkeypatch):
    rid = _rid()
    voice_store.add_recording("u1", rid, TURNS)
    # embed_speaker returns None → not enough of that speaker's voice.
    _enroll_ready(monkeypatch, None)
    res = await client.post(
        "/voice/enroll", json={"recording_id": rid, "speaker": "Speaker A"},
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 422
    assert "not enough" in res.json()["detail"]


async def test_forget_actually_deletes(client, voice_store, monkeypatch):
    rid = _rid()
    voice_store.add_recording("u1", rid, TURNS)
    _enroll_ready(monkeypatch, np.array([1.0, 0.0], dtype=np.float32))
    await client.post(
        "/voice/enroll", json={"recording_id": rid, "speaker": "Speaker A"},
        headers={"X-Test-Uid": "u1"},
    )
    assert "u1" in voice_store._voiceprints

    # First delete removes it for real.
    d1 = await client.request(
        "DELETE", "/voice/voiceprint", headers={"X-Test-Uid": "u1"},
    )
    assert d1.status_code == 200
    assert d1.json()["deleted"] is True
    assert "u1" not in voice_store._voiceprints

    # Idempotent: a second delete reports nothing was there.
    d2 = await client.request(
        "DELETE", "/voice/voiceprint", headers={"X-Test-Uid": "u1"},
    )
    assert d2.json()["deleted"] is False


async def test_enroll_is_uid_scoped(client, voice_store, monkeypatch):
    # A recording owned by u1 is invisible to u2 (reads as 404, never leaks).
    rid = _rid()
    voice_store.add_recording("u1", rid, TURNS)
    _enroll_ready(monkeypatch, np.array([1.0, 0.0], dtype=np.float32))
    res = await client.post(
        "/voice/enroll", json={"recording_id": rid, "speaker": "Speaker A"},
        headers={"X-Test-Uid": "u2"},
    )
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Part A (growth-voice-catchup) — "This is me" relabels the TAPPED recording
# ---------------------------------------------------------------------------

def _report_card(score):
    return {"score": score, "headline": "h", "did_well": "d", "work_on": "w"}


def _unidentified_analysis():
    return {
        "speaker_labels": {
            "Speaker A": {"display_label": "Speaker A", "label_source": "generic"},
            "Speaker B": {"display_label": "Speaker B", "label_source": "generic"},
        },
        "report_cards": {
            "Speaker A": _report_card(80),
            "Speaker B": _report_card(40),
        },
    }


async def test_enroll_relabels_this_recording_for_growth(client, voice_store, monkeypatch):
    # The owner's exact bug: a recording ALREADY has a stored analysis (two
    # generic speakers) when "This is me" is tapped on Speaker A. Growth must
    # see this recording as identified immediately — no reanalysis required.
    rid = _rid()
    voice_store.add_recording("u1", rid, TURNS, analysis=_unidentified_analysis())
    _enroll_ready(monkeypatch, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    pre = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    assert pre.json()["identified_recordings"] == 0

    res = await client.post(
        "/voice/enroll", json={"recording_id": rid, "speaker": "Speaker A"},
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 200, res.text

    post = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    body = post.json()
    assert body["identified_recordings"] == 1
    assert body["points"][0]["recording_id"] == rid
    assert body["points"][0]["my_score"] == 80  # Speaker A's card, not B's

    # The exact shape _identify_enrolled_speakers'/the label ladder's "enrolled"
    # rung normally produces: display_label "You", label_source "enrolled" —
    # nothing thinner, nothing extra (SpeakerLabelOut has only these 2 fields).
    detail = await client.get(f"/recordings/{rid}", headers={"X-Test-Uid": "u1"})
    assert detail.status_code == 200, detail.text
    labels = detail.json()["speaker_labels"]
    assert labels["Speaker A"] == {
        "display_label": "You", "label_source": "enrolled",
    }
    assert labels["Speaker B"]["label_source"] == "generic"  # untouched


async def test_enroll_with_no_analysis_yet_does_not_error(client, voice_store, monkeypatch):
    # A stored-but-never-analyzed recording: nothing to relabel — the endpoint
    # must not error just because there's no analysis.speaker_labels to touch.
    rid = _rid()
    voice_store.add_recording("u1", rid, TURNS, analysis=None)
    _enroll_ready(monkeypatch, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    res = await client.post(
        "/voice/enroll", json={"recording_id": rid, "speaker": "Speaker A"},
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 200, res.text
    assert res.json()["enrolled"] is True


async def test_enroll_persist_failure_is_swallowed(client, voice_store, monkeypatch):
    # The relabel-and-persist step is best-effort: a storage hiccup here must
    # NEVER fail the enrollment itself — the voiceprint write is what matters.
    rid = _rid()
    voice_store.add_recording("u1", rid, TURNS, analysis=_unidentified_analysis())
    _enroll_ready(monkeypatch, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    async def _boom(*a, **k):
        raise RuntimeError("gcs hiccup")

    monkeypatch.setattr(voice_store, "overwrite_analysis", _boom)

    res = await client.post(
        "/voice/enroll", json={"recording_id": rid, "speaker": "Speaker A"},
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 200, res.text
    assert res.json()["enrolled"] is True
    assert "u1" in voice_store._voiceprints  # the part that matters most


# ---------------------------------------------------------------------------
# Critical regression: a correction tap must never leave TWO "enrolled"
# speakers (main._growth_point requires EXACTLY one — two silently drops the
# recording out of Growth entirely).
# ---------------------------------------------------------------------------

def _identified_analysis(me="Speaker B"):
    """Analysis where ``me`` is ALREADY "enrolled" — simulates a prior (right
    or wrong) auto-match / "This is me" tap, for testing the demotion fix."""
    a = _unidentified_analysis()
    a["speaker_labels"][me] = {"display_label": "You", "label_source": "enrolled"}
    return a


async def test_enroll_correcting_a_wrong_match_demotes_the_old_one(
    client, voice_store, monkeypatch,
):
    # The exact critical-bug scenario: the original analysis auto-matched the
    # WRONG speaker (B) as "You". The user opens the recording and taps "This
    # is me" on the RIGHT speaker (A) to correct it. Before the fix this left
    # BOTH speakers "enrolled" — the recording would silently vanish from
    # Growth (strictly worse than before this feature existed: the wrong
    # single point at least stayed visible).
    rid = _rid()
    voice_store.add_recording(
        "u1", rid, TURNS, analysis=_identified_analysis(me="Speaker B"),
    )
    _enroll_ready(monkeypatch, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    res = await client.post(
        "/voice/enroll", json={"recording_id": rid, "speaker": "Speaker A"},
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 200, res.text

    growth = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    body = growth.json()
    assert body["identified_recordings"] == 1
    assert len(body["points"]) == 1
    assert body["points"][0]["recording_id"] == rid

    detail = await client.get(f"/recordings/{rid}", headers={"X-Test-Uid": "u1"})
    labels = detail.json()["speaker_labels"]
    assert labels["Speaker A"] == {
        "display_label": "You", "label_source": "enrolled",
    }
    # Speaker B — the old (wrong) match — is demoted, never left "enrolled".
    assert labels["Speaker B"]["label_source"] != "enrolled"


async def test_enroll_recomputes_stale_episodes_and_speaker_identity(
    client, voice_store, monkeypatch,
):
    # The stored episodes + speaker_identity were both stale: matched_speaker
    # said "Speaker B" (wrong). episodes_from_analysis PREFERS
    # speaker_identity.matched_speaker over speaker_labels, so a stale
    # identity can keep showing the WRONG speaker as "You" in the day
    # timeline even after speaker_labels is correctly relabeled.
    import episodes as episodes_mod

    rid = _rid()
    analysis = _unidentified_analysis()
    analysis["per_turn"] = [{"heat": 10}, {"heat": 20}]
    analysis["title"] = "Kitchen talk"
    analysis["speaker_identity"] = {
        "matched_speaker": "Speaker B",  # wrong — the stale-bug scenario
        "match_threshold": 0.5,
        "model": "test-model",
        "speakers": {
            "Speaker A": {"score": 0.1, "is_you": False},
            "Speaker B": {"score": 0.6, "is_you": True},
        },
    }
    analysis["episodes"] = episodes_mod.segment_episodes(
        TURNS,
        per_turn=analysis["per_turn"],
        speaker_labels=analysis["speaker_labels"],
        speaker_identity=analysis["speaker_identity"],
        title=analysis["title"],
    )
    # Confirm the stale fixture really does show the WRONG speaker as "You".
    assert analysis["episodes"][0]["participants"] == ["Speaker A", "You"]

    voice_store.add_recording("u1", rid, TURNS, analysis=analysis)
    _enroll_ready(monkeypatch, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    res = await client.post(
        "/voice/enroll", json={"recording_id": rid, "speaker": "Speaker A"},
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 200, res.text

    detail = await client.get(f"/recordings/{rid}", headers={"X-Test-Uid": "u1"})
    body = detail.json()
    assert body["analysis"]["speaker_identity"]["matched_speaker"] == "Speaker A"
    assert body["analysis"]["episodes"][0]["participants"] == ["You", "Speaker B"]
