"""People labeling — name who was talking ONCE, recognize them everywhere.

Covers, torch-free (synthetic embeddings, a fake store):

* ``speaker_id.enrollment_conflict`` — the "sounds like someone else" guard
  that keeps a mis-tapped speaker from poisoning another person's print;
* ``POST /voice/people/{id}/enroll-from-recording`` — learn a voice from a
  stored recording's diarized speaker: creates the person, appends a sample
  with recording/speaker/seconds provenance, relabels the recording; and the
  honest 422s (too little speech, sounds like someone else, live session
  with no audio, nameless new person) + 404/503;
* ``PATCH /voice/people/{id}`` — rename;
* the ``manual-person`` ladder rung — ``PATCH …/speaker-labels`` with
  ``people``: the effective label carries ``person_id``, "self" makes the
  recording count for /growth, unknown people are refused, the person's
  display name fills in a missing manual name, clearing works;
* episodes' participants honor a manual label over a stale enrolled match;
* ``live_sessions.manual_overlay`` — /growth people rows and the therapist
  GET /sessions rows follow the person id / name the user chose.

Plus ONE live-gated test (torch + speechbrain + the scene pack): enroll
"self" from the couple scene's Speaker A THROUGH THE ENDPOINT, then check
``identify_speakers_multi`` finds that person — and only that person — in
the family3 scene (Foundation D measured 0.89–0.94 cross-scene).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import audio_ingest
import episodes
import live_sessions
import main
import speaker_id
from main import app, init_db

pytestmark = pytest.mark.anyio

E_SELF = np.array([1.0, 0.0, 0.0], dtype=np.float32)
E_MOM = np.array([0.0, 1.0, 0.0], dtype=np.float32)
E_DAD = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _unit(*xs):
    return speaker_id.l2_normalize(np.array(xs, dtype=np.float32))


def _doc(embedding, person_id="self", display_name=None, **extra):
    d = {"version": 2, "person_id": person_id, "embedding": list(map(float, embedding)),
         "dim": len(embedding), "enroll_count": 1,
         "samples": [{"id": f"s-{person_id}", "embedding": list(map(float, embedding))}],
         **extra}
    if display_name is not None:
        d["display_name"] = display_name
    return d


# ---------------------------------------------------------------------------
# enrollment_conflict — pure
# ---------------------------------------------------------------------------

def test_conflict_none_when_nobody_else_is_close():
    profiles = [_doc(E_SELF), _doc(E_MOM, "mom", "Mom")]
    assert speaker_id.enrollment_conflict(E_DAD, profiles, "dad") is None
    # Closest to the target person itself → fine, even with others enrolled.
    assert speaker_id.enrollment_conflict(E_MOM, profiles, "mom") is None


def test_conflict_when_voice_matches_someone_else_and_person_is_new():
    profiles = [_doc(E_SELF), _doc(E_MOM, "mom", "Mom")]
    near_mom = _unit(0.1, 0.99, 0.0)
    c = speaker_id.enrollment_conflict(near_mom, profiles, "dad")
    assert c is not None
    assert c["person_id"] == "mom" and c["display_name"] == "Mom"
    assert c["score"] >= speaker_id.MATCH_THRESHOLD and c["own_score"] is None


def test_conflict_respects_margin_toward_the_target_person():
    # A voice that matches BOTH mom (0.70) and dad (0.71) — dad is not
    # clearer by the margin → refused. Dad at 0.85 vs mom 0.70 → accepted.
    profiles = [_doc(E_MOM, "mom", "Mom"), _doc(E_DAD, "dad", "Dad")]
    ambiguous = _unit(0.0, 0.70, 0.71)
    c = speaker_id.enrollment_conflict(ambiguous, profiles, "dad")
    assert c is not None and c["person_id"] == "mom"
    assert c["own_score"] == pytest.approx(0.71, abs=0.01)
    clearer = _unit(0.0, 0.70, 0.85)
    # cosine(clearer, dad) - cosine(clearer, mom) = (0.85-0.70)/|v| ≈ 0.136 ≥ 0.10
    assert speaker_id.enrollment_conflict(clearer, profiles, "dad") is None


def test_conflict_self_resolves_to_you_and_below_threshold_is_safe():
    profiles = [_doc(E_SELF)]
    c = speaker_id.enrollment_conflict(_unit(0.9, 0.44, 0.0), profiles, "mom")
    assert c is not None and c["display_name"] == "You"
    assert speaker_id.enrollment_conflict(_unit(0.5, 0.87, 0.0), profiles, "mom") is None
    # Documents without a usable vector contribute nothing.
    assert speaker_id.enrollment_conflict(E_SELF, [{"person_id": "x"}], "mom") is None


def test_new_profile_records_seconds_provenance():
    p = speaker_id.new_profile(
        E_MOM, None, recording_id="r", speaker="Speaker B", now_iso="t",
        person_id="mom", display_name="Mom", seconds=12.34,
    )
    assert p["samples"][0]["seconds"] == 12.3
    p2 = speaker_id.new_profile(E_MOM, None, recording_id="r", speaker="B", now_iso="t")
    assert "seconds" not in p2["samples"][0]


# ---------------------------------------------------------------------------
# The manual-person rung — pure
# ---------------------------------------------------------------------------

BASE = {
    "Speaker A": {"display_label": "Speaker A", "label_source": "generic"},
    "Speaker B": {"display_label": "You", "label_source": "enrolled"},
}


def test_effective_labels_manual_person_carries_person_id():
    eff = main._effective_speaker_labels(
        BASE, {"Speaker A": "Mom", "Speaker B": "Dad"}, {"Speaker A", "Speaker B"},
        {"Speaker A": "mom"},
    )
    assert eff["Speaker A"] == {
        "display_label": "Mom", "label_source": "manual-person", "person_id": "mom",
    }
    assert eff["Speaker B"] == {"display_label": "Dad", "label_source": "manual"}
    # A person id with no manual NAME is ignored (never an invented label).
    eff2 = main._effective_speaker_labels(BASE, {}, {"Speaker A"}, {"Speaker A": "mom"})
    assert eff2["Speaker A"]["label_source"] == "generic"


def test_is_me_label_enrolled_you_or_manual_self():
    assert main._is_me_label({"display_label": "You", "label_source": "enrolled"})
    assert not main._is_me_label({"display_label": "Alex", "label_source": "enrolled"})
    assert main._is_me_label(
        {"display_label": "You", "label_source": "manual-person", "person_id": "self"},
    )
    assert not main._is_me_label(
        {"display_label": "Mom", "label_source": "manual-person", "person_id": "mom"},
    )
    assert not main._is_me_label({"display_label": "You", "label_source": "manual"})


def test_growth_point_counts_manual_self_as_me():
    rec = {
        "id": "r1", "created_at": "2026-08-01T00:00:00+00:00",
        "turns": [{"speaker": "Speaker A", "text": "hi"}, {"speaker": "Speaker B", "text": "yo"}],
        "analysis": {"speaker_labels": {
            "Speaker A": {"display_label": "Speaker A", "label_source": "generic"},
            "Speaker B": {"display_label": "Speaker B", "label_source": "generic"},
        }, "report_cards": {"Speaker A": {"score": 77}}},
        "manual_speaker_labels": {"Speaker A": "You", "Speaker B": "Mom"},
        "manual_speaker_people": {"Speaker A": "self", "Speaker B": "mom"},
    }
    point = main._growth_point(rec)
    assert point is not None and point.my_score == 77
    assert point.partner_names == ["Mom"]


def test_episodes_manual_label_beats_stale_enrolled_match():
    turns = [
        {"speaker": "Speaker A", "text": "hello there", "start_time": 0.0, "end_time": 2.0},
        {"speaker": "Speaker B", "text": "hi", "start_time": 2.0, "end_time": 4.0},
    ]
    identity = {"matched": {"Speaker A": "self"}, "people": {"self": {"display_name": "You", "is_self": True}}}
    labels = {
        "Speaker A": {"display_label": "Mom", "label_source": "manual-person", "person_id": "mom"},
        "Speaker B": {"display_label": "Speaker B", "label_source": "generic"},
    }
    [ep] = episodes.segment_episodes(turns, speaker_labels=labels, speaker_identity=identity)
    assert ep["participants"] == ["Mom", "Speaker B"]
    # Without a manual label the enrolled match still wins over a lower rung.
    labels["Speaker A"] = {"display_label": "Sam", "label_source": "name"}
    [ep2] = episodes.segment_episodes(turns, speaker_labels=labels, speaker_identity=identity)
    assert ep2["participants"] == ["You", "Speaker B"]


def test_manual_overlay_and_dashboard_rows_follow_the_person():
    rec = {
        "id": "r", "created_at": "2026-08-01T00:00:00+00:00", "media_type": "none",
        "turns": [{"speaker": "Speaker A", "text": "hi"}, {"speaker": "Speaker B", "text": "yo"}],
        "analysis": {"speaker_labels": {
            "Speaker A": {"display_label": "You", "label_source": "enrolled"},
            "Speaker B": {"display_label": "Speaker B", "label_source": "generic"},
        }, "per_turn": []},
        "manual_speaker_labels": {"Speaker B": "Mom", "Ghost": "Nobody"},
        "manual_speaker_people": {"Speaker B": "mom"},
    }
    overlay = live_sessions.manual_overlay(rec)
    assert overlay == {"Speaker B": {
        "display_label": "Mom", "label_source": "manual-person", "person_id": "mom",
    }}
    row = live_sessions.dashboard_session(rec, patient="You", shared=False)
    assert [t["speaker"] for t in row["turns"]] == ["You", "Mom"]
    assert row["turns"][1]["speakerId"] == "Speaker B"
    assert row["turns"][1]["personId"] == "mom"
    assert row["turns"][1]["labelSource"] == "manual-person"
    assert row["speakers"] == [
        {"id": "Speaker A", "display": "You", "labelSource": "enrolled", "personId": None},
        {"id": "Speaker B", "display": "Mom", "labelSource": "manual-person", "personId": "mom"},
    ]
    assert row["hasAudio"] is False


def test_growth_extras_people_rows_take_manual_person():
    rec = {
        "id": "r", "source": {"type": "live"}, "mode": "earpiece",
        "turns": [{"speaker": "Speaker A", "text": "hi"}, {"speaker": "Speaker B", "text": "yo"}],
        "analysis": {"live": {"tone_summary": {
            "self": {"scored_turns": 1, "labels": {"warm": 1}, "mean": {}, "escalation_count": 0},
            "people": [{"speaker": "Speaker B", "person_id": None, "display_name": None,
                        "scored_turns": 1, "labels": {"warm": 1}, "escalation_count": 0}],
        }}},
        "manual_speaker_labels": {"Speaker B": "Mom"},
        "manual_speaker_people": {"Speaker B": "mom"},
    }
    [p] = live_sessions.growth_extras(rec)["self_tone"]["people"]
    assert p["person_id"] == "mom" and p["display_name"] == "Mom"
    [row] = live_sessions.aggregate_people([rec, {**rec, "id": "r2"}])
    assert row["person_id"] == "mom" and row["sessions"] == 2


# ---------------------------------------------------------------------------
# Router — fake store (recordings + manual labels + voiceprints)
# ---------------------------------------------------------------------------

class FakeStore:
    def __init__(self):
        self._recordings: dict[tuple, dict] = {}
        self._people: dict[tuple, dict] = {}

    def add_recording(self, uid, rid, turns, *, audio=b"AUDIO", analysis=None,
                      media_type="audio", manual=None, people=None):
        self._recordings[(uid, rid)] = {
            "turns": turns, "audio": audio, "analysis": analysis,
            "media_type": media_type, "manual": dict(manual or {}),
            "people": dict(people or {}), "created_at": "2026-08-01T00:00:00+00:00",
            "source": {"type": "live" if media_type == "none" else "upload"},
        }

    def _rec(self, uid, rid):
        r = self._recordings.get((uid, rid))
        if r is None:
            return None
        out = {"id": rid, "created_at": r["created_at"], "filename": "rec.m4a",
               "title": "A talk", "media_type": r["media_type"], "duration_seconds": 60.0,
               "source": r["source"], "turns": r["turns"], "analysis": r["analysis"]}
        if r["manual"]:
            out["manual_speaker_labels"] = dict(r["manual"])
        if r["people"]:
            out["manual_speaker_people"] = dict(r["people"])
        return out

    async def get_recording(self, uid, rid):
        return self._rec(uid, rid)

    async def list_recordings(self, uid):
        return [{"id": rid, "created_at": r["created_at"], "has_analysis": r["analysis"] is not None}
                for (u, rid), r in self._recordings.items() if u == uid]

    async def list_shared_with(self, uid):
        return []

    async def find_share(self, uid, rid):
        return None

    async def get_audio_bytes(self, uid, rid):
        r = self._recordings.get((uid, rid))
        return None if r is None else r["audio"]

    async def overwrite_analysis(self, uid, rid, *, turns, analysis, reanalyzed_at):
        r = self._recordings.get((uid, rid))
        if r is None:
            return None
        r["turns"], r["analysis"] = turns, analysis
        return {"id": rid}

    async def update_manual_speaker_labels(self, uid, rid, labels):
        r = self._recordings.get((uid, rid))
        if r is None:
            return None
        r["manual"] = dict(labels)
        return self._rec(uid, rid)

    async def update_manual_speaker_people(self, uid, rid, people):
        r = self._recordings.get((uid, rid))
        if r is None:
            return None
        r["people"] = dict(people)
        return self._rec(uid, rid)

    async def read_voiceprint(self, uid, person_id=None):
        pid = person_id or speaker_id.SELF_PERSON_ID
        return speaker_id.as_person(self._people.get((uid, pid)), person_id=pid)

    async def list_voiceprints(self, uid):
        return [speaker_id.as_person(p, person_id=pid)
                for (u, pid), p in self._people.items() if u == uid]

    async def write_voiceprint(self, uid, profile):
        doc = speaker_id.as_person(profile)
        self._people[(uid, doc["person_id"])] = doc

    async def delete_voiceprint(self, uid, person_id=None):
        return self._people.pop((uid, person_id or speaker_id.SELF_PERSON_ID), None) is not None


@pytest.fixture
async def client():
    await init_db()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
def store():
    fake = FakeStore()
    app.state.recordings_store = fake
    yield fake
    del app.state.recordings_store


H = {"X-Test-Uid": "u1"}
SR = 16000
# Speaker A speaks 0–5 s, Speaker B 5–10 s, Speaker C only 11–12 s (1 s — too
# little to enroll, enough to match).
TURNS = [
    {"speaker": "Speaker A", "text": "…", "start_time": 0.0, "end_time": 5.0},
    {"speaker": "Speaker B", "text": "…", "start_time": 5.0, "end_time": 10.0},
    {"speaker": "Speaker C", "text": "…", "start_time": 11.0, "end_time": 12.0},
]


def _generic_analysis():
    return {"speaker_labels": {
        sp: {"display_label": sp, "label_source": "generic"}
        for sp in ("Speaker A", "Speaker B", "Speaker C")
    }, "report_cards": {"Speaker A": {"score": 80}, "Speaker B": {"score": 40}}}


def _voice_ready(monkeypatch, by_offset):
    """Fake decode → a 20 s ramp whose sample VALUE is its time in seconds,
    so the pooled slice for a speaker starts at that speaker's first turn
    time; the fake embedder maps that start time to a synthetic vector.
    This exercises the REAL pool_speaker_pcm (seconds are measured from the
    turns) with no torch."""
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    import routers.voice as voice_router
    ramp = (np.arange(SR * 20, dtype=np.float32) / SR)
    monkeypatch.setattr(voice_router, "decode_to_pcm_16k", lambda data, name: (ramp, SR))

    def _embed(pcm, sr):
        start = float(pcm[0])
        for lo, vec in by_offset.items():
            if abs(start - lo) < 0.5:
                return vec
        raise AssertionError(f"unexpected pooled start {start}")
    monkeypatch.setattr(speaker_id, "embed_pcm", _embed)


async def test_enroll_from_recording_creates_person_with_provenance_and_relabels(
    client, store, monkeypatch,
):
    rid = str(uuid.uuid4())
    store.add_recording("u1", rid, TURNS, analysis=_generic_analysis())
    _voice_ready(monkeypatch, {0.0: E_SELF, 5.0: E_MOM})

    resp = await client.post(
        "/voice/people/mom/enroll-from-recording",
        json={"recording_id": rid, "speaker_label": "Speaker B", "display_name": "Mom"},
        headers=H,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["created"] is True and body["person_id"] == "mom"
    assert body["display_name"] == "Mom" and body["is_self"] is False
    assert body["enroll_count"] == 1 and body["seconds"] == 5.0
    assert body["speaker_labels"]["Speaker B"] == {"display_label": "Mom", "label_source": "enrolled"}
    assert "embedding" not in json.dumps(body)

    # Provenance on the stored sample: recording + speaker + seconds.
    people = (await client.get("/voice/people", headers=H)).json()["people"]
    [mom] = people
    [sample] = mom["samples"]
    assert sample["recording_id"] == rid and sample["speaker"] == "Speaker B"
    assert sample["seconds"] == 5.0

    # The recording itself now shows Mom on the enrolled rung.
    detail = (await client.get(f"/recordings/{rid}", headers=H)).json()
    assert detail["speaker_labels"]["Speaker B"]["display_label"] == "Mom"
    assert detail["analysis"]["speaker_identity"]["matched"] == {"Speaker B": "mom"}

    # A second sample for the same person appends (no name needed).
    rid2 = str(uuid.uuid4())
    store.add_recording("u1", rid2, TURNS, analysis=_generic_analysis())
    again = await client.post(
        "/voice/people/mom/enroll-from-recording",
        json={"recording_id": rid2, "speaker_label": "Speaker B"}, headers=H,
    )
    assert again.status_code == 200 and again.json()["created"] is False
    assert again.json()["enroll_count"] == 2


async def test_enroll_from_recording_too_little_speech_422(client, store, monkeypatch):
    rid = str(uuid.uuid4())
    store.add_recording("u1", rid, TURNS, analysis=_generic_analysis())
    _voice_ready(monkeypatch, {11.0: E_DAD})
    resp = await client.post(
        "/voice/people/dad/enroll-from-recording",
        json={"recording_id": rid, "speaker_label": "Speaker C", "display_name": "Dad"},
        headers=H,
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"].startswith("[too-little-speech]")
    assert "1.0s" in resp.json()["detail"] and "3s" in resp.json()["detail"]
    assert await store.read_voiceprint("u1", "dad") is None  # nothing stored


async def test_enroll_from_recording_sounds_like_someone_else_422(client, store, monkeypatch):
    rid = str(uuid.uuid4())
    store.add_recording("u1", rid, TURNS, analysis=_generic_analysis())
    await store.write_voiceprint("u1", _doc(E_MOM, "mom", "Mom"))
    # Speaker B IS Mom's voice, but the user tapped it as "Dad".
    _voice_ready(monkeypatch, {5.0: _unit(0.05, 0.99, 0.1)})
    resp = await client.post(
        "/voice/people/dad/enroll-from-recording",
        json={"recording_id": rid, "speaker_label": "Speaker B", "display_name": "Dad"},
        headers=H,
    )
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("[sounds-like-someone-else]")
    assert "sounds like Mom" in detail and "0.65" in detail
    assert await store.read_voiceprint("u1", "dad") is None
    # Mom's print is untouched (still one sample).
    assert (await store.read_voiceprint("u1", "mom"))["enroll_count"] == 1
    # The recording was NOT relabeled.
    detail_rec = (await client.get(f"/recordings/{rid}", headers=H)).json()
    assert detail_rec["speaker_labels"]["Speaker B"]["label_source"] == "generic"


async def test_enroll_from_recording_live_session_has_no_audio_422(client, store, monkeypatch):
    rid = str(uuid.uuid4())
    store.add_recording("u1", rid, TURNS, analysis=_generic_analysis(), media_type="none", audio=None)
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    resp = await client.post(
        "/voice/people/mom/enroll-from-recording",
        json={"recording_id": rid, "speaker_label": "Speaker B", "display_name": "Mom"},
        headers=H,
    )
    assert resp.status_code == 422
    assert resp.json()["detail"].startswith("[no-audio]")
    assert "live session" in resp.json()["detail"]
    # An upload whose derivative vanished is the same honest reason.
    rid2 = str(uuid.uuid4())
    store.add_recording("u1", rid2, TURNS, analysis=_generic_analysis(), audio=None)
    resp2 = await client.post(
        "/voice/people/mom/enroll-from-recording",
        json={"recording_id": rid2, "speaker_label": "Speaker B", "display_name": "Mom"},
        headers=H,
    )
    assert resp2.status_code == 422 and resp2.json()["detail"].startswith("[no-audio]")


async def test_enroll_from_recording_gates(client, store, monkeypatch):
    rid = str(uuid.uuid4())
    store.add_recording("u1", rid, TURNS, analysis=_generic_analysis())
    # Deps absent → 503.
    monkeypatch.setattr(speaker_id, "is_available", lambda: False)
    r = await client.post("/voice/people/mom/enroll-from-recording",
                          json={"recording_id": rid, "speaker_label": "Speaker B", "display_name": "Mom"},
                          headers=H)
    assert r.status_code == 503
    _voice_ready(monkeypatch, {0.0: E_SELF, 5.0: E_MOM})
    # Foreign / missing recording → 404.
    r = await client.post("/voice/people/mom/enroll-from-recording",
                          json={"recording_id": str(uuid.uuid4()), "speaker_label": "Speaker B",
                                "display_name": "Mom"}, headers=H)
    assert r.status_code == 404
    r = await client.post("/voice/people/mom/enroll-from-recording",
                          json={"recording_id": rid, "speaker_label": "Speaker B", "display_name": "Mom"},
                          headers={"X-Test-Uid": "u2"})
    assert r.status_code == 404
    # Speaker not in the recording → 422.
    r = await client.post("/voice/people/mom/enroll-from-recording",
                          json={"recording_id": rid, "speaker_label": "Speaker Z", "display_name": "Mom"},
                          headers=H)
    assert r.status_code == 422 and "not in this recording" in r.json()["detail"]
    # New person with no name → 422 (nothing stored).
    r = await client.post("/voice/people/mom/enroll-from-recording",
                          json={"recording_id": rid, "speaker_label": "Speaker B"}, headers=H)
    assert r.status_code == 422 and "display_name" in r.json()["detail"]
    assert await store.read_voiceprint("u1", "mom") is None
    # Bad slug → 422 at the path.
    r = await client.post("/voice/people/Not%20A%20Slug/enroll-from-recording",
                          json={"recording_id": rid, "speaker_label": "Speaker B", "display_name": "X"},
                          headers=H)
    assert r.status_code == 422
    # Self works too and relabels "You".
    r = await client.post("/voice/people/self/enroll-from-recording",
                          json={"recording_id": rid, "speaker_label": "Speaker A"}, headers=H)
    assert r.status_code == 200 and r.json()["is_self"] is True
    assert r.json()["speaker_labels"]["Speaker A"]["display_label"] == "You"


async def test_rename_person(client, store, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    await store.write_voiceprint("u1", _doc(E_MOM, "mom", "Mom"))
    r = await client.patch("/voice/people/mom", json={"display_name": "  Mum "}, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["display_name"] == "Mum" and r.json()["person_id"] == "mom"
    assert (await store.read_voiceprint("u1", "mom"))["display_name"] == "Mum"
    assert (await client.patch("/voice/people/self", json={"display_name": "Me"}, headers=H)).status_code == 422
    assert (await client.patch("/voice/people/dad", json={"display_name": "Dad"}, headers=H)).status_code == 404
    assert (await client.patch("/voice/people/mom", json={"display_name": "Mum"},
                               headers={"X-Test-Uid": "u2"})).status_code == 404


# ---------------------------------------------------------------------------
# PATCH …/speaker-labels with people → the manual-person rung
# ---------------------------------------------------------------------------

async def test_patch_people_labels_speaker_as_enrolled_person(client, store):
    rid = str(uuid.uuid4())
    store.add_recording("u1", rid, TURNS, analysis=_generic_analysis())
    await store.write_voiceprint("u1", _doc(E_MOM, "mom", "Mom"))

    # Name given explicitly + person → manual-person with person_id.
    r = await client.patch(f"/recordings/{rid}/speaker-labels",
                           json={"labels": {"Speaker B": "Mom"}, "people": {"Speaker B": "mom"}},
                           headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["manual_speaker_people"] == {"Speaker B": "mom"}
    assert body["speaker_labels"]["Speaker B"] == {
        "display_label": "Mom", "label_source": "manual-person", "person_id": "mom",
    }
    # Person alone → the person's display name fills the manual name ("You" for self).
    r = await client.patch(f"/recordings/{rid}/speaker-labels",
                           json={"people": {"Speaker A": "self"}}, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["manual_speaker_labels"]["Speaker A"] == "You"
    assert r.json()["speaker_labels"]["Speaker A"]["person_id"] == "self"

    detail = (await client.get(f"/recordings/{rid}", headers=H)).json()
    assert detail["manual_speaker_people"] == {"Speaker A": "self", "Speaker B": "mom"}
    assert detail["speaker_labels"]["Speaker B"]["label_source"] == "manual-person"
    # Episodes' participants follow the manual names.
    assert detail["episodes"][0]["participants"] == ["You", "Mom", "Speaker C"]

    # /growth: the manual "self" makes this recording count, Mom is a partner.
    growth = (await client.get("/growth", headers=H)).json()
    assert growth["identified_recordings"] == 1
    assert growth["points"][0]["my_score"] == 80
    assert growth["points"][0]["partner_names"] == ["Mom"]

    # Therapist rows carry the names + ids.
    sessions = (await client.get("/sessions", headers=H)).json()["sessions"]
    [row] = sessions
    assert [s["display"] for s in row["speakers"]] == ["You", "Mom", "Speaker C"]
    assert row["speakers"][1]["personId"] == "mom" and row["hasAudio"] is True

    # Clearing the person keeps the name (plain manual); clearing the name drops both.
    r = await client.patch(f"/recordings/{rid}/speaker-labels",
                           json={"people": {"Speaker B": ""}}, headers=H)
    assert r.json()["speaker_labels"]["Speaker B"] == {"display_label": "Mom", "label_source": "manual"}
    assert "Speaker B" not in r.json()["manual_speaker_people"]
    r = await client.patch(f"/recordings/{rid}/speaker-labels",
                           json={"labels": {"Speaker A": ""}}, headers=H)
    assert "Speaker A" not in r.json()["manual_speaker_people"]
    assert r.json()["speaker_labels"]["Speaker A"]["label_source"] == "generic"


async def test_patch_people_rejects_unknown_person_or_speaker(client, store):
    rid = str(uuid.uuid4())
    store.add_recording("u1", rid, TURNS, analysis=_generic_analysis())
    r = await client.patch(f"/recordings/{rid}/speaker-labels",
                           json={"people": {"Speaker B": "mom"}}, headers=H)
    assert r.status_code == 422 and "not enrolled" in r.json()["detail"]
    r = await client.patch(f"/recordings/{rid}/speaker-labels",
                           json={"people": {"Speaker Z": "self"}}, headers=H)
    assert r.status_code == 422 and "unknown speaker" in r.json()["detail"]
    r = await client.patch(f"/recordings/{rid}/speaker-labels",
                           json={"people": {"Speaker B": "Not A Slug"}}, headers=H)
    assert r.status_code == 422 and "invalid person id" in r.json()["detail"]
    # Nothing was written by the refused requests.
    detail = (await client.get(f"/recordings/{rid}", headers=H)).json()
    assert detail["manual_speaker_people"] == {} and detail["manual_speaker_labels"] == {}


async def test_catch_up_skips_recording_already_identified_by_manual_self(
    client, store, monkeypatch,
):
    rid = str(uuid.uuid4())
    store.add_recording("u1", rid, TURNS, analysis=_generic_analysis(),
                        manual={"Speaker A": "You"}, people={"Speaker A": "self"})
    await store.write_voiceprint("u1", _doc(E_SELF))
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    calls = []
    monkeypatch.setattr(store, "get_audio_bytes", lambda *a: calls.append(a) or None)
    r = await client.post("/voice/catch-up", headers=H)
    assert r.status_code == 200 and r.json() == {"checked": 0, "newly_identified": 0, "remaining": 0}
    assert calls == []  # never decoded — already identified by the human


# ---------------------------------------------------------------------------
# LIVE-GATED — the real ECAPA model on the scene pack
# ---------------------------------------------------------------------------

_AUDIO_DIR = Path(__file__).parent / "fixtures" / "audio"
_COUPLE = _AUDIO_DIR / "test_recording_scene_couple_escalation.wav"
_FAMILY3 = _AUDIO_DIR / "test_recording_scene_family3.wav"


def _scene_turns(wav_path: Path) -> tuple[dict, list[dict]]:
    meta = json.loads(wav_path.with_name(wav_path.stem + "_meta.json").read_text())
    gap = meta["silence_gap_sec"]
    turns, t = [], 0.0
    for m in meta["turns"]:
        turns.append({"speaker": m["speaker"], "text": m["text"],
                      "start_time": round(t, 4), "end_time": round(t + m["duration_sec"], 4)})
        t += m["duration_sec"] + gap
    return meta, turns


@pytest.mark.skipif(not speaker_id.is_available(), reason="voice deps (torch + speechbrain) not installed")
@pytest.mark.skipif(not (_COUPLE.exists() and _FAMILY3.exists()), reason="scene fixtures missing")
async def test_live_enroll_self_from_couple_scene_is_found_in_family3(client, store):
    """Through the endpoint: enroll "self" from the couple scene's Speaker A
    (the pack's designated self voice, pooled from the stored audio), then
    match every family3 speaker against the account's people — self must be
    found, and only self (Foundation D: 0.89–0.94 vs ≤0.28)."""
    meta, turns = _scene_turns(_COUPLE)
    rid = str(uuid.uuid4())
    store.add_recording("u1", rid, turns, audio=_COUPLE.read_bytes(), analysis={
        "speaker_labels": {s: {"display_label": s, "label_source": "generic"} for s in meta["speakers"]},
    })
    resp = await client.post(
        "/voice/people/self/enroll-from-recording",
        json={"recording_id": rid, "speaker_label": meta["self_speaker"]}, headers=H,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["seconds"] >= speaker_id.MIN_ENROLL_SECONDS
    assert resp.json()["speaker_labels"][meta["self_speaker"]]["display_label"] == "You"

    meta3, turns3 = _scene_turns(_FAMILY3)
    pcm, sr = audio_ingest.decode_to_pcm_16k(_FAMILY3.read_bytes(), _FAMILY3.name)
    prints = {p["person_id"]: np.asarray(p["embedding"], dtype=np.float32)
              for p in await store.list_voiceprints("u1")}
    report = speaker_id.identify_speakers_multi(pcm, sr, turns3, prints)
    assert report["matched"] == {meta3["self_speaker"]: "self"}, report["speakers"]
    assert report["speakers"][meta3["self_speaker"]]["scores"]["self"] >= speaker_id.MATCH_THRESHOLD
    for spk, entry in report["speakers"].items():
        if spk != meta3["self_speaker"]:
            assert entry["scores"]["self"] < speaker_id.MATCH_THRESHOLD

    # And the guard: family3's partner voice, tapped as "self", is refused.
    rid3 = str(uuid.uuid4())
    store.add_recording("u1", rid3, turns3, audio=_FAMILY3.read_bytes(), analysis=None)
    other = next(s for s in meta3["speakers"] if s != meta3["self_speaker"])
    ok = await client.post("/voice/people/mom/enroll-from-recording",
                           json={"recording_id": rid3, "speaker_label": other, "display_name": "Mom"},
                           headers=H)
    assert ok.status_code == 200, ok.text
    bad = await client.post("/voice/people/dad/enroll-from-recording",
                            json={"recording_id": rid3, "speaker_label": other, "display_name": "Dad"},
                            headers=H)
    assert bad.status_code == 422 and bad.json()["detail"].startswith("[sounds-like-someone-else]")
