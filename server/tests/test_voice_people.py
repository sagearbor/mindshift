"""Multi-person named voiceprints (Foundation B) — torch-free tests.

One account, N named people: the owner ("self" → "You") plus partners the
user names ("alex" → "Alex"). Covers, without ever loading the ECAPA model:

* the greedy one-to-one matcher (``speaker_id.identify_speakers_multi``) on
  synthetic embeddings — best person per speaker, a person wins at most one
  speaker, below-threshold stays unlabeled, the legacy single-print wrapper
  is shape-identical to before;
* the pure profile/report helpers (``as_person``, ``new_profile`` person
  fields, ``enrolled_display_labels`` for both report shapes,
  ``without_matches_for``) and the "exactly one self" structural invariant;
* the GCS store's per-person layout against a fake bucket — incl. the LEGACY
  ``voiceprints/{uid}/profile.json`` read-through shim (read as self without
  rewriting; retired by the first self write; removed by a self delete);
* the label ladder + episodes consuming a multi-person report (a named
  partner gets ``label_source="enrolled"`` under their name, never an
  invented one), and main's ``_identify_enrolled_speakers`` wiring;
* the router: enroll a partner, list people, per-person profile, delete a
  person, the 422s (nameless new partner, bad slug), and the relabel that
  keeps "You" and "Alex" coexisting in one recording.
"""

from __future__ import annotations

import json
import uuid

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import episodes
import main
import recordings_store
import speaker_id
from main import app, init_db

pytestmark = pytest.mark.anyio

E_SELF = np.array([1.0, 0.0, 0.0], dtype=np.float32)
E_ALEX = np.array([0.0, 1.0, 0.0], dtype=np.float32)
E_MOM = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _unit(*xs):
    return speaker_id.l2_normalize(np.array(xs, dtype=np.float32))


def _fake_embed_by_speaker(mapping):
    def _embed(pcm, sr, turns, speaker, **kw):
        return mapping.get(speaker)
    return _embed


def _turns(*speakers):
    return [
        {"speaker": s, "text": "…", "start_time": float(i * 3), "end_time": float(i * 3 + 3)}
        for i, s in enumerate(speakers)
    ]


PCM = np.zeros(10, dtype=np.float32)


# ---------------------------------------------------------------------------
# identify_speakers_multi — greedy one-to-one matching on synthetic vectors
# ---------------------------------------------------------------------------

def test_multi_each_speaker_gets_best_person(monkeypatch):
    monkeypatch.setattr(
        speaker_id, "embed_speaker",
        _fake_embed_by_speaker({"Speaker A": E_SELF, "Speaker B": E_ALEX, "Speaker C": E_MOM}),
    )
    report = speaker_id.identify_speakers_multi(
        PCM, 16000, _turns("Speaker A", "Speaker B", "Speaker C"),
        {"self": E_SELF, "alex": E_ALEX, "mom": E_MOM},
        threshold=0.5,
        people={"alex": {"display_name": "Alex"}, "mom": {"display_name": "Mom"}},
    )
    assert report["matched"] == {"Speaker A": "self", "Speaker B": "alex", "Speaker C": "mom"}
    assert report["matched_speaker"] == "Speaker A"  # legacy self key
    a = report["speakers"]["Speaker A"]
    assert a["matched_person_id"] == "self" and a["is_self"] is True
    assert a["display_name"] == "You"
    assert a["scores"] == {"self": 1.0, "alex": 0.0, "mom": 0.0}
    assert a["score"] == 1.0 and a["is_you"] is True  # legacy per-speaker keys
    b = report["speakers"]["Speaker B"]
    assert b["matched_person_id"] == "alex" and b["is_self"] is False
    assert b["display_name"] == "Alex" and b["is_you"] is False
    assert report["people"]["self"] == {"display_name": "You", "is_self": True}
    assert report["people"]["alex"] == {"display_name": "Alex", "is_self": False}


def test_multi_person_wins_at_most_one_speaker_greedy_by_score(monkeypatch):
    # The diarizer split ONE voice into two clusters: both A and B look like
    # Alex (0.9 and 0.8). Only the stronger gets the label; the other stays
    # generic rather than two "Alex"es — a person is one voice.
    near = _unit(0.0, 0.9, np.sqrt(1 - 0.81))
    nearer = _unit(0.0, 0.8, np.sqrt(1 - 0.64))
    monkeypatch.setattr(
        speaker_id, "embed_speaker",
        _fake_embed_by_speaker({"Speaker A": nearer, "Speaker B": near}),
    )
    report = speaker_id.identify_speakers_multi(
        PCM, 16000, _turns("Speaker A", "Speaker B"), {"alex": E_ALEX}, threshold=0.5,
    )
    assert report["matched"] == {"Speaker B": "alex"}
    assert report["speakers"]["Speaker A"]["matched_person_id"] is None
    assert report["speakers"]["Speaker A"]["scores"]["alex"] == pytest.approx(0.8, abs=1e-3)
    assert report["matched_speaker"] is None  # no self print supplied
    assert "score" not in report["speakers"]["Speaker A"]  # legacy keys need self


def test_multi_greedy_resolves_cross_conflicts_highest_first(monkeypatch):
    # A is a decent self (0.7) and a great alex (0.9); B is a decent self
    # (0.75). Greedy takes (A, alex, 0.9) first, then (B, self, 0.75) — A's
    # weaker self candidacy never blocks B.
    a = _unit(0.7, 0.9, 0.0)
    b = _unit(0.75, 0.2, 0.0)
    monkeypatch.setattr(
        speaker_id, "embed_speaker", _fake_embed_by_speaker({"Speaker A": a, "Speaker B": b}),
    )
    report = speaker_id.identify_speakers_multi(
        PCM, 16000, _turns("Speaker A", "Speaker B"),
        {"self": E_SELF, "alex": E_ALEX}, threshold=0.5,
    )
    assert report["matched"] == {"Speaker A": "alex", "Speaker B": "self"}
    assert report["matched_speaker"] == "Speaker B"


def test_multi_below_threshold_no_label_scores_kept(monkeypatch):
    weak = _unit(0.4, np.sqrt(1 - 0.16), 0.0)  # 0.4 to self, ~0.92 to alex
    monkeypatch.setattr(speaker_id, "embed_speaker", _fake_embed_by_speaker({"Speaker A": weak}))
    report = speaker_id.identify_speakers_multi(
        PCM, 16000, _turns("Speaker A"), {"self": E_SELF}, threshold=0.5,
    )
    assert report["matched"] == {}
    assert report["matched_speaker"] is None
    assert report["speakers"]["Speaker A"]["scores"]["self"] == pytest.approx(0.4, abs=1e-3)
    assert report["speakers"]["Speaker A"]["is_you"] is False


def test_multi_skips_speaker_with_too_little_audio(monkeypatch):
    monkeypatch.setattr(speaker_id, "embed_speaker", _fake_embed_by_speaker({"Speaker A": None}))
    report = speaker_id.identify_speakers_multi(
        PCM, 16000, _turns("Speaker A"), {"self": E_SELF, "alex": E_ALEX},
    )
    assert report["speakers"] == {} and report["matched"] == {}


def test_multi_embeds_each_speaker_once(monkeypatch):
    calls = []

    def _embed(pcm, sr, turns, speaker, **kw):
        calls.append(speaker)
        return E_SELF
    monkeypatch.setattr(speaker_id, "embed_speaker", _embed)
    speaker_id.identify_speakers_multi(
        PCM, 16000, _turns("Speaker A", "Speaker B", "Speaker A"),
        {"self": E_SELF, "alex": E_ALEX, "mom": E_MOM},
    )
    assert calls == ["Speaker A", "Speaker B"]  # not once per (speaker, person)


def test_legacy_identify_speakers_shape_unchanged(monkeypatch):
    monkeypatch.setattr(
        speaker_id, "embed_speaker",
        _fake_embed_by_speaker({"Speaker A": E_SELF, "Speaker B": E_ALEX}),
    )
    report = speaker_id.identify_speakers(
        PCM, 16000, _turns("Speaker A", "Speaker B"), E_SELF, threshold=0.5,
    )
    assert set(report) == {"matched_speaker", "match_threshold", "model", "speakers"}
    assert report["speakers"] == {
        "Speaker A": {"score": 1.0, "is_you": True},
        "Speaker B": {"score": 0.0, "is_you": False},
    }


# ---------------------------------------------------------------------------
# Pure helpers — profile person view + report readers
# ---------------------------------------------------------------------------

def test_as_person_defaults_legacy_doc_to_self():
    legacy = {"embedding": [1.0, 0.0], "enroll_count": 2}
    view = speaker_id.as_person(legacy)
    assert view["person_id"] == "self" and view["display_name"] == "You"
    assert view["is_self"] is True
    assert view["version"] == 2 and len(view["samples"]) == 1  # v2 view too
    assert speaker_id.as_person(None) is None


def test_as_person_is_self_is_structural_not_a_flag():
    # A doc that CLAIMS is_self but isn't keyed "self" is not the owner —
    # exactly-one-self is guaranteed by the reserved key, not trusted flags.
    doc = {"version": 2, "samples": [], "person_id": "alex", "display_name": "Alex", "is_self": True}
    assert speaker_id.as_person(doc)["is_self"] is False
    # And the owner is always "You", whatever a display_name says.
    doc = {"version": 2, "samples": [], "person_id": "self", "display_name": "Bob"}
    assert speaker_id.as_person(doc)["display_name"] == "You"


def test_new_profile_carries_person_fields_and_rename():
    p1 = speaker_id.new_profile(
        E_ALEX, None, recording_id="r1", speaker="Speaker B", now_iso="t0",
        person_id="alex", display_name="Alex",
    )
    assert (p1["person_id"], p1["display_name"], p1["is_self"]) == ("alex", "Alex", False)
    # Re-enroll without a name keeps it; with a new one renames.
    p2 = speaker_id.new_profile(E_ALEX, p1, recording_id="r2", speaker="Speaker B", now_iso="t1")
    assert p2["display_name"] == "Alex" and p2["enroll_count"] == 2
    p3 = speaker_id.new_profile(
        E_ALEX, p2, recording_id="r3", speaker="Speaker B", now_iso="t2", display_name="Alexander",
    )
    assert p3["display_name"] == "Alexander" and p3["enroll_count"] == 3
    # Default (every pre-existing caller): the owner.
    p_self = speaker_id.new_profile(E_SELF, None, recording_id="r1", speaker="A", now_iso="t0")
    assert (p_self["person_id"], p_self["display_name"], p_self["is_self"]) == ("self", "You", True)
    # remove_sample preserves the person fields.
    kept = speaker_id.remove_sample(p3, p3["samples"][0]["id"], now_iso="t3")
    assert kept["person_id"] == "alex" and kept["display_name"] == "Alexander"


def test_enrolled_display_labels_multi_and_legacy_shapes():
    multi = {
        "matched": {"Speaker A": "self", "Speaker B": "alex", "Speaker C": "ghost"},
        "people": {"self": {"display_name": "You", "is_self": True},
                   "alex": {"display_name": "Alex", "is_self": False}},
    }
    # "ghost" has no display name → NO label (never invented); others labeled.
    assert speaker_id.enrolled_display_labels(multi) == {"Speaker A": "You", "Speaker B": "Alex"}
    assert speaker_id.enrolled_display_labels({"matched_speaker": "Speaker B"}) == {"Speaker B": "You"}
    assert speaker_id.enrolled_display_labels({"matched_speaker": "  "}) == {}
    assert speaker_id.enrolled_display_labels(None) == {}
    assert speaker_id.enrolled_display_labels({"matched": "nope"}) == {}


def test_without_matches_for_suppresses_manual_speakers():
    identity = {"matched_speaker": "Speaker A", "matched": {"Speaker A": "self", "Speaker B": "alex"}}
    out = speaker_id.without_matches_for(identity, {"Speaker A"})
    assert out["matched_speaker"] is None and out["matched"] == {"Speaker B": "alex"}
    assert identity["matched"] == {"Speaker A": "self", "Speaker B": "alex"}  # pure


# ---------------------------------------------------------------------------
# Storage — RecordingsStore per-person layout on a fake bucket
# ---------------------------------------------------------------------------

class _FakeBlob:
    def __init__(self, bucket, name):
        self._bucket, self.name = bucket, name

    def exists(self):
        return self.name in self._bucket.objects

    def download_as_bytes(self):
        return self._bucket.objects[self.name]

    def upload_from_string(self, data, content_type=None):
        self._bucket.objects[self.name] = data.encode() if isinstance(data, str) else data

    def delete(self):
        del self._bucket.objects[self.name]


class FakeBucket:
    """The minimal google-cloud-storage bucket surface RecordingsStore's
    voiceprint methods touch (blob/exists/download/upload/delete/list_blobs)."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def blob(self, name):
        return _FakeBlob(self, name)

    def list_blobs(self, prefix=""):
        return [_FakeBlob(self, n) for n in sorted(self.objects) if n.startswith(prefix)]


def _doc(embedding, **extra):
    return {"version": 2, "embedding": list(map(float, embedding)), "dim": len(embedding),
            "enroll_count": 1, "samples": [{"id": "s1", "embedding": list(map(float, embedding))}],
            **extra}


@pytest.fixture
def bucket():
    return FakeBucket()


@pytest.fixture
def store(bucket):
    return recordings_store.RecordingsStore(bucket)


async def test_store_writes_each_person_under_its_own_path(store, bucket):
    await store.write_voiceprint("u1", _doc(E_SELF))  # no person_id → self
    await store.write_voiceprint("u1", _doc(E_ALEX, person_id="alex", display_name="Alex"))
    assert set(bucket.objects) == {
        "voiceprints/u1/self/profile.json", "voiceprints/u1/alex/profile.json",
    }
    me = await store.read_voiceprint("u1")
    assert me["person_id"] == "self" and me["is_self"] is True and me["display_name"] == "You"
    alex = await store.read_voiceprint("u1", "alex")
    assert alex["display_name"] == "Alex" and alex["is_self"] is False
    assert await store.read_voiceprint("u1", "mom") is None
    assert await store.read_voiceprint("u2", "alex") is None  # uid-scoped


async def test_store_list_orders_self_first_then_by_name(store):
    await store.write_voiceprint("u1", _doc(E_MOM, person_id="mom", display_name="Mom"))
    await store.write_voiceprint("u1", _doc(E_ALEX, person_id="alex", display_name="Alex"))
    await store.write_voiceprint("u1", _doc(E_SELF))
    await store.write_voiceprint("u2", _doc(E_SELF))
    people = await store.list_voiceprints("u1")
    assert [p["person_id"] for p in people] == ["self", "alex", "mom"]
    assert sum(p["is_self"] for p in people) == 1
    assert await store.list_voiceprints("u3") == []


async def test_store_delete_is_per_person_and_idempotent(store, bucket):
    await store.write_voiceprint("u1", _doc(E_SELF))
    await store.write_voiceprint("u1", _doc(E_ALEX, person_id="alex", display_name="Alex"))
    assert await store.delete_voiceprint("u1", "alex") is True
    assert await store.delete_voiceprint("u1", "alex") is False
    assert await store.read_voiceprint("u1") is not None  # self untouched
    assert await store.delete_voiceprint("u1") is True
    assert bucket.objects == {}


async def test_store_legacy_document_reads_through_as_self_without_rewrite(store, bucket):
    legacy_name = "voiceprints/u1/profile.json"
    bucket.objects[legacy_name] = json.dumps({"embedding": [1.0, 0.0, 0.0], "enroll_count": 2}).encode()

    me = await store.read_voiceprint("u1")
    assert me["person_id"] == "self" and me["display_name"] == "You" and me["is_self"] is True
    assert me["embedding"] == [1.0, 0.0, 0.0]
    people = await store.list_voiceprints("u1")
    assert [p["person_id"] for p in people] == ["self"]
    # Reads are side-effect free: nothing migrated, nothing rewritten.
    assert set(bucket.objects) == {legacy_name}

    # A partner alongside the legacy owner doc lists both, the owner first.
    await store.write_voiceprint("u1", _doc(E_ALEX, person_id="alex", display_name="Alex"))
    assert [p["person_id"] for p in await store.list_voiceprints("u1")] == ["self", "alex"]
    assert legacy_name in bucket.objects  # a partner write never touches it


async def test_store_first_self_write_retires_legacy_document(store, bucket):
    legacy_name = "voiceprints/u1/profile.json"
    bucket.objects[legacy_name] = json.dumps({"embedding": [1.0, 0.0, 0.0]}).encode()
    await store.write_voiceprint("u1", _doc(E_SELF))
    assert legacy_name not in bucket.objects
    assert "voiceprints/u1/self/profile.json" in bucket.objects
    assert len(await store.list_voiceprints("u1")) == 1  # never two "self"s


async def test_store_self_delete_removes_legacy_document(store, bucket):
    bucket.objects["voiceprints/u1/profile.json"] = json.dumps({"embedding": [1.0, 0.0]}).encode()
    assert await store.delete_voiceprint("u1") is True
    assert bucket.objects == {}
    assert await store.read_voiceprint("u1") is None


# ---------------------------------------------------------------------------
# Label ladder + episodes + main wiring consume the multi-person report
# ---------------------------------------------------------------------------

MULTI_IDENTITY = {
    "matched_speaker": "Speaker A",
    "matched": {"Speaker A": "self", "Speaker B": "alex"},
    "people": {"self": {"display_name": "You", "is_self": True},
               "alex": {"display_name": "Alex", "is_self": False}},
}


def test_ladder_labels_named_partner_as_enrolled():
    labels = main._resolve_speaker_labels(
        ["Speaker A", "Speaker B", "Speaker C"], {"Speaker B": "Bob", "Speaker C": "Cy"}, None,
        speaker_identity=MULTI_IDENTITY,
    )
    assert labels["Speaker A"].model_dump() == {"display_label": "You", "label_source": "enrolled"}
    # A voiceprint beats a transcript-inferred name.
    assert labels["Speaker B"].model_dump() == {"display_label": "Alex", "label_source": "enrolled"}
    assert labels["Speaker C"].model_dump() == {"display_label": "Cy", "label_source": "name"}


def test_episodes_participants_use_partner_names():
    eps = episodes.segment_episodes(
        _turns("Speaker A", "Speaker B", "Speaker C"), speaker_identity=MULTI_IDENTITY,
    )
    assert eps[0]["participants"] == ["You", "Alex", "Speaker C"]


def test_growth_point_me_is_you_not_an_enrolled_partner():
    rec = {
        "id": "r", "created_at": "2026-08-01T00:00:00+00:00", "title": "t",
        "turns": _turns("Speaker A", "Speaker B"),
        "analysis": {
            "speaker_labels": {
                "Speaker A": {"display_label": "Alex", "label_source": "enrolled"},
                "Speaker B": {"display_label": "You", "label_source": "enrolled"},
            },
            "report_cards": {"Speaker A": {"score": 30}, "Speaker B": {"score": 70}},
        },
    }
    point = main._growth_point(rec)
    assert point is not None
    assert point.my_score == 70  # Speaker B — the "You", not the enrolled partner
    assert point.partner_names == ["Alex"]  # an enrolled partner IS a named partner


class FakeVoiceStore:
    def __init__(self):
        self._recordings: dict[tuple, dict] = {}
        self._people: dict[tuple, dict] = {}  # (uid, person_id) → profile

    def add_recording(self, uid, rid, turns, audio=b"AUDIO", analysis=None):
        self._recordings[(uid, rid)] = {"turns": turns, "audio": audio, "analysis": analysis,
                                        "created_at": "2026-08-01T00:00:00+00:00"}

    async def get_recording(self, uid, recording_id):
        r = self._recordings.get((uid, recording_id))
        if r is None:
            return None
        return {"id": recording_id, "created_at": r["created_at"], "filename": "rec.m4a",
                "title": None, "media_type": "audio", "duration_seconds": 60.0,
                "turns": r["turns"], "analysis": r["analysis"]}

    async def list_recordings(self, uid):
        return [{"id": rid, "created_at": r["created_at"], "has_analysis": r["analysis"] is not None}
                for (u, rid), r in self._recordings.items() if u == uid]

    async def get_audio_bytes(self, uid, recording_id):
        r = self._recordings.get((uid, recording_id))
        return None if r is None else r["audio"]

    async def overwrite_analysis(self, uid, recording_id, *, turns, analysis, reanalyzed_at):
        r = self._recordings.get((uid, recording_id))
        if r is None:
            return None
        r["turns"], r["analysis"] = turns, analysis
        return {"id": recording_id, "reanalyzed_at": reanalyzed_at}

    async def read_voiceprint(self, uid, person_id=None):
        pid = person_id or speaker_id.SELF_PERSON_ID
        return speaker_id.as_person(self._people.get((uid, pid)), person_id=pid)

    async def list_voiceprints(self, uid):
        return [speaker_id.as_person(p, person_id=pid) for (u, pid), p in self._people.items() if u == uid]

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
def voice_store():
    fake = FakeVoiceStore()
    app.state.recordings_store = fake
    yield fake
    del app.state.recordings_store


TURNS = _turns("Speaker A", "Speaker B")
H = {"X-Test-Uid": "u1"}


def _enroll_ready(monkeypatch, by_speaker):
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    import routers.voice as voice_router
    monkeypatch.setattr(
        voice_router, "decode_to_pcm",
        lambda data, name: (np.zeros(16000 * 5, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(speaker_id, "embed_speaker", _fake_embed_by_speaker(by_speaker))


async def test_identify_enrolled_speakers_matches_every_person(voice_store, monkeypatch):
    # main's analyze-path wiring: reads ALL people, matches each speaker.
    await voice_store.write_voiceprint("u1", _doc(E_SELF))
    await voice_store.write_voiceprint("u1", _doc(E_ALEX, person_id="alex", display_name="Alex"))
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    monkeypatch.setattr(
        speaker_id, "embed_speaker", _fake_embed_by_speaker({"Speaker A": E_SELF, "Speaker B": E_ALEX}),
    )
    turns = [main.AnalyzeTurn(**t) for t in TURNS]
    report = await main._identify_enrolled_speakers("u1", PCM, 16000, turns)
    assert report["matched"] == {"Speaker A": "self", "Speaker B": "alex"}
    assert report["people"]["alex"]["display_name"] == "Alex"
    assert speaker_id.enrolled_display_labels(report) == {"Speaker A": "You", "Speaker B": "Alex"}
    # Nobody enrolled → None (skip), exactly as before.
    assert await main._identify_enrolled_speakers("u2", PCM, 16000, turns) is None


# ---------------------------------------------------------------------------
# Router — enroll a partner, list people, delete a person
# ---------------------------------------------------------------------------

def _generic_analysis():
    return {"speaker_labels": {
        "Speaker A": {"display_label": "Speaker A", "label_source": "generic"},
        "Speaker B": {"display_label": "Speaker B", "label_source": "generic"},
    }, "report_cards": {"Speaker A": {"score": 80}, "Speaker B": {"score": 40}}}


async def test_enroll_partner_then_list_people(client, voice_store, monkeypatch):
    rid = str(uuid.uuid4())
    voice_store.add_recording("u1", rid, TURNS, analysis=_generic_analysis())
    _enroll_ready(monkeypatch, {"Speaker A": E_SELF, "Speaker B": E_ALEX})

    me = await client.post("/voice/enroll", json={"recording_id": rid, "speaker": "Speaker A"}, headers=H)
    assert me.status_code == 200, me.text
    assert me.json()["person_id"] == "self" and me.json()["display_name"] == "You"
    assert me.json()["is_self"] is True

    alex = await client.post(
        "/voice/enroll",
        json={"recording_id": rid, "speaker": "Speaker B", "person_id": "alex", "display_name": "Alex"},
        headers=H,
    )
    assert alex.status_code == 200, alex.text
    body = alex.json()
    assert body["person_id"] == "alex" and body["display_name"] == "Alex" and body["is_self"] is False
    assert body["enroll_count"] == 1

    people = await client.get("/voice/people", headers=H)
    assert people.status_code == 200
    rows = people.json()["people"]
    assert [(p["person_id"], p["display_name"], p["is_self"]) for p in rows] == [
        ("self", "You", True), ("alex", "Alex", False),
    ]
    assert all(p["enrolled"] and p["enroll_count"] == 1 for p in rows)
    assert "embedding" not in json.dumps(rows)  # never the signature

    prof = await client.get("/voice/profile", params={"person_id": "alex"}, headers=H)
    assert prof.json()["display_name"] == "Alex" and prof.json()["enrolled"] is True
    unknown = await client.get("/voice/profile", params={"person_id": "mom"}, headers=H)
    assert unknown.status_code == 200 and unknown.json()["enrolled"] is False
    assert unknown.json()["display_name"] is None  # no name invented

    # Both people labeled in the SAME recording — "You" survives "Alex".
    detail = await client.get(f"/recordings/{rid}", headers=H)
    labels = detail.json()["speaker_labels"]
    assert labels["Speaker A"] == {"display_label": "You", "label_source": "enrolled"}
    assert labels["Speaker B"] == {"display_label": "Alex", "label_source": "enrolled"}
    identity = detail.json()["analysis"]["speaker_identity"]
    assert identity["matched"] == {"Speaker A": "self", "Speaker B": "alex"}
    assert identity["matched_speaker"] == "Speaker A"

    # Growth still sees exactly one "me" and names the enrolled partner.
    growth = await client.get("/growth", headers=H)
    assert growth.json()["identified_recordings"] == 1
    assert growth.json()["points"][0]["my_score"] == 80
    assert growth.json()["points"][0]["partner_names"] == ["Alex"]


async def test_enroll_partner_corrects_a_speaker_previously_marked_you(
    client, voice_store, monkeypatch,
):
    # Speaker B was (wrongly) "You"; the user says "no, that's Alex". Alex
    # takes B and the stale "You" on B is replaced — never two labels on one
    # speaker, and speaker_identity's self match is cleared honestly.
    rid = str(uuid.uuid4())
    analysis = _generic_analysis()
    analysis["speaker_labels"]["Speaker B"] = {"display_label": "You", "label_source": "enrolled"}
    analysis["speaker_identity"] = {"matched_speaker": "Speaker B"}
    voice_store.add_recording("u1", rid, TURNS, analysis=analysis)
    _enroll_ready(monkeypatch, {"Speaker B": E_ALEX})
    res = await client.post(
        "/voice/enroll",
        json={"recording_id": rid, "speaker": "Speaker B", "person_id": "alex", "display_name": "Alex"},
        headers=H,
    )
    assert res.status_code == 200, res.text
    detail = await client.get(f"/recordings/{rid}", headers=H)
    assert detail.json()["speaker_labels"]["Speaker B"]["display_label"] == "Alex"
    identity = detail.json()["analysis"]["speaker_identity"]
    assert identity["matched"] == {"Speaker B": "alex"} and identity["matched_speaker"] is None


async def test_enroll_new_partner_without_name_422(client, voice_store, monkeypatch):
    rid = str(uuid.uuid4())
    voice_store.add_recording("u1", rid, TURNS)
    _enroll_ready(monkeypatch, {"Speaker B": E_ALEX})
    res = await client.post(
        "/voice/enroll", json={"recording_id": rid, "speaker": "Speaker B", "person_id": "alex"}, headers=H,
    )
    assert res.status_code == 422
    assert "display_name" in res.json()["detail"]
    assert await voice_store.list_voiceprints("u1") == []  # nothing stored


async def test_reenroll_partner_keeps_name_and_rename_updates_it(client, voice_store, monkeypatch):
    rid = str(uuid.uuid4())
    voice_store.add_recording("u1", rid, TURNS)
    _enroll_ready(monkeypatch, {"Speaker B": E_ALEX})
    base = {"recording_id": rid, "speaker": "Speaker B", "person_id": "alex"}
    assert (await client.post("/voice/enroll", json={**base, "display_name": "Alex"}, headers=H)).status_code == 200
    again = await client.post("/voice/enroll", json=base, headers=H)
    assert again.status_code == 200 and again.json()["display_name"] == "Alex"
    assert again.json()["enroll_count"] == 2
    renamed = await client.post("/voice/enroll", json={**base, "display_name": "Alexander"}, headers=H)
    assert renamed.json()["display_name"] == "Alexander" and renamed.json()["enroll_count"] == 3


async def test_enroll_rejects_bad_person_id(client, voice_store, monkeypatch):
    rid = str(uuid.uuid4())
    voice_store.add_recording("u1", rid, TURNS)
    _enroll_ready(monkeypatch, {"Speaker B": E_ALEX})
    for bad in ("../self", "Alex Smith", "a/b", ""):
        res = await client.post(
            "/voice/enroll",
            json={"recording_id": rid, "speaker": "Speaker B", "person_id": bad, "display_name": "x"},
            headers=H,
        )
        assert res.status_code == 422, bad
    assert (await client.request("DELETE", "/voice/people/..%2Fself", headers=H)).status_code in (404, 422)
    assert (await client.request("DELETE", "/voice/people/Alex%20Smith", headers=H)).status_code == 422


async def test_delete_person_is_real_scoped_and_idempotent(client, voice_store, monkeypatch):
    rid = str(uuid.uuid4())
    voice_store.add_recording("u1", rid, TURNS)
    _enroll_ready(monkeypatch, {"Speaker A": E_SELF, "Speaker B": E_ALEX})
    await client.post("/voice/enroll", json={"recording_id": rid, "speaker": "Speaker A"}, headers=H)
    await client.post(
        "/voice/enroll",
        json={"recording_id": rid, "speaker": "Speaker B", "person_id": "alex", "display_name": "Alex"},
        headers=H,
    )
    # Another account can't delete u1's Alex.
    other = await client.request("DELETE", "/voice/people/alex", headers={"X-Test-Uid": "u2"})
    assert other.json() == {"deleted": False, "person_id": "alex"}
    assert ("u1", "alex") in voice_store._people

    d1 = await client.request("DELETE", "/voice/people/alex", headers=H)
    assert d1.status_code == 200 and d1.json() == {"deleted": True, "person_id": "alex"}
    assert ("u1", "alex") not in voice_store._people
    assert ("u1", "self") in voice_store._people  # the owner is untouched
    d2 = await client.request("DELETE", "/voice/people/alex", headers=H)
    assert d2.json()["deleted"] is False
    # "self" through the people route == "forget my voice".
    d3 = await client.request("DELETE", "/voice/people/self", headers=H)
    assert d3.json()["deleted"] is True and voice_store._people == {}


async def test_delete_sample_for_a_partner(client, voice_store, monkeypatch):
    rid = str(uuid.uuid4())
    voice_store.add_recording("u1", rid, TURNS)
    _enroll_ready(monkeypatch, {"Speaker B": E_ALEX})
    await client.post(
        "/voice/enroll",
        json={"recording_id": rid, "speaker": "Speaker B", "person_id": "alex", "display_name": "Alex"},
        headers=H,
    )
    sample_id = (await client.get("/voice/profile", params={"person_id": "alex"}, headers=H)).json()["samples"][0]["id"]
    # The owner's profile doesn't know this sample → 404 (per-person scoping).
    assert (await client.request("DELETE", f"/voice/samples/{sample_id}", headers=H)).status_code == 404
    res = await client.request(
        "DELETE", f"/voice/samples/{sample_id}", params={"person_id": "alex"}, headers=H,
    )
    assert res.status_code == 200 and res.json() == {"deleted": True, "enrolled": False, "enroll_count": 0}
    assert ("u1", "alex") not in voice_store._people  # last sample → gone entirely


async def test_people_endpoints_storage_disabled(client, monkeypatch):
    # No store on app.state: list answers honestly, delete 503s — the same
    # split GET /voice/profile vs DELETE /voice/voiceprint already have.
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    res = await client.get("/voice/people", headers=H)
    assert res.status_code == 200
    assert res.json() == {"available": True, "storage_enabled": False, "people": []}
    assert (await client.request("DELETE", "/voice/people/alex", headers=H)).status_code == 503


async def test_enroll_direct_accepts_person_form_fields(client, voice_store, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    import routers.voice as voice_router
    speech = (0.1 * np.ones(16000 * 5, dtype=np.float32), 16000)
    monkeypatch.setattr(voice_router, "decode_to_pcm_16k", lambda data, name: speech)
    monkeypatch.setattr(speaker_id, "embed_pcm", lambda pcm, sr: E_ALEX)
    res = await client.post(
        "/voice/enroll-direct",
        files={"file": ("clip.wav", b"RIFF-not-really", "audio/wav")},
        data={"person_id": "alex", "display_name": "Alex"},
        headers=H,
    )
    assert res.status_code == 200, res.text
    assert res.json()["person_id"] == "alex" and res.json()["display_name"] == "Alex"
    assert ("u1", "alex") in voice_store._people
    # Default (no fields) is still the owner.
    res = await client.post(
        "/voice/enroll-direct",
        files={"file": ("clip.wav", b"RIFF-not-really", "audio/wav")}, headers=H,
    )
    assert res.status_code == 200 and res.json()["person_id"] == "self"


# ---------------------------------------------------------------------------
# review 2026-08-24: /voice/enroll is behind the per-IP limiter like every
# other expensive route (GCS download + ffmpeg decode + ECAPA embed per call)
# ---------------------------------------------------------------------------

async def test_enroll_from_recording_is_rate_limited(client, monkeypatch):
    main._rate_limiter.reset()
    monkeypatch.setattr(main._rate_limiter, "limit", 1)
    body = {"recording_id": str(uuid.uuid4()), "speaker": "Speaker A"}
    first = await client.post("/voice/enroll", json=body, headers=H)
    assert first.status_code != 429  # the budget's one request (503/404 here)
    second = await client.post("/voice/enroll", json=body, headers=H)
    assert second.status_code == 429
    main._rate_limiter.reset()
