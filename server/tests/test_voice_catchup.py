"""POST /voice/catch-up — bulk re-match already-stored recordings against an
enrolled voiceprint, for recordings that predate enrollment (or predate any
"This is me" tap on them).

The owner's exact bug this fixes: enroll via the guided "Train my voice" flow
(POST /voice/enroll-direct — writes ONLY the account-level voiceprint, never
touches any recording), then open "Your Growth" and see "No growth data yet"
for 5 already-stored, already-analyzed recordings. Tapping "This is me" one at
a time (Part A) requires being confident which diarized speaker is you; catch-up
re-matches everything already stored in ONE call, cheaply — decode + embed
against the already-computed turns, NOT a full re-transcription.

Torch-free, same house style as test_voice_enrollment.py: the real
speaker_id.identify_speakers cosine-match logic runs for real (never
reimplemented here); only speaker_id.embed_speaker / is_available and the
router's decode_to_pcm are monkeypatched with deterministic doubles.
"""

import uuid

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import routers.voice as voice_router
import speaker_id
from main import app, init_db

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# In-memory fake store — list + get + audio + voiceprint + overwrite_analysis
# ---------------------------------------------------------------------------

class FakeCatchUpStore:
    def __init__(self):
        self._recordings: dict[tuple, dict] = {}  # (uid, rid) → recording dict
        self._voiceprints: dict[str, dict] = {}

    def add_recording(
        self, uid, rid, turns, audio=b"AUDIO", analysis=None,
        created_at="2026-08-01T00:00:00+00:00", manual_speaker_labels=None,
    ):
        rec = {
            "turns": turns, "audio": audio, "analysis": analysis,
            "created_at": created_at,
        }
        if manual_speaker_labels:
            rec["manual_speaker_labels"] = manual_speaker_labels
        self._recordings[(uid, rid)] = rec

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

    async def get_recording(self, uid, recording_id):
        r = self._recordings.get((uid, recording_id))
        if r is None:
            return None
        out = {
            "id": recording_id,
            "created_at": r["created_at"],
            "filename": "rec.m4a",
            "title": None,
            "media_type": "audio",
            "duration_seconds": 60.0,
            "turns": r["turns"],
            "analysis": r["analysis"],
        }
        if "manual_speaker_labels" in r:
            out["manual_speaker_labels"] = r["manual_speaker_labels"]
        return out

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
    # This route has its OWN, much tighter rate limiter (_CATCHUP_RATE_LIMIT
    # _PER_MINUTE, default 5/min) than the generic per-route one — reset it
    # per test so this file's ~15 catch-up calls across many tests never
    # trip it (same hygiene test_reanalyze.py/test_endpoints.py already do
    # for main._rate_limiter).
    voice_router._catchup_rate_limiter.reset()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def store():
    fake = FakeCatchUpStore()
    app.state.recordings_store = fake
    yield fake
    del app.state.recordings_store


def _rid():
    return str(uuid.uuid4())


TURNS_AB = [
    {"speaker": "Speaker A", "text": "hi", "start_time": 0.0, "end_time": 3.0},
    {"speaker": "Speaker B", "text": "yo", "start_time": 3.0, "end_time": 6.0},
]


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


def _identified_analysis(me="Speaker A"):
    a = _unidentified_analysis()
    a["speaker_labels"][me] = {"display_label": "You", "label_source": "enrolled"}
    return a


def _fake_embed_by_speaker(mapping):
    def _embed(pcm, sr, turns, speaker, **kw):
        return mapping.get(speaker)
    return _embed


def _catchup_ready(monkeypatch, embed_map):
    """Wire is_available + decode + the per-speaker embedder to deterministic
    doubles — same pattern as test_voice_enrollment.py's _enroll_ready, minus
    the single-embedding shortcut (catch-up matches ALL speakers, so it needs
    a full speaker → embedding map)."""
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    import routers.voice as voice_router
    monkeypatch.setattr(
        voice_router, "decode_to_pcm",
        lambda data, name: (np.zeros(16000 * 5, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(
        speaker_id, "embed_speaker", _fake_embed_by_speaker(embed_map),
    )


# ---------------------------------------------------------------------------
# Availability / honest-empty-state gates
# ---------------------------------------------------------------------------

async def test_catchup_unavailable_dep_503(client, store, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: False)
    res = await client.post("/voice/catch-up", headers={"X-Test-Uid": "u1"})
    assert res.status_code == 503


async def test_catchup_storage_disabled_503(client, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    res = await client.post("/voice/catch-up", headers={"X-Test-Uid": "u1"})
    assert res.status_code == 503


async def test_catchup_no_voiceprint_is_honest_zero_not_422(client, store, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    rid = _rid()
    store.add_recording("u1", rid, TURNS_AB, analysis=_unidentified_analysis())
    res = await client.post("/voice/catch-up", headers={"X-Test-Uid": "u1"})
    assert res.status_code == 200, res.text
    assert res.json() == {"checked": 0, "newly_identified": 0, "remaining": 0}


async def test_catchup_empty_store_is_zero(client, store, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    await store.write_voiceprint("u1", {"embedding": [1.0, 0.0, 0.0]})
    res = await client.post("/voice/catch-up", headers={"X-Test-Uid": "u1"})
    assert res.status_code == 200, res.text
    assert res.json() == {"checked": 0, "newly_identified": 0, "remaining": 0}


# ---------------------------------------------------------------------------
# The core behavior: match, skip-already-identified, honest no-match
# ---------------------------------------------------------------------------

async def test_catchup_matches_and_persists_reflected_in_growth(
    client, store, monkeypatch,
):
    e_you = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e_other = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    await store.write_voiceprint("u1", {"embedding": e_you.tolist()})
    rid = _rid()
    store.add_recording("u1", rid, TURNS_AB, analysis=_unidentified_analysis())
    _catchup_ready(monkeypatch, {"Speaker A": e_you, "Speaker B": e_other})

    res = await client.post("/voice/catch-up", headers={"X-Test-Uid": "u1"})
    assert res.status_code == 200, res.text
    assert res.json() == {"checked": 1, "newly_identified": 1, "remaining": 0}

    growth = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    body = growth.json()
    assert body["identified_recordings"] == 1
    assert body["points"][0]["recording_id"] == rid
    assert body["points"][0]["my_score"] == 80

    detail = await client.get(f"/recordings/{rid}", headers={"X-Test-Uid": "u1"})
    labels = detail.json()["speaker_labels"]
    assert labels["Speaker A"] == {
        "display_label": "You", "label_source": "enrolled",
    }
    assert labels["Speaker B"]["label_source"] == "generic"


async def test_catchup_skips_already_identified_recording(client, store, monkeypatch):
    e_you = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    await store.write_voiceprint("u1", {"embedding": e_you.tolist()})
    rid = _rid()
    store.add_recording("u1", rid, TURNS_AB, analysis=_identified_analysis())
    _catchup_ready(monkeypatch, {
        "Speaker A": e_you, "Speaker B": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    })

    res = await client.post("/voice/catch-up", headers={"X-Test-Uid": "u1"})
    assert res.status_code == 200, res.text
    # Already identified — no wasted work re-matching it.
    assert res.json() == {"checked": 0, "newly_identified": 0, "remaining": 0}


async def test_catchup_no_match_is_honest_zero(client, store, monkeypatch):
    # Neither speaker clears the match threshold — checked, but not identified.
    e_you = np.array([1.0, 0.0], dtype=np.float32)
    weak = np.array([0.1, np.sqrt(1 - 0.01)], dtype=np.float32)
    await store.write_voiceprint("u1", {"embedding": e_you.tolist()})
    rid = _rid()
    store.add_recording("u1", rid, TURNS_AB, analysis=_unidentified_analysis())
    _catchup_ready(monkeypatch, {"Speaker A": weak, "Speaker B": weak})

    res = await client.post("/voice/catch-up", headers={"X-Test-Uid": "u1"})
    assert res.status_code == 200, res.text
    assert res.json() == {"checked": 1, "newly_identified": 0, "remaining": 0}

    growth = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    assert growth.json()["identified_recordings"] == 0


async def test_catchup_unanalyzed_recording_is_not_checked(client, store, monkeypatch):
    # Stored but never analyzed (no analysis.json) — not a candidate at all.
    e_you = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    await store.write_voiceprint("u1", {"embedding": e_you.tolist()})
    rid = _rid()
    store.add_recording("u1", rid, TURNS_AB, analysis=None)
    _catchup_ready(monkeypatch, {"Speaker A": e_you})

    res = await client.post("/voice/catch-up", headers={"X-Test-Uid": "u1"})
    assert res.status_code == 200, res.text
    assert res.json() == {"checked": 0, "newly_identified": 0, "remaining": 0}


async def test_catchup_one_bad_recording_does_not_abort_the_batch(
    client, store, monkeypatch,
):
    # Three candidates sharing the same speaker/embedding wiring: one with NO
    # stored audio at all (corrupted/missing), two that genuinely match — the
    # bad one must not stop the other two from being processed.
    e_you = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e_other = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    await store.write_voiceprint("u1", {"embedding": e_you.tolist()})

    rid_bad = _rid()
    store.add_recording(
        "u1", rid_bad, TURNS_AB, audio=None, analysis=_unidentified_analysis(),
        created_at="2026-08-01T00:00:00+00:00",
    )
    rid_match_1 = _rid()
    store.add_recording(
        "u1", rid_match_1, TURNS_AB, analysis=_unidentified_analysis(),
        created_at="2026-08-02T00:00:00+00:00",
    )
    rid_match_2 = _rid()
    store.add_recording(
        "u1", rid_match_2, TURNS_AB, analysis=_unidentified_analysis(),
        created_at="2026-08-03T00:00:00+00:00",
    )
    _catchup_ready(monkeypatch, {"Speaker A": e_you, "Speaker B": e_other})

    res = await client.post("/voice/catch-up", headers={"X-Test-Uid": "u1"})
    assert res.status_code == 200, res.text
    body = res.json()
    # rid_bad has no audio → attempted (counted in "checked") but fails to
    # decode, honestly skipped; both real candidates still get processed.
    assert body["checked"] == 3
    assert body["newly_identified"] == 2

    growth = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    ids = {p["recording_id"] for p in growth.json()["points"]}
    assert rid_bad not in ids
    assert rid_match_1 in ids
    assert rid_match_2 in ids


# ---------------------------------------------------------------------------
# uid scoping
# ---------------------------------------------------------------------------

async def test_catchup_is_uid_scoped(client, store, monkeypatch):
    e_you = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    await store.write_voiceprint("u2", {"embedding": e_you.tolist()})
    rid = _rid()
    store.add_recording("u1", rid, TURNS_AB, analysis=_unidentified_analysis())
    _catchup_ready(monkeypatch, {"Speaker A": e_you})

    # u2 is enrolled but has no recordings of their own — u1's recording must
    # never be touched by u2's catch-up call.
    res = await client.post("/voice/catch-up", headers={"X-Test-Uid": "u2"})
    assert res.status_code == 200, res.text
    assert res.json() == {"checked": 0, "newly_identified": 0, "remaining": 0}

    growth = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    assert growth.json()["identified_recordings"] == 0


# ---------------------------------------------------------------------------
# A human's manual label always wins — never silently overwritten by a match
# ---------------------------------------------------------------------------

async def test_catchup_skips_a_speaker_with_an_existing_manual_label(
    client, store, monkeypatch,
):
    # Speaker A was manually re-tagged "Bob" by the user — a human override.
    # Catch-up's embedder still matches A to the enrolled voiceprint (a
    # perfectly plausible false-negative-turned-into-a-match scenario, or
    # just the user being wrong about who Bob is), but the match must NEVER
    # silently overwrite the human's explicit correction, even though the
    # manual overlay already hides the effect at read time — persisting it
    # anyway would corrupt the base label for when the manual tag is cleared.
    e_you = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e_other = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    await store.write_voiceprint("u1", {"embedding": e_you.tolist()})
    rid = _rid()
    store.add_recording(
        "u1", rid, TURNS_AB, analysis=_unidentified_analysis(),
        manual_speaker_labels={"Speaker A": "Bob"},
    )
    _catchup_ready(monkeypatch, {"Speaker A": e_you, "Speaker B": e_other})

    res = await client.post("/voice/catch-up", headers={"X-Test-Uid": "u1"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["checked"] == 1  # attempted — the match itself succeeded
    assert body["newly_identified"] == 0  # but never persisted or counted

    growth = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    assert growth.json()["identified_recordings"] == 0

    detail = await client.get(f"/recordings/{rid}", headers={"X-Test-Uid": "u1"})
    labels = detail.json()["speaker_labels"]
    # The manual label is untouched — still "Bob", never silently "You".
    assert labels["Speaker A"] == {"display_label": "Bob", "label_source": "manual"}


# ---------------------------------------------------------------------------
# Batch cap + remaining — catch-up must stay bounded per call
# ---------------------------------------------------------------------------

async def test_catchup_caps_the_batch_and_reports_remaining(
    client, store, monkeypatch,
):
    e_you = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e_other = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    await store.write_voiceprint("u1", {"embedding": e_you.tolist()})

    limit = voice_router._CATCHUP_BATCH_LIMIT
    total = limit + 3
    rids = []
    for i in range(total):
        rid = _rid()
        rids.append(rid)
        store.add_recording(
            "u1", rid, TURNS_AB, analysis=_unidentified_analysis(),
            created_at=f"2026-08-{i + 1:02d}T00:00:00+00:00",
        )
    _catchup_ready(monkeypatch, {"Speaker A": e_you, "Speaker B": e_other})

    res = await client.post("/voice/catch-up", headers={"X-Test-Uid": "u1"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["checked"] == limit
    assert body["newly_identified"] == limit
    assert body["remaining"] == 3

    growth = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    # The most-recent `limit` recordings were processed (list_recordings is
    # newest-first) — the 3 oldest are the ones left for "remaining".
    assert growth.json()["identified_recordings"] == limit
    identified_ids = {p["recording_id"] for p in growth.json()["points"]}
    most_recent = set(rids[-limit:])  # created_at ascending in insertion order
    assert identified_ids == most_recent


# ---------------------------------------------------------------------------
# Rate limiting — this route has its OWN, much tighter budget
# ---------------------------------------------------------------------------

async def test_catchup_has_its_own_tighter_rate_limit(client, store, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    monkeypatch.setattr(voice_router._catchup_rate_limiter, "limit", 2)
    voice_router._catchup_rate_limiter.reset()

    r1 = await client.post("/voice/catch-up", headers={"X-Test-Uid": "u1"})
    r2 = await client.post("/voice/catch-up", headers={"X-Test-Uid": "u1"})
    r3 = await client.post("/voice/catch-up", headers={"X-Test-Uid": "u1"})

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429
    assert "rate limit" in r3.json()["detail"].lower()
