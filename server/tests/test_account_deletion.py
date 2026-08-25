"""Tests for self-serve account deletion — ``DELETE /me``.

Nothing is mocked that could hide a real bug:

* the recordings tier runs the REAL ``recordings_store.RecordingsStore``
  against an in-memory fake GCS bucket (the ``_FakeBucket`` pattern
  test_therapist_links.py / test_voice_people.py already use), so the object
  layout — ``recordings/``, ``shared/``, ``voiceprints/``, ``uploads/``,
  ``jobs/``, ``therapist_links/``, ``therapist_patients/``,
  ``therapist_notes/`` — is exercised for real;
* the watch tier runs the REAL ``MemoryLiveSessionStore`` /
  ``MemoryPairingStore`` / ``MemoryTelemetryStore`` / ``MemoryBlobStore``;
* the relational tier runs the REAL aiosqlite database the suite already
  creates.

Only Firebase is faked — ``auth.delete_firebase_user`` is monkeypatched to
record its calls, which is also how "the auth user is deleted LAST" is proven.
"""

import json
import uuid
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

import account_deletion
import auth as auth_module
import main
import recordings_store
import speaker_id
import therapist_links
from main import app, init_db
from routers import account as account_router
from watch.app import WatchDeps
from watch.blobs import MemoryBlobStore
from watch.models import (
    Account,
    Capture,
    ConsentRecord,
    EnrollmentBaseline,
    Group,
    GroupInvite,
    GroupMember,
    LiveSession,
    Pairing,
    DeviceToken,
    SpeakerProfile,
    TelemetryEvent,
    VectorSubscription,
)
from watch.pairing_store import MemoryPairingStore
from watch.store import MemoryLiveSessionStore
from watch.telemetry_store import MemoryTelemetryStore

pytestmark = pytest.mark.anyio

# The account under test, a therapist who can see one of its sessions, and a
# bystander who must be provably untouched by any of it.
ME, THERAPIST, BYSTANDER = "user-me", "user-therapist", "user-bystander"

CONFIRM = {"confirm": "DELETE"}


def _h(uid):
    return {"X-Test-Uid": uid}


# ---------------------------------------------------------------------------
# Fake GCS bucket (same minimal surface RecordingsStore touches)
# ---------------------------------------------------------------------------

class _FakeBlob:
    def __init__(self, bucket, name):
        self._bucket, self.name = bucket, name

    def exists(self):
        return self.name in self._bucket.objects

    def download_as_bytes(self):
        return self._bucket.objects[self.name]

    def upload_from_string(self, data, content_type=None):
        self._bucket.objects[self.name] = (
            data.encode() if isinstance(data, str) else data
        )

    def delete(self):
        del self._bucket.objects[self.name]


class _FakeBucket:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def blob(self, name):
        return _FakeBlob(self, name)

    def list_blobs(self, prefix=""):
        return [
            _FakeBlob(self, n) for n in sorted(self.objects) if n.startswith(prefix)
        ]

    def names(self, prefix=""):
        return sorted(n for n in self.objects if n.startswith(prefix))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _profile(person_id, name):
    return {
        "version": 2, "embedding": [1.0, 0.0], "dim": 2, "enroll_count": 1,
        "model": "test", "created_at": _now(), "updated_at": _now(),
        "samples": [{"id": "s1", "embedding": [1.0, 0.0]}],
        "person_id": person_id, "display_name": name,
        "is_self": person_id == speaker_id.SELF_PERSON_ID,
    }


# ---------------------------------------------------------------------------
# Fixtures — a fully-populated account, wired into the real app
# ---------------------------------------------------------------------------

@pytest.fixture
def bucket():
    return _FakeBucket()


@pytest.fixture
def store(bucket):
    return recordings_store.RecordingsStore(bucket)


@pytest.fixture
def watch_deps():
    return WatchDeps(
        store=MemoryLiveSessionStore(),
        pairing_store=MemoryPairingStore(),
        telemetry_store=MemoryTelemetryStore(),
        blobs=MemoryBlobStore(),
    )


@pytest.fixture
def deleted_users(monkeypatch):
    """Records every ``auth.delete_firebase_user`` call, in order, alongside a
    marker for the data tiers so ordering is checkable."""
    calls: list[str] = []

    def _delete(uid):
        calls.append(uid)
        return True

    monkeypatch.setattr(auth_module, "delete_firebase_user", _delete)
    return calls


@pytest.fixture
async def client(store, watch_deps, deleted_users):
    await init_db()
    app.state.recordings_store = store
    previous_deps = getattr(app.state, "watch_deps", None)
    app.state.watch_deps = watch_deps
    account_router._delete_rate_limiter.reset()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as ac:
        yield ac
    del app.state.recordings_store
    app.state.watch_deps = previous_deps


async def _seed_recording(store, uid, *, rid=None, title="Kitchen talk"):
    """One stored episode with all four objects, written the way the real
    ingest path writes them."""
    rid = rid or str(uuid.uuid4())
    prefix = f"recordings/{uid}/{rid}/"
    store._bucket.blob(prefix + "meta.json").upload_from_string(
        json.dumps({
            "id": rid, "created_at": _now(), "filename": f"{title}.m4a",
            "title": title, "media_type": "audio", "duration_seconds": 12.0,
        }),
    )
    store._bucket.blob(prefix + "audio.m4a").upload_from_string(b"AUDIO")
    store._bucket.blob(prefix + "turns.json").upload_from_string(
        json.dumps([{"speaker": "Speaker A", "text": "hi"}]),
    )
    store._bucket.blob(prefix + "analysis.json").upload_from_string(
        json.dumps({"per_turn": [], "speaker_labels": {}}),
    )
    return rid


async def seed_everything(store, watch_deps, db):
    """Populate every tier for ME, plus the two other accounts that must
    survive. Returns the ids the assertions need."""
    # -- recordings + shares ------------------------------------------------
    shared_rid = await _seed_recording(store, ME, title="Session with therapist")
    private_rid = await _seed_recording(store, ME, title="Just me")
    await store.add_share(
        ME, shared_rid, recipient_uid=THERAPIST,
        recipient_email="t@example.com", owner_email="me@example.com",
    )

    # Something the BYSTANDER shared with ME (a grant pointing the other way).
    bystander_rid = await _seed_recording(store, BYSTANDER, title="Bystander clip")
    await store.add_share(
        BYSTANDER, bystander_rid, recipient_uid=ME,
        recipient_email="me@example.com", owner_email="b@example.com",
    )

    # -- voiceprints, uploads, jobs ----------------------------------------
    await store.write_voiceprint(ME, _profile(speaker_id.SELF_PERSON_ID, "You"))
    await store.write_voiceprint(ME, _profile("alex", "Alex"))
    await store.write_voiceprint(BYSTANDER, _profile(speaker_id.SELF_PERSON_ID, "You"))
    await store.write_upload_manifest(ME, "up1", {"chunks": 2})
    await store.write_upload_part(ME, "up1", 0, b"PART")
    await store.write_job_state(ME, "job1", {"status": "running"})

    # -- therapist links, both directions ----------------------------------
    # ME is a patient of THERAPIST...
    await store.write_therapist_link(ME, therapist_links.new_link(
        patient_uid=ME, patient_email="me@example.com",
        therapist_uid=THERAPIST, therapist_email="t@example.com",
    ))
    # ...and is ALSO the therapist for BYSTANDER.
    await store.write_therapist_link(BYSTANDER, therapist_links.new_link(
        patient_uid=BYSTANDER, patient_email="b@example.com",
        therapist_uid=ME, therapist_email="me@example.com",
    ))

    # -- notes -------------------------------------------------------------
    # The therapist's note ABOUT ME's shared session (must go with the session).
    await store.write_therapist_note(
        THERAPIST, shared_rid, {"text": "about my patient", "updated_at": _now()},
    )
    # The therapist's note about their OWN unrelated episode (must survive).
    await store.write_therapist_note(
        THERAPIST, "some-other-episode", {"text": "mine", "updated_at": _now()},
    )
    # A note ME wrote as a viewer (must go).
    await store.write_therapist_note(
        ME, bystander_rid, {"text": "my own note", "updated_at": _now()},
    )

    # -- watch tier --------------------------------------------------------
    ws = watch_deps.store
    await ws.put_live_session(LiveSession(
        id="ls-mine", owner_account=ME, started_at="2026-08-01T00:00:00Z",
        ended_at=None, status="analyzed", participants=[], vector_events=[],
        nudge_events=[],
    ))
    await ws.put_live_session(LiveSession(
        id="ls-theirs", owner_account=BYSTANDER, started_at="2026-08-02T00:00:00Z",
        ended_at=None, status="analyzed", participants=[], vector_events=[],
        nudge_events=[], shared_with=[ME],
    ))
    await ws.put_baseline(EnrollmentBaseline(
        account_id=ME, rms_db=-30.0, f0_median=120.0, updated_at=_now(),
    ))
    await ws.put_subscriptions(ME, [VectorSubscription(vector="interrupting")])
    await ws.put_account(Account(id=ME, email="me@example.com", created_at=_now(), updated_at=_now()))
    await ws.put_account(Account(id=BYSTANDER, email="b@example.com", created_at=_now(), updated_at=_now()))
    await ws.put_speaker_profile(SpeakerProfile(
        account_id=ME, embedding=[1.0, 0.0], dim=2, enroll_count=1,
        model="test", created_at=_now(), updated_at=_now(),
    ))
    await ws.put_capture(Capture(
        id="cap1", account_id=ME, captured_at=_now(), received_at=_now(),
        duration_s=3.0, status="stored", audio_uri="memory://captures/user-me/cap1.pcm",
    ))
    await watch_deps.blobs.put("captures/user-me/cap1.pcm", b"PCMPCM")
    # A pair group ME shares with BYSTANDER (survives, minus ME) and a solo
    # group only ME is in (deleted outright once ME leaves).
    await ws.put_group(Group(
        id="g-pair", kind="pair", name="Us", created_by=ME, created_at=_now(),
        members=[
            GroupMember(account_id=ME, joined_at=_now()),
            GroupMember(account_id=BYSTANDER, joined_at=_now()),
        ],
        invites=[GroupInvite(code="abc", invited_by=ME, created_at=_now(),
                             accepted_by=BYSTANDER, accepted_at=_now())],
        consents=[
            ConsentRecord(id="c1", participant_id=ME, kind="mutual_visibility",
                          attested_by=ME, confirmed=True, ts=_now()),
            ConsentRecord(id="c2", participant_id=BYSTANDER, kind="mutual_visibility",
                          attested_by=BYSTANDER, confirmed=True, ts=_now()),
        ],
    ))
    await ws.put_group(Group(
        id="g-solo", kind="team", name="Solo", created_by=ME, created_at=_now(),
        members=[GroupMember(account_id=ME, joined_at=_now())],
    ))

    ps = watch_deps.pairing_store
    ps.put_device_token(DeviceToken(
        token_hash="hash-mine", account_id=ME, created_at=_now(), pairing_id="p1",
    ))
    ps.put_device_token(DeviceToken(
        token_hash="hash-theirs", account_id=BYSTANDER, created_at=_now(), pairing_id="p2",
    ))
    await ps.create_pairing(Pairing(
        id="p1", code_hash="ch1", status="claimed", created_at=_now(),
        expires_at=_now(), claimed_account_id=ME, device_token="raw-token",
    ))
    await ps.set_failed_claim_record(ME, 2, _now())

    await watch_deps.telemetry_store.add_events([
        TelemetryEvent(
            id="tel-mine", device=f"phone:android:{ME}", app_version="1.18.0",
            level="info", tag="diagnostics", message="hi", ts=_now(),
            received_at=_now(), data={"uid": ME, "email": "me@example.com"},
        ),
        TelemetryEvent(
            id="tel-theirs", device=f"phone:android:{BYSTANDER}",
            app_version="1.18.0", level="info", tag="diagnostics", message="hi",
            ts=_now(), received_at=_now(), data={"uid": BYSTANDER},
        ),
        TelemetryEvent(
            id="tel-watch", device="watch-abc", app_version="1.0", level="info",
            tag="boot", message="hi", ts=_now(), received_at=_now(), data=None,
        ),
    ])

    # -- relational tier ---------------------------------------------------
    # The suite shares one SQLite file across tests, so clear this fixture's
    # own rows first (by id) — never the whole table, which other suites use.
    for sql, params in (
        ("DELETE FROM sessions WHERE id IN (?,?)", ("sess-mine", "sess-theirs")),
        ("DELETE FROM participants WHERE relationship_id IN (?,?)",
         ("rel-mine", "rel-theirs")),
        ("DELETE FROM voice_profiles WHERE relationship_id IN (?,?)",
         ("rel-mine", "rel-theirs")),
        ("DELETE FROM relationships WHERE id IN (?,?)", ("rel-mine", "rel-theirs")),
    ):
        await db.execute(sql, params)
    await db.execute(
        "INSERT INTO relationships (id, type, name, created_at, user_id) "
        "VALUES (?,?,?,?,?)", ("rel-mine", "partner", "Us", _now(), ME),
    )
    await db.execute(
        "INSERT INTO relationships (id, type, name, created_at, user_id) "
        "VALUES (?,?,?,?,?)", ("rel-theirs", "partner", "Them", _now(), BYSTANDER),
    )
    await db.execute(
        "INSERT INTO participants (id, relationship_id, role, display_name) "
        "VALUES (?,?,?,?)", ("p-a", "rel-mine", "self", "Me"),
    )
    await db.execute(
        "INSERT INTO voice_profiles (relationship_id, participant_id, pairs, updated_at) "
        "VALUES (?,?,?,?)", ("rel-mine", "p-a", "[]", _now()),
    )
    await db.execute(
        "INSERT INTO sessions (id, created_at, turns, metadata, relationship_id, user_id) "
        "VALUES (?,?,?,?,?,?)",
        ("sess-mine", _now(), "[]", "{}", "rel-mine", ME),
    )
    await db.execute(
        "INSERT INTO sessions (id, created_at, turns, metadata, relationship_id, user_id) "
        "VALUES (?,?,?,?,?,?)",
        ("sess-theirs", _now(), "[]", "{}", "rel-theirs", BYSTANDER),
    )
    await db.commit()

    return {
        "shared_rid": shared_rid,
        "private_rid": private_rid,
        "bystander_rid": bystander_rid,
    }


# ---------------------------------------------------------------------------
# The confirm guard
# ---------------------------------------------------------------------------

async def test_delete_me_requires_the_typed_confirmation(client):
    for body in (None, {}, {"confirm": "delete"}, {"confirm": "DELETE ME"}, {"confirm": ""}):
        kwargs = {} if body is None else {"json": body}
        # The per-IP budget is deliberately tiny and IS spent by refused
        # attempts too; reset between cases so this test measures the confirm
        # guard rather than the limiter (which has its own test below).
        account_router._delete_rate_limiter.reset()
        r = await client.request("DELETE", "/me", headers=_h(ME), **kwargs)
        assert r.status_code == 422, f"{body!r} should have been refused"


async def test_delete_me_requires_auth(client):
    """No override in play: the real dependency answers 401 without a token."""
    override = app.dependency_overrides.pop(auth_module.get_fresh_uid)
    try:
        r = await client.request("DELETE", "/me", json=CONFIRM)
        assert r.status_code == 401
    finally:
        app.dependency_overrides[auth_module.get_fresh_uid] = override


# ---------------------------------------------------------------------------
# The full walk
# ---------------------------------------------------------------------------

async def test_deletes_every_tier_and_leaves_other_accounts_alone(
    client, store, bucket, watch_deps, deleted_users,
):
    db = await main.get_db()
    try:
        ids = await seed_everything(store, watch_deps, db)
    finally:
        await db.close()

    r = await client.request("DELETE", "/me", headers=_h(ME), json=CONFIRM)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deleted"] is True
    assert body["firebase_user_deleted"] is True
    counts = body["counts"]
    # Every declared category is present, so a 0 means "none", not "unknown".
    assert set(counts) == set(account_deletion.COUNT_KEYS)
    assert counts["recordings"] == 2
    assert counts["shares_you_granted"] == 1
    assert counts["shares_granted_to_you"] == 1
    assert counts["notes_others_wrote_about_your_sessions"] == 1
    assert counts["notes_you_wrote"] == 1
    assert counts["voiceprints"] == 2
    assert counts["unfinished_uploads"] == 2      # manifest + one part
    assert counts["analysis_jobs"] == 1
    assert counts["therapist_links"] == 2         # as patient AND as therapist
    assert counts["live_sessions"] == 1
    assert counts["watch_captures"] == 1
    assert counts["watch_pairings"] == 2          # device token + claimed pairing
    assert counts["groups_left"] == 2
    assert counts["diagnostic_reports"] == 1
    assert counts["text_sessions"] == 1
    assert counts["relationships"] == 1

    # --- the bucket: nothing of ME's is left anywhere ---------------------
    assert bucket.names(f"recordings/{ME}/") == []
    assert bucket.names(f"voiceprints/{ME}/") == []
    assert bucket.names(f"uploads/{ME}/") == []
    assert bucket.names(f"jobs/{ME}/") == []
    assert bucket.names(f"shared/{ME}/") == []
    assert bucket.names(f"therapist_links/{ME}/") == []
    assert bucket.names(f"therapist_patients/{ME}/") == []
    assert bucket.names(f"therapist_notes/{ME}/") == []
    # ...and no other account is left holding a pointer to ME.
    assert bucket.names(f"shared/{THERAPIST}/") == []
    assert bucket.names(f"therapist_patients/{THERAPIST}/") == []
    assert bucket.names(f"therapist_links/{BYSTANDER}/") == []

    # --- the bystander is untouched ---------------------------------------
    assert await store.get_recording(BYSTANDER, ids["bystander_rid"]) is not None
    assert await store.read_voiceprint(BYSTANDER) is not None
    # The grant they made to ME is revoked on THEIR side too — no phantom share.
    meta = await store.get_recording(BYSTANDER, ids["bystander_rid"])
    assert meta.get("shares") == []

    # --- watch tier -------------------------------------------------------
    ws = watch_deps.store
    assert await ws.get_live_session("ls-mine") is None
    theirs = await ws.get_live_session("ls-theirs")
    assert theirs is not None and theirs.shared_with == []   # only the pointer went
    assert await ws.get_baseline(ME) is None
    assert await ws.has_subscriptions(ME) is False
    assert await ws.get_account(ME) is None
    assert await ws.get_account(BYSTANDER) is not None
    assert await ws.get_speaker_profile(ME) is None
    assert await ws.get_capture("cap1") is None
    assert await watch_deps.blobs.get("captures/user-me/cap1.pcm") is None
    assert await ws.get_group("g-solo") is None             # last member left
    pair = await ws.get_group("g-pair")
    assert pair is not None
    assert [m.account_id for m in pair.members] == [BYSTANDER]
    assert [c.participant_id for c in pair.consents] == [BYSTANDER]
    assert pair.invites == []                                # ME minted that one
    assert pair.created_by == ""                             # no dangling creator
    ps = watch_deps.pairing_store
    assert await ps.has_device_tokens_for_account(ME) is False
    assert await ps.has_device_tokens_for_account(BYSTANDER) is True
    assert await ps.get_pairing("p1") is None
    assert await ps.get_failed_claim_record(ME) is None
    remaining_telemetry = await watch_deps.telemetry_store.list_events(None, None, 50)
    assert {e.id for e in remaining_telemetry} == {"tel-theirs", "tel-watch"}

    # --- relational tier --------------------------------------------------
    db = await main.get_db()
    try:
        rows = await (await db.execute(
            "SELECT id FROM sessions WHERE id IN ('sess-mine','sess-theirs')",
        )).fetchall()
        assert [row[0] for row in rows] == ["sess-theirs"]
        rows = await (await db.execute(
            "SELECT id FROM relationships WHERE id IN ('rel-mine','rel-theirs')",
        )).fetchall()
        assert [row[0] for row in rows] == ["rel-theirs"]
        rows = await (await db.execute(
            "SELECT id FROM participants WHERE relationship_id = 'rel-mine'",
        )).fetchall()
        assert rows == []
        rows = await (await db.execute(
            "SELECT relationship_id FROM voice_profiles WHERE relationship_id = 'rel-mine'",
        )).fetchall()
        assert rows == []
    finally:
        await db.close()

    assert deleted_users == [ME]


async def test_shared_session_and_the_notes_about_it_go_together(
    client, store, watch_deps, deleted_users,
):
    """The documented shared-data rule: the patient owns the episode, so it is
    deleted along with the therapist's access AND the therapist's private notes
    about THAT episode — while the therapist's other notes survive."""
    db = await main.get_db()
    try:
        ids = await seed_everything(store, watch_deps, db)
    finally:
        await db.close()

    r = await client.request("DELETE", "/me", headers=_h(ME), json=CONFIRM)
    assert r.status_code == 200

    assert await store.get_recording(ME, ids["shared_rid"]) is None
    assert await store.find_share(THERAPIST, ids["shared_rid"]) is None
    assert await store.list_shared_with(THERAPIST) == []
    # Their note about the deleted episode is gone...
    assert await store.read_therapist_note(THERAPIST, ids["shared_rid"]) is None
    # ...but their own unrelated note is untouched.
    assert await store.read_therapist_note(THERAPIST, "some-other-episode") is not None


async def test_second_call_is_idempotent_and_reports_zeros(
    client, store, watch_deps, deleted_users, monkeypatch,
):
    db = await main.get_db()
    try:
        await seed_everything(store, watch_deps, db)
    finally:
        await db.close()

    first = await client.request("DELETE", "/me", headers=_h(ME), json=CONFIRM)
    assert first.status_code == 200

    # The Firebase user is gone now, so the second delete_user finds nothing.
    monkeypatch.setattr(auth_module, "delete_firebase_user", lambda uid: False)
    second = await client.request("DELETE", "/me", headers=_h(ME), json=CONFIRM)
    assert second.status_code == 200
    body = second.json()
    assert body["deleted"] is True
    assert body["firebase_user_deleted"] is False
    assert set(body["counts"].values()) == {0}


async def test_one_account_cannot_reach_another(
    client, store, bucket, watch_deps, deleted_users,
):
    """The endpoint takes no target: acting as the therapist deletes only the
    therapist, and the patient's data is provably still there."""
    db = await main.get_db()
    try:
        ids = await seed_everything(store, watch_deps, db)
    finally:
        await db.close()

    r = await client.request("DELETE", "/me", headers=_h(THERAPIST), json=CONFIRM)
    assert r.status_code == 200
    assert deleted_users == [THERAPIST]

    # ME's episodes, voiceprints and link survive untouched.
    assert await store.get_recording(ME, ids["shared_rid"]) is not None
    assert await store.get_recording(ME, ids["private_rid"]) is not None
    assert await store.read_voiceprint(ME) is not None
    assert bucket.names(f"recordings/{ME}/") != []
    # ...and ME's link to the now-deleted therapist is cleaned up from BOTH
    # sides rather than left dangling.
    assert await store.read_therapist_link(ME) is None


# ---------------------------------------------------------------------------
# Failure handling — the Firebase user is deleted LAST, and only on success
# ---------------------------------------------------------------------------

async def test_firebase_user_is_not_deleted_when_a_tier_fails(
    client, store, watch_deps, deleted_users, monkeypatch,
):
    db = await main.get_db()
    try:
        await seed_everything(store, watch_deps, db)
    finally:
        await db.close()

    async def _boom(uid):
        raise RuntimeError("simulated GCS outage")

    monkeypatch.setattr(store, "delete_all_recordings", _boom)

    r = await client.request("DELETE", "/me", headers=_h(ME), json=CONFIRM)
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert detail["failed"] == ["recordings: RuntimeError"]
    # Other tiers still ran and are reported honestly...
    assert detail["counts"]["live_sessions"] == 1
    # ...and the account itself is still there, so the user can retry.
    assert deleted_users == []


async def test_firebase_delete_runs_after_every_tier(
    client, store, watch_deps, monkeypatch,
):
    """Ordering, proven rather than asserted from reading: the fake records
    when it ran, and by then every tier must already be empty."""
    db = await main.get_db()
    try:
        await seed_everything(store, watch_deps, db)
    finally:
        await db.close()

    observed: dict = {}

    def _delete(uid):
        observed["recordings_left"] = len(
            [n for n in store._bucket.objects if n.startswith(f"recordings/{uid}/")]
        )
        observed["live_sessions_left"] = len(watch_deps.store._live_sessions)
        return True

    monkeypatch.setattr(auth_module, "delete_firebase_user", _delete)

    r = await client.request("DELETE", "/me", headers=_h(ME), json=CONFIRM)
    assert r.status_code == 200
    assert observed == {"recordings_left": 0, "live_sessions_left": 1}  # only ls-theirs


async def test_data_survives_a_failing_firebase_delete_as_a_reported_error(
    client, store, watch_deps, monkeypatch,
):
    def _boom(uid):
        raise RuntimeError("admin sdk unavailable")

    monkeypatch.setattr(auth_module, "delete_firebase_user", _boom)
    r = await client.request("DELETE", "/me", headers=_h(ME), json=CONFIRM)
    assert r.status_code == 500
    assert r.json()["detail"]["failed"] == ["firebase_auth"]


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

async def test_delete_me_has_its_own_tight_rate_limit(client, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(account_router._delete_rate_limiter, "limit", 2)
    account_router._delete_rate_limiter.reset()
    try:
        codes = [
            (await client.request("DELETE", "/me", headers=_h(ME), json=CONFIRM)).status_code
            for _ in range(3)
        ]
    finally:
        account_router._delete_rate_limiter.reset()
    assert codes[:2] == [200, 200]
    assert codes[2] == 429


# ---------------------------------------------------------------------------
# The fresh-token gate (auth.get_fresh_uid), exercised for real
# ---------------------------------------------------------------------------

async def test_fresh_token_dependency_accepts_a_just_issued_token(monkeypatch):
    import time as _time

    monkeypatch.setattr(
        auth_module, "verify_id_token_claims",
        lambda token: {"uid": "u1", "iat": _time.time() - 5},
    )
    assert await auth_module.get_fresh_uid("Bearer tok") == "u1"


async def test_fresh_token_dependency_rejects_a_stale_but_valid_token(monkeypatch):
    import time as _time

    monkeypatch.setattr(
        auth_module, "verify_id_token_claims",
        lambda token: {"uid": "u1", "iat": _time.time() - 3600},
    )
    with pytest.raises(Exception) as excinfo:
        await auth_module.get_fresh_uid("Bearer tok")
    assert excinfo.value.status_code == 401
    assert "freshly issued" in excinfo.value.detail


async def test_fresh_token_dependency_fails_closed_without_a_timestamp(monkeypatch):
    monkeypatch.setattr(
        auth_module, "verify_id_token_claims", lambda token: {"uid": "u1"},
    )
    with pytest.raises(Exception) as excinfo:
        await auth_module.get_fresh_uid("Bearer tok")
    assert excinfo.value.status_code == 401


async def test_fresh_token_dependency_rejects_a_missing_header():
    with pytest.raises(Exception) as excinfo:
        await auth_module.get_fresh_uid("")
    assert excinfo.value.status_code == 401


def test_token_age_falls_back_to_auth_time_and_clamps_skew():
    assert auth_module.token_age_seconds({"auth_time": 100}, now=160) == 60
    assert auth_module.token_age_seconds({"iat": 100, "auth_time": 0}, now=110) == 10
    assert auth_module.token_age_seconds({"iat": 200}, now=100) == 0  # clock skew
    assert auth_module.token_age_seconds({}) is None
    assert auth_module.token_age_seconds({"iat": "nonsense"}) is None


# ---------------------------------------------------------------------------
# The store's own bulk-delete surface, against the real GCS layout
# ---------------------------------------------------------------------------

async def test_store_bulk_deletes_are_uid_scoped(store, bucket):
    rid = await _seed_recording(store, ME)
    await _seed_recording(store, BYSTANDER)
    await store.write_voiceprint(ME, _profile(speaker_id.SELF_PERSON_ID, "You"))
    await store.write_voiceprint(BYSTANDER, _profile(speaker_id.SELF_PERSON_ID, "You"))
    await store.write_upload_manifest(ME, "up1", {"chunks": 1})
    await store.write_job_state(ME, "j1", {"status": "done"})

    assert await store.delete_all_recordings(ME) == 1
    assert await store.delete_all_voiceprints(ME) == 1
    assert await store.delete_all_uploads(ME) == 1
    assert await store.delete_all_jobs(ME) == 1

    assert bucket.names(f"recordings/{ME}/") == []
    assert bucket.names(f"voiceprints/{ME}/") == []
    assert bucket.names(f"recordings/{BYSTANDER}/") != []
    assert bucket.names(f"voiceprints/{BYSTANDER}/") != []
    # Idempotent second pass.
    assert await store.delete_all_recordings(ME) == 0
    assert await store.delete_all_voiceprints(ME) == 0
    assert rid  # the id was real


async def test_store_refuses_an_unscoped_bulk_prefix(store):
    for bad in ("", "recordings/", "recordings", "/"):
        with pytest.raises(ValueError):
            store._delete_prefix_sync(bad)


async def test_share_recipient_index_survives_being_read_before_deletion(store):
    rid = await _seed_recording(store, ME)
    await store.add_share(
        ME, rid, recipient_uid=THERAPIST, recipient_email="t@x", owner_email="m@x",
    )
    assert await store.list_recording_share_recipients(ME) == {rid: [THERAPIST]}
    grants = await store.list_received_share_grants(THERAPIST)
    assert [(g["owner_uid"], g["recording_id"]) for g in grants] == [(ME, rid)]
    assert await store.list_received_share_grants(BYSTANDER) == []


async def test_delete_account_data_runs_with_no_stores_configured():
    """Storage disabled everywhere: nothing to delete, reported as zeros rather
    than an error — the same honest-degradation posture the recordings
    endpoints already take."""
    summary = await account_deletion.delete_account_data("u1")
    assert summary.ok
    assert summary.total == 0
    assert set(summary.counts) == set(account_deletion.COUNT_KEYS)
