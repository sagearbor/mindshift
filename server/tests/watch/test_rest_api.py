# Ported from gauge@2157433 server/tests/test_rest_api.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B5): Episode -> LiveSession, /episodes* -> /live-sessions*,
# server.main.create_app -> watch.testing.create_watch_test_app (keyword-only
# assembly, no create_app-style positional store/transcriber/llm args), and
# GAUGE_ALLOW_LEGACY_ACCOUNT env var -> the explicit allow_legacy=True kwarg
# (testing.py takes no env vars at all -- see its own docstring). Wire field
# names (e.g. PeriodStats' `episodes` count) are UNCHANGED per the locked
# rename map -- only the "episode" TYPE name and its HTTP paths move.
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from server.tests.watch.test_auth import StubVerifier
from server.tests.watch.test_vectors import pcm
from watch.models import LiveSession, Participant
from watch.store import MemoryLiveSessionStore
from watch.testing import create_watch_test_app

TOKENS = {
    "tok-a": {"sub": "uid-a", "email": "a@example.com"},
    "tok-b": {"sub": "uid-b", "email": "b@example.com"},
}
AUTH_A = {"Authorization": "Bearer tok-a"}
AUTH_B = {"Authorization": "Bearer tok-b"}


def _ls(id, owner, started="2026-07-30T00:00:00Z", shared=(), participants=None):
    return LiveSession(
        id=id, owner_account=owner, started_at=started, ended_at=None, status="captured",
        participants=participants or [Participant(id="self", role="self", speaker_label="You")],
        vector_events=[], nudge_events=[], shared_with=list(shared),
    )


def _client(store=None):
    store = store or MemoryLiveSessionStore()
    return store, TestClient(create_watch_test_app(store=store, allow_legacy=True))


def _authed_client():
    """Same shape as test_claim_legacy.py's _client -- a StubVerifier-backed app so
    DELETE /live-sessions/{id}'s strict_auth (full-auth-only) gate has a real bearer
    identity to test against, plus the legacy ?account= ladder step for the
    401-on-legacy-delete assertion."""
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(
        store=store, verifier=StubVerifier(TOKENS), allow_legacy=True,
    ))
    return store, client


def wav_bytes(pcm_bytes: bytes) -> bytes:
    """Wrap raw PCM16 mono 16k in a minimal 44-byte WAV header."""
    return b"RIFF" + b"\x00" * 40 + pcm_bytes


# ----------------------------------------------------------------- live-sessions --

def test_list_live_sessions_owner_and_shared():
    store, client = _client()
    import asyncio
    asyncio.run(store.put_live_session(_ls("e1", "alice")))
    asyncio.run(store.put_live_session(_ls("e2", "bob", started="2026-07-31T00:00:00Z", shared=("alice",))))
    asyncio.run(store.put_live_session(_ls("e3", "carol")))

    resp = client.get("/live-sessions", params={"account": "alice"})
    assert resp.status_code == 200
    assert [e["id"] for e in resp.json()] == ["e2", "e1"]


def test_get_live_session_by_owner():
    store, client = _client()
    import asyncio
    asyncio.run(store.put_live_session(_ls("e1", "alice")))

    resp = client.get("/live-sessions/e1", params={"account": "alice"})
    assert resp.status_code == 200
    assert resp.json()["id"] == "e1"


def test_get_live_session_response_never_contains_pcm_b64():
    store, client = _client()
    import asyncio
    ls = _ls("e1", "alice")
    ls.pcm_b64 = "c29tZS1hdWRpby1ieXRlcw=="
    asyncio.run(store.put_live_session(ls))

    resp = client.get("/live-sessions/e1", params={"account": "alice"})
    assert resp.status_code == 200
    assert "pcm_b64" not in resp.json()


def test_get_live_session_unknown_404():
    _, client = _client()
    resp = client.get("/live-sessions/nope", params={"account": "alice"})
    assert resp.status_code == 404


def test_get_live_session_by_non_owner_non_shared_403():
    store, client = _client()
    import asyncio
    asyncio.run(store.put_live_session(_ls("e1", "alice")))

    resp = client.get("/live-sessions/e1", params={"account": "mallory"})
    assert resp.status_code == 403


def test_delete_live_session_owner_only_and_gone_for_shared_viewer():
    import asyncio
    store, client = _authed_client()      # StubVerifier: tok-a->uid-a, tok-b->uid-b
    asyncio.run(store.put_live_session(_ls("e1", "uid-a", shared=["uid-b"])))
    # the shared-with viewer cannot delete someone else's behavior record
    resp = client.delete("/live-sessions/e1", headers=AUTH_B)
    assert resp.status_code == 403
    assert resp.json()["detail"] == "only the episode owner may do this"
    # the owner can — and it disappears for EVERYONE (D4: live queries, no tombstones)
    assert client.delete("/live-sessions/e1", headers=AUTH_A).status_code == 204
    assert client.get("/live-sessions", headers=AUTH_A).json() == []
    assert client.get("/live-sessions", headers=AUTH_B).json() == []          # shared_with view gone
    assert client.get("/live-sessions/e1", headers=AUTH_B).status_code == 404  # honest 404, not 403


def test_delete_live_session_unknown_is_404_and_legacy_is_401():
    import asyncio
    store, client = _authed_client()
    assert client.delete("/live-sessions/nope", headers=AUTH_A).status_code == 404
    asyncio.run(store.put_live_session(_ls("e1", "default")))
    # destructive surface is full-auth: the legacy param may read, never delete
    assert client.delete("/live-sessions/e1", params={"account": "default"}).status_code == 401
    assert asyncio.run(store.get_live_session("e1")) is not None


def test_deleted_live_session_leaves_aggregates_naturally():
    import asyncio
    store, client = _authed_client()
    started = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    asyncio.run(store.put_live_session(_ls("e1", "uid-a", started=started)))
    assert client.get("/me/standing", headers=AUTH_A).json()["current"]["episodes"] == 1
    client.delete("/live-sessions/e1", headers=AUTH_A)
    assert client.get("/me/standing", headers=AUTH_A).json()["current"]["episodes"] == 0


# --------------------------------------------------------------------- labels --

def test_label_participant_happy_path():
    store, client = _client()
    import asyncio
    asyncio.run(store.put_live_session(_ls("e1", "alice", participants=[
        Participant(id="p2", role="other", speaker_label="Speaker 2"),
    ])))

    resp = client.post(
        "/live-sessions/e1/labels",
        params={"account": "alice"},
        json={"participant_id": "p2", "display_name": "Jordan", "attested": True},
    )
    assert resp.status_code == 200

    ls = asyncio.run(store.get_live_session("e1"))
    participant = next(p for p in ls.participants if p.id == "p2")
    assert participant.display_name == "Jordan"
    assert len(ls.consents) == 1
    rec = ls.consents[0]
    assert rec.kind == "labeling" and rec.confirmed is False and rec.attested_by == "alice"
    assert rec.ts  # server-generated, non-empty


def test_label_without_attested_true_422():
    store, client = _client()
    import asyncio
    asyncio.run(store.put_live_session(_ls("e1", "alice", participants=[
        Participant(id="p2", role="other", speaker_label="Speaker 2"),
    ])))

    resp = client.post(
        "/live-sessions/e1/labels",
        params={"account": "alice"},
        json={"participant_id": "p2", "display_name": "Jordan", "attested": False},
    )
    assert resp.status_code == 422


def test_label_unknown_live_session_404():
    _, client = _client()
    resp = client.post(
        "/live-sessions/nope/labels",
        params={"account": "alice"},
        json={"participant_id": "p2", "display_name": "Jordan", "attested": True},
    )
    assert resp.status_code == 404


def test_label_unknown_participant_404():
    store, client = _client()
    import asyncio
    asyncio.run(store.put_live_session(_ls("e1", "alice")))

    resp = client.post(
        "/live-sessions/e1/labels",
        params={"account": "alice"},
        json={"participant_id": "ghost", "display_name": "Jordan", "attested": True},
    )
    assert resp.status_code == 404


def test_label_by_non_owner_403():
    store, client = _client()
    import asyncio
    asyncio.run(store.put_live_session(_ls("e1", "alice", participants=[
        Participant(id="p2", role="other", speaker_label="Speaker 2"),
    ])))

    resp = client.post(
        "/live-sessions/e1/labels",
        params={"account": "mallory"},
        json={"participant_id": "p2", "display_name": "Jordan", "attested": True},
    )
    assert resp.status_code == 403


def test_label_attested_truthy_string_422():
    # "yes" must NOT be coerced to True — attested is a strict boolean gate,
    # not "anything truthy".
    store, client = _client()
    import asyncio
    asyncio.run(store.put_live_session(_ls("e1", "alice", participants=[
        Participant(id="p2", role="other", speaker_label="Speaker 2"),
    ])))

    resp = client.post(
        "/live-sessions/e1/labels",
        params={"account": "alice"},
        json={"participant_id": "p2", "display_name": "Jordan", "attested": "yes"},
    )
    assert resp.status_code == 422


def test_label_attested_int_one_422():
    store, client = _client()
    import asyncio
    asyncio.run(store.put_live_session(_ls("e1", "alice", participants=[
        Participant(id="p2", role="other", speaker_label="Speaker 2"),
    ])))

    resp = client.post(
        "/live-sessions/e1/labels",
        params={"account": "alice"},
        json={"participant_id": "p2", "display_name": "Jordan", "attested": 1},
    )
    assert resp.status_code == 422


def test_label_attested_literal_true_200():
    store, client = _client()
    import asyncio
    asyncio.run(store.put_live_session(_ls("e1", "alice", participants=[
        Participant(id="p2", role="other", speaker_label="Speaker 2"),
    ])))

    resp = client.post(
        "/live-sessions/e1/labels",
        params={"account": "alice"},
        json={"participant_id": "p2", "display_name": "Jordan", "attested": True},
    )
    assert resp.status_code == 200


# --------------------------------------------------------------------- share --

def test_share_happy_path():
    store, client = _client()
    import asyncio
    asyncio.run(store.put_live_session(_ls("e1", "alice")))

    resp = client.post("/live-sessions/e1/share", params={"account": "alice"}, json={"with_account": "bob"})
    assert resp.status_code == 200

    ls = asyncio.run(store.get_live_session("e1"))
    assert ls.shared_with == ["bob"]
    assert any(c.kind == "sharing" and c.attested_by == "alice" for c in ls.consents)


def test_share_by_non_owner_403():
    store, client = _client()
    import asyncio
    asyncio.run(store.put_live_session(_ls("e1", "alice")))

    resp = client.post("/live-sessions/e1/share", params={"account": "mallory"}, json={"with_account": "bob"})
    assert resp.status_code == 403


def test_share_idempotent():
    store, client = _client()
    import asyncio
    asyncio.run(store.put_live_session(_ls("e1", "alice")))

    client.post("/live-sessions/e1/share", params={"account": "alice"}, json={"with_account": "bob"})
    client.post("/live-sessions/e1/share", params={"account": "alice"}, json={"with_account": "bob"})

    ls = asyncio.run(store.get_live_session("e1"))
    assert ls.shared_with == ["bob"]


# --------------------------------------------------------------------- settings --

def test_get_settings_defaults():
    _, client = _client()
    resp = client.get("/settings/vectors", params={"account": "alice"})
    assert resp.status_code == 200
    subs = resp.json()
    assert {s["vector"] for s in subs} == {"yelling", "aggressive_tone", "interrupting", "airtime", "hr_spike"}


def test_put_settings_roundtrip_changes_channel():
    _, client = _client()
    subs = client.get("/settings/vectors", params={"account": "alice"}).json()
    for s in subs:
        if s["vector"] == "yelling":
            s["channel"] = "B"

    put_resp = client.put("/settings/vectors", params={"account": "alice"}, json=subs)
    assert put_resp.status_code == 200

    get_resp = client.get("/settings/vectors", params={"account": "alice"})
    yelling = next(s for s in get_resp.json() if s["vector"] == "yelling")
    assert yelling["channel"] == "B"


# --------------------------------------------------------------------- enroll --

def test_enroll_1_second_clip_422():
    _, client = _client()
    resp = client.post("/enroll", params={"account": "alice"}, content=pcm(0.2, seconds=1.0))
    assert resp.status_code == 422


def test_enroll_raw_pcm_happy_path():
    store, client = _client()
    resp = client.post("/enroll", params={"account": "alice"}, content=pcm(0.2, seconds=3.0))
    assert resp.status_code == 200
    body = resp.json()
    assert body["account_id"] == "alice"
    assert isinstance(body["rms_db"], float)
    assert isinstance(body["f0_median"], float)
    assert body["updated_at"]

    import asyncio
    baseline = asyncio.run(store.get_baseline("alice"))
    assert baseline is not None and baseline.rms_db == body["rms_db"]


def test_enroll_with_wav_header_works():
    _, client = _client()
    resp = client.post("/enroll", params={"account": "alice"}, content=wav_bytes(pcm(0.2, seconds=3.0)))
    assert resp.status_code == 200


def test_enroll_all_zero_clip_422():
    # A fully-silent (muted mic) clip must never be enrolled as a baseline —
    # it would poison every subsequent yelling comparison for this account.
    _, client = _client()
    resp = client.post("/enroll", params={"account": "alice"}, content=b"\x00" * (16000 * 2 * 4))
    assert resp.status_code == 422


def test_enroll_near_silent_clip_422():
    # Amplitude below VectorEngine's SILENCE_FLOOR_DBFS (-45 dBFS), but not
    # literally all zero -- must still be rejected, same as true silence.
    _, client = _client()
    resp = client.post("/enroll", params={"account": "alice"}, content=pcm(0.001, seconds=4.0))
    assert resp.status_code == 422


def test_enroll_normal_amplitude_still_200_with_finite_rms():
    _, client = _client()
    resp = client.post("/enroll", params={"account": "alice"}, content=pcm(0.2, seconds=4.0))
    assert resp.status_code == 200
    body = resp.json()
    assert body["rms_db"] is not None
    assert body["rms_db"] > -1000.0  # finite, not -inf
