"""Guest mode on the server — anonymous tokens, the ``is_guest`` flag, the
cost quota, and deleting a guest account.

"Continue as guest" (Firebase Anonymous Auth) mints a real, signed ID token
with a uid, no email, and ``firebase.sign_in_provider == "anonymous"``. The
four things that must be true, each proved here rather than asserted in a
doc:

1. **An anonymous token authenticates.** It is a verified Firebase identity;
   nothing rejects it for having no email.
2. **The account record says so.** ``is_guest`` / ``provider="anonymous"`` on
   the watch-domain accounts row and on ``GET /me`` — and a guest who LINKS a
   real credential stops being one on the next request, while a paired
   WATCH's device token (which cannot know the provider) leaves the flag
   alone instead of guessing.
3. **The quota bites.** Sessions-per-day and minutes-per-session, on the
   WebSocket that spends the money and on the ingest that buys the analysis,
   with the same sentence in both places.
4. **A guest can delete everything.** ``DELETE /me`` is uid-scoped and never
   looks at how that uid signed in, so this is a regression guard on a
   Play-required promise, not new behaviour.

The whole file runs keyless: ``server/conftest.py`` maps ``tok-guest`` to the
uid ``guest-user`` with an anonymous provider, and ``X-Test-Guest: 1`` makes a
REST request act as a guest.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from starlette.websockets import WebSocketDisconnect

import auth as auth_module
import guest_quota
import main
from main import app, init_db
from routers import sessions as sessions_router

# Siblings, imported by bare name — see pyproject's pythonpath note.
from test_audio_pipeline import FakeTranscriber, FakeTTS, MOCK_LLM_JSON
from test_sessions_live import FakeLiveStore, SESSION_ID, TURNS

pytestmark = pytest.mark.anyio

GUEST_HEADERS = {"X-Test-Uid": "guest-user", "X-Test-Guest": "1"}
REAL_HEADERS = {"X-Test-Uid": "real-user"}


# ---------------------------------------------------------------------------
# 1. Claims → identity
# ---------------------------------------------------------------------------

class TestAnonymousClaims:
    def test_anonymous_provider_reads_as_guest(self):
        identity = auth_module.identity_from_claims(
            {"uid": "g1", "firebase": {"sign_in_provider": "anonymous"}}
        )
        assert identity.uid == "g1"
        assert identity.email is None
        assert identity.is_guest is True

    def test_password_account_is_not_a_guest(self):
        identity = auth_module.identity_from_claims(
            {"uid": "u1", "email": "a@b.test",
             "firebase": {"sign_in_provider": "password"}}
        )
        assert identity.is_guest is False
        assert identity.email == "a@b.test"

    @pytest.mark.parametrize(
        "claims",
        [
            {"uid": "u1"},                          # no firebase block at all
            {"uid": "u1", "firebase": None},        # present but null
            {"uid": "u1", "firebase": "anonymous"}, # wrong type entirely
            {"uid": "u1", "firebase": {}},          # no provider key
            {"uid": "u1", "firebase": {"sign_in_provider": 7}},  # wrong type
        ],
    )
    def test_unreadable_provider_is_never_a_guest(self, claims):
        """Fail SAFE, in the direction that costs the user nothing: an
        unreadable provider claim must never apply the guest quota to what
        might be a paying, signed-up account."""
        assert auth_module.identity_from_claims(claims).is_guest is False

    async def test_anonymous_token_authenticates(self, monkeypatch):
        """No email, still a real identity — the dependency accepts it."""
        monkeypatch.delitem(app.dependency_overrides,
                            auth_module.get_current_identity, raising=False)
        identity = await auth_module.get_current_identity("Bearer tok-guest")
        assert identity.uid == "guest-user"
        assert identity.is_guest is True

    async def test_missing_token_is_still_401(self, monkeypatch):
        monkeypatch.delitem(app.dependency_overrides,
                            auth_module.get_current_identity, raising=False)
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            await auth_module.get_current_identity("")
        assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# 2. The account record
# ---------------------------------------------------------------------------

class TestAccountRecord:
    def _client(self, verifier_claims):
        from watch.testing import create_watch_test_app
        from watch.auth import InvalidToken
        from watch.store import MemoryLiveSessionStore

        class StubVerifier:
            def verify(self, token):
                try:
                    return verifier_claims[token]
                except KeyError:
                    raise InvalidToken("bad token")

        store = MemoryLiveSessionStore()
        app_ = create_watch_test_app(
            store=store, verifier=StubVerifier(), allow_legacy=True,
        )
        return store, TestClient(app_)

    def test_guest_account_is_provisioned_as_a_guest(self):
        store, client = self._client({
            "anon": {"sub": "guest-1", "email": None,
                     "sign_in_provider": "anonymous"},
        })
        body = client.get("/me", headers={"Authorization": "Bearer anon"}).json()
        assert body["is_guest"] is True
        assert body["sign_in_provider"] == "anonymous"

        account = asyncio.run(store.get_account("guest-1"))
        assert account.is_guest is True
        assert account.provider == "anonymous"

    def test_real_account_is_not_marked_a_guest(self):
        store, client = self._client({
            "pw": {"sub": "u-1", "email": "a@b.test",
                   "sign_in_provider": "password"},
        })
        assert client.get("/me", headers={"Authorization": "Bearer pw"}).json()[
            "is_guest"] is False
        assert asyncio.run(store.get_account("u-1")).is_guest is False

    def test_linking_an_account_clears_the_guest_flag(self):
        """The upgrade path: the SAME uid comes back with a real provider
        (the phone linked an email/password or Google credential onto the
        anonymous user). The stored account must stop saying guest, or the
        quota would follow them forever."""
        claims = {
            "anon": {"sub": "same-uid", "email": None,
                     "sign_in_provider": "anonymous"},
            "linked": {"sub": "same-uid", "email": "new@b.test",
                       "sign_in_provider": "password"},
        }
        store, client = self._client(claims)
        client.get("/me", headers={"Authorization": "Bearer anon"})
        assert asyncio.run(store.get_account("same-uid")).is_guest is True

        client.get("/me", headers={"Authorization": "Bearer linked"})
        account = asyncio.run(store.get_account("same-uid"))
        assert account.is_guest is False
        assert account.provider == "google"
        assert account.email == "new@b.test"

    def test_a_provider_less_verifier_leaves_the_flag_alone(self):
        """A paired WATCH authenticates with an opaque device token that
        carries no sign-in provider. It must not silently promote a guest to
        a real account (guest mode has to survive pairing a watch) — nor
        demote a real one."""
        claims = {
            "anon": {"sub": "same-uid", "email": None,
                     "sign_in_provider": "anonymous"},
            # Exactly what DeviceTokenVerifier returns: no provider key.
            "device": {"sub": "same-uid", "email": None},
        }
        store, client = self._client(claims)
        client.get("/me", headers={"Authorization": "Bearer anon"})
        client.get("/me", headers={"Authorization": "Bearer device"})
        account = asyncio.run(store.get_account("same-uid"))
        assert account.is_guest is True
        assert account.provider == "anonymous"


# ---------------------------------------------------------------------------
# 3a. The quota primitives
# ---------------------------------------------------------------------------

class TestQuotaPrimitives:
    async def test_counts_distinct_sessions_up_to_the_limit(self, monkeypatch):
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSIONS_PER_DAY", 3)
        counter = guest_quota.GuestSessionCounter()
        for i in range(3):
            assert await counter.admit("g1", f"s{i}") is True
        assert await counter.admit("g1", "s3") is False
        assert counter.used("g1") == 3

    async def test_the_same_session_never_double_charges(self, monkeypatch):
        """A WS reconnect mid-conversation, then the POST that stores the very
        same session, must cost ONE of the day's three — not three."""
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSIONS_PER_DAY", 3)
        counter = guest_quota.GuestSessionCounter()
        for _ in range(5):
            assert await counter.admit("g1", "same-session") is True
        assert counter.used("g1") == 1

    async def test_guests_do_not_share_an_allowance(self, monkeypatch):
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSIONS_PER_DAY", 1)
        counter = guest_quota.GuestSessionCounter()
        assert await counter.admit("g1", "a") is True
        assert await counter.admit("g2", "a") is True
        assert await counter.admit("g1", "b") is False

    async def test_a_new_utc_day_restores_the_allowance(self, monkeypatch):
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSIONS_PER_DAY", 1)
        counter = guest_quota.GuestSessionCounter()
        day1 = datetime(2026, 9, 20, 23, 59, tzinfo=timezone.utc)
        day2 = day1 + timedelta(minutes=2)
        assert await counter.admit("g1", "a", now=day1) is True
        assert await counter.admit("g1", "b", now=day1) is False
        assert await counter.admit("g1", "b", now=day2) is True
        assert counter.used("g1", now=day1) == 0  # the old bucket is gone

    @pytest.mark.parametrize(
        "minutes,over",
        [(1, False), (9.9, False), (10, False), (10.5, True), (60, True)],
    )
    def test_session_time_cap(self, minutes, over, monkeypatch):
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSION_MIN", 10)
        start = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
        end = start + timedelta(minutes=minutes)
        assert guest_quota.session_over_time_cap(
            start.isoformat(), end.isoformat()) is over

    @pytest.mark.parametrize(
        "started,ended",
        [
            ("not-a-time", "2026-09-20T12:00:00+00:00"),
            ("2026-09-20T12:00:00+00:00", "2026-09-20T11:00:00+00:00"),
        ],
    )
    def test_a_nonsense_span_is_not_treated_as_over_limit(self, started, ended):
        """A client reporting a broken span is a bug to notice, not a user to
        cut off — and the socket has already bounded the real spend."""
        assert guest_quota.session_over_time_cap(started, ended) is False

    def test_a_broken_env_value_falls_back_instead_of_uncapping(self, monkeypatch):
        monkeypatch.setenv("MINDSHIFT_GUEST_MAX_SESSIONS_PER_DAY", "lots")
        assert guest_quota._positive_int_env(
            "MINDSHIFT_GUEST_MAX_SESSIONS_PER_DAY", 3) == 3
        monkeypatch.setenv("MINDSHIFT_GUEST_MAX_SESSIONS_PER_DAY", "0")
        assert guest_quota._positive_int_env(
            "MINDSHIFT_GUEST_MAX_SESSIONS_PER_DAY", 3) == 3
        monkeypatch.setenv("MINDSHIFT_GUEST_MAX_SESSIONS_PER_DAY", "7")
        assert guest_quota._positive_int_env(
            "MINDSHIFT_GUEST_MAX_SESSIONS_PER_DAY", 3) == 7

    def test_the_defaults_are_the_documented_ones(self):
        """The numbers in docs/decisions and the Play packs quote these."""
        assert guest_quota.GUEST_MAX_SESSIONS_PER_DAY == 3
        assert guest_quota.GUEST_MAX_SESSION_MIN == 10


# ---------------------------------------------------------------------------
# 3b. The quota on the WebSocket that actually spends
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_ws():
    """TestClient with fake transcriber/TTS — no Deepgram, no Anthropic."""
    for attr in ("transcriber_factory", "tts_client", "diarizer_factory"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)
    mock_llm = MagicMock()
    mock_llm.complete.return_value = MOCK_LLM_JSON
    app.state.llm_client = mock_llm
    app.state.transcriber_factory = lambda: FakeTranscriber()
    app.state.tts_client = FakeTTS()
    try:
        yield TestClient(app)
    finally:
        for attr in ("transcriber_factory", "tts_client", "diarizer_factory"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)


def _open_guest(client, session_id):
    """Send the guest auth handshake; return the first server frame."""
    ws = client.websocket_connect(f"/ws/session/{session_id}").__enter__()
    ws.send_text(json.dumps({"type": "config", "id_token": "tok-guest"}))
    return ws, json.loads(ws.receive_text())


class TestWebSocketQuota:
    def test_a_guest_token_opens_a_session(self, fake_ws):
        ws, first = _open_guest(fake_ws, str(uuid.uuid4()))
        try:
            assert first["type"] == "config_ack"
        finally:
            ws.__exit__(None, None, None)

    def test_config_ack_tells_a_guest_the_allowance_up_front(self, fake_ws, monkeypatch):
        """The numbers must arrive on CONNECT, not only when a limit refuses.

        They used to be sent solely by _close_ws_guest_limit — i.e. at the
        moment a session was cut off — so the app could not warn anyone before
        the fact, and a device that had never hit a limit had nothing to show
        at all. Deliberately not the 3/10 defaults: a server that hardcoded
        them would pass a 3/10 assertion by accident.
        """
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSIONS_PER_DAY", 7)
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSION_MIN", 4)
        ws, first = _open_guest(fake_ws, str(uuid.uuid4()))
        try:
            assert first["type"] == "config_ack"
            assert first["guest_limits"] == {
                "max_sessions_per_day": 7,
                "max_session_minutes": 4,
            }
        finally:
            ws.__exit__(None, None, None)

    def test_a_signed_up_account_gets_no_guest_limits(self, fake_ws):
        """Only a guest's ack carries them — a real account has no allowance."""
        ws = fake_ws.websocket_connect(f"/ws/session/{uuid.uuid4()}").__enter__()
        try:
            ws.send_text(json.dumps({"type": "config", "id_token": "fake-id-token"}))
            ack = json.loads(ws.receive_text())
            assert ack["type"] == "config_ack"
            assert "guest_limits" not in ack
        finally:
            ws.__exit__(None, None, None)

    def test_the_fourth_session_of_the_day_is_refused(self, fake_ws, monkeypatch):
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSIONS_PER_DAY", 3)
        for _ in range(3):
            ws, first = _open_guest(fake_ws, str(uuid.uuid4()))
            assert first["type"] == "config_ack"
            ws.__exit__(None, None, None)

        ws, frame = _open_guest(fake_ws, str(uuid.uuid4()))
        try:
            assert frame["type"] == "guest_limit"
            assert frame["message"] == guest_quota.GUEST_LIMIT_MESSAGE
            assert frame["max_sessions_per_day"] == 3
            # Closed 4429, NOT 4401 — the app must not read this as a
            # revoked sign-in and clear the session.
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_text()
            assert exc.value.code == guest_quota.GUEST_LIMIT_WS_CODE
        finally:
            ws.__exit__(None, None, None)

    def test_reconnecting_the_same_session_does_not_burn_the_allowance(
        self, fake_ws, monkeypatch,
    ):
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSIONS_PER_DAY", 1)
        sid = str(uuid.uuid4())
        for _ in range(4):
            ws, first = _open_guest(fake_ws, sid)
            assert first["type"] == "config_ack", first
            ws.__exit__(None, None, None)

    def test_a_signed_up_account_has_no_daily_limit(self, fake_ws, monkeypatch):
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSIONS_PER_DAY", 1)
        for _ in range(4):
            sid = str(uuid.uuid4())
            with fake_ws.websocket_connect(f"/ws/session/{sid}") as ws:
                ws.send_text(json.dumps(
                    {"type": "config", "id_token": "fake-id-token"}))
                assert json.loads(ws.receive_text())["type"] == "config_ack"

    def test_a_guest_session_is_ended_at_the_minute_cap(self, fake_ws, monkeypatch):
        """The cap is wall-clock, so the test moves the clock rather than
        waiting ten minutes: after the deadline passes, the NEXT inbound
        frame ends the session with the same honest message."""
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSION_MIN", 10)
        ws, first = _open_guest(fake_ws, str(uuid.uuid4()))
        try:
            assert first["type"] == "config_ack"
            # Anything sent BEFORE the deadline flows normally.
            ws.send_bytes(b"\x00" * 50)
            assert json.loads(ws.receive_text())["type"] in {
                "transcript", "suggestion"}

            # Jump past the deadline (monotonic, so patch the clock the
            # pipeline reads) and send one more frame.
            import audio_pipeline

            real_monotonic = audio_pipeline.time.monotonic
            monkeypatch.setattr(
                audio_pipeline.time, "monotonic",
                lambda: real_monotonic() + 11 * 60,
            )
            ws.send_bytes(b"\x00" * 50)
            # The pre-deadline frame's own transcript/suggestion may still be
            # in flight (the suggestion worker is async) — skip past the
            # coaching stream to the verdict.
            for _ in range(10):
                frame = json.loads(ws.receive_text())
                if frame.get("type") not in {"transcript", "suggestion"}:
                    break
            assert frame["type"] == "guest_limit"
            assert frame["max_session_minutes"] == 10
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_text()
            assert exc.value.code == guest_quota.GUEST_LIMIT_WS_CODE
        finally:
            ws.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# 3c. The quota on POST /sessions/live (what buys the batch LLM spend)
# ---------------------------------------------------------------------------

@pytest.fixture
async def live_client():
    await init_db()
    main._rate_limiter.reset()
    store = FakeLiveStore()
    app.state.recordings_store = store
    sessions_router._REFLECT_LOCKS.clear()
    sessions_router._REFLECT_LOCK_USERS.clear()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as ac:
        yield ac, store
    del app.state.recordings_store


def _live_body(session_id=SESSION_ID, *, started=None, ended=None):
    start = started or "2026-09-20T18:05:00+00:00"
    end = ended or "2026-09-20T18:07:00+00:00"
    turns = [dict(t, session_id=session_id) for t in TURNS]
    return {
        "session_id": session_id, "started_at": start, "ended_at": end,
        "mode": "earpiece", "turns": turns,
        # The record itself is cheap; the analysis/reflection are the spend.
        # Off here so the assertions are about the gate, not the LLM.
        "analyze": False, "reflect": False,
    }


class TestIngestQuota:
    async def test_a_guest_can_store_a_session(self, live_client):
        client, _ = live_client
        res = await client.post(
            "/sessions/live", json=_live_body(), headers=GUEST_HEADERS)
        assert res.status_code == 201

    async def test_the_fourth_session_of_the_day_is_429(
        self, live_client, monkeypatch,
    ):
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSIONS_PER_DAY", 3)
        client, _ = live_client
        for i in range(3):
            res = await client.post(
                "/sessions/live", json=_live_body(str(uuid.uuid4())),
                headers=GUEST_HEADERS)
            assert res.status_code == 201, (i, res.text)

        res = await client.post(
            "/sessions/live", json=_live_body(str(uuid.uuid4())),
            headers=GUEST_HEADERS)
        assert res.status_code == 429
        assert res.json()["detail"] == guest_quota.GUEST_LIMIT_MESSAGE

    async def test_a_session_longer_than_the_cap_is_429(
        self, live_client, monkeypatch,
    ):
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSION_MIN", 10)
        client, _ = live_client
        res = await client.post(
            "/sessions/live",
            json=_live_body(
                started="2026-09-20T18:00:00+00:00",
                ended="2026-09-20T18:45:00+00:00",
            ),
            headers=GUEST_HEADERS,
        )
        assert res.status_code == 429
        assert res.json()["detail"] == guest_quota.GUEST_LIMIT_MESSAGE

    async def test_a_signed_up_account_is_never_gated(
        self, live_client, monkeypatch,
    ):
        """Same two payloads that just failed for a guest — over the daily
        count and hours long — both land for a real account."""
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSIONS_PER_DAY", 1)
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSION_MIN", 10)
        client, _ = live_client
        for _ in range(3):
            res = await client.post(
                "/sessions/live", json=_live_body(str(uuid.uuid4())),
                headers=REAL_HEADERS)
            assert res.status_code == 201
        res = await client.post(
            "/sessions/live",
            json=_live_body(
                str(uuid.uuid4()),
                started="2026-09-20T18:00:00+00:00",
                ended="2026-09-20T20:45:00+00:00",
            ),
            headers=REAL_HEADERS,
        )
        assert res.status_code == 201

    async def test_reposting_the_same_session_is_free(
        self, live_client, monkeypatch,
    ):
        """A retry after a flaky network must not spend a second slot."""
        monkeypatch.setattr(guest_quota, "GUEST_MAX_SESSIONS_PER_DAY", 1)
        client, _ = live_client
        sid = str(uuid.uuid4())
        for _ in range(3):
            res = await client.post(
                "/sessions/live", json=_live_body(sid), headers=GUEST_HEADERS)
            assert res.status_code == 201, res.text


# ---------------------------------------------------------------------------
# 4. A guest can delete everything
# ---------------------------------------------------------------------------

class TestGuestDeletion:
    async def test_delete_me_works_for_an_anonymous_account(self, monkeypatch):
        """Play requires an in-app delete for any account the app creates —
        a guest account is one. The walk is uid-scoped and never reads the
        sign-in provider, so this is a regression guard: it must keep
        working, including the Firebase Auth user itself."""
        await init_db()
        import account_deletion
        from routers import account as account_router

        deleted_firebase_uids: list[str] = []
        monkeypatch.setattr(
            auth_module, "delete_firebase_user",
            lambda uid: deleted_firebase_uids.append(uid) or True,
        )
        monkeypatch.setattr(
            account_router.auth, "delete_firebase_user",
            lambda uid: deleted_firebase_uids.append(uid) or True,
        )
        account_router._delete_rate_limiter.reset()

        seen: dict = {}

        async def _fake_walk(uid, **kwargs):
            seen["uid"] = uid
            return account_deletion.DeletionSummary(
                counts={k: 0 for k in account_deletion.COUNT_KEYS},
                errors=[], warnings=[],
            )

        monkeypatch.setattr(account_deletion, "delete_account_data", _fake_walk)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            res = await client.request(
                "DELETE", "/me", json={"confirm": "DELETE"},
                headers=GUEST_HEADERS,
            )

        assert res.status_code == 200, res.text
        body = res.json()
        assert body["deleted"] is True
        assert body["firebase_user_deleted"] is True
        # Acted on the guest's own uid, and on nothing else.
        assert seen["uid"] == "guest-user"
        assert deleted_firebase_uids == ["guest-user"]
