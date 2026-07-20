"""Tests for account-to-account read-only recording sharing.

GCS and Firebase are never touched: an in-memory :class:`FakeShareStore` (the
repo's ``app.state.recordings_store`` DI style) implements the same async
interface the real store exposes for sharing + reads, and ``main``'s
email→uid / uid→email resolvers are monkeypatched. The fake mirrors the real
store's grant semantics (owner meta ``shares`` + a per-recipient reverse index)
so the access-control matrix is exercised end-to-end through the real endpoints.
"""

import uuid
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

import main
import recordings_store
from main import app, init_db

# Fake Firebase directory: email ↔ uid. Monkeypatched onto main's resolvers.
ACCOUNTS = {
    "linda@example.com": "user-linda",
    "sage@example.com": "user-sage",
    "bystander@example.com": "user-bystander",
}
UID_TO_EMAIL = {v: k for k, v in ACCOUNTS.items()}


# ---------------------------------------------------------------------------
# In-memory fake store (async interface subset used by the sharing endpoints)
# ---------------------------------------------------------------------------

class FakeShareStore:
    def __init__(self):
        # {uid: {recording_id: {meta, turns, analysis, audio, media, ct}}}
        self._by_uid: dict[str, dict[str, dict]] = {}
        # reverse index: {recipient_uid: {(owner_uid, rid): grant_dict}}
        self._index: dict[str, dict[tuple, dict]] = {}

    # -- seeding helper (stands in for save_recording) --------------------
    def seed(self, uid, *, rid=None, title="Kitchen talk", media_type="audio",
             turns=None, analysis=None, audio=b"AUDIO", media=b"MEDIA-BYTES",
             ct="audio/mp4", source=None):
        rid = rid or str(uuid.uuid4())
        meta = {
            "id": rid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "filename": f"{title}.m4a",
            "title": title,
            "media_type": media_type,
            "duration_seconds": 12.0,
            "source": source or {"type": "upload", "url": None,
                                 "original_filename": f"{title}.m4a"},
        }
        self._by_uid.setdefault(uid, {})[rid] = {
            "meta": meta,
            "turns": turns if turns is not None else [
                {"speaker": "Speaker A", "text": "hi", "start_time": 0.0,
                 "end_time": 1.0},
                {"speaker": "Speaker B", "text": "hey", "start_time": 1.0,
                 "end_time": 2.0},
            ],
            "analysis": analysis if analysis is not None else {
                "per_turn": [
                    {"index": 0, "speaker": "Speaker A", "heat": 10,
                     "markers": [], "is_spike": False, "trigger_phrase": None},
                    {"index": 1, "speaker": "Speaker B", "heat": 12,
                     "markers": [], "is_spike": False, "trigger_phrase": None},
                ],
                "speaker_labels": {},
            },
            "audio": audio,
            "media": media,
            "ct": ct,
        }
        return rid

    # -- reads -------------------------------------------------------------
    async def list_recordings(self, uid):
        out = [{**r["meta"], "has_analysis": r["analysis"] is not None}
               for r in self._by_uid.get(uid, {}).values()]
        out.sort(key=lambda m: m["created_at"], reverse=True)
        return out

    async def get_recording(self, uid, rid):
        r = self._by_uid.get(uid, {}).get(rid)
        if r is None:
            return None
        return {**r["meta"], "turns": r["turns"], "analysis": r["analysis"]}

    async def recording_exists(self, uid, rid):
        return rid in self._by_uid.get(uid, {})

    async def get_audio_bytes(self, uid, rid):
        r = self._by_uid.get(uid, {}).get(rid)
        return r["audio"] if r else None

    async def open_media_stream(self, uid, rid, range_header):
        r = self._by_uid.get(uid, {}).get(rid)
        if r is None:
            return None
        payload = r["media"]
        start, end, status, headers = recordings_store.plan_media_response(
            len(payload), r["ct"], range_header,
        )
        return recordings_store._iter_bytes(payload[start:end + 1]), status, headers

    # -- owner writes ------------------------------------------------------
    async def update_title(self, uid, rid, title):
        r = self._by_uid.get(uid, {}).get(rid)
        if r is None:
            return None
        r["meta"]["title"] = title
        return r["meta"]

    async def update_source(self, uid, rid, source):
        r = self._by_uid.get(uid, {}).get(rid)
        if r is None:
            return None
        r["meta"]["source"] = source
        return source

    async def update_manual_speaker_labels(self, uid, rid, labels):
        r = self._by_uid.get(uid, {}).get(rid)
        if r is None:
            return None
        if labels:
            r["meta"]["manual_speaker_labels"] = labels
        else:
            r["meta"].pop("manual_speaker_labels", None)
        return r["meta"]

    async def overwrite_analysis(self, uid, rid, *, turns, analysis, reanalyzed_at):
        r = self._by_uid.get(uid, {}).get(rid)
        if r is None:
            return None
        r["meta"]["reanalyzed_at"] = reanalyzed_at
        r["turns"], r["analysis"] = turns, analysis
        return r["meta"]

    async def delete_recording(self, uid, rid):
        r = self._by_uid.get(uid, {}).pop(rid, None)
        if r is None:
            return False
        # Kill every recipient's reverse-index grant (spec §4).
        for share in (r["meta"].get("shares") or []):
            self._index.get(share["uid"], {}).pop((uid, rid), None)
        return True

    # -- sharing -----------------------------------------------------------
    async def add_share(self, owner_uid, rid, *, recipient_uid, recipient_email,
                        owner_email):
        r = self._by_uid.get(owner_uid, {}).get(rid)
        if r is None:
            return None
        created = datetime.now(timezone.utc).isoformat()
        shares = [s for s in (r["meta"].get("shares") or [])
                  if s["uid"] != recipient_uid]
        shares.append({"uid": recipient_uid, "email": recipient_email,
                       "created_at": created})
        r["meta"]["shares"] = shares
        self._index.setdefault(recipient_uid, {})[(owner_uid, rid)] = {
            "owner_uid": owner_uid, "recording_id": rid,
            "owner_email": owner_email, "created_at": created,
        }
        return shares

    async def remove_share(self, owner_uid, rid, recipient_uid):
        r = self._by_uid.get(owner_uid, {}).get(rid)
        if r is None:
            return False
        r["meta"]["shares"] = [s for s in (r["meta"].get("shares") or [])
                               if s["uid"] != recipient_uid]
        self._index.get(recipient_uid, {}).pop((owner_uid, rid), None)
        return True

    async def find_share(self, recipient_uid, rid):
        for (owner_uid, r_rid), grant in self._index.get(recipient_uid, {}).items():
            if r_rid == rid:
                return grant
        return None

    async def list_shared_with(self, recipient_uid):
        out = []
        for (owner_uid, rid), grant in self._index.get(recipient_uid, {}).items():
            r = self._by_uid.get(owner_uid, {}).get(rid)
            if r is None:
                continue
            meta = {**r["meta"], "has_analysis": r["analysis"] is not None}
            meta["owner_email"] = grant["owner_email"]
            meta["shared"] = True
            meta.pop("shares", None)
            meta["_at"] = grant["created_at"]
            out.append(meta)
        out.sort(key=lambda m: m.pop("_at"), reverse=True)
        return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def client():
    await init_db()
    main._rate_limiter.reset()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def store(monkeypatch):
    fake = FakeShareStore()
    app.state.recordings_store = fake

    def _resolve_uid(email):
        return ACCOUNTS.get(email.strip().lower())

    def _resolve_email(uid):
        return UID_TO_EMAIL.get(uid)

    monkeypatch.setattr(main, "resolve_uid_by_email", _resolve_uid)
    monkeypatch.setattr(main, "resolve_email_by_uid", _resolve_email)
    yield fake
    del app.state.recordings_store


def _h(uid):
    return {"X-Test-Uid": uid}


# Owner is "user-linda"; recipient is "user-sage".
OWNER = "user-linda"
RECIPIENT = "user-sage"
STRANGER = "user-bystander"


# ---------------------------------------------------------------------------
# Grant
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_owner_shares_recording(client, store):
    rid = store.seed(OWNER)
    resp = await client.post(
        f"/recordings/{rid}/shares", json={"email": "sage@example.com"},
        headers=_h(OWNER),
    )
    assert resp.status_code == 200, resp.text
    shares = resp.json()["shares"]
    assert len(shares) == 1
    assert shares[0]["uid"] == RECIPIENT
    assert shares[0]["email"] == "sage@example.com"
    assert shares[0]["created_at"]


@pytest.mark.anyio
async def test_share_email_not_found_is_404(client, store):
    rid = store.seed(OWNER)
    resp = await client.post(
        f"/recordings/{rid}/shares", json={"email": "nobody@example.com"},
        headers=_h(OWNER),
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "no MindShift account with that email"


@pytest.mark.anyio
async def test_cannot_share_with_self(client, store):
    rid = store.seed(OWNER)
    resp = await client.post(
        f"/recordings/{rid}/shares", json={"email": "linda@example.com"},
        headers=_h(OWNER),
    )
    assert resp.status_code == 400
    assert "yourself" in resp.json()["detail"]


@pytest.mark.anyio
async def test_share_requires_ownership_before_email_lookup(client, store):
    # Recording belongs to OWNER; STRANGER must not be able to probe emails
    # against it — they get a 404 on the recording, never a "no account" signal.
    rid = store.seed(OWNER)
    resp = await client.post(
        f"/recordings/{rid}/shares", json={"email": "sage@example.com"},
        headers=_h(STRANGER),
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Recording not found"


@pytest.mark.anyio
async def test_share_invalid_email_is_422(client, store):
    rid = store.seed(OWNER)
    resp = await client.post(
        f"/recordings/{rid}/shares", json={"email": "not-an-email"},
        headers=_h(OWNER),
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_reshare_is_idempotent(client, store):
    rid = store.seed(OWNER)
    for _ in range(2):
        resp = await client.post(
            f"/recordings/{rid}/shares", json={"email": "sage@example.com"},
            headers=_h(OWNER),
        )
        assert resp.status_code == 200
    assert len(resp.json()["shares"]) == 1  # no duplicate entry


# ---------------------------------------------------------------------------
# List views
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_owner_list_shows_shares(client, store):
    rid = store.seed(OWNER)
    await store.add_share(OWNER, rid, recipient_uid=RECIPIENT,
                          recipient_email="sage@example.com",
                          owner_email="linda@example.com")
    resp = await client.get("/recordings", headers=_h(OWNER))
    assert resp.status_code == 200
    body = resp.json()
    row = next(r for r in body["recordings"] if r["id"] == rid)
    assert row["shares"][0]["email"] == "sage@example.com"
    assert body["shared_with_me"] == []


@pytest.mark.anyio
async def test_recipient_sees_shared_with_me(client, store):
    rid = store.seed(OWNER, title="Sunday budget")
    await store.add_share(OWNER, rid, recipient_uid=RECIPIENT,
                          recipient_email="sage@example.com",
                          owner_email="linda@example.com")
    resp = await client.get("/recordings", headers=_h(RECIPIENT))
    assert resp.status_code == 200
    body = resp.json()
    assert body["recordings"] == []  # recipient owns nothing
    assert len(body["shared_with_me"]) == 1
    entry = body["shared_with_me"][0]
    assert entry["id"] == rid
    assert entry["owner_email"] == "linda@example.com"
    assert entry["shared"] is True
    assert entry["title"] == "Sunday budget"
    # A recipient must never see co-recipients.
    assert "shares" not in entry


@pytest.mark.anyio
async def test_list_defensive_when_no_shares(client, store):
    store.seed(OWNER)
    resp = await client.get("/recordings", headers=_h(OWNER))
    assert resp.status_code == 200
    assert resp.json()["shared_with_me"] == []


# ---------------------------------------------------------------------------
# Recipient read access
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_recipient_reads_detail(client, store):
    rid = store.seed(OWNER)
    await store.add_share(OWNER, rid, recipient_uid=RECIPIENT,
                          recipient_email="sage@example.com",
                          owner_email="linda@example.com")
    resp = await client.get(f"/recordings/{rid}", headers=_h(RECIPIENT))
    assert resp.status_code == 200
    body = resp.json()
    assert body["shared"] is True
    assert body["owner_email"] == "linda@example.com"
    assert body["shares"] == []  # co-recipients never leaked
    assert body["turns"]  # full transcript available
    assert body["analysis"] is not None


@pytest.mark.anyio
async def test_recipient_can_mint_media_url_and_stream(client, store):
    rid = store.seed(OWNER, media=b"THE-OWNER-MEDIA-BYTES")
    await store.add_share(OWNER, rid, recipient_uid=RECIPIENT,
                          recipient_email="sage@example.com",
                          owner_email="linda@example.com")
    resp = await client.get(f"/recordings/{rid}/media_url", headers=_h(RECIPIENT))
    assert resp.status_code == 200
    url = resp.json()["url"]
    # The token streams the OWNER's bytes even though the recipient requested it.
    media = await client.get(url.replace("http://test", ""))
    assert media.status_code == 200
    assert media.content == b"THE-OWNER-MEDIA-BYTES"


@pytest.mark.anyio
async def test_stranger_cannot_read_detail_or_media(client, store):
    rid = store.seed(OWNER)
    await store.add_share(OWNER, rid, recipient_uid=RECIPIENT,
                          recipient_email="sage@example.com",
                          owner_email="linda@example.com")
    # A user with no grant sees a plain 404 — never confirmed to exist.
    assert (await client.get(f"/recordings/{rid}", headers=_h(STRANGER))
            ).status_code == 404
    assert (await client.get(f"/recordings/{rid}/media_url", headers=_h(STRANGER))
            ).status_code == 404


# ---------------------------------------------------------------------------
# Recipient writes are all denied 403 (read-only)
# ---------------------------------------------------------------------------

@pytest.fixture
async def shared(store):
    rid = store.seed(OWNER)
    await store.add_share(OWNER, rid, recipient_uid=RECIPIENT,
                          recipient_email="sage@example.com",
                          owner_email="linda@example.com")
    return rid


@pytest.mark.anyio
async def test_recipient_cannot_rename(client, store, shared):
    resp = await client.patch(
        f"/recordings/{shared}", json={"title": "hijack"}, headers=_h(RECIPIENT),
    )
    assert resp.status_code == 403
    assert "read-only" in resp.json()["detail"]


@pytest.mark.anyio
async def test_recipient_cannot_patch_source(client, store, shared):
    resp = await client.patch(
        f"/recordings/{shared}/source", json={"url": "https://evil.example/x"},
        headers=_h(RECIPIENT),
    )
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_recipient_cannot_set_speaker_labels(client, store, shared):
    resp = await client.patch(
        f"/recordings/{shared}/speaker-labels",
        json={"labels": {"Speaker A": "Me"}}, headers=_h(RECIPIENT),
    )
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_recipient_cannot_delete(client, store, shared):
    resp = await client.delete(f"/recordings/{shared}", headers=_h(RECIPIENT))
    assert resp.status_code == 403
    # And the owner's recording is untouched.
    assert await store.recording_exists(OWNER, shared)


@pytest.mark.anyio
async def test_recipient_cannot_reanalyze(client, store, shared):
    resp = await client.post(
        f"/recordings/{shared}/reanalyze", headers=_h(RECIPIENT),
    )
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_stranger_write_is_404_not_403(client, store, shared):
    # A user with no grant gets 404 (not 403) — the recording is never confirmed.
    resp = await client.patch(
        f"/recordings/{shared}", json={"title": "x"}, headers=_h(STRANGER),
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Revoke + revocation immediacy
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_revoke_removes_access(client, store, shared):
    resp = await client.delete(
        f"/recordings/{shared}/shares/{RECIPIENT}", headers=_h(OWNER),
    )
    assert resp.status_code == 204
    # Recipient can no longer read the detail or mint a fresh media URL.
    assert (await client.get(f"/recordings/{shared}", headers=_h(RECIPIENT))
            ).status_code == 404
    assert (await client.get(f"/recordings/{shared}/media_url",
                             headers=_h(RECIPIENT))).status_code == 404
    # And it's gone from their shared-with-me list.
    body = (await client.get("/recordings", headers=_h(RECIPIENT))).json()
    assert body["shared_with_me"] == []


@pytest.mark.anyio
async def test_revoke_is_idempotent(client, store):
    rid = store.seed(OWNER)  # never shared
    resp = await client.delete(
        f"/recordings/{rid}/shares/{RECIPIENT}", headers=_h(OWNER),
    )
    assert resp.status_code == 204  # nothing to remove, still succeeds


@pytest.mark.anyio
async def test_revoke_foreign_recording_is_404(client, store, shared):
    # STRANGER doesn't own the recording — can't revoke its grants.
    resp = await client.delete(
        f"/recordings/{shared}/shares/{RECIPIENT}", headers=_h(STRANGER),
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_owner_delete_kills_recipient_access(client, store, shared):
    resp = await client.delete(f"/recordings/{shared}", headers=_h(OWNER))
    assert resp.status_code == 204
    # The reverse-index grant was cleaned up: recipient sees nothing.
    assert (await client.get(f"/recordings/{shared}", headers=_h(RECIPIENT))
            ).status_code == 404
    body = (await client.get("/recordings", headers=_h(RECIPIENT))).json()
    assert body["shared_with_me"] == []


# ---------------------------------------------------------------------------
# Storage-disabled honesty
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_share_503_when_storage_disabled(client):
    # No `store` fixture → storage disabled → honest 503, not a crash.
    rid = str(uuid.uuid4())
    resp = await client.post(
        f"/recordings/{rid}/shares", json={"email": "sage@example.com"},
        headers=_h(OWNER),
    )
    assert resp.status_code == 503
