import asyncio
import os
import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Header
from httpx import ASGITransport, AsyncClient

# Use a temp database for each test session
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["MINDSHIFT_DB_PATH"] = _tmp.name
_tmp.close()

from main import app, init_db  # noqa: E402 — must set env before import
from auth import (  # noqa: E402
    ANONYMOUS_PROVIDER,
    Identity,
    get_current_identity,
    get_current_uid,
    get_fresh_uid,
)


# ---------------------------------------------------------------------------
# Auth test harness — keyless, never touches real Firebase
# ---------------------------------------------------------------------------
# Fake Firebase tokens → uids. The default token/uid keep the REST + WS tests
# green once they present it; the two extra users drive cross-user isolation.
# ``tok-guest`` is a "Continue as guest" (Firebase Anonymous Auth) token —
# same shape, no email, ``sign_in_provider == "anonymous"``.
FAKE_TOKENS = {
    "fake-id-token": "test-user",
    "tok-user-a": "user-a",
    "tok-user-b": "user-b",
    "tok-guest": "guest-user",
}
DEFAULT_TEST_UID = "test-user"
# Which of the fake tokens above were minted by anonymous sign-in.
GUEST_TOKENS = {"tok-guest"}
# Sending ``X-Test-Guest: 1`` makes the dependency override below act as a
# guest, exactly the way ``X-Test-Uid`` makes it act as another user.
GUEST_TEST_HEADER_TRUE = {"1", "true", "yes", "on"}


def _test_uid_override(x_test_uid: str = Header(default=DEFAULT_TEST_UID)) -> str:
    """Stand-in for :func:`auth.get_current_uid`: returns the ``X-Test-Uid``
    header (or the default), so the REST suite runs authenticated WITHOUT real
    Firebase. Cross-user tests send a different ``X-Test-Uid`` to act as another
    user; the 401/valid-token tests remove this override to exercise the real
    dependency against the fake verifier below.
    """
    return x_test_uid


def _test_identity_override(
    x_test_uid: str = Header(default=DEFAULT_TEST_UID),
    x_test_guest: str = Header(default=""),
) -> Identity:
    """Stand-in for :func:`auth.get_current_identity` — the richer sibling of
    ``_test_uid_override``.

    Endpoints that need to know HOW the caller signed in (today: the guest
    quota on ``POST /sessions/live``) depend on the identity rather than the
    bare uid. Without this override those endpoints would fall through to real
    Firebase for every test in the suite. ``X-Test-Guest: 1`` opts a request
    into being an anonymous ("Continue as guest") caller; everything else is a
    password account, which is what the suite has always simulated.
    """
    guest = x_test_guest.strip().lower() in GUEST_TEST_HEADER_TRUE
    return Identity(
        uid=x_test_uid,
        email=None if guest else f"{x_test_uid}@example.test",
        sign_in_provider=ANONYMOUS_PROVIDER if guest else "password",
    )


# Installed once on the shared app: every TestClient/AsyncClient built from
# ``main.app`` (this conftest, tests/conftest, or any test module) inherits it.
app.dependency_overrides[get_current_uid] = _test_uid_override
app.dependency_overrides[get_current_identity] = _test_identity_override
# Same stand-in for the FRESH-token dependency (auth.get_fresh_uid, used by
# DELETE /me): the suite runs authenticated without real Firebase, so there is
# no real ``iat`` to age. The freshness gate itself is exercised directly
# against auth.get_fresh_uid in test_account_deletion.py, where the override is
# removed and a fake verifier supplies the claims.
app.dependency_overrides[get_fresh_uid] = _test_uid_override


@pytest.fixture(autouse=True)
def _server_test_auth(monkeypatch):
    """Per-test auth harness for the server suite.

    * Ensures the DB schema exists — the WS auth handshake checks session
      ownership in the ``sessions`` table, so it must be present even for the
      WS tests that never build the ``client`` fixture.
    * Replaces ``auth.verify_id_token`` AND ``auth.verify_id_token_identity``
      with keyless fakes used by the WS handshake and by the REST tests that
      drop the dependency override to hit the real
      :func:`auth.get_current_uid`. Both are patched (and agree with each
      other) because the WS handshake reads the identity form — it needs the
      sign-in provider for the guest quota — while REST still reads the uid
      form.
    * Clears the process-wide guest session counter, so a test that exhausts
      a guest's daily allowance cannot leak that state into the next test.
    """
    asyncio.run(init_db())

    import auth
    import guest_quota

    def _verify(token: str) -> str:
        try:
            return FAKE_TOKENS[token]
        except KeyError:
            raise ValueError("invalid test token")

    def _verify_identity(token: str) -> auth.Identity:
        uid = _verify(token)
        guest = token in GUEST_TOKENS
        return auth.Identity(
            uid=uid,
            email=None if guest else f"{uid}@example.test",
            sign_in_provider=ANONYMOUS_PROVIDER if guest else "password",
        )

    monkeypatch.setattr(auth, "verify_id_token", _verify)
    monkeypatch.setattr(auth, "verify_id_token_identity", _verify_identity)
    guest_quota.SESSION_COUNTER.reset()

    # Re-assert THIS module's overrides for every test under server/.
    # tests/conftest.py installs its own on the very same shared ``main.app``
    # at import time, and in a run that collects both trees the later import
    # wins for everybody. The two are kept behaviourally identical on purpose
    # (see the note on tests/conftest.py's _test_identity_override), but
    # re-installing here means a future divergence can't silently disarm the
    # guest-quota tests instead of failing loudly in the file that diverged.
    app.dependency_overrides[get_current_uid] = _test_uid_override
    app.dependency_overrides[get_current_identity] = _test_identity_override


MOCK_RESPOND_JSON = json.dumps({
    "suggestions": [
        "I hear what you're saying.",
        "That sounds really frustrating.",
        "Can you tell me more about how that made you feel?",
    ],
    "tone_score": {
        "warmth": 60,
        "defensiveness": 30,
        "sarcasm": 10,
        "constructiveness": 55,
        "overall": 65,
    },
})

MOCK_SCORE_JSON = json.dumps({
    "warmth": 70,
    "defensiveness": 20,
    "sarcasm": 5,
    "constructiveness": 80,
    "overall": 75,
})


@pytest.fixture
def mock_llm():
    """Patch the LLMClient so no real API calls are made."""
    mock_client = MagicMock()
    with patch("main.get_llm_client", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_respond(mock_llm):
    mock_llm.complete.return_value = MOCK_RESPOND_JSON
    return mock_llm


@pytest.fixture
def mock_score(mock_llm):
    mock_llm.complete.return_value = MOCK_SCORE_JSON
    return mock_llm


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture(autouse=True)
def _natural_turns_off_by_default(monkeypatch):
    """The NaturalTurn post-pass (main.NATURAL_TURNS_ENV, default ON in
    production) merges rapid same-speaker turns. Most pipeline tests pin
    exact turn counts against count-locked fake LLMs, so it is OFF for the
    suite; tests/test_natural_turns_pass.py enables it for itself.

    Lives HERE rather than in a server/tests/conftest.py: server/tests/ is a
    package (``server/tests/__init__.py``) with ``server/`` on sys.path, so a
    conftest there imports as ``tests.conftest`` and collides with the
    top-level ``tests/conftest.py`` — which breaks `pytest` from the repo
    root, the exact command AGENTS.md documents and CI's backend job runs.
    This file's scope over server/ is identical: every server test lives
    under server/tests/.
    """
    monkeypatch.setenv("MINDSHIFT_NATURAL_TURNS", "0")
