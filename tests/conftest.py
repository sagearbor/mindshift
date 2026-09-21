import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi import Header
from httpx import ASGITransport, AsyncClient

# Add server/ to sys.path so we can import main
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

# Use a temp database for each test session
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["MINDSHIFT_DB_PATH"] = _tmp.name
_tmp.close()

from main import app, init_db  # noqa: E402
from auth import Identity, get_current_identity, get_current_uid  # noqa: E402

# Auth is required on every data route now; these end-to-end tests run as a
# single fixed user via a dependency override (keyless — no real Firebase). The
# X-Test-Uid header can name a different user, but these suites don't need to.
DEFAULT_TEST_UID = "test-user"


def _test_uid_override(x_test_uid: str = Header(default=DEFAULT_TEST_UID)) -> str:
    return x_test_uid


def _test_identity_override(
    x_test_uid: str = Header(default=DEFAULT_TEST_UID),
    x_test_guest: str = Header(default=""),
) -> Identity:
    """Mirror of ``server/conftest.py``'s override for the richer identity
    dependency (endpoints that need the sign-in provider, e.g. the guest quota
    on ``POST /sessions/live``).

    It must behave IDENTICALLY to that one, ``X-Test-Guest`` included, even
    though these end-to-end suites never send that header themselves. Both
    conftests install their overrides on the SAME shared ``main.app`` at
    import time, so in a run that collects both trees (which is exactly how
    CI runs it — see pyproject's testpaths) whichever imported last wins for
    every test in both. The duplication is deliberate: these two modules
    cannot import each other (see the pythonpath comment in pyproject.toml),
    and a subtly weaker copy here would silently disarm the guest tests over
    there.
    """
    guest = x_test_guest.strip().lower() in {"1", "true", "yes", "on"}
    return Identity(
        uid=x_test_uid,
        email=None if guest else f"{x_test_uid}@example.test",
        sign_in_provider="anonymous" if guest else "password",
    )


app.dependency_overrides[get_current_uid] = _test_uid_override
app.dependency_overrides[get_current_identity] = _test_identity_override

# Shared mock payloads live in _mock_data so test modules can import them
# unambiguously even when pytest collects both server/ and tests/ in one run
# (two conftest.py modules would otherwise shadow each other).
from _mock_data import (  # noqa: E402,F401 — re-exported for backward compatibility
    MOCK_ASSERTIVE_JSON,
    MOCK_FULL_EMPATHY_JSON,
    MOCK_RESPOND_JSON,
    MOCK_SCORE_JSON,
    TONE_SCORE_KEYS,
)


@pytest.fixture
async def client():
    await init_db()
    # P1-5: isolate each test's per-IP rate-limit window so cumulative
    # cost-endpoint traffic across the suite (all tests share one client IP)
    # cannot trip the limiter. The generous 60/min default stays in force.
    import main  # noqa: E402 — app already imported above
    main._rate_limiter.reset()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def sample_transcripts():
    """Load the sample_transcripts.json fixture file."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_transcripts.json"
    with open(fixture_path) as f:
        return json.load(f)
