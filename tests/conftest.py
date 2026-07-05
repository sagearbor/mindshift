import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# Add server/ to sys.path so we can import main
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

# Use a temp database for each test session
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["MINDSHIFT_DB_PATH"] = _tmp.name
_tmp.close()

from main import app, init_db  # noqa: E402

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
