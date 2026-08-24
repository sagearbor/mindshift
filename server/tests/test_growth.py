"""GET /growth — the per-user "Your growth" aggregate.

One point per STORED recording in which the user's voice was confidently
identified (the stored per-recording labels are the source of truth: a speaker
whose EFFECTIVE label_source is "enrolled" after the manual overlay). Pure GCS
reads via the store — no LLM. Honest gaps everywhere:

* a recording with no confident "me" is COUNTED (total) but never scored;
* a manual re-tag of the machine's "You" removes that recording from the chart
  WITHOUT re-analysis (the correction flows straight in);
* a missing/malformed report-card score yields ``my_score: null`` — never 0.

GCS is never touched: the in-memory fake mirrors ``test_recordings.py``'s DI
style (``app.state.recordings_store``).
"""

import uuid
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from auth import get_current_uid
from main import app, init_db

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# In-memory fake store — only what /growth reads (list + get)
# ---------------------------------------------------------------------------

class FakeGrowthStore:
    def __init__(self):
        # {uid: {rid: {meta, turns, analysis}}}
        self._by_uid: dict[str, dict[str, dict]] = {}

    def add(
        self,
        uid: str,
        *,
        created_at: str,
        title: str = "A talk",
        turns=None,
        analysis=None,
        manual_speaker_labels=None,
    ) -> str:
        rid = str(uuid.uuid4())
        meta = {
            "id": rid,
            "created_at": created_at,
            "filename": f"{title}.m4a",
            "title": title,
            "media_type": "audio",
            "duration_seconds": 60.0,
        }
        if manual_speaker_labels:
            meta["manual_speaker_labels"] = manual_speaker_labels
        self._by_uid.setdefault(uid, {})[rid] = {
            "meta": meta,
            "turns": turns or [],
            "analysis": analysis,
        }
        return rid

    async def list_recordings(self, uid):
        recs = self._by_uid.get(uid, {})
        out = [
            {**r["meta"], "has_analysis": r["analysis"] is not None}
            for r in recs.values()
        ]
        out.sort(key=lambda m: m["created_at"], reverse=True)
        return out

    async def get_recording(self, uid, recording_id):
        r = self._by_uid.get(uid, {}).get(recording_id)
        if r is None:
            return None
        return {**r["meta"], "turns": r["turns"], "analysis": r["analysis"]}


@pytest.fixture
async def client():
    await init_db()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def store():
    fake = FakeGrowthStore()
    app.state.recordings_store = fake
    yield fake
    del app.state.recordings_store


def _iso(day: int, hour: int = 12) -> str:
    return datetime(2026, 7, day, hour, tzinfo=timezone.utc).isoformat()


TURNS_AB = [
    {"speaker": "Speaker A", "text": "hi", "start_time": 0.0, "end_time": 3.0},
    {"speaker": "Speaker B", "text": "yo", "start_time": 3.0, "end_time": 6.0},
]


def _analysis(
    *,
    me: str | None = "Speaker A",
    partner_label: tuple[str, str] | None = None,
    scores: dict | None = None,
    speakers: tuple[str, ...] = ("Speaker A", "Speaker B"),
) -> dict:
    """A minimal stored analysis.json: speaker_labels + report_cards.

    ``me`` (if set) gets label_source "enrolled"; ``partner_label`` is an
    optional (display_label, label_source) for Speaker B; everyone else is
    generic. ``scores`` maps speaker → report-card score (default 70/50).
    """
    labels = {}
    for sp in speakers:
        labels[sp] = {"display_label": sp, "label_source": "generic"}
    if me is not None:
        labels[me] = {"display_label": "You", "label_source": "enrolled"}
    if partner_label is not None:
        labels["Speaker B"] = {
            "display_label": partner_label[0],
            "label_source": partner_label[1],
        }
    scores = scores or {"Speaker A": 70, "Speaker B": 50}
    cards = {
        sp: {
            "score": s, "headline": "h", "did_well": "d", "work_on": "w",
        }
        for sp, s in scores.items()
    }
    return {"speaker_labels": labels, "report_cards": cards}


# ---------------------------------------------------------------------------
# Availability + shape
# ---------------------------------------------------------------------------

async def test_growth_storage_disabled_503(client):
    res = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    assert res.status_code == 503


async def test_growth_requires_auth_401(client, store, monkeypatch):
    monkeypatch.delitem(app.dependency_overrides, get_current_uid)
    res = await client.get("/growth")
    assert res.status_code == 401


async def test_growth_empty_store(client, store):
    res = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    assert res.status_code == 200
    body = res.json()
    assert body == {
        "points": [], "total_recordings": 0, "identified_recordings": 0,
        # Track 2: per-person "how I sound with X" rows — none without sessions.
        "people": [],
    }


# ---------------------------------------------------------------------------
# Point construction — the enrolled speaker is "me"
# ---------------------------------------------------------------------------

async def test_growth_point_for_identified_recording(client, store):
    rid = store.add(
        "u1", created_at=_iso(1), title="Kitchen talk",
        turns=TURNS_AB, analysis=_analysis(),
    )
    res = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    body = res.json()
    assert body["total_recordings"] == 1
    assert body["identified_recordings"] == 1
    assert len(body["points"]) == 1
    p = body["points"][0]
    assert p["recording_id"] == rid
    assert p["timestamp"] == _iso(1)
    assert p["title"] == "Kitchen talk"
    assert p["my_score"] == 70  # report_cards["Speaker A"].score — mine, not B's


async def test_growth_counts_but_never_scores_unidentified(client, store):
    # Analyzed, but no enrolled match → counted, no point, no guess.
    store.add("u1", created_at=_iso(1), turns=TURNS_AB, analysis=_analysis(me=None))
    # Stored but never analyzed → counted, no point.
    store.add("u1", created_at=_iso(2), turns=TURNS_AB, analysis=None)
    res = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    body = res.json()
    assert body["total_recordings"] == 2
    assert body["identified_recordings"] == 0
    assert body["points"] == []


async def test_growth_points_sorted_by_time_ascending(client, store):
    # Inserted newest-first; the chart wants a TIME axis, oldest → newest.
    store.add("u1", created_at=_iso(9), turns=TURNS_AB, analysis=_analysis())
    store.add("u1", created_at=_iso(3), turns=TURNS_AB, analysis=_analysis())
    store.add("u1", created_at=_iso(6), turns=TURNS_AB, analysis=_analysis())
    res = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    times = [p["timestamp"] for p in res.json()["points"]]
    assert times == [_iso(3), _iso(6), _iso(9)]


async def test_growth_missing_report_card_is_null_score_not_zero(client, store):
    # Identified, but the stored analysis has no card for "me" (old/degraded
    # analysis) → the point exists with my_score null. NEVER zero.
    analysis = _analysis()
    del analysis["report_cards"]["Speaker A"]
    store.add("u1", created_at=_iso(1), turns=TURNS_AB, analysis=analysis)
    res = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    body = res.json()
    assert body["identified_recordings"] == 1
    assert body["points"][0]["my_score"] is None


async def test_growth_malformed_score_is_null(client, store):
    analysis = _analysis()
    analysis["report_cards"]["Speaker A"]["score"] = "great"
    store.add("u1", created_at=_iso(1), turns=TURNS_AB, analysis=analysis)
    res = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    assert res.json()["points"][0]["my_score"] is None


# ---------------------------------------------------------------------------
# Manual re-tags flow into the chart WITHOUT re-analysis
# ---------------------------------------------------------------------------

async def test_growth_manual_retag_of_you_removes_point(client, store):
    # The machine said Speaker A is "You", but the user manually re-tagged that
    # speaker as "Alex" (i.e. the voiceprint was wrong). The manual label is the
    # top rung — the recording is no longer confidently "me".
    store.add(
        "u1", created_at=_iso(1), turns=TURNS_AB, analysis=_analysis(),
        manual_speaker_labels={"Speaker A": "Alex"},
    )
    res = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    body = res.json()
    assert body["total_recordings"] == 1
    assert body["identified_recordings"] == 0
    assert body["points"] == []


async def test_growth_manual_partner_name_flows_into_partner_names(client, store):
    store.add(
        "u1", created_at=_iso(1), turns=TURNS_AB, analysis=_analysis(),
        manual_speaker_labels={"Speaker B": "Sam"},
    )
    res = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    assert res.json()["points"][0]["partner_names"] == ["Sam"]


# ---------------------------------------------------------------------------
# Partner names — real names only, generic/relative labels stay anonymous
# ---------------------------------------------------------------------------

async def test_growth_partner_names_from_name_rung(client, store):
    store.add(
        "u1", created_at=_iso(1), turns=TURNS_AB,
        analysis=_analysis(partner_label=("Linda", "name")),
    )
    res = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    assert res.json()["points"][0]["partner_names"] == ["Linda"]


async def test_growth_generic_and_voice_partners_stay_anonymous(client, store):
    # "Speaker B" / "Higher voice" are per-recording relative labels — grouping
    # by them across recordings would fabricate an identity. Honest: no name.
    store.add(
        "u1", created_at=_iso(1), turns=TURNS_AB,
        analysis=_analysis(partner_label=("Higher voice", "voice")),
    )
    store.add(
        "u1", created_at=_iso(2), turns=TURNS_AB, analysis=_analysis(),
    )
    res = await client.get("/growth", headers={"X-Test-Uid": "u1"})
    for p in res.json()["points"]:
        assert p["partner_names"] == []


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

async def test_growth_is_uid_scoped(client, store):
    store.add("u1", created_at=_iso(1), turns=TURNS_AB, analysis=_analysis())
    res = await client.get("/growth", headers={"X-Test-Uid": "u2"})
    body = res.json()
    assert body["total_recordings"] == 0
    assert body["points"] == []
