"""Cost guardrails — usage counters, soft daily quotas, owner visibility.

The load-bearing property these tests defend: a spent budget must DEGRADE the
cloud half of the product and never touch the parts the user would call data
loss. Concretely — with every quota exhausted, a live session still emits its
``transcript`` events, still accepts the phone's ``turn_local`` frames, and
still says exactly once what stopped and when it resets.
"""

import json
import uuid
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

import usage_meter
from main import app
from tests.test_audio_pipeline import (  # noqa: E402 — shared WS doubles
    FRAME_100MS,
    FakeTTS,
    RecordingSegmentTranscriber,
    _turn_local,
    open_ws,
    recv_until,
)
from audio_pipeline import TranscriptSegment

TEST_UID = "test-user"

MOCK_LLM_JSON = json.dumps({
    "suggestions": ["I hear you.", "Tell me more.", "That sounds hard."],
    "importance": 5,
})

# /respond's contract (suggestions + a full tone_score) — see conftest.
MOCK_RESPOND_TONE_JSON = json.dumps({
    "suggestions": ["I hear you.", "Tell me more.", "That sounds hard."],
    "tone_score": {
        "warmth": 60, "defensiveness": 30, "sarcasm": 10,
        "constructiveness": 55, "overall": 65,
    },
})


@pytest.fixture(autouse=True)
def _clean_meter():
    """Every test starts with empty counters and no store — the meter is a
    process-wide singleton, so leaking counters between tests would make the
    quota assertions order-dependent."""
    usage_meter.meter().reset()
    usage_meter.bind_store(None)
    yield
    usage_meter.meter().reset()
    usage_meter.bind_store(None)


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

class TestCounters:
    def test_add_and_total(self):
        usage_meter.record("u1", **{usage_meter.KEY_STT_SECONDS: 12.5})
        usage_meter.record("u1", **{usage_meter.KEY_STT_SECONDS: 7.5})
        assert usage_meter.totals("u1")[usage_meter.KEY_STT_SECONDS] == 20.0

    def test_counters_are_per_uid(self):
        usage_meter.record("u1", **{usage_meter.KEY_CALLS_STARTED: 1})
        assert usage_meter.totals("u2") == {}

    def test_non_positive_and_nonsense_values_are_ignored(self):
        usage_meter.record("u1", **{usage_meter.KEY_STT_SECONDS: -5})
        usage_meter.record_counters("u1", {usage_meter.KEY_STT_SECONDS: "lots"})
        assert usage_meter.totals("u1") == {}

    def test_llm_usage_is_keyed_by_call_site(self):
        with usage_meter.attribute("u1", usage_meter.SITE_LIVE_SUGGESTION):
            usage_meter.note_llm_usage({"input_tokens": 300, "output_tokens": 80})
        with usage_meter.attribute("u1", usage_meter.SITE_BATCH_ANALYSIS):
            usage_meter.note_llm_usage({"input_tokens": 4000, "output_tokens": 900})
        totals = usage_meter.totals("u1")
        assert totals[usage_meter.llm_key("live_suggestion", "input_tokens")] == 300
        assert totals[usage_meter.llm_key("batch_analysis", "output_tokens")] == 900
        # One call recorded per site, even when the provider sent no usage.
        assert totals[usage_meter.llm_key("live_suggestion", "calls")] == 1

    def test_usage_outside_a_scope_is_bucketed_not_dropped(self):
        usage_meter.note_llm_usage({"input_tokens": 42})
        totals = usage_meter.totals(usage_meter.UNATTRIBUTED_UID)
        assert totals[usage_meter.llm_key("unattributed", "input_tokens")] == 42

    def test_hedge_surcharge_lands_on_the_same_uid_and_site(self):
        with usage_meter.attribute("u1", usage_meter.SITE_LIVE_SUGGESTION):
            usage_meter.note_hedge_extra(311)
        key = usage_meter.llm_key("live_suggestion", "hedge_extra_input_tokens")
        assert usage_meter.totals("u1")[key] == 311

    def test_llm_client_record_usage_feeds_the_meter(self):
        """The real LLMClient hook, not a hand-rolled call: this is what
        attributes every Anthropic response in the server."""
        from llm_client import LLMClient

        client = LLMClient.__new__(LLMClient)  # no provider/credentials needed
        client.model = "claude-haiku-4-5"
        usage = MagicMock(
            input_tokens=250, output_tokens=90,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        with usage_meter.attribute("u1", usage_meter.SITE_LIVE_SUGGESTION):
            client._record_usage(usage)
        totals = usage_meter.totals("u1")
        assert totals[usage_meter.llm_key("live_suggestion", "input_tokens")] == 250
        assert totals[usage_meter.llm_key("live_suggestion", "output_tokens")] == 90

    def test_attribute_scope_survives_a_worker_thread(self):
        """Every LLM call in this codebase runs under asyncio.to_thread, which
        COPIES the context — if that stopped holding, all live-session tokens
        would silently land in the unattributed bucket."""
        import asyncio

        def blocking() -> None:
            usage_meter.note_llm_usage({"input_tokens": 7})

        async def run() -> None:
            with usage_meter.attribute("u1", usage_meter.SITE_LIVE_NUDGE):
                await asyncio.to_thread(blocking)

        asyncio.run(run())
        assert usage_meter.totals("u1")[
            usage_meter.llm_key("live_nudge", "input_tokens")
        ] == 7

    def test_summarize_groups_by_site_and_totals(self):
        with usage_meter.attribute("u1", usage_meter.SITE_LIVE_SUGGESTION):
            usage_meter.note_llm_usage({"input_tokens": 100, "output_tokens": 20})
        usage_meter.record("u1", **{usage_meter.KEY_LIVE_MINUTES: 30})
        summary = usage_meter.summarize(usage_meter.totals("u1"))
        assert summary["llm_input_tokens"] == 100
        assert summary["llm_output_tokens"] == 20
        assert summary["live_minutes"] == 30
        assert summary["llm"]["live_suggestion"]["calls"] == 1


class TestPersistence:
    def test_flush_writes_one_shard_per_uid_day(self):
        import asyncio

        written = []

        class FakeStore:
            async def write_usage_shard(self, uid, day, instance, counters):
                written.append((uid, day, instance, dict(counters)))

        usage_meter.bind_store(FakeStore())
        usage_meter.record("u1", **{usage_meter.KEY_STT_SECONDS: 5})
        usage_meter.record("u2", **{usage_meter.KEY_STT_SECONDS: 9})
        assert asyncio.run(usage_meter.meter().flush()) == 2
        assert {row[0] for row in written} == {"u1", "u2"}
        # Nothing new -> nothing rewritten (the shard is a full overwrite, so
        # a clean tick must not burn a Class A operation per user).
        assert asyncio.run(usage_meter.meter().flush()) == 0

    def test_flush_failure_keeps_the_counters_for_the_next_tick(self):
        import asyncio

        class BrokenStore:
            calls = 0

            async def write_usage_shard(self, *a, **kw):
                BrokenStore.calls += 1
                raise RuntimeError("GCS is having a day")

        usage_meter.bind_store(BrokenStore())
        usage_meter.record("u1", **{usage_meter.KEY_STT_SECONDS: 5})
        assert asyncio.run(usage_meter.meter().flush()) == 0
        assert asyncio.run(usage_meter.meter().flush()) == 0
        assert BrokenStore.calls == 2  # retried, not dropped
        assert usage_meter.totals("u1")[usage_meter.KEY_STT_SECONDS] == 5

    def test_flush_prunes_finished_days_but_never_unflushed_ones(self):
        """A Cloud Run instance can live for days across thousands of
        accounts — yesterday's flushed counters must not stay resident."""
        import asyncio

        class FakeStore:
            async def write_usage_shard(self, *a, **kw):
                pass

        meter = usage_meter.meter()
        usage_meter.bind_store(FakeStore())
        usage_meter.record("u1", **{usage_meter.KEY_STT_SECONDS: 5})
        yesterday = ("u-old", "2020-01-01")
        with meter._lock:
            meter._own[yesterday] = {usage_meter.KEY_STT_SECONDS: 1}
            meter._dirty.add(yesterday)
        asyncio.run(meter.flush())          # writes both, then prunes
        assert yesterday not in meter._own
        # Today's stays resident for the quota check.
        assert usage_meter.totals("u1")[usage_meter.KEY_STT_SECONDS] == 5

    def test_prune_never_drops_counters_that_failed_to_flush(self):
        import asyncio

        class BrokenStore:
            async def write_usage_shard(self, *a, **kw):
                raise RuntimeError("nope")

        meter = usage_meter.meter()
        usage_meter.bind_store(BrokenStore())
        yesterday = ("u-old", "2020-01-01")
        with meter._lock:
            meter._own[yesterday] = {usage_meter.KEY_STT_SECONDS: 1}
            meter._dirty.add(yesterday)
        asyncio.run(meter.flush())
        assert meter._own[yesterday][usage_meter.KEY_STT_SECONDS] == 1

    def test_prime_folds_in_other_instances(self):
        import asyncio

        class FakeStore:
            async def read_usage_totals(self, uid, day, *, exclude_instance=None):
                assert exclude_instance == usage_meter.INSTANCE_ID
                return {usage_meter.KEY_STT_SECONDS: 1000}

        usage_meter.bind_store(FakeStore())
        usage_meter.record("u1", **{usage_meter.KEY_STT_SECONDS: 5})
        asyncio.run(usage_meter.prime("u1"))
        assert usage_meter.totals("u1")[usage_meter.KEY_STT_SECONDS] == 1005


# ---------------------------------------------------------------------------
# Quota arithmetic
# ---------------------------------------------------------------------------

class TestQuotas:
    def test_under_cap_is_never_blocked(self, monkeypatch):
        monkeypatch.setattr(usage_meter, "DAILY_LLM_TOKEN_CAP", 1000)
        with usage_meter.attribute("u1", usage_meter.SITE_LIVE_SUGGESTION):
            usage_meter.note_llm_usage({"input_tokens": 999})
        assert usage_meter.check("u1", "cloud_suggestions") is None

    def test_over_cap_reports_which_limit_and_when_it_resets(self, monkeypatch):
        monkeypatch.setattr(usage_meter, "DAILY_LLM_TOKEN_CAP", 1000)
        with usage_meter.attribute("u1", usage_meter.SITE_LIVE_SUGGESTION):
            usage_meter.note_llm_usage({"input_tokens": 900, "output_tokens": 200})
        exceeded = usage_meter.check("u1", "cloud_suggestions")
        assert exceeded is not None
        assert exceeded.limit == "llm_tokens"
        assert exceeded.used == 1100
        assert exceeded.resets_at.endswith("T00:00:00Z")
        notice = exceeded.notice()
        assert notice["type"] == "quota_notice"
        assert notice["stopped"] == ["cloud_suggestions"]
        assert "transcript" in notice["still_working"]
        assert "1100" in notice["message"] and notice["resets_at"] in notice["message"]

    def test_token_cap_sums_every_call_site(self, monkeypatch):
        monkeypatch.setattr(usage_meter, "DAILY_LLM_TOKEN_CAP", 1000)
        for site in (usage_meter.SITE_LIVE_SUGGESTION, usage_meter.SITE_REFLECTION):
            with usage_meter.attribute("u1", site):
                usage_meter.note_llm_usage({"input_tokens": 600})
        assert usage_meter.check("u1", "batch_analysis") is not None

    def test_zero_cap_means_no_cap(self, monkeypatch):
        monkeypatch.setattr(usage_meter, "DAILY_LLM_TOKEN_CAP", 0)
        with usage_meter.attribute("u1", usage_meter.SITE_LIVE_SUGGESTION):
            usage_meter.note_llm_usage({"input_tokens": 10_000_000})
        assert usage_meter.check("u1", "cloud_suggestions") is None

    def test_one_exhausted_limit_does_not_stop_an_unrelated_feature(
        self, monkeypatch,
    ):
        """Spending the STT budget must not also stop cloud coaching — the
        limits are separate budgets, not one shared kill switch."""
        monkeypatch.setattr(usage_meter, "DAILY_STT_SECONDS_CAP", 10)
        monkeypatch.setattr(usage_meter, "DAILY_LLM_TOKEN_CAP", 1_000_000)
        usage_meter.record("u1", **{usage_meter.KEY_STT_SECONDS: 99})
        assert usage_meter.check("u1", "cloud_transcription") is not None
        assert usage_meter.check("u1", "cloud_suggestions") is None

    def test_unknown_feature_and_unattributed_uid_are_never_blocked(
        self, monkeypatch,
    ):
        monkeypatch.setattr(usage_meter, "DAILY_LLM_TOKEN_CAP", 1)
        usage_meter.record(usage_meter.UNATTRIBUTED_UID, **{
            usage_meter.KEY_STT_SECONDS: 10_000,
        })
        assert usage_meter.check(usage_meter.UNATTRIBUTED_UID, "cloud_suggestions") is None
        assert usage_meter.check("u1", "not_a_feature") is None


# ---------------------------------------------------------------------------
# Live session — the degradation path
# ---------------------------------------------------------------------------

SEGMENTS = [
    TranscriptSegment(text="You never listen to me.", speaker=0,
                      start_time=0.0, end_time=1.0),
]


@pytest.fixture
def quota_ws():
    """A live-session TestClient whose transcriber RECORDS the frames it was
    given, so a test can prove Deepgram was (or was not) fed."""
    for attr in ("transcriber_factory", "tts_client", "diarizer_factory",
                 "monotonic_clock", "recordings_store"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)
    mock_llm = MagicMock()
    mock_llm.complete.return_value = MOCK_LLM_JSON
    app.state.llm_client = mock_llm
    transcriber = RecordingSegmentTranscriber(list(SEGMENTS))
    app.state.transcriber_factory = lambda: transcriber
    app.state.tts_client = FakeTTS()
    app.state.transcriber = transcriber  # handed to the test
    try:
        yield TestClient(app), transcriber, mock_llm
    finally:
        for attr in ("transcriber_factory", "tts_client", "transcriber"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)


class TestLiveDegradation:
    def test_exhausted_llm_budget_keeps_the_transcript_and_says_why(
        self, quota_ws, monkeypatch,
    ):
        client, transcriber, mock_llm = quota_ws
        monkeypatch.setattr(usage_meter, "DAILY_LLM_TOKEN_CAP", 100)
        usage_meter.record(TEST_UID, **{
            usage_meter.llm_key("live_suggestion", "input_tokens"): 500,
        })
        sid = str(uuid.uuid4())
        with open_ws(client, f"/ws/session/{sid}") as ws:
            ws.send_bytes(FRAME_100MS)
            transcript, seen = recv_until(ws, lambda m: m.get("type") == "transcript")
            assert transcript["text"] == "You never listen to me."
            notice, _ = recv_until(ws, lambda m: m.get("type") == "quota_notice")

        # The client was told exactly what stopped, what didn't, and when.
        assert notice["limit"] == "llm_tokens"
        assert notice["stopped"] == ["cloud_suggestions"]
        assert "transcript" in notice["still_working"]
        assert "on_device_coaching" in notice["still_working"]
        assert notice["resets_at"].endswith("T00:00:00Z")
        # And no LLM credit was spent for this turn.
        assert mock_llm.complete.call_count == 0
        # Transcription itself was NOT stopped — the STT budget is separate.
        assert transcriber.frames == [FRAME_100MS]

    def test_quota_notice_is_sent_once_not_per_turn(self, quota_ws, monkeypatch):
        client, transcriber, _ = quota_ws
        transcriber._segments = [
            TranscriptSegment(text=f"turn {i}", speaker=0,
                              start_time=float(i), end_time=float(i) + 1)
            for i in range(3)
        ]
        monkeypatch.setattr(usage_meter, "DAILY_LLM_TOKEN_CAP", 1)
        usage_meter.record(TEST_UID, **{
            usage_meter.llm_key("live_suggestion", "input_tokens"): 500,
        })
        sid = str(uuid.uuid4())
        seen: list[dict] = []
        with open_ws(client, f"/ws/session/{sid}") as ws:
            ws.send_bytes(FRAME_100MS)  # yields all three queued segments
            for _ in range(6):
                seen.append(json.loads(ws.receive_text()))
                if len([m for m in seen if m.get("type") == "transcript"]) == 3 \
                        and any(m.get("type") == "quota_notice" for m in seen):
                    break
        assert len([m for m in seen if m.get("type") == "transcript"]) == 3
        assert len([m for m in seen if m.get("type") == "quota_notice"]) == 1

    def test_exhausted_stt_budget_stops_deepgram_not_the_socket(
        self, quota_ws, monkeypatch,
    ):
        """The other degradation: the vendor stream stops, the phone's own
        transcription (turn_local) keeps producing transcript AND coaching."""
        client, transcriber, mock_llm = quota_ws
        monkeypatch.setattr(usage_meter, "DAILY_STT_SECONDS_CAP", 10)
        monkeypatch.setattr(usage_meter, "DAILY_LLM_TOKEN_CAP", 1_000_000)
        usage_meter.record(TEST_UID, **{usage_meter.KEY_STT_SECONDS: 10_000})
        sid = str(uuid.uuid4())
        with open_ws(client, f"/ws/session/{sid}") as ws:
            ws.send_bytes(FRAME_100MS)
            notice = json.loads(ws.receive_text())
            assert notice["type"] == "quota_notice"
            assert notice["limit"] == "stt_seconds"
            assert notice["stopped"] == ["cloud_transcription"]
            assert "on_device_transcription" in notice["still_working"]
            # Not one byte went to the vendor.
            assert transcriber.frames == []

            # The on-device loop is untouched: a phone-finalized turn is
            # still accepted and still coached.
            ws.send_text(json.dumps(_turn_local(sid)))
            suggestion, _ = recv_until(
                ws, lambda m: m.get("type") == "suggestion", limit=8,
            )
            assert suggestion["suggestions"]
        assert mock_llm.complete.call_count >= 1

    def test_live_session_records_minutes_and_stt_seconds(self, quota_ws):
        client, transcriber, _ = quota_ws
        sid = str(uuid.uuid4())
        with open_ws(client, f"/ws/session/{sid}") as ws:
            ws.send_bytes(FRAME_100MS)
            recv_until(ws, lambda m: m.get("type") == "transcript")
        totals = usage_meter.totals(TEST_UID)
        # 3200 bytes of PCM16 @ 16 kHz = 0.1 s of audio sent to the vendor.
        assert totals[usage_meter.KEY_STT_SECONDS] == pytest.approx(0.1, abs=1e-6)
        assert totals[usage_meter.KEY_LIVE_MINUTES] > 0

    def test_a_session_under_budget_is_completely_unchanged(self, quota_ws):
        client, transcriber, mock_llm = quota_ws
        sid = str(uuid.uuid4())
        with open_ws(client, f"/ws/session/{sid}") as ws:
            ws.send_bytes(FRAME_100MS)
            suggestion, seen = recv_until(
                ws, lambda m: m.get("type") == "suggestion", limit=6,
            )
        assert suggestion["suggestions"]
        assert not any(m.get("type") == "quota_notice" for m in seen)


# ---------------------------------------------------------------------------
# REST quota gates
# ---------------------------------------------------------------------------

@pytest.mark.anyio
class TestRestQuotas:
    async def test_analyze_refuses_with_the_same_notice_body(
        self, client, mock_respond, monkeypatch,
    ):
        monkeypatch.setattr(usage_meter, "DAILY_LLM_TOKEN_CAP", 10)
        usage_meter.record(TEST_UID, **{
            usage_meter.llm_key("batch_analysis", "input_tokens"): 5000,
        })
        resp = await client.post("/analyze", json={
            "turns": [
                {"speaker": "A", "text": "hello"},
                {"speaker": "B", "text": "hi"},
            ],
            "context": "",
        })
        assert resp.status_code == 429
        detail = resp.json()["detail"]
        assert detail["type"] == "quota_notice"
        assert detail["limit"] == "llm_tokens"
        assert detail["resets_at"].endswith("T00:00:00Z")
        assert int(resp.headers["Retry-After"]) > 0

    async def test_respond_under_budget_works_and_bills_the_right_uid(
        self, client, mock_respond,
    ):
        """End-to-end proof that the FastAPI dependency actually binds the
        ContextVar for the handler: the LLM double reports usage from inside
        the endpoint and it must land on this uid, at the `respond` site."""
        def _complete(*args, **kwargs):
            usage_meter.note_llm_usage({"input_tokens": 1234, "output_tokens": 56})
            return MOCK_RESPOND_TONE_JSON

        mock_respond.complete.side_effect = _complete
        resp = await client.post("/respond", json={
            "transcript_turn": "You never listen to me!",
            "role": "Husband",
            "empathy_slider": 50,
        })
        assert resp.status_code == 200, resp.text
        totals = usage_meter.totals(TEST_UID)
        assert totals[usage_meter.llm_key("respond", "input_tokens")] == 1234
        assert totals[usage_meter.llm_key("respond", "output_tokens")] == 56

    async def test_model_download_is_capped(self, client, monkeypatch):
        monkeypatch.setattr(usage_meter, "DAILY_MODEL_DOWNLOAD_CAP", 2)
        usage_meter.record(TEST_UID, **{usage_meter.KEY_MODEL_DOWNLOADS: 2})
        resp = await client.get("/models/ecapa.onnx")
        assert resp.status_code == 429
        assert resp.json()["detail"]["limit"] == "model_downloads"


# ---------------------------------------------------------------------------
# Owner visibility
# ---------------------------------------------------------------------------

@pytest.mark.anyio
class TestAdminUsage:
    async def test_closed_by_default(self, client, monkeypatch):
        monkeypatch.delenv("MINDSHIFT_ADMIN_UIDS", raising=False)
        resp = await client.get("/admin/usage")
        assert resp.status_code == 404

    async def test_non_allowlisted_uid_gets_404_not_403(self, client, monkeypatch):
        monkeypatch.setenv("MINDSHIFT_ADMIN_UIDS", "someone-else")
        resp = await client.get("/admin/usage")
        assert resp.status_code == 404

    async def test_allowlisted_uid_sees_every_users_rollup(
        self, client, monkeypatch,
    ):
        monkeypatch.setenv("MINDSHIFT_ADMIN_UIDS", f"other, {TEST_UID}")
        with usage_meter.attribute("heavy-user", usage_meter.SITE_LIVE_SUGGESTION):
            usage_meter.note_llm_usage({"input_tokens": 5000, "output_tokens": 900})
        with usage_meter.attribute("light-user", usage_meter.SITE_REFLECTION):
            usage_meter.note_llm_usage({"input_tokens": 10, "output_tokens": 3})
        usage_meter.record("heavy-user", **{usage_meter.KEY_STT_SECONDS: 1800})

        resp = await client.get("/admin/usage")
        assert resp.status_code == 200
        body = resp.json()
        assert body["persistent"] is False  # no bucket configured in tests
        uids = [u["uid"] for u in body["users"]]
        assert uids[0] == "heavy-user"  # loudest spender first
        assert set(uids) == {"heavy-user", "light-user"}
        heavy = body["users"][0]
        assert heavy["llm_input_tokens"] == 5000
        assert heavy["llm_output_tokens"] == 900
        assert heavy["stt_seconds"] == 1800
        assert heavy["llm"]["live_suggestion"]["calls"] == 1
        assert body["caps"]["llm_tokens"] == usage_meter.DAILY_LLM_TOKEN_CAP

    async def test_bad_since_is_422(self, client, monkeypatch):
        monkeypatch.setenv("MINDSHIFT_ADMIN_UIDS", TEST_UID)
        assert (await client.get("/admin/usage?since=yesterday")).status_code == 422
        assert (await client.get("/admin/usage?since=2099-01-01")).status_code == 422

    async def test_window_is_bounded(self, client, monkeypatch):
        monkeypatch.setenv("MINDSHIFT_ADMIN_UIDS", TEST_UID)
        resp = await client.get("/admin/usage?since=2000-01-01")
        assert resp.status_code == 200
        assert resp.json()["days"] <= usage_meter.MAX_ROLLUP_DAYS + 1


class TestAdminAllowlistParsing:
    def test_whitespace_and_empties_are_tolerated(self, monkeypatch):
        monkeypatch.setenv("MINDSHIFT_ADMIN_UIDS", " a , ,b ")
        assert usage_meter.admin_uids() == frozenset({"a", "b"})
        assert usage_meter.is_admin("a") and not usage_meter.is_admin("")

    def test_unset_is_empty(self, monkeypatch):
        monkeypatch.delenv("MINDSHIFT_ADMIN_UIDS", raising=False)
        assert usage_meter.admin_uids() == frozenset()
