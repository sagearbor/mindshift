"""Endpoint tests for routers/sessions.py — Track 2's live-session ingest,
"what you could have said" reflection, and the therapist-dashboard list.

GCS/LLM are never touched: an in-memory fake store is injected at
``app.state.recordings_store`` (the DI style of test_recordings.py) and the
LLM is the conftest ``mock_llm`` MagicMock, whose ``complete`` answers the
BATCH analysis prompt and the REFLECTION prompt differently (told apart by
the system prompt) so one fixture drives the whole post-ingest tail. The
background task is awaited explicitly via ``routers.sessions.BACKGROUND_TASKS``
so the async path is observed deterministically.
"""

import asyncio
import json
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

import live_sessions
import main
from main import app, init_db
from routers import sessions as sessions_router

pytestmark = pytest.mark.anyio

SESSION_ID = "0f2b1c9e-5a4d-4e8f-9c1a-2b3c4d5e6f70"
STARTED = "2026-08-24T18:05:00+00:00"
ENDED = "2026-08-24T18:07:00+00:00"


def _turn(i, speaker, text, *, is_self=None, tone=None, pid=None, start=None):
    start = i * 3.0 if start is None else start
    return {
        "type": "turn_local", "session_id": SESSION_ID, "speaker": speaker,
        "speaker_person_id": pid, "speaker_match_score": 0.8 if pid else None,
        "is_self": is_self, "text": text, "start_time": start,
        "end_time": start + 2.5, "transcript_source": "on-device",
        "prosody": {"rms_dbfs": -20.0, "pitch_hz": None, "speech_rate": 3.1},
        "text_tone": tone, "suggestion": None, "suggestion_source": None,
    }


TURNS = [
    _turn(0, "Speaker A", "Hey Mom, I got your message.", is_self=True,
          tone={"warmth": 80, "frustration": 10}),
    _turn(1, "Speaker B", "You never call back.", pid="p-mom"),
    _turn(2, "Speaker A", "I was working, I told you that.", is_self=True,
          tone={"warmth": 20, "frustration": 75}),
    _turn(3, "Speaker B", "That's what you always say.", pid="p-mom"),
    _turn(4, "Speaker A", "Fine. Whatever you want.", is_self=True,
          tone={"label": "defensive"}),
    _turn(5, "Speaker B", "Don't be like that.", pid="p-mom"),
]
IDENTITIES = [{
    "type": "speaker_identity", "session_id": SESSION_ID, "speaker": "Speaker B",
    "person_id": "p-mom", "display_name": "Mom", "is_self": False, "score": 0.81,
}]


def _body(**overrides):
    body = {
        "session_id": SESSION_ID, "started_at": STARTED, "ended_at": ENDED,
        "mode": "earpiece", "turns": TURNS, "speaker_identities": IDENTITIES,
    }
    body.update(overrides)
    return body


def _analysis_json(n_turns, speakers):
    return json.dumps({
        "per_turn": [
            {"heat": 15 + 10 * i, "markers": [], "trigger_phrase": None}
            for i in range(n_turns)
        ],
        "requests": [],
        "narrative": "You both kept reaching for each other.",
        "speaker_names": {},
        "report_cards": {
            sp: {"score": 64, "headline": f"{sp} stayed present",
                 "did_well": "Kept talking.", "work_on": "Slow down."}
            for sp in speakers
        },
    })


REFLECTIONS = [
    {"turn_index": 0, "could_have_said": "Hey Mom — thanks for the message.",
     "why": "Already warm.", "tone_read": "warm"},
    {"turn_index": 2, "could_have_said": "I hear you. Work swallowed me — I'm sorry.",
     "why": "Owns it without defending.", "tone_read": "defensive"},
    {"turn_index": 4, "could_have_said": "I don't want to fight. Can we start over?",
     "why": "Names the wish instead of shutting down.", "tone_read": "shut down"},
]


def _llm_side_effect(*, reflect_payload=None, analysis_ok=True):
    """Route the MagicMock's complete() by system prompt."""
    def complete(system, user, max_tokens=512, **_):
        if system.startswith(live_sessions.REFLECT_SYSTEM_PROMPT):
            if reflect_payload is None:
                return json.dumps({"reflections": REFLECTIONS})
            return reflect_payload
        if not analysis_ok:
            return "not json at all"
        n = user.count("\n") - 1  # "Conversation (...)" header + N numbered lines
        n = user.split("Conversation (")[1].split(" turns")[0]
        return _analysis_json(int(n), ["Speaker A", "Speaker B"])
    return complete


# ---------------------------------------------------------------------------
# In-memory fake store — the recording read/write + share methods this
# router and the reads it feeds (GET /recordings, /growth, /sessions) use.
# ---------------------------------------------------------------------------

class FakeLiveStore:
    def __init__(self):
        self._by_uid: dict = {}     # {uid: {rid: {meta, turns, analysis}}}
        self._shares: dict = {}     # {recipient: {rid: grant}}
        self.save_calls = 0
        self.update_calls = 0

    async def save_live_session(self, uid, recording_id, *, meta, turns, analysis):
        self.save_calls += 1
        slot = self._by_uid.setdefault(uid, {})
        written = dict(meta)
        existing = slot.get(recording_id)
        if existing:
            for key in ("manual_speaker_labels", "shares"):
                if key in existing["meta"] and key not in written:
                    written[key] = existing["meta"][key]
        slot[recording_id] = {"meta": written, "turns": turns, "analysis": analysis}
        return written

    async def update_analysis(self, uid, recording_id, analysis):
        self.update_calls += 1
        r = self._by_uid.get(uid, {}).get(recording_id)
        if r is None:
            return False
        r["analysis"] = analysis
        return True

    async def list_recordings(self, uid):
        out = [
            {**r["meta"], "has_analysis": r["analysis"] is not None}
            for r in self._by_uid.get(uid, {}).values()
        ]
        out.sort(key=lambda m: m["created_at"], reverse=True)
        return out

    async def get_recording(self, uid, recording_id):
        r = self._by_uid.get(uid, {}).get(recording_id)
        if r is None:
            return None
        return {**r["meta"], "turns": r["turns"], "analysis": r["analysis"]}

    async def update_manual_speaker_labels(self, uid, recording_id, labels):
        r = self._by_uid.get(uid, {}).get(recording_id)
        if r is None:
            return None
        r["meta"]["manual_speaker_labels"] = labels
        return r["meta"]

    async def open_media_stream(self, uid, recording_id, range_header):
        return None

    async def recording_exists(self, uid, recording_id):
        return recording_id in self._by_uid.get(uid, {})

    # Foundation B — the account's enrolled people (person views).
    voiceprints: dict = {}

    async def list_voiceprints(self, uid):
        return list(self.voiceprints.get(uid, []))

    # -- sharing (the therapist ← patient grant) --
    def grant(self, owner_uid, recording_id, recipient_uid, owner_email):
        self._shares.setdefault(recipient_uid, {})[recording_id] = {
            "owner_uid": owner_uid, "recording_id": recording_id,
            "owner_email": owner_email, "created_at": "2026-08-24T19:00:00+00:00",
        }

    async def find_share(self, recipient_uid, recording_id):
        return self._shares.get(recipient_uid, {}).get(recording_id)

    async def list_shared_with(self, recipient_uid):
        out = []
        for rid, grant in self._shares.get(recipient_uid, {}).items():
            rec = self._by_uid.get(grant["owner_uid"], {}).get(rid)
            if rec is None:
                continue
            out.append({
                **rec["meta"], "has_analysis": rec["analysis"] is not None,
                "owner_email": grant["owner_email"],
            })
        return out


@pytest.fixture
async def client():
    await init_db()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def store():
    fake = FakeLiveStore()
    app.state.recordings_store = fake
    sessions_router._REFLECT_LOCKS.clear()
    yield fake
    del app.state.recordings_store


async def _drain():
    """Await every scheduled post-ingest task (analysis + reflection)."""
    while sessions_router.BACKGROUND_TASKS:
        await asyncio.gather(*list(sessions_router.BACKGROUND_TASKS))
        # The done-callback that discards a finished task runs on the NEXT
        # loop tick; yield once so it can, else this loop spins forever on
        # an already-finished task.
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# POST /sessions/live
# ---------------------------------------------------------------------------

class TestIngest:
    async def test_stores_lite_then_full_analysis_and_reflection(self, client, store, mock_llm):
        mock_llm.complete.side_effect = _llm_side_effect()
        res = await client.post("/sessions/live", json=_body())
        assert res.status_code == 201, res.text
        body = res.json()
        rid = body["episode_id"]
        assert body["recording_id"] == rid
        assert body["created"] is True and body["turn_count"] == 6
        assert body["self_speaker"] == "Speaker A"
        assert body["analysis_status"] == "lite"
        assert body["analysis_scheduled"] is True and body["reflect_scheduled"] is True

        # Let the background tail run: batch analysis then reflection. (The
        # 201 above carried the PRE-LLM "lite" status — the response never
        # waited on the model; the lite record itself is checked in
        # test_lite_record_is_readable_before_any_llm_pass.)
        await _drain()
        assert mock_llm.complete.call_count == 2
        detail = (await client.get(f"/recordings/{rid}")).json()
        assert detail["source"] == {"type": "live", "url": None, "original_filename": None}
        assert detail["mode"] == "earpiece" and detail["session_id"] == SESSION_ID
        assert detail["media_type"] == "none"
        assert detail["created_at"] == STARTED           # when it HAPPENED
        assert detail["speaker_labels"]["Speaker A"] == {
            "display_label": "You", "label_source": "enrolled",
        }
        assert detail["speaker_labels"]["Speaker B"]["display_label"] == "Mom"
        # The phone's per-turn tone is preserved verbatim (pydantic dump —
        # unmeasured dims are explicit nulls, never zeros).
        tone = detail["turns"][2]["text_tone"]
        assert tone["warmth"] == 20 and tone["frustration"] == 75 and tone["sarcasm"] is None
        assert detail["turns"][1]["speaker_person_id"] == "p-mom"
        # Listing badges it with source_type "live" + the mode.
        rows = (await client.get("/recordings")).json()["recordings"]
        assert rows[0]["id"] == rid and rows[0]["source_type"] == "live"
        assert rows[0]["mode"] == "earpiece" and rows[0]["has_analysis"] is True

        analysis = detail["analysis"]
        assert analysis["live"]["analysis_status"] == "full"
        assert [p["heat"] for p in analysis["per_turn"]] == [15, 25, 35, 45, 55, 65]
        assert analysis["report_cards"]["Speaker A"]["score"] == 64
        # Identity labels survive the merge (voiceprint name beats LLM guess).
        assert analysis["speaker_labels"]["Speaker B"]["display_label"] == "Mom"
        assert analysis["speaker_labels"]["Speaker A"]["label_source"] == "enrolled"
        [ep] = detail["episodes"]
        assert ep["peak_heat"] == 65 and ep["self_escalation_count"] == 2
        assert [r["turn_index"] for r in analysis["live"]["could_have_said"]] == [0, 2, 4]
        assert analysis["live"]["reflection"]["turns_hash"] == live_sessions.turns_hash(detail["turns"])

        # Growth: the live session is a scored point with self tone + people.
        growth = (await client.get("/growth")).json()
        assert growth["identified_recordings"] == 1
        [pt] = growth["points"]
        assert pt["recording_id"] == rid and pt["my_score"] == 64
        assert pt["source"] == "live" and pt["mode"] == "earpiece"
        assert pt["partner_names"] == ["Mom"]
        assert pt["self_tone"]["escalation_count"] == 2
        assert pt["self_tone"]["labels"] == {"warm": 1, "frustrated": 1, "defensive": 1}
        [mom] = growth["people"]
        assert mom["display_name"] == "Mom" and mom["person_id"] == "p-mom"
        assert mom["sessions"] == 1 and mom["escalation_count"] == 2

    async def test_lite_record_is_readable_before_any_llm_pass(self, client, store, mock_llm):
        """With the LLM tail switched off, what ingest wrote is exactly what
        a reader sees while the background pass is still pending."""
        mock_llm.complete.side_effect = _llm_side_effect()
        res = await client.post("/sessions/live", json=_body(analyze=False, reflect=False))
        rid = res.json()["episode_id"]
        assert mock_llm.complete.call_count == 0
        detail = (await client.get(f"/recordings/{rid}")).json()
        assert detail["analysis"]["per_turn"] == []
        [ep] = detail["episodes"]
        assert ep["participants"] == ["You", "Mom"] and ep["mean_heat"] is None
        assert ep["self_tone_labels"] == {"warm": 1, "frustrated": 1, "defensive": 1}
        assert ep["self_escalation_count"] == 2
        assert detail["analysis"]["live"]["analysis_status"] == "lite"
        assert detail["analysis"]["live"]["could_have_said"] is None
        # Growth counts it, scores it honestly as a gap (no report card yet).
        growth = (await client.get("/growth")).json()
        [pt] = growth["points"]
        assert pt["my_score"] is None and pt["self_tone"]["escalation_count"] == 2
        # No audio → the media STREAM 404s honestly rather than 500s (the
        # client never asks: media_type "none" hides the player entirely).
        media = (await client.get(f"/recordings/{rid}/media_url")).json()
        path = media["url"].split("http://test", 1)[-1]
        assert (await client.get(path)).status_code == 404

    async def test_repost_is_idempotent_and_keeps_human_edits(self, client, store, mock_llm):
        mock_llm.complete.side_effect = _llm_side_effect()
        first = (await client.post("/sessions/live", json=_body(analyze=False, reflect=False))).json()
        rid = first["episode_id"]
        # The user names a speaker afterwards…
        res = await client.patch(
            f"/recordings/{rid}/speaker-labels", json={"labels": {"Speaker B": "Mum"}},
        )
        assert res.status_code == 200, res.text
        # …then the phone re-sends the same session (e.g. a retry).
        second = (await client.post("/sessions/live", json=_body(analyze=False, reflect=False))).json()
        assert second["episode_id"] == rid and second["created"] is False
        assert store.save_calls == 2
        assert len(store._by_uid["test-user"]) == 1
        detail = (await client.get(f"/recordings/{rid}")).json()
        assert detail["manual_speaker_labels"] == {"Speaker B": "Mum"}
        assert detail["speaker_labels"]["Speaker B"]["label_source"] == "manual"
        # Nothing scheduled, nothing billed.
        await _drain()
        assert mock_llm.complete.call_count == 0

    async def test_person_names_resolve_from_enrolled_voiceprints(self, client, store, mock_llm):
        """The phone sent only person ids (no identity events); the names
        come from the account's Foundation B voiceprint documents."""
        mock_llm.complete.side_effect = _llm_side_effect()
        store.voiceprints["test-user"] = [
            {"person_id": "self", "display_name": "You", "is_self": True},
            {"person_id": "p-mom", "display_name": "Mum", "is_self": False},
        ]
        res = await client.post(
            "/sessions/live", json=_body(speaker_identities=[], analyze=False, reflect=False),
        )
        detail = (await client.get(f"/recordings/{res.json()['episode_id']}")).json()
        assert detail["speaker_labels"]["Speaker B"] == {
            "display_label": "Mum", "label_source": "enrolled",
        }
        assert detail["episodes"][0]["participants"] == ["You", "Mum"]
        growth = (await client.get("/growth")).json()
        assert growth["points"][0]["partner_names"] == ["Mum"]
        assert growth["people"][0]["display_name"] == "Mum"

    async def test_ids_are_per_user(self, client, store):
        a = (await client.post(
            "/sessions/live", json=_body(analyze=False, reflect=False),
            headers={"X-Test-Uid": "user-a"},
        )).json()
        b = (await client.post(
            "/sessions/live", json=_body(analyze=False, reflect=False),
            headers={"X-Test-Uid": "user-b"},
        )).json()
        assert a["episode_id"] != b["episode_id"]
        # user-a can't read user-b's episode.
        res = await client.get(f"/recordings/{b['episode_id']}", headers={"X-Test-Uid": "user-a"})
        assert res.status_code == 404

    async def test_short_session_skips_analysis_but_still_reflects(self, client, store, mock_llm):
        mock_llm.complete.side_effect = _llm_side_effect(
            reflect_payload=json.dumps({"reflections": [REFLECTIONS[0]]}),
        )
        res = await client.post("/sessions/live", json=_body(turns=TURNS[:2]))
        body = res.json()
        assert body["analysis_scheduled"] is False and body["reflect_scheduled"] is True
        await _drain()
        assert mock_llm.complete.call_count == 1
        detail = (await client.get(f"/recordings/{body['episode_id']}")).json()
        assert detail["analysis"]["live"]["analysis_status"] == "lite"
        assert detail["analysis"]["per_turn"] == []
        assert detail["analysis"]["live"]["could_have_said"] == [REFLECTIONS[0]]

    async def test_analysis_failure_is_recorded_not_raised(self, client, store, mock_llm):
        mock_llm.complete.side_effect = _llm_side_effect(
            analysis_ok=False, reflect_payload="still not json",
        )
        res = await client.post("/sessions/live", json=_body())
        assert res.status_code == 201
        rid = res.json()["episode_id"]
        await _drain()
        detail = (await client.get(f"/recordings/{rid}")).json()
        live = detail["analysis"]["live"]
        assert live["analysis_status"] == "failed"
        assert "invalid JSON" in live["analysis_error"]
        assert live["could_have_said"] is None
        # Still a readable, listed episode with its tone summary intact.
        assert live["tone_summary"]["self"]["escalation_turns"] == [2, 4]
        assert detail["episodes"][0]["self_escalation_count"] == 2

    async def test_no_self_turns_means_no_reflection(self, client, store, mock_llm):
        mock_llm.complete.side_effect = _llm_side_effect()
        turns = [dict(t, is_self=None) for t in TURNS]
        res = await client.post("/sessions/live", json=_body(turns=turns))
        body = res.json()
        assert body["self_speaker"] is None and body["reflect_scheduled"] is False
        await _drain()
        assert mock_llm.complete.call_count == 1   # analysis only
        detail = (await client.get(f"/recordings/{body['episode_id']}")).json()
        assert detail["analysis"]["live"]["tone_summary"]["self"] is None
        assert detail["analysis"]["speaker_labels"]["Speaker A"]["label_source"] != "enrolled"
        # Not a growth point: the user isn't confidently in it.
        growth = (await client.get("/growth")).json()
        assert growth["identified_recordings"] == 0 and growth["total_recordings"] == 1

    async def test_validation(self, client, store):
        # A turn from another session is refused at the door.
        bad = _body(turns=[dict(TURNS[0], session_id="other")])
        assert (await client.post("/sessions/live", json=bad)).status_code == 422
        assert (await client.post("/sessions/live", json=_body(started_at="yesterday"))).status_code == 422
        assert (await client.post("/sessions/live", json=_body(mode="loud"))).status_code == 422
        assert (await client.post("/sessions/live", json=_body(turns=[]))).status_code == 422
        assert store.save_calls == 0

    async def test_storage_disabled_is_503(self, client):
        assert not hasattr(app.state, "recordings_store")
        res = await client.post("/sessions/live", json=_body())
        assert res.status_code == 503
        assert res.json()["detail"] == "recording storage is not enabled"

    async def test_requires_auth(self, client, store):
        from auth import get_current_uid
        saved = app.dependency_overrides.pop(get_current_uid)
        try:
            res = await client.post("/sessions/live", json=_body())
            assert res.status_code == 401
        finally:
            app.dependency_overrides[get_current_uid] = saved

    def test_bounds_mirror_main(self):
        assert sessions_router.LIVE_MAX_TURNS == main.ANALYZE_MAX_TURNS
        assert sessions_router.LIVE_MAX_TRANSCRIPT_CHARS == main.ANALYZE_MAX_TRANSCRIPT_CHARS
        rid = sessions_router.live_recording_id("u", "s")
        assert uuid.UUID(rid) and rid == sessions_router.live_recording_id("u", "s")


# ---------------------------------------------------------------------------
# POST /episodes/{id}/reflect
# ---------------------------------------------------------------------------

class TestReflect:
    async def _ingest(self, client, mock_llm, **overrides):
        mock_llm.complete.side_effect = _llm_side_effect()
        body = _body(analyze=False, reflect=False)
        body.update(overrides)
        res = await client.post("/sessions/live", json=body)
        assert res.status_code == 201
        return res.json()["episode_id"]

    async def test_on_demand_then_cached_then_forced(self, client, store, mock_llm):
        rid = await self._ingest(client, mock_llm)
        res = await client.post(f"/episodes/{rid}/reflect")
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["cached"] is False and body["self_speaker"] == "Speaker A"
        assert [r["turn_index"] for r in body["could_have_said"]] == [0, 2, 4]
        assert body["could_have_said"][1]["tone_read"] == "defensive"
        assert body["reflected_at"]
        assert mock_llm.complete.call_count == 1
        # The prompt tagged the user's turns and named the partner.
        _, kwargs = mock_llm.complete.call_args
        assert "0. (YOU) Hey Mom" in kwargs["user"] and "(Mom) You never call back" in kwargs["user"]
        assert "Session mode: earpiece" in kwargs["user"]

        # Second view: served from the cache, no spend.
        again = (await client.post(f"/episodes/{rid}/reflect")).json()
        assert again["cached"] is True and again["could_have_said"] == body["could_have_said"]
        assert mock_llm.complete.call_count == 1

        # force=true re-bills once.
        forced = (await client.post(f"/episodes/{rid}/reflect?force=true")).json()
        assert forced["cached"] is False
        assert mock_llm.complete.call_count == 2

    async def test_concurrent_requests_bill_once(self, client, store, mock_llm):
        rid = await self._ingest(client, mock_llm)
        results = await asyncio.gather(
            client.post(f"/episodes/{rid}/reflect"),
            client.post(f"/episodes/{rid}/reflect"),
        )
        assert {r.status_code for r in results} == {200}
        assert mock_llm.complete.call_count == 1
        assert sorted(r.json()["cached"] for r in results) == [False, True]

    async def test_changed_transcript_invalidates_cache(self, client, store, mock_llm):
        rid = await self._ingest(client, mock_llm)
        await client.post(f"/episodes/{rid}/reflect")
        # The phone re-POSTs with corrected words → the old reflection is stale.
        edited = [dict(t, text=t["text"] + " (edited)") for t in TURNS]
        await client.post("/sessions/live", json=_body(turns=edited, analyze=False, reflect=False))
        res = (await client.post(f"/episodes/{rid}/reflect")).json()
        assert res["cached"] is False
        assert mock_llm.complete.call_count == 2

    async def test_reflects_on_an_upload_with_enrolled_voice(self, client, store, mock_llm):
        """Not just live sessions: an upload whose stored labels carry the
        enrolled "You" has a self speaker too."""
        mock_llm.complete.side_effect = _llm_side_effect()
        rid = str(uuid.uuid4())
        store._by_uid.setdefault("test-user", {})[rid] = {
            "meta": {"id": rid, "created_at": "2026-08-01T10:00:00+00:00",
                     "filename": "talk.m4a", "title": "Talk", "media_type": "audio",
                     "duration_seconds": 18.0, "source": {"type": "upload", "url": None}},
            "turns": [{"speaker": t["speaker"], "text": t["text"],
                       "start_time": t["start_time"], "end_time": t["end_time"]} for t in TURNS],
            "analysis": {
                "per_turn": [], "per_speaker": {}, "dynamics": {}, "narrative": "",
                "report_cards": {},
                "speaker_labels": {
                    "Speaker A": {"display_label": "You", "label_source": "enrolled"},
                    "Speaker B": {"display_label": "Speaker B", "label_source": "generic"},
                },
            },
        }
        res = await client.post(f"/episodes/{rid}/reflect")
        assert res.status_code == 200, res.text
        assert [r["turn_index"] for r in res.json()["could_have_said"]] == [0, 2, 4]
        detail = (await client.get(f"/recordings/{rid}")).json()
        assert len(detail["analysis"]["live"]["could_have_said"]) == 3
        # The upload's own analysis fields are untouched by the reflection.
        assert detail["analysis"]["speaker_labels"]["Speaker A"]["label_source"] == "enrolled"

    async def test_errors(self, client, store, mock_llm):
        # 404 for a missing / foreign episode.
        rid = await self._ingest(client, mock_llm)
        missing = str(uuid.uuid4())
        assert (await client.post(f"/episodes/{missing}/reflect")).status_code == 404
        res = await client.post(f"/episodes/{rid}/reflect", headers={"X-Test-Uid": "user-b"})
        assert res.status_code == 404
        # 403 when the episode was SHARED with the caller (visible, read-only).
        store.grant("test-user", rid, "user-b", "patient@example.com")
        res = await client.post(f"/episodes/{rid}/reflect", headers={"X-Test-Uid": "user-b"})
        assert res.status_code == 403
        # 422 when nobody in the episode is the caller.
        anon = [dict(t, is_self=None) for t in TURNS]
        anon_id = (await client.post(
            "/sessions/live", json=_body(session_id="anon-1", turns=[dict(t, session_id="anon-1") for t in anon],
                                          analyze=False, reflect=False),
        )).json()["episode_id"]
        res = await client.post(f"/episodes/{anon_id}/reflect")
        assert res.status_code == 422
        # 502 when the LLM never yields a parseable answer (after the retry).
        mock_llm.complete.side_effect = None
        mock_llm.complete.return_value = "nope"
        res = await client.post(f"/episodes/{rid}/reflect")
        assert res.status_code == 502
        assert mock_llm.complete.call_count == 2   # one retry, then honest
        # A retry that recovers is served normally.
        mock_llm.complete.side_effect = ["garbage", json.dumps({"reflections": REFLECTIONS})]
        res = await client.post(f"/episodes/{rid}/reflect")
        assert res.status_code == 200 and res.json()["cached"] is False

    async def test_storage_disabled(self, client):
        res = await client.post(f"/episodes/{uuid.uuid4()}/reflect")
        assert res.status_code == 503


# ---------------------------------------------------------------------------
# GET /sessions — therapist dashboard
# ---------------------------------------------------------------------------

class TestDashboardList:
    async def test_own_and_shared_sessions_in_dashboard_shape(self, client, store, mock_llm):
        mock_llm.complete.side_effect = _llm_side_effect()
        # The PATIENT (user-a) records a live session and shares it with the
        # THERAPIST (test-user).
        res = await client.post(
            "/sessions/live", json=_body(), headers={"X-Test-Uid": "user-a"},
        )
        patient_rid = res.json()["episode_id"]
        await _drain()
        store.grant("user-a", patient_rid, "test-user", "patient@example.com")
        # The therapist's own live session, unanalyzed beyond lite.
        own = (await client.post(
            "/sessions/live",
            json=_body(session_id="own-1", turns=[dict(t, session_id="own-1") for t in TURNS[:2]],
                       mode="therapist", analyze=False, reflect=False),
        )).json()["episode_id"]

        body = (await client.get("/sessions")).json()
        sessions = body["sessions"]
        assert [s["id"] for s in sessions] == [own, patient_rid] or \
            [s["id"] for s in sessions] == [patient_rid, own]
        by_id = {s["id"]: s for s in sessions}
        mine = by_id[own]
        assert mine["role"] == "You" and mine["shared"] is False
        assert mine["mode"] == "therapist" and mine["source"] == "live"
        assert mine["avgPleasantness"] is None            # no heats yet
        assert mine["analysisStatus"] == "lite"
        theirs = by_id[patient_rid]
        assert theirs["role"] == "patient@example.com" and theirs["shared"] is True
        assert theirs["avgPleasantness"] == 60            # 100 − mean heat (40)
        assert theirs["turns"][0]["speaker"] == "You" and theirs["turns"][0]["isSelf"] is True
        assert theirs["turns"][1]["speaker"] == "Mom"
        assert theirs["turns"][2]["toneLabel"] == "frustrated" and theirs["turns"][2]["escalated"] is True
        assert theirs["turns"][0]["toneScores"] == {"pleasantness": 85, "warmth": 80}
        assert [r["turn_index"] for r in theirs["couldHaveSaid"]] == [0, 2, 4]
        assert theirs["toneSummary"]["people"][0]["display_name"] == "Mom"

    async def test_empty_and_disabled(self, client, store):
        assert (await client.get("/sessions")).json() == {"sessions": []}
        del app.state.recordings_store
        try:
            assert (await client.get("/sessions")).status_code == 503
        finally:
            app.state.recordings_store = store
