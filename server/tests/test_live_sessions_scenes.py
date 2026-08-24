"""End-to-end check of the live-session ANALYSIS path against Foundation D's
labeled scene fixtures — no audio, no LLM.

Each ``test_recording_scene_*_meta.json`` scripts a multi-speaker scene with
one SELF speaker, a per-turn ``emotion_coarse`` (what the actor performed)
and a hand-authored ``expected_nudges`` list: the self turns a coach should
react to (mild = tense/defensive rising, strong = shout/contempt), with the
explicit rule that NON-self escalations must never produce one. That is
exactly the question Track 2's aggregates answer after the fact ("which of
MY turns escalated?"), so the scene pack doubles as ground truth here:

* the meta's turns become a synthetic live session (timing rebuilt from
  ``duration_sec`` + the scene's silence gap; ``is_self`` from the speaker
  table; ``text_tone`` DERIVED from ``emotion_coarse`` by the fixed mapping
  below — the phone's on-device classifier would emit the same coarse
  vocabulary), POSTed through the real ``/sessions/live`` endpoint into the
  in-memory store;
* the stored ``tone_summary.self.escalation_turns`` must equal the nudge
  indexes, every non-self escalation must be absent, and the per-person /
  per-episode / growth views must agree with each other.

LLM passes are switched off (``analyze=false``, ``reflect=false``) — this is
the derived-analysis path only, so it runs in milliseconds and needs no key.
"""

import glob
import json
import os

import pytest
from httpx import ASGITransport, AsyncClient

from main import app, init_db
from routers import sessions as sessions_router

pytestmark = pytest.mark.anyio


class FakeLiveStore:
    """The slice of the recordings-store surface this path touches (a
    deliberate near-copy of test_sessions_live.FakeLiveStore rather than an
    import: the repo collects BOTH ``server/tests`` and the top-level
    ``tests/`` as a package named ``tests``, so a cross-module import here
    resolves to the wrong package under the full run)."""

    def __init__(self):
        self._by_uid: dict = {}

    async def save_live_session(self, uid, recording_id, *, meta, turns, analysis):
        self._by_uid.setdefault(uid, {})[recording_id] = {
            "meta": dict(meta), "turns": turns, "analysis": analysis,
        }
        return dict(meta)

    async def update_analysis(self, uid, recording_id, analysis):
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

    async def find_share(self, recipient_uid, recording_id):
        return None

    async def list_shared_with(self, recipient_uid):
        return []

    async def list_voiceprints(self, uid):
        return []

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "audio")
SCENE_METAS = sorted(glob.glob(os.path.join(FIXTURE_DIR, "test_recording_scene_*_meta.json")))

# emotion_coarse → the phone's TurnTextTone. The scene pack's coarse
# vocabulary is {neutral, angry, sad, happy}; the mapping is deliberately
# simple and one-way so the assertion below tests the AGGREGATION, not a
# clever tone model: "angry" is the only escalating coarse emotion (sad and
# happy are not escalations by live_sessions' rule; neutral scores stay under
# the dominant threshold and derive to "neutral").
COARSE_TO_TEXT_TONE = {
    "angry": {"label": "angry", "frustration": 80, "warmth": 15},
    "sad": {"label": "sad", "sadness": 80, "warmth": 40},
    "happy": {"label": "warm", "warmth": 85, "frustration": 5},
    "neutral": {"warmth": 45, "frustration": 15},
}


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _session_from_meta(meta: dict, session_id: str) -> dict:
    """The POST /sessions/live body for one scene meta."""
    gap = float(meta.get("silence_gap_sec") or 0.0)
    self_speakers = {sp for sp, info in meta["speakers"].items() if info.get("is_self")}
    turns = []
    clock = 0.0
    for turn in meta["turns"]:
        duration = float(turn["duration_sec"])
        turns.append({
            "type": "turn_local",
            "session_id": session_id,
            "speaker": turn["speaker"],
            "speaker_person_id": "self" if turn["speaker"] in self_speakers else None,
            "is_self": turn["speaker"] in self_speakers,
            "text": turn["text"],
            "start_time": round(clock, 3),
            "end_time": round(clock + duration, 3),
            "transcript_source": "on-device",
            "text_tone": COARSE_TO_TEXT_TONE[turn["emotion_coarse"]],
        })
        clock += duration + gap
    started = "2026-08-24T09:00:00+00:00"
    return {
        "session_id": session_id,
        "started_at": started,
        "ended_at": started,
        "mode": "earpiece",
        "title": meta["scene"],
        "turns": turns,
        "analyze": False,
        "reflect": False,
    }


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
    sessions_router._REFLECT_LOCK_USERS.clear()
    import main
    main._rate_limiter.reset()  # process-wide per-IP budget; see test_sessions_live
    yield fake
    del app.state.recordings_store


@pytest.mark.skipif(not SCENE_METAS, reason="Foundation D scene fixtures not present")
@pytest.mark.parametrize("meta_path", SCENE_METAS, ids=lambda p: os.path.basename(p))
async def test_scene_self_escalations_match_expected_nudges(client, store, meta_path):
    meta = _load(meta_path)
    expected = [n["after_turn_index"] for n in meta["expected_nudges"]]
    assert expected == sorted(expected)
    self_speaker = meta["self_speaker"]
    session_id = f"scene-{meta['scene']}"

    res = await client.post("/sessions/live", json=_session_from_meta(meta, session_id))
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["self_speaker"] == self_speaker
    assert body["turn_count"] == len(meta["turns"])

    detail = (await client.get(f"/recordings/{body['episode_id']}")).json()
    live = detail["analysis"]["live"]
    summary = live["tone_summary"]

    # The headline: the user's own escalations are exactly the scripted ones.
    assert summary["self"]["escalation_turns"] == expected
    assert summary["self"]["escalation_count"] == len(expected)
    # …and every escalation the OTHER people performed is absent (a nudge
    # on those would be a false positive — the fixture's explicit rule).
    other_angry = [
        i for i, t in enumerate(meta["turns"])
        if t["speaker"] != self_speaker and t["emotion_coarse"] == "angry"
    ]
    assert not set(other_angry) & set(summary["self"]["escalation_turns"])
    rows = live["turn_tone"]
    for i in other_angry:
        assert rows[i]["is_self"] is False and rows[i]["label"] == "angry"

    # Per-turn rows line up with the script for the self turns.
    for i, t in enumerate(meta["turns"]):
        row = rows[i]
        assert row["index"] == i and row["speaker"] == t["speaker"]
        assert row["is_self"] == (t["speaker"] == self_speaker)
        # `escalated` is per SPEAKER (the other side's anger is recorded on
        # their row); only the SELF rows feed the self aggregates above.
        assert row["escalated"] == (t["emotion_coarse"] == "angry")

    # Self label distribution == the script's coarse emotions over self turns.
    self_turns = [t for t in meta["turns"] if t["speaker"] == self_speaker]
    want = {}
    for t in self_turns:
        label = {"angry": "angry", "sad": "sad", "happy": "warm", "neutral": "neutral"}[t["emotion_coarse"]]
        want[label] = want.get(label, 0) + 1
    assert summary["self"]["labels"] == want
    assert summary["self"]["turns"] == len(self_turns)

    # Every other speaker is a "with ___" row (unnamed → raw label), with the
    # right number of their own turns, and the per-person escalations sum to
    # the self total (each self turn is attributed to exactly one partner).
    others = [sp for sp in meta["speakers"] if sp != self_speaker]
    assert {p["speaker"] for p in summary["people"]} == set(others)
    for person in summary["people"]:
        assert person["their_turns"] == sum(
            1 for t in meta["turns"] if t["speaker"] == person["speaker"]
        )
        assert person["display_name"] is None and person["person_id"] is None
    assert sum(p["escalation_count"] for p in summary["people"]) == len(expected)
    assert sum(p["self_turns"] for p in summary["people"]) == len(self_turns)

    # The scene's short pauses never split an episode; the episode carries
    # the same self-tone facts the timeline chip renders.
    [episode] = detail["episodes"]
    assert episode["turn_count"] == len(meta["turns"])
    assert episode["self_escalation_count"] == len(expected)
    assert episode["self_tone_labels"] == want
    assert episode["participants"][0:1] == ["You"] or "You" in episode["participants"]
    assert episode["mean_heat"] is None   # no LLM pass ran — honest gap

    # Growth agrees: the session is an identified point (self is "You") with
    # the same escalation count, scored honestly as a gap (no report card).
    growth = (await client.get("/growth")).json()
    point = next(p for p in growth["points"] if p["recording_id"] == body["episode_id"])
    assert point["my_score"] is None
    assert point["self_tone"]["escalation_count"] == len(expected)
    assert point["source"] == "live" and point["mode"] == "earpiece"
    # Unnamed partners never become cross-session people rows.
    assert growth["people"] == []
