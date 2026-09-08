"""Tests for POST /recordings/{id}/reanalyze-with-segments — "Use these voices
for this recording".

The phone's experimental engine B produced a speaker timeline; applying it
regroups the stored RAW transcript's WORDS by segment (proportional split when
there are no word timings), runs the SAME re-analysis job with STT skipped
and the local-diarization cross-check DISABLED (the user's segmentation must
win), overwrites analysis.json + turns.json in place, stamps the provenance on
meta, and clears the manual speaker names (they were keyed by the old ids).
Same fakes/patching as test_reanalyze; GCS/Deepgram/LLM are never touched.
"""

import json
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

import main
from main import (
    SpeakerSegment,
    _regroup_transcript_by_segments,
    app,
    init_db,
)
from tests.test_reanalyze import (
    FIXTURE_WAV,
    GOOD_UUID,
    STORED_TURNS,
    _STORED_SPEAKERS,
    FakeReanalyzeStore,
    _analyze_llm_json,
    _drain_jobs,
    _get_job,
    _mock_llm,
)

_ABC = ["Speaker A", "Speaker B", "Speaker C"]

# The phone's timeline over the STORED_TURNS fixture (0–5.5 s). Row 0 (with
# word timings 0.0–1.2) is WELDED across A→B at 0.87 s; rows 2–4 have no word
# timings (proportional split); segment C stops at 4.0 s so row 4's words
# (4.0–5.5) exercise the snap (first word, 0.375 s away) and the
# neighbour-fill (second word, 1.1 s away — beyond the 0.5 s snap).
SEGMENTS = [
    {"start": 0.0, "end": 0.87, "label": "Speaker A"},
    {"start": 0.87, "end": 2.5, "label": "Speaker B"},
    {"start": 2.5, "end": 4.0, "label": "Speaker C"},
]
EXPECTED_TURNS = [
    ("Speaker A", "Did you remember to", 0.0, 0.9),
    ("Speaker B", "call the plumber?", 0.9, 1.2),
    ("Speaker B", "I did, they come Tuesday.", 1.4, 2.5),
    ("Speaker C", "Great, thank you for handling that.", 2.5, 4.0),
    ("Speaker C", "Of course.", 4.0, 5.5),
]


def _segs(raw):
    return [SpeakerSegment(**s) for s in raw]


class FakeSegmentsStore(FakeReanalyzeStore):
    """test_reanalyze's fake + the meta-stamp method this endpoint adds."""

    def __init__(self):
        super().__init__()
        self.segment_stamps: list = []

    async def apply_speaker_segments_meta(
        self, uid, recording_id, *, source, applied_at, segments,
    ):
        r = self._recordings.get(uid, {}).get(recording_id)
        if r is None:
            return None
        meta = r["meta"]
        meta["speaker_segments_source"] = source
        meta["speaker_segments_applied_at"] = applied_at
        meta["speaker_segments"] = segments
        meta.pop("manual_speaker_labels", None)
        meta.pop("manual_speaker_people", None)
        self.segment_stamps.append({
            "recording_id": recording_id, "source": source,
            "applied_at": applied_at, "segments": segments,
        })
        return dict(meta)


@pytest.fixture
async def client():
    await init_db()
    main._rate_limiter.reset()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def store():
    fake = FakeSegmentsStore()
    app.state.recordings_store = fake
    yield fake
    del app.state.recordings_store


async def _post(client, rid, body, uid="test-user"):
    return await client.post(
        f"/recordings/{rid}/reanalyze-with-segments",
        headers={"X-Test-Uid": uid}, json=body,
    )


# ---------------------------------------------------------------------------
# The pure regrouping
# ---------------------------------------------------------------------------

def test_regroup_words_follow_segments_and_split_welded_utterance():
    out = _regroup_transcript_by_segments(STORED_TURNS, _segs(SEGMENTS))
    assert [(t["speaker"], t["text"], t["start_time"], t["end_time"]) for t in out] == \
        [(s, x, pytest.approx(a), pytest.approx(b)) for s, x, a, b in EXPECTED_TURNS]
    # The blank row contributed nothing (no placeholder turn).
    assert all(t["text"] for t in out)


def test_regroup_proportional_split_without_word_timings():
    # turns.json-style rows: text + span only. A 4-word row over 0–4 s gets
    # one word per second; a segment change at 2.0 splits it 2/2.
    rows = [
        {"speaker": "old", "text": "one two three four", "start_time": 0.0, "end_time": 4.0},
        {"speaker": "old", "text": "five six", "start_time": 4.0, "end_time": 6.0},
    ]
    segs = _segs([
        {"start": 0.0, "end": 2.0, "label": "X"},
        {"start": 2.0, "end": 6.0, "label": "Y"},
    ])
    out = _regroup_transcript_by_segments(rows, segs)
    assert [(t["speaker"], t["text"]) for t in out] == [
        ("X", "one two"), ("Y", "three four"), ("Y", "five six"),
    ]
    assert out[0]["start_time"] == pytest.approx(0.0)
    assert out[0]["end_time"] == pytest.approx(2.0)
    assert out[1]["start_time"] == pytest.approx(2.0)
    assert out[1]["end_time"] == pytest.approx(4.0)


def test_regroup_never_drops_a_word_far_from_every_segment():
    # A whole row outside every segment (and beyond the snap) keeps the last
    # label placed before it; a row BEFORE any placement takes the nearest.
    rows = [
        {"speaker": "old", "text": "early words", "start_time": 0.0, "end_time": 1.0},
        {"speaker": "old", "text": "inside", "start_time": 10.0, "end_time": 11.0},
        {"speaker": "old", "text": "late words", "start_time": 20.0, "end_time": 21.0},
    ]
    segs = _segs([{"start": 10.0, "end": 11.0, "label": "P"}])
    out = _regroup_transcript_by_segments(rows, segs)
    assert [(t["speaker"], t["text"]) for t in out] == [
        ("P", "early words"), ("P", "inside"), ("P", "late words"),
    ]


def test_regroup_untimed_rows_ride_with_neighbours():
    rows = [
        {"speaker": "old", "text": "timed", "start_time": 0.0, "end_time": 1.0},
        {"speaker": "old", "text": "no timing at all"},
    ]
    segs = _segs([{"start": 0.0, "end": 1.0, "label": "Q"}])
    out = _regroup_transcript_by_segments(rows, segs)
    assert [(t["speaker"], t["text"]) for t in out] == [("Q", "timed"), ("Q", "no timing at all")]
    assert out[1]["start_time"] is None


# ---------------------------------------------------------------------------
# Happy path — regrouped, cross-check skipped, overwritten, stamped, cleared
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_apply_segments_happy_path(client, store, monkeypatch):
    # Production runs the cross-check on 2+-speaker transcripts too; it must
    # NOT run here — the phone's segmentation is the point.
    monkeypatch.setenv("MINDSHIFT_DIARIZE_CROSSCHECK", "1")
    src = {"type": "upload", "url": None, "original_filename": "x.m4a"}
    store.seed("test-user", GOOD_UUID, audio=FIXTURE_WAV, title="Plumber",
               source=src, transcript=STORED_TURNS,
               turns=[{"speaker": "Alice", "text": "stale"}],
               analysis={"narrative": "old"})
    meta = store._recordings["test-user"][GOOD_UUID]["meta"]
    meta["manual_speaker_labels"] = {"Alice": "Mum", "Bob": "Dad"}
    meta["manual_speaker_people"] = {"Alice": "self"}

    with patch("main.transcribe_upload") as stt, \
         patch("main.get_llm_client",
               return_value=_mock_llm(_analyze_llm_json(len(EXPECTED_TURNS), _ABC))), \
         patch("main.diarize_local.diarize_turns") as dz:
        resp = await _post(client, GOOD_UUID, {"segments": SEGMENTS, "source": "device-B"})
        assert resp.status_code == 202, resp.text
        body = resp.json()
        job_id = body["job_id"]
        assert "cleared" in (body["note"] or "")
        await _drain_jobs()
        done = await _get_job(client, job_id)

    stt.assert_not_called()
    dz.assert_not_called()
    assert store.transcript_saves == []

    body = done.json()
    assert body["status"] == "done", body
    assert body["error"] is None
    result = body["result"]
    assert [(t["speaker"], t["text"]) for t in result["turns"]] == \
        [(s, x) for s, x, _, _ in EXPECTED_TURNS]
    assert len(result["per_turn"]) == len(EXPECTED_TURNS)
    assert set(result["word_metrics"]["speakers"]) == set(_ABC)
    assert "skipped" in (result["voice_analysis"] or "")
    assert "device-B" in (result["transcription_note"] or "")
    assert "cleared" in (result["storage_note"] or "")
    assert result["stored"] is True
    assert result["recording_id"] == GOOD_UUID
    assert result["title"] == "Plumber"

    history = store.status_history[job_id]
    assert "transcribing" not in history
    deduped = [s for i, s in enumerate(history) if i == 0 or s != history[i - 1]]
    assert deduped == ["queued", "analyzing", "storing", "done"]

    # Persisted in place, provenance stamped, manual names gone.
    assert len(store.overwrite_calls) == 1
    call = store.overwrite_calls[0]
    assert [t["speaker"] for t in call["turns"]] == [s for s, _, _, _ in EXPECTED_TURNS]
    assert call["analysis"]["word_metrics"] is not None
    assert len(store.segment_stamps) == 1
    stamp = store.segment_stamps[0]
    assert stamp["source"] == "device-B"
    assert stamp["applied_at"] == call["reanalyzed_at"]
    assert stamp["segments"] == SEGMENTS
    stored_meta = store._recordings["test-user"][GOOD_UUID]["meta"]
    assert stored_meta["speaker_segments_source"] == "device-B"
    assert stored_meta["speaker_segments_applied_at"] == call["reanalyzed_at"]
    assert stored_meta["reanalyzed_at"] == call["reanalyzed_at"]
    assert "manual_speaker_labels" not in stored_meta
    assert "manual_speaker_people" not in stored_meta
    assert stored_meta["title"] == "Plumber"
    assert stored_meta["source"] == src

    # The detail read carries the provenance and the cleared name maps.
    detail = await client.get(
        f"/recordings/{GOOD_UUID}", headers={"X-Test-Uid": "test-user"},
    )
    assert detail.status_code == 200, detail.text
    d = detail.json()
    assert d["speaker_segments_source"] == "device-B"
    assert d["speaker_segments_applied_at"] == call["reanalyzed_at"]
    assert d["manual_speaker_labels"] == {}
    assert d["manual_speaker_people"] == {}


@pytest.mark.anyio
async def test_apply_segments_without_word_timings_uses_proportional_split(client, store):
    # No transcript.json — the stored turns.json (text + span only) is split
    # proportionally. Four 1 s rows; the segment boundary at 2.5 s lands in
    # the middle of row 2 ("c1 c2" → c1 before, c2 after).
    rows = [
        {"speaker": "old", "text": "a1 a2", "start_time": 0.0, "end_time": 1.0},
        {"speaker": "old", "text": "b1 b2", "start_time": 1.0, "end_time": 2.0},
        {"speaker": "old", "text": "c1 c2", "start_time": 2.0, "end_time": 3.0},
        {"speaker": "old", "text": "d1 d2", "start_time": 3.0, "end_time": 4.0},
    ]
    store.seed("test-user", GOOD_UUID, audio=FIXTURE_WAV, turns=rows, transcript=None)
    segs = [
        {"start": 0.0, "end": 2.5, "label": "Speaker A"},
        {"start": 2.5, "end": 4.0, "label": "Speaker B"},
    ]
    expected = [("Speaker A", "a1 a2"), ("Speaker A", "b1 b2"),
                ("Speaker A", "c1"), ("Speaker B", "c2"), ("Speaker B", "d1 d2")]
    with patch("main.transcribe_upload") as stt, \
         patch("main.get_llm_client", return_value=_mock_llm(
             _analyze_llm_json(len(expected), ["Speaker A", "Speaker B"]))):
        resp = await _post(client, GOOD_UUID, {"segments": segs})
        assert resp.status_code == 202, resp.text
        await _drain_jobs()
        done = await _get_job(client, resp.json()["job_id"])
    stt.assert_not_called()
    body = done.json()
    assert body["status"] == "done", body
    assert [(t["speaker"], t["text"]) for t in body["result"]["turns"]] == expected
    # Default source when the body omits it.
    assert store.segment_stamps[0]["source"] == "device-B"


# ---------------------------------------------------------------------------
# Validation + errors — nothing is spawned for a request we can't honour
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_apply_segments_overlapping_422(client, store):
    store.seed("test-user", GOOD_UUID, audio=FIXTURE_WAV, transcript=STORED_TURNS)
    segs = [
        {"start": 0.0, "end": 3.0, "label": "Speaker A"},
        {"start": 2.0, "end": 5.5, "label": "Speaker B"},  # 1 s overlap
    ]
    resp = await _post(client, GOOD_UUID, {"segments": segs})
    assert resp.status_code == 422
    assert "overlap" in json.dumps(resp.json()).lower()
    assert store.status_history == {}


@pytest.mark.anyio
async def test_apply_segments_tolerates_rounding_overlap_and_any_order(client, store):
    store.seed("test-user", GOOD_UUID, audio=FIXTURE_WAV, transcript=STORED_TURNS)
    segs = [  # 0.03 s overlap (≤ 0.05 slop) and out of order
        {"start": 2.47, "end": 4.0, "label": "Speaker C"},
        {"start": 0.0, "end": 0.87, "label": "Speaker A"},
        {"start": 0.87, "end": 2.5, "label": "Speaker B"},
    ]
    with patch("main.transcribe_upload"), \
         patch("main.get_llm_client",
               return_value=_mock_llm(_analyze_llm_json(len(EXPECTED_TURNS), _ABC))):
        resp = await _post(client, GOOD_UUID, {"segments": segs})
        assert resp.status_code == 202, resp.text
        await _drain_jobs()
        done = await _get_job(client, resp.json()["job_id"])
    assert done.json()["status"] == "done", done.json()
    # Stored sorted.
    assert [s["label"] for s in store.segment_stamps[0]["segments"]] == _ABC


@pytest.mark.anyio
async def test_apply_segments_bad_bodies_422(client, store):
    store.seed("test-user", GOOD_UUID, audio=FIXTURE_WAV, transcript=STORED_TURNS)
    # Too many distinct labels.
    many = [{"start": i, "end": i + 0.5, "label": f"S{i}"} for i in range(11)]
    assert (await _post(client, GOOD_UUID, {"segments": many})).status_code == 422
    # Empty list.
    assert (await _post(client, GOOD_UUID, {"segments": []})).status_code == 422
    # end before start.
    bad = [{"start": 2.0, "end": 1.0, "label": "A"}]
    assert (await _post(client, GOOD_UUID, {"segments": bad})).status_code == 422
    assert store.status_history == {}


@pytest.mark.anyio
async def test_apply_segments_regrouped_transcript_too_short_422(client, store):
    # Two rows under one segment → 2 turns, below the analysis minimum: an
    # honest 422 up front, never a job that silently falls back to STT.
    store.seed("test-user", GOOD_UUID, audio=FIXTURE_WAV, transcript=STORED_TURNS[3:])
    segs = [{"start": 0.0, "end": 6.0, "label": "Speaker A"}]
    with patch("main.transcribe_upload") as stt:
        resp = await _post(client, GOOD_UUID, {"segments": segs})
    assert resp.status_code == 422
    assert "out of bounds" in resp.json()["detail"]
    stt.assert_not_called()
    assert store.status_history == {}


@pytest.mark.anyio
async def test_apply_segments_unknown_and_foreign_404(client, store):
    resp = await _post(client, GOOD_UUID, {"segments": SEGMENTS})
    assert resp.status_code == 404
    store.seed("user-a", GOOD_UUID, audio=FIXTURE_WAV, transcript=STORED_TURNS)
    resp = await _post(client, GOOD_UUID, {"segments": SEGMENTS}, uid="user-b")
    assert resp.status_code == 404
    assert store.status_history == {}


@pytest.mark.anyio
async def test_apply_segments_no_stored_audio_422(client, store):
    store.seed("test-user", GOOD_UUID, audio=None, transcript=STORED_TURNS)
    resp = await _post(client, GOOD_UUID, {"segments": SEGMENTS})
    assert resp.status_code == 422
    assert "audio" in resp.json()["detail"].lower()
    assert store.status_history == {}


@pytest.mark.anyio
async def test_apply_segments_no_transcript_422(client, store):
    store.seed("test-user", GOOD_UUID, audio=FIXTURE_WAV, turns=[], transcript=None)
    resp = await _post(client, GOOD_UUID, {"segments": SEGMENTS})
    assert resp.status_code == 422
    assert "transcript" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_apply_segments_503_when_storage_disabled(client):
    resp = await client.post(
        f"/recordings/{GOOD_UUID}/reanalyze-with-segments",
        headers={"X-Test-Uid": "test-user"}, json={"segments": SEGMENTS},
    )
    assert resp.status_code == 503


@pytest.mark.anyio
async def test_apply_segments_requires_auth_401(client, store, monkeypatch):
    from auth import get_current_uid
    monkeypatch.delitem(app.dependency_overrides, get_current_uid)
    resp = await client.post(
        f"/recordings/{GOOD_UUID}/reanalyze-with-segments", json={"segments": SEGMENTS},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST …/reanalyze on a recording WITH applied voices (2026-08-30): the applied
# segments are kept by default ("Re-analyze with the latest engine" must not
# undo a hand-checked result); fresh=true (query or body) ignores them.
# ---------------------------------------------------------------------------

def _seed_applied(store, *, manual=True):
    store.seed("test-user", GOOD_UUID, audio=FIXTURE_WAV, title="Plumber",
               transcript=STORED_TURNS,
               turns=[{"speaker": "Speaker A", "text": "stale"}],
               analysis={"narrative": "old"})
    meta = store._recordings["test-user"][GOOD_UUID]["meta"]
    meta["speaker_segments"] = [dict(s) for s in SEGMENTS]
    meta["speaker_segments_source"] = "device-B"
    meta["speaker_segments_applied_at"] = "2026-08-30T10:00:00+00:00"
    if manual:
        meta["manual_speaker_labels"] = {"Speaker A": "Mum", "Speaker B": "Dad"}
        meta["manual_speaker_people"] = {"Speaker A": "self"}
    return meta


async def _post_reanalyze(client, rid, *, query="", body=None, uid="test-user"):
    return await client.post(
        f"/recordings/{rid}/reanalyze{query}", headers={"X-Test-Uid": uid},
        **({"json": body} if body is not None else {}),
    )


@pytest.mark.anyio
async def test_reanalyze_keeps_applied_voices_by_default(client, store, monkeypatch):
    monkeypatch.setenv("MINDSHIFT_DIARIZE_CROSSCHECK", "1")
    monkeypatch.delenv("MINDSHIFT_DIARIZE_ENGINE", raising=False)
    meta = _seed_applied(store)

    with patch("main.transcribe_upload") as stt, \
         patch("main.get_llm_client",
               return_value=_mock_llm(_analyze_llm_json(len(EXPECTED_TURNS), _ABC))), \
         patch("main.diarize_local.diarize_windows_first") as wf, \
         patch("main.diarize_local.diarize_turns") as dz:
        resp = await _post_reanalyze(client, GOOD_UUID)
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]
        assert resp.json()["note"] is None
        await _drain_jobs()
        done = await _get_job(client, job_id)

    # No STT, no engine at all — the applied voices are the speakers.
    stt.assert_not_called()
    wf.assert_not_called()
    dz.assert_not_called()
    assert store.transcript_saves == []

    body = done.json()
    assert body["status"] == "done", body
    result = body["result"]
    assert [(t["speaker"], t["text"]) for t in result["turns"]] == \
        [(s, x) for s, x, _, _ in EXPECTED_TURNS]
    assert set(result["word_metrics"]["speakers"]) == set(_ABC)
    assert "skipped" in (result["voice_analysis"] or "")
    note = result["transcription_note"] or ""
    assert "device-B" in note and "kept" in note and "fresh=true" in note
    assert result["storage_note"] is None
    assert result["stored"] is True and result["recording_id"] == GOOD_UUID

    # Persisted in place; the applied segments, their provenance and the manual
    # names all survive (the speaker ids did not change) — nothing re-stamped.
    assert len(store.overwrite_calls) == 1
    assert store.segment_stamps == []
    assert meta["speaker_segments"] == SEGMENTS
    assert meta["speaker_segments_source"] == "device-B"
    assert meta["speaker_segments_applied_at"] == "2026-08-30T10:00:00+00:00"
    assert meta["manual_speaker_labels"] == {"Speaker A": "Mum", "Speaker B": "Dad"}
    assert meta["manual_speaker_people"] == {"Speaker A": "self"}
    assert meta["reanalyzed_at"] == store.overwrite_calls[0]["reanalyzed_at"]
    history = store.status_history[job_id]
    assert "transcribing" not in history and history[-1] == "done"


@pytest.mark.parametrize("how", ["query", "body"])
@pytest.mark.anyio
async def test_reanalyze_fresh_ignores_applied_voices(client, store, monkeypatch, how):
    monkeypatch.setenv("MINDSHIFT_DIARIZE_CROSSCHECK", "1")
    monkeypatch.delenv("MINDSHIFT_DIARIZE_ENGINE", raising=False)
    meta = _seed_applied(store, manual=False)
    seen: list = []

    def _spy(pcm, sr, turns, **kw):
        seen.append([dict(t) for t in turns])
        return None

    with patch("main.transcribe_upload") as stt, \
         patch("main.get_llm_client",
               return_value=_mock_llm(_analyze_llm_json(len(STORED_TURNS), _STORED_SPEAKERS))), \
         patch("main.diarize_local.diarize_windows_first", side_effect=_spy) as wf, \
         patch("main.diarize_local.diarize_turns", return_value=None) as dz:
        resp = await _post_reanalyze(
            client, GOOD_UUID,
            query="?fresh=true" if how == "query" else "",
            body={"fresh": True} if how == "body" else None,
        )
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]
        await _drain_jobs()
        done = await _get_job(client, job_id)

    stt.assert_not_called()
    # The engines ran (windows first, then the utterance fallback) on the
    # STORED transcript — not on the applied segments.
    wf.assert_called_once()
    dz.assert_called_once()
    assert seen and seen[0][0].get("words") == STORED_TURNS[0]["words"]
    body = done.json()
    assert body["status"] == "done", body
    result = body["result"]
    assert [t["speaker"] for t in result["turns"]] == [t["speaker"] for t in STORED_TURNS]
    assert "stored transcript" in (result["transcription_note"] or "")
    assert "kept" not in (result["transcription_note"] or "")
    # The applied segments stay on the meta for a later re-apply.
    assert meta["speaker_segments"] == SEGMENTS
    assert store.segment_stamps == []


@pytest.mark.anyio
async def test_reanalyze_unusable_applied_voices_is_422_with_fresh_hint(client, store):
    meta = _seed_applied(store, manual=False)
    meta["speaker_segments"] = [{"start": 2.0, "end": 1.0, "label": "X"}]
    resp = await _post_reanalyze(client, GOOD_UUID)
    assert resp.status_code == 422, resp.text
    assert "fresh=true" in resp.json()["detail"]
