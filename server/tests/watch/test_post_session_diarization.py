# Ported from gauge@2157433 server/tests/test_post_episode_diarization.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
import asyncio

import numpy as np

import speaker_id
from audio_pipeline import TranscriberUnavailable
from server.tests.watch.test_post_session import FakeLLM, FakeTranscriber
from watch.models import LiveSession, Participant, SpeakerProfile, VectorEvent
from watch.post_session import analyze_live_session
from watch.store import MemoryLiveSessionStore

ME = np.eye(192, dtype=np.float32)[0]
THEM = np.eye(192, dtype=np.float32)[1]


class ScriptedDiarizer:
    """Returns fixed turns; asserts it was handed a voiceprint when expected."""

    def __init__(self, turns):
        self.turns = turns
        self.seen_print = "unset"

    def diarize(self, pcm, sr, self_print):
        self.seen_print = self_print
        return list(self.turns)


def _live_session(id="ls1", owner="alice", vector_events=None, participants=None):
    return LiveSession(id=id, owner_account=owner, started_at="2026-08-01T00:00:00Z",
                        ended_at="2026-08-01T00:05:00Z", status="captured",
                        participants=participants if participants is not None else
                        [Participant(id="self", role="self", speaker_label="You")],
                        vector_events=list(vector_events or []), nudge_events=[])


async def _store_with(ls, *, profile=True):
    s = MemoryLiveSessionStore()
    await s.put_live_session(ls)
    if profile:
        # v2 adaptation: server/speaker_id.py's new_profile() returns a dict
        # already shaped for SpeakerProfile(account_id=..., **new_profile(...))
        # (version/embedding/dim/enroll_count/model/created_at/updated_at/
        # samples) — see server/watch/routers/rest.py's _maybe_enroll_voice
        # for the same construction pattern (Task B5 precedent). No v1-shim
        # translation needed: this call is a drop-in vs. gauge's v1
        # engine.speaker_id.new_profile(...).
        await s.put_speaker_profile(SpeakerProfile(
            account_id=ls.owner_account,
            **speaker_id.new_profile(ME, None, recording_id="r", speaker="self", now_iso="t")))
    return s


TURNS = [("self", 0.0, 50.0), ("other-1", 49.0, 55.0), ("self", 54.0, 90.0)]


def test_diarization_activates_interrupting_and_airtime():
    async def run():
        store = await _store_with(_live_session())
        d = ScriptedDiarizer(TURNS)
        result = await analyze_live_session("ls1", store, FakeTranscriber(), FakeLLM(),
                                             pcm=b"\x00\x00" * 16000, diarizer=d)
        names = {e.vector for e in result.vector_events}
        assert "interrupting" in names and "airtime" in names
    asyncio.run(run())


def test_diarization_events_are_attributed_to_the_wearer():
    async def run():
        store = await _store_with(_live_session())
        result = await analyze_live_session("ls1", store, FakeTranscriber(), FakeLLM(),
                                             pcm=b"\x00\x00" * 16000, diarizer=ScriptedDiarizer(TURNS))
        diar = [e for e in result.vector_events if e.detail.startswith("diar:")]
        assert diar and all(e.participant_id == "self" for e in diar)
    asyncio.run(run())


def test_other_speakers_become_anonymous_participants():
    async def run():
        store = await _store_with(_live_session())
        result = await analyze_live_session("ls1", store, FakeTranscriber(), FakeLLM(),
                                             pcm=b"\x00\x00" * 16000, diarizer=ScriptedDiarizer(TURNS))
        other = next(p for p in result.participants if p.id == "other-1")
        assert other.role == "other" and other.speaker_label == "Speaker A"
        assert other.display_name is None                 # anonymous by default (spec §6)
    asyncio.run(run())


def test_diarizer_receives_the_stored_voiceprint():
    async def run():
        store = await _store_with(_live_session())
        d = ScriptedDiarizer(TURNS)
        await analyze_live_session("ls1", store, FakeTranscriber(), FakeLLM(),
                                    pcm=b"\x00\x00" * 16000, diarizer=d)
        assert isinstance(d.seen_print, np.ndarray) and d.seen_print.shape == (192,)
    asyncio.run(run())


def test_no_speaker_profile_means_no_diarization_events():
    async def run():
        store = await _store_with(_live_session(), profile=False)
        d = ScriptedDiarizer(TURNS)
        result = await analyze_live_session("ls1", store, FakeTranscriber(), FakeLLM(),
                                             pcm=b"\x00\x00" * 16000, diarizer=d)
        assert [e for e in result.vector_events if e.detail.startswith("diar:")] == []
        assert d.seen_print == "unset"                    # never even attempted
    asyncio.run(run())


def test_no_diarizer_leaves_live_session_exactly_as_before():
    async def run():
        store = await _store_with(_live_session())
        result = await analyze_live_session("ls1", store, FakeTranscriber(), FakeLLM(),
                                             pcm=b"\x00\x00" * 16000)
        assert result.vector_events == []
    asyncio.run(run())


def test_diarization_events_survive_transcription_unavailable():
    async def run():
        store = await _store_with(_live_session())
        result = await analyze_live_session(
            "ls1", store, FakeTranscriber(error=TranscriberUnavailable("no whisper")), FakeLLM(),
            pcm=b"\x00\x00" * 16000, diarizer=ScriptedDiarizer(TURNS))
        assert result.status == "transcription_unavailable" and result.summary is None
        assert any(e.detail.startswith("diar:") for e in result.vector_events)
        persisted = await store.get_live_session("ls1")
        assert any(e.detail.startswith("diar:") for e in persisted.vector_events)
    asyncio.run(run())


def test_reanalysis_replaces_rather_than_duplicates():
    async def run():
        store = await _store_with(_live_session())
        for _ in range(2):
            result = await analyze_live_session("ls1", store, FakeTranscriber(), FakeLLM(),
                                                 pcm=b"\x00\x00" * 16000, diarizer=ScriptedDiarizer(TURNS))
        once = [e for e in result.vector_events if e.detail.startswith("diar:")]
        assert len(once) == len({(e.vector, e.t, e.value) for e in once})
        assert len(result.participants) == 2              # no duplicate "other-1" participant
    asyncio.run(run())


def test_prosody_events_are_never_touched_by_the_diarization_pass():
    async def run():
        prior = VectorEvent(vector="hr_spike", level=1, t=5.0, value=90.0)
        store = await _store_with(_live_session(vector_events=[prior]))
        result = await analyze_live_session("ls1", store, FakeTranscriber(), FakeLLM(),
                                             pcm=b"\x00\x00" * 16000, diarizer=ScriptedDiarizer(TURNS))
        assert prior in result.vector_events
    asyncio.run(run())


def test_diarizer_that_raises_does_not_break_analysis():
    class Boom:
        def diarize(self, pcm, sr, self_print):
            raise RuntimeError("diarizer exploded")

    async def run():
        store = await _store_with(_live_session())
        result = await analyze_live_session("ls1", store, FakeTranscriber(), FakeLLM(),
                                             pcm=b"\x00\x00" * 16000, diarizer=Boom())
        assert result.status == "analyzed"
        assert [e for e in result.vector_events if e.detail.startswith("diar:")] == []
    asyncio.run(run())
