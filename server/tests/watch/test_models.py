# Ported from gauge@2157433 server/tests/test_models.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
from watch.models import LiveSession, Participant, VectorEvent, NudgeEvent, VectorSubscription


def test_wire_roundtrip():
    ls = LiveSession(id="e1", owner_account="a1", started_at="2026-07-31T22:00:00Z", ended_at=None,
                      status="live",
                      participants=[Participant(id="p1", role="self", speaker_label="You")],
                      vector_events=[VectorEvent(vector="yelling", level=2, t=3.5, value=14.2)],
                      nudge_events=[NudgeEvent(channel="A", level=2, t=3.5, vectors=["yelling"])])
    assert LiveSession.model_validate_json(ls.model_dump_json()) == ls


def test_subscription_default_channels():
    assert VectorSubscription(vector="hr_spike").channel == "B"
    assert VectorSubscription(vector="yelling").channel == "A"


def test_level_bounds():
    import pytest
    with pytest.raises(Exception):
        VectorEvent(vector="yelling", level=4, t=0, value=0)


def test_live_session_consents_roundtrip():
    from watch.models import ConsentRecord
    consents = [
        ConsentRecord(id="c1", participant_id="p1", kind="labeling", attested_by="admin", confirmed=True, ts="2026-07-31T22:00:00Z"),
        ConsentRecord(id="c2", participant_id="p1", kind="sharing", attested_by="user", confirmed=False, ts="2026-07-31T22:01:00Z")
    ]
    ls = LiveSession(id="e2", owner_account="a1", started_at="2026-07-31T22:00:00Z", ended_at=None,
                      status="live",
                      participants=[Participant(id="p1", role="self", speaker_label="You")],
                      vector_events=[],
                      nudge_events=[],
                      consents=consents)
    roundtripped = LiveSession.model_validate_json(ls.model_dump_json())
    assert roundtripped == ls
    assert roundtripped.consents == consents
