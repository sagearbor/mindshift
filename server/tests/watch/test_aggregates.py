# Ported from gauge@2157433 server/tests/test_aggregates.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B4): Episode -> LiveSession per the locked rename map; the
# ported `episode_calm_score` function is renamed `live_session_calm_score`
# to match (it operates on a LiveSession, and there is no Episode type left
# in this repo's watch vocabulary to name it after). `PeriodStats.episodes`
# is UNCHANGED -- that field name was already ported as-is by Task B1 (it's
# a count, not a type reference) -- so assertions/kwargs below still say
# `episodes=`.
from datetime import datetime, timedelta, timezone

from watch.aggregates import (
    live_session_calm_score, group_standing, member_standing, parse_iso, period_stats,
)
from watch.models import LiveSession, NudgeEvent, Participant, VectorEvent

NOW = datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)


def _ls(id, owner, started: datetime, levels=(), nudges=0):
    return LiveSession(
        id=id, owner_account=owner, started_at=started.isoformat(), ended_at=None,
        status="analyzed",
        participants=[Participant(id="self", role="self", speaker_label="You")],
        vector_events=[VectorEvent(vector="yelling", level=lv, t=float(i), value=1.0,
                                   participant_id="self")
                       for i, lv in enumerate(levels)],
        nudge_events=[NudgeEvent(channel="A", level=1, t=float(i), vectors=["yelling"])
                      for i in range(nudges)],
    )


def test_calm_score_is_100_without_events():
    assert live_session_calm_score(_ls("e", "a", NOW)) == 100


def test_calm_score_drops_25_per_worst_level():
    assert live_session_calm_score(_ls("e", "a", NOW, levels=(1,))) == 75
    assert live_session_calm_score(_ls("e", "a", NOW, levels=(1, 3, 2))) == 25
    assert live_session_calm_score(_ls("e", "a", NOW, levels=(3, 3))) == 25


def test_calm_score_ignores_events_attributed_to_others():
    ls = _ls("e", "a", NOW)
    ls.vector_events = [VectorEvent(vector="yelling", level=3, t=0.0, value=1.0,
                                    participant_id="other-1")]
    assert live_session_calm_score(ls) == 100


def test_calm_score_counts_unattributed_legacy_events_as_the_wearers():
    # Every event written before Task 8 has participant_id=None and IS the
    # wearer's (v1 vectors are self-coaching only) — excluding them would
    # silently erase months of history from the graphs.
    ls = _ls("e", "a", NOW)
    ls.vector_events = [VectorEvent(vector="yelling", level=2, t=0.0, value=1.0)]
    assert live_session_calm_score(ls) == 50


def test_parse_iso_accepts_z_and_rejects_garbage():
    assert parse_iso("2026-08-02T00:00:00Z") == datetime(2026, 8, 2, tzinfo=timezone.utc)
    assert parse_iso("2026-08-02T00:00:00+00:00") == datetime(2026, 8, 2, tzinfo=timezone.utc)
    assert parse_iso("not-a-date") is None
    assert parse_iso(None) is None


def test_period_stats_empty_reports_none_calm_not_zero():
    s = period_stats([], NOW - timedelta(days=7), NOW)
    assert s.episodes == 0 and s.calm is None and s.nudges == 0 and s.escalations == 0


def test_period_stats_windows_by_started_at():
    sessions = [
        _ls("in1", "a", NOW - timedelta(days=1), levels=(1,)),      # 75
        _ls("in2", "a", NOW - timedelta(days=6), levels=(3,)),      # 25
        _ls("old", "a", NOW - timedelta(days=20), levels=(0,)),     # excluded
    ]
    s = period_stats(sessions, NOW - timedelta(days=7), NOW)
    assert s.episodes == 2 and s.calm == 50.0


def test_period_stats_end_is_exclusive_and_start_inclusive():
    start, end = NOW - timedelta(days=7), NOW
    assert period_stats([_ls("edge", "a", end)], start, end).episodes == 0
    assert period_stats([_ls("edge", "a", start)], start, end).episodes == 1


def test_period_stats_excludes_unparseable_timestamps():
    ls = _ls("bad", "a", NOW - timedelta(days=1))
    ls.started_at = "whenever"
    assert period_stats([ls], NOW - timedelta(days=7), NOW).episodes == 0


def test_period_stats_counts_nudges_and_escalations():
    sessions = [_ls("e1", "a", NOW - timedelta(days=1), levels=(1, 2), nudges=3),
                _ls("e2", "a", NOW - timedelta(days=2), levels=(3,), nudges=1)]
    s = period_stats(sessions, NOW - timedelta(days=7), NOW)
    assert s.nudges == 4 and s.escalations == 2


def test_member_standing_delta_is_relative_to_own_prior_period():
    sessions = [_ls("cur", "a", NOW - timedelta(days=2), levels=(1,)),    # current: 75
                _ls("pri", "a", NOW - timedelta(days=9), levels=(3,))]    # prior:   25
    st = member_standing("a", sessions, NOW, period_days=7)
    assert st.current.calm == 75.0 and st.prior.calm == 25.0
    assert st.delta_vs_self == 50.0 and st.improving is True


def test_member_standing_delta_is_none_without_a_prior_period():
    st = member_standing("a", [_ls("cur", "a", NOW - timedelta(days=2))], NOW, period_days=7)
    assert st.prior.episodes == 0 and st.prior.calm is None
    assert st.delta_vs_self is None and st.improving is None


def test_member_standing_flags_regression():
    sessions = [_ls("cur", "a", NOW - timedelta(days=2), levels=(3,)),
                _ls("pri", "a", NOW - timedelta(days=9), levels=(1,))]
    st = member_standing("a", sessions, NOW, period_days=7)
    assert st.delta_vs_self == -50.0 and st.improving is False


def test_group_standing_both_improving_is_the_win_win_headline():
    a = member_standing("a", [_ls("c", "a", NOW - timedelta(days=1), levels=(1,)),
                              _ls("p", "a", NOW - timedelta(days=9), levels=(3,))], NOW, 7)
    b = member_standing("b", [_ls("c", "b", NOW - timedelta(days=1), levels=(2,)),
                              _ls("p", "b", NOW - timedelta(days=9), levels=(3,))], NOW, 7)
    gs = group_standing("g1", [a, b], NOW, 7)
    assert gs.both_improving is True         # both beat their OWN prior week
    assert gs.ahead == "a"                   # secondary only: 75 vs 50
    assert gs.period_days == 7 and gs.group_id == "g1"
    assert gs.period_start < gs.period_end


def test_group_standing_ahead_is_none_on_tie():
    a = member_standing("a", [_ls("c", "a", NOW - timedelta(days=1), levels=(1,))], NOW, 7)
    b = member_standing("b", [_ls("c", "b", NOW - timedelta(days=1), levels=(1,))], NOW, 7)
    assert group_standing("g1", [a, b], NOW, 7).ahead is None


def test_group_standing_ahead_is_none_without_two_measurable_members():
    a = member_standing("a", [_ls("c", "a", NOW - timedelta(days=1), levels=(1,))], NOW, 7)
    b = member_standing("b", [], NOW, 7)
    gs = group_standing("g1", [a, b], NOW, 7)
    assert gs.ahead is None and gs.both_improving is False


def test_group_standing_both_improving_false_when_one_is_unknown():
    a = member_standing("a", [_ls("c", "a", NOW - timedelta(days=1), levels=(1,)),
                              _ls("p", "a", NOW - timedelta(days=9), levels=(3,))], NOW, 7)
    b = member_standing("b", [_ls("c", "b", NOW - timedelta(days=1), levels=(1,))], NOW, 7)
    assert group_standing("g1", [a, b], NOW, 7).both_improving is False
