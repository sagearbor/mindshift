# Ported from gauge@2157433 server/aggregates.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B4): Episode -> LiveSession per the locked rename map.
# `episode_calm_score` is renamed `live_session_calm_score` to match (it
# takes and describes a LiveSession, and there is no Episode type left in
# this repo's watch vocabulary to name it after). `PeriodStats.episodes`
# itself is UNCHANGED -- Task B1 already ported that field name as-is (it's
# a count, not a type reference) -- so this module keeps writing/reading it
# as `episodes=`/`.episodes`.
"""Pure period-aggregate math for the couples "standing" endpoint.

No I/O, no store, no FastAPI here — this module is deliberately just
functions over in-memory LiveSession lists so it can be unit tested without
Firestore and reused unchanged by the /me/standing handler (Task B6).
"""
from datetime import datetime, timedelta

from watch.models import LiveSession, GroupStanding, MemberStanding, PeriodStats

CALM_MAX = 100
CALM_PER_LEVEL = 25   # T8 review minor 3: the watch's GaugeViewModel no longer shows a
                       # calmScore at all (replaced by the live green/red threshold meter,
                       # per gauge's Phase 3 main-screen redesign) -- this formula is now
                       # purely a dashboard/standing-graph concept, not something mirroring
                       # any live on-wrist value.

# Every event written before Task 8 has participant_id=None; v1's vectors
# (interrupting/airtime/yelling/aggressive_tone) are self-coaching only, so
# an unattributed legacy event IS the wearer's own. Excluding it would
# silently erase months of history from the graphs.
_SELF_PARTICIPANT_ID = "self"


def live_session_calm_score(ls: LiveSession) -> int:
    """100 minus 25 per level of the live session's WORST self-attributed
    vector event, clamped to [0, 100]. Formerly the same formula the watch
    showed live (see CALM_PER_LEVEL above) -- the watch dropped its
    calmScore display for the live green/red threshold meter (Phase 3), so
    this is now a dashboard/standing-graph-only concept. Events attributed
    to someone else (participant_id set to another participant) are
    excluded — Gauge measures its own wearer (spec §4.4)."""
    worst = 0
    for event in ls.vector_events:
        if event.participant_id is not None and event.participant_id != _SELF_PARTICIPANT_ID:
            continue
        if event.level > worst:
            worst = event.level
    return max(0, min(CALM_MAX, CALM_MAX - CALM_PER_LEVEL * worst))


def parse_iso(ts: str | None) -> datetime | None:
    """Tolerant ISO-8601 parse (accepts a trailing 'Z'); None on anything
    unparseable — a live session with a garbage timestamp is EXCLUDED from a
    period rather than silently bucketed into it."""
    if ts is None:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def period_stats(live_sessions: list[LiveSession], start: datetime, end: datetime) -> PeriodStats:
    """LiveSessions whose started_at falls in [start, end). calm is the mean
    of per-session calm scores rounded to 1dp, or None when the period is
    empty (never 0 — an empty week is 'no data', not 'maximally bad')."""
    in_period = []
    for ls in live_sessions:
        started = parse_iso(ls.started_at)
        if started is None:
            continue
        if start <= started < end:
            in_period.append(ls)

    nudges = sum(len(ls.nudge_events) for ls in in_period)
    escalations = sum(
        1
        for ls in in_period
        for event in ls.vector_events
        if (event.participant_id is None or event.participant_id == _SELF_PARTICIPANT_ID)
        and event.level >= 2
    )

    if not in_period:
        return PeriodStats(episodes=0, calm=None, nudges=nudges, escalations=escalations)

    scores = [live_session_calm_score(ls) for ls in in_period]
    calm = round(sum(scores) / len(scores), 1)
    return PeriodStats(episodes=len(in_period), calm=calm, nudges=nudges, escalations=escalations)


def member_standing(account_id: str, live_sessions: list[LiveSession], now: datetime,
                     period_days: int, display_name: str | None = None) -> MemberStanding:
    """current = [now - period_days, now); prior = [now - 2*period_days, now - period_days).
    `live_sessions` must already be filtered to live sessions this account OWNS."""
    current_start = now - timedelta(days=period_days)
    prior_start = now - timedelta(days=2 * period_days)

    current = period_stats(live_sessions, current_start, now)
    prior = period_stats(live_sessions, prior_start, current_start)

    delta_vs_self: float | None = None
    improving: bool | None = None
    if current.calm is not None and prior.calm is not None:
        delta_vs_self = round(current.calm - prior.calm, 1)
        improving = delta_vs_self > 0

    return MemberStanding(
        account_id=account_id,
        display_name=display_name,
        current=current,
        prior=prior,
        delta_vs_self=delta_vs_self,
        improving=improving,
    )


def group_standing(group_id: str, standings: list[MemberStanding], now: datetime,
                    period_days: int) -> GroupStanding:
    """`both_improving` is True only when EVERY member has improving is True.
    `ahead` is None whenever fewer than two members have a non-None
    current.calm, or when the top two tie."""
    period_start = now - timedelta(days=period_days)
    period_end = now

    both_improving = bool(standings) and all(m.improving is True for m in standings)

    measurable = [m for m in standings if m.current.calm is not None]
    ahead: str | None = None
    if len(measurable) >= 2:
        ranked = sorted(measurable, key=lambda m: m.current.calm, reverse=True)
        top, second = ranked[0], ranked[1]
        if top.current.calm != second.current.calm:
            ahead = top.account_id

    return GroupStanding(
        group_id=group_id,
        period_days=period_days,
        period_start=period_start.isoformat(),
        period_end=period_end.isoformat(),
        members=standings,
        both_improving=both_improving,
        ahead=ahead,
    )
