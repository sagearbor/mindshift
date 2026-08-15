# Ported from gauge@2157433 server/tests/test_nudge_policy.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
from watch.models import VectorEvent, VectorSubscription
from watch.nudge_policy import NudgePolicy

SUBS = [VectorSubscription(vector="yelling"), VectorSubscription(vector="hr_spike")]

def ev(vector, level, t):
    return VectorEvent(vector=vector, level=level, t=t, value=0)

def test_channels_are_independent():
    p = NudgePolicy(SUBS)
    out = p.on_events([ev("yelling", 2, 1.0), ev("hr_spike", 1, 1.0)], t=1.0)
    assert {(n.channel, n.level) for n in out} == {("A", 2), ("B", 1)}

def test_no_event_when_level_unchanged():
    p = NudgePolicy(SUBS)
    p.on_events([ev("yelling", 2, 1.0)], t=1.0)
    assert p.on_events([ev("yelling", 2, 2.0)], t=2.0) == []

def test_deescalation_after_cooldown():
    p = NudgePolicy(SUBS, cooldown_s=20.0)
    p.on_events([ev("yelling", 3, 1.0)], t=1.0)
    assert p.on_events([], t=10.0) == []                       # still hot
    out = p.on_events([], t=22.0)
    assert out and out[0].channel == "A" and out[0].level == 2

def test_haptics_off_vector_never_nudges():
    p = NudgePolicy([VectorSubscription(vector="yelling", haptics=False)])
    assert p.on_events([ev("yelling", 3, 1.0)], t=1.0) == []

def test_sensitivity_scales_level():
    p = NudgePolicy([VectorSubscription(vector="yelling", sensitivity=0.5)])
    out = p.on_events([ev("yelling", 3, 1.0)], t=1.0)
    assert out and out[0].level == 2   # round(3*0.5)=2

def test_stepwise_deescalation():
    """Channel drops stepwise 3→2→1→0 across successive cooldown windows."""
    p = NudgePolicy([VectorSubscription(vector="yelling")], cooldown_s=20.0)

    # t=1: Set level to 3
    out = p.on_events([ev("yelling", 3, 1.0)], t=1.0)
    assert len(out) == 1 and out[0].channel == "A" and out[0].level == 3

    # t=22: Should drop to 2 (20 seconds after t=1)
    out = p.on_events([], t=22.0)
    assert len(out) == 1 and out[0].channel == "A" and out[0].level == 2

    # t=43: Should drop to 1 (20 seconds after t=22)
    out = p.on_events([], t=43.0)
    assert len(out) == 1 and out[0].channel == "A" and out[0].level == 1

    # t=64: Should drop to 0 (20 seconds after t=43)
    out = p.on_events([], t=64.0)
    assert len(out) == 1 and out[0].channel == "A" and out[0].level == 0

def test_sustained_qualifying_event():
    """Clock refreshes when event at current level arrives (Critical 2 repro).

    When a qualifying event (level >= current) arrives, the decay clock resets.
    Subsequent drops only happen after cooldown_s seconds from that refresh.
    """
    p = NudgePolicy([VectorSubscription(vector="yelling")], cooldown_s=20.0)

    # t=1: yelling=3 → level becomes 3, clock set to t=1
    out = p.on_events([ev("yelling", 3, 1.0)], t=1.0)
    assert len(out) == 1 and out[0].level == 3

    # t=15: yelling=3 → E==current, refresh clock to t=15, no emit
    out = p.on_events([ev("yelling", 3, 15.0)], t=15.0)
    assert out == []

    # t=22: no events → 22-15=7 < 20, NO drop (clock was refreshed at t=15)
    out = p.on_events([], t=22.0)
    assert out == []
    assert p.current()["A"] == 3

def test_two_vectors_multi_step():
    """Two vectors on same channel with stepwise de-escalation (Important 1).

    When a lower vector arrives, level doesn't snap; only drops stepwise after cooldown.
    When the lower vector sustains, it refreshes the clock once it reaches current level.
    """
    p = NudgePolicy([
        VectorSubscription(vector="yelling", channel="A", sensitivity=1.0),
        VectorSubscription(vector="hr_spike", channel="A", sensitivity=1.0),
    ], cooldown_s=20.0)

    # t=1: yelling=3 → level 3, clock at t=1
    out = p.on_events([ev("yelling", 3, 1.0)], t=1.0)
    assert len(out) == 1 and out[0].level == 3

    # t=5: hr_spike=1 → E=1 < current=3, no cooldown yet, no event
    out = p.on_events([ev("hr_spike", 1, 5.0)], t=5.0)
    assert out == []
    assert p.current()["A"] == 3

    # t=22: hr_spike=1 → 22-1=21 > 20, drop to 2 (stepwise, not snap)
    out = p.on_events([ev("hr_spike", 1, 22.0)], t=22.0)
    assert len(out) == 1 and out[0].level == 2
    assert p.current()["A"] == 2

    # t=43: hr_spike=1 → 43-22=21 > 20, drop to 1 (stepwise)
    out = p.on_events([ev("hr_spike", 1, 43.0)], t=43.0)
    assert len(out) == 1 and out[0].level == 1
    assert p.current()["A"] == 1

    # t=50: hr_spike=1 → E==current, refresh clock, no event
    out = p.on_events([ev("hr_spike", 1, 50.0)], t=50.0)
    assert out == []
    assert p.current()["A"] == 1

def test_rounding_half_up():
    """Rounding uses half-up (not banker's) to match Kotlin Math.round.

    sensitivity=0.5, level=1 → 0.5 should round to 1 (half-up), not 0 (banker's).
    """
    # Test: 0.5 rounds to 1
    p = NudgePolicy([VectorSubscription(vector="yelling", sensitivity=0.5)])
    out = p.on_events([ev("yelling", 1, 1.0)], t=1.0)
    assert len(out) == 1 and out[0].level == 1  # Half-up: 0.5 → 1

    # Test: 1.5 rounds to 2 (also works with half-up)
    p2 = NudgePolicy([VectorSubscription(vector="yelling", sensitivity=0.5)])
    out = p2.on_events([ev("yelling", 3, 1.0)], t=1.0)
    assert len(out) == 1 and out[0].level == 2  # Half-up: 1.5 → 2
