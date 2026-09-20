"""server/watch/heat_judge.py — the server-side heat judge (Track 1, step 3).

Five layers, cheapest first:

1. **Pure rules** on synthetic dims: the smoother's weights, the buffer/hop
   cadence, and every branch of ``verdict_from`` — including the ordering
   decision (confirm outranks the valence veto) that the CONFER numbers
   forced, because reverting it silently would cost 12 points of heated-window
   survival with every test still green.
2. **The calibrated thresholds**, pinned. They are the output of
   ``scripts/heat_judge_calibrate.py``; a change to either number must change
   this file and the decision record together.
3. **Degrade and latency.** A model that raises must produce ``unknown`` and
   the wrist must still buzz (a model outage can never mute the product); a
   model that takes 130 ms must produce a verdict well inside 1.5 s of the
   window's end.
4. **The wiring**: relay veto/confirm/unknown x mode x rung, the tone-flag
   surfacing gate, and ``audio_escalated`` in the stored analysis.
5. **Corpus gates** (skipped without the gitignored eval data): CONFER's
   human-rated heated windows must not be vetoed more than 10% of the time,
   and replaying AMI + SBCSAE through the judge must never RAISE the dose.

The corpus layer is the one that can only run on a machine with tmp/ built;
everything above it runs in CI. That split is deliberate — the arithmetic is
pinned unconditionally, the corpus-scale claims are pinned where the corpora
live.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from models.audio import ToneFlagEvent, TurnLocalEvent, TurnProsody, TurnTextTone
from watch import heat_judge as hj
from watch import relay

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))


SR = hj.SAMPLE_RATE


def dims(arousal: float, valence: float, dominance: float = 0.5) -> dict[str, float]:
    return {"arousal": arousal, "valence": valence, "dominance": dominance}


def frame(seconds: float, *, amplitude: int = 3000) -> bytes:
    """``seconds`` of PCM16 at the wire contract's rate."""
    n = int(seconds * SR)
    return (np.full(n, amplitude, dtype="<i2")).tobytes()


@pytest.fixture(autouse=True)
def _clean_registries(monkeypatch):
    """No test in this file may ever touch the real 1.3 GB model.

    ``relay.register_live_session`` attaches a judge, and with the weights
    on the developer's disk that judge would be built with the DEFAULT
    scorer — ``tone_id.classify_pcm``. Pinning ``weights_present`` to False
    makes every attach a no-op, so a test that wants a judge has to say what
    it scores with. (Found the expensive way: the first version of this file
    spent four minutes of CPU inside WavLM before anyone noticed.)
    """
    monkeypatch.setattr(hj, "weights_present", lambda: False)
    # A first-rung tap with no verdict yet WAITS (2 s in production). Tests
    # assert what happens after the wait, not how long it is.
    monkeypatch.setattr(hj, "JUDGE_WAIT_S", 0.02)
    monkeypatch.setattr(hj, "JUDGE_POLL_S", 0.005)
    hj.reset()
    relay._registry.clear()
    yield
    hj.reset()
    relay._registry.clear()


# --------------------------------------------------------------- 1. pure --

def test_smooth_weights_the_newest_score_most_and_renormalises():
    assert hj.smooth([]) is None
    assert hj.smooth([0.4]) == pytest.approx(0.4), "one score is its own average"
    # alpha .5 over three: weights .5/.25/.125 -> .571/.286/.143 after
    # renormalising, newest LAST in the input.
    got = hj.smooth([0.0, 0.0, 1.0])
    assert got == pytest.approx(0.5 / 0.875)
    assert hj.smooth([1.0, 0.0, 0.0]) == pytest.approx(0.125 / 0.875)
    # Only the last SMOOTHING_N matter.
    assert hj.smooth([9.9, 0.2, 0.2, 0.2]) == pytest.approx(0.2)


def test_one_odd_window_does_not_flip_a_verdict_that_three_do():
    """The whole reason the smoother exists — and its honest limit.

    One window reading 0.60 valence against two at 0.30 smooths to 0.472 and
    stays ``unknown``; three windows at 0.60 are a real state and veto. The
    newest score still carries 57% of the weight, so this buys ONE window of
    patience, not immunity: a window at 0.90 would cross the bar on its own
    under any weighting, which is correct — 0.90 against 0.30 twice averages
    above 0.48 however you slice it.
    """
    calm = dims(0.50, 0.30)
    odd = dims(0.50, 0.60)
    assert hj.verdict_from([calm, calm, odd]).verdict == hj.VERDICT_UNKNOWN
    assert hj.verdict_from([odd, odd, odd]).verdict == hj.VERDICT_VETO


def test_the_buffer_keeps_two_seconds_and_a_score_comes_due_every_hop():
    judge = hj.HeatJudge(scorer=lambda pcm, sr: dims(0.5, 0.3))
    # Nothing is due until a FULL window exists.
    assert judge.feed(frame(0.5)) is False
    assert judge.feed(frame(1.0)) is False
    assert judge.feed(frame(0.5)) is True, "2.0 s buffered and >= 1 s new"
    judge.score_now()
    assert judge.feed(frame(0.5)) is False, "half a hop is not a hop"
    assert judge.feed(frame(0.5)) is True
    judge.score_now()
    # And the buffer never grows past the window.
    for _ in range(20):
        judge.feed(frame(1.0))
    assert len(judge._buf) == int(hj.WINDOW_S * SR) * 2


def test_a_frame_at_the_wrong_sample_rate_is_dropped_not_resampled():
    judge = hj.HeatJudge(scorer=lambda pcm, sr: dims(0.5, 0.3))
    assert judge.feed(frame(3.0), 8000) is False
    assert len(judge._buf) == 0, "a wire bug must not be silently papered over"


@pytest.mark.parametrize("arousal,valence,expected,why", [
    (0.90, 0.30, hj.VERDICT_CONFIRM, "activated and unpleasant"),
    (0.90, 0.60, hj.VERDICT_CONFIRM, "confirm outranks the valence veto"),
    (0.50, 0.60, hj.VERDICT_VETO, "loud but PLEASANT — laughing, not shouting"),
    (0.30, 0.30, hj.VERDICT_VETO, "loud but FLAT — a projected voice"),
    (0.50, 0.30, hj.VERDICT_UNKNOWN, "between the floor and the bar"),
])
def test_every_verdict_branch(arousal, valence, expected, why):
    verdict = hj.verdict_from([dims(arousal, valence)] * hj.MIN_SCORES)
    assert verdict.verdict == expected, why
    assert verdict.arousal == pytest.approx(arousal)
    assert verdict.valence == pytest.approx(valence)
    assert verdict.reason


def test_fewer_than_two_scores_is_unknown_and_unknown_never_suppresses():
    for n in (0, 1):
        verdict = hj.verdict_from([dims(0.2, 0.9)] * n, acts=True)
        assert verdict.verdict == hj.VERDICT_UNKNOWN
        assert verdict.n_scores == n
        assert verdict.suppresses_escalation() is False, (
            "the first second of a session must never take a buzz away"
        )


def test_boundaries_are_where_the_constants_say():
    """Which side of each bar is inclusive. The margins are 1e-4 rather than
    exact equality because the smoother is a weighted mean: ``smooth([x, x])``
    is x to fifteen decimal places, not to sixteen, and a test that pins an
    exact float would fail on arithmetic rather than on behaviour."""
    eps = 1e-4
    assert hj.verdict_from([dims(0.50, hj.VALENCE_VETO_MAX - eps)] * 2).verdict == hj.VERDICT_UNKNOWN
    assert hj.verdict_from([dims(0.50, hj.VALENCE_VETO_MAX + eps)] * 2).verdict == hj.VERDICT_VETO
    assert hj.verdict_from([dims(hj.CALM_AROUSAL_FLOOR + eps, 0.30)] * 2).verdict == hj.VERDICT_UNKNOWN
    assert hj.verdict_from([dims(hj.CALM_AROUSAL_FLOOR - eps, 0.30)] * 2).verdict == hj.VERDICT_VETO
    assert hj.verdict_from([dims(hj.CONFIRM_AROUSAL + eps, 0.30)] * 2).verdict == hj.VERDICT_CONFIRM
    assert hj.verdict_from([dims(hj.CONFIRM_AROUSAL - eps, 0.30)] * 2).verdict == hj.VERDICT_UNKNOWN


def test_only_confirm_surfaces():
    for arousal, valence, expected in ((0.90, 0.30, True), (0.50, 0.60, False), (0.50, 0.30, False)):
        assert hj.surfaces(hj.verdict_from([dims(arousal, valence)] * 2)) is expected


# ------------------------------------------------------- 2. the thresholds --

def test_the_thresholds_are_calibrated():
    """Pinned output of scripts/heat_judge_calibrate.py (2026-09-20).

    Chosen as the pair maximising the acted-HAPPY veto subject to three
    gates at once: <= 10% of CONFER's human-rated heated windows vetoed,
    >= 50% of over-rung acted HAPPY vetoed, <= 10% of over-rung acted ANGRY
    vetoed (PR #186's own recall gate). Move either number and the decision
    record moves with it.
    """
    assert hj.CALM_AROUSAL_FLOOR == 0.40
    assert hj.CONFIRM_AROUSAL == 0.78
    assert hj.CALM_AROUSAL_FLOOR < hj.CONFIRM_AROUSAL, "there must be an undecided band"


def test_the_valence_bar_is_the_same_number_the_relay_uses():
    """Two constants, one threshold. heat_judge cannot import relay (relay
    imports heat_judge), so the equality is pinned here instead."""
    assert hj.VALENCE_VETO_MAX == relay.VALENCE_VETO_MAX == 0.48


def test_the_window_is_two_seconds_scored_every_second():
    assert (hj.WINDOW_S, hj.HOP_S, hj.SMOOTHING_N, hj.MIN_SCORES) == (2.0, 1.0, 3, 2)


# -------------------------------------------------- 3. degrade and latency --

def test_a_broken_model_degrades_to_unknown_and_logs_once(caplog):
    calls = []

    def explode(pcm, sr):
        calls.append(1)
        raise RuntimeError("model is on fire")

    judge = hj.HeatJudge(scorer=explode)
    with caplog.at_level("WARNING", logger="watch.heat_judge"):
        for _ in range(4):
            judge.feed(frame(1.0))
            judge.score_now()
    assert len(calls) == 3, "kept trying"
    assert judge.verdict().verdict == hj.VERDICT_UNKNOWN
    assert caplog.text.count("scorer failed") == 1, "one line per session, not per second"


def test_a_categorical_backend_has_no_opinion_rather_than_a_wrong_one():
    """superb_er / iemocap return a 4-class softmax with no valence axis.
    That is a configuration answer, not a failure — and certainly not a veto."""
    judge = hj.HeatJudge(scorer=lambda pcm, sr: {"angry": 0.9, "happy": 0.1})
    for _ in range(3):
        judge.feed(frame(1.0))
        judge.score_now()
    assert judge.verdict(acts=True).verdict == hj.VERDICT_UNKNOWN


def test_a_broken_model_still_lets_the_wrist_buzz(monkeypatch):
    """The product rule, end to end: model raises -> unknown -> escalation
    happens anyway. A model outage must never mute the product."""
    monkeypatch.setattr(hj, "_mode", lambda: "on")

    def explode(pcm, sr):
        raise RuntimeError("model is on fire")

    async def run():
        session, rec = _live_session()
        judge = hj.attach("alice", scorer=explode)
        for _ in range(4):
            judge.feed(frame(1.0))
            judge.score_now()
        assert judge.verdict(acts=True).verdict == hj.VERDICT_UNKNOWN
        relay.push_turn_local("alice", _loud_turn())
        await _settle()
        assert [(e.vector, e.level) for e in rec.calls[0][0]] == [("yelling", 1)]

    asyncio.run(run())


def test_a_verdict_lands_well_inside_the_latency_budget():
    """A fake 130 ms model — the measured cost of tone_id.classify_pcm on a
    2 s clip — must produce a verdict within 1.5 s of the window's END.
    Latency is stamped from the arrival of the window's last sample, which
    is what the wearer experiences; the model call is the whole of it."""
    def slow(pcm, sr):
        time.sleep(0.130)
        return dims(0.90, 0.30)

    judge = hj.HeatJudge(scorer=slow)
    for _ in range(2):
        judge.feed(frame(1.0))
    verdict = judge.score_now()
    judge.feed(frame(1.0))
    verdict = judge.score_now()
    assert verdict.verdict == hj.VERDICT_CONFIRM
    assert verdict.latency_ms is not None
    assert 100.0 < verdict.latency_ms < 1500.0, verdict.latency_ms


def test_a_slow_model_drops_windows_rather_than_queueing_them():
    """One score at a time. A 4 s-old verdict is worse than no verdict, so a
    model slower than the hop must DROP windows, never queue them."""
    async def run():
        judge = hj.HeatJudge(scorer=lambda pcm, sr: dims(0.9, 0.3))
        judge.scoring = True          # pretend a score is already in flight
        hj._judges["alice"] = judge
        for _ in range(5):
            hj.feed_pcm("alice", frame(1.0))
        await _settle()
        assert judge.verdict().n_scores == 0, "nothing queued behind the in-flight score"
        judge.scoring = False
        hj.feed_pcm("alice", frame(1.0))
        await _settle()
        assert judge.verdict().n_scores == 1, "and it resumes once that one lands"

    asyncio.run(run())


def test_the_first_second_of_a_session_does_not_deadlock():
    """Regression guard. ``score_now`` reads the buffer under the judge's
    lock and returns ``verdict()`` when there is not yet a full window —
    which takes the same lock. With a non-reentrant Lock that hangs the
    session's very first second, with no exception and no log line."""
    judge = hj.HeatJudge(scorer=lambda pcm, sr: dims(0.9, 0.3))
    judge.feed(frame(0.5))
    assert judge.score_now().verdict == hj.VERDICT_UNKNOWN


# ------------------------------------------------------------ 4. the wiring --

class _Recorder:
    def __init__(self):
        self.calls: list[tuple[list, float]] = []

    async def emit(self, events, t):
        self.calls.append((events, t))


def _live_session(account: str = "alice"):
    from watch.models import EnrollmentBaseline
    from watch.vectors import VectorEngine

    engine = VectorEngine(EnrollmentBaseline(
        account_id=account, rms_db=-30.0, f0_median=120.0, updated_at="x"))
    engine.t = 7.0
    rec = _Recorder()
    session = relay.LiveWatchSession(
        account_id=account, live_session_id="ls-1", engine=engine, emit=rec.emit,
        loop=asyncio.get_running_loop(),
    )
    relay.register_live_session(session)
    return session, rec


def _loud_turn(rms_dbfs: float = -22.0) -> TurnLocalEvent:
    """-22 dBFS against a -30 dB baseline = +8 dB = the FIRST rung."""
    return TurnLocalEvent(
        session_id="phone-1", speaker="Speaker A", is_self=True, text="I said I'm fine.",
        start_time=12.5, end_time=14.25, transcript_source="on-device",
        prosody=TurnProsody(rms_dbfs=rms_dbfs),
    )


async def _settle():
    """Let everything the relay scheduled actually run — including a
    first-rung escalation that is waiting on the judge."""
    await asyncio.sleep(hj.JUDGE_WAIT_S + 0.05)
    for _ in range(40):
        await asyncio.sleep(0)


def _seed(account: str, arousal: float, valence: float):
    """Give ``account`` a judge that has already settled on these dims.

    Replaces any existing judge outright: ``judge_for(create=True, ...)``
    returns the EXISTING one and silently ignores the kwargs, which is right
    for production (attach is idempotent) and a trap in a test.
    """
    judge = hj.HeatJudge(scorer=lambda pcm, sr: dims(arousal, valence))
    hj._judges[account] = judge
    for _ in range(4):
        judge.feed(frame(1.0))
        judge.score_now()
    return judge


@pytest.mark.parametrize("mode,arousal,valence,expect_yelling,why", [
    ("on", 0.50, 0.60, False, "veto suppresses the first rung"),
    ("on", 0.90, 0.30, True, "confirm escalates"),
    ("on", 0.50, 0.30, True, "unknown escalates — never mute the product"),
    ("dark", 0.50, 0.60, True, "dark computes and logs, never changes a buzz"),
    ("off", 0.50, 0.60, True, "off has no judge at all"),
])
def test_mode_by_verdict(monkeypatch, mode, arousal, valence, expect_yelling, why):
    monkeypatch.setattr(hj, "_mode", lambda: mode)

    async def run():
        session, rec = _live_session()
        _seed("alice", arousal, valence)
        relay.push_turn_local("alice", _loud_turn())
        await _settle()
        vectors = [e.vector for call in rec.calls for e in call[0]]
        assert (("yelling" in vectors) is expect_yelling), why

    asyncio.run(run())


def test_higher_rungs_are_vetoed_but_never_delayed(monkeypatch):
    """By rung 2 the wearer is 10 dB over their own baseline; a late buzz is
    worse than an imperfect one. So the judge is CONSULTED at every rung but
    only the first rung may wait for it."""
    monkeypatch.setattr(hj, "_mode", lambda: "on")
    # rung 2 with a verdict already in hand -> vetoed, synchronously.
    assert hj.judge_escalation("nobody", 2) is None, "no judge, no opinion"

    async def run():
        session, rec = _live_session()
        _seed("alice", 0.50, 0.60)
        assert isinstance(hj.judge_escalation("alice", 2), hj.HeatVerdict), "never an awaitable"
        relay.push_turn_local("alice", _loud_turn(rms_dbfs=-18.0))  # +12 dB = rung 2
        await _settle()
        assert rec.calls == [], "a loud PLEASANT rung-2 turn is still vetoed"

    asyncio.run(run())


def test_a_first_rung_escalation_waits_for_the_judge_then_goes_ahead(monkeypatch):
    """The wait is bounded and fails OPEN: nothing ever answers, so after
    JUDGE_WAIT_S the unknown verdict escalates exactly as today."""
    monkeypatch.setattr(hj, "_mode", lambda: "on")
    monkeypatch.setattr(hj, "JUDGE_WAIT_S", 0.05)
    monkeypatch.setattr(hj, "JUDGE_POLL_S", 0.01)

    async def run():
        session, rec = _live_session()
        hj._judges["alice"] = hj.HeatJudge(scorer=lambda pcm, sr: dims(0.5, 0.3))
        pending = hj.judge_escalation("alice", hj.FIRST_RUNG)
        assert asyncio.iscoroutine(pending), "an unknown first rung waits"
        verdict = await pending
        assert verdict.verdict == hj.VERDICT_UNKNOWN
        relay.push_turn_local("alice", _loud_turn())
        await asyncio.sleep(0.12)
        await _settle()
        assert [(e.vector, e.level) for call in rec.calls for e in call[0]] == [("yelling", 1)]
        assert rec.calls[0][1] == 7.0, "the escalation keeps its ORIGINAL stream clock"

    asyncio.run(run())


def test_the_judge_never_touches_the_tone_lane(monkeypatch):
    """The words are the better signal (PR #186's rule, kept). A vetoed
    loudness turn whose TEXT reads hostile must still reach the wrist."""
    monkeypatch.setattr(hj, "_mode", lambda: "on")

    async def run():
        session, rec = _live_session()
        _seed("alice", 0.50, 0.60)
        turn = _loud_turn().model_copy(update={
            "text_tone": TurnTextTone(frustration=90, label="defensive")})
        relay.push_turn_local("alice", turn)
        await _settle()
        assert [e.vector for call in rec.calls for e in call[0]] == ["aggressive_tone"]

    asyncio.run(run())


def test_tone_level_stays_inert_for_a_dimensional_backend():
    """odyssey_dim reports confidence 0.0 by construction (a raw dimensional
    reading carries no label to be confident about), and relay.tone_level
    ignores a flag below TONE_FLAG_MIN_CONFIDENCE. So flipping
    MINDSHIFT_TONE_AUDIO to `on` cannot make the AUDIO tone drive the
    aggressive_tone lane — only the phone's TEXT tone does. Pinned because
    the flag flip would otherwise be a silent second escalation source."""
    flag = ToneFlagEvent(
        session_id="s", speaker="Speaker A", start_time=0.0, end_time=2.0, source="audio",
        scores={"arousal": 0.9, "valence": 0.2, "dominance": 0.9,
                "frustration": 99.0, "defensiveness": 99.0},
        label=("unscored"), confidence=0.0,
    )
    assert relay.TONE_FLAG_MIN_CONFIDENCE == 0.5
    assert relay.tone_level(None, flag) == 0


def test_the_series_are_recorded_per_second_and_survive_detach():
    judge = hj._judges.setdefault("alice", hj.HeatJudge(scorer=lambda pcm, sr: dims(0.9, 0.3)))
    for _ in range(5):
        judge.feed(frame(1.0))
        judge.score_now()
    series = hj.series_for("alice")
    assert len(series["arousal"]) == len(series["valence"]) == len(series["judge"]) == 4
    assert series["arousal"][0] == pytest.approx(0.9)
    # The first window has one score -> unknown; then confirm.
    assert series["judge"] == [hj.VERDICT_CODES[hj.VERDICT_UNKNOWN]] + [
        hj.VERDICT_CODES[hj.VERDICT_CONFIRM]] * 3
    # The phone's socket may close before the watch's; the series must not
    # vanish with it.
    handed = hj.detach("alice")
    assert handed == series
    assert hj.series_for("alice") == series
    hj.forget_series("alice")
    assert hj.series_for("alice") is None


def test_the_ws_handler_persists_the_series_as_floats(monkeypatch):
    """LiveSession.series is typed dict[str, list[float]], so the verdict
    rides as a code rather than as a word. Pinned so a later widening of
    that model does not silently change what is stored."""
    from watch.models import LiveSession, Participant
    from watch.routers import ws as ws_router

    judge = hj._judges.setdefault("alice", hj.HeatJudge(scorer=lambda pcm, sr: dims(0.9, 0.3)))
    for _ in range(4):
        judge.feed(frame(1.0))
        judge.score_now()
    extra = ws_router._heat_series("alice")
    assert set(extra) == {"arousal", "valence", "judge"}
    doc = LiveSession(
        id="x", owner_account="alice", started_at="t", ended_at="t", status="captured",
        participants=[Participant(id="self", role="self", speaker_label="You")],
        vector_events=[], nudge_events=[],
        series={"rms_db": [], "hr_bpm": [], "hr_t": [], **extra},
    )
    assert doc.series["judge"][-1] == 1.0
    assert ws_router._heat_series("nobody") == {}, "no judge -> no keys at all"


# ------------------------------------------- 4b. surfacing and the analysis --

def test_only_a_confirm_surfaces_a_tone_flag(monkeypatch):
    """MINDSHIFT_TONE_AUDIO=on flips tone_id.surface_allowed(), which sends a
    ToneFlagEvent to the phone, fans it out to every other call participant,
    and hands it to the watch relay. All three now require a confirm."""
    import audio_pipeline as ap

    class _Ctx:
        uid = "alice"
        session_id = "s"

    ctx = _Ctx()
    confirmed = {"scores": dims(0.90, 0.30)}
    vetoed = {"scores": dims(0.50, 0.60)}
    undecided = {"scores": dims(0.50, 0.30)}
    def gated(result):
        return ap._heat_confirms(ctx, ap._heat_verdict_for(ctx, result), dimensional=True)

    assert gated(confirmed) is True
    assert gated(vetoed) is False
    assert gated(undecided) is False
    assert ap._heat_confirms(ctx, None, dimensional=True) is False, "fails CLOSED with no judge"


def test_a_categorical_backend_is_passed_through_not_silently_switched_off():
    """superb_er / iemocap produce a 4-class softmax with no arousal or
    valence, so the judge cannot have an opinion about one by construction.
    Gating those to death would turn a working configuration off without
    anyone deciding to. odyssey_dim — the default, and the one being turned
    on — is dimensional and IS gated."""
    import audio_pipeline as ap

    class _Ctx:
        uid = "alice"
        session_id = "s"

    assert ap._heat_confirms(_Ctx(), None, dimensional=False) is True


def test_the_live_judge_outranks_the_turns_own_dims(monkeypatch):
    """verdict_for_turn prefers the rolling 2 s judge when it has an opinion
    — that IS the module. The turn's own dims are the fallback for a session
    with no relayed frame feed."""
    monkeypatch.setattr(hj, "_mode", lambda: "on")
    _seed("alice", 0.50, 0.60)          # live judge says VETO
    verdict = hj.verdict_for_turn("alice", dims(0.95, 0.10))   # this turn looks hot
    assert verdict.verdict == hj.VERDICT_VETO
    assert hj.verdict_for_turn("bob", dims(0.95, 0.10)).verdict == hj.VERDICT_CONFIRM
    assert hj.verdict_for_turn("bob").verdict == hj.VERDICT_UNKNOWN


@pytest.mark.parametrize("verdict_code,expected", [
    (hj.VERDICT_CODES[hj.VERDICT_CONFIRM], True),
    (hj.VERDICT_CODES[hj.VERDICT_UNKNOWN], False),
    (hj.VERDICT_CODES[hj.VERDICT_VETO], False),
    (None, False),
])
def test_only_a_confirmed_turn_is_stored_as_audio_escalated(verdict_code, expected):
    """audio_escalated goes into the stored analysis that Growth, Replay and
    YourDay render back weeks later — the most durable claim the tone model
    makes about anybody. The dims survive either way; only the ESCALATION
    verdict needs a confirm."""
    import live_sessions as ls

    scores = {"arousal": 0.9, "valence": 0.2, "dominance": 0.9}
    if verdict_code is not None:
        scores[hj.HEAT_VERDICT_KEY] = verdict_code
    flag = {
        "source": "audio", "start_time": 0.0, "end_time": 2.0, "confidence": 0.0,
        "label": next(iter(ls.ESCALATION_LABELS)), "scores": scores,
    }
    turns = [{"speaker": "You", "start_time": 0.0, "end_time": 2.0}]
    rows = ls.turn_tone_rows(turns, "You", {}, [flag], audio_allowed=True)
    assert rows[0]["audio_label"] is not None, "the dims/label are kept either way"
    assert rows[0]["audio_escalated"] is expected


# ------------------------------------------------------- 5. corpus gates --
# These need the gitignored eval corpora. A git WORKTREE has its own empty
# tmp/, so the calibration script's eval_tmp() looks up to the main checkout
# — the corpora are multi-GB and are never copied per worktree.

try:
    import heat_judge_calibrate as calib
    _CALIB_OK = calib.REFS.exists()
except Exception:  # pragma: no cover — scripts/ deps absent
    calib = None
    _CALIB_OK = False

corpus = pytest.mark.skipif(
    not _CALIB_OK,
    reason="heat-map reference series not built — run scripts/heat_reference.py",
)


@corpus
def test_confer_gate_at_most_a_tenth_of_heated_windows_are_vetoed():
    """The gate that stops the judge silencing real arguments. CONFER is ten
    human raters' continuous conflict score — people, not a model — which
    makes it the strongest y-axis in the repo."""
    arousal, valence, rating, _ = calib.confer_windows()
    if not len(arousal):
        pytest.skip("CONFER manifest absent")
    heated = rating >= calib.HEATED_AT
    assert heated.sum() >= 20, "too few heated windows to gate on"
    verdicts = calib.verdicts(
        arousal[heated], valence[heated], hj.CALM_AROUSAL_FLOOR, hj.CONFIRM_AROUSAL)
    vetoed = float((verdicts == hj.VERDICT_VETO).mean())
    assert vetoed <= calib.MAX_HEATED_VETOED, (
        f"{vetoed:.1%} of human-rated heated windows vetoed")
    confirmed = float((verdicts == hj.VERDICT_CONFIRM).mean())
    assert confirmed >= 0.80, f"only {confirmed:.1%} of heated windows confirmed"


@corpus
def test_the_rule_order_is_what_the_confer_numbers_chose():
    """Reverting to "valence vetoes before confirm is considered" is a
    twelve-point regression in heated-window survival that no unit test can
    see. It is pinned HERE, against the corpus that measured it."""
    arousal, valence, rating, _ = calib.confer_windows()
    if not len(arousal):
        pytest.skip("CONFER manifest absent")
    heated = rating >= calib.HEATED_AT
    shipped = calib.verdicts(arousal[heated], valence[heated],
                             hj.CALM_AROUSAL_FLOOR, hj.CONFIRM_AROUSAL)
    veto_first = calib.verdicts(arousal[heated], valence[heated],
                                hj.CALM_AROUSAL_FLOOR, hj.CONFIRM_AROUSAL, veto_first=True)
    assert float((veto_first == hj.VERDICT_VETO).mean()) > 0.15
    assert float((shipped == hj.VERDICT_VETO).mean()) <= calib.MAX_HEATED_VETOED


@corpus
def test_cremad_gate_the_happy_false_alarms_fall_and_the_anger_survives():
    """The defect this module exists to fix, and the recall PR #186 refused
    to give up, measured on the SAME two artefacts that decision used."""
    cremad = calib.cremad_dims()
    if "happy" not in cremad or "angry" not in cremad:
        pytest.skip("CREMA-D valence probe / heat rubric absent")
    rates = {}
    for emotion in ("happy", "angry"):
        arousal, valence, over_db = cremad[emotion]
        loud = over_db >= calib.FIRST_RUNG_DB
        verdicts = calib.verdicts(arousal[loud], valence[loud],
                                  hj.CALM_AROUSAL_FLOOR, hj.CONFIRM_AROUSAL)
        rates[emotion] = float((verdicts == hj.VERDICT_VETO).mean())
    assert rates["happy"] >= calib.MIN_HAPPY_VETOED, rates
    assert rates["angry"] <= calib.MAX_ANGRY_VETOED, rates


@corpus
@pytest.mark.parametrize("corpus_name", ["AMI", "SBCSAE"])
def test_the_judge_gate_never_raises_the_dose(corpus_name):
    """Replay the shipped nudge chain over every recording with and without
    the judge, from the CACHED tone series — no audio, no model.

    The claim is about the MEDIAN, not every recording, and that is not a
    weaker claim dressed up: a veto removes an ESCALATION, and removing one
    can leave the ladder at a lower level that later re-escalates — a second
    first-rung buzz with a fresh PRD §6 reminder cycle where the ungated
    replay sat at one high level and reminded slowly. So an individual
    recording can come out marginally worse. Measured below, and bounded: a
    clear majority of recordings must fall or stay flat, the median must not
    rise, and the total across the corpus must not rise.
    """
    import conversation_audit as ca

    rows = []
    for path in sorted(calib.REFS.glob("*.json")):
        name = path.stem
        if corpus_name == "AMI" and not name.startswith(("ES", "IS", "TS")):
            continue
        if corpus_name == "SBCSAE" and not name.startswith("sbcsae_"):
            continue
        ref = json.loads(path.read_text())
        over, gate = _proxy_over_and_gate(ref)
        if over is None:
            continue
        before = len(ca.replay(over, False, True))
        after = len(ca.replay(over, False, True, gate=gate))
        rows.append((name, before, after))
    assert len(rows) >= 4, f"only {len(rows)} {corpus_name} recordings found"
    before_all = [b for _, b, _ in rows]
    after_all = [a for _, _, a in rows]
    fell = [name for name, before, after in rows if after < before]
    rose = [name for name, before, after in rows if after > before]
    assert np.median(after_all) <= np.median(before_all), (
        f"median dose rose: {np.median(before_all)} -> {np.median(after_all)}")
    assert sum(after_all) <= sum(before_all), "total dose rose"
    assert len(rose) <= len(fell) // 4, (
        f"{len(rose)} of {len(rows)} {corpus_name} recordings got worse: {rose}")
    assert fell, f"the judge changed nothing on any {corpus_name} recording"


def _proxy_over_and_gate(ref: dict):
    """Per-second dB-over-baseline and judge gate from ONE cached series.

    The reference series hold 5 s windows, subsampled for long recordings, so
    this expands each SCORED window over its own seconds and leaves the rest
    at 0.0 (the detector's own "unvoiced" convention). It is a proxy for the
    recording's real per-second loudness — the absolute dose differs from
    scripts/heat_map.py's — but both sides of the comparison read the same
    proxy, which is what the assertion is about. The full-audio numbers are
    in scripts/heat_judge_calibrate.py --dose.
    """
    idx = ref.get("window_idx")
    over_db = ref.get("ours_db_over")
    if not idx or over_db is None or len(idx) != len(over_db):
        return None, None
    w = int(ref["win_s"])
    n = (max(idx) + 1) * w
    over = np.zeros(n)
    gate = np.ones(n, dtype=bool)
    verdicts = calib.verdicts(
        np.asarray(ref["arousal"], dtype=float), np.asarray(ref["valence"], dtype=float),
        hj.CALM_AROUSAL_FLOOR, hj.CONFIRM_AROUSAL,
    )
    for k, value, verdict in zip(idx, over_db, verdicts):
        over[k * w:(k + 1) * w] = float(value)
        if verdict == hj.VERDICT_VETO:
            gate[k * w:(k + 1) * w] = False
    return over, gate
