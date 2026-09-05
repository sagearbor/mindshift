# Track 1 — realtime watch nudge (2026-08-24). New module, no gauge ancestor.
"""Phone -> watch relay: feeds the phone's realtime ``turn_local`` reports into
the paired watch's live escalation lane.

Why this exists
---------------
The watch's own nudge path (``watch/routers/ws.py`` -> ``VectorEngine`` ->
``NudgePolicy``) only ever hears the watch mic, so its escalation input is
"how many dB over the wearer's baseline" — physics, never judgment. That is
deliberately blind to a calm-volume, hostile-tone turn ("I said I'm *fine*"
at conversational loudness). The phone's realtime path (``server/audio_pipeline.py``,
Track 3-server) DOES see tone: each phone-finalized turn arrives as a
``models.audio.TurnLocalEvent`` carrying the on-device ``text_tone`` scores
and ``prosody``. This module is the seam between the two: Track 3-server
calls ``push_turn_local(uid, event)`` (guarded import — the call is optional
on their side) and this module turns that turn into ``VectorEvent``s for the
uid's *currently live* watch session, so the wrist buzzes for tone as well
as volume.

Contract (locked with Track 3-server)
-------------------------------------
* ``push_turn_local(uid: str, event: TurnLocalEvent, *, tone_flag=None) -> None``.
  Synchronous, never raises for "nothing to do", never blocks on the socket.
* Escalation counts ONLY for the wearer's OWN turns: ``event.is_self is True``.
  ``False`` (someone else) and ``None`` (the phone couldn't tell) are both
  ignored — the whole product is "measure the wearer, never the other party"
  (``watch/vectors.py``'s bias guard), and an unknown speaker is not evidence
  about the wearer.
* Safe when the uid has no paired watch mid-session: a debug log and a no-op.
  "Paired watch" here means "this account has an OPEN watch WS live session
  right now" — that is the only moment a nudge can physically reach a wrist,
  and it's exactly what ``watch/routers/ws.py`` registers/unregisters via
  ``register_live_session``/``unregister_live_session``. (A watch that has
  completed ``POST /me/pair/claim`` but isn't streaming has no socket to push
  a nudge down, so ``pairing_store`` is the wrong lookup for this purpose.)

How the two inputs combine
--------------------------
``NudgePolicy`` already takes the MAX scaled level over every event a call
delivers for a channel, so "combine as max(level_db, level_tone)" falls out
of feeding both a ``yelling`` event (from ``prosody.rms_dbfs`` over baseline)
and an ``aggressive_tone`` event (from the tone scores) into the same call —
no second combiner, and the watch's own PCM path is untouched (byte-identical:
this module never calls ``VectorEngine.push_pcm`` and never appends to its
running-median history — see ``LiveWatchSession.phone_baseline_rms_db``).

Tone -> level mapping (``TONE_LEVELS``)
---------------------------------------
``TurnTextTone`` scores are 0-100 per dimension. The escalation signal is
``max(frustration, defensiveness)`` — the two dimensions that read as
"turning on the other person"; sarcasm/sadness/warmth are real tone but not
escalation, and an over-broad trigger would teach the wearer to ignore the
buzz. Thresholds, highest-first, mirroring the ``+6/+10/+14 dB`` shape of
``YELLING_LEVELS`` (three rungs, each rung "clearly more" than the last):

* ``>= 85`` -> level 3: unmistakable — a classifier this sure is rarely wrong
  and the wearer is almost certainly already aware of it.
* ``>= 70`` -> level 2: strong — the "you're getting heated" tap.
* ``>= 55`` -> level 1: clearly present, a single soft cue. 55 (not 50) so a
  classifier's coin-flip midpoint never nudges; on-device tone models
  hover around 50 on neutral speech.

A ``ToneFlagEvent`` (the server's own audio/text tone verdict, when Track
3-server chooses to relay one) is read the same way from its ``scores``
dict — same 0-100 scale, same keys — and is only trusted at
``confidence >= TONE_FLAG_MIN_CONFIDENCE``. The two tone sources are combined
as a max too; whichever is louder about it wins.

Golden vectors: ``server/tests/fixtures/policy_vectors/tone_escalation.json``
(driver ``server/tests/watch/test_tone_escalation_vectors.py``) pin every
rung above and the max-combination rule. Edit the constants and the JSON
together, never one without the other.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from models.audio import ToneFlagEvent, TurnLocalEvent, TurnTextTone
from watch.models import VectorEvent, VectorName
from watch.vectors import (
    RUNNING_STAT_WINDOW,
    SILENCE_FLOOR_DBFS,
    YELLING_LEVELS,
    VectorEngine,
    level_for,
    running_median,
)

logger = logging.getLogger(__name__)

# (tone score 0-100, level), checked highest-first — see module docstring.
TONE_LEVELS: tuple[tuple[float, int], ...] = ((85.0, 3), (70.0, 2), (55.0, 1))
# Which TurnTextTone / ToneFlagEvent.scores dimensions count as escalation.
TONE_ESCALATION_KEYS: tuple[str, ...] = ("frustration", "defensiveness")
# The VectorName the tone lane rides on. `aggressive_tone` is the watch's own
# "hostile delivery" vector (F0-over-baseline in VectorEngine), lives on
# channel A by default (watch/models.py's set_default_channel), and is in
# every account's DEFAULT_VECTOR_NAMES subscription — so a tone nudge reaches
# a wearer who never touched their vector settings. A new VectorName literal
# would have needed a wire-protocol bump on the watch (WireModels.kt) for no
# gain: to the wrist, "your tone is aggressive" is the same message either way.
TONE_VECTOR: VectorName = "aggressive_tone"
# Below this a ToneFlagEvent is an observation, not a verdict — don't buzz.
TONE_FLAG_MIN_CONFIDENCE = 0.5

Emit = Callable[[list[VectorEvent], float], Awaitable[None]]


# ----------------------------------------------------------------- pure --

def tone_level(text_tone: TurnTextTone | None, tone_flag: ToneFlagEvent | None = None) -> int:
    """0-3 escalation level for a turn's tone, from the phone's text-tone
    scores and/or a server ToneFlagEvent — max of the two sources, each
    read as max(frustration, defensiveness) against ``TONE_LEVELS``.

    Missing scores are "couldn't measure", never 0: a turn with no tone data
    contributes level 0 because there is no evidence, not because it's calm.
    """
    score = 0.0
    if text_tone is not None:
        for key in TONE_ESCALATION_KEYS:
            value = getattr(text_tone, key, None)
            if value is not None:
                score = max(score, float(value))
    if tone_flag is not None and tone_flag.confidence >= TONE_FLAG_MIN_CONFIDENCE:
        for key in TONE_ESCALATION_KEYS:
            value = tone_flag.scores.get(key)
            if value is not None:
                score = max(score, float(value))
    return level_for(score, TONE_LEVELS)


def loudness_level(rms_dbfs: float | None, baseline_rms_db: float | None) -> tuple[int, float | None]:
    """(yelling level, dB over baseline) for a phone-measured turn loudness,
    using the watch's own ``YELLING_LEVELS`` ladder and silence floor.

    Returns ``(0, None)`` when it can't measure: no loudness reported, no
    baseline to compare against, or a turn at/below the silence floor (the
    same ``SILENCE_FLOOR_DBFS`` rule ``VectorEngine.push_pcm`` applies —
    noise-floor audio never reads as loudness).
    """
    if rms_dbfs is None or baseline_rms_db is None:
        return 0, None
    if rms_dbfs <= SILENCE_FLOOR_DBFS:
        return 0, None
    over_db = rms_dbfs - baseline_rms_db
    return level_for(over_db, YELLING_LEVELS), over_db


def turn_local_to_vector_events(
    event: TurnLocalEvent,
    *,
    t: float,
    baseline_rms_db: float | None,
    tone_flag: ToneFlagEvent | None = None,
) -> list[VectorEvent]:
    """Pure conversion: one phone turn -> the VectorEvents to feed NudgePolicy.

    Order is ``yelling`` then ``aggressive_tone``, mirroring ``push_pcm``'s
    emission order so a persisted live session reads the same regardless of
    which mic produced the event. ``t`` is the caller's stream clock — the
    watch session's ``engine.t``, NOT the phone's ``start_time`` (the policy
    must run on ONE clock; see ws.py's ``hr`` handling for the same rule).

    Returns [] (not a level-0 event) when neither input clears its first
    rung — level-0 VectorEvents are never emitted anywhere in this codebase.
    """
    if event.is_self is not True:
        return []

    events: list[VectorEvent] = []
    rms = event.prosody.rms_dbfs if event.prosody is not None else None
    yelling, over_db = loudness_level(rms, baseline_rms_db)
    if yelling:
        events.append(VectorEvent(
            vector="yelling", level=yelling, t=t, value=over_db,
            detail=f"phone turn: {over_db:.1f} dB over baseline",
        ))

    tone = tone_level(event.text_tone, tone_flag)
    if tone:
        label = (event.text_tone.label if event.text_tone is not None and event.text_tone.label else None) \
            or (tone_flag.label if tone_flag is not None else None) or "hostile"
        events.append(VectorEvent(
            vector=TONE_VECTOR, level=tone, t=t, value=float(tone),
            detail=f"phone turn tone: {label} (level {tone})",
        ))
    return events


# -------------------------------------------------------------- registry --

@dataclass
class LiveWatchSession:
    """One OPEN watch WS live session, as far as the relay needs to know it.

    ``emit`` is ws.py's own per-connection ``emit(events, t)`` closure — the
    same function its PCM/HR paths call — so relayed events go through the
    identical bookkeeping (appended to the session's ``vector_events``,
    pushed as ``vector_event`` frames, run through THIS session's
    ``NudgePolicy``, nudges recorded + pushed). ``loop`` is the event loop
    that socket lives on: ``push_turn_local`` may be called from another
    loop/thread (the phone's pipeline), and a websocket must only be
    touched from its own loop.
    """

    account_id: str
    live_session_id: str
    engine: VectorEngine
    emit: Emit
    loop: asyncio.AbstractEventLoop
    # Phone-side loudness history for the no-enrollment fallback baseline.
    # Deliberately SEPARATE from engine._rms_db_history: the phone's mic
    # gain is not the watch's, so mixing the two would corrupt the watch's
    # own running median (and the dB path must stay byte-identical).
    _phone_rms_history: deque[float] = field(default_factory=lambda: deque(maxlen=RUNNING_STAT_WINDOW))

    def phone_baseline_rms_db(self) -> float | None:
        """Enrollment baseline when the account has one (the watch's own
        rule), else the running median of this session's PRIOR phone turns
        (the same live-session-relative fallback ``VectorEngine`` uses, on
        the phone's own history). None until there is anything to compare
        against — the first phone turn of an un-enrolled session can never
        read as yelling, exactly like the first watch window can't."""
        if self.engine.baseline is not None:
            return self.engine.baseline.rms_db
        return running_median(self._phone_rms_history)

    def observe_phone_rms(self, rms_dbfs: float | None) -> None:
        """Record a self turn's loudness for the fallback baseline — AFTER
        the baseline for that turn was read (push_pcm's read-then-append
        ordering), and never for silence-floor turns (push_pcm skips those
        too, so a long quiet stretch can't drag the median down)."""
        if rms_dbfs is None or rms_dbfs <= SILENCE_FLOOR_DBFS:
            return
        self._phone_rms_history.append(rms_dbfs)


_registry: dict[str, LiveWatchSession] = {}
_registry_lock = threading.Lock()

# Strong refs for tasks scheduled onto the socket's own loop — same
# asyncio-weak-ref gotcha ws.py's _background_tasks guards against.
_relay_tasks: set[asyncio.Future] = set()


def register_live_session(session: LiveWatchSession) -> None:
    """Called by ws.py once the socket is accepted and the engine/policy exist.
    Last writer wins per account: if a wearer somehow has two live sessions
    open, phone turns go to the newest one (the one they're actually in)."""
    with _registry_lock:
        _registry[session.account_id] = session
    logger.debug("watch relay: live session %s registered for %s", session.live_session_id, session.account_id)


def unregister_live_session(session: LiveWatchSession) -> None:
    """Called from ws.py's ``finally`` — only removes THIS session, so a
    newer session registered for the same account isn't clobbered by an
    older one's teardown."""
    with _registry_lock:
        if _registry.get(session.account_id) is session:
            del _registry[session.account_id]
    logger.debug("watch relay: live session %s unregistered for %s", session.live_session_id, session.account_id)


def live_session_for(uid: str) -> LiveWatchSession | None:
    with _registry_lock:
        return _registry.get(uid)


def _on_relay_done(fut: asyncio.Future) -> None:
    _relay_tasks.discard(fut)
    if fut.cancelled():
        return
    exc = fut.exception()
    if exc is not None:
        # A socket that died between our lookup and the send lands here —
        # ws.py's own finally will unregister it; nothing to retry.
        logger.warning("watch relay: emit failed (socket gone?)", exc_info=exc)


def push_vector_events(uid: str, events: list[VectorEvent], t: float) -> bool:
    """Feed already-computed vector events (e.g. call-mode ``interrupting``
    from server/calls.py) into ``uid``'s live watch session — the wrist
    runs its own NudgePolicy over them exactly like phone turn_local
    vectors. Returns False (no-op) when no watch is live for the account."""
    if not events:
        return False
    session = live_session_for(uid)
    if session is None:
        return False
    _schedule(session, events, t)
    return True


def _schedule(session: LiveWatchSession, events: list[VectorEvent], t: float) -> None:
    """Run ``session.emit`` on the socket's loop from wherever we're called."""
    if session.loop.is_closed():
        logger.debug("watch relay: loop for %s already closed; dropping %d event(s)", session.account_id, len(events))
        return
    coro = session.emit(events, t)
    try:
        current = asyncio.get_running_loop()
    except RuntimeError:
        current = None
    if current is session.loop:
        fut: asyncio.Future = session.loop.create_task(coro)
    else:
        fut = asyncio.run_coroutine_threadsafe(coro, session.loop)
    _relay_tasks.add(fut)
    fut.add_done_callback(_on_relay_done)


# ------------------------------------------------------------- entrypoint --

def push_turn_local(uid: str, event: TurnLocalEvent, *, tone_flag: ToneFlagEvent | None = None) -> None:
    """Relay one phone-finalized turn to ``uid``'s live watch session.

    See module docstring for the contract. Never raises for the ordinary
    "nothing to do" cases (not the wearer's turn, no live watch, nothing
    over threshold) — Track 3-server calls this inline on its hot path.
    """
    if event.is_self is not True:
        logger.debug("watch relay: ignoring non-self turn (is_self=%r) for %s", event.is_self, uid)
        return

    session = live_session_for(uid)
    if session is None:
        logger.debug("watch relay: no live watch session for %s; turn dropped", uid)
        return

    # ONE clock: the watch session's stream clock, not the phone's turn
    # timestamps (different device, different epoch, and NudgePolicy's
    # cooldown hysteresis must see monotonic time from a single source).
    t = session.engine.t
    rms = event.prosody.rms_dbfs if event.prosody is not None else None
    baseline = session.phone_baseline_rms_db()
    events = turn_local_to_vector_events(event, t=t, baseline_rms_db=baseline, tone_flag=tone_flag)
    session.observe_phone_rms(rms)

    if not events:
        # No policy tick for an empty relay: the watch's own 1 s windows
        # already drive cooldown de-escalation, and ticking here would add
        # a second, phone-cadenced clock to a path that must stay
        # byte-identical to the shipped watch behaviour.
        logger.debug("watch relay: self turn for %s under every threshold; no events", uid)
        return

    logger.info(
        "watch relay: %s -> live session %s: %s",
        uid, session.live_session_id, ", ".join(f"{e.vector}={e.level}" for e in events),
    )
    _schedule(session, events, t)
