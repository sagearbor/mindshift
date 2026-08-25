"""Session resume (2026-08-25) — a live session survives a network drop.

Why
---
A real phone call drops WiFi/cellular for ten to thirty seconds. Before this
module every such drop cost the session three things, because ``/ws/session``
state is strictly per-CONNECTION:

1. **Turns.** The phone's ``turn_local`` frames produced while the socket was
   down went nowhere (they were only kept for the end-of-session POST), so the
   cloud coach — and, in a call, everybody else's screen — silently missed
   whole turns of the conversation.
2. **The clock.** The server's session timeline (``PcmRingBuffer``: t = 0 is
   the first audio frame of the CONNECTION, advancing by audio received) and
   ``calls.Participant.offset_s`` both restarted on the replacement socket,
   while the phone's capture clock kept counting. Every later turn_local then
   addressed audio the ring buffer did not hold, and in a call the resumed
   member's turns landed before turns already merged.
3. **Context.** Nothing replayed the merged call transcript the reconnecting
   phone missed, so its screen had a hole in the conversation.

The handshake
-------------
On reconnect the client sends, right after the ``config`` that re-authenticates
it and BEFORE any audio::

    {"type": "resume", "session_id": …, "since_seq": N, "last_local_time": t}

* ``since_seq`` — the highest merged-call ``seq`` this client has rendered.
  The server replays the call turns after it (bounded, see
  ``RESUME_REPLAY_MAX``); 0 means "I have nothing".
* ``last_local_time`` — seconds on the PHONE's capture clock of the NEXT audio
  byte it will send (captured-so-far minus what is still queued on the phone).
  The server re-anchors its session timeline to that instead of restarting at
  0, so the phone's turn_local times and the server's ring buffer keep meaning
  the same thing. The audio lost during the drop is a HOLE in the ring buffer,
  not a shift: a slice over it comes back short/empty and enrichment degrades
  to "no verdict", which is the honest outcome.

The phone buffers the ``turn_local``s it could not send (bounded, oldest
dropped) and flushes them in order after the resume frame. Every turn carries
a client-generated ``turn_uid``; the server remembers the ones it has already
processed ACROSS connections (that is what this module holds) and ignores a
repeat, so a turn that was half-sent when the socket died is never coached,
enriched, merged or delivered twice.

Scope — PROCESS-LOCAL, exactly like ``calls.registry`` and the watch relay:
the reconnecting socket must land on the same process, which is why production
runs Cloud Run with ``--max-instances 1``. A multi-instance deployment would
need this state in a shared store (a flagged, later decision). Losing it is
not fatal — a resume against an unknown session_id is answered honestly with
``resumed: false`` and the session simply starts fresh.

What this deliberately does NOT do: keep the session's transcript. A SOLO
live session's words are never stored server-side (see
``audio_pipeline._remember_utterance``), so there is nothing to replay to a
solo client — and the phone is the one holding that transcript anyway. Only
turn ids (opaque, client-generated) and two numbers live here.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field

from models.audio import TURN_UID_MAX

logger = logging.getLogger(__name__)

# How long a dropped session's resume state is kept before it is swept. Long
# enough for a lift ride or a tunnel; short enough that a process holding
# thousands of abandoned sessions cleans itself out. A session whose socket is
# LIVE is touched on every turn, so this is a bound on idle time, not on
# session length.
RESUME_TTL_S = float(os.getenv("MINDSHIFT_RESUME_TTL_S", "600"))
# Bound on sessions held in this process (abuse guard). Expired entries are
# swept before the oldest live one is evicted.
RESUME_MAX_SESSIONS = int(os.getenv("MINDSHIFT_RESUME_MAX_SESSIONS", "500"))
# Bound on remembered turn ids per session (oldest dropped). A duplicate can
# only arrive from the client's own pending queue, which is itself bounded and
# flushed at once — so a few hundred is far more than the window in which a
# repeat is possible, and the memory is ~64 bytes each.
RESUME_SEEN_UIDS_MAX = int(os.getenv("MINDSHIFT_RESUME_SEEN_UIDS_MAX", "400"))
# Merged call turns replayed on ONE resume. A 30 s drop in a real conversation
# is a handful of turns; the cap stops a client that resumes with
# ``since_seq: 0`` after an hour from being handed the whole call at once.
# When more were missed the OLDEST are dropped (the recent context is what
# matters on screen) and the count is reported in ``resume_replay``.
RESUME_REPLAY_MAX = int(os.getenv("MINDSHIFT_RESUME_REPLAY_MAX", "50"))

# A turn_uid is opaque to the server — it is only ever compared for equality
# and logged. Bound its shape anyway: it is client-supplied and ends up in log
# lines, so no control characters and no unbounded length. The LENGTH bound is
# models.audio.TURN_UID_MAX (the wire model enforces it); this pins the shape.
_TURN_UID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,%d}$" % TURN_UID_MAX)

# Bound on ``last_local_time``: a session's capture clock in seconds. 24 h is
# past any plausible session and keeps an absurd value out of the arithmetic
# that addresses the ring buffer.
LAST_LOCAL_TIME_MAX_S = 24 * 60 * 60.0


def clean_turn_uid(raw: object) -> str | None:
    """The validated ``turn_uid`` of a turn_local, or None when absent/unusable.

    None is not an error: a client that predates this field simply gets no
    cross-connection de-duplication (its turns are processed exactly as
    before), which is the same behaviour the pipeline had all along.
    """
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value or not _TURN_UID_RE.match(value):
        return None
    return value


def clean_since_seq(raw: object) -> int:
    """``since_seq`` as a non-negative int; anything else means "nothing yet"."""
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 0
    return raw if raw > 0 else 0


def clean_local_time(raw: object) -> float | None:
    """``last_local_time`` in seconds, or None when it cannot be trusted.

    None means "do not re-anchor the clock" — the session keeps the timeline
    it has, which is exactly the pre-resume behaviour.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return None
    if value < 0 or value > LAST_LOCAL_TIME_MAX_S:
        return None
    return value


@dataclass
class ResumeState:
    """What survives a dropped socket for one session id: the turn ids already
    processed (so a re-sent turn is ignored) and the last clock anchor."""

    session_id: str
    uid: str
    # Insertion-ordered id set; the oldest is dropped past RESUME_SEEN_UIDS_MAX.
    _seen: dict[str, None] = field(default_factory=dict, repr=False)
    # The most recent ``last_local_time`` a resume re-anchored to (diagnostics).
    last_local_time: float | None = None
    # Counters for the session's own honesty: how many turns were remembered,
    # how many repeats were ignored, how many times this session resumed.
    turns_seen: int = 0
    duplicates: int = 0
    resumes: int = 0
    # Sockets that have attached to this session. > 1 is the whole definition
    # of "the server remembers you": a second connection for the same session
    # id and account is a reconnect, and only then can a resume find anything.
    connections: int = 0
    touched_at: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.touched_at = time.monotonic()

    def expired(self, now: float | None = None, ttl: float = RESUME_TTL_S) -> bool:
        return (now if now is not None else time.monotonic()) - self.touched_at > ttl

    @property
    def known_turns(self) -> int:
        return len(self._seen)

    def seen(self, turn_uid: str) -> bool:
        return turn_uid in self._seen

    def remember(self, turn_uid: str | None) -> bool:
        """Record ``turn_uid`` as processed. Returns True when it is NEW (the
        caller should handle the turn) and False when it is a repeat (ignore
        it). An absent id is always "new" — a client that sends none gets the
        pre-resume behaviour, never a silently swallowed turn."""
        self.touch()
        if turn_uid is None:
            self.turns_seen += 1
            return True
        if turn_uid in self._seen:
            self.duplicates += 1
            return False
        self._seen[turn_uid] = None
        self.turns_seen += 1
        while len(self._seen) > RESUME_SEEN_UIDS_MAX:
            self._seen.pop(next(iter(self._seen)))
        return True


class ResumeRegistry:
    """Process-local ``session_id → ResumeState``, bounded and TTL-swept."""

    def __init__(self) -> None:
        self._states: dict[str, ResumeState] = {}

    def __len__(self) -> int:
        return len(self._states)

    def reset(self) -> None:
        self._states.clear()

    def sweep(self, now: float | None = None) -> int:
        now = now if now is not None else time.monotonic()
        stale = [sid for sid, st in self._states.items() if st.expired(now)]
        for sid in stale:
            del self._states[sid]
        return len(stale)

    def get(self, session_id: str, uid: str) -> ResumeState | None:
        """The live state for this session AND owner, or None.

        The uid check is the whole security story of resume: a session id is
        client-chosen, so state is only ever handed back to the account that
        created it. A mismatch is treated as "no state" (and logged), never as
        an error the caller has to handle.
        """
        state = self._states.get(session_id)
        if state is None:
            return None
        if state.expired():
            del self._states[session_id]
            return None
        if state.uid != uid:
            logger.warning(
                "Resume state for session %s belongs to another account — ignoring",
                session_id,
            )
            return None
        return state

    def attach(self, session_id: str, uid: str) -> tuple[ResumeState, bool]:
        """The state one CONNECTION works against, and whether the server
        already had it before this connection opened.

        Every session gets a state whether or not it ever reconnects: that is
        what makes the FIRST connection's turn ids known to the second. The
        flag is what a ``resume`` answers with — a client whose state is gone
        (a restart, a TTL sweep, a different process) is told so.

        A session id is client-chosen, so two ACCOUNTS can pick the same one.
        The second gets an isolated, unregistered state: it can neither read
        nor clobber the first's.
        """
        existing = self._states.get(session_id)
        if existing is not None and not existing.expired():
            if existing.uid != uid:
                logger.warning(
                    "Resume state for session %s belongs to another account — "
                    "this connection gets an isolated one",
                    session_id,
                )
                return ResumeState(session_id=session_id, uid=uid), False
            existing.touch()
            existing.connections += 1
            return existing, existing.connections > 1
        self.sweep()
        if len(self._states) >= RESUME_MAX_SESSIONS:
            # Evict the least recently touched rather than refuse: resume is a
            # best-effort convenience, and refusing would break a live session.
            oldest = min(self._states.values(), key=lambda s: s.touched_at)
            del self._states[oldest.session_id]
            logger.info(
                "Resume registry at %d sessions — evicted %s",
                RESUME_MAX_SESSIONS, oldest.session_id,
            )
        state = ResumeState(session_id=session_id, uid=uid, connections=1)
        self._states[session_id] = state
        return state, False

    def acquire(self, session_id: str, uid: str) -> ResumeState:
        """``attach`` without the "did it exist" flag."""
        return self.attach(session_id, uid)[0]

    def forget(self, session_id: str) -> bool:
        """Drop a session's state — called on a GRACEFUL stop, where there is
        nothing left to resume."""
        return self._states.pop(session_id, None) is not None


registry = ResumeRegistry()
