"""Guest (anonymous) accounts — the cost quota that bounds what they can spend.

WHY THIS EXISTS
---------------
"Continue as guest" (``signInAnonymously`` on the login screen) removes the
account wall in front of the core product: a Play reviewer, or anyone who
wants to try Live Coach before deciding, taps one button and is in. That is
worth a great deal for first-run friction and for review turnaround.

It also removes the only thing that used to make a session cost *somebody*
something. An anonymous uid is free to mint and free to mint again, and every
live session behind it can spend real money: Deepgram streaming STT and
Anthropic completions (see ``server/audio_pipeline.py``). So a guest account
gets a deliberately small allowance:

* ``MINDSHIFT_GUEST_MAX_SESSIONS_PER_DAY`` — default **3** live sessions per
  UTC day.
* ``MINDSHIFT_GUEST_MAX_SESSION_MIN`` — default **10** minutes per session.

A signed-up account (password, Google, Apple) is never touched by any of
this: :func:`admit_session` is only ever reached for an identity whose
``firebase.sign_in_provider`` is ``anonymous``.

WHAT THIS IS, HONESTLY
----------------------
This is a **cost brake, not a security control**, and the code should not be
read as claiming more:

1. The counter is **per process**. Cloud Run runs several instances, so a
   guest's real ceiling across a fleet of N instances is up to N × the limit.
2. A guest can **sign out and sign in again** and get a brand-new anonymous
   uid with a brand-new allowance, forever. No per-uid counter can fix that;
   only device attestation or a payment/identity signal could, and neither is
   worth adding for a free tier's try-before-you-sign-up path.

What it *does* buy is the thing that actually matters: no single guest session
can run unbounded (the per-session minute cap is hard, in-process, and applies
to the socket actually spending the money), and the boring failure mode — a
guest phone left on a desk streaming audio all afternoon, or a client bug that
reconnects in a loop — is capped rather than open-ended. Deliberate, repeated
abuse is bounded by the per-session cap and by the existing global
``MAX_WS_SESSIONS`` / rate limiters, not by this counter.

TODO (not done, deliberately): persist the counter (Firestore, keyed
``guest_usage/<uid>/<day>``) so it survives a restart and holds across
instances, and add a nightly sweep that deletes anonymous accounts — and the
data under them — that have been idle for N days. Neither is required for the
quota to bound a single session's spend, which is the part that has teeth; a
guest account and all of its data is already deletable today, in-app, via
``DELETE /me`` (the uid-scoped walk in ``server/account_deletion.py`` never
looks at how the uid signed in).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

# The one sentence the user sees whenever any of these limits bites. Kept here
# so the WebSocket close reason, the REST 429 detail and the app's copy are
# literally the same string.
GUEST_LIMIT_MESSAGE = "Guest limit reached — create a free account to continue."

# WebSocket close code for "you are over your quota". 4429 is the private-use
# analogue of HTTP 429, matching the existing 4401/4403 convention in
# server/audio_pipeline.py. The app does not retry on it (see the reconnect
# logic in apps/mobile/src/hooks/useAudioStream.ts) — a retry could not help.
GUEST_LIMIT_WS_CODE = 4429


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back on nonsense.

    A malformed or non-positive value falls back to ``default`` rather than
    disabling the quota: a typo in a deploy env must not silently uncap guest
    spend. (Set the value deliberately high if you want that.)
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


#: Live sessions one anonymous uid may START per UTC day.
GUEST_MAX_SESSIONS_PER_DAY = _positive_int_env(
    "MINDSHIFT_GUEST_MAX_SESSIONS_PER_DAY", 3
)

#: Wall-clock minutes one anonymous live session may run before the server
#: ends it. Enforced on the socket that is actually spending (audio_pipeline)
#: AND re-checked when the session's record is ingested, so a client that
#: ignores the close cannot buy an unbounded batch analysis with it either.
GUEST_MAX_SESSION_MIN = _positive_int_env("MINDSHIFT_GUEST_MAX_SESSION_MIN", 10)


def guest_max_session_seconds() -> float:
    """The per-session cap in seconds, read at call time.

    Read through a function (not a module constant computed at import) so a
    test — or a live reconfiguration — can ``monkeypatch.setattr`` the minute
    constant above and have every enforcement point agree immediately.
    """
    return GUEST_MAX_SESSION_MIN * 60.0


def _utc_day(now: datetime | None = None) -> str:
    """The UTC calendar day a usage bucket belongs to, as ``YYYY-MM-DD``.

    UTC, not the device's local day, on purpose: the server has no honest way
    to know a guest's timezone (there is no profile to read it from), and a
    client-supplied one would just be the knob to turn to get a fourth
    session.
    """
    moment = now or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d")


class GuestSessionCounter:
    """Per-uid, per-UTC-day count of guest live sessions admitted.

    Idempotent per ``session_id``: the same live session re-presenting itself
    (the WebSocket reconnect the phone does after a network blip, and then the
    ``POST /sessions/live`` that stores the very same session at the end)
    consumes exactly ONE of the day's allowance, not three. That matters —
    counting a reconnect as a new session would cut a real guest off in the
    middle of their first conversation, which is precisely the experience
    guest mode exists to provide.

    Buckets for elapsed days are dropped whenever a uid is touched, so the
    dict cannot grow without bound over a long-lived process.
    """

    def __init__(self) -> None:
        # uid -> (utc_day, {session_id, ...})
        self._seen: dict[str, tuple[str, set[str]]] = {}
        self._lock = asyncio.Lock()

    async def admit(
        self, uid: str, session_id: str, *, now: datetime | None = None
    ) -> bool:
        """Reserve one session of today's allowance for ``uid``.

        Returns ``True`` when the session may proceed — either because it is
        within the allowance or because this exact ``session_id`` was already
        admitted today. ``False`` means the guest is over the limit and the
        caller must refuse with :data:`GUEST_LIMIT_MESSAGE`.
        """
        day = _utc_day(now)
        async with self._lock:
            bucket_day, sessions = self._seen.get(uid, (day, set()))
            if bucket_day != day:
                bucket_day, sessions = day, set()  # a new UTC day: fresh allowance
            if session_id in sessions:
                self._seen[uid] = (bucket_day, sessions)
                return True  # same session, already counted — never double-charge
            if len(sessions) >= GUEST_MAX_SESSIONS_PER_DAY:
                self._seen[uid] = (bucket_day, sessions)
                return False
            sessions.add(session_id)
            self._seen[uid] = (bucket_day, sessions)
            return True

    def used(self, uid: str, *, now: datetime | None = None) -> int:
        """How many of today's sessions ``uid`` has already used (0 if none)."""
        bucket_day, sessions = self._seen.get(uid, (None, set()))
        return len(sessions) if bucket_day == _utc_day(now) else 0

    def reset(self) -> None:
        """Drop every counter. Used by tests to isolate days/limits."""
        self._seen.clear()


#: The process-wide counter. One instance so the WebSocket gate and the
#: ``POST /sessions/live`` gate agree about what a guest has already used.
SESSION_COUNTER = GuestSessionCounter()


async def admit_session(uid: str, session_id: str) -> bool:
    """Module-level shorthand for ``SESSION_COUNTER.admit`` (see it for the
    contract). Exists so callers do not have to reach through the singleton,
    and so a test can patch one name."""
    return await SESSION_COUNTER.admit(uid, session_id)


def session_span_seconds(started_at: str, ended_at: str) -> float | None:
    """Wall-clock seconds between two ISO-8601 stamps, or ``None`` if unusable.

    ``None`` (unparseable, or ends before it starts) is deliberately NOT
    treated as over-limit by callers: a client that reports a nonsense span is
    a bug to notice, not a user to cut off, and the socket-side cap has
    already bounded the real spend.
    """
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    span = (end - start).total_seconds()
    return span if span >= 0 else None


def session_over_time_cap(started_at: str, ended_at: str) -> bool:
    """True when a reported session span exceeds the guest per-session cap.

    A small grace of one wire-round-trip's worth of slack is NOT added on
    purpose: the socket closes a guest at the cap, so a span meaningfully past
    it means the client kept going on its own.
    """
    span = session_span_seconds(started_at, ended_at)
    if span is None:
        return False
    return span > guest_max_session_seconds()
