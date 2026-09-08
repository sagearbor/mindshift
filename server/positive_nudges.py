"""Positive nudges — the four codes that tell the user what they did WELL.

Every nudge that shipped before these was a complaint: too loud, too hot, you
cut in, you're hogging. A coach that only ever says "stop that" is one a person
turns off. D / E / R / K (nudge vocabulary, owner-approved 2026-09-06) are the
other half:

===  ==  ============  ==================================================
📉   D   De-escalated  heat dropped a level within two turns of a spike
👂   E   Listened      you let them finish a long turn without cutting in
🤝   R   Repair        you validated or apologised and their tone softened
🧘   K   Calm streak   N minutes with no escalation — a SUMMARY badge, never live
===  ==  ============  ==================================================

Rules that keep praise from becoming its own nag: one positive cue per two
minutes across D/E/R together (:class:`~nudge_vocabulary.PositiveNudgeGate`); a
detection the cap drops is still REPORTED with ``delivered=False`` so a replay
can show what the user nearly felt; the cue is soft and unleveled; K never
buzzes at all.

Mirror contract: this is the server-side counterpart of
``apps/mobile/src/live/positiveNudges.ts``. The cross-language golden vectors
in ``server/tests/fixtures/policy_vectors/positive_nudges.json`` (driven by
``server/tests/test_positive_nudges_vectors.py`` and, on the phone,
``apps/mobile/__tests__/positiveNudges.test.ts``) are the executable form of
that contract.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal, Sequence

from nudge_vocabulary import POSITIVE_CAP_S, PositiveNudgeGate

PositiveCode = Literal["D", "E", "R"]

# --- constants: each one pinned by the fixture on both runtimes ---------------

#: A partner turn at least this long is one you had to actively let run. The
#: CANDOR median turn is a couple of seconds; 12 s is someone holding the floor,
#: which is exactly when the urge to cut in shows up.
LISTEN_MIN_TURN_S = 12.0

#: How many of the user's OWN turns a de-escalation may take. "Within two turns
#: of a spike" (owner, 2026-09-06): later than that and the calm is not a
#: recovery, it is a different part of the conversation. A turn whose heat could
#: not be measured still spends one of the two — silence is not evidence.
DEESCALATION_WINDOW_TURNS = 2

#: How far the other person's tone must fall (0..100 points) for "they softened"
#: to be a measurement rather than noise.
REPAIR_SOFTEN_DROP = 15

#: No escalation for this long earns the 🧘 badge.
CALM_STREAK_MIN_S = 300.0

#: The text-tone ladder the live policy uses (``aggressive_tone``); repeated
#: here rather than imported so this module stays free of the watch package.
_TONE_LEVELS: tuple[tuple[int, int], ...] = ((85, 3), (70, 2), (50, 1))

#: Repair language: apology and validation, the two moves that actually turn a
#: fight around. Matched as substrings of the lowercased, punctuation-stripped
#: turn — deliberately a small, literal list rather than a classifier, because a
#: false "nice repair!" after something that was not one is worse than saying
#: nothing. First-person on purpose ("sorry not sorry" is sarcasm; "you are
#: wrong" is the opposite of a repair). Apostrophe-less spellings are listed
#: because on-device STT drops them.
REPAIR_PHRASES: tuple[str, ...] = (
    "im sorry",
    "i am sorry",
    "i apologize",
    "i apologise",
    "my bad",
    "that was on me",
    "youre right",
    "you are right",
    "thats fair",
    "that is fair",
    "fair enough",
    "i hear you",
    "i understand",
    "i get that",
    "that makes sense",
    "i see what you mean",
    "i shouldnt have",
    "i should not have",
    "i didnt mean",
    "i did not mean",
    "let me try again",
)

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class PositiveTurn:
    """One finalized turn, reduced to what the positive detectors need.

    Built from the phone's ``LocalTurn`` and from the server's analysis rows —
    neither detector ever sees audio.
    """

    index: int
    start: float
    end: float
    #: Whose voice the loop believes this is — a cluster label or a person's
    #: name. :func:`coalesce_turns` groups on it: a turn ends when someone ELSE
    #: speaks, which is the definition code E depends on.
    speaker: str
    is_self: bool
    text: str
    #: The loudness rung this turn raised over the speaker's own baseline, or
    #: ``None`` when loudness could not be measured.
    loud_level: int | None
    #: AGGRESSION: ``max(frustration, defensiveness)`` 0..100, or ``None`` when
    #: the turn was not scored. This is the SELF side's heat — the same rung the
    #: live policy's ``aggressive_tone`` vector reads.
    tone_heat: float | None
    #: NEGATIVE AFFECT: ``max(frustration, defensiveness, sadness)``, or
    #: ``None``. What R measures the other person's softening on — and the
    #: reason it is a separate number: someone you have just shouted at usually
    #: goes HURT, not aggressive, so an aggression-only reading would score a
    #: real repair as no change at all (measured on the couple_escalation
    #: fixture, 2026-09-06: their aggression moved 5 points, their sadness 65).
    #: Heat is never read from this — being sad is not being heated.
    tone_negativity: float | None = None
    #: This self turn started inside the previous partner turn and kept going
    #: long enough to raise the alert code C.
    cut_in: bool = False


@dataclass(frozen=True)
class PositiveNudge:
    code: PositiveCode
    #: Session seconds the detection landed on.
    t: float
    #: The turn that earned it.
    turn_index: int
    #: ``False`` when the two-minute cap dropped it — detected, not felt.
    delivered: bool
    #: One line a report (or Developer mode) can show verbatim.
    detail: str


@dataclass(frozen=True)
class CalmStreak:
    """K is not an event: it is a number on the summary."""

    #: The longest run of session seconds with no alert nudge.
    longest_s: float
    #: Whether that run earns the 🧘 badge.
    badge: bool


@dataclass(frozen=True)
class PositiveResult:
    nudges: list[PositiveNudge]
    calm: CalmStreak


def _half_up(x: float) -> int:
    """Round like the phone's ``Math.round``, not like Python's ``round``.

    These numbers reach the user inside the ``detail`` line ("let a 13 s turn
    finish"), and Python's banker's rounding would print 12 where the phone
    prints 13 for the same 12.5 s turn — a cross-runtime drift the golden file
    cannot see, because it only asserts that a detail exists.
    """
    return math.floor(x + 0.5)


def normalize_for_repair(text: str) -> str:
    """Lowercase, drop everything that is not a letter/digit/space, collapse
    runs of spaces — so ``"I'm sorry."`` and ``"im  sorry"`` both match."""
    return _SPACES.sub(" ", _NON_ALNUM.sub("", text.lower())).strip()


def has_repair_language(text: str) -> bool:
    """True when a turn says one of the repair moves."""
    norm = normalize_for_repair(text)
    if not norm:
        return False
    return any(p in norm for p in REPAIR_PHRASES)


def _tone_level(tone_heat: float | None) -> int | None:
    if tone_heat is None:
        return None
    for threshold, level in _TONE_LEVELS:
        if tone_heat >= threshold:
            return level
    return 0


def self_heat_level(turn: PositiveTurn) -> int | None:
    """The alert level a self turn raised: the higher of its loudness rung and
    its text-tone rung (the same two ladders the live policy maxes). ``None``
    when neither could be measured — never 0-as-a-guess."""
    tone = _tone_level(turn.tone_heat)
    if turn.loud_level is None and tone is None:
        return None
    return max(turn.loud_level or 0, tone or 0)


def coalesce_turns(fragments: Sequence[PositiveTurn]) -> list[PositiveTurn]:
    """Merge consecutive same-speaker fragments into conversational turns.

    Not cosmetic. The live loop's "turn" is a VAD fragment — it cuts at a 300 ms
    pause — so someone holding the floor for sixteen seconds arrives as five
    three-second pieces, and code E ("you let them finish a LONG turn") would
    never fire on a real device, only on a tidy fixture. A turn ends when
    somebody ELSE starts talking; that is what this restores.

    Merged fields take the worst/widest reading, because a turn is as loud as
    its loudest moment and as hot as its hottest: ``loud_level``, ``tone_heat``
    and ``tone_negativity`` are maxima over the SCORED fragments (``None`` only
    when no fragment was scored — never 0-as-a-guess), ``cut_in`` is true if any
    fragment cut in, and ``index`` stays the FIRST fragment's so the turn is
    still addressable in a report.
    """

    def widen(a: float | None, b: float | None) -> float | None:
        if a is None:
            return b
        if b is None:
            return a
        return max(a, b)

    out: list[PositiveTurn] = []
    for f in fragments:
        if not out or out[-1].speaker != f.speaker:
            out.append(f)
            continue
        prev = out[-1]
        text = f"{prev.text} {f.text}" if prev.text and f.text else (prev.text or f.text)
        loud = widen(prev.loud_level, f.loud_level)
        out[-1] = PositiveTurn(
            index=prev.index,
            start=prev.start,
            end=max(prev.end, f.end),
            speaker=prev.speaker,
            is_self=prev.is_self,
            text=text,
            loud_level=None if loud is None else int(loud),
            tone_heat=widen(prev.tone_heat, f.tone_heat),
            tone_negativity=widen(prev.tone_negativity, f.tone_negativity),
            cut_in=prev.cut_in or f.cut_in,
        )
    return out


def _cut_into_by(turns: Sequence[PositiveTurn], partner: PositiveTurn) -> bool:
    """True when any SELF turn started strictly inside ``partner`` and was
    tagged a cut-in — you talked over them, not just past their last word."""
    return any(t.is_self and t.cut_in and partner.start < t.start < partner.end for t in turns)


def detect_positive_nudges(
    turns: Sequence[PositiveTurn],
    alert_nudge_times: Sequence[float] = (),
    session_end_s: float | None = None,
    cap_s: float = POSITIVE_CAP_S,
) -> PositiveResult:
    """Run the positive detectors over a whole conversation.

    ``alert_nudge_times`` are the session seconds at which ALERT nudges fired;
    K is the longest gap between them, bounded by ``session_end_s`` (defaulting
    to the last turn's end). Turns must be in time order.
    """
    gate = PositiveNudgeGate(cap_s)
    out: list[PositiveNudge] = []

    # D: the most recent self turn that raised heat, and how many self turns have
    # gone by since. Reset whenever heat is raised again at or above it — a spike
    # that keeps repeating is not a recovery in progress.
    spike_level: int | None = None
    self_turns_since_spike = 0

    # R: the other person's NEGATIVE AFFECT before and after the self turn that
    # tried to repair. `pending_repair` is the self turn we are waiting on the
    # partner to answer; `last_partner_negativity` is how they sounded BEFORE it.
    last_partner_negativity: float | None = None
    pending_repair: tuple[PositiveTurn, float] | None = None

    def emit(code: PositiveCode, t: float, turn_index: int, detail: str) -> None:
        out.append(PositiveNudge(code, t, turn_index, gate.admit(t), detail))

    for turn in turns:
        if turn.is_self:
            heat = self_heat_level(turn)

            # --- D: did this turn pull the heat back down? ---
            if spike_level is not None:
                self_turns_since_spike += 1
                if heat is not None and heat < spike_level:
                    plural = "" if self_turns_since_spike == 1 else "s"
                    emit(
                        "D",
                        turn.end,
                        turn.index,
                        f"heat {spike_level} -> {heat} within {self_turns_since_spike} turn{plural} of the spike",
                    )
                    spike_level = None
                    self_turns_since_spike = 0
                elif self_turns_since_spike >= DEESCALATION_WINDOW_TURNS:
                    # The window closed without a measured drop. If this turn is
                    # itself a spike it becomes the new one below.
                    spike_level = None
                    self_turns_since_spike = 0
            if heat is not None and heat >= 1 and (spike_level is None or heat >= spike_level):
                spike_level = heat
                self_turns_since_spike = 0

            # --- R: remember an attempt, to be judged by their next turn ---
            if has_repair_language(turn.text) and last_partner_negativity is not None:
                pending_repair = (turn, last_partner_negativity)
            continue

        # --- partner turn ---
        # E: a long turn you did not cut into. Judged at its END, because a turn
        # that has ended can no longer be interrupted.
        if turn.end - turn.start >= LISTEN_MIN_TURN_S and not _cut_into_by(turns, turn):
            emit(
                "E",
                turn.end,
                turn.index,
                f"let a {_half_up(turn.end - turn.start)} s turn finish with no cut-in",
            )

        # R: their answer. "Their tone softened NEXT turn" means the very next
        # partner turn — so the attempt is spent here either way, including when
        # that turn was never scored (an unmeasured turn is not evidence of
        # softening, and waiting for a later one would let a repair claim credit
        # for a mood change minutes afterwards).
        if pending_repair is not None:
            repair_turn, before = pending_repair
            if (
                turn.tone_negativity is not None
                and before - turn.tone_negativity >= REPAIR_SOFTEN_DROP
            ):
                emit(
                    "R",
                    turn.end,
                    repair_turn.index,
                    f"repair language, then their tone fell {_half_up(before - turn.tone_negativity)} points",
                )
            pending_repair = None
        if turn.tone_negativity is not None:
            last_partner_negativity = turn.tone_negativity

    if session_end_s is None:
        session_end_s = max((t.end for t in turns), default=0.0)
    return PositiveResult(nudges=out, calm=calm_streak(alert_nudge_times, session_end_s))


def calm_streak(alert_nudge_times: Sequence[float], session_end_s: float) -> CalmStreak:
    """🧘 K — the longest stretch of the session with no alert nudge.

    Counted from 0 to the first alert, between consecutive alerts, and from the
    last alert to ``session_end_s``, so a session that was calm throughout is
    ONE long streak rather than none.
    """
    marks = sorted(t for t in alert_nudge_times if t == t and abs(t) != float("inf"))
    longest = 0.0
    prev = 0.0
    for t in marks:
        longest = max(longest, t - prev)
        prev = t
    longest = max(0.0, max(longest, session_end_s - prev))
    return CalmStreak(longest_s=longest, badge=longest >= CALM_STREAK_MIN_S)
