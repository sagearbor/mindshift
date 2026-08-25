"""Pleasantness scoreboard — the server twin of apps/mobile/src/live/pleasantness.ts.

PRD §6's per-turn score (0–100) from what a live turn already carries in
turns.json: ``text_tone`` (warmth, defensiveness, sarcasm, frustration,
label), ``prosody`` (loudness against the speaker's own running baseline,
speech rate) and the conversation's turn balance. The phone computes the
SAME numbers live; this module recomputes them from the stored turns so the
post-session view (SessionDetail / the therapist's session view) matches the
board the couple watched — both replay
``server/tests/fixtures/policy_vectors/pleasantness.json``, which is the
spec (its ``_schema`` spells out every rule).

Mapping (weights per PRD §6):

    warmth           30%  = text_tone.warmth
    constructiveness 25%  = 100 − text_tone.defensiveness
    calmness         20%  = (100 − frustration | NEUTRAL_CALM_PRIOR when only
                             prosody is measurable) − loudness penalty
                             (4 pts/dB over +3 dB above the speaker's own
                             baseline, max 30) − 10 when > 3.5 words/s
    respect          15%  = 100 − text_tone.sarcasm; a contempt/dismissive
                             label caps it at 20
    engagement       10%  = turn balance over the last 6 turns (50/50 → 100);
                             None with one voice in the window

Score = weighted mean over the dimensions actually measured (weights
renormalized), half-up rounded; None when none of the four content
dimensions is present. Missing inputs are honest ``None``s, never 0.

Pure — no I/O, no clock. House rule: a number that was never measured is
``None``.
"""

from __future__ import annotations

import math
from typing import Iterable

DIMENSIONS: tuple[str, ...] = (
    "warmth", "constructiveness", "calmness", "respect", "engagement",
)
WEIGHTS: dict[str, float] = {
    "warmth": 0.30,
    "constructiveness": 0.25,
    "calmness": 0.20,
    "respect": 0.15,
    "engagement": 0.10,
}
NEUTRAL_CALM_PRIOR = 70
LOUD_DB_FREE = 3.0
LOUD_PENALTY_PER_DB = 4.0
LOUD_PENALTY_MAX = 30
FAST_RATE_WPS = 3.5
FAST_PENALTY = 10
CONTEMPT_RESPECT_CAP = 20
CONTEMPT_LABELS: frozenset[str] = frozenset({
    "contempt", "contemptuous", "dismissive", "hostile", "mocking",
})
BALANCE_WINDOW = 6
CURRENT_WINDOW = 5
SERIES_LENGTH = 10
LEAD_MIN_MARGIN = 3


def round_half_up(x: float) -> int:
    """floor(x + 0.5) — the phone's Math.round; NOT Python's banker's round."""
    return int(math.floor(x + 0.5))


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _num(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    return cleaned or None


def engagement_from_window(window: list[str], speaker: str) -> int | None:
    """Turn-balance engagement over ``window`` (current turn included).
    None unless ≥ 2 turns from ≥ 2 speakers are in it."""
    if len(window) < 2 or len(set(window)) < 2:
        return None
    share = sum(1 for s in window if s == speaker) / len(window)
    return int(_clamp(round_half_up(100 - 200 * abs(share - 0.5))))


def score_turn(
    text_tone: dict | None,
    prosody: dict | None,
    *,
    baseline_rms: float | None,
    engagement: int | None,
) -> dict:
    """Score ONE turn → ``{"dims": {dim: int|None}, "score": int|None}``."""
    tone = text_tone if isinstance(text_tone, dict) else {}
    pros = prosody if isinstance(prosody, dict) else {}
    warmth_in = _num(tone.get("warmth"))
    defensiveness = _num(tone.get("defensiveness"))
    sarcasm = _num(tone.get("sarcasm"))
    frustration = _num(tone.get("frustration"))
    label = _label(tone.get("label"))
    rms = _num(pros.get("rms_dbfs"))
    rate = _num(pros.get("speech_rate"))

    warmth = None if warmth_in is None else int(_clamp(warmth_in))
    constructiveness = None if defensiveness is None else int(_clamp(100 - defensiveness))

    respect = None if sarcasm is None else int(_clamp(100 - sarcasm))
    if label is not None and label in CONTEMPT_LABELS:
        respect = CONTEMPT_RESPECT_CAP if respect is None else min(respect, CONTEMPT_RESPECT_CAP)

    loud_penalty = 0
    loud_measurable = False
    if rms is not None and baseline_rms is not None:
        loud_measurable = True
        over = rms - baseline_rms
        if over > LOUD_DB_FREE:
            loud_penalty = min(LOUD_PENALTY_MAX, round_half_up(LOUD_PENALTY_PER_DB * (over - LOUD_DB_FREE)))
    fast_measurable = rate is not None
    fast_penalty = FAST_PENALTY if (rate is not None and rate > FAST_RATE_WPS) else 0
    calmness: int | None = None
    if frustration is not None:
        calmness = int(_clamp(100 - frustration - loud_penalty - fast_penalty))
    elif loud_measurable or fast_measurable:
        calmness = int(_clamp(NEUTRAL_CALM_PRIOR - loud_penalty - fast_penalty))

    dims = {
        "warmth": warmth,
        "constructiveness": constructiveness,
        "calmness": calmness,
        "respect": respect,
        "engagement": engagement,
    }
    if all(dims[d] is None for d in ("warmth", "constructiveness", "calmness", "respect")):
        return {"dims": dims, "score": None}
    weighted = 0.0
    weight_sum = 0.0
    for d in DIMENSIONS:
        v = dims[d]
        if v is None:
            continue
        weighted += WEIGHTS[d] * v
        weight_sum += WEIGHTS[d]
    return {"dims": dims, "score": int(_clamp(round_half_up(weighted / weight_sum)))}


def person_score(speaker: str, scored: list[int]) -> dict:
    recent = scored[-CURRENT_WINDOW:]
    current = round_half_up(sum(recent) / len(recent)) if recent else None
    return {
        "speaker": speaker,
        "current": current,
        "series": list(scored[-SERIES_LENGTH:]),
        "scored_turns": len(scored),
    }


def lead_of(people: Iterable[dict]) -> dict | None:
    scored = sorted(
        (p for p in people if isinstance(p.get("current"), int)),
        key=lambda p: -p["current"],
    )
    if len(scored) < 2:
        return None
    margin = scored[0]["current"] - scored[1]["current"]
    if margin >= LEAD_MIN_MARGIN:
        return {"speaker": scored[0]["speaker"], "margin": margin}
    return None


class PleasantnessTracker:
    """Session-scoped: feed turns in order, read the board any time. Keyed
    by the raw speaker label (the phone keys the same way)."""

    def __init__(self) -> None:
        self._rms: dict[str, tuple[float, int]] = {}
        self._window: list[str] = []
        self._order: list[str] = []
        self._scores: dict[str, list[int]] = {}

    def baseline_for(self, speaker: str) -> float | None:
        s = self._rms.get(speaker)
        return (s[0] / s[1]) if s and s[1] > 0 else None

    def observe(self, speaker: str, text_tone: dict | None, prosody: dict | None) -> dict:
        self._window.append(speaker)
        if len(self._window) > BALANCE_WINDOW:
            self._window.pop(0)
        engagement = engagement_from_window(self._window, speaker)
        result = score_turn(
            text_tone, prosody,
            baseline_rms=self.baseline_for(speaker), engagement=engagement,
        )
        rms = _num((prosody or {}).get("rms_dbfs")) if isinstance(prosody, dict) else None
        if rms is not None:
            total, n = self._rms.get(speaker, (0.0, 0))
            self._rms[speaker] = (total + rms, n + 1)
        if speaker not in self._order:
            self._order.append(speaker)
        scores = self._scores.setdefault(speaker, [])
        if result["score"] is not None:
            scores.append(result["score"])
        return result

    def board(self) -> dict:
        people = [person_score(sp, self._scores.get(sp, [])) for sp in self._order]
        return {"people": people, "lead": lead_of(people)}


def score_session(turns: list[dict]) -> dict:
    """Score a stored live session's turns in one pass →
    ``{"per_turn": [{dims, score}…] (index-aligned), "people": […], "lead": …}``.
    A turn without a string speaker is skipped for balance/baselines but
    still yields an unscored row so the list stays index-aligned."""
    tracker = PleasantnessTracker()
    per_turn: list[dict] = []
    for turn in turns:
        speaker = turn.get("speaker") if isinstance(turn, dict) else None
        if not isinstance(speaker, str):
            per_turn.append({"dims": {d: None for d in DIMENSIONS}, "score": None})
            continue
        tone = turn.get("text_tone")
        pros = turn.get("prosody")
        per_turn.append(tracker.observe(
            speaker,
            tone if isinstance(tone, dict) else None,
            pros if isinstance(pros, dict) else None,
        ))
    board = tracker.board()
    return {"per_turn": per_turn, "people": board["people"], "lead": board["lead"]}
