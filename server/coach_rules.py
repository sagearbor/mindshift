"""The no-LLM coach: deterministic triggers only (``MINDSHIFT_COACH_RULES=1``,
the replay's ``--coach rules``).

Two jobs: (1) a replay can run fully offline at $0 — nothing here calls a
model; (2) it is the prototype of the fast tier (decisions that need no LLM,
tmp/recordings/fable_latency/REPORT.md §3). Every line it sends carries
``suggestion_source: "rules"`` on the wire so no report, score or person
mistakes it for model output.

Rules
-----
* **floor** — the wearer has held the floor for ``x_s`` (45 s) or more, at
  least ``gap_s`` (30 s) between fires. This is exp_FT's best LIVE cell
  (tmp/recordings/exp_FT/RESULTS.md: 7 fires / 6 helpful on DEV, CI spans 0
  — a promising-but-unproven rule, not a validated one). Same definition as
  exp_FT's ``floor_state`` in live mode: the wearer's turns merged across
  pauses <= 3 s, broken by another speaker's non-backchannel turn >= 2 s;
  a turn is usable 0.3 s after it closes. One difference: exp_FT re-checks
  every 0.5 s between turns; this runs at turn close only (the server sees
  nothing between turns), so a floor that crosses 45 s in the gap after a
  close fires at the next wearer turn instead.
* **lexical** — a hook: ``LexicalTrigger`` regexes over the turn text. None
  ship by default (their precision is unmeasured — REPORT.md §3 option E);
  load a list with ``MINDSHIFT_COACH_RULES_LEXICON=<json file>``:
  ``[{"name", "pattern", "line", "importance"?, "applies_to"?: other|self|any,
  "kind"?: response|nudge}]``.

Turns with no fire produce no suggestion event (silence is the default).
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

FLOOR_LINE = "You've had the floor a while. Check in with them."
FLOOR_IMPORTANCE = 90


def enabled() -> bool:
    return (os.getenv("MINDSHIFT_COACH_RULES") or "").strip() == "1"


@dataclass(frozen=True)
class RuleFire:
    rule: str
    line: str
    importance: int
    kind: str                    # "nudge" (wearer's own turn) | "response"
    floor_s: float | None = None


@dataclass(frozen=True)
class LexicalTrigger:
    name: str
    pattern: str
    line: str
    importance: int = FLOOR_IMPORTANCE
    applies_to: str = "other"    # other | self | any
    kind: str = "response"

    def matches(self, text: str, is_self: bool | None) -> bool:
        if self.applies_to == "other" and is_self is True:
            return False
        if self.applies_to == "self" and is_self is not True:
            return False
        return re.search(self.pattern, text or "", flags=re.IGNORECASE) is not None


@dataclass
class FloorRule:
    x_s: float = 45.0
    gap_s: float = 30.0
    pause_s: float = 3.0          # wearer silence longer than this breaks the floor
    break_s: float = 2.0          # another speaker's turn this long breaks it
    backchannel_s: float = 1.0    # another speaker's turn shorter than this is a backchannel
    lat_s: float = 0.3            # a closed turn is usable this long after its end
    _turns: list = field(default_factory=list)
    _last_fire: float = -1e9

    def observe(self, start: float, end: float, is_self: bool | None) -> RuleFire | None:
        wearer = is_self is True
        self._turns.append((float(start), float(end), wearer))
        if not wearer:
            return None
        t = float(end) + self.lat_s
        breakers = [e for s, e, w in self._turns
                    if not w and (e - s) >= self.backchannel_s and (e - s) >= self.break_s]
        cut = max(breakers) if breakers else -1e9
        merged: list[list[float]] = []
        for s, e in sorted((s, e) for s, e, w in self._turns if w and e > cut):
            if merged and s - merged[-1][1] <= self.pause_s:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        if not merged:
            return None
        floor = t - max(merged[-1][0], cut)
        if floor >= self.x_s and t - self._last_fire >= self.gap_s:
            self._last_fire = t
            return RuleFire("floor", FLOOR_LINE, FLOOR_IMPORTANCE, "nudge", floor_s=round(floor, 3))
        return None


class RulesCoach:
    """One per session. ``observe_turn`` is called for EVERY finalized turn
    (in order), and returns the line to send, or None."""

    def __init__(self, floor: FloorRule | None = None, lexicon: tuple[LexicalTrigger, ...] = ()) -> None:
        self.floor = floor or FloorRule()
        self.lexicon = tuple(lexicon)
        self.stats = {"turns": 0, "fires": 0, "by_rule": {}}

    def observe_turn(self, text: str, start: float, end: float, *, is_self: bool | None) -> RuleFire | None:
        self.stats["turns"] += 1
        fire = self.floor.observe(start, end, is_self)
        if fire is None:
            for trig in self.lexicon:
                if trig.matches(text, is_self):
                    fire = RuleFire(f"lexical:{trig.name}", trig.line, int(trig.importance), trig.kind)
                    break
        _note(fire)
        if fire is not None:
            self.stats["fires"] += 1
            self.stats["by_rule"][fire.rule] = self.stats["by_rule"].get(fire.rule, 0) + 1
        return fire


def load_lexicon(path: str | Path) -> tuple[LexicalTrigger, ...]:
    rows = json.loads(Path(path).read_text())
    return tuple(LexicalTrigger(**{k: v for k, v in r.items()
                                   if k in ("name", "pattern", "line", "importance", "applies_to", "kind")})
                 for r in rows)


def from_env() -> RulesCoach:
    lex = (os.getenv("MINDSHIFT_COACH_RULES_LEXICON") or "").strip()
    return RulesCoach(lexicon=load_lexicon(lex) if lex else ())


# Process-wide tally (an in-process replay reads it after the session, the
# way it reads the LLM cache's stats). Bounded: three counters.
_STATS_LOCK = threading.Lock()
_STATS = {"turns": 0, "fires": 0, "by_rule": {}}


def _note(fire: RuleFire | None) -> None:
    with _STATS_LOCK:
        _STATS["turns"] += 1
        if fire is not None:
            _STATS["fires"] += 1
            _STATS["by_rule"][fire.rule] = _STATS["by_rule"].get(fire.rule, 0) + 1


def reset_stats() -> None:
    with _STATS_LOCK:
        _STATS.update({"turns": 0, "fires": 0, "by_rule": {}})


def stats_snapshot() -> dict:
    with _STATS_LOCK:
        return {"turns": _STATS["turns"], "fires": _STATS["fires"], "by_rule": dict(_STATS["by_rule"])}
