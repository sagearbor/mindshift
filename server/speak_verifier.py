"""Speak verifier — a second, tiny LLM check on a coach line that already
passed the speak gate (coach_gate.SpeakGate): "should this line be SAID in
the wearer's ear right now?"

Why: the speak gate cut DEV spoken lines from 290 to 23, but a calibrated
Sonnet judge (tmp/recordings/judge/RUBRIC.md v2) still said 20 of those 23
should have stayed silent — the coach's own importance score sits near 72
whatever the moment, so it cannot tell "you're spiraling" after a
backchannel from "let him respond" after the other side asked to answer.
The verifier condenses the rubric's should-stay-silent rules into one
strict yes/no prompt over the last few turns, their roles, and the
candidate line.

Contract:

* Behind ``MINDSHIFT_SPEAK_VERIFIER`` (default OFF). Read at CALL time.
* Runs only for a line the gate would voice, so the cost is a few calls an
  hour (Haiku, ~600 input tokens, <= 60 output).
* Fails CLOSED: any error, unparseable answer, or a call slower than
  ``MINDSHIFT_SPEAK_VERIFIER_TIMEOUT_S`` (default 1.2 s) means "don't
  speak". The line is still SENT (speak=false, shown dimmed) — the
  verifier only ever silences, never adds or rewrites a line.
* ``MINDSHIFT_SPEAK_VERIFIER_PROMPT`` picks a prompt variant (see
  ``VARIANTS``); the default is the one that agreed best with the Sonnet
  reference labels on DEV (scripts/recreplay/verifier_eval.py).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass

DEFAULT_VARIANT = "v4"
MAX_TOKENS = 120
CONTEXT_TURNS = 6


def enabled() -> bool:
    return os.getenv("MINDSHIFT_SPEAK_VERIFIER", "0").strip() not in ("0", "false", "off", "")


def timeout_s() -> float:
    try:
        return float(os.getenv("MINDSHIFT_SPEAK_VERIFIER_TIMEOUT_S", "") or 1.2)
    except ValueError:
        return 1.2


def variant() -> str:
    v = os.getenv("MINDSHIFT_SPEAK_VERIFIER_PROMPT", "").strip() or DEFAULT_VARIANT
    return v if v in VARIANTS else DEFAULT_VARIANT


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_ROLE_INTRO = (
    "A private coach speaks into ONE person's earpiece: the WEARER. It may "
    "speak only rarely; a line said at the wrong moment is worse than "
    "silence. You decide whether ONE candidate line should be SPOKEN right "
    "now. In ordinary calm conversation the answer is almost always no. "
    "Turns may be in any language (often Greek); read them in that language."
)

_RULES = """Say NO when ANY of these holds:
1. The last turn is a backchannel ("yeah", "mm-hm", "right"), laughter, or a fragment of 2 words or fewer.
2. Nothing real just happened: small talk, banter, an ordinary calm exchange, a monologue, a calm explanation. Disagreement stated calmly is NOT a moment.
3. The line is generic: it would fit after almost any turn (e.g. "Pause. What's your main point?", "Take a breath", "Ask what they enjoy").
4. The line claims something about the wearer that the turns do not show ("you're rushing / repeating / spiraling / angry", "again" with no earlier instance).
5. "Let them finish / let them respond / don't interrupt" when the turns show no real overlap or cut-off and nobody asked for the floor.
6. Its "you" is really the other person (it answers the other person's turn as if the wearer said it), or it tells the wearer what someone ELSE should do.
7. The wearer is not involved and nothing calls for the wearer to act.

Say YES only when a REAL moment just happened in the last turns — someone was talked over or asked for the floor ("let me finish", "can I answer"), heat is rising, a dismissal, a direct question to the wearer left hanging, or a repair worth reinforcing — AND the line addresses THAT moment with a short, kind, concrete action for the wearer. When unsure, say no."""

_V1 = f"""{_ROLE_INTRO}

{_RULES}

Return ONLY JSON: {{"speak": true or false, "reason": "<= 12 words"}}"""

_V2 = f"""{_ROLE_INTRO}

{_RULES}

First quote (8 words or fewer, original language) the words from the turns (not the line) that show the moment, or "" if there is none. A YES needs a non-empty quote.
Return ONLY JSON: {{"evidence": "<short quote or empty>", "speak": true or false, "reason": "<= 12 words"}}"""

_V3 = f"""{_ROLE_INTRO}

Classify the last few turns and the line.
- "moment": one of "none", "talked_over", "floor_request", "heat", "dismissal", "hanging_question", "repair". Use "none" for small talk, banter, calm discussion, calm disagreement, monologue, backchannel or fragment.
- "fits": 0-3. 3 = the line addresses that exact moment with a concrete, grounded action for the wearer; 2 = fits the mood only; 1 = generic; 0 = misreads the moment, its "you" is the other person, or it claims wearer behaviour the turns don't show.
Be strict: when torn, take the lower value.
Return ONLY JSON: {{"moment": "...", "fits": 0, "reason": "<= 12 words"}}"""

# v4/v5: v1/v2 after reading their misses on the labelled DEV lines. Haiku
# said NO to "Let him respond fully" right after the other side asked "can I
# answer?" (it demanded a timestamp overlap), and YES to delivery polish
# ("slow down, finish your thought") on fragmented-but-calm talk.
_RULES_B = """Say NO when ANY of these holds:
1. The last turn is a backchannel ("yeah", "mm-hm", "right"), laughter, or a fragment of 2 words or fewer.
2. Nothing real just happened: small talk, banter, an ordinary calm exchange, a monologue, a calm explanation. Disagreement stated calmly is NOT a moment.
3. The line is generic: it would fit after almost any turn (e.g. "Pause. What's your main point?", "Take a breath", "Ask what they enjoy").
4. The line is delivery polish ("slow down", "finish your thought", "say it once, clearly", "collect your thoughts") — halting, repeated or fragmented speech is normal talk, not a moment.
5. The line claims something about the wearer that the turns do not show ("you're rushing / repeating / spiraling / angry", "again" with no earlier instance).
6. Its "you" is really the other person (it answers the other person's turn as if the wearer said it), or it tells the wearer what someone ELSE should do.
7. "Let them finish / respond / don't interrupt" when nobody was talked over and nobody asked for the floor. (A turn that merely looks cut off by transcription is not a cut-off.)

Say YES when a REAL moment just happened and the line answers exactly that moment with a short, concrete action for the wearer:
- the other person asks for the floor or says they are being talked over ("let me finish", "can I answer?", "μπορώ να απαντήσω;", "let me speak") — especially right after the wearer held the floor — and the line tells the wearer to let them speak. No timestamp overlap is needed: the request itself is the evidence;
- the wearer talks over the other person (overlap marked) and the line tells the wearer to stop and listen;
- clear heat (insults, accusations, raised stakes), a dismissal, a direct question to the wearer left hanging, or a repair worth reinforcing, and the line addresses it.
Otherwise, and whenever unsure, say no."""

_V4 = f"""{_ROLE_INTRO}

{_RULES_B}

Return ONLY JSON: {{"speak": true or false, "reason": "<= 12 words"}}"""

_V5 = f"""{_ROLE_INTRO}

{_RULES_B}

First quote (8 words or fewer, original language) the words from the turns (not the line) that show the moment, or "" if there is none. A YES needs a non-empty quote.
Return ONLY JSON: {{"evidence": "<short quote or empty>", "speak": true or false, "reason": "<= 12 words"}}"""

VARIANTS = {"v1": _V1, "v2": _V2, "v3": _V3, "v4": _V4, "v5": _V5}
_EVIDENCE_VARIANTS = {"v2", "v5"}
_MOMENTS = {"talked_over", "floor_request", "heat", "dismissal", "hanging_question", "repair"}


@dataclass(frozen=True)
class Turn:
    """One turn as the verifier sees it. ``role``: "wearer" | "other" |
    "unknown" (wearer not yet identified)."""

    role: str
    speaker: str
    start: float
    end: float
    text: str


def _role_name(t: Turn) -> str:
    if t.role == "wearer":
        return "WEARER"
    if t.role == "other":
        return f"OTHER ({t.speaker})"
    return f"UNIDENTIFIED ({t.speaker})"


def build_user(
    turns: list[Turn],
    line: str,
    *,
    kind: str = "response",
    setting: str | None = None,
    relationship: str | None = None,
    wearer_unknown: bool = False,
) -> str:
    """Deterministic user prompt (stable bytes = stable LLM cache keys)."""
    turns = [t for t in turns if (t.text or "").strip()][-CONTEXT_TURNS:]
    out: list[str] = []
    if setting:
        out.append(f"Setting: {setting.strip()[:200]}")
    if relationship:
        out.append(f"Relationship: {relationship.strip()[:60]}")
    if wearer_unknown:
        out.append("The wearer's voice is not identified yet: any speaker may be the wearer.")
    out.append("Last turns (oldest first; times in seconds):")
    prev: Turn | None = None
    for t in turns:
        mark = ""
        if prev is not None and t.speaker != prev.speaker:
            if t.start < prev.end - 0.2:
                mark = f" [overlaps the previous turn by {prev.end - t.start:.1f}s]"
            elif t.start - prev.end <= 0.1:
                mark = " [starts the instant the previous turn stops]"
        text = " ".join(t.text.split())
        if len(text) > 320:
            text = text[:150] + " … " + text[-150:]
        out.append(f"- {t.start:.1f}-{t.end:.1f} {_role_name(t)}{mark}: \"{text}\"")
        prev = t
    what = "a whispered nudge about the wearer's own delivery" if kind == "nudge" else \
        "a suggestion for what the wearer could do or say next"
    out.append(f"Candidate line ({what}): \"{line.strip()}\"")
    out.append("Speak this line now?")
    return "\n".join(out)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse(raw: str, which: str) -> tuple[bool, str]:
    """(speak, reason). Strict: only an explicit yes speaks."""
    m = _JSON_RE.search(raw or "")
    if not m:
        return False, "unparseable"
    try:
        data = json.loads(m.group(0))
    except (ValueError, TypeError):
        return False, "unparseable"
    if not isinstance(data, dict):
        return False, "unparseable"
    reason = str(data.get("reason") or "")[:120]
    if which == "v3":
        fits = data.get("fits")
        ok = (data.get("moment") in _MOMENTS and isinstance(fits, int)
              and not isinstance(fits, bool) and fits >= 3)
        return ok, reason
    speak = data.get("speak") is True
    if which in _EVIDENCE_VARIANTS and speak and not str(data.get("evidence") or "").strip():
        return False, "no evidence quoted"
    return speak, reason


# Bounded in-process record of recent verdicts (offline replays read it to
# measure the verifier; nothing is persisted, no transcript is kept beyond
# the candidate line itself).
RECENT: list[dict] = []
RECENT_MAX = 500


def note(v: "Verdict", *, line: str, t: float) -> None:
    RECENT.append({"t": t, "line": line, "speak": v.speak, "reason": v.reason,
                   "latency_ms": round(v.latency_ms, 1), "error": v.error})
    if len(RECENT) > RECENT_MAX:
        del RECENT[: RECENT_MAX // 2]


@dataclass(frozen=True)
class Verdict:
    speak: bool
    reason: str
    latency_ms: float
    error: str | None = None


def verify_sync(llm, user: str, which: str | None = None) -> Verdict:
    """Blocking call (offline evaluation). Errors fail closed."""
    which = which or variant()
    t0 = time.monotonic()
    try:
        raw = llm.complete(system=VARIANTS[which], user=user, temperature=0.0, max_tokens=MAX_TOKENS)
    except Exception as exc:  # noqa: BLE001 — fail closed
        return Verdict(False, "error", (time.monotonic() - t0) * 1000.0, type(exc).__name__)
    speak, reason = parse(raw, which)
    return Verdict(speak, reason, (time.monotonic() - t0) * 1000.0)


async def verify(llm, user: str, which: str | None = None, timeout: float | None = None) -> Verdict:
    """The live check: ``verify_sync`` in a thread under a hard deadline.
    A timeout fails closed (the thread finishes in the background; with the
    replay cache its answer is still recorded for the next run)."""
    which = which or variant()
    timeout = timeout_s() if timeout is None else timeout
    t0 = time.monotonic()
    try:
        return await asyncio.wait_for(asyncio.to_thread(verify_sync, llm, user, which), timeout)
    except asyncio.TimeoutError:
        return Verdict(False, "timeout", (time.monotonic() - t0) * 1000.0, "timeout")
