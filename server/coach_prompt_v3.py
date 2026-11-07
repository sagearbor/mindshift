"""Quiet coach, prompt v3: trigger-conditioned, one line type, SILENT allowed.

Why (docs plan 2026-11-05, Step 2): the quiet coach decides WHEN to call the
LLM (coach_trigger) but the prompt it sends is the old free-form one: three
cues on every call, no idea why it was called, no way to say "nothing
useful to add". v3 tells the model WHICH trigger fired (and the evidence),
asks for exactly one line of the one type that trigger calls for, makes
silence the default, and encodes the owner's taste rules. A regex
post-filter then drops lines that break the rules a model can still break
(raw speaker labels, names nobody said, praise, listen cues with no overlap).

Behind ``MINDSHIFT_COACH_PROMPT_V3`` (default OFF). With the flag off none of
this module is reached and every prompt is byte-identical to before.

Output contract (one JSON object)::

    {"line": "<=10 words, or empty>", "type": "heat|floor|question|read_other|silent",
     "importance": 0-100}

Line type per trigger and wearer identity (``line_type``):

=========  ==========================  ===========================================
trigger    who spoke                   type
=========  ==========================  ===========================================
heat       the wearer                  heat       one concrete de-escalating act
heat       someone else                read_other read their state + one move
floor      the wearer                  floor      hand the floor back on a topic
question   someone else                question   a scaffold for answering it
library    anyone                      question   (answer scaffold from the library)
=========  ==========================  ===========================================

A turn from an unidentified speaker is not sent to the LLM at all under v3:
the speak gate caps unknown-wearer lines below the speak threshold, so the
call could never be voiced.
"""

from __future__ import annotations

import os
import re

FLAG = "MINDSHIFT_COACH_PROMPT_V3"
LINE_TYPES = ("heat", "floor", "question", "read_other")
TRIGGER_PRECEDENCE = ("heat", "floor", "question", "library")
MAX_WORDS = 10
MIN_SPOKEN_IMPORTANCE = 80


def enabled() -> bool:
    """MINDSHIFT_COACH_PROMPT_V3 (default OFF; read at call time)."""
    return os.getenv(FLAG, "").strip().lower() in ("1", "true", "on", "yes")


def line_type(triggers: tuple[str, ...] | list[str], identity: str) -> str | None:
    """The one line type for this turn, or None when v3 makes no call
    (no trigger, or the wearer is unknown)."""
    if identity not in ("self", "other"):
        return None
    for t in TRIGGER_PRECEDENCE:
        if t not in triggers:
            continue
        if t == "heat":
            return "heat" if identity == "self" else "read_other"
        if t == "floor":
            if identity != "self":
                continue
            return "floor"
        return "question"          # question / library
    return None


_TYPE_RULES = {
    "heat": (
        "Line type HEAT. The wearer's own words just ran hot. Give ONE concrete "
        "act that lowers the temperature and names the thing in dispute, e.g. "
        "\"name the deadline, drop the 'always'\" or \"say what you need, not "
        "what they did\"."
    ),
    "floor": (
        "Line type FLOOR. The wearer has held the floor for a long time. Hand it "
        "back, naming the topic, e.g. \"ask what they think about the budget\" "
        "or \"stop and let them answer on the schedule\"."
    ),
    "question": (
        "Line type QUESTION. Someone asked the wearer a direct question. Give a "
        "scaffold for answering it that names what was asked, e.g. \"answer the "
        "price first, then the timeline\". Never supply facts the wearer has not "
        "stated."
    ),
    "read_other": (
        "Line type READ_OTHER. Someone else just spoke heatedly. Read their state "
        "from their words and give ONE move for the wearer, e.g. \"they sound "
        "guarded about the cost; ask what would help\"."
    ),
}

_COMMON_RULES = (
    "Give ONE whispered line of at most {max_words} words, of the type above and "
    "no other. SILENT is the default and is the right answer whenever the line "
    "would not clearly help right now: then return an empty line. Name the "
    "concrete thing from the transcript (a word, a number, the topic). Never "
    "praise. Never tell the wearer to \"engage\" or \"ask an open question\" "
    "in ordinary talk. {listen_rule}Use only names "
    "and facts that appear in the conversation; never say a speaker label "
    "(\"Speaker A\", \"S1\", \"the wearer\"). Do not speak as the wearer."
)

_LISTEN_RULE = (
    "Never give a listen, pause or \"let them finish\" cue unless the exchange "
    "shows the wearer talking over someone. "
)

_CONTRACT = (
    "Respond with ONLY a JSON object (no prose, no code fences) with keys in "
    "this order: \"line\" (the whispered line, or \"\" to stay silent), "
    "\"type\" (\"{type}\", or \"silent\" when the line is empty) and "
    "\"importance\" (an integer 0-100: 80-100 when you give a line, 0 when "
    "silent). No other keys."
)


def system_prompt(
    ltype: str, slider: int, role: str | None = None, *,
    relationship: str | None = None, session_context: str | None = None,
    library: bool = False,
) -> str:
    """System prompt for one line type (the preamble is the production one:
    core job, empathy stance, ground rules, wearer hints)."""
    from main import _coach_preamble  # lazy: main imports this module's caller

    preamble = _coach_preamble(slider, role, relationship, session_context, library)
    return "\n\n".join([
        preamble,
        _TYPE_RULES[ltype],
        _COMMON_RULES.format(max_words=MAX_WORDS, listen_rule="" if ltype == "floor" else _LISTEN_RULE),
        _CONTRACT.format(type=ltype),
    ])


def trigger_block(triggers: tuple[str, ...] | list[str], evidence: dict[str, str] | None = None) -> str:
    """The user-turn paragraph that names why the coach was called."""
    evidence = evidence or {}
    parts = []
    for t in TRIGGER_PRECEDENCE:
        if t not in triggers:
            continue
        ev = (evidence.get(t) or "").strip()
        label = {
            "heat": "heated words",
            "floor": "the wearer has held the floor",
            "question": "a direct question to the wearer",
            "library": "a phrase from the wearer's library",
        }[t]
        parts.append(f"{label}" + (f" ({ev})" if ev else ""))
    return "Why the coach was called: " + "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Parse + post-filter
# ---------------------------------------------------------------------------


def parse(data: dict) -> tuple[str, str, int]:
    """(line, type, importance) from the model's parsed JSON. Anything off
    contract is silence; a model that says ``type: silent`` is silence."""
    if not isinstance(data, dict):
        return "", "silent", 0
    line = data.get("line")
    ltype = data.get("type")
    if not isinstance(line, str) or not line.strip() or ltype == "silent":
        return "", "silent", 0
    if ltype not in LINE_TYPES:
        return "", "silent", 0
    raw = data.get("importance")
    imp = 100 if isinstance(raw, bool) or not isinstance(raw, (int, float)) else max(0, min(100, int(raw)))
    return line.strip(), ltype, imp


_PRAISE = re.compile(
    r"\b(great|good|nice|well)\s+(job|work|point|one|going|call|move)\b|\bnicely\b|\bwell done\b|"
    r"\bthat'?s?\s+(great|good|nice)\b|\bproud\b|\bkeep it up\b|\bhold that tone\b", re.I)
_ENGAGE = re.compile(r"\b(open[- ]ended|open question|engage|show (?:interest|curiosity)|be curious)\b", re.I)
_LISTEN = re.compile(
    r"\b(let (?:him|her|them|\w+) (?:finish|speak|talk)|pause|listen|stop (?:talking|interrupting)|"
    r"stop and (?:listen|let)|slow down|give (?:him|her|them) (?:space|room|a (?:turn|chance)))\b", re.I)
_WORD = re.compile(r"[A-Za-z0-9_'’À-￿]+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text or "")


def _foreign_names(line: str, context: str) -> list[str]:
    """Capitalised words in the line that never appear in the conversation
    text. A word that opens a sentence ("Name the deadline. Let them speak.")
    is not a name unless it is the whole line ("Pete"); "I" never is."""
    seen = {w.lower() for w in _words(context)}
    out = []
    single = len(_words(line)) == 1
    for m in _WORD.finditer(line):
        w = m.group(0)
        if not w[:1].isupper() or w == "I" or w.lower() in ("i'm", "i'll", "i've", "i'd"):
            continue
        before = line[: m.start()].rstrip()
        if (not before or before[-1] in ".!?;:\u2014\u2013-\"\u201c(") and not single:
            continue                      # sentence-initial imperative
        if w.lower() in seen:
            continue
        out.append(w)
    return out


def post_filter(
    line: str, ltype: str, context: str, *, wearer_overlapped: bool = False,
) -> tuple[str, str | None]:
    """(line, None) when the line may stand, ("", reason) when it is dropped.

    ``context`` is the conversation text the model saw (this turn + the
    recent exchange); a name absent from it was invented. ``floor`` lines are
    allowed to be listen-type cues (that is their job).
    """
    from coach_gate import has_label

    if not line or not line.strip():
        return "", "empty"
    if has_label(line) or re.search(r"\bthe wearer\b", line, re.I):
        return "", "label"
    if len(_words(line)) > MAX_WORDS + 2:
        return "", "too_long"
    if _PRAISE.search(line):
        return "", "praise"
    if _ENGAGE.search(line):
        return "", "engage"
    if ltype != "floor" and not wearer_overlapped and _LISTEN.search(line):
        return "", "listen_no_overlap"
    if _foreign_names(line, context):
        return "", "unseen_name"
    return line, None


def wearer_overlapped(history: dict | None, turn_start: float, turn_end: float) -> bool:
    """True when another voice's turn overlaps the coached (wearer) turn by
    more than 0.3 s - the only case in which a listen cue is justified."""
    if not history:
        return False
    for t in history.get("turns") or []:
        if t.get("who") == "You":
            continue
        s, e = float(t.get("start", t["t"])), float(t["t"])
        if min(e, turn_end) - max(s, turn_start) > 0.3:
            return True
    return False


def context_text(utterance_text: str, history: dict | None) -> str:
    parts = [utterance_text or ""]
    if history:
        parts += [t.get("text", "") for t in history.get("turns") or []]
    return "\n".join(parts)
