"""Quiet coach: the cheap PRE-LLM trigger gate for free-form cloud coaching.

Before this gate the cloud coach LLM ran on EVERY substantive turn (~1 call
per turn, hundreds an hour), and the speak gate (coach_gate.SpeakGate) then
threw most of the lines away. The owner's rule for an outside tester
(2026-10-29): a rare but accurate intervention beats a frequent one, so a
free-form cloud suggestion is only generated when something on the turn says
a coach could be useful. A turn with no trigger is never sent to the LLM (no
spend, no latency, nothing shown) - the transcript line has already gone out.

Triggers (each one a pure function of data the server already holds; no
model, microseconds):

``heat``      the turn's words score >= ``MINDSHIFT_TRIGGER_HEAT_MIN`` (1.5)
              on the curated counterproductive-conduct lexicon - the SAME
              patterns as the phone's heated-words buzz
              (apps/mobile/src/live/heatLexicon.ts; server/heat_lexicon.json
              is a byte-identical copy, pinned by a test). Measured offline
              10/10 on judge-heated units, 0 on judge-calm material.
``floor``     the wearer has held the floor >= ``MINDSHIFT_TRIGGER_FLOOR_S``
              (45 s) on this turn (the server's own continuous-floor clock,
              audio_pipeline._history_for_prompt), or the phone says its
              mid-turn floor cue fired during the turn (turn_local.triggers).
``question``  someone OTHER than the wearer (or an unknown voice) asked a
              direct question: the last sentence addresses "you", has >= 5
              words, ends in "?" or opens with an interrogative, and is not
              addressed to a name ("Where'd you go, Mel?"). Tag questions
              ("you know?", "right?") do not count.
``library``   the turn mentions a distinctive phrase from the user's selected
              Coach library items (LiveLibrary.phrase_hit: content-word
              bigrams of the selected items' text; no embedding call).

Flags (read at CALL time, so a test or a replay can flip them per run; all
reversible without a client release):

``MINDSHIFT_TRIGGER_GATE``     default ON; "0" = every substantive turn goes
                               to the LLM again (the pre-2026-10-29 behaviour).
``MINDSHIFT_TRIGGER_HEAT`` / ``_FLOOR`` / ``_QUESTION`` / ``_LIBRARY``
                               each trigger individually, default ON.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

TRIGGERS = ("heat", "floor", "question", "library")
LEXICON_PATH = Path(__file__).with_name("heat_lexicon.json")


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "off", "no", "")


def _num(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def enabled() -> bool:
    """MINDSHIFT_TRIGGER_GATE (default ON)."""
    return _flag("MINDSHIFT_TRIGGER_GATE")


# ---------------------------------------------------------------------------
# heat: the curated lexicon (shared with the phone)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _lexicon() -> tuple[tuple[re.Pattern, float, str], ...]:
    doc = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    return tuple(
        (re.compile(p["pattern"], re.IGNORECASE), float(p["weight"]), str(p["cat"]))
        for p in doc["patterns"]
    )


def heat_score(text: str) -> tuple[float, list[str]]:
    """(score, sorted unique categories) - same rule as scoreHeatText on the
    phone: the sum of the weights of every pattern that matches once."""
    if not text:
        return 0.0, []
    t = text.lower()
    score, cats = 0.0, set()
    for rx, w, cat in _lexicon():
        if rx.search(t):
            score += w
            cats.add(cat)
    return score, sorted(cats)


# ---------------------------------------------------------------------------
# question: a direct question to the wearer
# ---------------------------------------------------------------------------

_WH = r"(?:what|why|how|when|where|who|whom|whose|which)"
_AUX = (r"(?:do|does|did|are|is|was|were|can|could|would|will|should|have|has|had|"
        r"shall|may|might|must|won't|wont|don't|dont|didn't|didnt|aren't|arent|isn't|isnt|"
        r"can't|cant|wouldn't|wouldnt|shouldn't|shouldnt|haven't|havent)")
_OPENER = re.compile(rf"^(?:(?:so|and|but|okay|ok|well|um|uh|hey|then|like|now)[, ]+)*(?:{_WH}|{_AUX})\b", re.I)
_YOU = re.compile(r"\b(?:you|your|yours|yourself|you're|youre|u|ya)\b", re.I)
_TAG = re.compile(
    r"^(?:(?:you know|right|okay|ok|yeah|huh|eh|no|yes|really|isn't it|isnt it|"
    r"don't you think|dont you think|you see|see|or what|what|sorry|pardon)[ ,.!?]*)+$",
    re.I,
)
# Same boundary rule as the phone (coachTrigger.ts: no lookbehind for Hermes).
_SENTENCE = re.compile(r"(?:[^.!?]|[.!?](?=\S))+[.!?]*")


def _words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_'À-￿]+", s)


def _last_sentence(text: str) -> str:
    parts = [p.strip() for p in _SENTENCE.findall(text.strip()) if p.strip()]
    return parts[-1] if parts else ""


QUESTION_MIN_WORDS = 5
_DISCOURSE = (r"(?:So|Well|Okay|OK|Ok|And|But|Now|Then|Like|Um|Uh|Hey|Yes|No|Yeah|Look|Listen|Honestly|"
              r"Seriously|Sorry|Please|Oh|Wait|Right|Alright|Anyway|Actually|Man|Dude|Guys|God)")
# A question put to a NAMED person ("Where'd you go, Mel?", "Mel, are you
# coming?"): the listener is whoever was named, which the coach cannot tell
# is the wearer, and in practice it is casual address, not a moment to coach.
_UNPUNCT_Q = re.compile(
    rf"^(?:(?:so|and|but|okay|ok|well|um|uh|hey|then|now)[, ]+)*(?:{_WH}\s+)?{_AUX}\s+(?:you|u|ya)\b", re.I)
_VOCATIVE_END = re.compile(r",\s*[A-Z][a-z]+\s*[?!.]*$")
_VOCATIVE_START = re.compile(rf"^(?:[Hh]ey\s+)?(?!{_DISCOURSE}\b)[A-Z][a-z]+,\s")


def is_direct_question(text: str) -> bool:
    """True when the turn's LAST sentence is a question put to the listener.

    v2 (2026-10-30, tightened after replay review): the question must
    address "you" in its body, have >= ``QUESTION_MIN_WORDS`` words once a
    trailing tag ("..., right?") is stripped, end in "?" or open with an
    interrogative (on-device STT drops punctuation), and not be addressed to
    a NAME ("Where'd you go, Mel?"). Tag questions ("you know?", "right?")
    never count.
    """
    last = _last_sentence(text or "")
    if not last or _TAG.match(last):
        return False
    body = re.sub(r"[, ]+(?:you know|right|okay|ok|yeah|huh|eh)\s*\?*$", "", last, flags=re.I).strip()
    if len(_words(body)) < QUESTION_MIN_WORDS:
        return False
    if _VOCATIVE_END.search(body) or _VOCATIVE_START.match(body):
        return False
    opener = bool(_OPENER.match(body))
    if not _YOU.search(body):
        return False
    ends_q = last.endswith("?")
    if ends_q and body != last.rstrip("?").strip() and not opener:
        # "You went to the shop, right?" - the question mark belongs to a tag.
        return False
    # Without a "?" (on-device STT drops punctuation) only an inverted
    # question counts: "do you...", "what do you...", "are you...". A bare
    # wh-word opener is usually a clause ("when you fold it open").
    return ends_q or bool(_UNPUNCT_Q.match(body))


# ---------------------------------------------------------------------------
# library: distinctive phrases from the selected items
# ---------------------------------------------------------------------------

_STOP = frozenset(
    """a about above after again against all am an and any are aren't as at be because been before being
    below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
    during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here
    here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
    let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves
    out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their
    theirs them themselves then there there's these they they'd they'll they're they've this those through
    to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's
    where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're
    you've your yours yourself yourselves also just like get got really yeah okay ok well one two use used
    will said says say thing things way make made know think want going go see look""".split()
)
_WORD = re.compile(r"[a-z0-9][a-z0-9'\-]*")


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower()) if len(w) >= 3 and w not in _STOP]


def library_phrases(text: str, *, max_phrases: int = 20000) -> frozenset[str]:
    """Content-word bigrams ("payment plan", "late fee") of a library block.
    Adjacent in the source text after dropping stopwords, so "the late fee
    is" -> "late fee". Bounded so a huge selection cannot blow up memory."""
    words = _content_words(text)
    out: set[str] = set()
    for a, b in zip(words, words[1:]):
        out.add(f"{a} {b}")
        if len(out) >= max_phrases:
            break
    return frozenset(out)


def library_hit(text: str, phrases: frozenset[str] | set[str] | None) -> bool:
    if not phrases:
        return False
    words = _content_words(text)
    return any(f"{a} {b}" in phrases for a, b in zip(words, words[1:]))


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriggerVerdict:
    """Why a turn may (or may not) go to the cloud coach LLM."""

    triggers: tuple[str, ...]
    heat: float = 0.0

    @property
    def fires(self) -> bool:
        return bool(self.triggers)


def evaluate(
    text: str,
    *,
    is_wearer: bool,
    wearer_floor_s: float = 0.0,
    wearer_floor_before_s: float | None = None,
    phone_triggers: list[str] | tuple[str, ...] | None = None,
    library: frozenset[str] | set[str] | None = None,
) -> TriggerVerdict:
    """Every trigger that holds for this turn (empty = do not call the LLM).

    ``is_wearer`` is True only for a CONFIRMED wearer turn (a question the
    wearer asks is not a question to the wearer; an unknown voice's is).
    """
    found: list[str] = []
    score, _ = heat_score(text)
    phone = {t for t in (phone_triggers or ()) if t in TRIGGERS}
    if _flag("MINDSHIFT_TRIGGER_HEAT") and (score >= _num("MINDSHIFT_TRIGGER_HEAT_MIN", 1.5) or "heat" in phone):
        found.append("heat")
    floor_x = _num("MINDSHIFT_TRIGGER_FLOOR_S", 45.0)
    crossed = is_wearer and wearer_floor_s >= floor_x and (
        wearer_floor_before_s is None or wearer_floor_before_s < floor_x
    )
    if _flag("MINDSHIFT_TRIGGER_FLOOR") and (crossed or "floor" in phone):
        found.append("floor")
    if _flag("MINDSHIFT_TRIGGER_QUESTION") and not is_wearer and is_direct_question(text):
        found.append("question")
    if _flag("MINDSHIFT_TRIGGER_LIBRARY") and library_hit(text, library):
        found.append("library")
    return TriggerVerdict(triggers=tuple(found), heat=score)
