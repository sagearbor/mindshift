"""Live Coach ROOM mode: an assistant for the whole meeting room.

The phone sits on a conference-room table (or a laptop shares its screen).
Unlike earpiece coaching, nothing here is about the wearer:

* **Addressed questions** — an utterance that STARTS with the wake phrase
  ("MindShift, ..." / "Hey MindShift ...", tolerant of STT spellings such as
  "Mind shift") is answered in one or two spoken sentences from the session's
  library and the recent conversation (one LLM call, prompts in
  room_prompts.py). If neither holds the answer the model says so; it is told
  never to guess. The phone speaks the answer aloud.
* **Info cards** — when an utterance mentions something the library covers (a
  client, a product, a metric), a short card with the matching library line
  and its source goes to the screen. Deterministic and extractive: NO LLM
  call, the fact is the library's own words. Rate-limited and de-duplicated.

Cheap by construction: a non-addressed utterance costs a regex, a token set
intersection and (only when a mention looks likely) a library lookup that is
cached in full mode. The only LLM call is for an addressed question.

Frames (server -> client), all additive:
    {"type": "room_card", "id", "title", "fact", "source_item_id",
     "source_title", "t"}
    {"type": "room_listening", "t"}      wake phrase heard with no question
                                          yet; the next utterance (within
                                          WAKE_FOLLOWUP_S) is the question
    {"type": "room_answer", "question", "text", "known", "source_item_ids",
     "t", "speak": true}
    {"type": "room_answer_error", "question", "reason", "t"}
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from llm_client import CachedPrefixPrompt
from room_prompts import room_answer_system, room_answer_user

logger = logging.getLogger("room_mode")

MODE_ROOM = "room"
WAKE_PHRASE = "MindShift"

# "MindShift" as speech-to-text writes it: "MindShift", "Mind shift",
# "mind-shift", "Mindshift's", and the common mishearing "mine shift". Only
# at the START of an utterance, after an optional greeting/filler word, so
# "we should ask MindShift later" is a mention, not a request.
_WAKE_RE = re.compile(
    r"^[\W_]*(?:(?:hey|hi|hello|ok|okay|so|um|uh|yo)[\W_]+)?"
    r"(?:mind|mine|mined|minds)[\s\-]*shift(?:'s|s)?\b"
    r"(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_LEADING_PUNCT = re.compile(r"^[\s,.:;!?\-–—…]+")

# The question may follow the bare wake phrase as its own utterance.
WAKE_FOLLOWUP_S = 8.0
# Cards: at most one per CARD_MIN_GAP_S of session time; a fact is shown at
# most once per session.
CARD_MIN_GAP_S = 12.0
CARD_FACT_MAX_CHARS = 220
# Long library lines are split into sentences so a card stays one fact.
_LONG_LINE_CHARS = 240
# Recent conversation lines given to the answer prompt.
ANSWER_HISTORY_TURNS = 12
ANSWER_MAX_TOKENS = 200
ANSWER_MAX_SENTENCES = 2
ANSWER_MAX_CHARS = 320


def detect_address(text: str) -> str | None:
    """The question when ``text`` starts with the wake phrase ("" when the
    phrase stands alone), else ``None``."""
    if not text:
        return None
    m = _WAKE_RE.match(text)
    if m is None:
        return None
    return _LEADING_PUNCT.sub("", m.group("rest")).strip()


# ---------------------------------------------------------------------------
# library block -> facts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fact:
    item_id: str
    item_title: str
    text: str
    index: int  # position in the block, for a stable card id + tie-break


_ITEM_RE = re.compile(
    r'<library_item id="(?P<id>[^"]*)" title="(?P<title>[^"]*)" kind="[^"]*">\n'
    r"(?P<body>.*?)\n?</library_item>",
    re.DOTALL,
)
_EXCERPT_TAG = re.compile(r'</?excerpt(?: n="\d+")?>')
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def facts_from_library(block: str) -> list[Fact]:
    """Split the rendered ``<wearer_library>`` block (full or retrieved, as
    library.context renders it) into one fact per non-empty line; very long
    lines into sentences."""
    facts: list[Fact] = []
    for m in _ITEM_RE.finditer(block or ""):
        item_id = html.unescape(m.group("id"))
        title = html.unescape(m.group("title"))
        body = _EXCERPT_TAG.sub("\n", m.group("body"))
        for line in body.splitlines():
            line = html.unescape(line).strip()
            if not line:
                continue
            parts = _SENTENCE_SPLIT.split(line) if len(line) > _LONG_LINE_CHARS else [line]
            for part in parts:
                part = part.strip()
                if part:
                    facts.append(Fact(item_id, title, part, len(facts)))
    return facts


# ---------------------------------------------------------------------------
# terms
# ---------------------------------------------------------------------------

# Capitalised words that are NOT a client, product or metric.
_STOPWORDS = frozenset("""
a about after again all also am an and any are as at be because been before
being but by can could did do does done each for from had has have he her here
hers him his how i if in into is it its just let lets like me more most my no
nor not now of off ok okay on once only or other our ours out over own same she
should so some such than that the their theirs them then there these they this
those through to too under until up very was we were what when where which
while who whom why will with would yes yet you your yours hey hi hello thanks
thank please well yeah right sure maybe really actually also mr mrs ms dr
monday tuesday wednesday thursday friday saturday sunday today tomorrow
yesterday morning afternoon evening week month year quarter annual total note
notes account accounts champion customer client team plan meeting call email
us pm am tbd n/a etc vs per new next last first second third fourth
mindshift mind shift
""".split())

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9&\-]*")
_CODE_RE = re.compile(r"^(?:q[1-4]|h[12]|fy\d{2,4})$", re.IGNORECASE)
_QUARTER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4",
                  "first": "1", "second": "2", "third": "3", "fourth": "4"}
_Q_SPOKEN = re.compile(r"\bq\s+(one|two|three|four|[1-4])\b", re.IGNORECASE)
_Q_ORDINAL = re.compile(r"\b(first|second|third|fourth)\s+quarter\b", re.IGNORECASE)
_Q_QUARTER_N = re.compile(r"\bquarter\s+(one|two|three|four|[1-4])\b", re.IGNORECASE)


def _normalize_quarters(text: str) -> str:
    def sub(m: re.Match) -> str:
        n = m.group(1).lower()
        return "Q" + _QUARTER_WORDS.get(n, n)
    text = _Q_SPOKEN.sub(sub, text)
    text = _Q_ORDINAL.sub(sub, text)
    return _Q_QUARTER_N.sub(sub, text)


def _tokens(text: str) -> list[str]:
    text = re.sub(r"['’]s\b", "", text)
    return _TOKEN_RE.findall(text)


def _is_anchor(token: str) -> bool:
    """A library token that can identify a client/product/metric: a
    capitalised non-stopword (>= 3 chars) or an all-caps acronym (>= 2)."""
    low = token.lower()
    if low in _STOPWORDS or _CODE_RE.match(token) or token.isdigit():
        return False
    if len(token) >= 2 and token.isupper() and token.isalpha():
        return True
    return len(token) >= 3 and token[0].isupper() and not token.isupper()


@dataclass
class _IndexedFact:
    fact: Fact
    anchors: dict[str, str]  # lower -> as written in the library
    codes: dict[str, str]


def _index(facts: list[Fact]) -> list[_IndexedFact]:
    out: list[_IndexedFact] = []
    for f in facts:
        anchors: dict[str, str] = {}
        codes: dict[str, str] = {}
        for tok in _tokens(f.text):
            if _CODE_RE.match(tok):
                codes.setdefault(tok.lower(), tok.upper())
            elif _is_anchor(tok):
                anchors.setdefault(tok.lower(), tok)
        out.append(_IndexedFact(f, anchors, codes))
    return out


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# per-session state
# ---------------------------------------------------------------------------

@dataclass
class RoomState:
    """One room session's card bookkeeping and pending wake phrase."""

    seen_card_ids: set[str] = field(default_factory=set)
    last_card_t: float | None = None
    pending_wake_t: float | None = None
    # Lower-cased anchor/code terms from the last library block seen: the
    # cheap pre-filter that decides whether a lookup is worth doing.
    known_terms: set[str] = field(default_factory=set)
    indexed_once: bool = False
    cards_sent: int = 0
    answers: int = 0
    _index_key: str | None = None
    _indexed: list[_IndexedFact] = field(default_factory=list)

    def likely_mention(self, text: str) -> bool:
        """True when ``text`` could mention something in the library: it
        shares a term with the library seen so far, or carries a capitalised
        non-stopword / period code (speech-to-text capitalises names)."""
        toks = _tokens(_normalize_quarters(text))
        if any(t.lower() in self.known_terms for t in toks):
            return True
        return any(_is_anchor(t) for t in toks)

    def _indexed_for(self, block: str) -> list[_IndexedFact]:
        key = hashlib.sha1(block.encode()).hexdigest()
        if key != self._index_key:
            self._index_key = key
            self._indexed = _index(facts_from_library(block))
            for ix in self._indexed:
                self.known_terms.update(ix.anchors)
                self.known_terms.update(ix.codes)
            self.indexed_once = True
        return self._indexed

    def card_for(self, text: str, library_ctx, *, t: float) -> dict | None:
        """The ``room_card`` frame for this utterance, or ``None`` (no library,
        no anchor match, already shown, or within CARD_MIN_GAP_S of the last
        card). Records the card as shown when it returns one."""
        if library_ctx is None or not getattr(library_ctx, "text", ""):
            return None
        indexed = self._indexed_for(library_ctx.text)
        said = {tok.lower() for tok in _tokens(_normalize_quarters(text))} - _STOPWORDS
        if not said:
            return None
        best: tuple[int, int, _IndexedFact, list[str], list[str]] | None = None
        for ix in indexed:
            anchors = [w for w in ix.anchors if w in said]
            if not anchors:
                continue
            codes = [c for c in ix.codes if c in said]
            score = 2 * len(anchors) + len(codes)
            cand = (score, -ix.fact.index, ix, anchors, codes)
            if best is None or cand[:2] > best[:2]:
                best = cand
        if best is None:
            return None
        _, _, ix, anchors, codes = best
        card_id = hashlib.sha1(f"{ix.fact.item_id}\n{ix.fact.text}".encode()).hexdigest()[:16]
        if card_id in self.seen_card_ids:
            return None
        if self.last_card_t is not None and t - self.last_card_t < CARD_MIN_GAP_S:
            return None
        self.seen_card_ids.add(card_id)
        self.last_card_t = t
        self.cards_sent += 1
        # Title: the matched names as the library writes them, in the order
        # they appear in the fact, then the matched period codes.
        order = {w.lower(): i for i, w in enumerate(_tokens(ix.fact.text))}
        names = sorted(anchors, key=lambda w: order.get(w, 1 << 30))
        title = " · ".join([ix.anchors[w] for w in names] + [ix.codes[c] for c in codes])
        return {
            "type": "room_card",
            "id": card_id,
            "title": title,
            "fact": _clip(ix.fact.text, CARD_FACT_MAX_CHARS),
            "source_item_id": ix.fact.item_id,
            "source_title": ix.fact.item_title,
            "t": t,
        }


# ---------------------------------------------------------------------------
# answers
# ---------------------------------------------------------------------------

class RoomAnswerUnavailable(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_UNKNOWN_RE = re.compile(
    r"\b(?:don'?t|do not|doesn'?t|does not)\s+(?:have|know|see|contain|mention)\b"
    r"|\bnot in the (?:library|conversation)\b|\bno information\b",
    re.IGNORECASE,
)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _first_sentences(text: str, n: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n]).strip()


def parse_answer(raw: str) -> dict:
    """``{"text", "known", "source_item_ids"}`` from the model's JSON reply.
    Raises :class:`RoomAnswerUnavailable` (``llm_parse_error``) when there is
    no usable answer — an empty or invented fallback would be spoken aloud."""
    s = _FENCE.sub("", (raw or "").strip())
    start, end = s.find("{"), s.rfind("}")
    try:
        data = json.loads(s[start : end + 1]) if start >= 0 and end > start else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise RoomAnswerUnavailable("llm_parse_error")
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise RoomAnswerUnavailable("llm_parse_error")
    text = _clip(_first_sentences(answer, ANSWER_MAX_SENTENCES), ANSWER_MAX_CHARS)
    known = data.get("known")
    if not isinstance(known, bool):
        known = not bool(_UNKNOWN_RE.search(text))
    ids = data.get("source_item_ids")
    ids = [i for i in ids if isinstance(i, str)] if isinstance(ids, list) else []
    return {"text": text, "known": known, "source_item_ids": ids}


async def answer_question(
    llm, question: str, *, asker: str | None, library_ctx, conversation: list[str], t: float,
) -> dict:
    """One LLM call -> the ``room_answer`` frame. The library block is placed
    like the coach's: a cache-marked system prefix in full mode, with the turn
    in retrieved mode. Raises RoomAnswerUnavailable / the client's error."""
    has_library = bool(library_ctx is not None and library_ctx.text)
    system: str = room_answer_system(library=has_library)
    user = room_answer_user(question, asker, conversation[-ANSWER_HISTORY_TURNS:])
    if has_library:
        if library_ctx.mode == "full":
            system = CachedPrefixPrompt(library_ctx.text + "\n", system)
        else:
            user = library_ctx.text + "\n" + user
    raw = await asyncio.to_thread(
        llm.complete, system=system, user=user, max_tokens=ANSWER_MAX_TOKENS,
    )
    parsed = parse_answer(raw)
    allowed = set(library_ctx.item_ids) if has_library else set()
    return {
        "type": "room_answer",
        "question": question,
        "text": parsed["text"],
        "known": parsed["known"],
        "source_item_ids": [i for i in parsed["source_item_ids"] if i in allowed],
        "t": t,
        "speak": True,
    }


# ---------------------------------------------------------------------------
# one utterance
# ---------------------------------------------------------------------------

async def handle_room_turn(
    state: RoomState,
    *,
    text: str,
    speaker: str | None,
    t: float,
    conversation: list[str],
    get_library: Callable[[str], Awaitable[object | None]] | None,
    llm,
    send: Callable[[dict], Awaitable[None]],
) -> None:
    """Run the room pipeline for one finalized utterance.

    ``conversation`` is the recent transcript as ``"Speaker A: text"`` lines,
    EXCLUDING this utterance. ``get_library(query)`` is the session's bounded
    library lookup (None when nothing is selected). Never raises: an answer
    that cannot be produced is reported as ``room_answer_error``.
    """
    question = detect_address(text)
    if question is None and state.pending_wake_t is not None:
        if t - state.pending_wake_t <= WAKE_FOLLOWUP_S and text.strip():
            question = text.strip()
        state.pending_wake_t = None
    if question is not None:
        if not question:
            state.pending_wake_t = t
            await send({"type": "room_listening", "t": t})
            return
        state.pending_wake_t = None
        library_ctx = None
        if get_library is not None:
            library_ctx = await get_library("\n".join(conversation[-6:] + [question]))
        try:
            frame = await answer_question(
                llm, question, asker=speaker, library_ctx=library_ctx,
                conversation=conversation, t=t,
            )
        except Exception as exc:  # noqa: BLE001 — reported, never fabricated
            reason = exc.reason if isinstance(exc, RoomAnswerUnavailable) else type(exc).__name__
            logger.warning("room answer failed: %s", reason)
            await send({"type": "room_answer_error", "question": question, "reason": reason, "t": t})
            return
        state.answers += 1
        await send(frame)
        return

    if get_library is None:
        return
    if state.indexed_once and not state.likely_mention(text):
        return
    library_ctx = await get_library("\n".join(conversation[-6:] + [text]))
    card = state.card_for(text, library_ctx, t=t)
    if card is not None:
        await send(card)
