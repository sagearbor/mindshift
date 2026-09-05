"""NaturalTurn port for the server's batch/recording path (Cooney & Reece 2025,
Sci Reports 41598-025-24381-1; upstream python: github.com/betterup/
natural-turn-transcription, MIT). Numpy-free, spacy-free: a regex tokenizer
mirrors spacy's non-punct tokens closely enough for our purposes (same
regex as the phone port's `wordsOf`).

Two BATCH-only additions this module has that
``apps/mobile/src/live/naturalTurn.ts`` intentionally does not:

* :func:`sentences_from_words` — port of upstream's
  ``transcription/text.py::_create_sentences_from_tokens``, adapted for our
  word-timed ASR shape (Deepgram gives one token per word, with punctuation
  attached to the word's own text via ``punctuated_word`` rather than emitted
  as a separate token the way AWS Transcribe's raw stream does upstream).
  Stitches a speaker's consecutive words into sentence-level utterances,
  splitting at terminal punctuation and closing whatever is left over at the
  end of the word list.
* :func:`collapse_short_pauses` — port of ``_collapse_short_pauses`` +
  its join: merges a SINGLE speaker's consecutive sentence-level utterances
  when the pause between them is under ``max_pause``. This runs BEFORE
  containment/classification.

Why these matter: the live phone path hands `classifyUtterance` a single
already-VAD-finalized turn's whole text, so there is nothing to stitch or
collapse — the ASR/VAD boundary IS the utterance boundary. The server's
batch/recording path instead gets raw word-timed ASR output, and running
containment directly on the ASR's OWN utterance/word boundaries (skipping
these two stages) is why an earlier port-vs-published validation
(tmp/candor/analysis/port_validation.json, 2026-09-05) measured backchannel
recall of 0.064 against BetterUp's published NaturalTurn transcripts despite
precision 1.00: real backchannels were there, but the raw ASR splits a
speaker's own continuous speech into far more fragments than a human
sentence boundary would, so the upstream sentence-stitch + short-pause
collapse stage is required to reproduce their published turns.

``classify_utterance``, ``label_turns`` and ``merge_primaries`` are kept in
lockstep with the TS port's decision order and constants — bit-identical,
verified by replaying the SAME fixture (tests/fixtures/policy_vectors/
natural_turn.json) from both runtimes. Only the two new pre-stages above are
server-only; feeding the TS port's whole-utterance input through this
module's ``label_turns``/``merge_primaries`` should reproduce the same
answers.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional, TypedDict

# ---------------------------------------------------------------------------
# Constants (mirror apps/mobile/src/live/naturalTurn.ts bit-for-bit)
# ---------------------------------------------------------------------------

# Upstream BACKCHANNEL_CUES (single tokens; the two-word "mm hmm" cue is inert
# upstream too — their Matcher builds one-token patterns).
BACKCHANNEL_CUES: frozenset[str] = frozenset({
    "a", "ah", "alright", "awesome", "cool", "dope", "e", "exactly", "god",
    "gotcha", "huh", "hmm", "mhm", "mm", "mmm", "nice", "oh", "okay",
    "really", "right", "sick", "sucks", "sure", "uh", "um", "wow", "yeah",
    "yep", "yes", "yup",
    # ASR (AWS/Deepgram) writes "ok" not "okay"; the upstream literal cue
    # never matches it on real transcripts (found mining CANDOR, 2026-09-05).
    "ok",
})

# Upstream NOT_BACKCHANNEL_CUES: a short turn STARTING with one of these is
# someone starting a real thought ("and then…", "but I…"), never a
# backchannel.
NOT_BACKCHANNEL_CUES: frozenset[str] = frozenset({
    "and", "but", "i", "i'm", "it", "it's", "like", "so", "that", "that's",
    "we", "we're", "well", "you", "you're",
})

BACKCHANNEL_WORD_MAX = 3
BACKCHANNEL_SECOND_MAX = 0.0  # natural_turn preset default
BACKCHANNEL_PROPORTION = 0.5
# Same-speaker primary turns closer than this merge into one turn.
MAX_PAUSE_SECONDS = 1.5
# Upstream token_conf: "anything below .6 is often hallucinated speech or
# incorrect transcription" (transcript_config.py comment, natural_turn
# preset's token_confidence_threshold).
TOKEN_CONFIDENCE_THRESHOLD = 0.6

TERMINAL_PUNCTUATION = (".", "!", "?")

UtteranceKind = str  # "primary" | "backchannel" | "secondary" | "other"

_WORD_RE = re.compile(r"[a-z0-9']+(?:-[a-z0-9']+)*")


def words_of(text: str) -> list[str]:
    """Lowercased word tokens with punctuation stripped (mirror of
    ``apps/mobile/src/live/naturalTurn.ts``'s ``wordsOf``, and of spacy's
    non-punct tokens for our purposes; apostrophes stay inside words so
    "i'm" matches the cue list)."""
    return _WORD_RE.findall(text.lower())


def classify_utterance(text: str, duration_seconds: float) -> Optional[str]:
    """Upstream ``determine_utterance_type``, same decision order:
    1. every word a cue -> backchannel (however long);
    2. more than WORD_MAX words (and longer than SECOND_MAX) -> secondary;
    3. first word a not-cue -> secondary;
    4. cue proportion >= PROPORTION -> backchannel; else other.
    Returns ``None`` for an utterance with no words. Bit-identical to the TS
    port's ``classifyUtterance``.
    """
    words = words_of(text)
    if not words:
        return None
    cues = sum(1 for w in words if w in BACKCHANNEL_CUES)
    prop = cues / len(words)
    if prop == 1:
        return "backchannel"
    if len(words) > BACKCHANNEL_WORD_MAX and duration_seconds > BACKCHANNEL_SECOND_MAX:
        return "secondary"
    if words[0] in NOT_BACKCHANNEL_CUES:
        return "secondary"
    return "backchannel" if prop >= BACKCHANNEL_PROPORTION else "other"


def live_turn_kind(text: str, duration_seconds: float) -> str:
    """The live tag for a finalized single-mic turn (mirror of the TS port's
    ``liveTurnKind``) — kept here mainly so the server can score a live-style
    input the same way the phone would, for parity checks."""
    return "backchannel" if classify_utterance(text, duration_seconds) == "backchannel" else "primary"


# ---------------------------------------------------------------------------
# Word shape helpers — flexible over the field names we actually see.
# ---------------------------------------------------------------------------
# Deepgram word objects: {"word": ..., "punctuated_word": ..., "start": ...,
# "end": ..., "confidence": ...}. server/audio_ingest.py's internal turn
# ``words`` plumbing instead uses {"word", "start_time", "end_time"} (no
# confidence, no punctuated_word — see _parse_utterance_words). Support both.


class Word(TypedDict, total=False):
    word: str
    punctuated_word: str
    start: float
    end: float
    start_time: float
    end_time: float
    confidence: float
    speaker: str


def _word_text(w: dict) -> str:
    text = w.get("punctuated_word") or w.get("word") or ""
    return str(text).strip()


def _word_start(w: dict) -> float:
    v = w.get("start", w.get("start_time"))
    return float(v)  # type: ignore[arg-type]


def _word_end(w: dict) -> float:
    v = w.get("end", w.get("end_time"))
    return float(v)  # type: ignore[arg-type]


def _word_confidence(w: dict) -> Optional[float]:
    v = w.get("confidence")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Batch path (channelized/diarized utterances with real overlap)
# ---------------------------------------------------------------------------


class Utterance(TypedDict):
    speaker: str
    start: float
    end: float
    text: str


class LabeledUtterance(Utterance, total=False):
    kind: str
    # Index (into the sorted utterance list) of the primary turn this
    # non-primary utterance interjects; None for primary turns.
    interjects: Optional[int]


class MergedTurn(TypedDict):
    speaker: str
    start: float
    end: float
    text: str
    # Non-primary utterances (backchannels etc.) that rode inside or between
    # this turn's merged parts.
    attached: list[LabeledUtterance]
    # How many primary utterances merged into this turn.
    parts: int


def sentences_from_words(
    words: Iterable[dict],
    speaker: str,
    *,
    confidence_threshold: float = TOKEN_CONFIDENCE_THRESHOLD,
) -> list[Utterance]:
    """Port of upstream ``_create_sentences_from_tokens`` for ONE speaker's
    chronologically-ordered word-timed ASR output.

    Stitches consecutive words into sentence-level utterances, closing a
    sentence when a word's text ends in terminal punctuation (``.``, ``!``,
    ``?`` — upstream instead watches for a separate punctuation TOKEN; our
    Deepgram-shaped words carry punctuation attached to the word itself, via
    ``punctuated_word``) and always flushing whatever is left over at the end
    of the input (mirrors upstream's "append final row regardless of terminal
    punc").

    When any word in ``words`` carries a ``confidence`` field, words below
    ``confidence_threshold`` are dropped BEFORE stitching — upstream's
    natural_turn preset treats sub-0.6-confidence tokens as likely
    hallucinated ASR noise (transcript_config.py ``token_conf`` comment).
    Confidence-free word shapes (e.g. our internal ``{word, start_time,
    end_time}`` turn plumbing) skip this filter entirely, per the task: the
    threshold only applies when confidences exist.
    """
    words_list = list(words)
    have_confidence = any(_word_confidence(w) is not None for w in words_list)
    if have_confidence:
        kept = [
            w for w in words_list
            if (_word_confidence(w) if _word_confidence(w) is not None else 1.0) >= confidence_threshold
        ]
    else:
        kept = words_list

    out: list[Utterance] = []
    start: Optional[float] = None
    end = 0.0
    content: list[str] = []

    def _flush() -> None:
        nonlocal start, end, content
        if content and start is not None:
            out.append({"speaker": speaker, "start": start, "end": end, "text": " ".join(content)})
        start = None
        end = 0.0
        content = []

    for w in kept:
        text = _word_text(w)
        if not text:
            continue
        w_start = _word_start(w)
        w_end = _word_end(w)
        if start is None or w_start < start:
            start = w_start
        if w_end > end:
            end = w_end
        content.append(text)
        if text[-1] in TERMINAL_PUNCTUATION:
            _flush()
    _flush()
    return out


def collapse_short_pauses(
    utterances: Iterable[Utterance],
    max_pause: float = MAX_PAUSE_SECONDS,
) -> list[Utterance]:
    """Port of upstream ``_collapse_short_pauses`` + its join: merges a SINGLE
    speaker's consecutive sentence-level utterances (already in chronological
    order) whenever the pause between them is STRICTLY less than
    ``max_pause`` (upstream: a gap ``>= max_pause`` starts a new group). Must
    run per-speaker BEFORE :func:`label_turns` — this is the pre-containment
    stage the live phone path skips (see module docstring) and whose absence
    measured a 0.064 backchannel recall against published NaturalTurn
    transcripts.
    """
    out: list[Utterance] = []
    current: Optional[Utterance] = None
    for u in utterances:
        if current is not None and u["start"] - current["end"] < max_pause:
            current["text"] = f"{current['text']} {u['text']}".strip()
            current["end"] = max(current["end"], u["end"])
            continue
        current = dict(u)  # type: ignore[assignment]
        out.append(current)
    return out


def label_turns(utterances: Iterable[Utterance]) -> list[LabeledUtterance]:
    """Upstream ``_label_turns``: sort by start; an utterance that begins
    before an earlier utterance ends AND ends within it (``start2 < stop1 and
    stop2 <= stop1``) is non-primary, attached to that turn; the forward scan
    stops at the first utterance that is not contained (upstream ``break``).
    Non-primary utterances are then classified; primaries stay "primary".
    Bit-identical decision order to the TS port's ``labelTurns``.
    """
    sorted_u = sorted(utterances, key=lambda u: (u["start"], u["end"]))
    out: list[LabeledUtterance] = [
        {**u, "kind": "primary", "interjects": None} for u in sorted_u  # type: ignore[dict-item]
    ]
    for i in range(len(out)):
        if out[i]["interjects"] is not None:
            continue  # already claimed by an earlier turn
        for j in range(i + 1, len(out)):
            if out[j]["start"] < out[i]["end"] and out[j]["end"] <= out[i]["end"]:
                out[j]["interjects"] = i
                out[j]["kind"] = classify_utterance(out[j]["text"], out[j]["end"] - out[j]["start"]) or "other"
            else:
                break  # first non-contained utterance ends this turn's window
    return out


def merge_primaries(
    labeled: Iterable[LabeledUtterance],
    max_pause: float = MAX_PAUSE_SECONDS,
) -> list[MergedTurn]:
    """Upstream merge: consecutive PRIMARY turns by the same speaker join
    when the pause between them is <= ``max_pause``. Non-primary utterances
    never break a merge — they attach to the merged turn (upstream reassigns
    their turn_id to the interjected primary). Bit-identical decision order
    to the TS port's ``mergePrimaries``.
    """
    out: list[MergedTurn] = []
    current: Optional[MergedTurn] = None
    for u in labeled:
        if u["kind"] != "primary":
            if current is not None:
                current["attached"].append(u)
            elif out:
                out[-1]["attached"].append(u)
            continue
        if (
            current is not None
            and u["speaker"] == current["speaker"]
            and u["start"] - current["end"] <= max_pause
        ):
            current["text"] = f"{current['text']} {u['text']}".strip()
            current["end"] = max(current["end"], u["end"])
            current["parts"] += 1
            continue
        current = {
            "speaker": u["speaker"], "start": u["start"], "end": u["end"],
            "text": u["text"], "attached": [], "parts": 1,
        }
        out.append(current)
    return out


def natural_turns(
    words_by_speaker: dict[str, Iterable[dict]],
    *,
    max_pause: float = MAX_PAUSE_SECONDS,
    confidence_threshold: float = TOKEN_CONFIDENCE_THRESHOLD,
) -> list[MergedTurn]:
    """Top-level entry point: word-timed ASR output PER SPEAKER in, merged
    NaturalTurn primary turns (with attached backchannels/secondaries) out.

    Pipeline (mirrors the upstream ``natural_turn`` preset's order):
    per speaker, stitch words into sentences (:func:`sentences_from_words`)
    and collapse short pauses between them (:func:`collapse_short_pauses`);
    then, across ALL speakers together, run containment (:func:`label_turns`)
    and the final same-speaker merge (:func:`merge_primaries`).
    """
    utterances: list[Utterance] = []
    for speaker, words in words_by_speaker.items():
        sentences = sentences_from_words(words, speaker, confidence_threshold=confidence_threshold)
        utterances.extend(collapse_short_pauses(sentences, max_pause=max_pause))
    labeled = label_turns(utterances)
    return merge_primaries(labeled, max_pause=max_pause)
