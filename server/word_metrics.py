"""Transparent, dictionary-based word-level metrics (Pennebaker-inspired).

Pure, deterministic, LOCALLY-computed per-speaker language statistics derived
straight from the diarized turns — no LLM, no network, no clock. Every number
here can be reproduced by hand from the transcript and the fixed word lists in
this file, which is the whole point: unlike the heat scores (an LLM judgment),
these are auditable counts the UI can fully explain ("here is exactly what was
counted").

What is measured
----------------

* **Pronoun profile** — the rate (per 100 words) at which a speaker uses
  first-person-singular pronouns (I/me/my), second-person pronouns (you/your),
  and first-person-plural pronouns (we/us/our). Pronoun use is among the most
  replicated findings in Pennebaker's language-and-relationships work: a high
  you-rate ("you always…", "you never…") tracks blame/accusation, while
  first-person-singular tracks ownership and self-disclosure.

* **Emotion-word density** — the rate (per 100 words) of anger / fear / sadness
  / joy / trust words, counted against the hand-curated word lists below.

Design constraints (house rules, mirroring dynamics.py / episodes.py)
---------------------------------------------------------------------

* PURE — plain data in, plain JSON-ready dict out. Trivially unit-testable.
* HONEST NULLS — a speaker with fewer than :data:`LOW_SAMPLE_MIN_WORDS` words
  gets ``low_sample: true`` and ``None`` for every RATE (raw counts are still
  reported — they are exact — but a "3.7 per 100 words" computed off 8 words is
  fake precision we refuse to print).
* TRANSPARENT BY DESIGN — these are deliberately naive surface counts. There is
  NO negation handling ("not happy" still counts "happy" as a joy word), no
  stemming, no word-sense disambiguation. Being clever here would make the
  number un-auditable. The exact method — including every word list — is
  returned in the ``method`` block so the client can show "how this is computed"
  and state the caveats plainly.

Lexicon provenance / licensing
------------------------------

The obvious choice for emotion words is the NRC Emotion Lexicon (EmoLex). We do
NOT use it: EmoLex is free for RESEARCH only — commercial use requires a paid
NRC licence (verified 2026-07 at saifmohammad.com/WebPages/AccessResource.htm) —
and MindShift is a commercial product. Rather than ship under an incompatible
licence or an ambiguous one, the word lists below are a COMPACT, HAND-CURATED,
ORIGINAL set of common English emotion terms authored for this file. Individual
common words are not copyrightable and this curation depends on no third-party
lexicon, so it carries no attribution burden. It is intentionally small and
high-precision (strong, unambiguous affect words) rather than exhaustive — an
honest, transparent floor, not a research-grade lexicon. Swapping in a
commercially-licensed EmoLex later is a drop-in replacement for these frozensets.
"""

from __future__ import annotations

import re

# A speaker below this many words gets honest null RATES (see module docstring).
# ~20 words is roughly the floor at which a "per 100 words" rate stops being
# dominated by the presence/absence of a single token.
LOW_SAMPLE_MIN_WORDS = 20

# Word tokenizer: lowercase alphabetic runs, keeping a single internal apostrophe
# so contractions ("i'm", "you're", "we'll") survive as one token and match the
# pronoun sets below. Everything else (digits, punctuation) is a boundary.
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


# ---------------------------------------------------------------------------
# Pronoun sets (Pennebaker's function-word categories). Contracted forms are
# listed explicitly because the tokenizer keeps the apostrophe.
# ---------------------------------------------------------------------------

FIRST_SINGULAR = frozenset({
    "i", "me", "my", "mine", "myself",
    "i'm", "i've", "i'll", "i'd",
})
SECOND_PERSON = frozenset({
    "you", "your", "yours", "yourself", "yourselves",
    "you're", "you've", "you'll", "you'd",
})
FIRST_PLURAL = frozenset({
    "we", "us", "our", "ours", "ourselves",
    "we're", "we've", "we'll", "we'd", "let's",
})


# ---------------------------------------------------------------------------
# Emotion word lists — hand-curated, original, licence-free (see docstring).
# Kept small and high-precision: strong, unambiguous affect words only.
# ---------------------------------------------------------------------------

ANGER_WORDS = frozenset({
    "angry", "anger", "mad", "furious", "fury", "rage", "enraged", "hate",
    "hateful", "hostile", "hostility", "irritated", "annoyed", "annoying",
    "resentful", "resent", "outraged", "outrage", "frustrated", "frustrating",
    "fuming", "livid", "bitter", "contempt", "provoked", "aggravated",
    "agitated", "indignant", "offended", "spite", "temper", "wrath", "seething",
    "yelling", "shouting", "aggressive", "vengeful", "pissed",
})
FEAR_WORDS = frozenset({
    "afraid", "fear", "fearful", "scared", "terrified", "terror", "anxious",
    "anxiety", "worried", "worry", "nervous", "panic", "panicked", "dread",
    "frightened", "threatened", "insecure", "uneasy", "apprehensive",
    "alarmed", "tense", "timid", "horror", "petrified", "spooked", "jittery",
    "paranoid", "vulnerable", "helpless", "distressed", "trembling",
})
SADNESS_WORDS = frozenset({
    "sad", "sadness", "unhappy", "depressed", "miserable", "grief", "sorrow",
    "lonely", "heartbroken", "hopeless", "despair", "gloomy", "disappointed",
    "hurt", "crying", "cry", "tears", "mourning", "melancholy", "dejected",
    "weeping", "regret", "guilt", "guilty", "ashamed", "sorry", "defeated",
    "empty", "numb", "grieving",
})
JOY_WORDS = frozenset({
    "happy", "happiness", "joy", "joyful", "glad", "delighted", "delight",
    "cheerful", "excited", "elated", "pleased", "content", "thrilled",
    "grateful", "gratitude", "wonderful", "love", "loving", "hopeful",
    "optimistic", "proud", "smile", "smiling", "laughing", "laugh", "fun",
    "enjoy", "enjoyed", "celebrate", "blessed", "bliss", "ecstatic",
    "satisfied",
})
TRUST_WORDS = frozenset({
    "trust", "trusted", "trusting", "honest", "honesty", "faith", "reliable",
    "dependable", "loyal", "loyalty", "respect", "respected", "support",
    "supportive", "safe", "secure", "confident", "sincere", "believe",
    "understanding", "understand", "comfort", "comfortable", "accept",
    "accepting", "commitment", "committed", "promise", "integrity",
    "faithful", "truthful",
})

# Ordered so the emitted per-speaker keys and the method block are stable.
_EMOTIONS: tuple[tuple[str, frozenset[str]], ...] = (
    ("anger", ANGER_WORDS),
    ("fear", FEAR_WORDS),
    ("sadness", SADNESS_WORDS),
    ("joy", JOY_WORDS),
    ("trust", TRUST_WORDS),
)


def tokenize(text: object) -> list[str]:
    """Lowercase word tokens of ``text`` (contractions kept whole). A non-string
    or empty text yields an empty list — never an error."""
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())


def _rate(count: int, total: int) -> float:
    """Occurrences of ``count`` per 100 of ``total`` words, rounded to 2 dp.
    ``total`` is guaranteed > 0 by the caller (low-sample speakers never reach
    here)."""
    return round(count * 100.0 / total, 2)


def _speaker_stats(tokens: list[str]) -> dict:
    """The metric block for ONE speaker's pooled tokens.

    Raw counts are always reported (they are exact). Rates are ``None`` when the
    speaker is below :data:`LOW_SAMPLE_MIN_WORDS` — honest, never fabricated
    precision — with ``low_sample`` flagging why.
    """
    word_count = len(tokens)
    i_count = sum(1 for t in tokens if t in FIRST_SINGULAR)
    you_count = sum(1 for t in tokens if t in SECOND_PERSON)
    we_count = sum(1 for t in tokens if t in FIRST_PLURAL)
    emotion_counts = {
        name: sum(1 for t in tokens if t in words) for name, words in _EMOTIONS
    }

    low_sample = word_count < LOW_SAMPLE_MIN_WORDS

    stats: dict = {
        "word_count": word_count,
        "low_sample": low_sample,
        # Raw counts — exact regardless of sample size.
        "i_count": i_count,
        "you_count": you_count,
        "we_count": we_count,
    }
    for name in emotion_counts:
        stats[f"{name}_count"] = emotion_counts[name]

    # Rates — honest None below the low-sample floor.
    if low_sample:
        stats["i_rate"] = None
        stats["you_rate"] = None
        stats["we_rate"] = None
        for name in emotion_counts:
            stats[f"{name}_rate"] = None
    else:
        stats["i_rate"] = _rate(i_count, word_count)
        stats["you_rate"] = _rate(you_count, word_count)
        stats["we_rate"] = _rate(we_count, word_count)
        for name, count in emotion_counts.items():
            stats[f"{name}_rate"] = _rate(count, word_count)
    return stats


def _method() -> dict:
    """The exact, self-describing recipe — every word list included — so the UI
    can show 'how this number is computed' and state the caveats honestly."""
    return {
        "description": (
            "Per-speaker surface counts over the diarized transcript, computed "
            "locally with no LLM. Text is lowercased and split into word tokens "
            "(alphabetic runs; a single internal apostrophe is kept so "
            "contractions match). Each rate is that category's token count per "
            "100 of the speaker's words."
        ),
        "unit": "occurrences per 100 words",
        "low_sample_min_words": LOW_SAMPLE_MIN_WORDS,
        "low_sample_note": (
            "A speaker with fewer than the minimum words reports low_sample=true "
            "and null rates (raw *_count fields are still exact); we refuse to "
            "print a rate that a single word would swing."
        ),
        "negation_caveat": (
            "Deliberately naive: these are transparent, auditable counts, not "
            "sentiment analysis. There is NO negation handling ('not happy' "
            "still counts 'happy' as a joy word), no stemming, and no "
            "word-sense disambiguation."
        ),
        "pronouns": {
            "i": sorted(FIRST_SINGULAR),
            "you": sorted(SECOND_PERSON),
            "we": sorted(FIRST_PLURAL),
        },
        "emotion_lexicon": {
            "source": (
                "Hand-curated, original, licence-free word list authored for "
                "MindShift (NOT the NRC Emotion Lexicon, which requires a "
                "commercial licence). Small and high-precision by design."
            ),
            "categories": {name: sorted(words) for name, words in _EMOTIONS},
        },
    }


def compute_word_metrics(turns: list[dict]) -> dict | None:
    """Per-speaker word metrics for a whole conversation, or ``None`` when there
    is nothing to count.

    ``turns`` are the diarized turn dicts ({speaker, text, ...}); only
    ``speaker`` and ``text`` are read, so this works identically on the analysis
    pipeline's live turns and on a stored recording's ``turns.json`` (the
    read-path backfill, mirroring episodes_from_analysis). Tokens are pooled per
    canonical speaker id across all their turns.

    Returns ``{speakers: {<id>: {...}}, method: {...}}`` — ``method`` fully
    describes what was counted. ``None`` when ``turns`` is empty or carries no
    speaker with any text (a stored-but-empty transcript has nothing honest to
    report).
    """
    if not turns:
        return None

    # Pool tokens per canonical speaker id, in first-appearance order.
    by_speaker: dict[str, list[str]] = {}
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = turn.get("speaker")
        if not isinstance(speaker, str) or not speaker:
            continue
        by_speaker.setdefault(speaker, []).extend(tokenize(turn.get("text")))

    if not by_speaker:
        return None

    return {
        "speakers": {sp: _speaker_stats(toks) for sp, toks in by_speaker.items()},
        "method": _method(),
    }
