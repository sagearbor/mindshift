"""Unit tests for the transparent word-level metrics (word_metrics.py).

Pure-function tests — no FastAPI, no store, no LLM. They pin down the pronoun
math, the emotion counts, the honest low-sample nulls, the empty-input contract,
and the self-describing method block, exactly as the module promises the UI it
can reproduce by hand.
"""

import word_metrics as wm


# A long single-speaker line (>= LOW_SAMPLE_MIN_WORDS words) so RATES are emitted
# rather than nulled. Exactly 25 words, hand-counted: first-singular me/me/i/my
# (=4), second-person you/you (=2), first-plural we (=1); the rest is filler.
_LONG = (
    "You blame me and you never hear me while I keep my calm and we should "
    "talk word word word word word word word word"
)


def _speaker(turns):
    return wm.compute_word_metrics(turns)["speakers"]


# ---------------------------------------------------------------------------
# Tokenizer — contractions kept whole, case-folded, punctuation dropped
# ---------------------------------------------------------------------------

def test_tokenize_keeps_contractions_and_lowercases():
    assert wm.tokenize("I'm SURE you're right, aren't we?") == [
        "i'm", "sure", "you're", "right", "aren't", "we",
    ]


def test_tokenize_non_string_is_empty():
    assert wm.tokenize(None) == []
    assert wm.tokenize(42) == []
    assert wm.tokenize("") == []


# ---------------------------------------------------------------------------
# Pronoun math
# ---------------------------------------------------------------------------

def test_pronoun_counts_are_exact():
    # 25 words. first-singular: me, me, i, my (=4). second-person: you, you (=2).
    # first-plural: we (=1).
    stats = _speaker([{"speaker": "A", "text": _LONG}])["A"]
    assert stats["word_count"] == 25
    assert stats["i_count"] == 4
    assert stats["you_count"] == 2
    assert stats["we_count"] == 1


def test_pronoun_rates_are_per_100_words():
    stats = _speaker([{"speaker": "A", "text": _LONG}])["A"]
    assert stats["low_sample"] is False
    # 4/25 * 100 = 16.0, 2/25 * 100 = 8.0, 1/25 * 100 = 4.0
    assert stats["i_rate"] == 16.0
    assert stats["you_rate"] == 8.0
    assert stats["we_rate"] == 4.0


def test_contracted_pronouns_count():
    text = "I'm sure you're right and we've agreed " + "word " * 20
    stats = _speaker([{"speaker": "A", "text": text}])["A"]
    # i'm -> first-singular, you're -> second-person, we've -> first-plural.
    assert stats["i_count"] == 1
    assert stats["you_count"] == 1
    assert stats["we_count"] == 1


def test_lets_counts_as_first_plural():
    text = "let's go " + "filler " * 25
    stats = _speaker([{"speaker": "A", "text": text}])["A"]
    assert stats["we_count"] == 1


# ---------------------------------------------------------------------------
# Emotion-word density
# ---------------------------------------------------------------------------

def test_emotion_counts_across_categories():
    # 4 padding words + 5 emotion words = enough to leave low-sample territory
    # once padded; here we pad to 20+ and check one hit per category.
    text = (
        "I felt angry and scared and sad but also happy and I trust you "
        + "padding " * 15
    )
    stats = _speaker([{"speaker": "A", "text": text}])["A"]
    assert stats["anger_count"] == 1   # angry
    assert stats["fear_count"] == 1    # scared
    assert stats["sadness_count"] == 1  # sad
    assert stats["joy_count"] == 1     # happy
    assert stats["trust_count"] == 1   # trust
    assert stats["anger_rate"] is not None


def test_emotion_rate_math():
    # 2 anger words among 25 total -> 8.0 per 100.
    text = "angry furious " + "word " * 23
    stats = _speaker([{"speaker": "A", "text": text}])["A"]
    assert stats["word_count"] == 25
    assert stats["anger_count"] == 2
    assert stats["anger_rate"] == 8.0


def test_negation_is_not_handled_by_design():
    # "not happy" still counts "happy" — the module documents this transparency.
    text = "I am not happy " + "word " * 20
    stats = _speaker([{"speaker": "A", "text": text}])["A"]
    assert stats["joy_count"] == 1


# ---------------------------------------------------------------------------
# Honest low-sample nulls
# ---------------------------------------------------------------------------

def test_low_sample_nulls_rates_but_keeps_counts():
    stats = _speaker([{"speaker": "A", "text": "I am so angry at you"}])["A"]
    assert stats["word_count"] == 6
    assert stats["low_sample"] is True
    # Raw counts are exact even at tiny samples ...
    assert stats["i_count"] == 1
    assert stats["you_count"] == 1
    assert stats["anger_count"] == 1
    # ... but every RATE is an honest None (no fake precision).
    for key in (
        "i_rate", "you_rate", "we_rate",
        "anger_rate", "fear_rate", "sadness_rate", "joy_rate", "trust_rate",
    ):
        assert stats[key] is None


def test_exactly_at_threshold_is_not_low_sample():
    text = "word " * wm.LOW_SAMPLE_MIN_WORDS  # exactly the minimum
    stats = _speaker([{"speaker": "A", "text": text}])["A"]
    assert stats["word_count"] == wm.LOW_SAMPLE_MIN_WORDS
    assert stats["low_sample"] is False
    assert stats["i_rate"] == 0.0


# ---------------------------------------------------------------------------
# Pooling per speaker + first-appearance ordering
# ---------------------------------------------------------------------------

def test_tokens_pooled_per_speaker_across_turns():
    turns = [
        {"speaker": "A", "text": "I think " + "word " * 12},
        {"speaker": "B", "text": "you know " + "word " * 12},
        {"speaker": "A", "text": "I do " + "word " * 12},
    ]
    speakers = _speaker(turns)
    assert set(speakers) == {"A", "B"}
    # A's two turns are pooled: two "i" tokens across both turns.
    assert speakers["A"]["i_count"] == 2
    assert speakers["A"]["low_sample"] is False  # pooled well over threshold


# ---------------------------------------------------------------------------
# Empty / degenerate inputs -> honest None
# ---------------------------------------------------------------------------

def test_empty_turns_returns_none():
    assert wm.compute_word_metrics([]) is None


def test_turns_without_usable_speaker_returns_none():
    # No dict carries a usable (str, non-empty) speaker id -> nothing to report.
    assert wm.compute_word_metrics([
        {"text": "orphan text with no speaker"},
        {"speaker": "", "text": "blank speaker"},
        "not-a-dict",
    ]) is None


def test_speaker_with_empty_text_is_zero_not_error():
    stats = _speaker([{"speaker": "A", "text": ""}])["A"]
    assert stats["word_count"] == 0
    assert stats["low_sample"] is True
    assert stats["i_count"] == 0
    assert stats["i_rate"] is None


# ---------------------------------------------------------------------------
# The self-describing method block (transparency contract)
# ---------------------------------------------------------------------------

def test_method_block_is_self_describing():
    result = wm.compute_word_metrics([{"speaker": "A", "text": _LONG}])
    method = result["method"]
    assert method["low_sample_min_words"] == wm.LOW_SAMPLE_MIN_WORDS
    assert "per 100" in method["unit"]
    # Negation caveat is stated plainly.
    assert "negation" in method["negation_caveat"].lower()
    # Every pronoun list and emotion category is included verbatim so the UI can
    # show exactly what was counted.
    assert set(method["pronouns"]) == {"i", "you", "we"}
    assert "i" in method["pronouns"]["i"]
    assert set(method["emotion_lexicon"]["categories"]) == {
        "anger", "fear", "sadness", "joy", "trust",
    }
    assert "angry" in method["emotion_lexicon"]["categories"]["anger"]
    # Provenance is honest about NOT being the NRC lexicon.
    assert "nrc" in method["emotion_lexicon"]["source"].lower()
