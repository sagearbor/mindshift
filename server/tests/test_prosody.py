"""Unit tests for prosody.py — pure numpy, no network, no key.

Synthetic signals with KNOWN properties (sine tones, noise, amplitude/tempo
changes) exercise the measurements and the relative labeling. This is the
key-free layer; the DEEPGRAM_API_KEY-gated live test closes the loop against
physically-modulated real speech.
"""

import numpy as np

import prosody

SR = 16000


def _sine(freq: float, seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Pitch estimation
# ---------------------------------------------------------------------------

def test_pure_tone_pitch_within_tolerance():
    """A 200 Hz sine's median F0 lands within +/-15 Hz."""
    f0, std, voiced = prosody.estimate_pitch(_sine(200.0, 1.0), SR)
    assert f0 is not None
    assert abs(f0 - 200.0) <= 15.0
    assert voiced > 0.9          # a clean tone is almost entirely voiced
    assert std is not None and std < 5.0  # a steady tone barely varies


def test_unvoiced_noise_pitch_is_none():
    """Broadband noise has no periodicity → honest None, never a fabricated F0."""
    rng = np.random.default_rng(0)
    noise = (0.3 * rng.standard_normal(SR)).astype(np.float32)
    f0, std, voiced = prosody.estimate_pitch(noise, SR)
    assert f0 is None
    assert std is None
    assert voiced < prosody.MIN_VOICED_FRACTION


def test_silence_pitch_is_none():
    f0, std, voiced = prosody.estimate_pitch(np.zeros(SR, dtype=np.float32), SR)
    assert f0 is None and std is None and voiced == 0.0


# ---------------------------------------------------------------------------
# Energy labeling (relative to the recording's own turns)
# ---------------------------------------------------------------------------

def test_loud_window_labelled_louder_than_quiet_window():
    """A 0.5-amplitude turn is labelled louder than a 0.05-amplitude turn."""
    quiet = _sine(200.0, 1.0, amp=0.05)
    mid = _sine(180.0, 1.0, amp=0.2)
    loud = _sine(200.0, 1.0, amp=0.5)
    pcm = np.concatenate([quiet, mid, loud]).astype(np.float32)
    turns = [
        {"text": "one two", "start_time": 0.0, "end_time": 1.0},
        {"text": "one two", "start_time": 1.0, "end_time": 2.0},
        {"text": "one two", "start_time": 2.0, "end_time": 3.0},
    ]
    feats = [
        prosody.turn_features(pcm, SR, t["start_time"], t["end_time"])
        for t in turns
    ]
    labels = prosody.label_turns(feats, turns)
    assert labels[0]["energy_label"] == "quiet"
    assert labels[2]["energy_label"] == "loud"
    # And the raw RMS strictly orders the same way.
    assert labels[0]["rms"] < labels[1]["rms"] < labels[2]["rms"]


# ---------------------------------------------------------------------------
# Speech rate math is exact
# ---------------------------------------------------------------------------

def test_speech_rate_is_words_over_duration():
    feats = [{"rms": 0.1, "f0_median": None, "f0_std": None, "voiced_fraction": 0.0}]
    turns = [{"text": "one two three four", "start_time": 0.0, "end_time": 2.0}]
    labels = prosody.label_turns(feats, turns)
    assert labels[0]["speech_rate"] == 2.0  # 4 words / 2.0 s


def test_zero_duration_turn_rate_is_zero_not_crash():
    feats = [{"rms": 0.1, "f0_median": None, "f0_std": None, "voiced_fraction": 0.0}]
    turns = [{"text": "hi", "start_time": 5.0, "end_time": 5.0}]
    labels = prosody.label_turns(feats, turns)
    assert labels[0]["speech_rate"] == 0.0


def test_faster_turn_labelled_fast():
    """Same word count, shorter duration → higher rate → 'fast' tertile."""
    feats = [
        {"rms": 0.1, "f0_median": None, "f0_std": None, "voiced_fraction": 0.0}
        for _ in range(3)
    ]
    turns = [
        {"text": "a b c d", "start_time": 0.0, "end_time": 4.0},   # 1.0 wps slow
        {"text": "a b c d", "start_time": 4.0, "end_time": 6.0},   # 2.0 wps mid
        {"text": "a b c d", "start_time": 6.0, "end_time": 7.0},   # 4.0 wps fast
    ]
    labels = prosody.label_turns(feats, turns)
    assert labels[0]["rate_label"] == "slow"
    assert labels[2]["rate_label"] == "fast"


# ---------------------------------------------------------------------------
# Unvoiced turns get null pitch labels; annotate() stays honest
# ---------------------------------------------------------------------------

def test_unvoiced_turn_pitch_labels_null_and_annotation_unclear():
    rng = np.random.default_rng(1)
    voiced = _sine(200.0, 1.0, amp=0.4)
    noise = (0.3 * rng.standard_normal(SR)).astype(np.float32)
    pcm = np.concatenate([voiced, noise]).astype(np.float32)
    turns = [
        {"text": "hello there", "start_time": 0.0, "end_time": 1.0},
        {"text": "shh", "start_time": 1.0, "end_time": 2.0},
    ]
    feats = [
        prosody.turn_features(pcm, SR, t["start_time"], t["end_time"])
        for t in turns
    ]
    labels = prosody.label_turns(feats, turns)
    assert labels[1]["pitch_label"] is None
    assert labels[1]["pitch_var_label"] is None
    assert "pitch unclear" in prosody.annotate(labels[1])
    # The voiced turn does carry a pitch label.
    assert labels[0]["pitch_label"] is not None


def test_annotate_format():
    labels = {
        "energy_label": "loud", "rate_label": "fast",
        "pitch_var_label": "varied", "pitch_label": "high",
    }
    assert prosody.annotate(labels) == "loud, fast, pitch varied"


def test_label_vocab_matches_client_contract():
    """The exact label unions the mobile client types against — a mismatch
    here broke chip rendering once (review CRITICAL); lock the vocab."""
    from prosody import ENERGY_LABELS, PITCH_LABELS, RATE_LABELS

    assert ENERGY_LABELS == ("quiet", "normal", "loud")
    assert PITCH_LABELS == ("low", "mid", "high")
    assert RATE_LABELS == ("slow", "normal", "fast")


def test_degenerate_distribution_labels_middle():
    """All-equal values = nothing stands out = baseline (middle) label,
    so an evenly-delivered recording doesn't render noteworthy chips."""
    from prosody import _tertile_label

    assert _tertile_label(5.0, [5.0, 5.0, 5.0], ("a", "b", "c")) == "b"
    assert _tertile_label(5.0, [5.0], ("a", "b", "c")) == "b"


# ---------------------------------------------------------------------------
# Per-speaker pitch aggregation + relative voice labels (§2b)
# ---------------------------------------------------------------------------

def _vl(f0):
    """A minimal label_turns-shaped dict carrying just the f0_median the
    aggregator reads (None = an unvoiced turn)."""
    return {"f0_median": f0}


def test_speaker_median_pitch_ignores_unvoiced_and_orders_first_seen():
    speakers = ["Speaker B", "Speaker A", "Speaker B", "Speaker A"]
    labels = [_vl(120.0), _vl(220.0), _vl(None), _vl(200.0)]
    out = prosody.speaker_median_pitch(speakers, labels)
    assert list(out) == ["Speaker B", "Speaker A"]   # first-seen order
    assert out["Speaker B"] == 120.0                 # lone voiced turn
    assert out["Speaker A"] == 210.0                 # median(220, 200)


def test_speaker_median_pitch_all_unvoiced_is_none():
    out = prosody.speaker_median_pitch(["A", "B"], [_vl(None), _vl(None)])
    assert out == {"A": None, "B": None}


def test_pitch_voice_labels_meaningful_difference():
    """>15% apart → deeper/higher assigned to the correct speakers."""
    out = prosody.pitch_voice_labels({"A": 110.0, "B": 200.0})
    assert out == {"A": "Deeper voice", "B": "Higher voice"}
    # Order of the input dict must not change WHO is deeper.
    out2 = prosody.pitch_voice_labels({"B": 200.0, "A": 110.0})
    assert out2 == {"A": "Deeper voice", "B": "Higher voice"}


def test_pitch_voice_labels_never_emits_gender_words():
    out = prosody.pitch_voice_labels({"A": 110.0, "B": 200.0})
    joined = " ".join(out.values()).lower()
    for banned in ("male", "female", "man", "woman", "men", "women"):
        assert banned not in joined


def test_pitch_voice_labels_within_threshold_declines():
    """A near-tie (≤15%) is honestly left unlabeled."""
    assert prosody.pitch_voice_labels({"A": 200.0, "B": 210.0}) is None
    # Exactly at the 15% boundary is NOT "more than" the threshold → None.
    assert prosody.pitch_voice_labels({"A": 100.0, "B": 115.0}) is None
    # Just over the boundary IS labeled.
    assert prosody.pitch_voice_labels({"A": 100.0, "B": 116.0}) == {
        "A": "Deeper voice", "B": "Higher voice",
    }


def test_pitch_voice_labels_exact_tie_declines():
    assert prosody.pitch_voice_labels({"A": 180.0, "B": 180.0}) is None


def test_pitch_voice_labels_one_speaker_declines():
    assert prosody.pitch_voice_labels({"A": 180.0}) is None


def test_pitch_voice_labels_three_speakers_declines():
    assert prosody.pitch_voice_labels(
        {"A": 110.0, "B": 180.0, "C": 250.0}
    ) is None


def test_pitch_voice_labels_missing_pitch_declines():
    """Two speakers but one has no measured pitch → decline."""
    assert prosody.pitch_voice_labels({"A": 110.0, "B": None}) is None
