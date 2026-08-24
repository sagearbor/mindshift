"""Unit tests for server/tone_id.py (audio tone: backends + escalation).

Everything except the live tests runs WITHOUT torch: flag + backend parsing,
``surface_allowed``, the per-turn slicing/skipping, the result shaping, the
pure ``EscalationTracker`` / ``annotate_escalation`` layer, and
``classify_turns`` driven by an injected ``classify_fn``. The fake-model
test exercises the torch glue for the SpeechBrain path without a checkpoint
(needs torch, skipped honestly otherwise). The LIVE tests load the real
pinned checkpoints — ONLY when the pinned snapshot for that backend is
already in the local cache dir, so a CI/base install never downloads. What
they pin is set by the measured round-2 eval
(docs/research/tone-audio/2026-08-24-round2.md), not by hope: each test's
docstring says exactly which measured number it guards.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest

import tone_id

_AUDIO_DIR = Path(__file__).resolve().parent / "fixtures" / "audio"
SR = tone_id.TARGET_SR


# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------

def test_mode_default_is_dark_when_unset(monkeypatch):
    monkeypatch.delenv(tone_id.TONE_AUDIO_ENV, raising=False)
    assert tone_id.mode() == "dark"
    assert tone_id.DEFAULT_TONE_MODE == "dark"
    assert tone_id.is_enabled() is True
    assert tone_id.surface_allowed() is False


@pytest.mark.parametrize("raw, expected", [
    ("off", "off"), ("dark", "dark"), ("on", "on"),
    ("ON", "on"), ("  Off ", "off"), ("Dark", "dark"),
])
def test_mode_parses_case_and_whitespace(monkeypatch, raw, expected):
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, raw)
    assert tone_id.mode() == expected


def test_mode_empty_string_is_default(monkeypatch):
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "")
    assert tone_id.mode() == tone_id.DEFAULT_TONE_MODE


def test_mode_unknown_value_falls_back_to_default_and_warns(monkeypatch, caplog):
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "loud")
    with caplog.at_level(logging.WARNING, logger="tone_id"):
        assert tone_id.mode() == tone_id.DEFAULT_TONE_MODE
    assert any("loud" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("raw, enabled, surface", [
    ("off", False, False),
    ("dark", True, False),
    ("on", True, True),
])
def test_enabled_and_surface_allowed_per_mode(monkeypatch, raw, enabled, surface):
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, raw)
    assert tone_id.is_enabled() is enabled
    assert tone_id.surface_allowed() is surface


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def test_backend_default_is_odyssey_dim(monkeypatch):
    monkeypatch.delenv(tone_id.TONE_BACKEND_ENV, raising=False)
    assert tone_id.backend() == "odyssey_dim" == tone_id.DEFAULT_TONE_BACKEND
    assert tone_id.is_dimensional() is True
    assert tone_id.escalation_threshold() == tone_id.ESCALATION_DELTA_THRESHOLD["odyssey_dim"]


@pytest.mark.parametrize("raw, expected", [
    ("iemocap", "iemocap"), ("SUPERB_ER", "superb_er"), ("  odyssey_dim ", "odyssey_dim"),
])
def test_backend_parses_case_and_whitespace(monkeypatch, raw, expected):
    monkeypatch.setenv(tone_id.TONE_BACKEND_ENV, raw)
    assert tone_id.backend() == expected
    assert tone_id.model_id() == f"{tone_id.BACKEND_INFO[expected]['source']}@{tone_id.BACKEND_INFO[expected]['revision']}"


def test_backend_unknown_falls_back_and_warns(monkeypatch, caplog):
    monkeypatch.setenv(tone_id.TONE_BACKEND_ENV, "whisper")
    with caplog.at_level(logging.WARNING, logger="tone_id"):
        assert tone_id.backend() == tone_id.DEFAULT_TONE_BACKEND
    assert any("whisper" in r.getMessage() for r in caplog.records)
    monkeypatch.setenv(tone_id.TONE_BACKEND_ENV, "")
    assert tone_id.backend() == tone_id.DEFAULT_TONE_BACKEND


def test_every_backend_is_pinned_and_thresholded():
    for name in tone_id.TONE_BACKENDS:
        info = tone_id.BACKEND_INFO[name]
        assert len(info["revision"]) == 40, f"{name} revision must be a full commit sha"
        assert info["kind"] in ("categorical", "dimensional")
        assert (name in tone_id.DIMENSIONAL_BACKENDS) == (info["kind"] == "dimensional")
        assert tone_id.ESCALATION_DELTA_THRESHOLD[name] > 0
        assert "@" in tone_id.model_id(name)
    with pytest.raises(tone_id.ToneUnavailable):
        tone_id.model_id("msp_dim")  # research-only (CC-BY-NC) model is NOT a backend
    assert tone_id.is_available("msp_dim") is False


def test_snapshot_present_per_backend(tmp_path):
    d = str(tmp_path)
    assert not tone_id.snapshot_present(d, "odyssey_dim")
    assert not tone_id.snapshot_present(d, "superb_er")
    assert not tone_id.snapshot_present(d, "iemocap")
    (tmp_path / "odyssey-dim").mkdir()
    (tmp_path / "odyssey-dim" / "config.json").write_text("{}")
    (tmp_path / "odyssey-dim" / "model.safetensors").write_bytes(b"x")
    assert not tone_id.snapshot_present(d, "odyssey_dim")  # backbone config still missing
    (tmp_path / "wavlm-large").mkdir()
    (tmp_path / "wavlm-large" / "config.json").write_text("{}")
    assert tone_id.snapshot_present(d, "odyssey_dim")
    (tmp_path / "superb-er").mkdir()
    (tmp_path / "superb-er" / "config.json").write_text("{}")
    (tmp_path / "superb-er" / "pytorch_model.bin").write_bytes(b"x")
    assert tone_id.snapshot_present(d, "superb_er")
    assert not tone_id.snapshot_present(d, "iemocap")


# ---------------------------------------------------------------------------
# Off / unavailable paths never touch a model
# ---------------------------------------------------------------------------

def _never_load(*_a, **_k):
    raise AssertionError("model load must not be attempted")


def test_classify_pcm_off_mode_raises_without_loading(monkeypatch):
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "off")
    monkeypatch.setattr(tone_id, "_load_model", _never_load)
    with pytest.raises(tone_id.ToneUnavailable):
        tone_id.classify_pcm(np.zeros(SR, dtype=np.float32), SR)


def test_load_model_off_mode_raises(monkeypatch):
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "off")
    with pytest.raises(tone_id.ToneUnavailable):
        tone_id._load_model()


def test_load_model_unavailable_deps_is_honest(monkeypatch):
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    monkeypatch.setattr(tone_id, "is_available", lambda name=None: False)
    monkeypatch.setattr(tone_id, "_models", {})
    with pytest.raises(tone_id.ToneUnavailable, match="requirements-voice.txt"):
        tone_id._load_model("odyssey_dim")


def test_classify_pcm_wrong_sample_rate_is_honest_error(monkeypatch):
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    monkeypatch.setattr(tone_id, "_load_model", _never_load)
    with pytest.raises(tone_id.ToneUnavailable, match="16000 Hz"):
        tone_id.classify_pcm(np.zeros(24000, dtype=np.float32), 24000)


def test_classify_pcm_empty_audio_is_honest_error(monkeypatch):
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    monkeypatch.setattr(tone_id, "_load_model", _never_load)
    with pytest.raises(tone_id.ToneUnavailable, match="zero-length"):
        tone_id.classify_pcm(np.zeros(0, dtype=np.float32), SR)


def test_unavailable_propagates_from_classify_turns(monkeypatch):
    """A ToneUnavailable from the model call is NOT swallowed per turn — the
    caller decides what "no tone" means for the whole recording."""
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")

    def boom(_pcm, _sr):
        raise tone_id.ToneUnavailable("deps missing")

    pcm = np.ones(SR * 3, dtype=np.float32) * 0.1
    with pytest.raises(tone_id.ToneUnavailable):
        tone_id.classify_turns(
            pcm, SR, [{"speaker": "A", "start_time": 0.0, "end_time": 2.0}],
            classify_fn=boom,
        )


def test_is_available_false_when_import_fails(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("no transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert tone_id.is_available() is False
    assert tone_id.is_available("iemocap") is False


# ---------------------------------------------------------------------------
# Result shaping — pure
# ---------------------------------------------------------------------------

def test_arousal_margin_is_angry_minus_best_other():
    assert tone_id.arousal_margin({"neutral": 1.0, "angry": 3.0, "happy": 2.0, "sad": -1.0}) == pytest.approx(1.0)
    assert tone_id.arousal_margin({"neutral": 5.0, "angry": 3.0, "happy": 2.0, "sad": -1.0}) == pytest.approx(-2.0)
    assert tone_id.arousal_margin({"angry": 2.0}) == pytest.approx(2.0)


def test_probs_to_result_schema_and_rounding(monkeypatch):
    monkeypatch.setenv(tone_id.TONE_BACKEND_ENV, "iemocap")
    r = tone_id._probs_to_result(np.array([0.1, 0.6, 0.2, 0.1]))
    assert r["label"] == "angry"
    assert r["confidence"] == pytest.approx(0.6)
    assert set(r["scores"]) == set(tone_id.LABELS)
    assert sum(r["scores"].values()) == pytest.approx(1.0, abs=1e-3)
    assert r["model"] == tone_id.model_id() and r["backend"] == "iemocap" and r["kind"] == "categorical"
    # without explicit logits the margin comes from log-probs: log .6 - log .2
    assert r["arousal"] == pytest.approx(math.log(0.6) - math.log(0.2), abs=1e-3)
    assert set(r["logits"]) == set(tone_id.LABELS)
    r2 = tone_id._probs_to_result(np.array([0.1, 0.6, 0.2, 0.1]), logits=np.array([0.0, 4.0, 1.0, -2.0]))
    assert r2["arousal"] == pytest.approx(3.0)
    with pytest.raises(tone_id.ToneUnavailable):
        tone_id._probs_to_result(np.array([0.5, 0.5]))


def test_dims_to_result_schema(monkeypatch):
    monkeypatch.setenv(tone_id.TONE_BACKEND_ENV, "odyssey_dim")
    r = tone_id._dims_to_result({"arousal": 0.71234, "dominance": 0.5, "valence": 0.31})
    assert r["kind"] == "dimensional" and r["backend"] == "odyssey_dim"
    assert r["label"] == tone_id.UNSCORED_LABEL and r["confidence"] == 0.0
    assert r["scores"] == {"arousal": 0.7123, "dominance": 0.5, "valence": 0.31}
    assert r["arousal"] == pytest.approx(0.7123)
    with pytest.raises(tone_id.ToneUnavailable):
        tone_id._dims_to_result({"arousal": 0.5})
    with pytest.raises(tone_id.ToneUnavailable):
        tone_id._dims_to_result({"arousal": "x", "dominance": 0.5, "valence": 0.3})


# ---------------------------------------------------------------------------
# Escalation tracker — pure
# ---------------------------------------------------------------------------

def test_tracker_first_turn_has_no_baseline_then_median_of_previous():
    tr = tone_id.EscalationTracker()
    assert tr.baseline("A") is None and tr.history("A") == 0
    assert tr.observe("A", 0.40) == {"delta": None, "baseline": None, "history": 0}
    assert tr.observe("A", 0.50) == {"delta": pytest.approx(0.10), "baseline": 0.40, "history": 1}
    # baseline is the median of the PREVIOUS turns only (the turn never baselines itself)
    obs = tr.observe("A", 0.30)
    assert obs["baseline"] == pytest.approx(0.45) and obs["delta"] == pytest.approx(-0.15) and obs["history"] == 2
    assert tr.history("A") == 3


def test_tracker_keeps_speakers_apart():
    tr = tone_id.EscalationTracker()
    tr.observe("A", 0.9)
    tr.observe("A", 0.9)
    # B's first turn is unscored even though A has a baseline — never compare voices
    assert tr.observe("B", 0.2)["delta"] is None
    assert tr.observe("B", 0.3)["delta"] == pytest.approx(0.1)
    assert tr.baseline("A") == pytest.approx(0.9)


def test_tracker_max_history_window():
    tr = tone_id.EscalationTracker(max_history=2)
    for v in (0.1, 0.2, 0.9, 0.9):
        tr.observe("A", v)
    assert tr.baseline("A") == pytest.approx(0.9)  # only the last two count
    assert tone_id.EscalationTracker(max_history=None).max_history is None
    with pytest.raises(ValueError):
        tone_id.EscalationTracker(max_history=0)


def test_escalation_confidence_squash():
    assert tone_id.escalation_confidence(0.03, 0.03) == pytest.approx(0.5)
    assert tone_id.escalation_confidence(0.06, 0.03) == pytest.approx(1.0)
    assert tone_id.escalation_confidence(0.09, 0.03) == 1.0
    assert tone_id.escalation_confidence(0.0, 0.03) == 0.0
    assert tone_id.escalation_confidence(1.0, 0.0) == 1.0


def test_annotate_escalation_dimensional_unscored_steady_escalating(monkeypatch):
    monkeypatch.setenv(tone_id.TONE_BACKEND_ENV, "odyssey_dim")
    tr = tone_id.EscalationTracker()
    t = tone_id.escalation_threshold("odyssey_dim")
    r0 = tone_id.annotate_escalation(tone_id._dims_to_result({"arousal": 0.40, "dominance": 0.5, "valence": 0.5}), "A", tr)
    assert r0["label"] == tone_id.UNSCORED_LABEL and r0["confidence"] == 0.0
    assert r0["escalation"] == {"delta": None, "baseline": None, "history": 0, "flag": False, "threshold": t}
    r1 = tone_id.annotate_escalation(tone_id._dims_to_result({"arousal": 0.41, "dominance": 0.5, "valence": 0.5}), "A", tr)
    assert r1["label"] == tone_id.STEADY_LABEL and r1["escalation"]["flag"] is False
    assert r1["escalation"]["delta"] == pytest.approx(0.01)
    r2 = tone_id.annotate_escalation(tone_id._dims_to_result({"arousal": 0.40 + 2 * t + 0.005, "dominance": 0.5, "valence": 0.5}), "A", tr)
    assert r2["label"] == tone_id.ESCALATION_LABEL and r2["escalation"]["flag"] is True
    assert r2["escalation"]["history"] == 2 and r2["confidence"] == pytest.approx(1.0, abs=0.05)
    # scores are untouched by the annotation
    assert set(r2["scores"]) == set(tone_id.DIMS)


def test_annotate_escalation_categorical_keeps_model_label_unless_flagged(monkeypatch):
    monkeypatch.setenv(tone_id.TONE_BACKEND_ENV, "superb_er")
    tr = tone_id.EscalationTracker()
    base = tone_id._probs_to_result(np.array([0.7, 0.1, 0.1, 0.1]), logits=np.array([2.0, 0.0, 0.0, 0.0]))
    r0 = tone_id.annotate_escalation(dict(base), "A", tr)
    assert r0["label"] == "neutral" and r0["confidence"] == pytest.approx(0.7)
    assert r0["escalation"]["delta"] is None and r0["escalation"]["threshold"] == tone_id.ESCALATION_DELTA_THRESHOLD["superb_er"]
    hot = tone_id._probs_to_result(np.array([0.4, 0.3, 0.2, 0.1]), logits=np.array([1.0, 0.7, 0.3, 0.0]))
    r1 = tone_id.annotate_escalation(dict(hot), "A", tr)  # margin -0.3 vs baseline -2.0 → +1.7 ≥ 0.57
    assert r1["label"] == tone_id.ESCALATION_LABEL and r1["escalation"]["flag"] is True
    assert r1["confidence"] == pytest.approx(tone_id.escalation_confidence(1.7, 0.57), abs=1e-3)
    # an explicit threshold overrides the backend's pinned one
    r2 = tone_id.annotate_escalation(dict(hot), "A", tr, threshold=10.0)
    assert r2["escalation"]["flag"] is False and r2["label"] == "neutral"


# ---------------------------------------------------------------------------
# Slicing / skipping / escalation with an injected classify_fn (no model)
# ---------------------------------------------------------------------------

def _fake_classify(label="neutral", arousal=None):
    """A classify_fn that records the slice lengths it was handed. With
    ``arousal`` a list, each call pops the next value (dimensional shape)."""
    seen: list[int] = []
    queue = list(arousal) if arousal is not None else None

    def fn(samples, sr):
        assert sr == SR
        assert samples.dtype == np.float32
        seen.append(int(samples.size))
        if queue is not None:
            a = queue.pop(0)
            return {"label": tone_id.UNSCORED_LABEL, "scores": {"arousal": a, "dominance": 0.5, "valence": 0.5},
                    "confidence": 0.0, "arousal": a, "kind": "dimensional", "backend": "odyssey_dim", "model": "fake"}
        scores = {lbl: 0.0 for lbl in tone_id.LABELS}
        scores[label] = 1.0
        return {"label": label, "scores": scores, "confidence": 1.0, "model": "fake"}

    fn.seen = seen
    return fn


def test_slice_turn_clamps_to_bounds_and_reports_truncation():
    pcm = np.arange(SR * 5, dtype=np.float32)
    samples, trunc = tone_id.slice_turn(pcm, SR, {"start_time": 1.0, "end_time": 2.5})
    assert samples.size == int(1.5 * SR) and trunc is False
    assert samples[0] == pytest.approx(SR * 1.0)
    samples, _ = tone_id.slice_turn(pcm, SR, {"start_time": 4.0, "end_time": 9.0})
    assert samples.size == SR * 1
    assert tone_id.slice_turn(pcm, SR, {"start_time": 3.0, "end_time": 2.0})[0].size == 0
    assert tone_id.slice_turn(pcm, SR, {"start_time": None, "end_time": 2.0})[0].size == 0
    assert tone_id.slice_turn(np.zeros(0, np.float32), SR, {"start_time": 0, "end_time": 1})[0].size == 0
    long_pcm = np.zeros(int(SR * (tone_id.MAX_TURN_SECONDS + 5)), dtype=np.float32)
    samples, trunc = tone_id.slice_turn(
        long_pcm, SR, {"start_time": 0.0, "end_time": tone_id.MAX_TURN_SECONDS + 5},
    )
    assert trunc is True
    assert samples.size == int(tone_id.MAX_TURN_SECONDS * SR)


def test_classify_turns_one_entry_per_turn_with_honest_skips(monkeypatch):
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    pcm = np.ones(SR * 10, dtype=np.float32) * 0.1
    turns = [
        {"speaker": "A", "start_time": 0.0, "end_time": 2.0},      # scored
        {"speaker": "B", "start_time": 2.0, "end_time": 2.5},      # too short
        {"speaker": "A", "start_time": None, "end_time": 4.0},     # no audio
        {"speaker": "B", "start_time": 5.0, "end_time": 5.999},    # just under 1s
        {"speaker": "A", "start_time": 6.0, "end_time": 7.0},      # exactly 1s → scored
        {"speaker": "B", "start_time": 9.0, "end_time": 20.0},     # clamped → 1s scored
    ]
    fn = _fake_classify("angry")
    out = tone_id.classify_turns(pcm, SR, turns, classify_fn=fn)

    assert [r["index"] for r in out] == [0, 1, 2, 3, 4, 5]
    assert [r["speaker"] for r in out] == ["A", "B", "A", "B", "A", "B"]
    assert [r["skipped"] for r in out] == [None, "too_short", "no_audio", "too_short", None, None]
    assert [r["tone"] is not None for r in out] == [True, False, False, False, True, True]
    assert out[0]["tone"]["label"] == "angry"
    # a classify_fn result WITHOUT an arousal number gets no escalation entry
    assert "escalation" not in out[0]["tone"]
    assert out[0]["seconds"] == pytest.approx(2.0)
    assert out[1]["seconds"] == pytest.approx(0.5)
    assert out[5]["seconds"] == pytest.approx(1.0)
    assert fn.seen == [SR * 2, SR * 1, SR * 1]
    assert all(r["latency_ms"] >= 0.0 for r in out)
    assert out[1]["latency_ms"] == 0.0 and out[2]["latency_ms"] == 0.0
    assert all(r["truncated"] is False for r in out)


def test_classify_turns_runs_escalation_causally_per_speaker(monkeypatch):
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    monkeypatch.setenv(tone_id.TONE_BACKEND_ENV, "odyssey_dim")
    pcm = np.ones(SR * 12, dtype=np.float32) * 0.1
    turns = [
        {"speaker": "A", "start_time": 0.0, "end_time": 2.0},   # A #1  0.40 unscored
        {"speaker": "B", "start_time": 2.0, "end_time": 4.0},   # B #1  0.60 unscored
        {"speaker": "A", "start_time": 4.0, "end_time": 4.5},   # too short — must NOT touch A's baseline
        {"speaker": "A", "start_time": 5.0, "end_time": 7.0},   # A #2  0.41 steady (+0.01)
        {"speaker": "B", "start_time": 7.0, "end_time": 9.0},   # B #2  0.59 steady (-0.01)
        {"speaker": "A", "start_time": 9.0, "end_time": 11.0},  # A #3  0.55 escalating (+0.145 vs median .405)
    ]
    fn = _fake_classify(arousal=[0.40, 0.60, 0.41, 0.59, 0.55])
    out = tone_id.classify_turns(pcm, SR, turns, classify_fn=fn)
    labels = [r["tone"]["label"] if r["tone"] else r["skipped"] for r in out]
    assert labels == ["unscored", "unscored", "too_short", "steady", "steady", "escalating"]
    esc = out[5]["tone"]["escalation"]
    assert esc["flag"] is True and esc["history"] == 2 and esc["baseline"] == pytest.approx(0.405)
    assert esc["delta"] == pytest.approx(0.145) and esc["threshold"] == tone_id.ESCALATION_DELTA_THRESHOLD["odyssey_dim"]
    assert out[3]["tone"]["escalation"]["history"] == 1  # the skipped turn didn't count
    assert tone_id.label_distribution(out) == {
        "neutral": 0, "angry": 0, "happy": 0, "sad": 0, "skipped": 1,
        "unscored": 2, "steady": 2, "escalating": 1,
    }


def test_classify_turns_tracker_continues_across_calls_and_threshold_override(monkeypatch):
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    monkeypatch.setenv(tone_id.TONE_BACKEND_ENV, "odyssey_dim")
    pcm = np.ones(SR * 4, dtype=np.float32) * 0.1
    tr = tone_id.EscalationTracker()
    one = [{"speaker": "A", "start_time": 0.0, "end_time": 2.0}]
    first = tone_id.classify_turns(pcm, SR, one, classify_fn=_fake_classify(arousal=[0.40]), tracker=tr)
    assert first[0]["tone"]["label"] == "unscored"
    second = tone_id.classify_turns(pcm, SR, one, classify_fn=_fake_classify(arousal=[0.50]), tracker=tr)
    assert second[0]["tone"]["label"] == "escalating" and second[0]["tone"]["escalation"]["history"] == 1
    third = tone_id.classify_turns(pcm, SR, one, classify_fn=_fake_classify(arousal=[0.55]), tracker=tr, threshold=0.5)
    assert third[0]["tone"]["label"] == "steady" and third[0]["tone"]["escalation"]["threshold"] == 0.5


def test_classify_turns_min_seconds_override(monkeypatch):
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    pcm = np.ones(SR * 4, dtype=np.float32) * 0.1
    turns = [{"speaker": "A", "start_time": 0.0, "end_time": 0.5}]
    fn = _fake_classify()
    out = tone_id.classify_turns(pcm, SR, turns, classify_fn=fn, min_seconds=0.25)
    assert out[0]["skipped"] is None and fn.seen == [SR // 2]


def test_classify_turns_empty_inputs(monkeypatch):
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    fn = _fake_classify()
    assert tone_id.classify_turns(np.zeros(SR, np.float32), SR, [], classify_fn=fn) == []
    out = tone_id.classify_turns(
        np.zeros(0, np.float32), SR,
        [{"speaker": "A", "start_time": 0.0, "end_time": 2.0}], classify_fn=fn,
    )
    assert out[0]["skipped"] == "no_audio" and fn.seen == []


def test_classify_turns_logs_distribution_in_dark_mode(monkeypatch, caplog):
    """Dark mode's entire output IS the log line — make sure it exists and
    names the mode and backend, so an operator can grep for it."""
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    monkeypatch.setenv(tone_id.TONE_BACKEND_ENV, "superb_er")
    pcm = np.ones(SR * 4, dtype=np.float32) * 0.1
    turns = [
        {"speaker": "A", "start_time": 0.0, "end_time": 2.0},
        {"speaker": "B", "start_time": 2.0, "end_time": 2.2},
    ]
    with caplog.at_level(logging.INFO, logger="tone_id"):
        out = tone_id.classify_turns(pcm, SR, turns, classify_fn=_fake_classify("sad"))
    msgs = [r.getMessage() for r in caplog.records]
    assert any("dark mode" in m and "superb_er backend" in m and "'sad': 1" in m and "'skipped': 1" in m for m in msgs)
    assert tone_id.label_distribution(out) == {
        "neutral": 0, "angry": 0, "happy": 0, "sad": 1, "skipped": 1,
    }


def test_live_sessions_treats_escalating_as_an_escalation_label():
    import live_sessions

    assert tone_id.ESCALATION_LABEL in live_sessions.ESCALATION_LABELS
    assert live_sessions.is_escalated({"label": tone_id.ESCALATION_LABEL}) is True
    assert live_sessions.is_escalated({"label": tone_id.STEADY_LABEL}) is False
    assert live_sessions.is_escalated({"label": tone_id.UNSCORED_LABEL}) is False


# ---------------------------------------------------------------------------
# Fake-model path: exercises the SpeechBrain torch glue without the checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not tone_id.is_available("iemocap"), reason="torch/speechbrain/transformers absent")
def test_classify_pcm_with_injected_fake_iemocap_model(monkeypatch):
    import torch

    class _Mods:
        def output_mlp(self, emb):
            return torch.tensor([[0.0, 0.0, 3.0, 0.0]]) + 0 * emb.sum()  # neu, ang, hap, sad

    class _FakeModel:
        def __init__(self):
            self.calls = []
            self.mods = _Mods()

        def encode_batch(self, wavs, wav_lens=None):
            self.calls.append(tuple(wavs.shape))
            return torch.zeros(1, 4)

    fake = _FakeModel()
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    monkeypatch.setenv(tone_id.TONE_BACKEND_ENV, "iemocap")
    monkeypatch.setattr(tone_id, "_models", {"iemocap": fake})
    r = tone_id.classify_pcm(np.zeros(SR * 2, dtype=np.float32), SR)
    assert r["label"] == "happy" and r["backend"] == "iemocap" and r["kind"] == "categorical"
    assert r["confidence"] == pytest.approx(math.exp(3) / (math.exp(3) + 3), abs=1e-3)
    assert r["arousal"] == pytest.approx(-3.0)  # angry logit 0 minus best other 3
    assert fake.calls == [(1, SR * 2)]


# ---------------------------------------------------------------------------
# Live: the real pinned checkpoints on real fixture turns
# ---------------------------------------------------------------------------

def _fixture_turns(name: str) -> tuple[np.ndarray, int, list[dict]]:
    """16 kHz PCM + reconstructed turns (duration_sec + silence_gap_sec, exactly
    as test_diarize_regression_ladder._build_turns does) with the scripted /
    coarse labels attached."""
    import audio_ingest

    wav = _AUDIO_DIR / f"test_recording_{name}.wav"
    meta = json.loads((_AUDIO_DIR / f"test_recording_{name}_meta.json").read_text())
    pcm, sr = audio_ingest.decode_to_pcm_16k(wav.read_bytes(), wav.name)
    turns, t = [], 0.0
    for m in meta["turns"]:
        turns.append({
            "speaker": m["speaker"], "start_time": round(t, 4), "end_time": round(t + m["duration_sec"], 4),
            "scripted_emotion": m["scripted_emotion"], "coarse": m.get("emotion_coarse"),
        })
        t += m["duration_sec"] + meta["silence_gap_sec"]
    return pcm, sr, turns


def _live(name: str):
    return pytest.mark.skipif(
        not tone_id.is_available(name) or not tone_id.snapshot_present(name=name),
        reason=f"{name} deps absent or pinned snapshot not in local cache (no download in tests)",
    )


@_live("iemocap")
def test_live_iemocap_shout_angry_turn_schema_and_label(monkeypatch):
    """Round-1 pin, unchanged: the SpeechBrain checkpoint still loads and still
    calls the gptaudio shout_angry slice 'angry' (p~1.00 on both acted
    fixtures). NOT evidence it reads emotion — round 1 found it calls every
    turn of that voice 'angry' — only a drift guard on the pinned revision."""
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    monkeypatch.setenv(tone_id.TONE_BACKEND_ENV, "iemocap")
    pcm, sr, turns = _fixture_turns("gptaudio")
    turn = next(t for t in turns if t["scripted_emotion"] == "shout_angry")
    out = tone_id.classify_turns(pcm, sr, [turn])
    assert len(out) == 1 and out[0]["skipped"] is None
    tone = out[0]["tone"]
    assert set(tone["scores"]) == set(tone_id.LABELS)
    assert sum(tone["scores"].values()) == pytest.approx(1.0, abs=1e-2)
    assert tone["model"] == tone_id.model_id("iemocap") and tone["kind"] == "categorical"
    assert isinstance(tone["arousal"], float) and out[0]["latency_ms"] > 0
    assert tone["escalation"]["delta"] is None  # a lone turn has no baseline
    assert tone["label"] == "angry"


@_live("odyssey_dim")
def test_live_odyssey_couple_escalation_timeline(monkeypatch):
    """The DEFAULT backend on the canonical self-escalation scene, pinned at
    the round-2 measurement: self (Speaker A) is flagged 'escalating' on
    exactly its three angry turns (4 tense_rising, 6 shout_angry, 8
    cold_contempt) and on none of its calm/repair/warm turns — the expected
    nudge timeline, 3 hit / 0 miss / 0 false. The partner's pleading
    hurt_sad (turn 7) is ALSO flagged (its arousal jumps): that is the one
    measured false alarm on this scene and is pinned so a change is noticed,
    not so it is celebrated. The first turn of each voice is 'unscored'."""
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    monkeypatch.setenv(tone_id.TONE_BACKEND_ENV, "odyssey_dim")
    pcm, sr, turns = _fixture_turns("scene_couple_escalation")
    out = tone_id.classify_turns(pcm, sr, turns)
    assert all(r["skipped"] is None for r in out)
    tones = [r["tone"] for r in out]
    assert all(t["kind"] == "dimensional" and set(t["scores"]) == set(tone_id.DIMS) for t in tones)
    assert all(0.0 <= t["scores"]["arousal"] <= 1.0 for t in tones)
    assert tones[0]["label"] == tones[1]["label"] == tone_id.UNSCORED_LABEL
    flagged = {r["index"] for r in out if r["tone"]["escalation"]["flag"]}
    self_angry = {i for i, t_ in enumerate(turns) if t_["speaker"] == "Speaker A" and t_["coarse"] == "angry"}
    assert self_angry == {4, 6, 8}
    assert flagged & {i for i, t_ in enumerate(turns) if t_["speaker"] == "Speaker A"} == self_angry
    assert flagged == {4, 5, 6, 7, 8}
    assert out[6]["tone"]["confidence"] == 1.0  # the shout: delta ~0.11 ≥ 2× threshold
    assert all(r["latency_ms"] > 0 for r in out)


@_live("superb_er")
def test_live_superb_er_schema_and_measured_flags(monkeypatch):
    """The Apache 'fits 2 GiB' backend on the same scene: schema + the
    measured flag set. Round 2 measured it noisier than odyssey_dim (delta
    AUC 0.75 vs 0.81; here it flags calm turn 2 and misses the shout at 6),
    which is exactly why it is not the default — pinned as a drift guard."""
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    monkeypatch.setenv(tone_id.TONE_BACKEND_ENV, "superb_er")
    pcm, sr, turns = _fixture_turns("scene_couple_escalation")
    out = tone_id.classify_turns(pcm, sr, turns)
    tones = [r["tone"] for r in out]
    assert all(t["kind"] == "categorical" and set(t["scores"]) == set(tone_id.LABELS) for t in tones)
    assert all(abs(sum(t["scores"].values()) - 1.0) < 1e-2 for t in tones)
    assert tones[0]["escalation"]["delta"] is None and tones[0]["label"] in tone_id.LABELS
    flagged = {r["index"] for r in out if r["tone"]["escalation"]["flag"]}
    assert flagged == {2, 4, 5, 7, 8, 12}
