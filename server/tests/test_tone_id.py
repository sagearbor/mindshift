"""Unit tests for server/tone_id.py (audio tone classification).

Everything here except the last two tests runs WITHOUT torch/speechbrain:
the flag parsing, ``surface_allowed``, and the per-turn slicing/skipping
logic are pure and are exercised with an injected ``classify_fn``. The
softmax-to-dict conversion is tested against a fake model object injected
via ``tone_id._model`` (needs torch, skipped honestly otherwise). The final
test is LIVE: it loads the real pinned checkpoint — but ONLY when both pinned
snapshots are already in the local cache dir, so a CI/base install never
triggers a ~750MB download. What it asserts about the label is set by the
measured eval (docs/research/tone-audio/2026-08-24-wav2vec2-iemocap-eval.md),
not by hope: see the test's own docstring.
"""
from __future__ import annotations

import json
import logging
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
# Off / unavailable paths never touch the model
# ---------------------------------------------------------------------------

def _never_load():
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


# ---------------------------------------------------------------------------
# Slicing / skipping with an injected classify_fn (no model)
# ---------------------------------------------------------------------------

def _fake_classify(label="neutral"):
    """A classify_fn that records the slice lengths it was handed."""
    seen: list[int] = []

    def fn(samples, sr):
        assert sr == SR
        assert samples.dtype == np.float32
        seen.append(int(samples.size))
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
    # end beyond the clip → clamped
    samples, _ = tone_id.slice_turn(pcm, SR, {"start_time": 4.0, "end_time": 9.0})
    assert samples.size == SR * 1
    # inverted / missing / empty → no audio
    assert tone_id.slice_turn(pcm, SR, {"start_time": 3.0, "end_time": 2.0})[0].size == 0
    assert tone_id.slice_turn(pcm, SR, {"start_time": None, "end_time": 2.0})[0].size == 0
    assert tone_id.slice_turn(np.zeros(0, np.float32), SR, {"start_time": 0, "end_time": 1})[0].size == 0
    # longer than the cap → first MAX_TURN_SECONDS, flagged
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
    assert [r["skipped"] for r in out] == [
        None, "too_short", "no_audio", "too_short", None, None,
    ]
    assert [r["tone"] is not None for r in out] == [True, False, False, False, True, True]
    assert out[0]["tone"]["label"] == "angry"
    assert out[0]["seconds"] == pytest.approx(2.0)
    assert out[1]["seconds"] == pytest.approx(0.5)
    assert out[5]["seconds"] == pytest.approx(1.0)
    # the model saw exactly the scored slices, in order
    assert fn.seen == [SR * 2, SR * 1, SR * 1]
    # latency is measured for scored turns, 0 for skipped
    assert all(r["latency_ms"] >= 0.0 for r in out)
    assert out[1]["latency_ms"] == 0.0 and out[2]["latency_ms"] == 0.0
    assert all(r["truncated"] is False for r in out)


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
    names the mode, so an operator can grep for it."""
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    pcm = np.ones(SR * 4, dtype=np.float32) * 0.1
    turns = [
        {"speaker": "A", "start_time": 0.0, "end_time": 2.0},
        {"speaker": "B", "start_time": 2.0, "end_time": 2.2},
    ]
    with caplog.at_level(logging.INFO, logger="tone_id"):
        out = tone_id.classify_turns(pcm, SR, turns, classify_fn=_fake_classify("sad"))
    msgs = [r.getMessage() for r in caplog.records]
    assert any("dark mode" in m and "'sad': 1" in m and "'skipped': 1" in m for m in msgs)
    assert tone_id.label_distribution(out) == {
        "neutral": 0, "angry": 0, "happy": 0, "sad": 1, "skipped": 1,
    }


def test_probs_to_result_schema_and_rounding():
    r = tone_id._probs_to_result(np.array([0.1, 0.6, 0.2, 0.1]))
    assert r["label"] == "angry"
    assert r["confidence"] == pytest.approx(0.6)
    assert set(r["scores"]) == set(tone_id.LABELS)
    assert sum(r["scores"].values()) == pytest.approx(1.0, abs=1e-3)
    assert r["model"] == tone_id.model_id()
    with pytest.raises(tone_id.ToneUnavailable):
        tone_id._probs_to_result(np.array([0.5, 0.5]))


# ---------------------------------------------------------------------------
# Fake-model path: exercises the torch glue without the checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not tone_id.is_available(), reason="torch/speechbrain/transformers absent")
def test_classify_pcm_with_injected_fake_model(monkeypatch):
    import torch

    class _FakeModel:
        def __init__(self):
            self.calls = []

        def classify_batch(self, wavs, wav_lens=None):
            self.calls.append(tuple(wavs.shape))
            probs = torch.tensor([[0.05, 0.05, 0.85, 0.05]])  # neu, ang, hap, sad
            return probs, probs.max(dim=-1).values, torch.tensor([2]), ["hap"]

    fake = _FakeModel()
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    monkeypatch.setattr(tone_id, "_model", fake)
    r = tone_id.classify_pcm(np.zeros(SR * 2, dtype=np.float32), SR)
    assert r["label"] == "happy" and r["confidence"] == pytest.approx(0.85)
    assert r["scores"]["happy"] == pytest.approx(0.85)
    assert fake.calls == [(1, SR * 2)]


# ---------------------------------------------------------------------------
# Live: the real pinned checkpoint on a real scripted turn
# ---------------------------------------------------------------------------

def _gptaudio_turn(emotion: str) -> tuple[np.ndarray, int, dict]:
    """The 16 kHz PCM + reconstructed turn for one scripted emotion in the
    gptaudio fixture (turn times from duration_sec + silence_gap_sec, exactly
    as test_diarize_regression_ladder._build_turns does)."""
    import audio_ingest

    wav = _AUDIO_DIR / "test_recording_gptaudio.wav"
    meta = json.loads((_AUDIO_DIR / "test_recording_gptaudio_meta.json").read_text())
    pcm, sr = audio_ingest.decode_to_pcm_16k(wav.read_bytes(), wav.name)
    t = 0.0
    for m in meta["turns"]:
        turn = {
            "speaker": m["speaker"], "start_time": round(t, 4),
            "end_time": round(t + m["duration_sec"], 4),
        }
        if m["scripted_emotion"] == emotion:
            return pcm, sr, turn
        t += m["duration_sec"] + meta["silence_gap_sec"]
    raise AssertionError(f"no {emotion} turn in fixture")


@pytest.mark.skipif(
    not tone_id.is_available() or not tone_id.snapshot_present(),
    reason="tone deps absent or pinned snapshot not in local cache (no download in tests)",
)
def test_live_shout_angry_turn_schema_and_label(monkeypatch):
    """Real pinned model, real fixture turn — schema first, then ONE label.

    Why the label assertion is safe to pin, honestly: the eval (see module
    docstring) measured shout_angry -> angry at p~1.00 on BOTH acted
    fixtures, deterministic at the pinned revision. It is NOT evidence the
    model reads emotion: the same eval found it calls EVERY turn of this
    fixture's Speaker A voice "angry" (calm_open included) and every Speaker
    B turn "neutral" — a per-voice bias, which is exactly why the default
    mode is dark. So this test guards "the pinned checkpoint still loads
    and still produces its measured output on this slice", i.e. a silent
    model/revision drift — nothing more. No other turn's label is asserted."""
    monkeypatch.setenv(tone_id.TONE_AUDIO_ENV, "dark")
    pcm, sr, turn = _gptaudio_turn("shout_angry")
    out = tone_id.classify_turns(pcm, sr, [turn])
    assert len(out) == 1 and out[0]["skipped"] is None
    tone = out[0]["tone"]
    assert set(tone["scores"]) == set(tone_id.LABELS)
    assert sum(tone["scores"].values()) == pytest.approx(1.0, abs=1e-2)
    assert tone["label"] in tone_id.LABELS
    assert tone["model"] == tone_id.model_id()
    assert out[0]["latency_ms"] > 0
    assert tone["label"] == "angry"
