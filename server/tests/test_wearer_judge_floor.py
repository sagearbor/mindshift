"""Wearer-identity rule (plan 2026-11-05, Step 3), server half: the voiceprint
matcher judges only >= 1.5 s of audio. ``audio_pipeline._confirm_wearer_by_voiceprint``
reads ``speaker_id.MIN_MATCH_SECONDS`` at call time, so this one constant is
the floor for the per-turn wearer confirmation and for pooled-cluster matching.
"""
from __future__ import annotations

import numpy as np

from server import speaker_id

SR = 16000


def _turns(seconds: float) -> list[dict]:
    return [{"speaker": "A", "start_time": 0.0, "end_time": seconds}]


def test_floor_is_one_and_a_half_seconds():
    assert speaker_id.MIN_MATCH_SECONDS == 1.5


def test_a_1_2_s_turn_is_not_judged(monkeypatch):
    called = []
    monkeypatch.setattr(speaker_id, "embed_pcm", lambda pcm, sr: called.append(pcm.size) or np.ones(4, np.float32))
    pcm = np.random.default_rng(0).standard_normal(3 * SR).astype(np.float32) * 0.1
    assert speaker_id.embed_speaker(pcm, SR, _turns(1.2), "A") is None
    assert called == []  # the embedder was never even asked


def test_a_1_6_s_turn_is_judged(monkeypatch):
    called = []
    monkeypatch.setattr(speaker_id, "embed_pcm", lambda pcm, sr: called.append(pcm.size) or np.ones(4, np.float32))
    pcm = np.random.default_rng(0).standard_normal(3 * SR).astype(np.float32) * 0.1
    assert speaker_id.embed_speaker(pcm, SR, _turns(1.6), "A") is not None
    assert called == [int(1.6 * SR)]
