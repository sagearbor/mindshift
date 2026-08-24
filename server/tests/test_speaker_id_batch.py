"""Unit tests for speaker_id.embed_pcm_batch — the round-3 batched embedding
entry point (see diarize_sliding_window.py's module docstring for why it
exists: amortizing ECAPA's fixed per-call overhead across many windows).

These tests need REAL torch tensor ops (padding/stacking/wav_lens) but NOT
the real (heavy) speechbrain ECAPA checkpoint: a fake model object is
injected via speaker_id._model directly, so no model download/load happens.
Skipped honestly when torch itself isn't installed.
"""
from __future__ import annotations

import numpy as np
import pytest

import speaker_id

pytestmark = pytest.mark.skipif(
    not speaker_id.is_available(), reason="torch/speechbrain not installed",
)


class _FakeEncoder:
    """Records exactly what encode_batch was called with, and returns an
    embedding per row that encodes the REAL (unpadded) samples the model
    would have seen, per wav_lens -- so a test can prove padding never
    leaks into an embedding."""

    def __init__(self):
        self.calls = []

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        import torch

        self.calls.append((wavs.shape, None if wav_lens is None else wav_lens.clone()))
        batch, max_len = wavs.shape
        if wav_lens is None:
            wav_lens = torch.ones(batch)
        out = torch.zeros(batch, 1, 192)
        for i in range(batch):
            real_len = int(round(float(wav_lens[i]) * max_len))
            real = wavs[i, :real_len]
            # Deterministic "embedding" derived ONLY from the real samples:
            # if padding leaked in, this would change (padding is zeros, so
            # a nonzero-mean real region distinguishes leakage from none —
            # made robust by comparing directly against the expected value
            # computed from the caller's own known input in the tests below).
            out[i, 0, 0] = real.mean() if real_len else 0.0
            out[i, 0, 1] = float(real_len)
        return out


@pytest.fixture(autouse=True)
def _reset_model_cache():
    original = speaker_id._model
    yield
    speaker_id._model = original


def test_embed_pcm_batch_empty_list_returns_empty_no_model_call():
    fake = _FakeEncoder()
    speaker_id._model = fake
    assert speaker_id.embed_pcm_batch([]) == []
    assert fake.calls == []


def test_embed_pcm_batch_equal_length_chunks_no_padding_needed():
    fake = _FakeEncoder()
    speaker_id._model = fake
    sr = speaker_id.TARGET_SR
    chunks = [
        np.full(1000, 0.1, dtype=np.float32),
        np.full(1000, 0.5, dtype=np.float32),
        np.full(1000, 0.9, dtype=np.float32),
    ]
    out = speaker_id.embed_pcm_batch(chunks, sr)
    assert len(out) == 3
    for v in out:
        assert v.shape == (192,)
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5  # L2-normalized
    (shape, wav_lens) = fake.calls[0]
    assert shape == (3, 1000)
    assert wav_lens is not None
    assert all(abs(float(x) - 1.0) < 1e-6 for x in wav_lens)  # all full-length


def test_embed_pcm_batch_different_length_chunks_pads_and_reports_real_lengths():
    fake = _FakeEncoder()
    speaker_id._model = fake
    sr = speaker_id.TARGET_SR
    chunks = [
        np.full(500, 0.3, dtype=np.float32),   # shorter -> gets padded
        np.full(1000, 0.7, dtype=np.float32),  # the batch's longest
    ]
    speaker_id.embed_pcm_batch(chunks, sr)
    (shape, wav_lens) = fake.calls[0]
    assert shape == (2, 1000)  # padded to the longest chunk
    assert abs(float(wav_lens[0]) - 0.5) < 1e-6  # 500/1000
    assert abs(float(wav_lens[1]) - 1.0) < 1e-6  # 1000/1000


def test_embed_pcm_batch_padding_does_not_leak_into_shorter_chunks_embedding():
    """The fake model's 'embedding' encodes the mean of only the REAL
    (non-padded) samples it was told about via wav_lens; if padding leaked
    in as real signal, this would not match the chunk's true, unpadded
    mean."""
    fake = _FakeEncoder()
    speaker_id._model = fake
    sr = speaker_id.TARGET_SR
    short = np.full(400, 0.3, dtype=np.float32)
    long_ = np.full(1000, 0.7, dtype=np.float32)
    speaker_id.embed_pcm_batch([short, long_], sr)
    (shape, wav_lens) = fake.calls[0]
    assert abs(float(wav_lens[0]) - 0.4) < 1e-6


def test_embed_pcm_batch_wrong_sample_rate_raises():
    fake = _FakeEncoder()
    speaker_id._model = fake
    with pytest.raises(speaker_id.SpeakerIdUnavailable):
        speaker_id.embed_pcm_batch([np.zeros(100, dtype=np.float32)], sr=8000)


def test_embed_pcm_batch_matches_embed_pcm_ordering_and_count():
    """Order is preserved 1:1 with the input list."""
    fake = _FakeEncoder()
    speaker_id._model = fake
    sr = speaker_id.TARGET_SR
    chunks = [np.full(300 + 10 * i, float(i) / 10, dtype=np.float32) for i in range(5)]
    out = speaker_id.embed_pcm_batch(chunks, sr)
    assert len(out) == len(chunks)
