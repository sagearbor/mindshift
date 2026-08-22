"""Unit tests for diarize_sliding_window — round-3 centroid-anchored margin
change-point detection + batched embedding pass.

All tests run WITHOUT torch/speechbrain: pure numpy, with injectable fake
``embed``/``embed_batch`` callables (same pattern as test_diarize_local.py).
"""

from __future__ import annotations

import numpy as np
import pytest

import diarize_sliding_window as dsw

SR = 16000


def _direction(seed: int, dim: int = 192) -> np.ndarray:
    """A fixed unit vector for a synthetic "voice" — deterministic per seed."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return dsw.l2n(v)


def _noisy(direction: np.ndarray, noise_seed: int, noise_scale: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(noise_seed)
    v = direction + noise_scale * rng.standard_normal(direction.shape).astype(np.float32)
    return dsw.l2n(v)


# ---------------------------------------------------------------------------
# _window_slices / sliding_window_embeddings / sliding_window_embeddings_batched
# ---------------------------------------------------------------------------

def test_window_slices_covers_expected_grid():
    pcm = np.arange(SR * 3, dtype=np.float32)  # 3 seconds
    starts, chunks = dsw._window_slices(pcm, SR, window=1.0, hop=0.5)
    assert starts == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert all(c.size == SR for c in chunks)


def test_sliding_window_embeddings_batched_matches_looped_stub():
    """The batched pass must produce the SAME (starts, embeddings) as the
    per-window loop for a given stub — batching is a performance change,
    not a behavior change."""
    pcm = np.arange(SR * 3, dtype=np.float32)

    def embed_single(chunk, sr):
        return dsw.l2n(np.array([chunk.mean(), chunk.std(), 1.0], dtype=np.float32))

    def embed_batch(chunks, sr):
        return [embed_single(c, sr) for c in chunks]

    starts_loop, embs_loop = dsw.sliding_window_embeddings(pcm, SR, 1.0, 0.5, embed_single)
    starts_batch, embs_batch = dsw.sliding_window_embeddings_batched(pcm, SR, 1.0, 0.5, embed_batch)

    assert starts_loop == starts_batch
    for a, b in zip(embs_loop, embs_batch):
        assert np.allclose(a, b, atol=1e-6)


def test_sliding_window_embeddings_batched_respects_max_batch_chunking():
    """A max_batch smaller than the window count must still produce every
    embedding, in order, via multiple embed_batch calls."""
    pcm = np.arange(SR * 5, dtype=np.float32)
    calls = []

    def embed_batch(chunks, sr):
        calls.append(len(chunks))
        return [dsw.l2n(np.array([c.mean(), 1.0, 1.0], dtype=np.float32)) for c in chunks]

    starts, embs = dsw.sliding_window_embeddings_batched(
        pcm, SR, window=1.0, hop=0.5, embed_batch=embed_batch, max_batch=3,
    )
    assert len(starts) == len(embs) == 9  # (5-1)/0.5 + 1 = 9 windows
    assert sum(calls) == 9
    assert all(c <= 3 for c in calls)
    assert len(calls) == 3  # 3 batches of <=3 to cover 9 windows


# ---------------------------------------------------------------------------
# centroid_margin_change_points — the round-3 noise fix
# ---------------------------------------------------------------------------

def _synthetic_speaker_embeddings(hop_count_per_speaker: int, n_speakers: int) -> tuple[list[float], list[np.ndarray]]:
    """n_speakers distinct "voices" (fixed random directions), each held for
    hop_count_per_speaker consecutive window positions with small per-window
    noise (mirrors real same-speaker embedding jitter)."""
    hop = 0.5
    directions = [_direction(seed=100 + s) for s in range(n_speakers)]
    starts: list[float] = []
    embs: list[np.ndarray] = []
    idx = 0
    for s, direction in enumerate(directions):
        for j in range(hop_count_per_speaker):
            starts.append(round(idx * hop, 4))
            embs.append(_noisy(direction, noise_seed=1000 + idx))
            idx += 1
    return starts, embs


def test_centroid_margin_change_points_finds_two_speaker_boundary():
    starts, embs = _synthetic_speaker_embeddings(hop_count_per_speaker=12, n_speakers=2)
    boundaries = dsw.centroid_margin_change_points(
        starts, embs, min_segment_windows=4, lookahead_windows=3,
        threshold=0.5, min_run=2,
    )
    # True change is at index 12 -> starts[12] = 6.0s.
    assert len(boundaries) >= 1
    assert min(abs(b - 6.0) for b in boundaries) <= 1.0


def test_centroid_margin_change_points_finds_all_boundaries_and_resets():
    """Three distinct voices back-to-back: the detector must find BOTH real
    changes, not just the first (this is the regression the non-resetting
    centroid_margin_curve is documented to be vulnerable to)."""
    starts, embs = _synthetic_speaker_embeddings(hop_count_per_speaker=12, n_speakers=3)
    boundaries = dsw.centroid_margin_change_points(
        starts, embs, min_segment_windows=4, lookahead_windows=3,
        threshold=0.5, min_run=2,
    )
    # True changes at index 12 (t=6.0) and index 24 (t=12.0).
    assert len(boundaries) >= 2
    assert min(abs(b - 6.0) for b in boundaries) <= 1.0
    assert min(abs(b - 12.0) for b in boundaries) <= 1.0


def test_centroid_margin_change_points_no_false_positive_on_one_speaker():
    """A single voice held throughout (with realistic small jitter) must
    yield NO change points — the sustained-run requirement exists precisely
    to reject noise like this."""
    direction = _direction(seed=7)
    starts = [round(i * 0.5, 4) for i in range(30)]
    embs = [_noisy(direction, noise_seed=2000 + i, noise_scale=0.08) for i in range(30)]
    boundaries = dsw.centroid_margin_change_points(
        starts, embs, min_segment_windows=4, lookahead_windows=3,
        threshold=0.5, min_run=2,
    )
    assert boundaries == []


def test_centroid_margin_change_points_requires_sustained_run():
    """A single-hop noise spike (not sustained for min_run consecutive
    hops) must NOT be accepted as a change — this is the exact failure mode
    round 2's raw window-pair approach had no defense against."""
    direction = _direction(seed=9)
    starts = [round(i * 0.5, 4) for i in range(20)]
    embs = [_noisy(direction, noise_seed=3000 + i, noise_scale=0.05) for i in range(20)]
    # Inject ONE lone outlier embedding (simulating a single noisy window,
    # e.g. a cough or mic artifact) that would exceed threshold in isolation.
    embs[10] = _direction(seed=999)
    boundaries = dsw.centroid_margin_change_points(
        starts, embs, min_segment_windows=4, lookahead_windows=3,
        threshold=0.5, min_run=3,
    )
    assert boundaries == []


# ---------------------------------------------------------------------------
# boundaries_to_turns (unchanged pure math, still covered here)
# ---------------------------------------------------------------------------

def test_boundaries_to_turns_merges_short_slivers():
    turns = dsw.boundaries_to_turns([5.0, 5.3, 10.0], duration=15.0, min_turn_seconds=1.0)
    # 5.0-5.3 is a 0.3s sliver -> merged into its shorter neighbor.
    assert all(t["end_time"] - t["start_time"] >= 0.3 for t in turns)
    assert turns[0]["start_time"] == 0.0
    assert turns[-1]["end_time"] == 15.0


def test_boundaries_to_turns_no_boundaries_yields_one_turn():
    turns = dsw.boundaries_to_turns([], duration=10.0, min_turn_seconds=1.0)
    assert turns == [{"start_time": 0.0, "end_time": 10.0}]


# ---------------------------------------------------------------------------
# detect_turns_from_audio — the full orchestrator, stubbed embed_batch
# ---------------------------------------------------------------------------

def test_detect_turns_from_audio_orchestrator_end_to_end():
    """Two "voices" (constant-fill PCM, mirroring test_diarize_local.py's
    fake-embedding convention) back to back; the orchestrator should
    propose a turn split near the real boundary."""
    seconds_per_voice = 6.0
    pcm = np.concatenate([
        np.full(int(seconds_per_voice * SR), 0.1, dtype=np.float32),
        np.full(int(seconds_per_voice * SR), 0.9, dtype=np.float32),
    ])

    def embed_batch(chunks, sr):
        # Embedding "angle" tracks the chunk's mean fill value, plus tiny
        # deterministic jitter so same-voice windows aren't bit-identical.
        out = []
        for i, c in enumerate(chunks):
            m = float(c.mean())
            out.append(dsw.l2n(np.array([m, 1.0 - m, 0.01 * ((i % 5) - 2)], dtype=np.float32)))
        return out

    result = dsw.detect_turns_from_audio(pcm, SR, embed_batch, threshold=0.05, min_run=2)
    assert result["turns"][0]["start_time"] == 0.0
    assert result["turns"][-1]["end_time"] == pytest.approx(2 * seconds_per_voice, abs=1e-6)
    # At least one turn boundary should land near the true 6.0s change.
    boundary_times = [t["end_time"] for t in result["turns"][:-1]]
    assert any(abs(b - seconds_per_voice) <= 1.0 for b in boundary_times)
