# Ported from gauge@2157433 server/diarize.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
"""Pure diarization: energy VAD segmentation + voiceprint-based speaker labels.

Wired into production by ``server/watch/post_session.py`` (Task B10) via the
``DiarizationService`` protocol at the bottom of this module. The
segmentation/labeling functions above have NO torch import anywhere — it is
pure numpy plus the already-tested pure vector math in ``speaker_id``
(``cosine`` / ``l2_normalize`` / ``running_mean_embedding``), so the whole
module is unit-testable without the optional voice dependencies installed.

Two-stage pipeline, mirroring how ``VectorEngine.push_diarization`` consumes
its input:

1. :func:`speech_segments` — a simple energy VAD over fixed-size frames using
   the SAME silence floor (``watch.vectors.SILENCE_FLOOR_DBFS``) the
   streaming engine already uses, so "speech" means one thing across the
   codebase. Adjacent bursts separated by a short gap are merged into one
   turn; bursts too short to be a real turn (a cough, a click) are dropped.
2. :func:`assign_speakers` — labels each segment's voiceprint embedding
   "self" (only ever assigned when it clears ``self_threshold`` against a
   real enrolled voiceprint — there is no "self" without one), or clusters
   it against the "other" speakers seen so far in this recording.

:func:`diarize` glues segmentation, embedding, and labeling together and
returns turns in exactly ``VectorEngine.push_diarization``'s tuple shape:
``(speaker, start_s, end_s)``, honestly dropping anything that can't be
embedded or labeled rather than guessing.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Protocol

import numpy as np

import speaker_id
from speaker_id import MATCH_THRESHOLD, cosine, l2_normalize, running_mean_embedding
from watch.vectors import SILENCE_FLOOR_DBFS, rms_dbfs

logger = logging.getLogger(__name__)

# (float32 mono PCM in [-1, 1], sample_rate) -> L2-normed voiceprint vector.
# Mirrors rest.py's Embedder alias — same shape, same contract.
EmbedFn = Callable[[np.ndarray, int], np.ndarray]

FRAME_SECONDS = 0.25            # VAD frame
MERGE_GAP_SECONDS = 0.30        # gaps this short don't split a turn
MIN_SEGMENT_SECONDS = 0.60      # shorter than this is a cough, not a turn
CLUSTER_THRESHOLD = 0.55        # two "other" segments are the same person above this

# PCM16 full-scale divisor — matches speaker_id.embed_pcm's expected input
# range (float32, roughly [-1, 1]) so embed_fn sees the same scale whether it
# is fed a real recording or a synthetic test fixture.
PCM16_FULL_SCALE = 32768.0


def speech_segments(
    pcm: np.ndarray,
    sr: int,
    *,
    floor_dbfs: float = SILENCE_FLOOR_DBFS,
    frame_seconds: float = FRAME_SECONDS,
    merge_gap_seconds: float = MERGE_GAP_SECONDS,
    min_seconds: float = MIN_SEGMENT_SECONDS,
) -> list[tuple[float, float]]:
    """Energy VAD: (start_s, end_s) spans of speech.

    Frames at/under the silence floor are silence — the same threshold
    ``VectorEngine`` uses, so "speech" means one thing across the codebase.
    """
    frame_samples = max(1, int(round(frame_seconds * sr)))
    n_frames = pcm.size // frame_samples
    trailing = pcm.size - n_frames * frame_samples  # < frame_samples, may be 0
    if n_frames == 0 and trailing == 0:
        return []

    # Raw speech/silence per frame (energy VAD only — no smoothing beyond
    # the merge/min-duration passes below). Frame boundaries are tracked as
    # explicit cumulative timestamps (not i * frame_seconds) so a trailing
    # sub-frame — shorter than frame_seconds, e.g. a clip whose length isn't
    # an exact multiple of it — is still evaluated for speech instead of
    # being silently dropped uncounted.
    is_speech: list[bool] = []
    boundaries: list[float] = [0.0]
    for i in range(n_frames):
        chunk = pcm[i * frame_samples:(i + 1) * frame_samples]
        is_speech.append(rms_dbfs(chunk) > floor_dbfs)
        boundaries.append(boundaries[-1] + frame_seconds)
    if trailing > 0:
        chunk = pcm[n_frames * frame_samples:]
        is_speech.append(rms_dbfs(chunk) > floor_dbfs)
        boundaries.append(boundaries[-1] + trailing / sr)

    # Frame-run -> raw (start_s, end_s) segments.
    raw_segments: list[tuple[float, float]] = []
    run_start: int | None = None
    for i, speech in enumerate(is_speech):
        if speech and run_start is None:
            run_start = i
        elif not speech and run_start is not None:
            raw_segments.append((boundaries[run_start], boundaries[i]))
            run_start = None
    if run_start is not None:
        raw_segments.append((boundaries[run_start], boundaries[len(is_speech)]))

    if not raw_segments:
        return []

    # Merge gaps short enough not to split a turn.
    merged: list[tuple[float, float]] = [raw_segments[0]]
    for start, end in raw_segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap_seconds:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # Drop bursts too short to be a real turn.
    return [(s, e) for s, e in merged if (e - s) >= min_seconds]


def assign_speakers(
    embeddings: list[np.ndarray | None],
    self_print: np.ndarray | None,
    *,
    self_threshold: float = MATCH_THRESHOLD,
    cluster_threshold: float = CLUSTER_THRESHOLD,
) -> list[str | None]:
    """Label each embedding "self" | "other-1" | "other-2" | ... | None.

    Without ``self_print`` NOTHING is ever labeled "self" — there is no
    honest basis for it, and the interrupting/airtime vectors are defined
    only for the wearer. ``None`` entries in (or out of) the list are
    segments that could not be embedded and carry no label — never a guess.

    Clustering is greedy and order-stable: for a non-self embedding, cosine
    against each existing other-centroid; join the best one clearing
    ``cluster_threshold`` and update that centroid via
    ``speaker_id.running_mean_embedding``; otherwise mint ``other-{n+1}``.
    """
    self_vec = l2_normalize(self_print) if self_print is not None else None
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    labels: list[str | None] = []

    for emb in embeddings:
        if emb is None:
            labels.append(None)
            continue

        if self_vec is not None and cosine(emb, self_vec) >= self_threshold:
            labels.append("self")
            continue

        best_idx: int | None = None
        best_score = -1.0
        for idx, centroid in enumerate(centroids):
            score = cosine(emb, centroid)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None and best_score >= cluster_threshold:
            centroids[best_idx] = running_mean_embedding(centroids[best_idx], counts[best_idx], emb)
            counts[best_idx] += 1
            labels.append(f"other-{best_idx + 1}")
        else:
            centroids.append(l2_normalize(emb))
            counts.append(1)
            labels.append(f"other-{len(centroids)}")

    return labels


def diarize(
    pcm: np.ndarray,
    sr: int,
    self_print: np.ndarray | None,
    embed_fn: EmbedFn,
) -> list[tuple[str, float, float]]:
    """Segment -> embed -> label.

    Returns turns in exactly ``VectorEngine.push_diarization``'s tuple
    shape: ``(speaker, start_s, end_s)``, ascending by start. Segments whose
    ``embed_fn`` raises are DROPPED (logged), never fabricated; unlabeled
    segments are dropped too.
    """
    segments = speech_segments(pcm, sr)
    if not segments:
        return []

    embeddings: list[np.ndarray | None] = []
    for start, end in segments:
        i0 = max(0, int(round(start * sr)))
        i1 = min(pcm.size, int(round(end * sr)))
        chunk = (pcm[i0:i1].astype(np.float32)) / PCM16_FULL_SCALE
        try:
            embeddings.append(embed_fn(chunk, sr))
        except Exception:  # noqa: BLE001 — an unembeddable segment is dropped, not fatal
            logger.warning("diarize: embed_fn failed for segment %.2f-%.2fs; dropping", start, end)
            embeddings.append(None)

    labels = assign_speakers(embeddings, self_print)

    turns = [
        (label, float(start), float(end))
        for (start, end), label in zip(segments, labels)
        if label is not None
    ]
    return sorted(turns, key=lambda t: t[1])


# ---------------------------------------------------------------------------
# Service wrappers (Task B10's production callers over the pure functions above)
# ---------------------------------------------------------------------------

class DiarizationService(Protocol):
    def diarize(self, pcm: bytes, sr: int, self_print: np.ndarray | None) -> list[tuple[str, float, float]]:
        ...


class NullDiarizationService:
    """No diarization configured/available — returns [] so interrupting/airtime
    stay honestly dark rather than being guessed from a single mono channel."""

    def diarize(self, pcm: bytes, sr: int, self_print: np.ndarray | None) -> list[tuple[str, float, float]]:
        return []


class EmbeddingDiarizationService:
    """Wraps an Embedder; converts PCM16 bytes -> float32 and calls diarize()."""

    def __init__(self, embedder: EmbedFn) -> None:
        self._embedder = embedder

    def diarize(self, pcm: bytes, sr: int, self_print: np.ndarray | None) -> list[tuple[str, float, float]]:
        # int16-scale floats, NOT normalized to [-1, 1] here: the pure
        # diarize() (and speech_segments -> rms_dbfs within it) expects
        # int16-scale input and does its own /PCM16_FULL_SCALE just before
        # embed_fn (line ~185). Normalizing a second time here was C1: it
        # silently sank every frame ~90 dB below the -45 dBFS silence floor,
        # so diarization always returned [].
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        return diarize(samples, sr, self_print, self._embedder)


class LazyDiarizationService:
    """Resolves to Embedding-or-Null on FIRST ``.diarize()`` call, never at
    construction (Task H3).

    ``build_watch_routers()`` runs at ``import main`` (``server/main.py``'s
    module-level ``for _r in build_watch_routers(): ...``), so eagerly
    calling ``speaker_id.is_available()`` there — which tries ``import
    torch; import speechbrain`` — paid that heavy import cost on EVERY
    Cloud Run cold start (min-instances 0), even for requests that never
    touch diarization. This proxy defers that check to the first actual
    use — request time, exactly mirroring how ``server/routers/voice.py``'s
    pre-existing routes (and this module's sibling, ``rest.py``'s
    ``_resolve_embedder``) already behave. Unlike the embedder, the
    diarizer has no per-request fallback in ``ws.py``/``live_sessions.py``
    — both just use whatever object they're handed — so it is still built
    ONCE and shared by both routers; only the WHEN of that one-time build
    moves from import time to first-use time.

    Resolution happens once under a lock and is cached for the life of the
    process — double-checked locking so concurrent live sessions (both
    routers' ``.diarize()`` calls run off the event loop via
    ``asyncio.to_thread``, so they can genuinely race here) resolve at most
    once, never rebuild per call.

    Degradation parity: when torch/speechbrain are genuinely unavailable at
    first use, this resolves to the SAME ``NullDiarizationService()`` (``[]``,
    honestly) eager construction would have produced at import time — never
    a crash, never behavior different from the eager code path it replaces.
    """

    def __init__(self) -> None:
        self._resolved: DiarizationService | None = None
        self._lock = threading.Lock()

    def _resolve(self) -> DiarizationService:
        resolved = self._resolved
        if resolved is not None:
            return resolved
        with self._lock:
            if self._resolved is None:
                self._resolved = (
                    EmbeddingDiarizationService(speaker_id.embed_pcm)
                    if speaker_id.is_available()
                    else NullDiarizationService()
                )
            return self._resolved

    def diarize(self, pcm: bytes, sr: int, self_print: np.ndarray | None) -> list[tuple[str, float, float]]:
        return self._resolve().diarize(pcm, sr, self_print)
