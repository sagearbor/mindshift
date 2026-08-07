"""Local speaker diarization — vendor-independent "who said each utterance".

Why this exists: Deepgram's nova-3 (model 2025-07-31.0) regressed prerecorded
diarization — it merged two distinct voices into one speaker on real recordings
(verified 2026-08-05 by direct nova-2/nova-3 comparison on identical bytes).
Renting diarization means a silent vendor model swap can corrupt every
per-speaker feature downstream (talk-share, heat, report cards). This module
re-derives speaker labels ON OUR OWN COMPUTE with the same PINNED ECAPA model
that already powers voice enrollment (``speaker_id``).

Algorithm — calibrated on the real recording that exposed the regression plus
the TTS fixture (2026-08-06):

1. Embed each transcript utterance that is long enough to carry voice signal.
   Per-utterance embeddings alone are NOISY — same-speaker cosine can dip
   below cross-speaker — so a plain similarity-threshold clustering
   over-fragments (it heard 4-5 "speakers" in a 2-person recording).
2. Force a 2-way split (average-linkage merge to exactly two clusters), then
   REFINE it: embed each cluster's POOLED audio (pooled embeddings are what
   ``speaker_id``'s calibration table trusts: same voice ≈0.73, different
   ≈0.19) and reassign every utterance to its closest pooled centroid, a few
   rounds until stable.
3. VALIDATE before believing it: accept the split only when the two pooled
   centroids are clearly two different voices (cosine ≤
   :data:`MAX_POOLED_COSINE`) and each cluster has enough pooled speech to be
   trustworthy (≥ :data:`MIN_CLUSTER_SECONDS`). A genuine monologue measures
   ≈0.73 pooled self-similarity and is REJECTED — we never invent a speaker.

Scope + honesty:

* Only a 2-way split is attempted — the fallback's job is "the transcript
  heard ONE voice; did it merge a two-person conversation?", not general
  N-speaker diarization.
* Segmentation still comes from the transcript's utterance boundaries — this
  fixes ATTRIBUTION per utterance, not a transcriber that merged two voices
  into ONE utterance (sub-utterance change detection is a planned follow-up).
* Requires the optional voice deps (torch + speechbrain). When they are
  missing, validation fails, or there is too little embeddable speech,
  :func:`diarize_turns` returns ``None`` and the caller keeps the transcript's
  labels.
* Pure math (merging, agreement) is torch-free; the orchestrator takes an
  injectable ``embed_fn(pcm_slice, sr)`` so the unit suite runs without torch.
"""

from __future__ import annotations

import logging
import os

import numpy as np

import speaker_id

logger = logging.getLogger(__name__)

# Accept a 2-way split only when the two clusters' POOLED embeddings are at
# most this similar. Calibration (2026-08-06, pinned ECAPA): different people
# pooled ≈0.19-0.26; the same real voice split in half ≈0.73; speaker_id's
# table puts merged/degraded artifacts at ≈0.48-0.56. 0.45 sits under all
# observed same-voice values with margin. Env-overridable for recalibration.
MAX_POOLED_COSINE = float(os.getenv("MINDSHIFT_DIARIZE_MAX_POOLED_COSINE", "0.45"))

# An utterance shorter than this is not embedded (too little voice signal); it
# inherits the nearest embedded utterance's cluster (nearest by midpoint).
MIN_SECONDS = 1.0

# Each side of an accepted split must have at least this much pooled speech —
# a "second voice" carried by one breath of audio is not evidence.
MIN_CLUSTER_SECONDS = 3.0

# Pooled-centroid reassignment rounds (converges in 1-2 on calibration data).
REFINE_ROUNDS = 3

# Cap pooled audio per cluster centroid, mirroring speaker_id.MAX_POOL_SECONDS.
MAX_POOL_SECONDS = 60.0

SOURCE = "local-ecapa"


def partition_agreement(a: list, b: list) -> float:
    """Pairwise (Rand) agreement of two labelings of the same items, in [0, 1].

    Label NAMES don't matter — only whether each pair of items is grouped
    together or apart in both labelings. Fewer than two items → 1.0 (nothing to
    disagree about). Used to log how far a local relabeling diverged from the
    transcript's own diarization.
    """
    if len(a) != len(b):
        raise ValueError(f"labelings differ in length: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        return 1.0
    agree = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if (a[i] == a[j]) == (b[i] == b[j]):
                agree += 1
    return agree / total


def _merge_to_two(embeddings: list[np.ndarray]) -> list[int]:
    """Average-linkage merging until exactly two clusters remain.

    Returns a 0/1 label per embedding, 0 = the cluster containing the earliest
    input. O(n^3) worst case — fine at transcript scale (≤ 400 turns).
    """
    n = len(embeddings)
    sim = np.array([[float(np.dot(a, b)) for b in embeddings] for a in embeddings])
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > 2:
        best = (-2.0, 0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                avg = float(np.mean([sim[a][b] for a in clusters[i] for b in clusters[j]]))
                if avg > best[0]:
                    best = (avg, i, j)
        _, i, j = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
    labels = [0] * n
    for cid, members in enumerate(sorted(clusters, key=min)):
        for m in members:
            labels[m] = cid
    return labels


def _slice(pcm: np.ndarray, sr: int, turn: dict) -> np.ndarray:
    i0 = max(0, int(float(turn.get("start_time") or 0.0) * sr))
    i1 = min(pcm.size, int(float(turn.get("end_time") or 0.0) * sr))
    return pcm[i0:i1]


def _pooled(
    pcm: np.ndarray, sr: int, turns: list[dict], idxs: list[int],
) -> np.ndarray:
    """Concatenate the given turns' audio, capped at MAX_POOL_SECONDS."""
    cap = int(MAX_POOL_SECONDS * sr)
    chunks, total = [], 0
    for i in idxs:
        chunk = _slice(pcm, sr, turns[i])
        chunks.append(chunk)
        total += chunk.size
        if total >= cap:
            break
    pooled = np.concatenate(chunks)
    return np.ascontiguousarray(pooled[:cap])


def _default_embed(pcm_slice: np.ndarray, sr: int) -> np.ndarray:
    """Embed an audio slice with the pinned ECAPA model."""
    if not speaker_id.is_available():
        raise speaker_id.SpeakerIdUnavailable(
            "voice deps not installed (torch + speechbrain)"
        )
    return speaker_id.embed_pcm(np.ascontiguousarray(pcm_slice), sr)


def _speaker_name(index: int) -> str:
    """Cluster index → display label, matching the transcript convention."""
    if index < 26:
        return f"Speaker {chr(ord('A') + index)}"
    return f"Speaker {index + 1}"


def diarize_turns(
    pcm: np.ndarray,
    sr: int,
    turns: list[dict],
    *,
    embed_fn=None,
    min_seconds: float = MIN_SECONDS,
    max_pooled_cosine: float = MAX_POOLED_COSINE,
) -> dict | None:
    """Attempt a validated two-voice relabeling of ``turns`` from the audio.

    Returns ``None`` whenever local diarization has nothing TRUSTWORTHY to
    say: voice model unavailable, fewer than two embeddable utterances, a
    cluster with too little pooled speech, or pooled centroids that are not
    clearly two different voices (a monologue). Otherwise returns::

        {
          "turns": [...],              # same dicts, speaker relabeled
          "num_speakers": 2,
          "source": "local-ecapa",
          "model": "<hf-source>@<pinned-revision>",
          "segments_total": int,
          "segments_embedded": int,
          "pooled_cosine": float,      # centroid similarity (low = distinct)
          "agreement_with_input": float,   # Rand agreement vs input labels
        }
    """
    embed = embed_fn or _default_embed
    try:
        embedded: dict[int, np.ndarray] = {}
        for idx, t in enumerate(turns):
            start = float(t.get("start_time") or 0.0)
            end = float(t.get("end_time") or 0.0)
            if end - start < min_seconds:
                continue
            embedded[idx] = speaker_id.l2_normalize(embed(_slice(pcm, sr, t), sr))

        if len(embedded) < 2:
            return None

        order = sorted(embedded)
        embs = [embedded[i] for i in order]
        labels = _merge_to_two(embs)

        # Refine against pooled centroids; bail out if refinement empties a
        # side (all utterances actually sound like one voice).
        centroids: dict[int, np.ndarray] = {}
        for _ in range(REFINE_ROUNDS):
            centroids = {}
            for c in set(labels):
                idxs = [order[i] for i in range(len(order)) if labels[i] == c]
                centroids[c] = speaker_id.l2_normalize(
                    embed(_pooled(pcm, sr, turns, idxs), sr)
                )
            if len(centroids) < 2:
                return None
            new = [
                max(centroids, key=lambda c: float(np.dot(embs[i], centroids[c])))
                for i in range(len(embs))
            ]
            if new == labels:
                break
            labels = new
        if len(set(labels)) < 2:
            return None

        # Validate: enough speech on each side, and clearly two voices.
        for c in set(labels):
            idxs = [order[i] for i in range(len(order)) if labels[i] == c]
            seconds = sum(_slice(pcm, sr, turns[i]).size for i in idxs) / sr
            if seconds < MIN_CLUSTER_SECONDS:
                logger.info(
                    "local diarization rejected split: cluster has only "
                    "%.1fs pooled speech", seconds,
                )
                return None
        pooled_cosine = float(np.dot(centroids[0], centroids[1]))
        if pooled_cosine > max_pooled_cosine:
            logger.info(
                "local diarization heard one voice (pooled cosine %.3f > %.2f)",
                pooled_cosine, max_pooled_cosine,
            )
            return None
    except speaker_id.SpeakerIdUnavailable as exc:
        logger.info("local diarization unavailable: %s", exc)
        return None

    cluster_of = dict(zip(order, labels))

    # Un-embedded (too-short) turns inherit the nearest embedded turn's
    # cluster, nearest by utterance midpoint.
    def midpoint(t: dict) -> float:
        return (float(t.get("start_time") or 0.0) + float(t.get("end_time") or 0.0)) / 2

    for idx, t in enumerate(turns):
        if idx in cluster_of:
            continue
        nearest = min(order, key=lambda e: abs(midpoint(turns[e]) - midpoint(t)))
        cluster_of[idx] = cluster_of[nearest]

    # Name clusters in order of first appearance across the full transcript.
    name_of: dict[int, str] = {}
    for idx in range(len(turns)):
        cid = cluster_of[idx]
        if cid not in name_of:
            name_of[cid] = _speaker_name(len(name_of))

    new_turns = [dict(t, speaker=name_of[cluster_of[i]]) for i, t in enumerate(turns)]
    return {
        "turns": new_turns,
        "num_speakers": len(name_of),
        "source": SOURCE,
        "model": f"{speaker_id.ECAPA_SOURCE}@{speaker_id.ECAPA_REVISION}",
        "segments_total": len(turns),
        "segments_embedded": len(embedded),
        "pooled_cosine": pooled_cosine,
        "agreement_with_input": partition_agreement(
            [t.get("speaker") for t in turns],
            [t["speaker"] for t in new_turns],
        ),
    }
