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
* Segmentation starts from the transcript's utterance boundaries, PLUS a
  word-level pass for the transcriber-welded case (two voices merged into ONE
  utterance): utterances longer than :data:`SPLIT_MIN_UTTERANCE_SECONDS` that
  carry per-word timings are scanned for a SUSTAINED voice-change point —
  windows on either side of a sliding candidate boundary are scored by their
  affinity MARGIN against the two POOLED cluster centroids from the first
  clustering pass (window-to-window cosine is useless on real speech; see the
  SPLIT_* constants), and a change requires consecutive opposite-sign margins
  clearing :data:`SPLIT_MIN_MARGIN`. The utterance is split at the nearest
  word boundary and everything is re-clustered + validated over the finer
  segments. No sustained evidence → no split.
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

# --- Word-level speaker-change splitting -----------------------------------
# A transcriber can weld a speaker handoff into ONE utterance; per-word
# timings let us split it at the change. Calibrated on the real recording
# (2026-08-07): comparing two short windows TO EACH OTHER is useless — on real
# speech, same-speaker 1.5-3s windows score cosine ≈0.0-0.35 against each
# other, indistinguishable from a genuine change. What separates cleanly is
# each window's affinity MARGIN against the two POOLED cluster centroids
# (margin = cos(win, c0) - cos(win, c1)): pure utterances keep both sides of
# every candidate boundary on ONE sign (measured: no flip anywhere), while a
# welded handoff shows a SUSTAINED run of opposite-sign margins (measured
# weaker-margin values ≥0.19 inside genuine runs; edge candidates ≈0.02-0.03).

# Only utterances longer than this get scanned (bounds compute; a shorter
# utterance can't yield two trustworthy sides anyway).
SPLIT_MIN_UTTERANCE_SECONDS = 5.0

# Window on each side of a candidate boundary. Below ~1.5s ECAPA embeddings
# carry too little voice signal (2026-08-06 calibration).
SPLIT_WINDOW_SECONDS = 1.5

# Candidate boundaries are tried every SPLIT_HOP_SECONDS.
SPLIT_HOP_SECONDS = 0.25

# A change point is believed only when at least this many CONSECUTIVE
# candidates flip with margin — a lone flip is noise.
SPLIT_SUSTAIN = 2

# Both sides of a flip candidate must clear this margin. Measured floor:
# genuine-change candidates ≥0.19, edge noise ≤0.03; 0.15 splits the gap.
SPLIT_MIN_MARGIN = float(os.getenv("MINDSHIFT_DIARIZE_SPLIT_MIN_MARGIN", "0.15"))


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


def _sustained_flip(
    times: list[float], left_margins: list[float], right_margins: list[float],
    min_margin: float, min_run: int = SPLIT_SUSTAIN,
) -> float | None:
    """Time of the strongest candidate inside a sustained opposite-sign run.

    Pure math. A candidate qualifies when its left and right margins have
    OPPOSITE signs and the weaker one still clears ``min_margin``; a change
    point is believed only when at least ``min_run`` CONSECUTIVE candidates
    qualify (calibration: lone flips and sub-margin flips appear only at the
    edges of genuine changes and in noise). Returns the qualifying candidate
    with the largest weaker-margin (the clearest separation), or None.
    """
    best_time: float | None = None
    best_q = -np.inf
    run: list[int] = []

    def flush(run: list[int]) -> None:
        nonlocal best_time, best_q
        if len(run) < min_run:
            return
        for j in run:
            q = min(abs(left_margins[j]), abs(right_margins[j]))
            if q > best_q:
                best_q = q
                best_time = times[j]

    for i in range(len(times)):
        opposite = left_margins[i] * right_margins[i] < 0
        strong = min(abs(left_margins[i]), abs(right_margins[i])) >= min_margin
        if opposite and strong:
            run.append(i)
            continue
        flush(run)
        run = []
    flush(run)
    return best_time


def find_change_point(
    pcm: np.ndarray, sr: int, start: float, end: float, embed,
    centroids: tuple[np.ndarray, np.ndarray],
    *,
    min_margin: float = SPLIT_MIN_MARGIN,
) -> float | None:
    """Scan [start, end] for ONE sustained speaker-change point, or None.

    Every SPLIT_HOP_SECONDS, embed the SPLIT_WINDOW_SECONDS of audio on each
    side of the candidate boundary and score each window's affinity margin
    against the two POOLED cluster centroids (margin = cos(win, c0) -
    cos(win, c1)). Window-to-window cosine is NOT used — on real speech it is
    noise (see module constants). A change point is a sustained run of
    opposite-sign margins (:func:`_sustained_flip`); no sustained flip, no
    change point, no split. Never fabricates.
    """
    c0, c1 = centroids
    times: list[float] = []
    lefts: list[float] = []
    rights: list[float] = []
    b = start + SPLIT_WINDOW_SECONDS
    while b <= end - SPLIT_WINDOW_SECONDS + 1e-9:
        left = pcm[int((b - SPLIT_WINDOW_SECONDS) * sr):int(b * sr)]
        right = pcm[int(b * sr):int((b + SPLIT_WINDOW_SECONDS) * sr)]
        e_left = speaker_id.l2_normalize(embed(np.ascontiguousarray(left), sr))
        e_right = speaker_id.l2_normalize(embed(np.ascontiguousarray(right), sr))
        times.append(b)
        lefts.append(float(np.dot(e_left, c0) - np.dot(e_left, c1)))
        rights.append(float(np.dot(e_right, c0) - np.dot(e_right, c1)))
        b += SPLIT_HOP_SECONDS
    return _sustained_flip(times, lefts, rights, min_margin)


def split_turn_at_word_boundary(
    turn: dict, change_point: float, *, min_seconds: float = MIN_SECONDS,
) -> list[dict] | None:
    """Split ``turn`` at the word boundary nearest ``change_point``.

    Returns ``[left, right]`` turn dicts — text divided by the turn's own
    words, times meeting at the midpoint of the chosen inter-word gap — or
    ``None`` when the split cannot be made honestly: no word timings, no
    interior boundary, or a resulting piece shorter than ``min_seconds``
    (a sliver piece carries too little voice signal to attribute).
    """
    words = turn.get("words")
    if not isinstance(words, list) or len(words) < 2:
        return None
    # Candidate boundaries: midpoints of the gaps between consecutive words.
    boundaries = [
        (i, (float(words[i]["end_time"]) + float(words[i + 1]["start_time"])) / 2)
        for i in range(len(words) - 1)
    ]
    i, boundary = min(boundaries, key=lambda ib: abs(ib[1] - change_point))
    start = float(turn.get("start_time") or 0.0)
    end = float(turn.get("end_time") or 0.0)
    if boundary - start < min_seconds or end - boundary < min_seconds:
        return None
    base = {k: v for k, v in turn.items() if k != "words"}
    left = dict(
        base,
        text=" ".join(w["word"] for w in words[: i + 1]),
        start_time=start, end_time=boundary,
    )
    right = dict(
        base,
        text=" ".join(w["word"] for w in words[i + 1:]),
        start_time=boundary, end_time=end,
    )
    return [left, right]


def split_long_utterances(
    pcm: np.ndarray, sr: int, turns: list[dict], embed,
    centroids: tuple[np.ndarray, np.ndarray],
) -> tuple[list[dict], dict]:
    """Split long word-timed utterances at sustained voice changes.

    ``centroids`` are the two POOLED cluster centroids from a first
    whole-utterance clustering pass — the reliable anchors every scan window
    is scored against. Returns ``(finer_turns, stats)`` where stats counts
    ``scanned``, ``split``, ``skipped_short`` (at or under
    SPLIT_MIN_UTTERANCE_SECONDS — bounded compute, by design) and
    ``skipped_no_words`` (long enough to scan but the transcriber gave no word
    timings — logged, never hidden). Turns that yield no sustained change
    point pass through unchanged.
    """
    stats = {"scanned": 0, "split": 0, "skipped_short": 0, "skipped_no_words": 0}
    out: list[dict] = []
    for t in turns:
        start = float(t.get("start_time") or 0.0)
        end = float(t.get("end_time") or 0.0)
        if end - start <= SPLIT_MIN_UTTERANCE_SECONDS:
            stats["skipped_short"] += 1
            out.append(t)
            continue
        if not t.get("words"):
            stats["skipped_no_words"] += 1
            logger.info(
                "split scan skipped %.1fs utterance at %.2fs: no word timings",
                end - start, start,
            )
            out.append(t)
            continue
        stats["scanned"] += 1
        change = find_change_point(pcm, sr, start, end, embed, centroids)
        pieces = (
            split_turn_at_word_boundary(t, change) if change is not None else None
        )
        if pieces is None:
            out.append(t)
            continue
        stats["split"] += 1
        logger.info(
            "split %.1fs utterance at %.2fs (change point %.2fs → word "
            "boundary %.2fs)",
            end - start, start, change, pieces[0]["end_time"],
        )
        out.extend(pieces)
    return out, stats


def _speaker_name(index: int) -> str:
    """Cluster index → display label, matching the transcript convention."""
    if index < 26:
        return f"Speaker {chr(ord('A') + index)}"
    return f"Speaker {index + 1}"


def _cluster(
    pcm: np.ndarray, sr: int, turns: list[dict], embed, min_seconds: float,
) -> tuple[list[int], list[int], dict[int, np.ndarray]] | None:
    """Embed + force-2 split + pooled refinement over ``turns``.

    Returns ``(order, labels, centroids)`` — the embeddable turn indices, one
    0/1 label per embeddable turn, and the two pooled cluster centroids — or
    ``None`` when there is nothing trustworthy to cluster: fewer than two
    embeddable turns, or refinement empties a side (all utterances actually
    sound like one voice).
    """
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
    return order, labels, centroids


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
          "turns": [...],              # speaker relabeled; a turn that was
                                       # split arrives as TWO turns with the
                                       # text divided at a word boundary
          "num_speakers": 2,
          "source": "local-ecapa",
          "model": "<hf-source>@<pinned-revision>",
          "segments_total": int,       # after any word-level splitting
          "segments_embedded": int,
          "split_utterances": int,     # utterances split at a voice change
          "pooled_cosine": float,      # centroid similarity (low = distinct)
          "agreement_with_input": float,   # Rand agreement vs input labels
        }

    Turns may carry an optional ``words`` list ([{word, start_time,
    end_time}, ...]); it enables the word-level split pre-pass and is never
    propagated to the output turns.
    """
    embed = embed_fn or _default_embed
    try:
        clustered = _cluster(pcm, sr, turns, embed, min_seconds)
        if clustered is None:
            return None
        order, labels, centroids = clustered

        # Word-level pre-pass: a transcriber can weld a speaker handoff into
        # ONE utterance. With the pass-1 pooled centroids as anchors, split
        # long word-timed utterances at sustained voice changes, then
        # re-cluster over the finer segments so every piece is attributed
        # against pooled centroids like any other turn.
        turns, split_stats = split_long_utterances(
            pcm, sr, turns, embed, (centroids[0], centroids[1]),
        )
        if split_stats["split"]:
            clustered = _cluster(pcm, sr, turns, embed, min_seconds)
            if clustered is None:
                return None
            order, labels, centroids = clustered

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

    # Output turns are plain {speaker, text, start_time, end_time, ...} — the
    # internal ``words`` plumbing is not propagated.
    new_turns = [
        dict({k: v for k, v in t.items() if k != "words"},
             speaker=name_of[cluster_of[i]])
        for i, t in enumerate(turns)
    ]
    return {
        "turns": new_turns,
        "num_speakers": len(name_of),
        "source": SOURCE,
        "model": f"{speaker_id.ECAPA_SOURCE}@{speaker_id.ECAPA_REVISION}",
        "segments_total": len(turns),
        "segments_embedded": len(order),
        "split_utterances": split_stats["split"],
        "pooled_cosine": pooled_cosine,
        "agreement_with_input": partition_agreement(
            [t.get("speaker") for t in turns],
            [t["speaker"] for t in new_turns],
        ),
    }
