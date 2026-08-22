"""EXPERIMENTAL — sliding-window voice-change-point detection (spike, round 2).

NOT wired into production. This is a standalone prototype answering one
question: can a speaker-change detector that works directly on RAW AUDIO via
a sliding window (no transcript turns, no silence/gap requirement) recover
MORE real speakers than the current transcript-turn-relabeling architecture
in ``diarize_local.py``?

Technique (standard sliding-window / BIC-style speaker-change detection,
e.g. Chen & Gopalakrishnan 1998's BIC segmentation and its many embedding-
based descendants): slide a window of length ``window`` seconds across the
audio at a fixed ``hop``; embed every window with the SAME pinned ECAPA
model ``diarize_local``/``speaker_id`` already use; compare each window to
the window exactly ``window`` seconds ahead of it (i.e. two adjacent,
NON-overlapping windows swept across the audio at ``hop``-second
resolution) via cosine distance; flag a local peak in that distance curve as
a candidate speaker-change point. This is the two-window-compare structure
``diarize_local.find_change_point`` already uses (see its module docstring)
generalized to run over the WHOLE clip with no dependency on a transcript
utterance or a detected pause — the two-window pair can sit anywhere,
including entirely inside what a transcript calls one welded utterance.

Compute note (measured on this machine, 2026-08-22, CPU-only via the
isolated ``/tmp/ecapa_cache_exp1`` cache): each ``speaker_id.embed_pcm``
call costs ~15-20 seconds regardless of the (short) window length — the
model's fixed per-call overhead dominates, not audio duration. A sliding
window at hop=0.5s over a 30s clip needs ~58 embed calls (~17 minutes); at
hop=0.25s, ~115 calls (~35 minutes). This module is written to make the
embedding pass as cheap as possible for a given hop: each window position is
embedded EXACTLY ONCE (:func:`sliding_window_embeddings`) and reused for
every pairwise comparison that needs it (:func:`window_pair_distances`),
rather than re-embedding the same audio slice from two different candidate
boundaries.

Downstream: the candidate boundaries this module proposes are turned into
plain ``{start_time, end_time}`` turn dicts (:func:`boundaries_to_turns`, no
speaker/text — voice evidence only) and handed to
``diarize_local.diarize_turns`` UNCHANGED, so the existing pooling /
k-selection / validation acceptance gates (``MAX_POOLED_COSINE``,
``MIN_CLUSTER_SECONDS``, the marginal-split + anchor rules, etc.) are the
ones that ultimately decide how many speakers are believed. This module
never invents a speaker count itself — it only proposes WHERE the audio
might change voice.
"""

from __future__ import annotations

import numpy as np

import speaker_id


def sliding_window_embeddings(
    pcm: np.ndarray, sr: int, window: float, hop: float, embed,
) -> tuple[list[float], list[np.ndarray]]:
    """Embed a window of ``window`` seconds at every ``hop``-second position.

    Returns ``(starts, embeddings)`` — the window START times (seconds) and
    their L2-normalized embeddings, in order. Each position is embedded
    EXACTLY ONCE; callers needing a comparison between two window positions
    reuse these (see :func:`window_pair_distances`) rather than re-embedding.
    The final window is dropped if it would run past ``pcm``'s end.
    """
    n_samples = pcm.size
    win_samples = int(round(window * sr))
    hop_samples = int(round(hop * sr))
    if win_samples <= 0 or hop_samples <= 0:
        raise ValueError("window and hop must be positive")
    starts: list[float] = []
    embs: list[np.ndarray] = []
    i = 0
    while i + win_samples <= n_samples:
        chunk = pcm[i:i + win_samples]
        embs.append(speaker_id.l2_normalize(embed(np.ascontiguousarray(chunk), sr)))
        starts.append(i / sr)
        i += hop_samples
    return starts, embs


def window_pair_distances(
    starts: list[float], embs: list[np.ndarray], window: float, hop: float,
    *, tol: float = 1e-6,
) -> tuple[list[float], list[float]]:
    """Cosine DISTANCE (1 - cosine) between each window and the window
    starting exactly ``window`` seconds later (i.e. two back-to-back,
    non-overlapping windows). Pure math — no embedding calls.

    Returns ``(boundary_times, distances)`` where ``boundary_times[i]`` is
    the time between the compared pair (``starts[i] + window``) — the
    candidate speaker-change point that pair is evidence for. Requires
    ``window`` to be (within ``tol``) an integer multiple of ``hop`` so every
    left window has a matching right window on the same grid; raises
    ``ValueError`` otherwise (a caller wanting a different hop/window ratio
    should re-embed with matching ones — this keeps the embedding pass the
    cheap single-pass it's designed to be, never re-embedding for a
    mismatched comparison).
    """
    steps = window / hop
    if abs(steps - round(steps)) > tol:
        raise ValueError(
            f"window ({window}) must be an integer multiple of hop ({hop}) "
            "for the single-pass embedding reuse this function relies on"
        )
    stride = int(round(steps))
    times: list[float] = []
    dists: list[float] = []
    for i in range(len(starts) - stride):
        # starts are on a uniform hop grid by construction, so starts[i+stride]
        # is (within float rounding) starts[i] + window.
        d = 1.0 - float(np.dot(embs[i], embs[i + stride]))
        times.append(starts[i] + window)
        dists.append(d)
    return times, dists


def pick_change_points(
    times: list[float], distances: list[float],
    *, threshold: float, min_sep: float,
) -> list[float]:
    """Local peaks of ``distances`` at or above ``threshold``, kept only when
    no earlier-picked (higher) peak already claimed a point within
    ``min_sep`` seconds. Pure math. Greedy highest-first non-max suppression
    — the standard way to turn a noisy peak-ish curve into a sparse set of
    believed change points without doubling up on one true change (a genuine
    change elevates the distance over a stretch of candidates around it, not
    just a single sample).
    """
    ranked = sorted(zip(distances, times), reverse=True)
    chosen: list[float] = []
    for d, t in ranked:
        if d < threshold:
            break
        if all(abs(t - c) >= min_sep for c in chosen):
            chosen.append(t)
    return sorted(chosen)


def boundaries_to_turns(
    boundaries: list[float], duration: float, *, min_turn_seconds: float,
) -> list[dict]:
    """Sorted change-point times -> plain ``{start_time, end_time}`` turn
    dicts spanning ``[0, duration]``. Any resulting segment shorter than
    ``min_turn_seconds`` is merged into a neighbor (the shorter of its two
    neighbors' segments, so short slivers don't survive as standalone
    turns — they carry too little voice signal to embed on their own,
    mirroring ``diarize_local.MIN_SECONDS``'s role). No ``speaker``/``text``
    keys: these turns carry voice evidence only, not transcript content.
    """
    bounds = [0.0, *sorted(b for b in boundaries if 0.0 < b < duration), duration]
    i = 1
    while i < len(bounds) - 1:
        left_len = bounds[i] - bounds[i - 1]
        right_len = bounds[i + 1] - bounds[i]
        if min(left_len, right_len) < min_turn_seconds:
            # Drop whichever neighboring boundary yields the smaller merged
            # segment's shorter piece — simplest honest rule: merge this
            # segment into its shorter neighbor by deleting the boundary on
            # that side.
            if left_len <= right_len:
                del bounds[i - 1]
                i = max(i - 1, 1)
            else:
                del bounds[i]
        else:
            i += 1
    return [
        {"start_time": round(s, 4), "end_time": round(e, 4)}
        for s, e in zip(bounds[:-1], bounds[1:])
    ]


def detect_change_points(
    pcm: np.ndarray, sr: int, embed, *, window: float = 1.5, hop: float = 0.5,
    threshold: float, min_sep: float | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Full pipeline: embed, compare, pick peaks. Returns ``(boundary_times,
    all_times, all_distances)`` — the accepted change points plus the full
    distance curve (for diagnostics/plotting). ``min_sep`` defaults to
    ``window`` (a true change elevates distance over roughly a window's
    width of candidates around it)."""
    starts, embs = sliding_window_embeddings(pcm, sr, window, hop, embed)
    times, dists = window_pair_distances(starts, embs, window, hop)
    boundaries = pick_change_points(
        times, dists, threshold=threshold, min_sep=min_sep or window,
    )
    return boundaries, times, dists


def detect_turns_from_audio(
    pcm: np.ndarray, sr: int, embed=None, *,
    window: float = 1.5, hop: float = 0.5, threshold: float,
    min_sep: float | None = None, min_turn_seconds: float = 1.0,
) -> dict:
    """Orchestrator: raw PCM -> candidate turn boundaries from voice evidence
    alone. Returns ``{"turns": [...], "boundaries": [...], "curve": {"times":
    [...], "distances": [...]}}``. ``embed`` defaults to
    ``speaker_id.embed_pcm`` (real ECAPA); tests inject a stub."""
    embed = embed or speaker_id.embed_pcm
    boundaries, times, dists = detect_change_points(
        pcm, sr, embed, window=window, hop=hop, threshold=threshold,
        min_sep=min_sep,
    )
    duration = pcm.size / sr
    turns = boundaries_to_turns(
        boundaries, duration, min_turn_seconds=min_turn_seconds,
    )
    return {
        "turns": turns,
        "boundaries": boundaries,
        "curve": {"times": times, "distances": dists},
    }
