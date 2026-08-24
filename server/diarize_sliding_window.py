"""EXPERIMENTAL — sliding-window voice-change-point detection (round 3).

NOT wired into production. Standalone module answering one question: can a
speaker-change detector that works directly on RAW AUDIO via a sliding
window (no transcript turns, no silence/gap requirement) recover MORE real
speakers than the current transcript-turn-relabeling architecture in
``diarize_local.py``?

History (see ``.superpowers/sdd/2026-08-22-poker6-v3-sliding-window-refine/``
for the full write-up):

* Round 2 built the first version of this module: slide a window across the
  audio, embed every position once, and flag a local peak in RAW
  window-to-window cosine DISTANCE (two adjacent, non-overlapping windows
  compared directly) as a candidate change point (:func:`window_pair_distances`
  + :func:`pick_change_points`, both KEPT below for compatibility/tests but no
  longer the default path). Result: recovered all 6 real speakers on the
  target fixture for the first time (vs. 4 from the existing architecture),
  but only 64% per-turn accuracy — round 2's own diagnosis: a SINGLE noisy
  point-to-point cosine is not a stable enough signal to threshold reliably;
  a threshold picked from one fixture's curve didn't transfer to the other.
  It was also expensive: ~58 separate ``embed_pcm`` calls (~17 minutes) for
  one 30-second clip's embedding pass alone.
* Round 3 (this version) fixes both problems:

  1. **Noise/calibration** — :func:`centroid_margin_change_points` replaces
     raw window-pair comparison with the same TECHNIQUE
     ``diarize_local.find_change_point``/``_select_k`` already use
     successfully: compare POOLED, multi-sample centroids instead of single
     noisy points. Concretely, an online growing-segment scan maintains a
     "current speaker so far" centroid (the mean of every window embedding
     since the last accepted change) and, at each hop, a "candidate new
     speaker" centroid (the mean of the next ``lookahead_windows`` window
     embeddings) and scores the candidate boundary by 1 - cosine between the
     two POOLED centroids. A change is accepted only on a SUSTAINED run of
     ``min_run`` consecutive hops at or above ``threshold`` (mirroring
     ``diarize_local._sustained_flip``'s guard against a lone noisy flip),
     at which point the running segment resets and scanning continues.
     Averaging multiple embeddings into each side of the comparison is what
     suppresses the single-window noise round 2's report identified as the
     root cause: calibrated on both real fixtures' actual embedding data
     (``tmp/exp/centroid_calibrate2.py``, not committed — throwaway
     calibration script, results captured in the report), pooled-centroid
     distances at genuine changes separate much more cleanly from
     within-segment noise than raw single-window pairs did.
  2. **Cost** — :func:`sliding_window_embeddings_batched` replaces the
     one-``embed_pcm``-call-per-window loop with ONE call to
     :func:`speaker_id.embed_pcm_batch`, which stacks every window into a
     single batched forward pass through the same pinned ECAPA model. Real
     measured numbers are in the report; the model's fixed per-call
     dispatch/setup overhead (the dominant cost per round 2's own
     measurement) is paid once per BATCH instead of once per window.

Downstream: the candidate boundaries this module proposes are turned into
plain ``{start_time, end_time}`` turn dicts (:func:`boundaries_to_turns`, no
speaker/text — voice evidence only) and handed to
``diarize_local.diarize_turns`` UNCHANGED, so the existing pooling /
k-selection / validation acceptance gates (``MAX_POOLED_COSINE``,
``MIN_CLUSTER_SECONDS``, the marginal-split + anchor rules, etc.) are the
ones that ultimately decide how many speakers are believed. This module
never invents a speaker count itself — it only proposes WHERE the audio
might change voice. Over-segmentation here (an extra turn boundary inside
one real speaker's speech) is CHEAP: ``diarize_local``'s pooled-centroid
clustering will merge same-speaker turns back together on its own evidence.
Under-segmentation (missing a real change, welding two voices into one
turn) is EXPENSIVE: it corrupts that turn's embedding and can corrupt
whatever cluster inherits it. The round-3 defaults are deliberately tuned
to favor recall of true changes over precision, for exactly this reason.
"""

from __future__ import annotations

import numpy as np

import speaker_id


def l2n(vec: np.ndarray) -> np.ndarray:
    """L2-normalize; a zero vector is returned as-is (matches
    ``speaker_id.l2_normalize`` — duplicated locally as trivial pure math so
    this module's core scan has no import-time dependency beyond numpy)."""
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


# ---------------------------------------------------------------------------
# Embedding passes (the ONLY functions that call the ECAPA model)
# ---------------------------------------------------------------------------

def sliding_window_embeddings(
    pcm: np.ndarray, sr: int, window: float, hop: float, embed,
) -> tuple[list[float], list[np.ndarray]]:
    """Embed a window of ``window`` seconds at every ``hop``-second position,
    ONE ``embed`` CALL PER WINDOW.

    Returns ``(starts, embeddings)`` — the window START times (seconds) and
    their L2-normalized embeddings, in order. The final window is dropped if
    it would run past ``pcm``'s end. KEPT for callers that inject a cheap
    stub ``embed`` (tests) or need per-window control; production callers
    with the REAL ECAPA model should prefer
    :func:`sliding_window_embeddings_batched`, which does the same windowing
    but in one batched forward pass (see module docstring, cost problem 2).
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


def _window_slices(pcm: np.ndarray, sr: int, window: float, hop: float) -> tuple[list[float], list[np.ndarray]]:
    """Pure slicing (no embedding): every window's start time + raw PCM
    chunk. Shared by both the looped and batched embedding passes so the
    windowing math lives in exactly one place."""
    n_samples = pcm.size
    win_samples = int(round(window * sr))
    hop_samples = int(round(hop * sr))
    if win_samples <= 0 or hop_samples <= 0:
        raise ValueError("window and hop must be positive")
    starts: list[float] = []
    chunks: list[np.ndarray] = []
    i = 0
    while i + win_samples <= n_samples:
        chunks.append(np.ascontiguousarray(pcm[i:i + win_samples]))
        starts.append(i / sr)
        i += hop_samples
    return starts, chunks


def sliding_window_embeddings_batched(
    pcm: np.ndarray, sr: int, window: float, hop: float, embed_batch=None,
    *, max_batch: int = 128,
) -> tuple[list[float], list[np.ndarray]]:
    """Same windowing as :func:`sliding_window_embeddings`, but embeds ALL
    windows in as FEW model calls as possible via ``embed_batch`` (defaults
    to :func:`speaker_id.embed_pcm_batch`) instead of one call per window —
    this is the round-3 cost fix (see module docstring). ``max_batch`` caps
    how many windows go into a single ``embed_batch`` call (memory safety
    for very long recordings — a 30s clip at the default grid needs ~58
    windows, well under the cap, so this fixture-sized case is ONE call).
    Returns the same ``(starts, embeddings)`` shape as the looped version —
    a drop-in replacement, not a different contract.
    """
    embed_batch = embed_batch or speaker_id.embed_pcm_batch
    starts, chunks = _window_slices(pcm, sr, window, hop)
    embs: list[np.ndarray] = []
    for i in range(0, len(chunks), max_batch):
        batch = chunks[i:i + max_batch]
        embs.extend(speaker_id.l2_normalize(v) for v in embed_batch(batch, sr))
    return starts, embs


# ---------------------------------------------------------------------------
# Round 2 baseline: raw window-pair cosine distance (KEPT for tests/
# comparison; no longer the default change-point technique — see module
# docstring for why).
# ---------------------------------------------------------------------------

def window_pair_distances(
    starts: list[float], embs: list[np.ndarray], window: float, hop: float,
    *, tol: float = 1e-6,
) -> tuple[list[float], list[float]]:
    """Cosine DISTANCE (1 - cosine) between each window and the window
    starting exactly ``window`` seconds later. Pure math — no embedding
    calls. ROUND 2 BASELINE — noisy single-point comparison, superseded by
    :func:`centroid_margin_change_points` as the default (see module
    docstring); kept for tests and side-by-side comparison.

    Returns ``(boundary_times, distances)``. Requires ``window`` to be
    (within ``tol``) an integer multiple of ``hop``; raises ``ValueError``
    otherwise.
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
        d = 1.0 - float(np.dot(embs[i], embs[i + stride]))
        times.append(starts[i] + window)
        dists.append(d)
    return times, dists


def pick_change_points(
    times: list[float], distances: list[float],
    *, threshold: float, min_sep: float,
) -> list[float]:
    """Local peaks of ``distances`` at or above ``threshold`` via greedy
    highest-first non-max suppression. ROUND 2 BASELINE — pairs with
    :func:`window_pair_distances`; kept for tests/comparison."""
    ranked = sorted(zip(distances, times), reverse=True)
    chosen: list[float] = []
    for d, t in ranked:
        if d < threshold:
            break
        if all(abs(t - c) >= min_sep for c in chosen):
            chosen.append(t)
    return sorted(chosen)


# ---------------------------------------------------------------------------
# Round 3: centroid-anchored margin change-point detection (the fix for
# problem 1 — see module docstring for the full rationale and calibration
# pointer).
# ---------------------------------------------------------------------------

# Calibrated 2026-08-22 against BOTH real fixtures' actual embedding data
# (family_real: precise 5-change ground truth; poker6_real: coarse ~5s-grid
# ground truth, +/-1-2s per the fixture's own note) using the SAME window=1.5/
# hop=0.5 embedding grid round 2 used — this is a scoring/decision change
# only, not a new embedding parameter surface. One setting recovered every
# true change on BOTH fixtures (a few extra same-speaker turn splits are
# tolerated by design — see module docstring on over- vs under-segmentation).
DEFAULT_MIN_SEGMENT_WINDOWS = 4  # 2.0s of accumulated "current speaker" evidence before a boundary test is trusted
DEFAULT_LOOKAHEAD_WINDOWS = 3    # 1.5s pooled into the "candidate new speaker" centroid
DEFAULT_CENTROID_THRESHOLD = 0.65  # 1-cosine; cosine ~0.35, comfortably inside diarize_local's own cross-speaker pooled-cosine territory (~0.19-0.34) once averaged
DEFAULT_MIN_RUN = 2             # consecutive qualifying hops required (1.0s sustained) -- a lone spike is noise


def centroid_margin_curve(
    starts: list[float], embs: list[np.ndarray],
    *, min_segment_windows: int = DEFAULT_MIN_SEGMENT_WINDOWS,
    lookahead_windows: int = DEFAULT_LOOKAHEAD_WINDOWS,
) -> tuple[list[float], list[float]]:
    """Diagnostic-only: the pooled-centroid distance curve WITHOUT the
    stateful reset walk (segment always grown from index 0). Useful for
    plotting/inspection; :func:`centroid_margin_change_points` is the real
    (stateful, reset-on-accept) detector production code should use — this
    function's non-resetting curve degrades on later changes in a
    many-speaker recording (the "current" centroid keeps diluting with every
    past speaker), which is exactly why the real detector resets."""
    n = len(embs)
    times: list[float] = []
    dists: list[float] = []
    i = min_segment_windows
    while i + lookahead_windows <= n:
        left = l2n(np.mean(embs[0:i], axis=0))
        right = l2n(np.mean(embs[i:i + lookahead_windows], axis=0))
        times.append(starts[i])
        dists.append(1.0 - float(np.dot(left, right)))
        i += 1
    return times, dists


def centroid_margin_change_points(
    starts: list[float], embs: list[np.ndarray],
    *, min_segment_windows: int = DEFAULT_MIN_SEGMENT_WINDOWS,
    lookahead_windows: int = DEFAULT_LOOKAHEAD_WINDOWS,
    threshold: float = DEFAULT_CENTROID_THRESHOLD,
    min_run: int = DEFAULT_MIN_RUN,
) -> list[float]:
    """The round-3 change-point detector. Pure math (no embedding calls) —
    walks the ALREADY-EMBEDDED window positions once.

    Stateful online scan: maintain a "current speaker so far" pooled
    centroid (the mean of every window embedding since the last accepted
    change, or the clip start) and, at each hop, a "candidate new speaker"
    pooled centroid (the mean of the next ``lookahead_windows`` window
    embeddings). Score = 1 - cosine(current, candidate). A change is
    accepted at the START of a run of ``min_run`` CONSECUTIVE hops scoring
    at or above ``threshold`` — a lone elevated hop is noise, a sustained
    run is a genuine voice change (same sustained-evidence principle as
    ``diarize_local._sustained_flip``). On acceptance the "current speaker"
    segment resets to start at the accepted boundary (the lookahead
    evidence that triggered acceptance becomes the seed of the new
    segment) and scanning continues — this reset is what keeps the
    "current" centroid representative of ONE speaker instead of slowly
    diluting across every speaker heard so far (see
    :func:`centroid_margin_curve`'s docstring for what happens without it).

    A fresh segment must accumulate ``min_segment_windows`` of its own
    evidence before it is tested against again — mirroring
    ``diarize_local.MIN_CLUSTER_SECONDS``'s "not enough evidence yet" idea —
    so a change is never proposed against a still-forming, too-short
    reference.
    """
    n = len(embs)
    boundaries: list[float] = []
    seg_start = 0
    i = min_segment_windows
    run_start: int | None = None
    run_len = 0
    while i + lookahead_windows <= n:
        left = l2n(np.mean(embs[seg_start:i], axis=0))
        right = l2n(np.mean(embs[i:i + lookahead_windows], axis=0))
        dist = 1.0 - float(np.dot(left, right))
        if dist >= threshold:
            if run_len == 0:
                run_start = i
            run_len += 1
        else:
            run_len = 0
            run_start = None
        if run_len >= min_run:
            boundaries.append(starts[run_start])
            seg_start = run_start
            i = seg_start + min_segment_windows
            run_len = 0
            run_start = None
            continue
        i += 1
    return boundaries


# ---------------------------------------------------------------------------
# Shared post-processing + orchestrators
# ---------------------------------------------------------------------------

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


def detect_turns_from_audio(
    pcm: np.ndarray, sr: int, embed_batch=None, *,
    window: float = 1.5, hop: float = 0.5,
    min_segment_windows: int = DEFAULT_MIN_SEGMENT_WINDOWS,
    lookahead_windows: int = DEFAULT_LOOKAHEAD_WINDOWS,
    threshold: float = DEFAULT_CENTROID_THRESHOLD,
    min_run: int = DEFAULT_MIN_RUN,
    min_turn_seconds: float = 1.0,
) -> dict:
    """Round-3 orchestrator: raw PCM -> candidate turn boundaries from voice
    evidence alone, using the BATCHED embedding pass (cost fix) and the
    CENTROID-MARGIN change-point detector (noise fix). ``embed_batch``
    defaults to ``speaker_id.embed_pcm_batch`` (real ECAPA); tests inject a
    stub. Returns ``{"turns": [...], "boundaries": [...], "curve": {"times":
    [...], "distances": [...]}}`` — the diagnostic ``curve`` is the
    non-resetting :func:`centroid_margin_curve` (for inspection only; the
    actual accepted ``boundaries`` come from the stateful detector).
    """
    starts, embs = sliding_window_embeddings_batched(pcm, sr, window, hop, embed_batch)
    boundaries = centroid_margin_change_points(
        starts, embs, min_segment_windows=min_segment_windows,
        lookahead_windows=lookahead_windows, threshold=threshold, min_run=min_run,
    )
    times, dists = centroid_margin_curve(
        starts, embs, min_segment_windows=min_segment_windows,
        lookahead_windows=lookahead_windows,
    )
    duration = pcm.size / sr
    turns = boundaries_to_turns(boundaries, duration, min_turn_seconds=min_turn_seconds)
    return {
        "turns": turns,
        "boundaries": boundaries,
        "curve": {"times": times, "distances": dists},
    }


def detect_turns_from_audio_legacy(
    pcm: np.ndarray, sr: int, embed=None, *,
    window: float = 1.5, hop: float = 0.5, threshold: float,
    min_sep: float | None = None, min_turn_seconds: float = 1.0,
) -> dict:
    """ROUND 2 BASELINE orchestrator (raw window-pair distance, one embed
    call per window) — kept for tests and side-by-side comparison against
    the round-3 default (:func:`detect_turns_from_audio`). Not used by
    production or by any new caller."""
    embed = embed or speaker_id.embed_pcm
    starts, embs = sliding_window_embeddings(pcm, sr, window, hop, embed)
    times, dists = window_pair_distances(starts, embs, window, hop)
    boundaries = pick_change_points(times, dists, threshold=threshold, min_sep=min_sep or window)
    duration = pcm.size / sr
    turns = boundaries_to_turns(boundaries, duration, min_turn_seconds=min_turn_seconds)
    return {
        "turns": turns,
        "boundaries": boundaries,
        "curve": {"times": times, "distances": dists},
    }
