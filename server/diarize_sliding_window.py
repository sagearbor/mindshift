"""Sliding-window voice-change-point detection (round 3) + the spectral
window-clustering tools ``diarize_local`` uses in production (2026-08-29).

The round-2/3 change-point DETECTORS below (:func:`window_pair_distances`,
:func:`centroid_margin_change_points`, :func:`detect_turns_from_audio`) are
still experimental and NOT wired into production. What IS production since
the 2026-08-29 voice-separation bake-off (docs/research/2026-08-29-voice-
separation/, approach B) is the last section of this module — the refined
cosine affinity, eigengap speaker count, spectral labels and run smoothing
over window embeddings — which ``diarize_local``'s window pass calls for its
boundary proposals, eigengap lower bound and spectral fallback partition —
and which, since 2026-08-30, IS the speaker labelling under production's
default engine (``diarize_local.diarize_windows_first``, MINDSHIFT_DIARIZE_
ENGINE=windows: spectral labels at the eigengap k → :func:`mode_filter` →
:func:`window_label_runs` → segments the transcript's words are regrouped
by).

The original round-3 question: can a
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


# ---------------------------------------------------------------------------
# 2026-08-29: spectral clustering of window embeddings (bake-off approach B,
# docs/research/2026-08-29-voice-separation/B-sliding-window/). PRODUCTION
# code now — ``diarize_local`` uses these pure-math pieces for (1) boundary
# proposals inside long utterances, (2) the eigengap speaker-count lower
# bound and (3) the spectral fallback partition in its k-selection. Every
# function below is torch-free numpy; the only model calls stay in the
# embedding passes above.
# ---------------------------------------------------------------------------

# Row-wise percentile of the affinity refinement (Wang et al. 2018, "Speaker
# diarization with LSTM"): entries under each row's p-quantile are damped
# x0.01 before symmetrizing + diffusing. Swept 0.95/0.90/0.80/0.70 on eight
# fixtures (2026-08-29): 0.80 was the only setting that found the right k on
# every 2-3-voice fixture (family_real 2, maggiano3 3, family3 3, the TTS
# pairs 2) — 0.95 shattered family_real into 8 and collapsed maggiano3 to 1.
SPECTRAL_PERCENTILE = 0.80

# Mode filter over +/- this many HOPS of temporal neighbours, then label runs
# shorter than SPECTRAL_MIN_RUN_SECONDS are absorbed into the longer
# neighbour (B's smoothing; a 1.5 s window cannot honestly resolve a shorter
# voice run anyway).
SPECTRAL_SMOOTH_HOPS = 2
SPECTRAL_MIN_RUN_SECONDS = 0.5

# k-means restarts for the spectral embedding (seeded, so results are
# deterministic run to run).
_KMEANS_RESTARTS = 10


def refine_affinity(embs: np.ndarray, p: float = SPECTRAL_PERCENTILE) -> np.ndarray:
    """Refined cosine affinity of L2-normalized window embeddings.

    Wang et al. 2018: diagonal = row max, row-wise percentile thresholding
    (entries under the row's ``p``-quantile are damped x0.01), symmetrize by
    max, diffusion ``A @ A.T``, row-max normalization. The Gaussian blur
    step is skipped — windows overlap heavily already. Pure numpy.
    """
    embs = np.asarray(embs, dtype=np.float64)
    a = embs @ embs.T
    np.fill_diagonal(a, 0.0)
    np.fill_diagonal(a, a.max(axis=1))
    thr = np.percentile(a, p * 100.0, axis=1, keepdims=True)
    a = np.where(a >= thr, a, a * 0.01)
    a = np.maximum(a, a.T)
    a = a @ a.T
    a = a / np.maximum(a.max(axis=1, keepdims=True), 1e-9)
    return a


def eigengap_k(affinity: np.ndarray, max_k: int) -> tuple[int, list[float]]:
    """``(k, eigenvalues)`` — k = argmax over 1..max_k of lambda_k / lambda_{k+1}
    on the symmetrized refined affinity's descending eigenvalues (clipped at
    1e-9 so a rank-deficient affinity — every window the same voice — yields
    k=1 rather than a division by zero)."""
    w = np.linalg.eigvalsh((affinity + affinity.T) / 2.0)[::-1]
    w = np.clip(w, 1e-9, None)
    max_k = max(1, min(int(max_k), len(w) - 1))
    ratios = [float(w[i] / w[i + 1]) for i in range(max_k)]
    return int(np.argmax(ratios)) + 1, [round(float(x), 6) for x in w[: max_k + 1]]


def _kmeans(feats: np.ndarray, k: int, *, restarts: int = _KMEANS_RESTARTS,
            seed: int = 0, iters: int = 100) -> np.ndarray:
    """Seeded k-means++ (best of ``restarts`` by inertia). Pure numpy — the
    production image has no scikit-learn."""
    n = feats.shape[0]
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    best_labels = np.zeros(n, dtype=int)
    best_inertia = np.inf
    for _ in range(restarts):
        centers = np.empty((k, feats.shape[1]), dtype=np.float64)
        centers[0] = feats[rng.integers(n)]
        d2 = np.sum((feats - centers[0]) ** 2, axis=1)
        for j in range(1, k):
            tot = float(d2.sum())
            idx = rng.integers(n) if tot <= 0 else rng.choice(n, p=d2 / tot)
            centers[j] = feats[idx]
            d2 = np.minimum(d2, np.sum((feats - centers[j]) ** 2, axis=1))
        labels = np.zeros(n, dtype=int)
        for _ in range(iters):
            dist = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new = dist.argmin(axis=1)
            if np.array_equal(new, labels) and _ > 0:
                break
            labels = new
            for j in range(k):
                members = feats[labels == j]
                if members.size:
                    centers[j] = members.mean(axis=0)
        dist = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        inertia = float(dist[np.arange(n), labels].sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.copy()
    return best_labels


def spectral_labels(affinity: np.ndarray, k: int) -> np.ndarray:
    """k-way spectral clustering of a refined affinity: k-means on the
    row-normalized top-k eigenvectors. ``k <= 1`` → every window label 0."""
    n = affinity.shape[0]
    k = max(1, min(int(k), n))
    if k == 1:
        return np.zeros(n, dtype=int)
    _, v = np.linalg.eigh((affinity + affinity.T) / 2.0)
    feats = v[:, ::-1][:, :k]
    feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-9)
    return _kmeans(feats, k)


def mode_filter(labels: np.ndarray, starts: list[float] | np.ndarray, hop: float,
                *, radius: int = SPECTRAL_SMOOTH_HOPS) -> np.ndarray:
    """Mode over each window's TEMPORAL neighbours within ``radius`` hops (a
    VAD gap breaks the neighbourhood). Ties keep the window's own label."""
    labels = np.asarray(labels)
    starts = np.asarray(starts, dtype=np.float64)
    out = labels.copy()
    for i in range(len(labels)):
        nb = labels[np.abs(starts - starts[i]) <= radius * hop + 1e-6]
        vals, cnt = np.unique(nb, return_counts=True)
        best = vals[cnt == cnt.max()]
        out[i] = labels[i] if labels[i] in best else best[0]
    return out


def window_label_runs(
    labels: np.ndarray, starts: list[float] | np.ndarray, window: float,
    lo: float, hi: float, *, min_run: float = SPECTRAL_MIN_RUN_SECONDS,
    step: float = 0.01,
) -> list[list]:
    """Window labels → ``[[start, end, label], ...]`` runs covering [lo, hi].

    Every ``step`` frame takes the label of the nearest window CENTRE (gaps
    inherit, nothing is left unlabelled); adjacent same-label frames merge;
    runs shorter than ``min_run`` are absorbed into the longer neighbour,
    shortest first, until every run is at least ``min_run`` (or one run is
    left). No windows → one run.
    """
    labels = np.asarray(labels)
    starts = np.asarray(starts, dtype=np.float64)
    n = max(1, int(round((hi - lo) / step)))
    if len(starts) == 0:
        return [[lo, hi, 0]]
    centres = starts + window / 2.0
    t = lo + (np.arange(n) + 0.5) * step
    frame_labels = labels[np.abs(t[:, None] - centres[None, :]).argmin(axis=1)]
    runs: list[list] = []
    s = 0
    for i in range(1, n + 1):
        if i == n or frame_labels[i] != frame_labels[s]:
            runs.append([lo + s * step, hi if i == n else lo + i * step, int(frame_labels[s])])
            s = i
    while len(runs) > 1:
        lens = [e - b for b, e, _ in runs]
        i = int(np.argmin(lens))
        if lens[i] >= min_run:
            break
        cand = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        j = max(cand, key=lambda j: runs[j][1] - runs[j][0])
        runs[i][2] = runs[j][2]
        merged: list[list] = []
        for b, e, lab in runs:
            if merged and merged[-1][2] == lab:
                merged[-1][1] = e
            else:
                merged.append([b, e, lab])
        runs = merged
    return runs
