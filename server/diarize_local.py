"""Local speaker diarization — vendor-independent "who said each utterance".

Why this exists: Deepgram's nova-3 (model 2025-07-31.0) regressed prerecorded
diarization — it merged two distinct voices into one speaker on real recordings
(verified 2026-08-05 by direct nova-2/nova-3 comparison on identical bytes).
Renting diarization means a silent vendor model swap can corrupt every
per-speaker feature downstream (talk-share, heat, report cards). This module
re-derives speaker labels ON OUR OWN COMPUTE with the same PINNED ECAPA model
that already powers voice enrollment (``speaker_id``).

Algorithm — calibrated on the real recording that exposed the regression plus
the TTS fixture (2026-08-06), extended to N-way k-selection on the real
3-person recording that exposed the forced-2 limit (2026-08-14):

1. Embed each transcript utterance that is long enough to carry voice signal.
   Per-utterance embeddings alone are NOISY — same-speaker cosine can dip
   below cross-speaker — so a plain similarity-threshold clustering
   over-fragments (it heard 4-5 "speakers" in a 2-person recording).
2. For each candidate speaker count k = 2 .. :data:`MAX_SPEAKERS_LOCAL`:
   average-linkage merge to exactly k clusters, then REFINE: embed each
   cluster's POOLED audio (pooled embeddings are what ``speaker_id``'s
   calibration table trusts: same voice ≈0.73, different ≈0.19) and reassign
   every utterance to its closest pooled centroid, a few rounds until stable.
3. VALIDATE each k before believing it: EVERY pair of pooled centroids must be
   clearly different voices (cosine ≤ :data:`MAX_POOLED_COSINE`); for k > 2
   the pair(s) the marginal split CREATED (vs the refined k-1 partition) must
   be VERY clearly different voices (≤ :data:`STRONG_SEPARATION_COSINE`) —
   measured on the real couple recording, ONE voice heard in calm + shouting
   registers forms two well-fed clusters at pooled cosine 0.359 that no
   seconds floor can reject, while the real 3-person recording's genuine
   third voice split off at 0.267 — AND the split must be ANCHORED by a half
   that is wildly unlike some other cluster (≤
   :data:`NEW_VOICE_ANCHOR_COSINE`; the TTS fixture's phantom split-pair
   0.277 is indistinguishable from the genuine 0.267, but its halves anchor
   at 0.216+ where the real child anchors at -0.017); and each cluster must
   carry enough pooled
   speech to be trustworthy — the full :data:`MIN_CLUSTER_SECONDS` normally,
   relaxed to :data:`MIN_CLUSTER_SECONDS_STRONG` only for a cluster that is
   VERY clearly distinct from every other centroid (all its pairwise cosines
   ≤ :data:`STRONG_SEPARATION_COSINE`) — a quiet third participant with one
   clean utterance is real evidence, a moderately-separated sliver is not.
   The LARGEST fully-validating k wins; a genuine monologue measures ≈0.73
   pooled self-similarity, validates at NO k, and is REJECTED — we never
   invent a speaker.

Scope + honesty:

* Speaker counts up to :data:`MAX_SPEAKERS_LOCAL` are attempted — enough for
  the recordings this app targets (family/couple conversations), not general
  N-speaker diarization. Every k tried is reported in ``k_evaluated`` so logs
  show why a count was chosen.
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

# Accept a k-way split only when EVERY pair of clusters' POOLED embeddings is
# at most this similar. Calibration (2026-08-06, pinned ECAPA): different
# people pooled ≈0.19-0.26; the same real voice split in half ≈0.73;
# speaker_id's table puts merged/degraded artifacts at ≈0.48-0.56. 0.45 sits
# under all observed same-voice values with margin. Env-overridable for
# recalibration.
MAX_POOLED_COSINE = float(os.getenv("MINDSHIFT_DIARIZE_MAX_POOLED_COSINE", "0.45"))

# Candidate speaker counts run k = 2 .. MAX_SPEAKERS_LOCAL (also capped by the
# number of embeddable utterances). 4 covers the family recordings this app
# targets while keeping the spurious-split surface small.
MAX_SPEAKERS_LOCAL = 4

# An utterance shorter than this is not embedded (too little voice signal); it
# inherits the nearest embedded utterance's cluster (nearest by midpoint).
MIN_SECONDS = 1.0

# Each cluster of an accepted split must have at least this much pooled
# speech — a "second voice" carried by one breath of audio is not evidence.
MIN_CLUSTER_SECONDS = 3.0

# "VERY clearly a different voice" — a stricter bar than the accept gate,
# used twice:
#
# 1. MARGINAL-SPLIT RULE: claiming k+1 speakers over k asserts that one of
#    k's clusters is really TWO voices; the pair that split creates must
#    measure at or below this bar, else it is one voice in two registers.
#    Calibration (2026-08-14, both real recordings, pinned ECAPA): the
#    GENUINE marginal split (the 3-person recording's third voice, a child
#    with 1.9s of solo speech) measured 0.267 against the cluster it split
#    from, while every SPURIOUS split measured 0.359 (couple recording k=3 —
#    ONE voice heard in calm + shouting registers, 6+s per half, so no
#    seconds floor can reject it), 0.391 (couple k=4) and 0.402 (3-person
#    k=4). Any bar in (0.267, 0.359) separates them; 0.30 sits on the
#    conservative (2-speaker-protecting) side. NOTE a global pairwise gate
#    CANNOT do this job: the couple's spurious k=3 max pair (0.359) is
#    nearly identical to the 3-person's genuine adult-adult pair (0.341).
# 2. EVIDENCE-FLOOR RELAXATION: a cluster whose EVERY pairwise cosine is at
#    or below this bar may carry MIN_CLUSTER_SECONDS_STRONG of speech
#    instead of the full MIN_CLUSTER_SECONDS (the real third voice above:
#    pairwise -0.017 / 0.267, only 1.9s of solo speech — real evidence).
#
# Env-overridable for recalibration.
STRONG_SEPARATION_COSINE = float(
    os.getenv("MINDSHIFT_DIARIZE_STRONG_SEPARATION_COSINE", "0.30")
)
MIN_CLUSTER_SECONDS_STRONG = 1.5

# ANCHOR RULE for a marginal split: the split-pair bar alone cannot separate
# a genuine new voice from a noisy same-voice split — measured split pairs
# are 0.267 (the real third voice, a child) vs 0.277 (the TTS fixture's
# phantom split), a 0.010 window no honest threshold fits. What separates
# them robustly: a GENUINE new voice announces itself by being wildly unlike
# at least one established cluster (the child vs her father: -0.017), while
# BOTH halves of a phantom split sit moderately far from everything (TTS
# fixture halves: 0.216 / 0.238 vs the other voice; couple recording:
# 0.221 / 0.277). So at least one half of a marginal split must have ALL its
# cosines to NON-sibling clusters at or below this anchor bar — window
# (-0.017, 0.216), 0.15 sits inside with margin on both sides. CONSEQUENCE
# (honest tradeoff, calibrated on ONE real 3-person file whose third voice
# is a child): three typical ADULTS (different-people pooled pairs measure
# ≈0.19-0.34) may fail to anchor and stay a 2-way split — the conservative
# failure direction, since the transcript's own diarization usually hears 3
# adults and the never-reduce guard keeps them. Env-overridable.
NEW_VOICE_ANCHOR_COSINE = float(
    os.getenv("MINDSHIFT_DIARIZE_NEW_VOICE_ANCHOR_COSINE", "0.15")
)

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


def _merge_to_k(embeddings: list[np.ndarray], k: int) -> list[int]:
    """Average-linkage merging until exactly ``k`` clusters remain.

    Returns a 0..k-1 label per embedding, 0 = the cluster containing the
    earliest input. O(n^3) worst case — fine at transcript scale (≤ 400
    turns).
    """
    n = len(embeddings)
    sim = np.array([[float(np.dot(a, b)) for b in embeddings] for a in embeddings])
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > k:
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


def _embed_turns(
    pcm: np.ndarray, sr: int, turns: list[dict], embed, min_seconds: float,
) -> tuple[list[int], list[np.ndarray]] | None:
    """Embed every turn long enough to carry voice signal.

    Returns ``(order, embs)`` — the embeddable turn indices and their
    normalized embeddings — or ``None`` with fewer than two embeddable turns
    (nothing trustworthy to cluster).
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
    return order, [embedded[i] for i in order]


def _refine_k(
    pcm: np.ndarray, sr: int, turns: list[dict], embed,
    order: list[int], embs: list[np.ndarray], k: int,
) -> tuple[list[int], dict[int, np.ndarray]] | None:
    """Force-k split + pooled-centroid refinement.

    Returns ``(labels, centroids)`` — one cluster label per embeddable turn
    and the k pooled cluster centroids — or ``None`` when refinement empties
    a cluster (the audio does not actually hold k distinct voices).
    """
    labels = _merge_to_k(embs, k)
    centroids: dict[int, np.ndarray] = {}
    for _ in range(REFINE_ROUNDS):
        centroids = {}
        for c in set(labels):
            idxs = [order[i] for i in range(len(order)) if labels[i] == c]
            centroids[c] = speaker_id.l2_normalize(
                embed(_pooled(pcm, sr, turns, idxs), sr)
            )
        if len(centroids) < k:
            return None
        new = [
            max(centroids, key=lambda c: float(np.dot(embs[i], centroids[c])))
            for i in range(len(embs))
        ]
        if new == labels:
            break
        labels = new
    if len(set(labels)) < k:
        return None
    return labels, centroids


def _marginal_pairs(
    labels: list[int], prev_labels: list[int], weights: list[float],
) -> list[tuple[int, int]]:
    """Cluster pairs the k-th split CREATED, vs the k-1 partition.

    Each current cluster's parent is the previous cluster holding the
    majority of its speech (weighted by ``weights``, seconds per embeddable
    turn); every pair of current clusters sharing a parent is a pair that
    exists only because of the extra split — the pair k must justify.
    """
    parent: dict[int, int] = {}
    for c in set(labels):
        w: dict[int, float] = {}
        for i, l in enumerate(labels):
            if l == c:
                w[prev_labels[i]] = w.get(prev_labels[i], 0.0) + weights[i]
        parent[c] = max(w, key=lambda p: w[p])
    by_parent: dict[int, list[int]] = {}
    for c in sorted(parent):
        by_parent.setdefault(parent[c], []).append(c)
    return [
        (a, b)
        for siblings in by_parent.values()
        for i, a in enumerate(siblings) for b in siblings[i + 1:]
    ]


def _validate_k(
    pcm: np.ndarray, sr: int, turns: list[dict],
    order: list[int], labels: list[int], centroids: dict[int, np.ndarray],
    max_pooled_cosine: float, strict_pairs: list[tuple[int, int]],
) -> dict:
    """Judge one refined k-way split; returns a ``k_evaluated`` entry.

    Always contains ``k``, ``ok``, ``max_pairwise_cosine`` and
    ``min_cluster_seconds``; ``reason`` says what failed. Rules:

    * every PAIR of pooled centroids must be clearly different voices
      (cosine ≤ ``max_pooled_cosine``);
    * every pair in ``strict_pairs`` (the pair(s) the marginal split created
      — see :func:`_marginal_pairs`) must be VERY clearly different voices
      (cosine ≤ :data:`STRONG_SEPARATION_COSINE`), else the split carved one
      voice's registers apart rather than finding a new voice;
    * every marginal split must also be ANCHORED: at least one of its halves
      must have all its cosines to non-sibling clusters ≤
      :data:`NEW_VOICE_ANCHOR_COSINE` — a genuine new voice is wildly unlike
      an existing one, a phantom split is moderately far from everything;
    * every cluster must carry ≥ :data:`MIN_CLUSTER_SECONDS` of speech, OR
      ≥ :data:`MIN_CLUSTER_SECONDS_STRONG` when it is VERY clearly distinct
      from every other centroid (all its pairwise cosines ≤
      :data:`STRONG_SEPARATION_COSINE`).
    """
    cids = sorted(centroids)
    pair_cos = {
        (a, b): float(np.dot(centroids[a], centroids[b]))
        for i, a in enumerate(cids) for b in cids[i + 1:]
    }
    seconds = {
        c: sum(
            _slice(pcm, sr, turns[order[i]]).size
            for i in range(len(order)) if labels[i] == c
        ) / sr
        for c in cids
    }
    entry = {
        "k": len(cids),
        "ok": True,
        "max_pairwise_cosine": round(max(pair_cos.values()), 3),
        "min_cluster_seconds": round(min(seconds.values()), 2),
    }
    worst = max(pair_cos.values())
    if worst > max_pooled_cosine:
        entry["ok"] = False
        entry["reason"] = (
            f"centroids not clearly distinct (worst pair cosine "
            f"{worst:.3f} > {max_pooled_cosine:.2f})"
        )
        return entry
    if strict_pairs:
        worst_split = max(
            pair_cos[tuple(sorted(p))] for p in strict_pairs
        )
        entry["marginal_pair_cosine"] = round(worst_split, 3)
        if worst_split > STRONG_SEPARATION_COSINE:
            entry["ok"] = False
            entry["reason"] = (
                f"marginal split pair too similar (cosine {worst_split:.3f} "
                f"> {STRONG_SEPARATION_COSINE:.2f} — one voice in two "
                "registers, not a new voice)"
            )
            return entry

        def _outside(c: int, sibling: int) -> float:
            """Worst (highest) cosine from cluster ``c`` to any non-sibling."""
            others = [
                pair_cos[tuple(sorted((c, o)))]
                for o in cids if o not in (c, sibling)
            ]
            return max(others) if others else -1.0

        for a, b in strict_pairs:
            anchor = min(_outside(a, b), _outside(b, a))
            entry["split_anchor_cosine"] = round(anchor, 3)
            if anchor > NEW_VOICE_ANCHOR_COSINE:
                entry["ok"] = False
                entry["reason"] = (
                    "marginal split not anchored by a clearly new voice "
                    f"(both halves ≥ cosine {anchor:.3f} from every other "
                    f"cluster; needs ≤ {NEW_VOICE_ANCHOR_COSINE:.2f})"
                )
                return entry
    for c in cids:
        own_pairs = [v for (a, b), v in pair_cos.items() if c in (a, b)]
        floor = (
            MIN_CLUSTER_SECONDS_STRONG
            if max(own_pairs) <= STRONG_SEPARATION_COSINE
            else MIN_CLUSTER_SECONDS
        )
        if seconds[c] < floor:
            entry["ok"] = False
            entry["reason"] = (
                f"cluster has only {seconds[c]:.1f}s pooled speech "
                f"(needs {floor:.1f}s at separation "
                f"{max(own_pairs):.3f})"
            )
            return entry
    return entry


def diarize_turns(
    pcm: np.ndarray,
    sr: int,
    turns: list[dict],
    *,
    embed_fn=None,
    min_seconds: float = MIN_SECONDS,
    max_pooled_cosine: float = MAX_POOLED_COSINE,
) -> dict | None:
    """Attempt a validated k-voice relabeling of ``turns`` from the audio.

    Every k = 2 .. :data:`MAX_SPEAKERS_LOCAL` (capped by the embeddable turn
    count) is merged, refined and validated; the LARGEST fully-validating k
    wins. Returns ``None`` whenever local diarization has nothing TRUSTWORTHY
    to say: voice model unavailable, fewer than two embeddable utterances, or
    NO k validates (a monologue, or clusters with too little pooled speech /
    insufficiently distinct centroids). Otherwise returns::

        {
          "turns": [...],              # speaker relabeled; a turn that was
                                       # split arrives as TWO turns with the
                                       # text divided at a word boundary
          "num_speakers": int,         # the chosen k (2..MAX_SPEAKERS_LOCAL)
          "source": "local-ecapa",
          "model": "<hf-source>@<pinned-revision>",
          "segments_total": int,       # after any word-level splitting
          "segments_embedded": int,
          "split_utterances": int,     # utterances split at a voice change
          "pooled_cosine": float,      # WORST (highest) pairwise centroid
                                       # cosine of the chosen k (low=distinct)
          "k_evaluated": [...],        # every k tried: {k, ok,
                                       # max_pairwise_cosine,
                                       # min_cluster_seconds,
                                       # marginal_pair_cosine? (k>2),
                                       # reason?}
          "agreement_with_input": float,   # Rand agreement vs input labels
        }

    Turns may carry an optional ``words`` list ([{word, start_time,
    end_time}, ...]); it enables the word-level split pre-pass and is never
    propagated to the output turns.
    """
    embed = embed_fn or _default_embed
    try:
        embedded = _embed_turns(pcm, sr, turns, embed, min_seconds)
        if embedded is None:
            return None
        order, embs = embedded

        # Word-level pre-pass: a transcriber can weld a speaker handoff into
        # ONE utterance. The 2-way pass's pooled centroids are the anchors
        # every scan window is scored against (the change-point margin is a
        # two-centroid affinity contrast by construction); a handoff between
        # ANY two of the loudest voices shows up against them. Split long
        # word-timed utterances at sustained voice changes, then re-embed the
        # finer segments so every piece is attributed like any other turn.
        pass1 = _refine_k(pcm, sr, turns, embed, order, embs, 2)
        if pass1 is None:
            return None
        turns, split_stats = split_long_utterances(
            pcm, sr, turns, embed, (pass1[1][0], pass1[1][1]),
        )
        if split_stats["split"]:
            embedded = _embed_turns(pcm, sr, turns, embed, min_seconds)
            if embedded is None:
                return None
            order, embs = embedded
            pass1 = None  # stale — computed over the unsplit turns

        # k-selection: refine + validate every candidate count, keep the
        # LARGEST k that fully validates. Each k > 2 must additionally
        # justify the pair(s) its extra split created against the refined
        # k-1 partition (the marginal-split rule — see
        # STRONG_SEPARATION_COSINE). Every verdict is recorded so the
        # caller's logs show why a count was chosen.
        weights = [
            _slice(pcm, sr, turns[i]).size / sr for i in order
        ]
        k_evaluated: list[dict] = []
        chosen: tuple[list[int], dict[int, np.ndarray], dict] | None = None
        prev_labels: list[int] | None = None
        for k in range(2, min(MAX_SPEAKERS_LOCAL, len(order)) + 1):
            refined = (
                pass1 if k == 2 and pass1 is not None
                else _refine_k(pcm, sr, turns, embed, order, embs, k)
            )
            if refined is None:
                k_evaluated.append({
                    "k": k, "ok": False,
                    "reason": "refinement collapsed clusters "
                              f"(audio does not hold {k} distinct voices)",
                })
                prev_labels = None
                continue
            labels_k, centroids_k = refined
            if k == 2:
                strict_pairs: list[tuple[int, int]] = []
            elif prev_labels is None:
                k_evaluated.append({
                    "k": k, "ok": False,
                    "reason": f"no refined k={k - 1} partition to justify "
                              "the marginal split against",
                })
                continue
            else:
                strict_pairs = _marginal_pairs(labels_k, prev_labels, weights)
            entry = _validate_k(
                pcm, sr, turns, order, labels_k, centroids_k,
                max_pooled_cosine, strict_pairs,
            )
            k_evaluated.append(entry)
            prev_labels = labels_k
            if entry["ok"]:
                chosen = (labels_k, centroids_k, entry)
        if chosen is None:
            logger.info(
                "local diarization heard one voice (no k validated: %s)",
                "; ".join(
                    f"k={e['k']}: {e.get('reason', 'ok')}" for e in k_evaluated
                ),
            )
            return None
        labels, centroids, chosen_entry = chosen
        pooled_cosine = chosen_entry["max_pairwise_cosine"]
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
        "k_evaluated": k_evaluated,
        "agreement_with_input": partition_agreement(
            [t.get("speaker") for t in turns],
            [t["speaker"] for t in new_turns],
        ),
    }
