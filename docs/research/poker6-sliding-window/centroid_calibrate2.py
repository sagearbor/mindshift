"""Offline calibration v2: FULL online state-machine (reset on accept) for
the centroid-anchored margin change detector, against cached round-2
embeddings. No new ECAPA calls.
"""
import json
import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).parent

GROUND_TRUTH = {
    "family": [6.08, 10.25, 15.28, 19.75, 25.755],
    "poker6": [5.0, 10.0, 15.0, 20.0, 25.0],
}


def load(fixture):
    d = json.loads((EXP_DIR / f"embeds_{fixture}_w1.5_h0.5.json").read_text())
    starts = d["starts"]
    embs = [np.array(d["embeddings"][str(i)], dtype=np.float32) for i in range(len(starts))]
    return starts, embs, d["window"], d["hop"], d["duration"]


def l2n(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def detect_online(
    starts, embs, duration, *, min_segment_windows, lookahead_windows,
    threshold, min_run,
):
    """Stateful walk: grow a 'current segment' centroid; test each hop's
    lookahead-block centroid against it; accept a change at the START of a
    sustained (>= min_run consecutive hops) run of dist >= threshold; reset
    the segment to start there and continue. Returns list of boundary times.
    """
    n = len(embs)
    boundaries = []
    seg_start = 0
    i = min_segment_windows
    run_start = None
    run_len = 0
    trace = []
    while i + lookahead_windows <= n:
        left = l2n(np.mean(embs[seg_start:i], axis=0))
        right = l2n(np.mean(embs[i:i + lookahead_windows], axis=0))
        d = 1.0 - float(np.dot(left, right))
        trace.append((starts[i], d, seg_start))
        if d >= threshold:
            if run_len == 0:
                run_start = i
            run_len += 1
        else:
            run_len = 0
            run_start = None
        if run_len >= min_run:
            # Accept: boundary at the START of the sustained run.
            boundaries.append(starts[run_start])
            seg_start = run_start
            i = seg_start + min_segment_windows
            run_len = 0
            run_start = None
            continue
        i += 1
    return boundaries, trace


def score(boundaries, truths, tol):
    """How many truths were matched (within tol) and how many boundaries
    were spurious (not near any truth)."""
    matched_truths = set()
    spurious = 0
    for b in boundaries:
        hit = False
        for k, t in enumerate(truths):
            if abs(b - t) <= tol and k not in matched_truths:
                matched_truths.add(k)
                hit = True
                break
        if not hit:
            spurious += 1
    return len(matched_truths), len(truths), spurious


def main():
    configs = [
        (4, 2, 0.65, 2), (4, 2, 0.70, 2), (4, 2, 0.60, 3),
        (4, 3, 0.65, 2), (4, 3, 0.70, 2), (4, 3, 0.60, 2),
        (6, 2, 0.65, 2), (6, 2, 0.60, 2), (6, 3, 0.65, 2),
        (3, 2, 0.65, 2), (3, 2, 0.70, 3), (3, 3, 0.65, 2),
        (4, 2, 0.65, 3), (4, 2, 0.75, 2),
    ]
    for fixture, tol in [("family", 1.0), ("poker6", 2.0)]:
        starts, embs, window, hop, duration = load(fixture)
        truths = GROUND_TRUTH[fixture]
        print(f"\n########## {fixture} (duration={duration:.1f}s, {len(truths)} true changes) ##########")
        for min_seg_w, look_w, thr, min_run in configs:
            boundaries, trace = detect_online(
                starts, embs, duration,
                min_segment_windows=min_seg_w, lookahead_windows=look_w,
                threshold=thr, min_run=min_run,
            )
            hit, total, spurious = score(boundaries, truths, tol)
            print(f"  seg={min_seg_w}({min_seg_w*hop:.1f}s) look={look_w}({look_w*hop:.1f}s) "
                  f"thr={thr} run={min_run} -> {len(boundaries)} boundaries "
                  f"[{hit}/{total} true matched, {spurious} spurious] "
                  f"boundaries={[round(b,2) for b in boundaries]}")


if __name__ == "__main__":
    main()
