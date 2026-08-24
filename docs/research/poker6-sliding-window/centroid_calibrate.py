"""Offline calibration of the centroid-anchored margin approach against the
round-2 cached window embeddings (NO new ECAPA calls -- pure numpy over
already-embedded windows). Explores the online growing-segment-vs-lookahead
centroid distance curve and compares it to known ground truth change points
for BOTH real fixtures, to pick min_segment_windows / lookahead_windows /
threshold / min_run before touching diarize_sliding_window.py for real.
"""
import json
import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).parent

GROUND_TRUTH = {
    # speaker-changing gap midpoints (same as round 2's inspect_curve.py)
    "family": [6.08, 10.25, 15.28, 19.75, 25.755],
    # poker6 approx grid (only accurate to +/-1-2s per fixture note)
    "poker6": [5.0, 10.0, 15.0, 20.0, 25.0],
}


def load(fixture):
    d = json.loads((EXP_DIR / f"embeds_{fixture}_w1.5_h0.5.json").read_text())
    starts = d["starts"]
    embs = [np.array(d["embeddings"][str(i)], dtype=np.float32) for i in range(len(starts))]
    return starts, embs, d["window"], d["hop"]


def l2n(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def centroid_margin_curve(starts, embs, *, min_segment_windows, lookahead_windows):
    """Online growing-segment-vs-lookahead centroid distance curve.

    Returns (times, dists, seg_start_indices) -- dists[i] is 1-cos between
    the POOLED centroid of the current running segment (embs[seg_start:i])
    and the POOLED centroid of the next `lookahead_windows` embeddings
    (embs[i:i+lookahead_windows]). Only positions where segment has
    accumulated >= min_segment_windows are scored (None otherwise).
    """
    n = len(embs)
    times = []
    dists = []
    seg_start = 0
    i = min_segment_windows
    while i + lookahead_windows <= n:
        left = l2n(np.mean(embs[seg_start:i], axis=0))
        right = l2n(np.mean(embs[i:i + lookahead_windows], axis=0))
        d = 1.0 - float(np.dot(left, right))
        times.append(starts[i])
        dists.append(d)
        i += 1
    return times, dists


def main():
    for fixture in ["family", "poker6"]:
        starts, embs, window, hop = load(fixture)
        truths = GROUND_TRUTH[fixture]
        for min_seg_w, look_w in [(4, 2), (6, 3), (8, 3), (4, 3), (6, 2)]:
            times, dists = centroid_margin_curve(
                starts, embs, min_segment_windows=min_seg_w, lookahead_windows=look_w,
            )
            print(f"\n=== {fixture} min_seg_w={min_seg_w} look_w={look_w} "
                  f"(seg={min_seg_w*hop:.1f}s, look={look_w*hop:.1f}s) ===")
            for t, d in zip(times, dists):
                gap = min(abs(t - g) for g in truths)
                marker = " <-- TRUE" if gap < 1.0 else ""
                print(f"  t={t:6.2f} dist={d:7.4f}{marker}")
            near = [d for t, d in zip(times, dists) if min(abs(t - g) for g in truths) < 1.0]
            far = [d for t, d in zip(times, dists) if min(abs(t - g) for g in truths) >= 1.0]
            if near and far:
                print(f"  near-truth: min={min(near):.3f} max={max(near):.3f} mean={np.mean(near):.3f}")
                print(f"  far-from-truth: min={min(far):.3f} max={max(far):.3f} mean={np.mean(far):.3f}")


if __name__ == "__main__":
    main()
