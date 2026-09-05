"""Diagnostic: how much of each variant's error is pyannote's VAD leaving GT
speech UNLABELLED (tight segment edges, missed quiet speech) versus WRONG
labels?  Re-score selected preds after nearest-neighbour gap filling — every
unlabelled 10 ms frame takes the label of the temporally-closest predicted
segment.  Our production path labels whole transcript utterances, so it
never leaves gaps; this is the fairer ceiling for "adopt pyannote's
segmenter in front of our clustering".  Stdlib only.

    python docs/research/2026-08-29-voice-separation/C-pyannote/gapfill_eval.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("score", HERE.parent / "score.py")
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)

VARIANTS = ["p31_default", "p31_oracle_k", "seg_pyannote__oracle_labels",
            "seg_pyannote__ecapa__pyclust_oracle_k", "seg_pyannote__ecapa__ours_avglink_oracle_k",
            "seg_pyannote__wespeaker__ours_avglink_oracle_k", "tuned_thr__wespeaker", "tuned_thr__ecapa"]


def gap_fill(pred: list, end: float) -> list:
    segs = sorted((float(s), float(e), l) for s, e, l in pred)
    if not segs:
        return []
    out, t = [], 0.0
    for i, (s, e, l) in enumerate(segs):
        if s > t:  # gap before this segment: split it between neighbours
            prev = out[-1] if out else None
            mid = (prev[1] + s) / 2 if prev else 0.0
            if prev:
                out[-1] = (prev[0], mid, prev[2])
            out.append((mid, s, l))
        out.append((s, e, l))
        t = max(t, e)
    if out[-1][1] < end:
        out[-1] = (out[-1][0], end, out[-1][2])
    return [[s, e, l] for s, e, l in out]


def main():
    res = {}
    for fx in score.all_fixtures():
        gt = score.load_fixture(fx)["gt"]
        end = max(e for _, e, _ in gt)
        res[fx] = {}
        for v in VARIANTS:
            p = HERE / "preds" / f"pred_{fx}__{v}.json"
            if not p.exists():
                continue
            pred = json.loads(p.read_text())
            raw = score.score_fixture(fx, [tuple(x) for x in pred])
            filled = score.score_fixture(fx, [tuple(x) for x in gap_fill(pred, end)])
            res[fx][v] = {"raw": raw["frame_accuracy"], "unlabelled": raw["unlabelled_frac"],
                          "gapfilled": filled["frame_accuracy"], "k_pred": raw["k_pred"], "k_true": raw["k_true"],
                          "owner_purity_gapfilled": filled["owner_purity"]}
    (HERE / "results_gapfill.json").write_text(json.dumps(res, indent=1))
    fxs = list(res)
    print("| variant | " + " | ".join(fxs) + " |")
    print("|---|" + "---|" * len(fxs))
    for v in VARIANTS:
        row = [f"{res[f][v]['raw']:.3f} -> {res[f][v]['gapfilled']:.3f} (unl {res[f][v]['unlabelled']:.2f})" if v in res[f] else "-" for f in fxs]
        print(f"| {v} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
