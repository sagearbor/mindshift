"""Single-feature ablation for Approach A.

For every fixture and every scalar feature: (a) Fisher ratio using the GT
speaker labels (between-speaker variance of window means / mean
within-speaker variance — a supervised upper bound on how separable the
speakers are along that ONE axis), and (b) unsupervised oracle-k accuracy
from clustering on that one feature alone (same pipeline as run_acoustic).
Writes ablation.json and prints a table.  Re-uses run_acoustic's cached
feature extraction (re-computed here; it is < 1 s per fixture).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_acoustic as R  # noqa: E402

SINGLE = ["f0_med", "f0_iqr", "centroid", "tilt", "rolloff", "rms", "f1", "f2", "f3",
          "mfcc1_m", "mfcc2_m", "mfcc3_m"]
GROUPS = {"pitch": ["f0_med", "f0_iqr"], "spectral": ["centroid", "tilt", "rolloff"],
          "formants": ["f1", "f2", "f3"], "mfcc": R.VARIANTS["mfcc"], "full": R.VARIANTS["full"]}


def fisher(rows, col):
    vals = {}
    for r in rows:
        if r["speaker_gt"] and np.isfinite(r[col]):
            vals.setdefault(r["speaker_gt"], []).append(r[col])
    if len(vals) < 2:
        return float("nan")
    means = np.array([np.mean(v) for v in vals.values()])
    within = np.mean([np.var(v) for v in vals.values() if len(v) > 1])
    return float(np.var(means) / (within + 1e-9))


def main():
    names = sys.argv[1:] or R.scorer.all_fixtures()
    out = {}
    for name in names:
        fx = R.scorer.load_fixture(name)
        y = R.load_audio(fx)
        tr = R.frame_tracks(y)
        speech = R.energy_vad(tr["rms"])
        rows, _ = R.window_features(tr, speech, fx["gt"])
        res = {"k_true": fx["k_true"], "single": {}, "groups": {}}
        for c in SINGLE:
            lab, _ = R.cluster_variant(rows, [c], fx["k_true"])
            sc = R.scorer.score_fixture(name, R.to_segments(rows, lab))
            res["single"][c] = {"fisher": round(fisher(rows, c), 3), "oracle_acc": sc["frame_accuracy"]}
        for g, cols in GROUPS.items():
            lab, _ = R.cluster_variant(rows, cols, fx["k_true"])
            sc = R.scorer.score_fixture(name, R.to_segments(rows, lab))
            res["groups"][g] = {"oracle_acc": sc["frame_accuracy"],
                                "fisher_mean": round(float(np.nanmean([fisher(rows, c) for c in cols])), 3)}
        out[name] = res
        print(f"\n== {name} (k_true={fx['k_true']})")
        for c, v in sorted(res["single"].items(), key=lambda kv: -kv[1]["oracle_acc"]):
            print(f"  {c:10s} fisher={v['fisher']:7.3f}  oracle_acc={v['oracle_acc']:.3f}")
        for g, v in res["groups"].items():
            print(f"  [{g:8s}] oracle_acc={v['oracle_acc']:.3f} fisher_mean={v['fisher_mean']:.3f}")
    (HERE / "ablation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
