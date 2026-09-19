"""Step 3: decompose segmentation vs clustering using the cached
intermediates (no neural nets re-run except loading the pipeline object for
its reconstruct/to_annotation helpers).

    tmp/venv-pyannote/bin/python docs/research/2026-08-29-voice-separation/C-pyannote/cluster_hybrid.py [fixture ...]
"""
from __future__ import annotations

import json
import sys

import numpy as np

from common import HERE, Timer, load_pipeline, score, scored, write_pred
from hybrid_lib import PY_THRESHOLD, Cached, cluster_gt_segments, ours_cluster, pyannote_cluster


def run_fixture(pipeline, name: str) -> dict:
    c = Cached(name)
    k = c.k_true
    out = {}

    def add(vname, pred, **extra):
        write_pred(name, vname, pred)
        out[vname] = scored(name, pred, **extra)
        r = out[vname]
        print(f"{name:14s} {vname:34s} acc={r['frame_accuracy']:.3f} k={r['k_pred']}/{k} own={r['owner_purity']}", flush=True)

    # sanity: replaying pyannote's own clustering on its own embeddings must reproduce p31_default
    t = Timer().__enter__()
    hard = pyannote_cluster(c, c.emb("wespeaker"))
    add("replay_p31_default", c.finish(pipeline, hard), clustering_wall_s=round(t.s, 3))
    # segmentation ceiling: pyannote segments + ORACLE unit labels
    add("seg_pyannote__oracle_labels", c.finish(pipeline, c.unit_gt_labels()))
    # pyannote segmentation + pyannote clustering, k given / bounded, for both embedders
    for emb in ("wespeaker", "ecapa"):
        add(f"seg_pyannote__{emb}__pyclust_thr", c.finish(pipeline, pyannote_cluster(c, c.emb(emb))))
        add(f"seg_pyannote__{emb}__pyclust_oracle_k", c.finish(pipeline, pyannote_cluster(c, c.emb(emb), num_clusters=k)))
        add(f"seg_pyannote__{emb}__pyclust_bounds_2_6", c.finish(pipeline, pyannote_cluster(c, c.emb(emb), min_clusters=2, max_clusters=6), max_speakers=6))
        add(f"seg_pyannote__{emb}__ours_avglink_oracle_k", c.finish(pipeline, ours_cluster(c, c.emb(emb), k)))
        # clustering-only arm: GT segmentation
        add(f"seg_GT__{emb}__pyclust_thr", cluster_gt_segments(c, c.gt_emb(emb)))
        add(f"seg_GT__{emb}__pyclust_oracle_k", cluster_gt_segments(c, c.gt_emb(emb), k=k))
        add(f"seg_GT__{emb}__ours_avglink_oracle_k", cluster_gt_segments(c, c.gt_emb(emb), k=k, ours=True))
    # a few diagnostics about the units themselves
    act = [u["active_sec"] for u in c.units]
    out["_units"] = {"n": len(c.units), "median_active_sec": round(float(np.median(act)), 2),
                     "ecapa_nan": int(np.isnan(c.ecapa[:, :, 0]).sum() - (np.sum(c.bin.data, axis=1) == 0).sum())}
    return out


def main(names):
    pipeline = load_pipeline()
    path = HERE / "results_hybrid.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for n in names:
        results[n] = run_fixture(pipeline, n)
        path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or score.all_fixtures())
