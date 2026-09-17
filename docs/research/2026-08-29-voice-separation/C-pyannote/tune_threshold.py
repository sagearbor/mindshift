"""Step 4: tune ONE knob — pyannote's clustering threshold — on the three
TTS scene fixtures only, then report on every other fixture untouched.
Done for both embedders (wespeaker = the stock pipeline; ecapa = the hybrid),
each tuned independently, because a distance threshold is embedder-specific.

    tmp/venv-pyannote/bin/python docs/research/2026-08-29-voice-separation/C-pyannote/tune_threshold.py
"""
from __future__ import annotations

import json

import numpy as np

from common import HERE, load_pipeline, score, scored, write_pred
from hybrid_lib import PY_THRESHOLD, Cached, pyannote_cluster

TUNE_ON = ["scene_couple", "scene_family3", "scene_meeting4"]
GRID = [round(x, 3) for x in np.arange(0.30, 1.101, 0.025)]


def main():
    pipeline = load_pipeline()
    names = score.all_fixtures()
    cached = {n: Cached(n) for n in names}
    out = {"grid": GRID, "tuned_on": TUNE_ON, "sweep": {}, "chosen": {}, "eval": {}}
    for emb in ("wespeaker", "ecapa"):
        sweep = {}
        for thr in GRID:
            accs = []
            for n in TUNE_ON:
                c = cached[n]
                pred = c.finish(pipeline, pyannote_cluster(c, c.emb(emb), threshold=thr))
                accs.append(scored(n, pred)["frame_accuracy"])
            sweep[str(thr)] = {"mean_acc": round(float(np.mean(accs)), 4), "per_fixture": dict(zip(TUNE_ON, accs))}
        best_acc = max(v["mean_acc"] for v in sweep.values())
        cands = [float(t) for t, v in sweep.items() if v["mean_acc"] == best_acc]
        thr = min(cands, key=lambda t: abs(t - PY_THRESHOLD))  # tie -> nearest to stock
        out["sweep"][emb] = sweep
        out["chosen"][emb] = {"threshold": thr, "tune_mean_acc": best_acc, "tied_candidates": cands,
                              "stock_threshold": PY_THRESHOLD, "stock_tune_mean_acc": sweep[str(round(min(GRID, key=lambda g: abs(g - PY_THRESHOLD)), 3))]["mean_acc"]}
        print(f"[{emb}] chosen threshold={thr} (tune mean acc {best_acc:.3f}; ties {cands})", flush=True)
        ev = {}
        for n in names:
            c = cached[n]
            pred = c.finish(pipeline, pyannote_cluster(c, c.emb(emb), threshold=thr))
            vname = f"tuned_thr__{emb}"
            write_pred(n, vname, pred)
            ev[n] = scored(n, pred, threshold=thr, held_out=n not in TUNE_ON)
            r = ev[n]
            print(f"  {n:14s} {'HELD-OUT' if r['held_out'] else 'tuned-on':8s} acc={r['frame_accuracy']:.3f} k={r['k_pred']}/{c.k_true} own={r['owner_purity']}", flush=True)
        out["eval"][emb] = ev
    (HERE / "results_tuned.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
