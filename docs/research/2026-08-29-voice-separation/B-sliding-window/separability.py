"""Window-embedding separability per fixture (no model calls — reads cache/).

For each cached fixture + grid: mean pairwise cosine between windows of every
GT-speaker pair (diagonal = within-speaker), and the accuracy of assigning
each window to its nearest ORACLE GT-centroid (an upper bound on what any
window-level clustering can reach). Writes separability.json + prints tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import score  # noqa: E402

out: dict = {}
for f in sorted((HERE / "cache").glob("*_w1.5_h0.25.npz")):
    name = f.name.split("_w1.5")[0]
    fx = score.load_fixture(name)
    z = np.load(f)
    st, E, w = z["starts"], z["embs"], float(z["window"])
    c = st + w / 2

    def lab_of(x):
        g = next((l for s, e, l in fx["gt"] if s <= x < e), None)
        return None if g is None else (g if isinstance(g, str) else "/".join(sorted(g)))

    lab = np.array([lab_of(x) or "gap" for x in c])
    S = E @ E.T
    labs = [l for l in sorted(set(lab)) if l != "gap" and "/" not in l]
    mat = {}
    for a in labs:
        for b in labs:
            ia, ib = np.where(lab == a)[0], np.where(lab == b)[0]
            sub = S[np.ix_(ia, ib)]
            if a == b:
                sub = sub[np.triu_indices(len(ia), 1)]
            mat[f"{a}|{b}"] = round(float(sub.mean()), 3) if sub.size else None
    cents = {a: E[lab == a].mean(0) for a in labs}
    for a in cents:
        cents[a] /= np.linalg.norm(cents[a])
    idx = [i for i in range(len(st)) if lab[i] in cents]
    ok = sum(1 for i in idx if max(cents, key=lambda k: float(E[i] @ cents[k])) == lab[i])
    within = [mat[f"{a}|{a}"] for a in labs]
    cross = [mat[f"{a}|{b}"] for a in labs for b in labs if a < b]
    out[name] = {"n_windows": int(len(st)), "speakers": labs, "matrix": mat,
                 "within_mean": round(float(np.mean(within)), 3), "within_min": min(within),
                 "cross_mean": round(float(np.mean(cross)), 3), "cross_max": max(cross),
                 "oracle_centroid_window_acc": round(ok / len(idx), 3)}
    print(f"{name:14s} n={len(st):3d} within mean {np.mean(within):.3f} (min {min(within):.3f}) | "
          f"cross mean {np.mean(cross):.3f} (max {max(cross):.3f}) | oracle-centroid window acc {ok/len(idx):.3f}")
(HERE / "separability.json").write_text(json.dumps(out, indent=1))
