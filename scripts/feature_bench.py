"""FEATURE BENCH — which signals, and which combinations, actually tell anger from everything else?

Reads the feature bank and evaluates every requested combination with the two
guards that make a number trustworthy:

  speaker-grouped CV   no speaker appears in both train and test (5 folds)
  cross-corpus         train on one corpus, test on the other — acted-to-acted
                       but different actors, rooms, scripts and label protocols;
                       a feature that survives this is far more likely to survive
                       a living room

Metrics, per combination:
  auc_all      angry vs every other emotion
  auc_happy    angry vs HAPPY only — the failure the shipped loudness ladder has
  recall@5fa   share of angry clips caught when only 5% of non-angry may fire —
               the operating point a coaching cue actually lives at
  n            clips with every feature in the combination present

Model: logistic regression on standardised features (a linear probe — the
point is to rank SIGNALS, not to tune a classifier). --gbm adds a gradient-
boosted tree for the combinations where interactions might matter.

    python scripts/feature_bench.py                       # all groups + key combos
    python scripts/feature_bench.py --combo egemaps+tone  # one
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
BANK = REPO / "tmp/feature-bank"
OUT = REPO / "tmp/feature-bank/bench.json"


def load() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    idx = pd.read_parquet(BANK / "index.parquet")
    groups: dict[str, list[str]] = {}
    df = idx
    for f in sorted(BANK.glob("*.parquet")):
        if f.stem == "index":
            continue
        g = pd.read_parquet(f)
        groups[f.stem] = list(g.columns)
        df = df.join(g, how="left")
    return df, groups


def recall_at_fa(y: np.ndarray, s: np.ndarray, fa: float = 0.05) -> float:
    thr = np.quantile(s[y == 0], 1 - fa)
    return float((s[y == 1] >= thr).mean())


def model(gbm: bool):
    if gbm:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))


def evaluate(df: pd.DataFrame, cols: list[str], gbm: bool) -> dict:
    d = df.dropna(subset=cols)
    X = d[cols].to_numpy(dtype=float)
    y = d.is_angry.to_numpy().astype(int)
    grp = d.speaker.to_numpy()
    happy = (d.emotion == "happy").to_numpy()
    res = {"n": int(len(d))}

    # ---- within-corpus, speaker-grouped (CREMA-D is the big one) ----
    for corpus in ("cremad", "ravdess"):
        m = (d.corpus == corpus).to_numpy()
        if m.sum() < 200 or y[m].sum() < 30:
            continue
        oof = np.zeros(m.sum())
        Xc, yc, gc = X[m], y[m], grp[m]
        for tr, te in GroupKFold(n_splits=5).split(Xc, yc, gc):
            oof[te] = model(gbm).fit(Xc[tr], yc[tr]).predict_proba(Xc[te])[:, 1]
        hm = happy[m] | (yc == 1)
        res[f"{corpus}_auc_all"] = round(roc_auc_score(yc, oof), 3)
        res[f"{corpus}_auc_happy"] = round(roc_auc_score(yc[hm], oof[hm]), 3)
        res[f"{corpus}_recall@5fa"] = round(recall_at_fa(yc, oof), 3)

    # ---- cross-corpus: the generalisation test ----
    for tr_c, te_c in (("cremad", "ravdess"), ("ravdess", "cremad")):
        tr, te = (d.corpus == tr_c).to_numpy(), (d.corpus == te_c).to_numpy()
        if tr.sum() < 200 or te.sum() < 200 or y[tr].sum() < 30 or y[te].sum() < 30:
            continue
        s = model(gbm).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        hm = happy[te] | (y[te] == 1)
        res[f"{tr_c}->{te_c}_auc_all"] = round(roc_auc_score(y[te], s), 3)
        res[f"{tr_c}->{te_c}_auc_happy"] = round(roc_auc_score(y[te][hm], s[hm]), 3)
        res[f"{tr_c}->{te_c}_recall@5fa"] = round(recall_at_fa(y[te], s), 3)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", action="append", help="e.g. egemaps+tone; repeatable")
    ap.add_argument("--gbm", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=2)
    args = ap.parse_args()
    df, groups = load()
    print(f"bank: {len(df)} clips · groups {', '.join(f'{k}({len(v)})' for k, v in groups.items())}")

    combos: list[list[str]] = []
    if args.combo:
        combos = [c.split("+") for c in args.combo]
    else:
        # the shipped signal on its own, every group alone, then pairs
        combos.append(["__loudness__"])
        combos += [[g] for g in groups]
        combos += [list(c) for k in range(2, args.max_pairs + 1) for c in combinations(groups, k)]

    results = {}
    for combo in combos:
        if combo == ["__loudness__"]:
            cols = ["prosody_db_over_own_neutral"]
            name = "loudness-over-own-baseline (shipped)"
        else:
            cols = [c for g in combo for c in groups.get(g, [])]
            name = "+".join(combo)
        if not cols:
            continue
        r = evaluate(df, cols, args.gbm)
        results[name] = r
        print(f"\n{name}  (n={r['n']})")
        for k, v in r.items():
            if k != "n":
                print(f"  {k:26} {v}")
    OUT.write_text(json.dumps(results, indent=1))
    print(f"\n-> {OUT}")

    # a compact ranking on the numbers that matter
    print(f"\n{'combination':44} {'crema all':>9} {'crema happy':>11} {'xcorp all':>9} {'xcorp happy':>11} {'rec@5fa':>8}")
    for name, r in sorted(results.items(), key=lambda kv: -(kv[1].get("cremad->ravdess_auc_happy", 0) or 0)):
        print(f"{name[:44]:44} {r.get('cremad_auc_all','-')!s:>9} {r.get('cremad_auc_happy','-')!s:>11} "
              f"{r.get('cremad->ravdess_auc_all','-')!s:>9} {r.get('cremad->ravdess_auc_happy','-')!s:>11} "
              f"{r.get('cremad->ravdess_recall@5fa','-')!s:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
