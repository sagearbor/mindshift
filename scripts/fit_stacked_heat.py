"""FIT STACKED HEAT — the coefficients server/tone_id.stacked_heat_score hard-codes.

Heat-judge plan step 4 (docs/decisions/2026-09-20-heat-judge-plan.md): stack
the SpeechBrain IEMOCAP angry-vs-happy vote (server/tone_id.angry_vote) on
top of odyssey_dim's arousal/dominance/valence, as ONE fixed linear formula
cheap enough to run every turn with no model call (the models already ran
once, when the feature bank was built).

Reads the feature bank (tmp/feature-bank/{index,tone,sbiemocap}.parquet —
gitignored, rebuild with scripts/feature_bank.py) and, using EXACTLY
scripts/feature_bench.py's protocol (StandardScaler + LogisticRegression,
train on ALL of CREMA-D's 6 emotions with y = is_angry, evaluate cross-corpus
on RAVDESS), fits two logistic regressions:

  STACKED    arousal, valence, dominance, angry_p   (angry_p = P(angry) - P(happy)
                                                       from the IEMOCAP softmax)
  DIMS-ONLY  arousal, valence, dominance             (angry_p unavailable fallback)

then folds the StandardScaler into the coefficients so the shipped formula
is a single dot product in RAW feature units (no scaler object to carry
around at inference time), and prints both the scaled-space numbers (for
provenance) and the raw ones to paste into server/tone_id.py.

"angry-vs-happy" in the printed metric names is the EVALUATED subset (only
angry + happy clips go into that AUC — the shipped loudness ladder's actual
failure mode), not a filtered training set: --train-angry-happy-only
reproduces the (worse) alternative of training on that subset directly, kept
as the ablation that justifies training on all 6 emotions instead.

    python scripts/fit_stacked_heat.py
    python scripts/fit_stacked_heat.py --C 3.0 --write-fixture
    python scripts/fit_stacked_heat.py --train-angry-happy-only   # ablation
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
# The feature bank (tmp/feature-bank/*.parquet, gitignored) lives in the
# MAIN repo's tmp/ — a worktree checkout does not carry it. Override with
# MINDSHIFT_FEATURE_BANK; otherwise prefer this repo's own tmp/ (correct when
# run from the main checkout) and fall back to the known main-repo path (this
# script is expected to run from a `.claude/worktrees/...` checkout).
_MAIN_REPO_BANK = Path("/Users/sagearbor/projects/githubs/mindshift/tmp/feature-bank")
_local_bank = REPO / "tmp/feature-bank"
BANK = Path(os.environ["MINDSHIFT_FEATURE_BANK"]) if os.environ.get("MINDSHIFT_FEATURE_BANK") else (
    _local_bank if _local_bank.exists() else _MAIN_REPO_BANK
)
FIXTURE_OUT = REPO / "server/tests/fixtures/stacked_heat_fixture.json"

STACKED_COLS = ["tone_arousal", "tone_valence", "tone_dominance", "angry_p"]
DIMS_ONLY_COLS = ["tone_arousal", "tone_valence", "tone_dominance"]


def load_bank() -> pd.DataFrame:
    idx = pd.read_parquet(BANK / "index.parquet")
    tone = pd.read_parquet(BANK / "tone.parquet")
    sb = pd.read_parquet(BANK / "sbiemocap.parquet")
    df = idx.join(tone, how="inner").join(sb, how="inner")
    df["angry_p"] = df["sbiemocap_ang"] - df["sbiemocap_hap"]
    return df


def recall_at_fa(y: np.ndarray, s: np.ndarray, fa: float = 0.05) -> float:
    thr = np.quantile(s[y == 0], 1 - fa)
    return float((s[y == 1] >= thr).mean())


def fit_one(df: pd.DataFrame, cols: list[str], C: float, angry_happy_only: bool, label: str) -> dict:
    y = df.is_angry.to_numpy().astype(int)
    happy = (df.emotion == "happy").to_numpy()
    tr = (df.corpus == "cremad").to_numpy()
    te = (df.corpus == "ravdess").to_numpy()
    if angry_happy_only:
        tr = tr & (happy | (y == 1))
    X = df[cols].to_numpy(dtype=float)

    scaler = StandardScaler().fit(X[tr])
    Xz = scaler.transform(X)
    clf = LogisticRegression(max_iter=5000, C=C).fit(Xz[tr], y[tr])

    s = clf.predict_proba(Xz[te])[:, 1]
    hm = happy[te] | (y[te] == 1)
    auc_happy = float(roc_auc_score(y[te][hm], s[hm]))
    rec = recall_at_fa(y[te], s)

    # Fold the scaler into raw-unit coefficients: z = (x - mu) / sd, so
    #   intercept + sum(coef_i * z_i)
    # = intercept + sum(coef_i * (x_i - mu_i) / sd_i)
    # = [intercept - sum(coef_i * mu_i / sd_i)] + sum((coef_i / sd_i) * x_i)
    coef, intercept = clf.coef_[0], float(clf.intercept_[0])
    mu, sd = scaler.mean_, scaler.scale_
    raw_coef = coef / sd
    raw_intercept = float(intercept - np.sum(coef * mu / sd))

    print(f"\n{label}  cols={cols}  C={C}  train_rows={int(tr.sum())}  test_rows={int(te.sum())}")
    print(f"  cremad -> ravdess   auc(angry-vs-happy)={auc_happy:.4f}   recall@5%FA={rec:.4f}")
    print(f"  scaler mean {mu.round(4).tolist()}")
    print(f"  scaler std  {sd.round(4).tolist()}")
    print(f"  scaled coef {coef.round(4).tolist()}   scaled intercept {intercept:.4f}")
    print(f"  RAW coef    {raw_coef.round(6).tolist()}")
    print(f"  RAW intercept {raw_intercept:.6f}")
    return {
        "cols": cols, "auc_happy": auc_happy, "recall_at_5fa": rec,
        "raw_coef": dict(zip([c.replace("tone_", "") for c in cols], raw_coef.round(6).tolist())),
        "raw_intercept": round(raw_intercept, 6),
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def score_with_raw(df: pd.DataFrame, cols: list[str], fit: dict) -> np.ndarray:
    X = df[cols].to_numpy(dtype=float)
    coef = np.array([fit["raw_coef"][c.replace("tone_", "")] for c in cols])
    return sigmoid(fit["raw_intercept"] + X @ coef)


def write_fixture(df: pd.DataFrame, stacked_fit: dict, dims_fit: dict, n: int, seed: int) -> None:
    """20 rows (default), sampled across both corpora and both emotions of
    interest, with the score the HARD-CODED formula in tone_id.py must
    reproduce (computed here from the same raw_coef/raw_intercept, rounded to
    the precision actually pasted into the module) — the unit-test fixture
    for test_stacked_heat.py's "the constants really are these numbers" check.
    """
    # DataFrameGroupBy.sample (not .apply(lambda g: g.sample(...))) — apply
    # silently drops the group-by columns from the result when every row in
    # a group shares them, which corpus/emotion always do here. Oversample
    # per group then trim to exactly `n` (6 groups: {cremad,ravdess} x
    # {angry,happy,neutral} don't divide evenly into n).
    per_group = max(1, -(-n // 6))  # ceil(n / 6)
    sample = df[df.emotion.isin(["angry", "happy", "neutral"])].groupby(
        ["corpus", "emotion"],
    ).sample(n=per_group, random_state=seed).sample(n=min(n, per_group * 6), random_state=seed)
    rows = []
    for _, r in sample.iterrows():
        dims = {"arousal": float(r.tone_arousal), "valence": float(r.tone_valence), "dominance": float(r.tone_dominance)}
        angry_p = float(r.angry_p)
        stacked_score = float(sigmoid(np.array(
            stacked_fit["raw_intercept"]
            + dims["arousal"] * stacked_fit["raw_coef"]["arousal"]
            + dims["valence"] * stacked_fit["raw_coef"]["valence"]
            + dims["dominance"] * stacked_fit["raw_coef"]["dominance"]
            + angry_p * stacked_fit["raw_coef"]["angry_p"],
        )))
        dims_only_score = float(sigmoid(np.array(
            dims_fit["raw_intercept"]
            + dims["arousal"] * dims_fit["raw_coef"]["arousal"]
            + dims["valence"] * dims_fit["raw_coef"]["valence"]
            + dims["dominance"] * dims_fit["raw_coef"]["dominance"],
        )))
        rows.append({
            "corpus": r.corpus, "emotion": r.emotion, "dims": dims, "angry_p": angry_p,
            "stacked_score": round(stacked_score, 6), "dims_only_score": round(dims_only_score, 6),
        })
    FIXTURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_OUT.write_text(json.dumps(rows, indent=1))
    print(f"\n-> {FIXTURE_OUT} ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=3.0)
    ap.add_argument("--train-angry-happy-only", action="store_true",
                     help="ablation: train on CREMA-D angry+happy rows only (worse recall@5fa)")
    ap.add_argument("--write-fixture", action="store_true", help="(re)write the 20-row unit-test fixture")
    ap.add_argument("--fixture-rows", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_bank()
    print(f"bank: {len(df)} clips (index ⋈ tone ⋈ sbiemocap), "
          f"{int((df.corpus=='cremad').sum())} cremad / {int((df.corpus=='ravdess').sum())} ravdess")

    stacked = fit_one(df, STACKED_COLS, args.C, args.train_angry_happy_only, "STACKED (dims + angry_p)")
    dims_only = fit_one(df, DIMS_ONLY_COLS, args.C, args.train_angry_happy_only, "DIMS-ONLY fallback")

    if args.write_fixture:
        write_fixture(df, stacked, dims_only, args.fixture_rows, args.seed)

    print("\n--- paste into server/tone_id.py ---")
    print(f"_STACKED_INTERCEPT = {stacked['raw_intercept']}")
    print(f"_STACKED_COEF = {json.dumps(stacked['raw_coef'])}")
    print(f"_DIMS_ONLY_INTERCEPT = {dims_only['raw_intercept']}")
    print(f"_DIMS_ONLY_COEF = {json.dumps(dims_only['raw_coef'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
