"""Agreement statistics with bootstrap CIs (numpy only, deterministic seeds)."""
from __future__ import annotations

import numpy as np

SEED = 20261105


def cohen_kappa(a, b, labels=None) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if labels is None:
        labels = sorted(set(a.tolist()) | set(b.tolist()))
    if len(a) == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == l)) * float(np.mean(b == l)) for l in labels)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(matrix: np.ndarray) -> float:
    """matrix: items x categories, counts of raters per category (same n raters per item)."""
    m = np.asarray(matrix, dtype=float)
    n = m.sum(1)
    if np.any(n != n[0]) or n[0] < 2:
        raise ValueError("Fleiss needs a constant >=2 raters per item")
    n = n[0]
    p_j = m.sum(0) / m.sum()
    P_i = (np.sum(m * m, 1) - n) / (n * (n - 1))
    Pbar, Pe = P_i.mean(), np.sum(p_j ** 2)
    if Pe >= 1.0:
        return float("nan")
    return float((Pbar - Pe) / (1 - Pe))


def boot_ci(fn, n_items: int, B: int = 2000, alpha: float = 0.05, seed: int = SEED):
    """Percentile bootstrap over items. fn(idx) -> statistic."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n_items, n_items)
        v = fn(idx)
        if v is not None and not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def kappa_ci(a, b, labels=None, B=2000):
    a = np.asarray(a)
    b = np.asarray(b)
    k = cohen_kappa(a, b, labels)
    lo, hi = boot_ci(lambda i: cohen_kappa(a[i], b[i], labels), len(a), B)
    return k, lo, hi


def fleiss_ci(matrix, B=2000):
    m = np.asarray(matrix)
    k = fleiss_kappa(m)
    lo, hi = boot_ci(lambda i: fleiss_kappa(m[i]), len(m), B)
    return k, lo, hi


def binary_class_stats(pred, ref, cls, B=2000):
    """One-vs-rest precision / recall / kappa for class `cls`, with bootstrap CIs."""
    pred = np.asarray(pred)
    ref = np.asarray(ref)

    def prec(i):
        p, r = pred[i] == cls, ref[i] == cls
        return float(np.sum(p & r) / np.sum(p)) if np.sum(p) else np.nan

    def rec(i):
        p, r = pred[i] == cls, ref[i] == cls
        return float(np.sum(p & r) / np.sum(r)) if np.sum(r) else np.nan

    def kap(i):
        return cohen_kappa(pred[i] == cls, ref[i] == cls, [True, False])

    allidx = np.arange(len(pred))
    out = {}
    for name, f in (("precision", prec), ("recall", rec), ("kappa", kap)):
        v = f(allidx)
        lo, hi = boot_ci(f, len(pred), B)
        out[name] = (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v), lo, hi)
    out["n_ref"] = int(np.sum(ref == cls))
    out["n_pred"] = int(np.sum(pred == cls))
    return out


def tvd(p, q) -> float:
    """Total variation distance between two distributions."""
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())
