"""Scoring an annotation against human ground truth (annotator validation).

* :func:`der` — DER-style frame scoring (10 ms) with the optimal one-to-one
  speaker mapping (Hungarian on overlap seconds). Overlapped truth frames
  count once per truth speaker, like NIST DER without a collar.
* :func:`best_shift` — the constant time offset that minimises DER, so a
  model that is consistently early/late is scored on alignment too.
* :func:`boundary_error` — for each TRUE speaker change, the distance to the
  nearest hypothesis speaker change.
* :func:`wer` — word error rate on normalised tokens.
* :func:`spearman` / :func:`pearson`.
"""

from __future__ import annotations

import math
import re

import numpy as np

HOP = 0.01


def _mask(spans, n: int, shift: float = 0.0) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in spans:
        a = max(0, int(round((s + shift) / HOP)))
        b = min(n, int(round((e + shift) / HOP)))
        if b > a:
            m[a:b] = True
    return m


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except ImportError:  # tiny greedy fallback
        pairs, used_r, used_c = [], set(), set()
        for idx in np.argsort(cost, axis=None):
            i, j = divmod(int(idx), cost.shape[1])
            if i not in used_r and j not in used_c:
                pairs.append((i, j))
                used_r.add(i)
                used_c.add(j)
        return pairs


def der(truth: dict, hyp: dict, duration: float, shift: float = 0.0) -> dict:
    n = int(math.ceil(duration / HOP)) + 1
    tk, hk = list(truth), list(hyp)
    T = np.stack([_mask(truth[k], n) for k in tk]) if tk else np.zeros((0, n), bool)
    H = np.stack([_mask(hyp[k], n, shift) for k in hk]) if hk else np.zeros((0, n), bool)
    mapping: dict[str, str] = {}
    if tk and hk:
        ov = (H[:, None, :] & T[None, :, :]).sum(-1).astype(float)
        for i, j in _hungarian(-ov):
            if ov[i, j] > 0:
                mapping[hk[i]] = tk[j]
    nt = T.sum(0)
    nh = H.sum(0)
    total = float(nt.sum())
    if total == 0:
        return {"der": None, "miss": None, "false_alarm": None, "confusion": None, "mapping": mapping}
    correct = np.zeros(n)
    for h, t in mapping.items():
        correct += (H[hk.index(h)] & T[tk.index(t)])
    miss = np.maximum(nt - nh, 0).sum()
    fa = np.maximum(nh - nt, 0).sum()
    conf = (np.minimum(nt, nh) - correct).sum()
    return {"der": float((miss + fa + conf) / total), "miss": float(miss / total),
            "false_alarm": float(fa / total), "confusion": float(conf / total), "mapping": mapping}


def best_shift(truth: dict, hyp: dict, duration: float, max_shift: float = 5.0, step: float = 0.1) -> float:
    best, best_d = 0.0, None
    k = int(round(max_shift / step))
    for i in range(-k, k + 1):
        s = i * step
        d = der(truth, hyp, duration, shift=s)["der"]
        if d is not None and (best_d is None or d < best_d - 1e-9 or (abs(d - best_d) < 1e-9 and abs(s) < abs(best))):
            best, best_d = s, d
    return round(best, 2)


def _changes(spk: dict, min_gap_merge: float = 0.0) -> list[float]:
    ev = sorted((s, e, k) for k, v in spk.items() for s, e in v)
    out = []
    prev = None
    for s, e, k in ev:
        if prev is not None and k != prev:
            out.append(s)
        prev = k
    return out


def boundary_error(truth: dict, hyp: dict) -> dict:
    tc, hc = _changes(truth), _changes(hyp)
    if not tc or not hc:
        return {"median_s": None, "within_1s": None, "n": len(tc)}
    hc_arr = np.array(hc)
    d = [float(np.min(np.abs(hc_arr - t))) for t in tc]
    return {"median_s": float(np.median(d)), "within_1s": float(np.mean(np.array(d) <= 1.0)), "n": len(tc)}


_TOK = re.compile(r"[a-z0-9']+")


def norm_words(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]*\]|<[^>]*>|\([^)]*\)", " ", text.lower())
    return [w.strip("'") for w in _TOK.findall(text) if w.strip("'")]


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = norm_words(ref), norm_words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


def _rank(x):
    x = np.asarray(x, float)
    order = x.argsort()
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    # average ties
    for v in np.unique(x):
        idx = np.where(x == v)[0]
        ranks[idx] = ranks[idx].mean()
    return ranks


def pearson(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b) -> float | None:
    if len(a) < 3:
        return None
    return pearson(_rank(a), _rank(b))
