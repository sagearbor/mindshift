"""Nudge-quality yardstick: U_lambda/h with session-level bootstrap CIs.

    U_lambda / h = (helpful_spoken - lambda * unhelpful_spoken) / audio_hours

Silence scores 0 by construction, so a gate/prompt only beats silence when
its spoken lines are helpful more than 1/(1+lambda) of the time (67% at the
default lambda = 2). Same definition as tmp/recordings/judge/validity_F/
quality_metric.py (the pre-registered yardstick), re-implemented here so the
numbers are reproducible from the repo; the resampling unit is the SESSION
(lines inside one recording are not independent), and the audio hours are
resampled with the sessions.

Pure functions, no I/O: the driver is quiet_coach_live_eval.py.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

LAMBDA = 2.0


def value(helpful: bool, lam: float = LAMBDA) -> float:
    return 1.0 if helpful else -lam


def u_per_h(labels: list[bool], hours: float, lam: float = LAMBDA) -> float:
    return sum(value(h, lam) for h in labels) / hours if hours else 0.0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    k = (len(s) - 1) * q / 100.0
    f = math.floor(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(lines: list[dict], hours: float, lam: float = LAMBDA) -> dict:
    """lines: [{"helpful": bool, ...}] = the SPOKEN lines of one arm."""
    n = len(lines)
    h = sum(1 for ln in lines if ln["helpful"])
    return {
        "n_spoken": n, "helpful": h, "unhelpful": n - h,
        "helpful_share": (h / n) if n else None,
        "helpful_share_ci95": wilson(h, n),
        "hours": hours,
        "spoken_per_h": n / hours if hours else None,
        "helpful_per_h": h / hours if hours else None,
        "unhelpful_per_h": (n - h) / hours if hours else None,
        "U_per_h": u_per_h([ln["helpful"] for ln in lines], hours, lam),
    }


def bootstrap(
    arm_a: list[dict], arm_b: list[dict] | None, hours_by_item: dict[str, float],
    *, lam: float = LAMBDA, B: int = 4000, seed: int = 20261105,
) -> dict:
    """Bootstrap over sessions. Every session in ``hours_by_item`` is a
    resampling unit (including sessions where an arm spoke nothing - that is
    exactly what silence-is-zero means). Returns U/h CIs for each arm and,
    when ``arm_b`` is given, the paired delta (a - b) CI."""
    items = sorted(hours_by_item)
    by_a: dict[str, list[dict]] = defaultdict(list)
    by_b: dict[str, list[dict]] = defaultdict(list)
    for ln in arm_a:
        by_a[ln["item"]].append(ln)
    for ln in arm_b or []:
        by_b[ln["item"]].append(ln)
    rng = random.Random(seed)
    ua, ub, dd = [], [], []
    for _ in range(B):
        pick = [rng.choice(items) for _ in items]
        hrs = sum(hours_by_item[i] for i in pick)
        if hrs <= 0:
            continue
        a = sum(value(ln["helpful"], lam) for i in pick for ln in by_a[i]) / hrs
        ua.append(a)
        if arm_b is not None:
            b = sum(value(ln["helpful"], lam) for i in pick for ln in by_b[i]) / hrs
            ub.append(b)
            dd.append(a - b)
    out = {"B": len(ua), "seed": seed, "a_ci95": (_pct(ua, 2.5), _pct(ua, 97.5))}
    if arm_b is not None:
        out["b_ci95"] = (_pct(ub, 2.5), _pct(ub, 97.5))
        out["delta_ci95"] = (_pct(dd, 2.5), _pct(dd, 97.5))
        out["p_delta_le_0"] = sum(1 for x in dd if x <= 0) / len(dd)
    return out


def combine_judges(grades: dict[str, dict[str, bool]], mode: str = "both") -> dict[str, bool]:
    """grades: {judge: {line_id: helpful}} -> {line_id: helpful}.

    ``both`` (the plan's rule: positive = every judge says helpful), ``any``
    (lenient sensitivity), or a single judge's name."""
    judges = sorted(grades)
    ids = set.intersection(*(set(g) for g in grades.values())) if grades else set()
    if mode == "both":
        return {i: all(grades[j][i] for j in judges) for i in ids}
    if mode == "any":
        return {i: any(grades[j][i] for j in judges) for i in ids}
    return {i: grades[mode][i] for i in grades[mode]}


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if not n or n != len(b):
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)
