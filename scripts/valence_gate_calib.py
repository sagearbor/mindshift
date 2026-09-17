"""Calibrate the valence veto — the threshold behind `relay.VALENCE_VETO_MAX`.

The heat rubric (scripts/heat_rubric.py) established that the shipped +6 dB
rung fires on 44.5% of HAPPY speech, because loudness measures AROUSAL and
anger and joy are both high-arousal. `tone_id`'s dimensional backend also
returns VALENCE — the pleasant/unpleasant axis — and every caller had only
ever read `arousal`.

This script joins the two artefacts those sessions left behind:

  tmp/corpora/heat_rubric_cremad.csv   dB over each speaker's OWN neutral
  tmp/corpora/valence_probe.json       tone_id's arousal/valence/dominance

and answers one question: among the clips that ALREADY clear the rung — the
only population a veto can act on — what valence threshold removes the most
happy false alarms while keeping the anger?

Neither input is committed (CREMA-D is ODbL, RAVDESS CC BY-NC-SA); see
docs/DATASETS.md. Regenerate them with scripts/heat_rubric.py and
tmp/corpora/valence_probe.py.

Conclusions: docs/decisions/2026-09-17-valence-veto.md.

    python scripts/valence_gate_calib.py
"""

from __future__ import annotations

import collections
import csv
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPORA = os.path.join(REPO, "tmp", "corpora")
RUBRIC_CSV = os.path.join(CORPORA, "heat_rubric_cremad.csv")
PROBE_JSON = os.path.join(CORPORA, "valence_probe.json")

#: The shipped first rung (watch/vectors.py YELLING_LEVELS).
RUNG_DB = 6.0
#: Keep at least this share of the anger the rung already catches. The veto
#: buys precision with recall; this is the budget for that trade.
MIN_ANGER_KEPT = 0.90


def load() -> list[dict]:
    """Clips present in BOTH the rubric and the valence probe."""
    for path in (RUBRIC_CSV, PROBE_JSON):
        if not os.path.exists(path):
            sys.exit(
                f"missing {path}\n"
                "Corpora are never committed — regenerate with scripts/heat_rubric.py "
                "and tmp/corpora/valence_probe.py (see docs/DATASETS.md)."
            )

    rubric: dict[str, tuple[str, str, float]] = {}
    with open(RUBRIC_CSV) as fh:
        for row in csv.DictReader(fh):
            try:
                rubric[row["path"]] = (
                    row["emotion"], row["speaker"], float(row["db_over_own_neutral"]),
                )
            except (KeyError, TypeError, ValueError):
                continue  # a clip with no usable own-neutral baseline

    joined = []
    for probe in json.load(open(PROBE_JSON)):
        hit = rubric.get(probe.get("path", ""))
        if hit is None:
            continue
        emotion, speaker, db = hit
        joined.append({
            "emotion": emotion, "speaker": speaker, "db": db,
            "valence": float(probe["valence"]), "arousal": float(probe["arousal"]),
        })
    return joined


def best_threshold(clips: list[dict]) -> float | None:
    """LOWEST veto threshold that still keeps MIN_ANGER_KEPT of the anger.

    Lowest, not highest. Both quantities move monotonically with the
    threshold: raising it lets more anger through (good) AND more happy
    speech (bad). So "keep >= 90% of the anger" is satisfied by every
    threshold above some point, and the best of those is the SMALLEST one —
    it vetoes the most happy speech for the recall we agreed to spend.
    Taking the largest instead would trivially return the end of the sweep
    and veto almost nothing.
    """
    angry = [c for c in clips if c["emotion"] == "angry"]
    if not angry:
        return None
    for step in range(20, 61):
        thr = step / 100.0
        kept = sum(1 for c in angry if c["valence"] <= thr) / len(angry)
        if kept >= MIN_ANGER_KEPT:
            return thr
    return None


def rate(clips: list[dict], emotion: str, thr: float | None) -> float:
    pool = [c for c in clips if c["emotion"] == emotion]
    if not pool:
        return 0.0
    if thr is None:
        return 1.0
    return sum(1 for c in pool if c["valence"] <= thr) / len(pool)


def main() -> None:
    clips = load()
    print(f"joined {len(clips)} clips  {dict(collections.Counter(c['emotion'] for c in clips))}\n")

    print("-- median valence / arousal / dB per label --")
    for label in ("angry", "happy", "neutral", "sad"):
        pool = [c for c in clips if c["emotion"] == label]
        if pool:
            print(f"   {label:8s} n={len(pool):4d}  valence {statistics.median(c['valence'] for c in pool):.3f}"
                  f"   arousal {statistics.median(c['arousal'] for c in pool):.3f}"
                  f"   dB {statistics.median(c['db'] for c in pool):+5.1f}")

    fires = [c for c in clips if c["db"] >= RUNG_DB]
    print(f"\n-- the only population a veto can act on: clips already >= +{RUNG_DB:.0f} dB --")
    print(f"   n={len(fires)}  {dict(collections.Counter(c['emotion'] for c in fires))}")

    total = {e: sum(1 for c in clips if c["emotion"] == e) for e in ("angry", "happy")}
    caught = {e: sum(1 for c in fires if c["emotion"] == e) for e in ("angry", "happy")}
    print(f"   shipped: angry recall {caught['angry']/total['angry']:.1%}, "
          f"HAPPY false alarm {caught['happy']/total['happy']:.1%}")

    print("\n-- veto sweep: fire only if dB >= rung AND valence <= thr --")
    print(f"   {'thr':>5} {'angry recall':>13} {'HAPPY FA':>9} {'anger kept':>11} {'happy FA removed':>17}")
    for step in range(28, 61, 2):
        thr = step / 100.0
        a = sum(1 for c in fires if c["emotion"] == "angry" and c["valence"] <= thr)
        h = sum(1 for c in fires if c["emotion"] == "happy" and c["valence"] <= thr)
        print(f"   {thr:5.2f} {a/total['angry']:12.1%} {h/total['happy']:8.1%} "
              f"{a/caught['angry']:10.1%} {1 - h/caught['happy']:16.1%}")

    thr = best_threshold(fires)
    a = sum(1 for c in fires if c["emotion"] == "angry" and c["valence"] <= thr)
    h = sum(1 for c in fires if c["emotion"] == "happy" and c["valence"] <= thr)
    print(f"\n-- chosen: lowest threshold keeping >= {MIN_ANGER_KEPT:.0%} of caught anger = {thr:.2f} --")
    print(f"   angry recall {a/total['angry']:.1%} (was {caught['angry']/total['angry']:.1%}), "
          f"HAPPY FA {h/total['happy']:.1%} (was {caught['happy']/total['happy']:.1%})")
    print(f"   anger kept {a/caught['angry']:.1%}, happy false alarms removed {1 - h/caught['happy']:.1%}")

    print("\n-- per rung: where does the veto actually pay? --")
    print(f"   {'rung':>22} {'n ang':>6} {'ang vetoed':>11} {'n hap':>6} {'hap vetoed':>11}")
    for lo, hi, name in ((6, 10, "level 1  +6..10 dB"), (10, 14, "level 2 +10..14 dB"), (14, 1e9, "level 3  >=+14 dB")):
        band = [c for c in fires if lo <= c["db"] < hi]
        ang = [c for c in band if c["emotion"] == "angry"]
        hap = [c for c in band if c["emotion"] == "happy"]
        av = sum(1 for c in ang if c["valence"] > thr)
        hv = sum(1 for c in hap if c["valence"] > thr)
        fa = f"{av/len(ang):.0%}" if ang else "   -"
        fh = f"{hv/len(hap):.0%}" if hap else "   -"
        print(f"   {name:>22} {len(ang):>6} {av:>5} ({fa:>4}) {len(hap):>6} {hv:>5} ({fh:>4})")
    print("   Loud+happy concentrates at the TOP rung, so the veto applies at every rung.")

    # --- does the threshold transfer to speakers it was never chosen on?
    print("\n-- leave-one-speaker-out (threshold re-chosen per fold) --")
    speakers = sorted({c["speaker"] for c in fires})
    oof_a = oof_h = 0
    for held in speakers:
        train = [c for c in fires if c["speaker"] != held]
        test = [c for c in fires if c["speaker"] == held]
        fold_thr = best_threshold(train)
        if fold_thr is None:
            continue
        oof_a += sum(1 for c in test if c["emotion"] == "angry" and c["valence"] <= fold_thr)
        oof_h += sum(1 for c in test if c["emotion"] == "happy" and c["valence"] <= fold_thr)
    print(f"   {len(speakers)} held-out speakers")
    print(f"   anger kept {oof_a}/{caught['angry']} = {oof_a/caught['angry']:.1%}")
    print(f"   happy false alarms removed {1 - oof_h/caught['happy']:.1%}")
    print("   Matching the in-sample numbers means the threshold is not overfit to these speakers.")


if __name__ == "__main__":
    main()
