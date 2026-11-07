"""CREMA-D crowd votes (voice-only presentations), frozen balanced sample, human ceiling.

Human reference = plurality of the VOICE-ONLY crowd votes (queryType 1, ~9.8 raters
per clip, 2,443 raters; finishedResponses.csv, repeats/practice removed).
Ties are excluded from the sample.  Soft label = vote distribution.

Human ceiling = for every sampled clip, ONE rater (seeded random) vs the plurality of
the remaining raters (tie -> clip dropped for that statistic).  This is what a single
careful human labeller achieves against the crowd, i.e. the bar a model labeller has
to reach to be "as good as one more human".
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np

from env import MAIN, OUT

ROOT = MAIN / "tmp/corpora/cremad"
EMOS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
LETTER = {"A": "anger", "D": "disgust", "F": "fear", "H": "happy", "N": "neutral", "S": "sad"}
CODE = {"ANG": "anger", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
SENTENCES = {
    "IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.", "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.", "TAI": "The airplane is almost full.", "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.", "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.", "ITS": "I think I've seen this before.", "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}
BROKEN = {"1076_MTI_NEU_XX", "1076_MTI_SAD_XX"}
SEED = 20261105


def votes() -> dict[str, list[tuple[str, str]]]:
    """clip -> [(rater_id, emotion)] for voice-only presentations."""
    out: dict[str, list] = defaultdict(list)
    with open(ROOT / "finishedResponses.csv") as f:
        for r in csv.DictReader(f):
            if r["queryType"] == "1" and r["respEmo"] in LETTER:
                out[r["clipName"]].append((r["localid"], LETTER[r["respEmo"]]))
    return out


def plurality(labels: list[str]):
    c = Counter(labels).most_common()
    if not c or (len(c) > 1 and c[0][1] == c[1][1]):
        return None
    return c[0][0]


def dist(labels: list[str]) -> list[float]:
    c = Counter(labels)
    n = sum(c.values())
    return [c[e] / n for e in EMOS]


def build_sample(per_class: int = 50) -> list[dict]:
    v = votes()
    by = defaultdict(list)
    for clip, vs in sorted(v.items()):
        if clip in BROKEN or not (ROOT / "AudioWAV" / f"{clip}.wav").exists() or len(vs) < 6:
            continue
        lab = [e for _, e in vs]
        maj = plurality(lab)
        if maj is None:
            continue
        by[maj].append(clip)
    rng = np.random.default_rng(SEED)
    sample = []
    for e in EMOS:
        pool = by[e]
        pick = sorted(rng.choice(len(pool), size=min(per_class, len(pool)), replace=False).tolist())
        for i in pick:
            clip = pool[i]
            lab = [x for _, x in v[clip]]
            wav = ROOT / "AudioWAV" / f"{clip}.wav"
            sample.append({
                "clip": clip, "actor": clip[:4], "sentence": SENTENCES[clip.split("_")[1]],
                "intended": CODE[clip.split("_")[2]], "crowd_majority": e, "crowd_dist": dist(lab),
                "n_votes": len(lab), "crowd_agreement": max(dist(lab)),
                "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
            })
    return sample


def human_ceiling(sample: list[dict]) -> dict:
    """Single held-out rater vs plurality of the others, on the sampled clips."""
    from stats import binary_class_stats, kappa_ci
    v = votes()
    rng = np.random.default_rng(SEED + 1)
    pred, ref = [], []
    for s in sample:
        vs = v[s["clip"]]
        k = int(rng.integers(0, len(vs)))
        rest = [e for j, (_, e) in enumerate(vs) if j != k]
        maj = plurality(rest)
        if maj is None:
            continue
        pred.append(vs[k][1])
        ref.append(maj)
    k, lo, hi = kappa_ci(pred, ref, EMOS)
    # pooled over EVERY rater (each rater vs plurality of the others): more stable point estimate
    from stats import cohen_kappa
    pp, rr = [], []
    for s in sample:
        vs = v[s["clip"]]
        for j in range(len(vs)):
            maj = plurality([e for i, (_, e) in enumerate(vs) if i != j])
            if maj is not None:
                pp.append(vs[j][1])
                rr.append(maj)
    pooled = {"n_pairs": len(pp), "overall_kappa": cohen_kappa(pp, rr, EMOS),
              "per_class_kappa": {e: cohen_kappa(np.array(pp) == e, np.array(rr) == e, [True, False]) for e in EMOS}}
    return {"n": len(pred), "overall_kappa": (k, lo, hi), "pooled_all_raters": pooled,
            "accuracy": float(np.mean(np.array(pred) == np.array(ref))),
            "per_class": {e: binary_class_stats(pred, ref, e) for e in EMOS}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = build_sample()
    manifest = {"what": "CREMA-D voice-only crowd sample, balanced by crowd plurality, seed %d" % SEED,
                "n": len(s), "items": s,
                "manifest_sha256": hashlib.sha256(json.dumps([x["sha256"] for x in s]).encode()).hexdigest()}
    (OUT / "cremad_sample.json").write_text(json.dumps(manifest, indent=1))
    hc = human_ceiling(s)
    (OUT / "cremad_human_ceiling.json").write_text(json.dumps(hc, indent=1))
    print(len(s), "clips; manifest", manifest["manifest_sha256"][:16])
    print("human single-rater kappa", hc["overall_kappa"], "acc", hc["accuracy"])
    for e, d in hc["per_class"].items():
        print(e, {k: (round(x[0], 3) if isinstance(x, tuple) and x[0] is not None else x) for k, x in d.items()})


if __name__ == "__main__":
    main()
