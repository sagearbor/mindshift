"""Cross-corpus sanity check on RAVDESS (EVAL ONLY, CC BY-NC-SA): 96 clips, 16 per shared emotion,
reference = the actor's INTENDED emotion (no crowd votes on disk).  Audio only.

API labellers run here; emotion2vec+ is scored from emotion_results_ravdess_e2v.jsonl if present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from cremad import EMOS
from env import MAIN, OUT
from stats import binary_class_stats, kappa_ci

R = MAIN / "tmp/ravdess/audio"
CODE = {"01": "neutral", "03": "happy", "04": "sad", "05": "anger", "06": "fear", "07": "disgust"}
SEED = 20261105


def sample() -> list[dict]:
    files = sorted(R.glob("Actor_*/*.wav"))
    by = {e: [] for e in CODE.values()}
    for f in files:
        p = f.stem.split("-")
        if p[0] == "03" and p[2] in CODE:
            by[CODE[p[2]]].append(f)
    rng = np.random.default_rng(SEED)
    out = []
    for e, lst in by.items():
        for i in sorted(rng.choice(len(lst), 16, replace=False).tolist()):
            out.append({"id": "rav_" + lst[i].stem, "wav": str(lst[i]), "intended": e,
                        "intensity": "strong" if lst[i].stem.split("-")[3] == "02" else "normal"})
    return out


def score(rows, s):
    ok = [x for x in s if rows.get(x["id"], {}).get("label")]
    pred = [rows[x["id"]]["label"] for x in ok]
    ref = [x["intended"] for x in ok]
    return {"n": len(ok), "kappa": kappa_ci(pred, ref, EMOS), "accuracy": float(np.mean(np.array(pred) == np.array(ref))),
            "recall": {e: binary_class_stats(pred, ref, e)["recall"][0] for e in EMOS},
            "precision": {e: binary_class_stats(pred, ref, e)["precision"][0] for e in EMOS}}


def main(models):
    from emotion_labellers import label
    s = sample()
    (OUT / "ravdess_sample.json").write_text(json.dumps(s, indent=1))
    res = {}
    for m in models:
        rows = {}
        for x in s:
            rows[x["id"]] = label(m, "audio_only", x["id"], Path(x["wav"]), None, "emo-ravdess")
        res[m] = score(rows, s)
    p = OUT / "emotion_results_ravdess_e2v.jsonl"
    if p.exists():
        rows = {json.loads(l)["id"]: json.loads(l) for l in open(p)}
        res["emotion2vec-plus-large"] = score(rows, s)
    (OUT / "ravdess_check.json").write_text(json.dumps(res, indent=1))
    for m, r in res.items():
        print(m, r["n"], [round(x, 3) for x in r["kappa"]], round(r["accuracy"], 3), {k: round(v or 0, 2) for k, v in r["recall"].items()})


if __name__ == "__main__":
    main(sys.argv[1:])
