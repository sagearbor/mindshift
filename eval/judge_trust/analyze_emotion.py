"""Score emotion labellers against CREMA-D crowd votes; apply the pre-registered bars (PREREG.md A)."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from cremad import EMOS, plurality, votes
from emotion_labellers import TTS_SENTENCES, contra_emotion
from env import OUT
from stats import binary_class_stats, kappa_ci, tvd

H = {"anger": 0.62, "disgust": 0.38, "fear": 0.45, "happy": 0.60, "neutral": 0.35, "sad": 0.50}  # PREREG
H_OVERALL = 0.48


def load_results() -> dict:
    """(model, cond) -> {id: result} keeping the last parsed row."""
    res: dict = defaultdict(dict)
    for fn in ("emotion_results.jsonl", "emotion_results_e2v.jsonl"):
        p = OUT / fn
        if not p.exists():
            continue
        for line in open(p):
            d = json.loads(line)
            key = (d["model"], d["cond"])
            if d.get("label") or d["id"] not in res[key]:
                res[key][d["id"]] = d
    return res


def split_half_tvd(sample) -> tuple[float, float]:
    v = votes()
    rng = np.random.default_rng(7)
    sh, one = [], []
    for s in sample:
        lab = [e for _, e in v[s["clip"]]]
        idx = rng.permutation(len(lab))
        a, b = [lab[i] for i in idx[: len(lab) // 2]], [lab[i] for i in idx[len(lab) // 2:]]
        da = [a.count(e) / len(a) for e in EMOS]
        db = [b.count(e) / len(b) for e in EMOS]
        sh.append(tvd(da, db))
        k = idx[0]
        rest = [lab[i] for i in range(len(lab)) if i != k]
        one.append(tvd([1.0 if e == lab[k] else 0.0 for e in EMOS], [rest.count(e) / len(rest) for e in EMOS]))
    return float(np.mean(sh)), float(np.mean(one))


def verdict(st: dict, e: str) -> str:
    k, lo, _ = st["kappa"]
    p = st["precision"][0]
    if k is None:
        return "NOT TRUSTWORTHY"
    if k >= H[e] and lo >= 0.75 * H[e] and (p or 0) >= 0.60:
        return "TRUSTWORTHY"
    if k >= 0.75 * H[e] and lo >= 0.50 * H[e]:
        return "USABLE WITH CARE"
    return "NOT TRUSTWORTHY"


def score_cond(sample, rows: dict, ref_key="crowd_majority") -> dict | None:
    items = [s for s in sample if s["clip"] in rows]
    ok = [s for s in items if rows[s["clip"]].get("label")]
    if not ok:
        return None
    pred = [rows[s["clip"]]["label"] for s in ok]
    ref = [s[ref_key] for s in ok]
    k = kappa_ci(pred, ref, EMOS)
    out = {"n_attempted": len(items), "n_ok": len(ok), "fail_rate": 1 - len(ok) / max(1, len(items)),
           "kappa": k, "accuracy": float(np.mean(np.array(pred) == np.array(ref))),
           "pred_dist": {e: pred.count(e) / len(pred) for e in EMOS},
           "per_class": {}, "mean_tvd": float(np.mean([tvd(rows[s["clip"]]["probs"], s["crowd_dist"]) for s in ok]))}
    for e in EMOS:
        st = binary_class_stats(pred, ref, e)
        st["verdict"] = verdict(st, e)
        out["per_class"][e] = st
    out["kappa_vs_intended"] = kappa_ci(pred, [s["intended"] for s in ok], EMOS)
    return out


def main() -> dict:
    sample = json.loads((OUT / "cremad_sample.json").read_text())["items"]
    by_clip = {s["clip"]: s for s in sample}
    res = load_results()
    sh_tvd, one_tvd = split_half_tvd(sample)
    report = {"human": json.loads((OUT / "cremad_human_ceiling.json").read_text()),
              "split_half_tvd": sh_tvd, "single_rater_tvd": one_tvd, "models": {}}
    models = sorted({m for m, _ in res})
    for m in models:
        R = {}
        for cond in ("audio_only", "both", "contradict"):
            rows = {k: v for k, v in res.get((m, cond), {}).items() if k in by_clip}
            if rows:
                R[cond] = score_cond(sample, rows)
        # soft-label verdict
        if R.get("audio_only"):
            R["audio_only"]["soft_pass"] = R["audio_only"]["mean_tvd"] <= sh_tvd + 0.05
        # transcript-only: broadcast the per-sentence answer to every clip with that sentence
        t = res.get((m, "text_only"), {})
        if t:
            sent_lab = {k.split("_", 1)[1]: v.get("label") for k, v in t.items()}
            pred = [sent_lab.get(s["clip"].split("_")[1]) for s in sample]
            okp = [(p, s["crowd_majority"]) for p, s in zip(pred, sample) if p]
            kk = kappa_ci([a for a, _ in okp], [b for _, b in okp], EMOS)
            R["text_only"] = {"kappa": kk, "labels_by_sentence": sent_lab, "pass": kk[0] <= 0.10}
        # contradiction leakage
        a, c = res.get((m, "audio_only"), {}), res.get((m, "contradict"), {})
        moved, eligible, to_text_any = 0, 0, 0
        for clip, cr in c.items():
            if clip not in by_clip or not cr.get("label") or not a.get(clip, {}).get("label"):
                continue
            te = contra_emotion(clip, by_clip[clip]["crowd_majority"])
            to_text_any += cr["label"] == te
            if a[clip]["label"] != te:
                eligible += 1
                moved += cr["label"] == te
        if eligible:
            rate = moved / eligible
            R["leak_contradict"] = {"eligible": eligible, "moved_to_text": moved, "rate": rate,
                                    "verdict": "PASS" if rate <= 0.10 else ("WARN" if rate <= 0.25 else "FAIL")}
        # TTS flat voice reading emotional words
        tts = {k: v for k, v in res.get((m, "audio_only"), {}).items() if k.startswith("tts_")}
        if tts:
            emo_items = [(k, v) for k, v in tts.items() if k.split("_")[1] != "neutral" and v.get("label")]
            neu_items = [(k, v) for k, v in tts.items() if k.split("_")[1] == "neutral" and v.get("label")]
            word = sum(v["label"] == k.split("_")[1] for k, v in emo_items) / max(1, len(emo_items))
            R["leak_tts"] = {"n_emotional_words": len(emo_items), "labelled_word_emotion": word,
                             "labelled_neutral": sum(v["label"] == "neutral" for _, v in emo_items) / max(1, len(emo_items)),
                             "neutral_words_labelled_neutral": sum(v["label"] == "neutral" for _, v in neu_items) / max(1, len(neu_items)),
                             "by_word_emotion": {e: [v["label"] for k, v in emo_items if k.split("_")[1] == e]
                                                 for e in TTS_SENTENCES if e != "neutral"},
                             "verdict": "PASS" if word <= 0.20 else ("WARN" if word <= 0.40 else "FAIL")}
        report["models"][m] = R
    # agreement filter: keep a label only where two labellers agree (the cheap "two-labeller" strategy)
    pairs = [("gemini-3.8-flash", "emotion2vec-plus-large"), ("gemini-3.8-flash", "gemini-3.5-flash-lite"),
             ("gemini-3.8-flash", "gpt-audio-1.5")]
    report["agreement_filter"] = {}
    for x, y in pairs:
        A, B = res.get((x, "audio_only"), {}), res.get((y, "audio_only"), {})
        both = [s for s in sample if A.get(s["clip"], {}).get("label") and B.get(s["clip"], {}).get("label")]
        agree = [s for s in both if A[s["clip"]]["label"] == B[s["clip"]]["label"]]
        if not both:
            continue
        pred = [A[s["clip"]]["label"] for s in agree]
        ref = [s["crowd_majority"] for s in agree]
        report["agreement_filter"][f"{x} + {y}"] = {
            "n_both": len(both), "coverage": len(agree) / len(both),
            "accuracy_on_agreed": float(np.mean(np.array(pred) == np.array(ref))) if agree else None,
            "kappa_on_agreed": kappa_ci(pred, ref, EMOS) if len(agree) > 5 else None,
            "per_class_precision": {e: binary_class_stats(pred, ref, e)["precision"] for e in EMOS} if agree else None}
    (OUT / "emotion_analysis.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    r = main()
    for m, R in r["models"].items():
        ao = R.get("audio_only") or {}
        print(m, "audio kappa", [round(x, 3) for x in ao.get("kappa", [])], "acc", round(ao.get("accuracy", 0), 3),
              "fail", round(ao.get("fail_rate", 0), 3), "tvd", round(ao.get("mean_tvd", 0), 3))
        for e, st in (ao.get("per_class") or {}).items():
            print("   ", e, st["verdict"], "k", st["kappa"][0] and round(st["kappa"][0], 2), "P", st["precision"][0] and round(st["precision"][0], 2), "R", st["recall"][0] and round(st["recall"][0], 2))
        for k in ("both", "contradict"):
            if R.get(k):
                print("  ", k, "kappa", round(R[k]["kappa"][0], 3))
        print("  ", "text_only", R.get("text_only", {}).get("kappa"), "leak", R.get("leak_contradict"), "tts", {k: v for k, v in (R.get("leak_tts") or {}).items() if k != "by_word_emotion"})
    print("split-half tvd", r["split_half_tvd"], "single-rater tvd", r["single_rater_tvd"])
    print(json.dumps(r["agreement_filter"], indent=0)[:1500])
