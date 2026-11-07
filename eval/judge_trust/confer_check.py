"""Language-agnostic check on CONFER (Greek, eval-only): do labellers rank high-conflict windows
above low-conflict windows of the SAME recording?  Only vocal arousal is asked; no words used.

Per recording (>= 40 s of windows): the 3 highest- and 3 lowest-conflict 5 s windows (10-rater
per-second curve, mean over the window), at least 5 s apart.  Metric = within-recording pairwise
AUC (P[hot window scored above cold window], ties 0.5), with a bootstrap CI over recordings;
baseline = window loudness (dBFS), which earlier scored rho 0.00 against these raters.
Exploratory (PREREG.md lists it as optional); not gated.
"""
from __future__ import annotations

import io
import json
import sys
import wave
from pathlib import Path

import numpy as np

import ledger
from emotion_labellers import BACKENDS, EST
from env import MAIN, OUT

C = MAIN / "tmp/corpora/confer"
SR = 16000
W = 5
PROMPT = ("Listen to how the speakers SOUND (loudness, pitch, tempo, tension, overlap), ignoring the words and the "
          "language. Rate vocal arousal/heat from 1 (very calm) to 5 (very heated, raised voices). Also give the "
          'probability that the exchange sounds angry or confrontational. Reply JSON only: {"arousal": 3, "p_conflict": 0.0}')


def read_wav(p: Path) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(str(p), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    assert sr == SR, sr
    return y


def to_wav_bytes(y: np.ndarray) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    return b.getvalue()


def windows() -> list[dict]:
    man = json.loads((C / "heatmap_manifest.json").read_text())
    out = []
    for e in man:
        h = np.asarray(e["conflict_per_second"], float)
        p = C / "clips16k" / Path(e["audio"]).name
        if not p.exists():
            p = C / e["audio"]
        y = read_wav(p)
        n = min(int(len(y) / SR) - W + 1, len(h) - W + 1)
        if n < 40:
            continue
        sc = np.array([h[s:s + W].mean() for s in range(n)])
        chosen = []

        def take(order):
            got = []
            for s in order:
                if all(abs(s - c) >= W for c in chosen + got):
                    got.append(int(s))
                if len(got) == 3:
                    break
            return got
        hot = take(np.argsort(-sc))
        chosen += hot
        cold = take(np.argsort(sc))
        for tag, ss in (("hot", hot), ("cold", cold)):
            for s in ss:
                seg = y[s * SR:(s + W) * SR]
                out.append({"rec": e["id"], "start": s, "tag": tag, "human": float(sc[s]),
                            "dbfs": float(10 * np.log10(np.mean(seg.astype(np.float64) ** 2) + 1e-12)), "_y": seg})
    return out


def auc_within(rows, key) -> tuple[float, float, float]:
    recs = sorted({r["rec"] for r in rows})
    per = []
    for rec in recs:
        hot = [r[key] for r in rows if r["rec"] == rec and r["tag"] == "hot" and r.get(key) is not None]
        cold = [r[key] for r in rows if r["rec"] == rec and r["tag"] == "cold" and r.get(key) is not None]
        if not hot or not cold:
            continue
        per.append(np.mean([(a > b) + 0.5 * (a == b) for a in hot for b in cold]))
    per = np.array(per)
    rng = np.random.default_rng(3)
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(2000)]
    return float(per.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main(models: list[str]) -> None:
    rows = windows()
    print(len(rows), "windows from", len({r["rec"] for r in rows}), "recordings", flush=True)
    for m in models:
        for r in rows:
            wav = to_wav_bytes(r["_y"])
            key = ledger.cache_key("confer-v1", m, PROMPT, wav)
            hit = ledger.cache_get(key)
            if hit is None:
                ledger.check_budget(EST[m])
                raw = BACKENDS[m](m, PROMPT, wav, "confer-arousal")
                try:
                    import re
                    hit = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
                    ledger.cache_put(key, hit)
                except Exception:  # noqa: BLE001
                    hit = {}
            r[f"{m}:arousal"] = float(hit["arousal"]) if hit.get("arousal") is not None else None
            r[f"{m}:p_conflict"] = float(hit["p_conflict"]) if hit.get("p_conflict") is not None else None
    res = {"n_windows": len(rows), "n_recs": len({r["rec"] for r in rows}), "auc": {"dbfs (loudness)": auc_within(rows, "dbfs")}}
    for m in models:
        for f in ("arousal", "p_conflict"):
            res["auc"][f"{m}:{f}"] = auc_within(rows, f"{m}:{f}")
    (OUT / "confer_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["gemini-3.8-flash"])
