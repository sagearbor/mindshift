"""A SECOND OPINION on heat, per window, for every recording — no human labels needed.

heat_map.py can only say "how well we measure" where ground truth exists, and
today that is three scripted scenes. This adds a reference that exists for ANY
recording: tone_id's dimensional model (WavLM-large, arousal/valence/dominance),
scored over fixed 5 s windows of the same audio the loudness chain sees.

Per recording it then reports how well our loudness-derived heat AGREES with
that second opinion (Spearman correlation across windows), plus the model's own
view of the recording's volatility (sd of arousal) — an independent x-axis to
set beside loudness volatility.

Honest framing: the reference is a model, not a person. Agreement with it is
evidence, not proof — but disagreement is a strong hint that loudness is
measuring something other than heat on that recording, which is exactly the
anger-vs-enthusiasm failure the corpus rubric found.

    python scripts/heat_reference.py          # writes tmp/heat-map/heat_reference.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "server"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("MINDSHIFT_TONE_AUDIO", "dark")

import conversation_audit as ca  # noqa: E402
import heat_map as hm  # noqa: E402
import tone_id  # noqa: E402

WIN_S = 5.0
OUT = REPO / "tmp/heat-map/heat_reference.json"


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 6 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def score(entry: dict) -> dict | None:
    pcm, sr = hm.load_pcm(entry["audio"])
    if sr != tone_id.TARGET_SR:
        # Two of the older TTS fixtures are 44.1/48 kHz. Resample rather than
        # skip — a recording silently missing from the graph is worse than one
        # that cost a resample.
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(sr, tone_id.TARGET_SR)
        pcm = np.clip(resample_poly(pcm.astype(np.float64), tone_id.TARGET_SR // g, sr // g), -32768, 32767).astype("<i2")
        sr = tone_id.TARGET_SR
    n = int(sr * WIN_S)
    count = len(pcm) // n
    # Our chain's 1 s windows, then pooled to the same 5 s grid.
    over1 = ca.db_over_baseline(ca.windows_dbfs(pcm, sr))
    ours, arousal, valence = [], [], []
    x = pcm.astype(np.float32) / 32768.0
    for i in range(count):
        seg = x[i * n:(i + 1) * n]
        o = over1[i * 5:(i + 1) * 5]
        voiced = o[o != 0.0]
        if len(voiced) == 0:
            continue                       # silence: nothing to compare
        try:
            r = tone_id.classify_pcm(seg, sr)
        except Exception:
            continue
        sc = r.get("scores") or {}
        if "valence" not in sc:
            continue
        ours.append(float(np.mean(voiced)))
        arousal.append(float(r["arousal"]))
        valence.append(float(sc["valence"]))
    if len(ours) < 3:
        return None
    a, v, o = np.array(arousal), np.array(valence), np.array(ours)
    return {
        "id": entry["id"], "windows": len(o),
        "agreement_loudness_vs_arousal": spearman(o, a),
        "ref_arousal_mean": round(float(a.mean()), 3),
        "ref_arousal_sd": round(float(a.std()), 3),
        "ref_valence_mean": round(float(v.mean()), 3),
        # share of loud windows the model calls PLEASANT — the anger-vs-joy
        # confusion, per recording
        "loud_but_pleasant": round(float(np.mean((o >= 6.0) & (v > 0.48))) if (o >= 6.0).any() else 0.0, 3),
        "loud_windows": int((o >= 6.0).sum()),
    }


def main() -> int:
    entries = hm.ami_entries() + hm.fixture_entries() + hm.extra_entries()
    done = {}
    if OUT.exists():
        done = {r["id"]: r for r in json.loads(OUT.read_text())}
    t0 = time.time()
    for e in entries:
        if e["id"] in done:
            print(f"  ✓ {e['id']}")
            continue
        r = score(e)
        if r:
            done[e["id"]] = r
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(list(done.values()), indent=1))
            print(f"  {e['id']:26} windows {r['windows']:4}  agreement {r['agreement_loudness_vs_arousal']!s:>6}  "
                  f"loud-but-pleasant {r['loud_but_pleasant']:.2f}   ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n{len(done)} recordings -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
