"""Step 2 (runs under tmp/venv-voice — the production ECAPA venv): embed the
cached pyannote units and the GT intervals with OUR pinned ECAPA model via
server.speaker_id.embed_pcm.  Writes cache/<fixture>_ecapa.npz.

    tmp/venv-voice/bin/python docs/research/2026-08-29-voice-separation/C-pyannote/embed_ecapa.py [fixture ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CACHE = HERE / "cache"
sys.path.insert(0, str(ROOT / "server"))
import speaker_id  # noqa: E402

MIN_SEC = 0.3  # below this ECAPA output is noise; pyannote would still embed


def embed_many(arrs: dict[str, np.ndarray]) -> tuple[dict, float]:
    t0 = time.perf_counter()
    out = {}
    for k, pcm in arrs.items():
        if len(pcm) < MIN_SEC * 16000:
            out[k] = np.full(192, np.nan, dtype=np.float32)
            continue
        out[k] = speaker_id.embed_pcm(pcm, 16000)
    return out, time.perf_counter() - t0


def main(names):
    for name in names:
        u = np.load(CACHE / f"{name}_units.npz")
        g = np.load(CACHE / f"{name}_gt_units.npz")
        units = {k: u[k] for k in u.files}
        gts = {k: g[k] for k in g.files}
        ue, tu = embed_many(units)
        ge, tg = embed_many(gts)
        np.savez(CACHE / f"{name}_ecapa.npz", **{f"unit_{k}": v for k, v in ue.items()},
                 **{f"gt_{k}": v for k, v in ge.items()})
        info = {"units": len(units), "t_units_s": round(tu, 2), "gt": len(gts), "t_gt_s": round(tg, 2),
                "nan_units": int(sum(np.isnan(v).any() for v in ue.values()))}
        (CACHE / f"{name}_ecapa.json").write_text(json.dumps(info))
        print(name, info, flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.name[: -len("_units.npz")] for p in CACHE.glob("*_units.npz"))
    main(names)
