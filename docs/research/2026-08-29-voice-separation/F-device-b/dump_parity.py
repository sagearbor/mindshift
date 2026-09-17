"""Parity fixtures for the phone's port of the window pass (F-device-b).

Reads the window embeddings the node replay wrote (tmp/f-device-b/
emb_<fixture>.json, rounded to 1e-7 so both sides see identical inputs),
runs the SHIPPED numpy functions on them — ``diarize_sliding_window``'s
``refine_affinity`` / ``eigengap_k`` / ``spectral_labels`` / ``mode_filter``
/ ``window_label_runs`` — and dumps inputs + outputs to
``parity_<fixture>.json`` for apps/mobile/__tests__/diarizeWindows.parity
.test.ts, which re-runs the TypeScript port on the same embeddings and
asserts identical k and >= 0.99 label agreement. Also dumps a
``numpy.random.default_rng`` draw sequence so the RNG port is checked bit
for bit.

Run: tmp/venv-voice/bin/python docs/research/2026-08-29-voice-separation/F-device-b/dump_parity.py [fixture ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "server"))

import diarize_local  # noqa: E402
import diarize_sliding_window as dsw  # noqa: E402

TMP = ROOT / "tmp" / "f-device-b"
DEFAULT_FIXTURES = ["family_real", "poker6"]
# B's eigengap range (run_b.py SPEC_MAX_K) and production's (MAX_SPEAKERS_LOCAL).
K_POLICIES = {"b8": (8, 1), "prod6": (diarize_local.MAX_SPEAKERS_LOCAL, 2)}


def rng_dump() -> dict:
    out = {}
    for seed in (0, 1, 123456789):
        rng = np.random.default_rng(seed)
        seq = []
        for _ in range(40):
            seq.append(["i", 600, int(rng.integers(600))])
            seq.append(["i", 7, int(rng.integers(7))])
            seq.append(["r", None, float(rng.random())])
            seq.append(["i", 3, int(rng.integers(3))])
        p = [0.1, 0.2, 0.3, 0.4]
        out[str(seed)] = {"seq": seq, "choice_p": p, "choice": [int(rng.choice(4, p=p)) for _ in range(20)]}
    return out


def dump(name: str) -> None:
    src = json.loads((TMP / f"emb_{name}.json").read_text())
    embs = np.asarray(src["embeddings"], dtype=np.float64)
    starts = np.asarray(src["starts"], dtype=np.float64)
    hop, window, duration = float(src["hop_s"]), float(src["window_s"]), float(src["duration_s"])
    aff = dsw.refine_affinity(embs)
    out = {
        "fixture": name,
        "sample_rate": src["sample_rate"],
        "window_s": window, "hop_s": hop, "duration_s": duration,
        "starts": src["starts"],
        "embeddings": src["embeddings"],
        "affinity_row_sums": [float(x) for x in aff.sum(axis=1)],
        "affinity_trace": float(np.trace(aff)),
        "policies": {},
    }
    for pname, (max_k, min_k) in K_POLICIES.items():
        k_eig, eig = dsw.eigengap_k(aff, max_k)
        k = max(min_k, min(max_k, k_eig, len(starts)))
        raw = dsw.spectral_labels(aff, k)
        sm = dsw.mode_filter(raw, starts, hop)
        runs = dsw.window_label_runs(sm, starts, window, 0.0, duration)
        out["policies"][pname] = {
            "max_k": max_k, "min_k": min_k, "k_eigengap": int(k_eig), "k": int(k),
            "eigenvalues": eig,
            "raw_labels": [int(x) for x in raw],
            "labels": [int(x) for x in sm],
            "segments": [[float(a), float(b), int(c)] for a, b, c in runs],
        }
    (HERE / f"parity_{name}.json").write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"{name}: {len(starts)} windows, k = " + ", ".join(f"{p}:{v['k']}(eig {v['k_eigengap']})" for p, v in out["policies"].items()))


def main(argv: list[str]) -> None:
    names = argv or DEFAULT_FIXTURES
    (HERE / "parity_rng.json").write_text(json.dumps(rng_dump(), separators=(",", ":")) + "\n")
    print("parity_rng.json: numpy", np.__version__)
    for n in names:
        dump(n)


if __name__ == "__main__":
    main(sys.argv[1:])
