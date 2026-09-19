"""Experiment 3, step 2 — score the enhanced audio. Runs under tmp/venv-voice.

Variants of each fixture's PCM (same sample count, so transcript timings hold):
  orig     production-decoded 16 kHz audio
  dfn      DeepFilterNet3, full attenuation      (cache/<name>_dfn.wav)
  dfn12    DeepFilterNet3, attenuation <= 12 dB  (cache/<name>_dfn12.wav)
  rmsnorm  NO model: every chunk handed to the embedder (window, segment or
           pooled speaker audio) is gain-normalised to RMS 0.05 first
           (speaker_id.embed_pcm / embed_pcm_batch wrapped in this process)

Per variant: approach B (spectral eigengap p=0.80 on the 1.5 s / 0.25 s
grid), the window-level CEILING (each speech window -> nearest true
voiceprint: B's mean-of-windows GT centroid, and the pooled-audio print
production would store), within/cross-speaker window cosine, and the
production diarizer on GT boundaries + (maggiano3) the two real transcripts.

Usage: python exp3_score.py <fixture> <variant> [<variant> ...]   (partials -> cache/exp3_*.json)
       python exp3_score.py --merge                                 (-> results.json["exp3"])
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import common as C
from common import speaker_id

RMS_TARGET = 0.05


def rms_normalised(chunk: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.asarray(chunk, dtype=np.float32) ** 2)))
    return chunk if rms < 1e-6 else (chunk * (RMS_TARGET / rms)).astype(np.float32)


def patch_rms_norm():
    e1, eb = speaker_id.embed_pcm, speaker_id.embed_pcm_batch
    speaker_id.embed_pcm = lambda pcm, sr=speaker_id.TARGET_SR: e1(rms_normalised(pcm), sr)
    speaker_id.embed_pcm_batch = lambda chunks, sr=speaker_id.TARGET_SR: eb([rms_normalised(c) for c in chunks], sr)
    return lambda: setattr(speaker_id, "embed_pcm", e1) or setattr(speaker_id, "embed_pcm_batch", eb)


def load_variant(name: str, variant: str):
    if variant in ("orig", "rmsnorm"):
        return C.load_audio(name)
    return C.wav_read(C.CACHE / f"{name}_{variant}.wav")


def run(name: str, variant: str) -> dict:
    pcm, sr = load_variant(name, variant)
    dur = pcm.size / sr
    unpatch = patch_rms_norm() if variant == "rmsnorm" else (lambda: None)
    try:
        t0 = time.perf_counter()
        we = C.b_windows(name if variant == "orig" else f"{name}_{variant}", pcm, sr)
        starts, embs = we["starts"], we["embs"]
        segs, k_hat = C.b_cluster(starts, embs, dur)
        b = {"k_eigengap": k_hat, "n_windows": int(len(starts)), "vad_thr": round(float(we["vad_thr"]), 4),
             **C.score_segments(name, segs)}
        k_true = C.score.load_fixture(name)["k_true"]          # partition quality with k GIVEN
        segs_o, _ = C.b_cluster(starts, embs, dur, k=k_true)
        b["oracle_k"] = C.score_segments(name, segs_o)
        pools = C.speaker_pcm(name, pcm, sr)
        prints = {l: C.embed_pooled(p, sr) for l, p in pools.items()}
        labs = sorted(prints)
        M = np.stack([prints[l] for l in labs])
        pooled_cos = {f"{a}|{b_}": round(float(prints[a] @ prints[b_]), 3) for i, a in enumerate(labs) for b_ in labs[i + 1:]}
        ceiling = C.window_ceiling(name, starts, embs, dur, prints)
        prod = C.run_production(name, pcm, sr)
        out = {"B": b, "ceiling": ceiling, "pooled_print_cosine": pooled_cos, "production": prod,
               "t_total_s": round(time.perf_counter() - t0, 1)}
    finally:
        unpatch()
    return out


def main(argv: list[str]) -> None:
    if "--merge" in argv:
        res = {"fixtures": {}}
        for f in sorted(C.CACHE.glob("exp3_*.json")):
            _, name, variant = f.stem.split("_", 2)
            res["fixtures"].setdefault(name, {})[variant] = json.loads(f.read_text())
        tp = C.CACHE / "dfn_timing.json"
        if tp.exists():
            res["dfn_timing"] = json.loads(tp.read_text())
        C.merge_results("exp3", res)
        for name, vs in res["fixtures"].items():
            for v, r in vs.items():
                c = r["ceiling"]
                print(f"{name:10s} {v:8s} B acc={r['B']['frame_accuracy']} k={r['B']['k_pred']} pur={r['B']['owner_purity']} | "
                      f"ceiling centroid {c['centroid']['window_acc']}/{c['centroid']['frame_accuracy']} pooled {c.get('pooled', {}).get('window_acc')} "
                      f"within {c['within_mean']} cross_max {c['cross_max']} | prod "
                      + " ".join(f"{k}={x['frame_accuracy']}(k{x['k_pred']},p{x['owner_purity']})" for k, x in r["production"].items()))
        print("wrote results.json[exp3]")
        return
    C.torch_threads()
    speaker_id._load_model()
    name, variants = argv[0], argv[1:]
    for v in variants:
        r = run(name, v)
        (C.CACHE / f"exp3_{name}_{v}.json").write_text(json.dumps(r, indent=1))
        print(name, v, json.dumps({"B": {k: r["B"][k] for k in ("k_pred", "frame_accuracy", "owner_purity")},
                                   "ceiling": {k: r["ceiling"][k] for k in ("centroid", "pooled", "within_mean", "cross_max") if k in r["ceiling"]},
                                   "production": {k: (x["frame_accuracy"], x["k_pred"], x["owner_purity"]) for k, x in r["production"].items()},
                                   "t": r["t_total_s"]}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
