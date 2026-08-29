"""Baseline: today's production diarizer (server/diarize_local.py) under the shared scorer.

Input "utterances" = the GT intervals (the transcript boundaries production
normally gets, minus the welding problem) — an OPTIMISTIC baseline — plus, for
maggiano3, the two real Deepgram transcripts (with word timings) production
actually sees.
"""
import json, logging, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "server")); sys.path.insert(0, str(HERE.parent))
logging.basicConfig(level=logging.WARNING)
import numpy as np, audio_ingest, diarize_local, score

def pcm_of(fx):
    p = fx["audio_path"]; data = open(p, "rb").read()
    # Always 16 kHz — the TTS fixtures are natively 24 kHz and ECAPA refuses that.
    return audio_ingest.decode_to_pcm_16k(data, os.path.basename(p))

results = {}
for name in score.all_fixtures():
    fx = score.load_fixture(name); pcm, sr = pcm_of(fx)
    variants = {"gt_boundaries": [{"speaker": f"U{i}", "text": "…", "start_time": s, "end_time": e} for i, (s, e, _) in enumerate(fx["gt"])]}
    for t in fx.get("transcripts", []):
        variants[Path(t).stem] = json.loads(Path(t).read_text())
    results[name] = {}
    for vname, turns in variants.items():
        t0 = time.time(); out = diarize_local.diarize_turns(pcm, sr, [dict(t) for t in turns]); dt = time.time() - t0
        if out is None:
            pred = [(t["start_time"], t["end_time"], "one") for t in turns]
        else:
            pred = [(t["start_time"], t["end_time"], t["speaker"]) for t in out["turns"]]
        r = score.score_fixture(name, pred); r["runtime_s"] = round(dt, 1); r["k_evaluated"] = out and out.get("k_evaluated")
        results[name][vname] = r
        print(f"{name:14s} {vname:18s} k={r['k_pred']}/{r['k_true']} acc={r['frame_accuracy']:.3f} owner_purity={r['owner_purity']} {dt:.1f}s")
        json.dump([list(p) for p in pred], open(HERE / f"pred_{name}_{vname}.json", "w"))
json.dump(results, open(HERE / "results.json", "w"), indent=1, default=str)
