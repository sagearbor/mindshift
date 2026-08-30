"""Baseline: the production diarizer (server/diarize_local.py) under the shared
scorer — BOTH engines since 2026-08-30.

Input "utterances" = the GT intervals (the transcript boundaries production
normally gets, minus the welding problem) — an OPTIMISTIC baseline — plus the
REAL Deepgram transcripts (with word timings) production actually sees:
maggiano3's two variants (tmp/private_fixtures/maggiano3/transcript_*.json)
and the owner's cached poker / family transcripts
(tmp/private_fixtures/{poker_gcs,family_gcs}/transcript_run*.json — the GCS
audio is byte-for-byte the same recording as the checked-in WAV fixture, so
the transcripts are scored on the fixture's ground truth; identical runs are
collapsed to one).

Engines (``--engine utterances | windows | both``, default both):
  utterances  diarize_local.diarize_turns  — the transcript's utterances are
              embedded, clustered and validated (shipped until 2026-08-30)
  windows     diarize_local.diarize_windows_first — the transcript-free window
              engine labels the audio, the words are regrouped by its segments
              (production default since 2026-08-30). Its raw segment timeline
              is scored too (``segments_accuracy``) — the ceiling the regrouped
              turns can reach given the transcript's coverage.

Outputs: results.json (utterances, the shape the sibling READMEs cite),
results_windows.json, pred_<fixture>_<variant>[_windows].json, and a two-engine
table on stdout. Timing: torch at 4 threads (Cloud Run's 4 vCPU).
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(HERE.parent))
logging.basicConfig(level=logging.WARNING)
import audio_ingest  # noqa: E402
import diarize_local  # noqa: E402
import score  # noqa: E402

try:
    import torch
    torch.set_num_threads(4)
except Exception:  # noqa: BLE001
    pass

ENGINES = {
    "utterances": diarize_local.diarize_turns,
    "windows": diarize_local.diarize_windows_first,
}
REAL_TRANSCRIPTS = {"poker6": "poker_gcs", "family_real": "family_gcs"}

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=[*ENGINES, "both"], default="both")
ap.add_argument("fixtures", nargs="*", help="subset of fixtures (default: all)")
args = ap.parse_args()
engines = list(ENGINES) if args.engine == "both" else [args.engine]


def pcm_of(fx):
    p = fx["audio_path"]
    data = open(p, "rb").read()
    # Always 16 kHz — the TTS fixtures are natively 24 kHz and ECAPA refuses that.
    return audio_ingest.decode_to_pcm_16k(data, os.path.basename(p))


def variants_of(name, fx):
    out = {"gt_boundaries": [{"speaker": f"U{i}", "text": "…", "start_time": s, "end_time": e} for i, (s, e, _) in enumerate(fx["gt"])]}
    for t in fx.get("transcripts", []):
        out[Path(t).stem] = json.loads(Path(t).read_text())
    seen = set()
    for i, p in enumerate(sorted((score.PRIVATE / REAL_TRANSCRIPTS.get(name, "-")).glob("transcript_run*.json")), 1):
        raw = p.read_text()
        key = json.dumps(json.loads(raw), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out[f"deepgram_{p.stem.replace('transcript_', '')}"] = json.loads(raw)
    return out


results = {e: {} for e in engines}
rows = []
for name in (args.fixtures or score.all_fixtures()):
    fx = score.load_fixture(name)
    pcm, sr = pcm_of(fx)
    for vname, turns in variants_of(name, fx).items():
        row = {"fixture": name, "variant": vname, "k_true": fx["k_true"]}
        for engine in engines:
            t0 = time.time()
            out = ENGINES[engine](pcm, sr, [dict(t) for t in turns])
            dt = time.time() - t0
            if out is None:
                pred = [(t["start_time"], t["end_time"], "one") for t in turns]
            else:
                pred = [(t["start_time"], t["end_time"], t["speaker"]) for t in out["turns"]]
            r = score.score_fixture(name, pred)
            r["runtime_s"] = round(dt, 1)
            r["k_evaluated"] = out and out.get("k_evaluated")
            r["n_turns"] = len(pred)
            r["engine"] = engine
            r["source"] = out and out.get("source")
            r["uncovered_turns"] = out and out.get("uncovered_turns")
            r["split_utterances"] = out and out.get("split_utterances")
            if out is not None and out.get("segments"):
                seg = score.score_fixture(name, [(s["start"], s["end"], s["label"]) for s in out["segments"]])
                r["segments_accuracy"] = seg["frame_accuracy"]
                r["segments_k"] = seg["k_pred"]
                r["segments"] = out["segments"]
            results[engine].setdefault(name, {})[vname] = r
            suffix = "" if engine == "utterances" else f"_{engine}"
            json.dump([list(p) for p in pred], open(HERE / f"pred_{name}_{vname}{suffix}.json", "w"))
            row[engine] = r
            print(f"{engine:10s} {name:14s} {vname:18s} k={r['k_pred']}/{r['k_true']} acc={r['frame_accuracy']:.3f}"
                  + (f" seg_acc={r['segments_accuracy']:.3f} (k={r['segments_k']})" if "segments_accuracy" in r else "")
                  + f" owner_purity={r['owner_purity']} unlabelled={r['unlabelled_frac']:.3f} turns={r['n_turns']} {dt:.1f}s",
                  flush=True)
        rows.append(row)

for engine in engines:
    fname = "results.json" if engine == "utterances" else f"results_{engine}.json"
    json.dump(results[engine], open(HERE / fname, "w"), indent=1, default=str)

print("\n| fixture | variant | " + " | ".join(f"{e}: acc (k/k_true) purity, s" for e in engines)
      + (" | windows segments acc (k) |" if "windows" in engines else " |"))
print("|---|---|" + "---|" * (len(engines) + (1 if "windows" in engines else 0)))
for row in rows:
    cells = []
    for e in engines:
        r = row[e]
        cells.append(f"{r['frame_accuracy']:.3f} ({r['k_pred']}/{row['k_true']}) {r['owner_purity']}, {r['runtime_s']}s")
    if "windows" in engines:
        r = row["windows"]
        cells.append(f"{r.get('segments_accuracy', float('nan')):.3f} ({r.get('segments_k', '-')})")
    print(f"| {row['fixture']} | {row['variant']} | " + " | ".join(cells) + " |")
