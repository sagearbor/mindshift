"""Production diarizer under the shared scorer, with an Unknown column.

Same inputs as ../2026-08-29-voice-separation/baseline/run.py (GT boundaries
for every fixture + the real Deepgram transcripts for maggiano3), but:

* turns the diarizer labels ``diarize_local.UNKNOWN_SPEAKER`` are REMOVED
  from the prediction before scoring — the scorer then sees those frames as
  UNLABELLED, i.e. WRONG (``unlabelled_frac`` rises, ``frame_accuracy``
  falls). Unknown never earns credit; it can only cost accuracy. The seconds
  it claimed are reported separately (``unknown_s``).
* results + predictions go under ``out/<tag>/`` (never over the 08-29
  baseline's checked-in predictions).

Usage: ``MINDSHIFT_DIARIZE_UNKNOWN=0|1 python run_measure.py <tag> [fixture ...]``
"""
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(HERE.parent / "2026-08-29-voice-separation"))
logging.basicConfig(level=logging.WARNING)
import audio_ingest  # noqa: E402
import diarize_local  # noqa: E402
import score  # noqa: E402


def pcm_of(fx):
    p = fx["audio_path"]
    data = open(p, "rb").read()
    return audio_ingest.decode_to_pcm_16k(data, os.path.basename(p))


class Hang(Exception):
    pass


def _alarm(*_):
    raise Hang("diarize_turns exceeded the watchdog")


def measure(name: str, turns: list[dict]) -> tuple[dict, list]:
    fx = score.load_fixture(name)
    pcm, sr = pcm_of(fx)
    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(180)
    try:
        out = diarize_local.diarize_turns(pcm, sr, [dict(t) for t in turns])
    finally:
        signal.alarm(0)
    dt = time.time() - t0
    if out is None:
        pred = [(t["start_time"], t["end_time"], "one") for t in turns]
        unknown_s = 0.0
        unknown_turns = []
    else:
        unknown_turns = [
            t for t in out["turns"] if t["speaker"] == diarize_local.UNKNOWN_SPEAKER
        ]
        unknown_s = sum(t["end_time"] - t["start_time"] for t in unknown_turns)
        pred = [
            (t["start_time"], t["end_time"], t["speaker"]) for t in out["turns"]
            if t["speaker"] != diarize_local.UNKNOWN_SPEAKER
        ]
    r = score.score_fixture(name, pred)
    r["runtime_s"] = round(dt, 1)
    r["k_evaluated"] = out and out.get("k_evaluated")
    r["unknown_s"] = round(unknown_s, 2)
    r["unknown_turns"] = [
        (round(t["start_time"], 2), round(t["end_time"], 2), t["text"][:40])
        for t in unknown_turns
    ]
    r["num_speakers"] = out and out.get("num_speakers")
    return r, pred


def main() -> None:
    tag = sys.argv[1]
    only = set(sys.argv[2:])
    outdir = HERE / "out" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in score.all_fixtures():
        if only and name not in only:
            continue
        fx = score.load_fixture(name)
        variants = {
            "gt_boundaries": [
                {"speaker": f"U{i}", "text": "…", "start_time": s, "end_time": e}
                for i, (s, e, _) in enumerate(fx["gt"])
            ]
        }
        for t in fx.get("transcripts", []):
            variants[Path(t).stem] = json.loads(Path(t).read_text())
        results[name] = {}
        for vname, turns in variants.items():
            r, pred = measure(name, turns)
            results[name][vname] = r
            print(
                f"{name:14s} {vname:18s} k={r['k_pred']}/{r['k_true']} "
                f"acc={r['frame_accuracy']:.3f} unl={r['unlabelled_frac']:.3f} "
                f"owner_purity={r['owner_purity']} unknown_s={r['unknown_s']} "
                f"{r['unknown_turns']} {r['runtime_s']}s",
                flush=True,
            )
            json.dump(
                [list(p) for p in pred], open(outdir / f"pred_{name}_{vname}.json", "w"),
            )
    json.dump(results, open(outdir / "results.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
