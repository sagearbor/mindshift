"""Test ONE min_cluster_size value against ONE fixture — run repeatedly with
a shell timeout so a pathological config can't hang the whole sweep.
Usage: tune_one.py <min_cluster_size> <fixture_name>
"""
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import score_against_reference  # noqa: E402
from eval_pyannote import load_ref_turns, FIXTURES, FIXTURES_DIR  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import torch  # noqa: E402
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from pyannote.audio import Pipeline  # noqa: E402


def main():
    mcs = int(sys.argv[1])
    fixture_name = sys.argv[2]
    fx = next(f for f in FIXTURES if f[0] == fixture_name)
    name, wav_name, meta_name, turns_key = fx

    hf_token = os.environ["HF_TOKEN"]
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    params = pipeline.parameters(instantiated=True)
    params["clustering"]["min_cluster_size"] = mcs
    pipeline.instantiate(params)

    wav_path = FIXTURES_DIR / wav_name
    meta_path = FIXTURES_DIR / meta_name
    ref_turns, num_speakers_true = load_ref_turns(meta_path, turns_key)

    t0 = time.time()
    diarization = pipeline(str(wav_path), num_speakers=num_speakers_true)
    elapsed = time.time() - t0
    pred_turns = [
        {"start_time": t.start, "end_time": t.end, "speaker": s}
        for t, _, s in diarization.itertracks(yield_label=True)
    ]
    num_pred = len({t["speaker"] for t in pred_turns})
    score = score_against_reference(pred_turns, ref_turns, speaker_key="speaker")
    result = {
        "min_cluster_size": mcs,
        "fixture": name,
        "num_true": num_speakers_true,
        "num_pred": num_pred,
        "acc": round(score["per_turn_accuracy"], 3),
        "weighted_acc": round(score["duration_weighted_accuracy"], 3),
        "elapsed": round(elapsed, 1),
    }
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
