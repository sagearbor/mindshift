"""The metric that actually matters: assign the REFERENCE's own utterance
boundaries (what a real transcript would give diarize_pyannote.diarize_turns)
to their majority-overlap pyannote speaker — mirrors production exactly,
unlike eval_pyannote.py's per-pyannote-turn scoring (which penalizes every
small pyannote sub-segment individually, even ones a real transcript would
never surface as a separate unit).
"""
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import best_permutation_accuracy  # noqa: E402
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


def majority_overlap(seg_start, seg_end, pyannote_turns):
    best_speaker, best_overlap = None, 0.0
    for t, _, s in pyannote_turns:
        overlap = max(0.0, min(seg_end, t.end) - max(seg_start, t.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = s
    return best_speaker


def main():
    mcs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    fixture_name = sys.argv[2] if len(sys.argv) > 2 else "poker6"
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

    use_hint = "--hint" in sys.argv
    use_bounded = "--bounded" in sys.argv
    if use_hint:
        kwargs = {"num_speakers": num_speakers_true}
    elif use_bounded:
        kwargs = {"min_speakers": 2, "max_speakers": 6}
    else:
        kwargs = {}
    t0 = time.time()
    diarization = pipeline(str(wav_path), **kwargs)
    elapsed = time.time() - t0
    pyannote_turns = list(diarization.itertracks(yield_label=True))
    num_pred = len({s for _, _, s in pyannote_turns})

    # Assign EACH REFERENCE TURN (what a real transcript utterance would be)
    # to its majority-overlap pyannote speaker — production's actual unit.
    truth = [t["speaker"] for t in ref_turns]
    pred = [
        majority_overlap(
            t.get("start_time", t.get("approx_start")),
            t.get("end_time", t.get("approx_end")),
            pyannote_turns,
        )
        for t in ref_turns
    ]
    acc, correct, mapping = best_permutation_accuracy(truth, pred)

    result = {
        "min_cluster_size": mcs,
        "fixture": name,
        "num_true": num_speakers_true,
        "num_pred_pyannote_clusters": num_pred,
        "production_per_utterance_accuracy": round(acc, 3),
        "correct": correct,
        "total": len(truth),
        "elapsed": round(elapsed, 1),
        "truth": truth,
        "pred": pred,
    }
    print("PRODRESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
