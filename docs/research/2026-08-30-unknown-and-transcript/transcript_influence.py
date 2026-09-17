"""Task 2a — does the transcript's own speaker labelling push us wrong?

For each of the owner's three real recordings (GCS copies under
tmp/private_fixtures/<name>_gcs/, byte-identical / duration-identical to the
checked-in fixtures and the private maggiano3 clip) and each of three fresh
Deepgram transcriptions (transcript_run<i>.json, cached next to the audio):

* Deepgram's speaker count and the accuracy of ITS OWN labels vs ground truth
  (the shared scorer, ../2026-08-29-voice-separation/score.py);
* our diarize_turns output on that transcript (count + accuracy);
* what main.py's cross-check block would do with MINDSHIFT_DIARIZE_CROSSCHECK=1
  (the deployed config): transcript < 2 speakers -> local wins when it hears
  2+; else NEVER-REDUCE guard keeps the transcript when local k < Deepgram's
  count; else local wins when it changes anything.
"""
import json
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(HERE.parent / "2026-08-29-voice-separation"))
logging.basicConfig(level=logging.WARNING)
import audio_ingest  # noqa: E402
import diarize_local  # noqa: E402
import score  # noqa: E402

PRIV = ROOT / "tmp" / "private_fixtures"
RECS = {"maggiano_gcs": "maggiano3", "poker_gcs": "poker6", "family_gcs": "family_real"}
rows = []
for gcs, fixture in RECS.items():
    fx = score.load_fixture(fixture)
    gt, owner = fx["gt"], fx["owner_label"]
    data = (PRIV / gcs / "audio.m4a").read_bytes()
    pcm, sr = audio_ingest.decode_to_pcm_16k(data, "audio.m4a")
    for run in (1, 2, 3):
        turns = json.loads((PRIV / gcs / f"transcript_run{run}.json").read_text())
        dg_pred = [(t["start_time"], t["end_time"], t["speaker"]) for t in turns]
        dg = score.score_segments(gt, dg_pred, owner)
        n_dg = len({t["speaker"] for t in turns})
        local = diarize_local.diarize_turns(pcm, sr, [dict(t) for t in turns])
        if local is None:
            ours = None
            k_local = None
        else:
            ours = score.score_segments(
                gt, [(t["start_time"], t["end_time"], t["speaker"]) for t in local["turns"]], owner,
            )
            k_local = local["num_speakers"]
        # main.py's cross-check block, MINDSHIFT_DIARIZE_CROSSCHECK=1
        if local is None:
            winner = "transcript (local None)"
        elif n_dg >= 2 and k_local < n_dg:
            winner = "transcript (NEVER-REDUCE guard)"
        elif k_local >= 2 and (n_dg < 2 or len(local["turns"]) != len(turns) or local["agreement_with_input"] < 1.0):
            winner = "local"
        else:
            winner = "transcript (local agreed)"
        final = ours if winner == "local" else dg
        row = dict(
            recording=fixture, run=run, utterances=len(turns), dg_k=n_dg, dg_acc=dg["frame_accuracy"],
            dg_owner_purity=dg["owner_purity"], local_k=k_local,
            local_acc=ours and ours["frame_accuracy"], local_owner_purity=ours and ours["owner_purity"],
            winner=winner, final_acc=final["frame_accuracy"],
            guard_hurts=(winner.startswith("transcript (NEVER") and ours is not None
                         and ours["frame_accuracy"] > dg["frame_accuracy"]),
            k_evaluated=(local or {}).get("k_evaluated"),
        )
        rows.append(row)
        print(f"{fixture:12s} run{run} utt={len(turns):2d} DG k={n_dg} acc={dg['frame_accuracy']:.3f} "
              f"| ours k={k_local} acc={row['local_acc']} | winner={winner} final={row['final_acc']} "
              f"guard_hurts={row['guard_hurts']}", flush=True)
json.dump(rows, open(HERE / "out" / "transcript_influence.json", "w"), indent=1, default=str)
