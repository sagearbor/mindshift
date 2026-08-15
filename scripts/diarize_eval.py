#!/usr/bin/env python3
"""Score a diarization output against a ground-truth rubric.

The rubric is a JSON file describing who ACTUALLY spoke when (built from a
human listen-through or transcript addressing evidence), with per-segment
confidence weights so uncertain stretches count less::

    {
      "speakers": ["dad", "mom", "asher"],
      "confidence_weights": {"high": 1.0, "medium": 0.6, "low": 0.3},
      "segments": [{"start": s, "end": e, "speaker": "...",
                    "confidence": "high", "text": "..."}, ...]
    }

The produced side is the pipeline's turns.json shape:
``[{"speaker": "Speaker A", "start_time": s, "end_time": e, ...}, ...]``.

Metrics (rubric-time-weighted):
* speaker-count correctness,
* attribution accuracy under the best rubric->produced label mapping
  (brute-force over permutations/assignments - fine at family scale),
* per-true-speaker talk-share, rubric vs produced.

Usage: diarize_eval.py RUBRIC.json TURNS.json [--label NAME]

Rubrics live OUTSIDE the repo (they quote family conversations); this tool is
generic. See the 2026-08-15 calibration work for the pattern.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def produced_speaker_at(turns: list[dict], t0: float, t1: float) -> dict[str, float]:
    """Seconds of [t0,t1] attributed to each produced speaker label."""
    out: dict[str, float] = {}
    for turn in turns:
        ov = overlap(t0, t1, float(turn["start_time"]), float(turn["end_time"]))
        if ov > 0:
            out[turn["speaker"]] = out.get(turn["speaker"], 0.0) + ov
    return out


def score(rubric: dict, turns: list[dict]) -> dict:
    weights = rubric.get(
        "confidence_weights", {"high": 1.0, "medium": 0.6, "low": 0.3}
    )
    true_speakers = rubric["speakers"]
    produced_labels = sorted({t["speaker"] for t in turns})

    # Weighted overlap seconds for every (true, produced) pair. A segment's
    # "speaker" may be a LIST (genuine overlap / either-is-correct): its
    # seconds credit whichever listed speaker the mapping matches, once.
    pair: dict[tuple[str, str], float] = {}
    seg_speakers: list[list[str]] = []
    total_weighted = 0.0
    for seg in rubric["segments"]:
        w = weights.get(seg.get("confidence", "high"), 1.0)
        dur = float(seg["end"]) - float(seg["start"])
        total_weighted += w * dur
        accepted = (
            seg["speaker"] if isinstance(seg["speaker"], list) else [seg["speaker"]]
        )
        seg_speakers.append(accepted)
        for label, secs in produced_speaker_at(
            turns, float(seg["start"]), float(seg["end"])
        ).items():
            for true_s in accepted:
                key = (true_s, label)
                pair[key] = pair.get(key, 0.0) + w * secs / len(accepted)

    # Best injective mapping produced-label -> true-speaker (brute force).
    best_acc, best_map = 0.0, {}
    k = min(len(true_speakers), len(produced_labels))
    for true_subset in itertools.permutations(true_speakers, k):
        for prod_subset in itertools.permutations(produced_labels, k):
            mapping = dict(zip(prod_subset, true_subset))
            hit = sum(
                secs
                for (true_s, label), secs in pair.items()
                if mapping.get(label) == true_s
            )
            acc = hit / total_weighted if total_weighted else 0.0
            if acc > best_acc:
                best_acc, best_map = acc, mapping

    # Talk-share comparison under the best mapping.
    true_share: dict[str, float] = {s: 0.0 for s in true_speakers}
    for seg in rubric["segments"]:
        accepted = (
            seg["speaker"] if isinstance(seg["speaker"], list) else [seg["speaker"]]
        )
        dur = float(seg["end"]) - float(seg["start"])
        for s in accepted:
            true_share[s] += dur / len(accepted)
    true_total = sum(true_share.values()) or 1.0

    prod_share: dict[str, float] = {s: 0.0 for s in true_speakers}
    for t in turns:
        mapped = best_map.get(t["speaker"])
        if mapped:
            prod_share[mapped] += float(t["end_time"]) - float(t["start_time"])
    prod_total = sum(
        float(t["end_time"]) - float(t["start_time"]) for t in turns
    ) or 1.0

    return {
        "true_speaker_count": len(true_speakers),
        "produced_speaker_count": len(produced_labels),
        "count_correct": len(true_speakers) == len(produced_labels),
        "attribution_accuracy_weighted": round(best_acc, 3),
        "label_mapping": best_map,
        "talk_share_true": {
            s: round(v / true_total, 3) for s, v in true_share.items()
        },
        "talk_share_produced": {
            s: round(v / prod_total, 3) for s, v in prod_share.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rubric")
    ap.add_argument("turns")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    rubric = json.load(open(args.rubric))
    turns = json.load(open(args.turns))
    result = score(rubric, turns)
    if args.label:
        result["label"] = args.label
    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
