"""Shared scorer for the 2026-08-29 voice-separation bake-off.

Every approach is scored EXACTLY the same way so the results are
comparable:

* ``load_fixture(name)`` → ``{"audio_path", "gt", "k_true", "owner_label"}``
  where ``gt`` is a list of ``(start_s, end_s, label)`` ground-truth speech
  intervals (only the audio inside these intervals is scored — gaps and
  unlabelled audio are ignored).
* ``score_fixture(name, pred)`` where ``pred`` is a list of
  ``(start_s, end_s, label)`` predicted speaker intervals (any labels; they
  may overlap the GT boundaries however they like — scoring is frame-level).
  Returns::

      {
        "fixture": ..., "k_true": 6, "k_pred": 5,
        "frame_accuracy": 0.93,        # best one-to-one label mapping,
                                       # 10 ms frames inside GT speech
        "unlabelled_frac": 0.02,       # GT speech frames with NO prediction
        "mapping": {"pred_label": "gt_label"},
        "per_gt_recall": {"Player1": 0.98, ...},
        "owner_purity": 0.97 | None,   # of frames the mapped owner cluster
                                       # claims, the fraction that are the
                                       # owner (None when no owner in GT)
      }

Fixtures (all 16 kHz mono; decode with ``audio_ingest.decode_to_pcm_16k`` or
``scipy.io.wavfile`` — the .wav ones are already 16 kHz):

  family_real   owner + son, 8 GT turns, exact
  poker6        6 men, 6 approx 5 s turns (±1-2 s slop — expect ~0.9 ceiling)
  openai        2 TTS voices, exact
  gptaudio      2 TTS voices, exact
  scene_couple / scene_family3 / scene_meeting4   2/3/4 TTS voices, exact
  maggiano3     PRIVATE (tmp/private_fixtures/maggiano3/audio.m4a + rubric.json,
                the OWNER'S per-second listen-through: dad / mom / asher, overlap
                segments credit either speaker) — skipped when absent

CLI: ``python score.py <fixture> <pred.json>`` where pred.json is
``[[start, end, "label"], ...]``.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIX = ROOT / "server" / "tests" / "fixtures" / "audio"
PRIVATE = Path(os.getenv("MINDSHIFT_PRIVATE_FIXTURES") or (ROOT / "tmp" / "private_fixtures"))
FRAME = 0.01  # seconds

Seg = tuple[float, float, str]


def _meta(stem: str) -> dict:
    return json.loads((FIX / f"{stem}_meta.json").read_text())


def _scene_turns(meta: dict) -> list[Seg]:
    gap = meta["silence_gap_sec"]
    t, out = 0.0, []
    for m in meta["turns"]:
        out.append((round(t, 4), round(t + m["duration_sec"], 4), m["speaker"]))
        t += m["duration_sec"] + gap
    return out


def _maggiano_gt() -> list:
    """The OWNER'S per-second listen-through rubric (tmp/private_fixtures/
    maggiano3/rubric.json, v2 2026-08-15, ±1 s boundary slop). Overlap
    segments carry a LIST of speakers — either is credited. Labels are the
    rubric's own: dad / mom / asher."""
    rub = json.loads((PRIVATE / "maggiano3" / "rubric.json").read_text())
    out = []
    for seg in rub["segments"]:
        sp = seg["speaker"]
        out.append((float(seg["start"]), float(seg["end"]), tuple(sp) if isinstance(sp, list) else sp))
    return out


FIXTURES = {
    "family_real": ("test_recording_family_real", "Sage"),
    "poker6": ("test_recording_poker6_real", "Player6"),
    "openai": ("test_recording_openai", None),
    "gptaudio": ("test_recording_gptaudio", None),
    "scene_couple": ("test_recording_scene_couple_escalation", "Speaker A"),
    "scene_family3": ("test_recording_scene_family3", "Speaker A"),
    "scene_meeting4": ("test_recording_scene_meeting4", "Speaker A"),
    "maggiano3": (None, "dad"),
}


def load_fixture(name: str) -> dict:
    stem, owner = FIXTURES[name]
    if name == "maggiano3":
        audio = PRIVATE / "maggiano3" / "audio.m4a"
        if not audio.exists():
            raise FileNotFoundError(audio)
        return {"audio_path": str(audio), "gt": _maggiano_gt(), "k_true": 3,
                "owner_label": owner,
                "transcripts": sorted(str(p) for p in (PRIVATE / "maggiano3").glob("transcript_*.json"))}
    meta = _meta(stem)
    if name == "poker6":
        gt = [(t["approx_start"], t["approx_end"], t["speaker"]) for t in meta["approx_turns"]]
    elif name == "family_real":
        gt = [(t["start_time"], t["end_time"], t["speaker"]) for t in meta["turns"]]
    else:
        gt = _scene_turns(meta)
    return {"audio_path": str(FIX / f"{stem}.wav"), "gt": gt,
            "k_true": len({g[2] for g in gt}), "owner_label": owner}


def all_fixtures() -> list[str]:
    names = [n for n in FIXTURES if n != "maggiano3"]
    if (PRIVATE / "maggiano3" / "audio.m4a").exists() and (PRIVATE / "maggiano3" / "rubric.json").exists():
        names.append("maggiano3")
    return names


def _frames(segs: list, n: int, multi: bool = False) -> list:
    lab: list = [None] * n
    for s, e, l in segs:
        allowed = frozenset(l) if isinstance(l, (tuple, list, set, frozenset)) else frozenset([l])
        for i in range(max(0, int(s / FRAME)), min(n, int(e / FRAME))):
            lab[i] = (allowed if multi else next(iter(allowed)))
    return lab


def score_segments(gt: list, pred: list[Seg], owner_label: str | None = None) -> dict:
    """Frame-level (10 ms) accuracy under the best one-to-one pred→gt label
    mapping, over GT speech frames only. A GT frame may allow SEVERAL labels
    (overlap) — a prediction mapping to any of them is correct; for the
    per-speaker/purity tallies an overlap frame is credited to the allowed
    label the prediction maps to (else to its first listed speaker)."""
    end = max([e for _, e, _ in gt] + [e for _, e, _ in pred] + [0.0])
    n = int(end / FRAME) + 1
    g, p = _frames(gt, n, multi=True), _frames(pred, n)
    idx = [i for i in range(n) if g[i] is not None]
    gt_labels = sorted({l for i in idx for l in g[i]})
    pred_labels = sorted({p[i] for i in idx if p[i] is not None})
    unl = sum(1 for i in idx if p[i] is None)

    def score_of(m: dict) -> int:
        return sum(1 for i in idx if p[i] is not None and m.get(p[i]) in g[i])

    best, best_map = -1, {}
    small_is_pred = len(pred_labels) <= len(gt_labels)
    small, large = (pred_labels, gt_labels) if small_is_pred else (gt_labels, pred_labels)
    for perm in itertools.permutations(large, len(small)):
        m = dict(zip(small, perm)) if small_is_pred else {pl: gl for gl, pl in zip(small, perm)}
        sc = score_of(m)
        if sc > best:
            best, best_map = sc, m
    total = len(idx)
    # credited GT label per frame under the best mapping
    credited = []
    for i in idx:
        mapped = best_map.get(p[i]) if p[i] is not None else None
        credited.append(mapped if mapped in g[i] else sorted(g[i])[0])
    per_gt = {}
    for gl in gt_labels:
        tot = sum(1 for c in credited if c == gl)
        hit = sum(1 for i, c in zip(idx, credited) if c == gl and p[i] is not None and best_map.get(p[i]) == gl)
        per_gt[gl] = round(hit / tot, 3) if tot else 0.0
    owner_purity = None
    if owner_label in gt_labels:
        pl = next((k for k, v in best_map.items() if v == owner_label), None)
        if pl is not None:
            claimed = [i for i in idx if p[i] == pl]
            owner_purity = round(sum(1 for i in claimed if owner_label in g[i]) / len(claimed), 3) if claimed else None
    return {
        "k_true": len(gt_labels), "k_pred": len(pred_labels),
        "frame_accuracy": round(best / total, 3) if total else 0.0,
        "unlabelled_frac": round(unl / total, 3) if total else 0.0,
        "mapping": best_map, "per_gt_recall": per_gt, "owner_purity": owner_purity,
    }


def score_fixture(name: str, pred: list[Seg]) -> dict:
    fx = load_fixture(name)
    out = score_segments(fx["gt"], [(float(s), float(e), str(l)) for s, e, l in pred], fx["owner_label"])
    out["fixture"] = name
    return out


if __name__ == "__main__":
    name, pred_path = sys.argv[1], sys.argv[2]
    pred = [tuple(x) for x in json.loads(Path(pred_path).read_text())]
    print(json.dumps(score_fixture(name, pred), indent=1))
