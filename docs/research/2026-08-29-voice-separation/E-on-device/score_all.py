"""Step 4 of E: score every pred_*.json with the bake-off scorer and assemble
results.json (this experiment's numbers + the bake-off rows copied from the
sibling folders' results.json so the README table is one source).

Run: tmp/venv-voice/bin/python docs/research/2026-08-29-voice-separation/E-on-device/score_all.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAKEOFF = HERE.parent
sys.path.insert(0, str(BAKEOFF))
import score  # noqa: E402

FIXTURES = ["family_real", "poker6", "openai", "gptaudio", "scene_couple", "scene_family3", "scene_meeting4", "maggiano3"]
VARIANTS = ["live@0.48", "live@0.55", "gt@0.48", "gt@0.55", "energy@0.48", "energy@0.55"]


FILLED = ["live_filled@0.48", "live_filled@0.55"]


def fx_end(name: str) -> float:
    return max(e for _, e, _ in score.load_fixture(name)["gt"])


def pick(d: dict) -> dict:
    return {k: d.get(k) for k in ("k_true", "k_pred", "frame_accuracy", "owner_purity", "unlabelled_frac")}


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def bakeoff_rows() -> dict:
    base = load(BAKEOFF / "baseline/results.json")
    a = load(BAKEOFF / "A-acoustic/results.json")
    b = load(BAKEOFF / "B-sliding-window/results.json")
    c = load(BAKEOFF / "C-pyannote/results.json")
    rows: dict = {}
    for f in FIXTURES:
        r: dict = {}
        if f in base:
            r["production_gt_boundaries"] = pick(base[f]["gt_boundaries"])
            for k in ("transcript_7utt", "transcript_8utt"):
                if k in base[f]:
                    r[f"production_{k}"] = pick(base[f][k])
        if f in a:
            r["A_auto_k"] = pick(a[f]["variants"]["full_auto"])
            r["A_k_given"] = pick(a[f]["variants"]["full_oracle"])
        bv = b.get("fixtures", {}).get(f, {}).get("variants", {}).get("w1.5_h0.25/spec_eigengap_p0.80", {})
        for stage in ("merged", "refined", "smoothed"):
            if stage in bv:
                r["B_w1.5_h0.25_spec_p0.80"] = {**pick(bv[stage]), "stage": stage}
                break
        cf = _find_c(c, f)
        if cf:
            r["C_pyannote_default"] = pick(cf)
        rows[f] = r
    return rows


def _find_c(c: dict, fixture: str) -> dict | None:
    """C-pyannote/results.json: take the fixture's default-pipeline entry
    wherever it sits (the folder's own layout)."""
    stack = [(c, "")]
    hits = []
    while stack:
        o, p = stack.pop()
        if isinstance(o, dict):
            if "frame_accuracy" in o and fixture in p:
                hits.append((p, o))
            else:
                for k, v in o.items():
                    stack.append((v, f"{p}/{k}"))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                stack.append((v, f"{p}[{i}]"))
    for p, o in hits:
        if "default" in p or p.rstrip("/").endswith(fixture):
            return o
    return hits[0][1] if hits else None


def main() -> None:
    summary = load(HERE / "replay_summary.json")
    bench = load(HERE / "bench_ecapa.json")
    results: dict = {
        "experiment": "E-on-device: the phone's live SpeakerLabeler replayed on the bake-off fixtures + on-device ECAPA cost",
        "scorer": "docs/research/2026-08-29-voice-separation/score.py (10 ms frames, best 1:1 mapping, GT speech only)",
        "labeler": "apps/mobile/src/live/speakerId.ts SpeakerLabeler, no enrolled people; 'Unknown' turns (short-segment guard) left unlabelled",
        "variants": {
            "live": "real loop via apps/mobile/src/live/replay/sceneReplay.ts: 100 ms frames -> Silero VAD -> StreamingSegmenter (gap 0.3 s, min 0.6 s) -> ECAPA ONNX on the last <=10 s -> SpeakerLabeler",
            "gt": "ground-truth utterance boundaries fed as segments (perfect segmenter), same embed + labeler",
            "energy": "phone's fallback EnergyVad segmentation (energySpeechSegments: -45 dBFS, 0.25 s frames, same merge/min), same embed + labeler",
            "@0.48": "CLUSTER_THRESHOLD as shipped (2026-08-26 tuning)",
            "@0.55": "the pre-tuning value (server/batch still uses it)",
        },
        "fixtures": {},
        "bakeoff": bakeoff_rows(),
        "latency": bench,
    }
    means: dict = {v: [] for v in VARIANTS}
    for f in FIXTURES:
        s = summary[f]
        fr: dict = {"seconds": s["seconds"], "k_true": s["k_true"], "gt_segments": s["gt_segments"],
                    "live_turns": s["live"]["turns"], "energy_segments": s["energy_segments"],
                    "live_turns_per_minute": s["live"]["turns_per_minute"],
                    "live_embedded_seconds_per_minute": s["live"]["embedded_seconds_per_minute"],
                    "scores": {}}
        for v in VARIANTS:
            pred_file = HERE / s["preds"][v]
            pred = [tuple(x) for x in load(pred_file)]
            sc = score.score_fixture(f, pred)
            sc["pred_file"] = pred_file.name
            sc["n_segments"] = s["info"][v]["n_segments"]
            sc["n_unknown"] = s["info"][v]["n_unknown"]
            sc["accuracy_on_labelled"] = round(sc["frame_accuracy"] / (1 - sc["unlabelled_frac"]), 3) if sc["unlabelled_frac"] < 1 else None
            fr["scores"][v] = sc
            means[v].append(sc["frame_accuracy"])
        # Coverage-corrected: the loop only labels VAD-trimmed turns, so the
        # GT frames in the gaps (15-37 %) count as wrong. A session transcript
        # (production) labels contiguous utterances; this fills each gap by
        # nearest labelled turn (split at the midpoint) to separate the
        # segmenter's coverage loss from the clustering.
        for v in ("live@0.48", "live@0.55"):
            pred = sorted(tuple(x) for x in load(HERE / s["preds"][v]))
            filled = []
            for i, (st, en, lab) in enumerate(pred):
                a = 0.0 if i == 0 else (pred[i - 1][1] + st) / 2
                b_ = fx_end(f) if i == len(pred) - 1 else (en + pred[i + 1][0]) / 2
                filled.append((round(a, 3), round(b_, 3), lab))
            key = v.replace("live", "live_filled")
            fn = HERE / s["preds"][v].replace("_live_", "_livefilled_")
            fn.write_text(json.dumps(filled) + "\n")
            sc = score.score_fixture(f, filled)
            sc["pred_file"] = fn.name
            fr["scores"][key] = sc
            means.setdefault(key, []).append(sc["frame_accuracy"])
        results["fixtures"][f] = fr
    results["mean_frame_accuracy"] = {v: round(sum(x) / len(x), 3) for v, x in means.items()}
    results["mean_accuracy_on_labelled"] = {
        v: round(sum(results["fixtures"][f]["scores"][v]["accuracy_on_labelled"] for f in FIXTURES) / len(FIXTURES), 3) for v in VARIANTS
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=1) + "\n")

    # console table
    cols = VARIANTS + FILLED
    hdr = f"{'fixture':15s} " + " ".join(f"{v:>16s}" for v in cols) + f" {'prod-GT':>10s} {'B':>10s}"
    print(hdr)
    for f in FIXTURES:
        fr = results["fixtures"][f]
        cells = []
        for v in cols:
            sc = fr["scores"][v]
            op = "-" if sc["owner_purity"] is None else f"{sc['owner_purity']:.2f}"
            cells.append(f"{sc['frame_accuracy']:.2f} k{sc['k_pred']} p{op}".rjust(16))
        bo = results["bakeoff"][f]
        pg = bo.get("production_gt_boundaries", {})
        bb = bo.get("B_w1.5_h0.25_spec_p0.80", {})
        print(f"{f:15s} " + " ".join(cells) + f" {pg.get('frame_accuracy', '-')!s:>10s} {bb.get('frame_accuracy', '-')!s:>10s}")
    print(f"{'mean':15s} " + " ".join(f"{results['mean_frame_accuracy'][v]:.3f}".rjust(16) for v in cols))
    print(f"{'mean/labelled':15s} " + " ".join(f"{results['mean_accuracy_on_labelled'][v]:.3f}".rjust(16) for v in VARIANTS))
    print("wrote", HERE / "results.json")


if __name__ == "__main__":
    main()
