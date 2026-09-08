"""Score the phone-port replay (replay_b.ts) with the shared scorer and put
it next to B's Python numbers (../B-sliding-window/results.json, the
``w1.5_h0.25/spec_eigengap_p0.80`` smoothed stage) -> results.json + a table.

Run: tmp/venv-voice/bin/python docs/research/2026-08-29-voice-separation/F-device-b/score_f.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAKEOFF = HERE.parent
sys.path.insert(0, str(BAKEOFF))
import score  # noqa: E402

VARIANTS = ["b8", "prod6"]


def main() -> None:
    b = json.loads((BAKEOFF / "B-sliding-window" / "results.json").read_text())
    summary = json.loads((HERE / "replay_summary.json").read_text())
    out = {"scorer": "../score.py", "variants": {
        "b8": "phone port, eigengap over 1..8, no k floor (B's k policy)",
        "prod6": "phone port, same embeddings, production's k clamp (eigengap over 1..6, floor 2)",
    }, "fixtures": {}}
    rows = []
    for name in score.all_fixtures():
        fs = summary["fixtures"].get(name)
        if not fs:
            continue
        bv = b["fixtures"].get(name, {}).get("variants", {}).get("w1.5_h0.25/spec_eigengap_p0.80", {}).get("smoothed", {})
        entry = {"k_true": fs["k_true"], "python_B": {k: bv.get(k) for k in ("k_pred", "frame_accuracy", "owner_purity")},
                 "windows": f"{fs['windows_kept']}/{fs['windows_total']}", "hop_s": fs["hop_s"], "gate_rms": fs["gate_rms"],
                 "embed_ms_per_window": fs["embed_ms_per_window"], "timings_ms": fs["timings_ms"]}
        for v in VARIANTS:
            pred = json.loads((HERE / fs["variants"][v]["pred_file"]).read_text())
            sc = score.score_fixture(name, [tuple(x) for x in pred])
            entry[v] = {k: sc[k] for k in ("k_pred", "frame_accuracy", "owner_purity", "per_gt_recall")}
            entry[v]["k_eigengap"] = fs["variants"][v]["k_eigengap"]
        out["fixtures"][name] = entry
        d = entry["b8"]["frame_accuracy"] - (bv.get("frame_accuracy") or 0)
        rows.append(
            f"| {name} ({fs['k_true']}) | {entry['b8']['frame_accuracy']:.3f} ({entry['b8']['k_pred']}) "
            f"[{entry['b8']['owner_purity']}] | {entry['prod6']['frame_accuracy']:.3f} ({entry['prod6']['k_pred']}) | "
            f"{bv.get('frame_accuracy')} ({bv.get('k_pred')}) [{bv.get('owner_purity')}] | {d:+.3f} | "
            f"{fs['timings_ms']['wall'] / 1000:.1f} | {fs['embed_ms_per_window']['mean_ms']} / {fs['embed_ms_per_window']['p90_ms']} | "
            f"{fs['windows_kept']}/{fs['windows_total']} |"
        )
    (HERE / "results.json").write_text(json.dumps(out, indent=1) + "\n")
    print("| fixture (k) | phone port b8: acc (k) [owner purity] | prod6 k-clamp: acc (k) | Python B: acc (k) [purity] | Δ vs B | wall s (Mac) | embed ms/window mean / p90 | windows |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    accs = [e["b8"]["frame_accuracy"] for e in out["fixtures"].values()]
    print(f"\nmean b8 frame accuracy over {len(accs)}: {sum(accs) / len(accs):.3f}")


if __name__ == "__main__":
    main()
