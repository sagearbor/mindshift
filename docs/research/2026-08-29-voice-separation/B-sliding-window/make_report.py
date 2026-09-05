"""Turn results.json into the README tables + per-fixture pred_<fixture>.json.

Usage: python make_report.py            (prints markdown to stdout, writes preds)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import score  # noqa: E402

R = json.loads((HERE / "results.json").read_text()) if (HERE / "results.json").exists() else {"fixtures": {}}
for f in sorted(HERE.glob("results_*.json")):        # per-fixture files win
    part = json.loads(f.read_text())
    R["fixtures"][part["fixture"]["fixture"]] = part["fixture"]
    R["model_load_s"] = part.get("model_load_s", R.get("model_load_s"))
    R["torch_threads"] = part.get("torch_threads", R.get("torch_threads"))
(HERE / "results.json").write_text(json.dumps(R, indent=1))
FX = [n for n in ["family_real", "poker6", "maggiano3", "openai", "gptaudio",
                  "scene_couple", "scene_family3", "scene_meeting4"] if n in R["fixtures"]]
G = "w1.5_h0.25"
COLS = [  # (label, variant key, stage)
    ("agg t=0.80", f"{G}/agg_t0.80", "smoothed"),
    ("agg t=0.85", f"{G}/agg_t0.85", "smoothed"),
    ("agg t=0.90", f"{G}/agg_t0.90", "smoothed"),
    ("spec eigengap p=.95", f"{G}/spec_eigengap", "smoothed"),
    ("spec eigengap p=.80", f"{G}/spec_eigengap_p0.80", "smoothed"),
    ("agg t=0.85 +pooled-merge", f"{G}/agg_t0.85", "merged"),
    ("spec p=.80 +pooled-merge", f"{G}/spec_eigengap_p0.80", "merged"),
    ("agg ORACLE k", f"{G}/agg_oracle", "smoothed"),
    ("spec ORACLE k", f"{G}/spec_oracle", "smoothed"),
    ("agg ORACLE k (1.0/0.5)", "w1.0_h0.5/agg_oracle", "smoothed"),
    ("spec p=.80 (1.0/0.5)", "w1.0_h0.5/spec_eigengap_p0.80", "smoothed"),
]


def cell(fx: str, key: str, stage: str) -> tuple[float | None, int | None, float | None]:
    v = R["fixtures"][fx]["variants"].get(key)
    if not v or stage not in v:
        return None, None, None
    s = v[stage]
    return s["frame_accuracy"], s["k_pred"], s.get("owner_purity")


def fmt(a, k, kt, o):
    if a is None:
        return "—"
    return f"{a:.3f} (k {k}/{kt}{'' if o is None else f', own {o:.2f}'})"


out = []
out.append("| variant | " + " | ".join(FX) + " | mean acc |")
out.append("|---|" + "---|" * (len(FX) + 1))
for label, key, stage in COLS:
    row, accs = [], []
    for fx in FX:
        a, k, o = cell(fx, key, stage)
        row.append(fmt(a, k, R["fixtures"][fx]["k_true"], o))
        accs.append(a if a is not None else 0.0)
    out.append(f"| {label} | " + " | ".join(row) + f" | {sum(accs)/len(accs):.3f} |")
print("\n".join(out))

# runtime table
print("\n| fixture | dur s | speech s | windows (1.5/0.25) | embed wall s | embed cpu s | cluster s | pooled-merge s | total wall s |")
print("|---|---|---|---|---|---|---|---|---|")
for fx in FX:
    r = R["fixtures"][fx]
    w = r.get(f"windows_{G}", {})
    v = r["variants"].get(f"{G}/agg_t0.85", {})
    print(f"| {fx} | {r['duration_s']} | {r['vad_speech_s']} | {w.get('kept')}/{w.get('total')} | {w.get('t_embed')} | "
          f"{w.get('t_embed_cpu', '—')} | {v.get('t_cluster_all_variants', '—')} | "
          f"{v.get('merged', {}).get('t_merge_total', '—')} | {r['t_total_wall']} |")

# coherence table
print("\n| fixture | variant | cluster | n win | majority GT | purity | phantom? | mean pairwise cos | p10 |")
print("|---|---|---|---|---|---|---|---|---|")
for fx in FX:
    for vk, clusters in R["fixtures"][fx]["coherence"].items():
        if not vk.startswith(G):
            continue
        for c, x in clusters.items():
            print(f"| {fx} | {vk.split('/')[1]} | {c} | {x['n']} | {x['majority']} | {x['purity']} | "
                  f"{'PHANTOM' if x['phantom'] else ''} | {x['mean_pairwise_cos']} | {x.get('p10_pairwise_cos')} |")

# best transcript-free variant = single global choice by mean accuracy (non-oracle)
best = None
for label, key, stage in COLS:
    if "ORACLE" in label:
        continue
    accs = [cell(fx, key, stage)[0] or 0.0 for fx in FX]
    m = sum(accs) / len(accs)
    if best is None or m > best[0]:
        best = (m, label, key, stage)
print(f"\nBEST transcript-free variant (global): {best[1]} mean acc {best[0]:.3f}")
for fx in FX:
    v = R["fixtures"][fx]["variants"][best[2]]
    segs = v["merged_segments"] if best[3] == "merged" else v["segments"]
    (HERE / f"pred_{fx}.json").write_text(json.dumps([[round(s, 3), round(e, 3), f"S{l}"] for s, e, l in segs]))
    sc = score.score_fixture(fx, [(s, e, f"S{l}") for s, e, l in segs])
    print(f"  {fx}: acc {sc['frame_accuracy']} k {sc['k_pred']}/{sc['k_true']} recall {sc['per_gt_recall']}")
