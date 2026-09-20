"""Merge results_pipeline/hybrid/tuned into results.json and print the
README tables.  Any python (stdlib only)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["family_real", "poker6", "maggiano3", "openai", "gptaudio", "scene_couple", "scene_family3", "scene_meeting4"]


def load(n):
    p = HERE / n
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    pipe, hyb, tuned = load("results_pipeline.json"), load("results_hybrid.json"), load("results_tuned.json")
    merged = {}
    for fx in ORDER:
        if fx not in pipe and fx not in hyb:
            continue
        m = {}
        for k, v in pipe.get(fx, {}).items():
            m[k] = v
        for k, v in hyb.get(fx, {}).items():
            m[k] = v
        for emb, ev in tuned.get("eval", {}).items():
            if fx in ev:
                m[f"tuned_thr__{emb}"] = ev[fx]
        merged[fx] = m
    gap, segd = load("results_gapfill.json"), load("results_segdiag.json")
    for fx in merged:
        merged[fx]["_gapfill"] = gap.get(fx, {})
        merged[fx]["_segmentation_diag"] = segd.get(fx, {})
    out = {"fixtures": merged, "tuning": {k: tuned.get(k) for k in ("grid", "tuned_on", "chosen", "sweep")},
           "runtime_note": "CPU, torch 4 threads, Apple M-series Mac shared with 3 sibling experiments (load avg 8-28); "
                           "re-timed quieter (load ~8): family_real 30 s audio -> 24 s, openai 70 s audio -> 72 s.",
           "models": {"pyannote/speaker-diarization-3.1": "loaded (MIT, gated)",
                      "pyannote/speaker-diarization-3.0": "NOT loadable: gated, licence not accepted for this token",
                      "pyannote/speaker-diarization-community-1": "NOT loadable: gated + needs pyannote.audio 4.x (venv has 3.3.2)"}}
    (HERE / "results.json").write_text(json.dumps(out, indent=1))

    def cell(r):
        if not r or "frame_accuracy" not in r:
            return "-"
        own = "" if r.get("owner_purity") is None else f" / {r['owner_purity']:.2f}"
        return f"{r['frame_accuracy']:.3f} ({r['k_pred']}/{r['k_true']}{own})"

    def table(title, variants, labels=None):
        labels = labels or variants
        fxs = [f for f in ORDER if f in merged]
        print(f"\n### {title}\n")
        print("| variant | " + " | ".join(fxs) + " | mean |")
        print("|---|" + "---|" * (len(fxs) + 1))
        for v, lab in zip(variants, labels):
            accs = [merged[f][v]["frame_accuracy"] for f in fxs if v in merged[f]]
            mean = f"{sum(accs)/len(accs):.3f}" if accs else "-"
            print(f"| {lab} | " + " | ".join(cell(merged[f].get(v)) for f in fxs) + f" | {mean} |")

    table("End-to-end pyannote 3.1 (frame_accuracy (k_pred/k_true / owner_purity))",
          ["p31_default", "p31_oracle_k", "p31_bounds_2_6", "tuned_thr__wespeaker"],
          ["3.1 default (thr 0.7046)", "3.1 num_speakers=k_true (oracle)", "3.1 min/max 2..6", "3.1 tuned thr (scene-only tuning)"])
    table("Decomposition: pyannote segmentation, swap the clustering/embedder",
          ["seg_pyannote__oracle_labels", "seg_pyannote__wespeaker__pyclust_thr", "seg_pyannote__wespeaker__pyclust_oracle_k",
           "seg_pyannote__wespeaker__ours_avglink_oracle_k", "seg_pyannote__ecapa__pyclust_thr", "seg_pyannote__ecapa__pyclust_oracle_k",
           "seg_pyannote__ecapa__pyclust_bounds_2_6", "seg_pyannote__ecapa__ours_avglink_oracle_k", "tuned_thr__ecapa"],
          ["seg ceiling: ORACLE unit labels", "wespeaker + pyannote clust (stock)", "wespeaker + pyannote clust, oracle k",
           "wespeaker + OUR avg-link, oracle k", "ECAPA + pyannote clust (thr 0.7046)", "ECAPA + pyannote clust, oracle k",
           "ECAPA + pyannote clust, bounds 2..6", "ECAPA + OUR avg-link, oracle k", "ECAPA + pyannote clust, tuned thr"])
    table("Decomposition: GT segmentation (clustering-only)",
          ["seg_GT__wespeaker__pyclust_thr", "seg_GT__wespeaker__pyclust_oracle_k", "seg_GT__wespeaker__ours_avglink_oracle_k",
           "seg_GT__ecapa__pyclust_thr", "seg_GT__ecapa__pyclust_oracle_k", "seg_GT__ecapa__ours_avglink_oracle_k"],
          ["GT seg + wespeaker, pyannote thr", "GT seg + wespeaker, oracle k", "GT seg + wespeaker, OUR avg-link k",
           "GT seg + ECAPA, pyannote thr", "GT seg + ECAPA, oracle k", "GT seg + ECAPA, OUR avg-link k"])
    print("\n### Runtime (CPU, 4 threads, seconds)\n")
    print("| fixture | audio s | default | oracle k | bounds | segmentation | wespeaker embed |")
    print("|---|---|---|---|---|---|---|")
    for f in ORDER:
        if f not in pipe:
            continue
        p = pipe[f]
        i = p.get("intermediates", {})
        print(f"| {f} | | {p['p31_default']['wall_s']} | {p['p31_oracle_k']['wall_s']} | {p['p31_bounds_2_6']['wall_s']} | {i.get('t_segmentation')} | {i.get('t_embedding')} |")
    if tuned:
        print("\n### Threshold sweep (mean frame_accuracy on the 3 scene fixtures)\n")
        for emb, ch in tuned["chosen"].items():
            print(f"- {emb}: chosen {ch['threshold']} (mean {ch['tune_mean_acc']:.3f}); stock 0.7046 gives {ch['stock_tune_mean_acc']:.3f}; ties {ch['tied_candidates']}")


if __name__ == "__main__":
    main()
