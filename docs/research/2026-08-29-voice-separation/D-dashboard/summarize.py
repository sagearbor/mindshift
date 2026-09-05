"""Print the per-fixture separability ranking from data_<fixture>.json
(the numbers quoted in README.md). Usage: python summarize.py [--md]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["maggiano3", "poker6", "family_real", "scene_family3", "scene_meeting4",
         "openai", "gptaudio", "scene_couple"]


def main() -> None:
    md = "--md" in sys.argv
    for name in ORDER:
        p = HERE / f"data_{name}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        ranked = sorted(((k, v) for k, v in d["sep"].items() if v.get("ratio") is not None),
                        key=lambda kv: -kv[1]["ratio"])
        prod = d["prod"][0]
        pooled = d["pooled"]
        worst = max(pooled["pairs"], key=lambda x: x[2]) if pooled["pairs"] else None
        feats = ", ".join(f"{d['features'][k]['label']} {v['ratio']:.2f} ({v['accuracy']*100:.0f}% alone)"
                          for k, v in ranked[:4])
        prod_txt = ("None" if prod["returned_none"] else f"{prod['k_pred']}/{d['k_true']}") + \
            f" voices, {prod['score']['frame_accuracy']*100:.0f}% frames"
        emb = f"ECAPA nearest-voiceprint {pooled['nearest_centroid_window_accuracy']*100:.0f}% of windows, " \
              f"silhouette {d['embed']['silhouette_cosine']}, closest pair {worst[0]}–{worst[1]} cos {worst[2]:.2f}"
        if md:
            print(f"- **{name}** (k={d['k_true']}): best single feature **{d['features'][ranked[0][0]]['label']}** "
                  f"ratio {ranked[0][1]['ratio']:.2f}; ranking: {feats}. {emb}. Production (GT turns in): {prod_txt}.")
        else:
            print(f"{name} k={d['k_true']}: {feats} | {emb} | prod {prod_txt}")


if __name__ == "__main__":
    main()
