"""Run analyze_nudge, optionally excluding judges (e.g. --without qwen-local while it is still running)."""
import argparse
import json

import analyze_nudge as a

ap = argparse.ArgumentParser()
ap.add_argument("--without", nargs="*", default=[])
args = ap.parse_args()
a.JUDGES = [j for j in a.JUDGES if j not in args.without]
r = a.main()
h = r["real_helpful"]
print("judges", r["judges"], "n real", h["n"])
print("rates", {k: round(v, 2) for k, v in h["rate"].items()})
print("fleiss real", [round(x, 2) for x in h["fleiss"]], "17", [round(x, 2) for x in r["e17_helpful"]["fleiss"]],
      "all", [round(x, 2) for x in r["all_helpful"]["fleiss"]], "derived real", [round(x, 2) for x in r["real_derived_helpful"]["fleiss"]])
for k in h["pairwise"]:
    print(f"  {k:48} k {h['pairwise'][k][0]:+.2f} agree {h['pct_agree'][k]:.2f} AC1 {h['ac1'][k][0]:+.2f} "
          f"[{h['ac1'][k][1]:+.2f},{h['ac1'][k][2]:+.2f}] rhoP {r['spearman_p'][k]:+.2f} rhoT {r['spearman_timing'][k]:+.2f} "
          f"| 17: k {r['e17_helpful']['pairwise'][k][0]:+.2f} agree {r['e17_helpful']['pct_agree'][k]:.2f}")
for j in r["judges"]:
    print("  loo", j, "kappa", [round(x, 2) for x in h["loo"][j]], "AC1", [round(x, 2) for x in h["loo_ac1"][j]],
          "17 rate", round(r["e17_helpful"]["rate"][j], 2), "17 loo k", [round(x, 2) for x in r["e17_helpful"]["loo"][j]])
print("checklist", {k: (round(v["fleiss"][0], 2) if v["fleiss"] else None, round(v["rate"] or 0, 2)) for k, v in r["checklist_fleiss"].items()})
print("owner", r["owner"])
print("pos", {j: r["synthetic"][j]["pos_controls_helpful"] for j in r["judges"]})
print("soft", json.dumps(r["soft"]))
print("existing17", json.dumps(r["existing17"]["rates"]), r["existing17"]["self_consistency"])
