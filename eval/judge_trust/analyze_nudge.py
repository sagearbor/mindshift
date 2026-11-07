"""Nudge judge-panel agreement, synthetic-case validity, owner anchor, minimum panel (PREREG.md B)."""
from __future__ import annotations

import itertools
import json
from collections import defaultdict

import numpy as np

from env import MAIN, OUT
from nudge_judges import FAMILY, FLAGS, OWNER_CARDS
from stats import ac1_ci, boot_ci, cohen_kappa, fleiss_ci, kappa_ci, spearman

JUDGES = ["claude-sonnet-5-5", "claude-haiku-5-5", "gpt-5-mini", "gpt-6.1-terra", "gemini-3.8-flash",
          "gemini-3.5-flash-lite", "qwen-local"]


def load() -> dict:
    g: dict = defaultdict(dict)
    for fn in ("nudge_results.jsonl", "nudge_results_qwen.jsonl"):
        p = OUT / fn
        if p.exists():
            for line in open(p):
                d = json.loads(line)
                if d.get("ok") or d["case_id"] not in g[d["model"]]:
                    g[d["model"]][d["case_id"]] = d
    return g


def cost_per_case() -> dict:
    tot, n = defaultdict(float), defaultdict(int)
    for line in open(OUT / "calls.jsonl"):
        d = json.loads(line)
        if d["task"] == "nudge-judge":
            tot[d["model"]] += d["usd"]
            n[d["model"]] += 1
    return {m: tot[m] / n[m] for m in tot} | {"qwen-local": 0.0}


def majority(vals) -> int:
    return int(sum(vals) * 2 > len(vals))  # ties -> not helpful (silence default)


def agreement(G, judges, ids, field) -> dict:
    ids = [i for i in ids if all(G[j].get(i, {}).get("ok") for j in judges)]
    M = {j: np.array([int(G[j][i][field]) for i in ids]) for j in judges}
    out = {"n": len(ids), "rate": {j: float(M[j].mean()) for j in judges}, "pairwise": {}, "loo": {}}
    for a, b in itertools.combinations(judges, 2):
        out["pairwise"][f"{a} | {b}"] = kappa_ci(M[a], M[b], [0, 1])
    for j in judges:
        others = [o for o in judges if o != j]
        maj = np.array([majority([M[o][k] for o in others]) for k in range(len(ids))])
        out["loo"][j] = kappa_ci(M[j], maj, [0, 1])
    cnt = np.array([[sum(M[j][k] == 0 for j in judges), sum(M[j][k] == 1 for j in judges)] for k in range(len(ids))])
    out["fleiss"] = fleiss_ci(cnt) if len(ids) > 3 else None
    out["mean_pairwise"] = float(np.mean([v[0] for v in out["pairwise"].values()]))
    # POST-HOC (not pre-registered): prevalence-robust agreement
    out["pct_agree"] = {f"{a} | {b}": float(np.mean(M[a] == M[b])) for a, b in itertools.combinations(judges, 2)}
    out["ac1"] = {f"{a} | {b}": ac1_ci(M[a], M[b]) for a, b in itertools.combinations(judges, 2)}
    out["loo_ac1"] = {}
    for j in judges:
        others = [o for o in judges if o != j]
        maj = np.array([majority([M[o][k] for o in others]) for k in range(len(ids))])
        out["loo_ac1"][j] = ac1_ci(M[j], maj)
    out["_M"] = M
    out["_ids"] = ids
    return out


def main() -> dict:
    G = load()
    judges = [j for j in JUDGES if G.get(j)]
    frozen = json.loads((OUT / "nudge_cases_frozen.json").read_text())
    real_ids = [c["case_id"] for c in frozen["cases"] if c["kind"] == "real"]
    syn = [c for c in frozen["cases"] if c["kind"] != "real"]
    rep = {"judges": judges, "families": {j: FAMILY[j] for j in judges}, "cost_per_case": cost_per_case()}
    rep["parse_fail"] = {j: sum(1 for c in frozen["cases"] if not G[j].get(c["case_id"], {}).get("ok")) for j in judges}
    for field in ("helpful", "derived_helpful"):
        a = agreement(G, judges, real_ids, field)
        rep[f"real_{field}"] = {k: v for k, v in a.items() if not k.startswith("_")}
        if field == "helpful":
            M, ids = a["_M"], a["_ids"]
    # POST-HOC: continuous agreement (Spearman) on P(helpful) and on timing 0-3, real frozen cases
    rep["spearman_p"], rep["spearman_timing"] = {}, {}
    for a_, b_ in itertools.combinations(judges, 2):
        rep["spearman_p"][f"{a_} | {b_}"] = spearman([G[a_][i].get("p_helpful", 0) for i in ids],
                                                     [G[b_][i].get("p_helpful", 0) for i in ids])
        rep["spearman_timing"][f"{a_} | {b_}"] = spearman([G[a_][i].get("timing", 0) for i in ids],
                                                          [G[b_][i].get("timing", 0) for i in ids])
    # the 17 PR #215 lines, and the combined set (real + 17 + synthetic): same statistics
    p17 = sorted({k for j in judges for k in G[j] if k.startswith("pr215:")})
    rep["e17_helpful"] = {k: v for k, v in agreement(G, judges, p17, "helpful").items() if not k.startswith("_")}
    allids = real_ids + p17 + [c["case_id"] for c in syn]
    rep["all_helpful"] = {k: v for k, v in agreement(G, judges, allids, "helpful").items() if not k.startswith("_")}
    # open-weight local judge, scored separately (slow on this Mac, may cover only part of the set)
    if G.get("qwen-local") and "qwen-local" not in judges:
        core3 = ["claude-haiku-5-5", "gpt-6.1-terra", "gemini-3.8-flash"]
        qids = [i for i in real_ids + p17 if G["qwen-local"].get(i, {}).get("ok") and all(G[j].get(i, {}).get("ok") for j in core3)]
        if qids:
            q = np.array([int(G["qwen-local"][i]["helpful"]) for i in qids])
            cm = np.array([majority([int(G[j][i]["helpful"]) for j in core3]) for i in qids])
            qs = [c for c in syn if G["qwen-local"].get(c["case_id"], {}).get("ok")]
            rep["qwen"] = {"n": len(qids), "rate": float(q.mean()), "core_rate": float(cm.mean()),
                           "kappa_vs_core": kappa_ci(q, cm, [0, 1]), "ac1_vs_core": ac1_ci(q, cm),
                           "pct_agree": float(np.mean(q == cm)),
                           "synthetic_n": len(qs),
                           "synthetic_neg_correct": float(np.mean([int(G["qwen-local"][c["case_id"]]["helpful"]) == 0 for c in qs
                                                                   if c["expected_helpful"] == 0])) if qs else None,
                           "secs_per_case": float(np.mean([G["qwen-local"][i].get("secs") or 0 for i in qids]))}
    # PREREG B2: confirmation of the post-hoc core panel on a fresh set
    cp = OUT / "nudge_cases_confirm.json"
    core = ["claude-haiku-5-5", "gpt-6.1-terra", "gemini-3.8-flash"]
    if cp.exists() and all(j in G for j in core):
        cids = [c["case_id"] for c in json.loads(cp.read_text())["cases"]]
        ac = agreement(G, core, cids, "helpful")
        if ac["n"]:
            ok = bool(ac["fleiss"][0] >= 0.40 and ac["fleiss"][1] >= 0.20 and min(ac["pct_agree"].values()) >= 0.85)
            rep["confirm"] = {**{k: v for k, v in ac.items() if not k.startswith("_")}, "panel": core,
                              "n_attempted": len(cids), "verdict": "CONFIRMED" if ok else "NOT CONFIRMED"}
    # admissibility per prereg
    rep["admissible"] = {j: bool(rep["real_helpful"]["loo"][j][0] >= 0.40 and rep["real_helpful"]["loo"][j][1] > 0.20)
                         for j in judges}
    # checklist item agreement (Fleiss over all judges, real cases)
    rep["checklist_fleiss"] = {}
    for flag in ["real_trigger", *FLAGS]:
        ids = [i for i in real_ids if all(G[j].get(i, {}).get("ok") for j in judges)]
        cnt = []
        for i in ids:
            v = [int(bool((G[j][i].get("checklist") or {}).get(flag))) for j in judges]
            cnt.append([v.count(0), v.count(1)])
        rep["checklist_fleiss"][flag] = {"fleiss": fleiss_ci(np.array(cnt)) if ids else None,
                                         "rate": float(np.mean([c[1] / len(judges) for c in cnt])) if cnt else None}
    # soft labels: panel mean P(helpful) and each judge's Brier vs leave-one-out majority
    rep["soft"] = {}
    for j in judges:
        others = [o for o in judges if o != j]
        ps = np.array([float(G[j][i].get("p_helpful", G[j][i]["helpful"])) for i in ids])
        maj = np.array([majority([M[o][k] for o in others]) for k in range(len(ids))])
        rep["soft"][j] = {"brier_vs_loo_majority": float(np.mean((ps - maj) ** 2)), "mean_p": float(ps.mean())}
    pm = np.mean([[float(G[j][i].get("p_helpful", 0)) for i in ids] for j in judges], axis=0)
    rep["panel_mean_p_hist"] = np.histogram(pm, bins=[0, .2, .4, .6, .8, 1.0001])[0].tolist()
    # synthetic validity
    rep["synthetic"] = {}
    for j in judges:
        per = defaultdict(list)
        for c in syn:
            r = G[j].get(c["case_id"])
            if r and r.get("ok"):
                per[c["kind"]].append(int(r["helpful"]) == c["expected_helpful"])
        neg = [x for k, v in per.items() if not k.startswith("pos") for x in v]
        rep["synthetic"][j] = {"by_kind": {k: float(np.mean(v)) for k, v in per.items()},
                               "neg_correct": float(np.mean(neg)) if neg else None,
                               "neg_pass": bool(neg and np.mean(neg) >= 0.90),
                               "pos_controls_helpful": float(np.mean(per.get("pos_regulate_heated", [np.nan])))}
    # existing 17 (PR #215): original A/B labels plus our panel
    p = MAIN / ".claude/worktrees/agent-a90ab94643480b5c9/tmp/nudge-eval/judge-baseline-all.json"
    if p.exists():
        d = json.loads(p.read_text())
        keys = list(d["views"])
        e17 = {"orig_A_sonnet": [int(d["grades"]["A"][k]["helpful"]) for k in keys],
               "orig_B_gpt5mini": [int(d["grades"]["B"][k]["helpful"]) for k in keys]}
        for j in judges:
            e17[j] = [int(G[j].get(f"pr215:{k}", {}).get("helpful", -1)) for k in keys]
        rep["existing17"] = {"rates": {k: float(np.mean([x for x in v if x >= 0])) for k, v in e17.items()},
                             "orig_kappa": kappa_ci(e17["orig_A_sonnet"], e17["orig_B_gpt5mini"], [0, 1]),
                             "orig_agree": int(sum(a == b for a, b in zip(e17["orig_A_sonnet"], e17["orig_B_gpt5mini"]))),
                             "self_consistency": {
                                 "sonnet_new_vs_orig": kappa_ci(e17["claude-sonnet-5-5"], e17["orig_A_sonnet"], [0, 1]) if "claude-sonnet-5-5" in e17 else None,
                                 "gpt5mini_new_vs_orig": kappa_ci(e17["gpt-5-mini"], e17["orig_B_gpt5mini"], [0, 1]) if "gpt-5-mini" in e17 else None},
                             "panel_majority": [majority([e17[j][i] for j in judges if e17[j][i] >= 0]) for i in range(len(keys))]}
    # owner anchor (contaminated)
    rep["owner"] = {}
    for j in judges:
        hits = [(int(G[j][f"owner:{k}"]["helpful"]) == v) for k, v in OWNER_CARDS.items() if G[j].get(f"owner:{k}", {}).get("ok")]
        rep["owner"][j] = {"n": len(hits), "agree": int(sum(hits))}
    # minimum panel
    full = np.array([majority([M[j][k] for j in judges]) for k in range(len(ids))])
    cpc = rep["cost_per_case"]
    cands = []
    for r in range(2, len(judges)):
        for sub in itertools.combinations(judges, r):
            if len({FAMILY[j] for j in sub}) < 2:
                continue
            maj = np.array([majority([M[j][k] for j in sub]) for k in range(len(ids))])
            k = cohen_kappa(maj, full, [0, 1])
            cands.append({"panel": list(sub), "size": r, "kappa_vs_full": k,
                          "ci": boot_ci(lambda i, maj=maj: cohen_kappa(maj[i], full[i], [0, 1]), len(ids)),
                          "usd_per_case": sum(cpc.get(j, 0) for j in sub)})
    cands.sort(key=lambda c: (c["size"], -c["kappa_vs_full"], c["usd_per_case"]))
    rep["min_panel_candidates"] = [c for c in cands if c["kappa_vs_full"] >= 0.80][:10]
    rep["best_by_size"] = {r: max([c for c in cands if c["size"] == r], key=lambda c: c["kappa_vs_full"]) for r in
                           sorted({c["size"] for c in cands})}
    rep["full_panel_rate"] = float(full.mean())
    (OUT / "nudge_analysis.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


if __name__ == "__main__":
    r = main()
    print("judges", r["judges"])
    h = r["real_helpful"]
    print("n", h["n"], "rates", {k: round(v, 2) for k, v in h["rate"].items()})
    print("fleiss helpful", h["fleiss"], "mean pairwise", round(h["mean_pairwise"], 3))
    print("fleiss derived", r["real_derived_helpful"]["fleiss"], "mean pairwise", round(r["real_derived_helpful"]["mean_pairwise"], 3))
    for j, v in h["loo"].items():
        print("  loo", j, [round(x, 2) for x in v], "admissible", r["admissible"][j])
    for k, v in h["pairwise"].items():
        print("  pair", k, [round(x, 2) for x in v])
    print("checklist", {k: (round(v["fleiss"][0], 2) if v["fleiss"] else None, round(v["rate"] or 0, 2)) for k, v in r["checklist_fleiss"].items()})
    print("synthetic", json.dumps(r["synthetic"], indent=0)[:2500])
    print("existing17", json.dumps(r.get("existing17", {}), default=float)[:1200])
    print("owner", r["owner"])
    print("min panel", json.dumps(r["min_panel_candidates"][:5], default=float))
    print("best by size", json.dumps(r["best_by_size"], default=float)[:1500])
    print("cost", r["cost_per_case"], "soft", r["soft"])
