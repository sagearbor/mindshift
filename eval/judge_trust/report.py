"""Render tmp/judge-trust-20261105/REPORT.html (mobile-first, light/dark, native <details>)."""
from __future__ import annotations

import html
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from env import OUT

HERE = Path(__file__).parent
E = lambda s: html.escape(str(s))  # noqa: E731

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--line:#e3e1db;--card:#fff;--ok:#1f7a45;--okbg:#e5f4ea;--warn:#8a5a00;
--warnbg:#fbf0d9;--bad:#a1261d;--badbg:#f9e3e0;--acc:#2f5d8a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151514;--fg:#ecebe7;--mut:#a3a19b;--line:#33322f;
--card:#1e1e1c;--ok:#7fd3a0;--okbg:#17321f;--warn:#f0c46b;--warnbg:#3a2e12;--bad:#f2948a;--badbg:#3c1a17;--acc:#8fb8e3}}
:root[data-theme="dark"]{--bg:#151514;--fg:#ecebe7;--mut:#a3a19b;--line:#33322f;--card:#1e1e1c;--ok:#7fd3a0;--okbg:#17321f;
--warn:#f0c46b;--warnbg:#3a2e12;--bad:#f2948a;--badbg:#3c1a17;--acc:#8fb8e3}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,Segoe UI,sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}h1{font-size:1.35rem;margin:.2em 0}h2{font-size:1.1rem;margin:0}
h3{font-size:1rem;margin:1em 0 .3em}.mut{color:var(--mut);font-size:.9em}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:10px 0;padding:0 12px}
summary{cursor:pointer;padding:12px 0;font-weight:600;list-style-position:outside}details[open]>summary{border-bottom:1px solid var(--line);margin-bottom:8px}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:6px 0 12px}table{border-collapse:collapse;font-size:.85em;min-width:100%}
th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top;white-space:nowrap}
td.w{white-space:normal;min-width:180px}th{color:var(--mut);font-weight:600}
.b{display:inline-block;border-radius:6px;padding:0 6px;font-size:.8em;font-weight:600}
.ok{background:var(--okbg);color:var(--ok)}.warn{background:var(--warnbg);color:var(--warn)}.bad{background:var(--badbg);color:var(--bad)}
.box{border-left:4px solid var(--acc);background:var(--card);padding:10px 12px;border-radius:6px;margin:10px 0}
pre{white-space:pre-wrap;font-size:.8em;background:var(--bg);border:1px solid var(--line);padding:8px;border-radius:6px}
ul{padding-left:1.2em}li{margin:.2em 0}
"""


def badge(text: str) -> str:
    t = text.upper()
    cls = "ok" if t.startswith(("TRUST", "PASS", "YES")) else ("warn" if t.startswith(("USABLE", "WARN")) else "bad")
    return f'<span class="b {cls}">{E(text)}</span>'


def k3(x) -> str:
    if not x or x[0] is None:
        return "-"
    return f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]"


def table(head, rows, wide=()):
    h = "".join(f"<th>{E(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f'<td class="w">{c}</td>' if i in wide else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>"
                for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def sec(title, body, open_=False):
    return f"<details{' open' if open_ else ''}><summary><h2 style='display:inline'>{E(title)}</h2></summary>{body}</details>"


def spend():
    tot, by = 0.0, defaultdict(lambda: [0, 0.0])
    for line in open(OUT / "calls.jsonl"):
        d = json.loads(line)
        tot += d["usd"]
        by[(d["model"], d["task"].split("-")[0] if d["task"].startswith("emo") else d["task"])][0] += 1
        by[(d["model"], d["task"].split("-")[0] if d["task"].startswith("emo") else d["task"])][1] += d["usd"]
    return tot, by


def prereg_commit():
    r = subprocess.run(["git", "-C", str(HERE), "log", "--diff-filter=A", "--format=%h %ci", "--", "PREREG.md"],
                       capture_output=True, text=True, timeout=20)
    return r.stdout.strip()


def build(recs_html: str, bottom_html: str) -> str:
    ea = json.loads((OUT / "emotion_analysis.json").read_text())
    na = json.loads((OUT / "nudge_analysis.json").read_text())
    cc = json.loads((OUT / "confer_check.json").read_text()) if (OUT / "confer_check.json").exists() else None
    man = json.loads((OUT / "nudge_cases_frozen.json").read_text())
    cs = json.loads((OUT / "cremad_sample.json").read_text())
    tot, by = spend()
    EM = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
    H = {"anger": 0.62, "disgust": 0.38, "fear": 0.45, "happy": 0.60, "neutral": 0.35, "sad": 0.50}
    parts = [f"<main><h1>Judge trust: can our AI judges be believed?</h1><p class='mut'>MindShift, 2026-11-05. "
             f"Spend ${tot:.2f} of the $5 cap. Pre-registration committed {E(prereg_commit())} before any judge ran. "
             "Code: eval/judge_trust/ on branch feat/judge-trust.</p>", bottom_html]
    # ---------------- emotion
    hum = ea["human"]["pooled_all_raters"]
    rows = [["one human rater (pooled)", f"{hum['overall_kappa']:.2f}", "-", "-", "-", "-"]]
    for m, R in ea["models"].items():
        ao = R.get("audio_only") or {}
        rows.append([E(m), k3(ao.get("kappa")), f"{ao.get('accuracy', 0):.2f}", f"{ao.get('fail_rate', 0):.0%}",
                     f"{ao.get('mean_tvd', 0):.2f} " + badge("PASS" if ao.get("soft_pass") else "FAIL"),
                     " ".join(f"{e[:3]} {v:.0%}" for e, v in (ao.get("pred_dist") or {}).items())])
    body = ("<p>CREMA-D, 300 clips (50 per crowd-majority emotion), reference = majority of ~10 voice-only crowd "
            "votes. Kappa 95% CI by bootstrap over clips. Soft labels: mean total-variation distance (TVD) between "
            f"the labeller's probabilities and the crowd vote shares; split-half crowd TVD = {ea['split_half_tvd']:.2f}, "
            f"one rater vs the rest = {ea['single_rater_tvd']:.2f}; pass bar = split-half + 0.05.</p>"
            + table(["labeller", "6-class kappa [CI]", "acc", "no answer", "soft TVD", "what it predicts"], rows))
    vr = [["human H_e"] + [f"{H[e]:.2f}" for e in EM]]
    for m, R in ea["models"].items():
        pc = (R.get("audio_only") or {}).get("per_class") or {}
        vr.append([E(m)] + [badge(pc[e]["verdict"]) + f"<br>k {k3(pc[e]['kappa'])}<br>P {pc[e]['precision'][0] or 0:.2f} R {pc[e]['recall'][0] or 0:.2f}"
                            if e in pc else "-" for e in EM])
    body += "<h3>Per emotion, audio only (pre-registered verdicts)</h3>" + table(["labeller"] + EM, vr)
    cr = []
    for m, R in ea["models"].items():
        cr.append([E(m), k3((R.get("audio_only") or {}).get("kappa")), k3((R.get("both") or {}).get("kappa")),
                   k3((R.get("contradict") or {}).get("kappa")),
                   ((k3(R["text_only"]["kappa"]) + " " + badge("PASS" if R["text_only"]["pass"] else "FAIL"))
                    if R.get("text_only") and R["text_only"]["kappa"][0] == R["text_only"]["kappa"][0]
                    else "n/a (model rejects text-only input)" if R.get("text_only") else "-"),
                   (f"{R['leak_contradict']['rate']:.0%} ({R['leak_contradict']['moved_to_text']}/{R['leak_contradict']['eligible']}) " + badge(R['leak_contradict']['verdict'])) if R.get("leak_contradict") else "-",
                   (f"{R['leak_tts']['labelled_word_emotion']:.0%} word / {R['leak_tts']['labelled_neutral']:.0%} neutral " + badge(R['leak_tts']['verdict'])) if R.get("leak_tts") else "-"])
    body += ("<h3>Conditions and text-leakage tests</h3><p class='mut'>contradict = same audio, transcript swapped for "
             "strong words of a different emotion; 'moved' = label switched to the words' emotion (of clips whose "
             "audio-only label was not already that emotion). TTS = flat macOS voices reading strongly emotional "
             "sentences (24 clips): share labelled with the words' emotion.</p>"
             + table(["labeller", "audio only", "audio + true text", "audio + contradicting text", "text only", "moved to text", "flat TTS"], cr))
    af = ea.get("agreement_filter") or {}
    if af:
        body += "<h3>Two-labeller agreement filter (keep a label only where both agree)</h3>" + table(
            ["pair", "coverage", "accuracy on kept", "kappa on kept"],
            [[E(k), f"{v['coverage']:.0%} of {v['n_both']}", f"{v['accuracy_on_agreed'] or 0:.2f}", k3(v.get("kappa_on_agreed"))] for k, v in af.items()])
    rv = json.loads((OUT / "ravdess_check.json").read_text()) if (OUT / "ravdess_check.json").exists() else None
    if rv:
        body += ("<h3>Cross-corpus: RAVDESS (eval only; reference = actor's intended emotion, 96 clips)</h3>"
                 + table(["labeller", "kappa [CI]", "acc"] + EM,
                         [[E(m), k3(r["kappa"]), f"{r['accuracy']:.2f}"] + [f"R {r['recall'][e] or 0:.2f}" for e in EM] for m, r in rv.items()]))
    parts.append(sec("1. Emotion labellers vs human crowd (CREMA-D)", body, True))
    if cc:
        parts.append(sec("2. Language-agnostic check (CONFER, Greek debates, eval only)",
                         f"<p>{cc['n_windows']} five-second windows from {cc['n_recs']} recordings: the 3 most- and 3 least-"
                         "conflicted windows per recording by the 10-rater curve. AUC = chance a hot window scores above "
                         "a cold one of the same recording (0.5 = no signal). CI over recordings.</p>"
                         + table(["signal", "within-recording AUC [CI]"], [[E(k), k3(v)] for k, v in cc["auc"].items()])))
    # ---------------- nudge
    h = na["real_helpful"]
    d = na["real_derived_helpful"]
    judges = na["judges"]
    rr = [[E(j), E(na["families"][j]), f"{h['rate'][j]:.0%}", k3(h["loo"][j]), badge("YES" if na["admissible"][j] else "NO"),
           f"{na['synthetic'][j]['neg_correct']:.0%} " + badge("PASS" if na["synthetic"][j]["neg_pass"] else "FAIL") if na["synthetic"][j]["neg_correct"] is not None else "-",
           f"{na['synthetic'][j]['pos_controls_helpful']:.0%}", f"{na['owner'][j]['agree']}/{na['owner'][j]['n']}",
           f"${na['cost_per_case'].get(j, 0):.4f}", str(na["parse_fail"][j])] for j in judges]
    body = (f"<p>Frozen set: {h['n']} real coach lines from held-out sessions (all judges answered), plus "
            f"{sum(v for k, v in man['by_kind'].items() if k != 'real')} synthetic cases whose answer is fixed by the owner's written rules. "
            "Rubric v4 (PR #215) plus P(helpful) and a 6-item checklist, one call per case.</p>"
            f"<div class='box'>Pre-registered verdict: panel "
            f"{badge('TRUSTWORTHY' if h['fleiss'][0] >= 0.40 and h['fleiss'][1] >= 0.20 else 'NOT TRUSTWORTHY')}; "
            f"admissible judges: {sum(na['admissible'].values())} of {len(judges)}. "
            f"Panel Fleiss kappa (helpful): <b>{k3(h['fleiss'])}</b>, mean pairwise {h['mean_pairwise']:.2f}. "
            f"Checklist-derived verdict: Fleiss <b>{k3(d['fleiss'])}</b>, mean pairwise {d['mean_pairwise']:.2f}. "
            f"Full-panel majority calls {na['full_panel_rate']:.0%} of real lines helpful.</div>"
            + table(["judge", "family", "says helpful", "kappa vs others' majority", "admissible?", "rule cases right", "pos. controls helpful",
                     "owner cards", "$/case", "no answer"], rr))
    pr = [[E(k.replace(" | ", " vs ")), k3(v)] for k, v in sorted(h["pairwise"].items(), key=lambda kv: -kv[1][0])]
    body += "<h3>Pairwise Cohen kappa (helpful), sorted</h3>" + table(["pair", "kappa [CI]"], pr)
    body += "<h3>Checklist items: Fleiss kappa across all judges</h3>" + table(
        ["item", "Fleiss [CI]", "mean yes-rate"], [[E(k), k3(v["fleiss"]), f"{(v['rate'] or 0):.0%}"] for k, v in na["checklist_fleiss"].items()])
    syn_kinds = sorted({k for j in judges for k in na["synthetic"][j]["by_kind"]})
    body += "<h3>Synthetic rule cases: share graded as expected</h3>" + table(
        ["judge"] + syn_kinds, [[E(j)] + [f"{na['synthetic'][j]['by_kind'].get(k, float('nan')):.0%}" for k in syn_kinds] for j in judges])
    if na.get("existing17"):
        e = na["existing17"]
        body += ("<h3>The 17 lines from PR #215 (where kappa 0.14 came from)</h3>"
                 f"<p>Original: Sonnet 5.5 vs gpt-5-mini agreed on {e['orig_agree']}/17, kappa {k3(e['orig_kappa'])}. "
                 "Same models re-run with this prompt (adds P(helpful) + checklist): Sonnet vs its own earlier verdicts "
                 f"{k3(e['self_consistency']['sonnet_new_vs_orig'])}, gpt-5-mini vs its earlier verdicts "
                 f"{k3(e['self_consistency']['gpt5mini_new_vs_orig'])}.</p>"
                 + table(["judge", "says helpful on the 17"], [[E(k), f"{v:.0%}"] for k, v in e["rates"].items()]))
        e17 = na["e17_helpful"]
        body += "<p>Pairwise on the 17 (sorted by kappa):</p>" + table(
            ["pair", "kappa [CI]", "% agree"],
            [[E(k.replace(" | ", " vs ")), k3(v), f"{e17['pct_agree'][k]:.0%}"] for k, v in sorted(e17["pairwise"].items(), key=lambda kv: -kv[1][0])])
    body += ("<h3>Why kappa collapses here: post-hoc, prevalence-robust view (NOT pre-registered)</h3>"
             "<p>Almost every held-out line is judged not helpful, so a judge that says 'no' to everything (Sonnet 5.5: "
             "0 of 93) gets kappa 0 even when it agrees on 90%+ of lines. Gwet's AC1 and raw agreement are not "
             "distorted by that; Spearman compares the judges' P(helpful) and timing scores as rankings.</p>" + table(
                 ["pair", "% agree", "AC1 [CI]", "rho P(helpful)", "rho timing"],
                 [[E(k.replace(" | ", " vs ")), f"{h['pct_agree'][k]:.0%}", k3(h["ac1"][k]), f"{na['spearman_p'][k]:+.2f}",
                   f"{na['spearman_timing'][k]:+.2f}"] for k in sorted(h["pct_agree"], key=lambda k: -h["pct_agree"][k])]))
    c = na.get("confirm")
    if c:
        body += (f"<h3>Confirmation of the post-hoc core panel on {c['n']} fresh lines (PREREG B2)</h3>"
                 f"<div class='box'>{E(' + '.join(c['panel']))}: Fleiss <b>{k3(c['fleiss'])}</b>, "
                 f"pairwise agreement {', '.join(f'{v:.0%}' for v in c['pct_agree'].values())}, helpful rates "
                 f"{', '.join(f'{v:.0%}' for v in c['rate'].values())}. {badge(c['verdict'])}</div>"
                 + table(["pair", "kappa [CI]", "% agree", "AC1 [CI]"],
                         [[E(k.replace(' | ', ' vs ')), k3(v), f"{c['pct_agree'][k]:.0%}", k3(c['ac1'][k])] for k, v in c["pairwise"].items()]))
    q = na.get("qwen")
    if q:
        body += (f"<h3>Open-weight local judge: Qwen3.5-9B 4-bit (Apache-2.0, MLX, $0)</h3><p>Scored on the {q['n']} real/17 "
                 f"lines it finished ({q['secs_per_case']:.0f} s per line on this Mac). Says helpful {q['rate']:.0%} vs core-panel "
                 f"majority {q['core_rate']:.0%}; agreement {q['pct_agree']:.0%}, kappa {k3(q['kappa_vs_core'])}, AC1 {k3(q['ac1_vs_core'])}. "
                 f"Rule cases right: {(q['synthetic_neg_correct'] or 0):.0%} of {q['synthetic_n']}.</p>")
    parts.append(sec("3. Nudge-quality judge panel (no human labels exist)", body, True))
    # ---------------- frozen set
    body = (f"<p>Set SHA-256 <code>{man['set_sha256']}</code>; rubric v4 SHA-256 <code>{man['rubric']['sha256'][:16]}...</code>. "
            f"Built from {man['pool_rows_heldout']} held-out pool rows; {man['excluded_ids_count']} ids excluded as tuning material "
            f"(rubric examples, owner cards, rubric-development grades, the PR #215 lines); exact-text leakage check dropped "
            f"{len(man['leakage_text_matches_dropped'])} more ({E(', '.join(man['leakage_text_matches_dropped']))}).</p>"
            + table(["kind / cell", "n"], [[E(k), v] for k, v in man["by_cell"].items()])
            + f"<p>CREMA-D sample manifest SHA-256 <code>{cs['manifest_sha256']}</code> (300 clips, per-file SHA-256 inside cremad_sample.json).</p>"
            "<p class='mut'>Files: tmp/judge-trust-20261105/nudge_cases_frozen.json, cremad_sample.json. Rebuild: "
            "eval/judge_trust/nudge_cases.py, cremad.py (seeded, deterministic).</p>")
    parts.append(sec("4. Frozen case sets", body))
    parts.append(sec("5. Recommendation", recs_html, True))
    rows = sorted(([E(m), E(t), n, f"${u:.3f}"] for (m, t), (n, u) in by.items()), key=lambda r: r[0])
    parts.append(sec("6. Spend (every call logged in calls.jsonl)", f"<p>Total ${tot:.3f} (estimated from token counts x public prices; "
                     "Gemini 3.8 Flash at its 2026 page price).</p>" + table(["model", "task", "calls", "est. USD"], rows)))
    parts.append(sec("7. Pre-registration (verbatim)", f"<pre>{E((HERE / 'PREREG.md').read_text())}</pre>"))
    parts.append("</main>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Judge Trust Report</title><style>{CSS}</style></head><body>" + "".join(parts) + "</body></html>")


if __name__ == "__main__":
    recs = (HERE / "report_text/recs.html").read_text()
    bottom = (HERE / "report_text/bottom.html").read_text()
    (OUT / "REPORT.html").write_text(build(recs, bottom))
    print("wrote", OUT / "REPORT.html")
