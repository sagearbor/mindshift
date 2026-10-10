#!/usr/bin/env python3
"""landscape.py — the shared experiment table ("energy landscape") that every
overnight tuning agent appends to and reads before choosing its next try.

  python scripts/recreplay/landscape.py add --agent B --exp B-007 --change "min gap 20s" \
      --params '{"min_gap_s":20}' --split dev --clips t1 --baseline B-000 \
      --metrics '{"calm_fires_h":31.0,"lift":0.08}' --verdict better --notes "..."
  python scripts/recreplay/landscape.py show [--agent B] [--last 20]
  python scripts/recreplay/landscape.py html            # -> tmp/recordings/landscape/landscape.html

Rows are JSON lines in tmp/recordings/landscape/results.jsonl (file-locked).
Metric keys (use these names so rows compare; omit what you didn't measure):
  calm_fires_h, heated_fires_h, hit_rate, chance_hit_rate, lift, wearer_recall,
  false_self_rate, time_to_confirm_s, alert_buzz_hits, alert_buzz_false_h,
  label_leaks, short_turn_line_pct, after_laugh_lines, judge_score, latency_p50_s,
  vad_coverage, merged_turn_pct, spend_usd
Verdict: better | worse | neutral | broken  (vs --baseline on the SAME split+clips).
"""
from __future__ import annotations

import argparse
import fcntl
import html
import json
import time
from pathlib import Path

ROOT = Path("/Users/sagearbor/projects/githubs/mindshift/tmp/recordings/landscape")
DB = ROOT / "results.jsonl"
LOWER_IS_BETTER = {"calm_fires_h", "heated_fires_h", "false_self_rate", "time_to_confirm_s",
                   "alert_buzz_false_h", "label_leaks", "short_turn_line_pct", "after_laugh_lines",
                   "latency_p50_s", "merged_turn_pct", "spend_usd"}


def _rows() -> list[dict]:
    if not DB.exists():
        return []
    return [json.loads(l) for l in DB.read_text().splitlines() if l.strip()]


def add(a) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "agent": a.agent, "exp": a.exp, "change": a.change,
           "params": json.loads(a.params or "{}"), "split": a.split, "clips": a.clips,
           "baseline": a.baseline, "metrics": json.loads(a.metrics or "{}"), "verdict": a.verdict,
           "commit": a.commit, "notes": a.notes}
    with open(DB, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(row) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)
    print(f"added {a.exp}")


def show(a) -> None:
    rows = [r for r in _rows() if not a.agent or r["agent"] == a.agent][-a.last:]
    for r in rows:
        m = " ".join(f"{k}={v}" for k, v in r["metrics"].items())
        print(f'{r["exp"]:10} {r["verdict"]:8} [{r["split"]}/{r["clips"]}] {r["change"][:60]} | {m}')


def _cell(k, v, base):
    cls = ""
    if isinstance(v, (int, float)) and isinstance(base, (int, float)) and v != base:
        good = (v < base) if k in LOWER_IS_BETTER else (v > base)
        cls = "g" if good else "b"
    return f'<td class="{cls}">{html.escape(str(v))}</td>'


def render(_a=None) -> Path:
    rows = _rows()
    by_exp = {r["exp"]: r for r in rows}
    keys = []
    for r in rows:
        for k in r["metrics"]:
            if k not in keys:
                keys.append(k)
    head = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
    body = []
    for r in reversed(rows):
        base = by_exp.get(r.get("baseline") or "", {}).get("metrics", {})
        cells = "".join(_cell(k, r["metrics"].get(k, ""), base.get(k)) for k in keys)
        body.append(f'<tr class="v-{html.escape(r["verdict"])}"><td>{html.escape(r["exp"])}</td>'
                    f'<td>{html.escape(r["agent"])}</td><td>{html.escape(r["verdict"])}</td>'
                    f'<td class="ch">{html.escape(r["change"])}</td><td>{html.escape(r["split"])}/{html.escape(r["clips"])}</td>{cells}</tr>')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tuning Landscape</title><style>
:root{{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--line:#e7e5e4;--g:#bbf7d0;--b:#fecaca}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--line:#44403c;--g:#14532d;--b:#7f1d1d}}}}
:root[data-theme="dark"]{{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--line:#44403c;--g:#14532d;--b:#7f1d1d}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif}}
.wrap{{overflow-x:auto}}table{{border-collapse:collapse;font-size:12px}}td,th{{border-bottom:1px solid var(--line);padding:4px 6px;white-space:nowrap;text-align:right}}
td.ch{{text-align:left;white-space:normal;min-width:220px}}td.g{{background:var(--g)}}td.b{{background:var(--b)}}
tr.v-better td:nth-child(3){{color:#16a34a;font-weight:700}}tr.v-worse td:nth-child(3){{color:#dc2626;font-weight:700}}
</style></head><body><h1 style="font-size:17px">Tuning landscape</h1>
<p style="color:var(--mut)">{len(rows)} experiments · newest first · green/red = better/worse than that row's baseline</p>
<div class="wrap"><table><tr><th>exp</th><th>agent</th><th>verdict</th><th>change</th><th>split/clips</th>{head}</tr>
{''.join(body)}</table></div></body></html>"""
    out = ROOT / "landscape.html"
    ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    for f in ("agent", "exp", "change"):
        p.add_argument(f"--{f}", required=True)
    for f in ("params", "split", "clips", "baseline", "metrics", "commit", "notes"):
        p.add_argument(f"--{f}", default="")
    p.add_argument("--verdict", default="neutral", choices=["better", "worse", "neutral", "broken"])
    p.set_defaults(fn=add)
    s = sub.add_parser("show"); s.add_argument("--agent", default=""); s.add_argument("--last", type=int, default=30)
    s.set_defaults(fn=show)
    h = sub.add_parser("html"); h.set_defaults(fn=render)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
