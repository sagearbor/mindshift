#!/usr/bin/env python3
"""regate.py — POST-HOC speak-gate sweeps over recorded corpus runs, at $0
(no server, no LLM, no STT): take every fire a recorded run produced, apply a
candidate speak gate, and re-score with the turn-end-anchored scorer.

    tmp/venv/bin/python scripts/recreplay/regate.py --split dev                 # grid on DEV, print the Pareto front
    tmp/venv/bin/python scripts/recreplay/regate.py --split dev --landscape     # ... and add rows to the landscape
    tmp/venv/bin/python scripts/recreplay/regate.py --split dev --gate '{"min_words":3,"importance":60,"min_gap_s":20}'

Gate knobs (each one only ever REMOVES server fires; phone alert buzzes pass
through untouched):
  min_words      the answered turn has at least this many words (\\w+, so Greek counts)
  skip_bc        skip turns that are only a backchannel token ("yeah", "mhm right")
  laugh_s        skip a line landing within N s after a laughter mark (corpus truth:
                 a live gate would need a laughter detector, so this is an upper bound)
  importance     the line's importance must be >= T
  unknown_cap    when the phone has not decided whether the answered turn is the
                 wearer's (is_self None), cap its importance at this value first
  min_gap_s      at least G s between two SPOKEN server lines (greedy, in time order)
  max_per_5min   at most K spoken server lines in any rolling 300 s

CAVEAT: post-hoc gating ignores cascade effects. In a live session a line the
gate silences still enters (or would leave) the coach's prompt history, the
server's own cooldowns react to what was spoken, and the next lines'
importance can change; this sweep holds every recorded line fixed. Treat its
Pareto front as a shortlist to confirm with real replays, not as the result.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from recreplay import corpus_summary as cs  # noqa: E402
from recreplay import score as score_mod  # noqa: E402
from recreplay.corpora import is_backchannel_text  # noqa: E402

RECORDINGS = HERE.parents[2] / "tmp" / "recordings"
DEFAULT_GATE = {"min_words": 0, "skip_bc": False, "laugh_s": 0.0, "importance": 0, "unknown_cap": None,
                "min_gap_s": 0.0, "max_per_5min": None}
GRID = {
    "min_words": [0, 3],
    "skip_bc": [False, True],
    "laugh_s": [0.0, 6.0],
    "importance": [0, 50, 60, 70, 75, 80, 85],
    "unknown_cap": [None, 45, 55, 65],
    "min_gap_s": [0.0, 15.0, 30.0, 45.0, 60.0],
    "max_per_5min": [None, 4],
}


def _words(text: str | None) -> int:
    return len(re.findall(r"\w+", text or ""))


def prepare(bundle: dict) -> dict:
    """Per-bundle facts the gates need, computed once."""
    sent = {t.get("turn_uid"): t for t in (bundle.get("phone") or {}).get("sent") or []}
    gt = cs._gt(bundle)
    laughs = sorted(float(e["t"]) for e in gt.get("events") or [] if e.get("type") == "laughter")
    return {"sent": sent, "laughs": laughs}


def apply_gate(lines: list[dict], facts: dict, gate: dict) -> list[dict]:
    """A copy of ``lines`` with ``fires`` recomputed under ``gate``."""
    g = {**DEFAULT_GATE, **gate}
    out = copy.deepcopy(lines)
    spoken: list[float] = []
    for ln in out:
        if not ln.get("fires") or ln.get("source") != "server" or ln.get("kind") not in ("response", "nudge"):
            continue
        utt = ln.get("utterance_text") or ""
        keep = True
        if g["min_words"] and _words(utt) < g["min_words"]:
            keep = False
        elif g["skip_bc"] and is_backchannel_text(utt):
            keep = False
        elif g["laugh_s"] and any(0.0 <= ln["at_s"] - t <= g["laugh_s"] for t in facts["laughs"]):
            keep = False
        else:
            imp = float(ln.get("importance") if ln.get("importance") is not None else 100)
            if g["unknown_cap"] is not None:
                turn = facts["sent"].get(ln.get("turn_uid")) or {}
                if turn.get("is_self") is None:
                    imp = min(imp, float(g["unknown_cap"]))
            if imp < g["importance"]:
                keep = False
            elif g["min_gap_s"] and spoken and ln["at_s"] - spoken[-1] < g["min_gap_s"]:
                keep = False
            elif g["max_per_5min"] is not None and sum(1 for t in spoken if ln["at_s"] - t < 300.0) >= g["max_per_5min"]:
                keep = False
        ln["fires"] = keep
        ln["speak"] = keep
        if keep:
            spoken.append(ln["at_s"])
    return out


def evaluate(prepared: list[tuple[dict, dict | None, dict]], gate: dict) -> dict:
    items = []
    for bundle, prov, facts in prepared:
        sc = bundle["score"]
        lines = apply_gate(sc["lines"], facts, gate)
        mom = score_mod.moments(bundle, lines, sc["moments"]["window_s"], pre_s=sc["moments"]["pre_s"],
                                anchor=sc["moments"]["anchor"])
        b2 = {**bundle, "score": {**sc, "lines": lines, "moments": mom}}
        items.append(cs.item_metrics(b2, prov))
    m = cs.landscape_metrics(items)
    m["spoken_lines"] = sum(i["server_spoken"] for i in items)
    return m


def load(split: str) -> list[tuple[dict, dict | None, dict]]:
    pairs = cs.load_bundles(RECORDINGS / "work", RECORDINGS / "inbox", cs.split_names(split))
    return [(cs.rescore(b), prov, prepare(b)) for b, prov in pairs]


def _knobs(g: dict) -> int:
    return sum(1 for k, v in g.items() if DEFAULT_GATE.get(k) != v)


def pareto(rows: list[dict], x: str = "calm_fires_h", y: str = "lift") -> list[dict]:
    """Rows not dominated on (lower x, higher y). Gates with identical
    outcomes collapse to the one with the fewest knobs switched on."""
    best: dict = {}
    for r in rows:
        m = r["metrics"]
        if m.get(x) is None or m.get(y) is None:
            continue
        key = (m.get(x), m.get(y), m.get("heated_fires_h"), m.get("nomoment_fires_h"), m.get("hits"))
        if key not in best or _knobs(r["gate"]) < _knobs(best[key]["gate"]):
            best[key] = r
    ok = list(best.values())
    front = []
    for r in ok:
        rx, ry = r["metrics"][x], r["metrics"][y]
        if not any((o["metrics"][x] <= rx and o["metrics"][y] >= ry) and (o["metrics"][x] < rx or o["metrics"][y] > ry)
                   for o in ok):
            front.append(r)
    return sorted(front, key=lambda r: r["metrics"][x])


def grid_points(grid: dict = GRID):
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def _short(g: dict) -> str:
    d = {k: v for k, v in g.items() if DEFAULT_GATE.get(k) != v}
    return ", ".join(f"{k}={v}" for k, v in d.items()) or "no gate (recorded fires)"


def _landscape_add(exp: str, change: str, params: dict, split: str, metrics: dict, verdict: str, notes: str, commit: str) -> None:
    keep = ("hit_rate", "chance_hit_rate", "lift", "calm_fires_h", "heated_fires_h", "nomoment_fires_h",
            "short_turn_line_pct", "after_laugh_lines", "label_leaks")
    m = {k: metrics[k] for k in keep if k in metrics}
    m["spend_usd"] = 0
    subprocess.run([sys.executable, str(HERE.parent / "landscape.py"), "add", "--agent", "A", "--exp", exp, "--change", change,
                    "--params", json.dumps(params), "--split", split, "--clips", "corpus30-posthoc", "--baseline", "A-G000",
                    "--metrics", json.dumps(m), "--verdict", verdict, "--notes", notes, "--commit", commit], check=True,
                   stdout=subprocess.DEVNULL)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "held_out", "all"])
    ap.add_argument("--gate", help="evaluate one gate (JSON) instead of the grid")
    ap.add_argument("--landscape", action="store_true", help="add the Pareto front + representative points to the landscape")
    ap.add_argument("--out", type=Path, default=RECORDINGS / "landscape" / "regate-dev.json")
    ap.add_argument("--commit", default="")
    a = ap.parse_args(argv)
    if a.landscape and a.split != "dev":
        ap.error("landscape rows are tuning results: DEV only")
    prepared = load(a.split)
    if a.gate:
        g = {**DEFAULT_GATE, **json.loads(a.gate)}
        print(json.dumps({"gate": g, "metrics": evaluate(prepared, g)}))
        return 0
    rows = []
    for n, g in enumerate(grid_points()):
        rows.append({"gate": g, "metrics": evaluate(prepared, g)})
        if n % 200 == 0:
            print(f"[regate] {n} points", file=sys.stderr, flush=True)
    base = next(r for r in rows if r["gate"] == DEFAULT_GATE)
    front = pareto(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"split": a.split, "grid": GRID, "baseline": base, "front": front, "rows": rows}, indent=0))
    print(f"baseline: {json.dumps(base['metrics'])}")
    print(f"Pareto front (calm_fires_h vs lift), {len(front)} of {len(rows)} points:")
    for r in front:
        m = r["metrics"]
        print(f"  calm {m.get('calm_fires_h'):6} /h  lift {m.get('lift'):+.3f}  hit {m.get('hit_rate')}  chance {m.get('chance_hit_rate')}"
              f"  heated {m.get('heated_fires_h')}/h  nomoment {m.get('nomoment_fires_h')}/h  | {_short(r['gate'])}")
    if a.landscape:
        caveat = "post-hoc gate over recorded fires ($0); ignores prompt-history/cooldown cascade effects."
        _landscape_add("A-G000", "regate baseline: recorded fires, no extra gate (DEV)", base["gate"], "dev",
                       base["metrics"], "neutral", caveat, a.commit)
        reps = []
        for k, vals in GRID.items():                      # one knob at a time, the rest off
            for v in vals:
                if DEFAULT_GATE[k] != v:
                    reps.append({**DEFAULT_GATE, k: v})
        seen = set()
        n = 0
        for tag, gates in (("front", [r["gate"] for r in front]), ("single", reps)):
            for g in gates:
                key = json.dumps(g, sort_keys=True)
                if key in seen or g == DEFAULT_GATE:
                    continue
                seen.add(key)
                n += 1
                r = next(x for x in rows if x["gate"] == g)
                bm = base["metrics"]
                better = (r["metrics"].get("lift", -9) >= bm.get("lift", -9)
                          and (r["metrics"].get("calm_fires_h") or 0) < (bm.get("calm_fires_h") or 0))
                _landscape_add(f"A-G{n:03d}", f"regate[{tag}]: {_short(g)}", g, "dev", r["metrics"],
                               "better" if better else "neutral", caveat, a.commit)
        print(f"[regate] {n + 1} landscape rows added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
