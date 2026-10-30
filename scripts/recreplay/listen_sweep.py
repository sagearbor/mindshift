#!/usr/bin/env python3
"""listen_sweep.py — phone-only replays of inbox items with listening-stage
knobs (apps/mobile/src/live/replay/tuning.ts env vars), each experiment in
its OWN work root (never the shared tmp/recordings/work/<name>), reading the
shared reference-STT cache read-only and OFFLINE ($0: no Deepgram, no LLM).

    tmp/venv/bin/python scripts/recreplay/listen_sweep.py --exp D-001 \
        --env MINDSHIFT_VAD_AGC=1 ami_TS3004a sbcsae_SBC033 ...

Prints one JSON line per item and a summary line; metrics:
  vad_coverage / vad_precision   A's time-based scorer (score.turn_coverage)
  merged_turn_pct   % phone turns holding >= 2 ground-truth voices (>= 0.5 s each)
  frag_pct          % ground-truth single-voice segments >= 2 s cut across >= 2 phone turns
  turns_per_min     phone turns per minute of ground-truth speech
  wearer_recall     score.identity.phone wearer recall (per turn)
  purity / self_time_recall / self_time_precision   time-weighted (time_identity)
Re-score saved runs without replaying: --rescore D-000 D-001 ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
for p in (REPO / "server", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from recreplay import pipeline  # noqa: E402
from recreplay import score as score_mod  # noqa: E402

MERGE_MIN_S = 0.5
FRAG_MIN_SEG_S = 2.0
FRAG_MIN_OV_S = 0.3


def merged_turn_pct(sent: list[dict], segs: list[dict]) -> float | None:
    real = [s for s in segs if not s.get("is_backchannel")]
    if not sent or not real:
        return None
    merged = 0
    for t in sent:
        a, b = float(t["start_time"]), float(t["end_time"])
        per: Counter = Counter()
        for s in real:
            ov = min(b, s["end"]) - max(a, s["start"])
            if ov > 0:
                per[s["label"]] += ov
        if sum(1 for v in per.values() if v >= MERGE_MIN_S) >= 2:
            merged += 1
    return 100.0 * merged / len(sent)


def frag_pct(sent: list[dict], segs: list[dict]) -> float | None:
    long = [s for s in segs if not s.get("is_backchannel") and s["end"] - s["start"] >= FRAG_MIN_SEG_S]
    if not long:
        return None
    n = 0
    for s in long:
        k = sum(1 for t in sent
                if min(s["end"], float(t["end_time"])) - max(s["start"], float(t["start_time"])) >= FRAG_MIN_OV_S)
        if k >= 2:
            n += 1
    return 100.0 * n / len(long)


def time_identity(sent: list[dict], segs: list[dict]) -> dict:
    """Time-weighted, so splitting a turn cannot game a per-turn count:
    ``purity`` — share of GT speech inside phone turns that belongs to each
    turn's majority voice; ``self_time_recall`` — share of the wearer's GT
    speech inside turns the phone called self; ``self_time_precision`` —
    share of GT speech in self-called turns that is the wearer's."""
    real = [s for s in segs if not s.get("is_backchannel")]
    tot = maj = 0.0
    self_w = self_all = 0.0
    for t in sent:
        a, b = float(t["start_time"]), float(t["end_time"])
        per: Counter = Counter()
        w = 0.0
        for s in real:
            ov = min(b, s["end"]) - max(a, s["start"])
            if ov > 0:
                per[s["label"]] += ov
                if s.get("wearer"):
                    w += ov
        if not per:
            continue
        tot += sum(per.values())
        maj += max(per.values())
        if t.get("is_self") is True:
            self_w += w
            self_all += sum(per.values())
    wearer_total = sum(s["end"] - s["start"] for s in real if s.get("wearer"))
    return {"purity": (maj / tot) if tot else None,
            "self_time_recall": (self_w / wearer_total) if wearer_total else None,
            "self_time_precision": (self_w / self_all) if self_all else None}


def item_metrics(bundle: dict) -> dict:
    segs, src = score_mod.truth_segments(bundle)
    sent = (bundle.get("phone") or {}).get("sent") or []
    tc = score_mod.turn_coverage(bundle, segs, src)
    gt = [s for s in segs if not s.get("is_backchannel")] if src.startswith("annotation") else []
    speech_min = sum(s["end"] - s["start"] for s in gt) / 60.0
    ident = ((bundle.get("score") or {}).get("identity") or {}).get("phone") or {}
    return {
        "truth": src, "phone_turns": len(sent),
        "vad_coverage": tc.get("vad_coverage"), "vad_precision": tc.get("vad_precision"),
        "merged_turn_pct": merged_turn_pct(sent, gt) if gt else None,
        "frag_pct": frag_pct(sent, gt) if gt else None,
        "turns_per_min": (len(sent) / speech_min) if speech_min else None,
        "wearer_recall": ident.get("wearer_recall"),
        **(time_identity(sent, gt) if gt else {"purity": None, "self_time_recall": None, "self_time_precision": None}),
    }


def run_item(name: str, exp_root: Path) -> dict:
    folder = pipeline.RECORDINGS / "inbox" / name
    inp = pipeline.inputs_from_inbox(folder, work_root=exp_root)
    shared = pipeline.RECORDINGS / "work" / name / inp.deepgram_cache.name
    inp.deepgram_cache = shared            # read-only reference transcript
    inp.offline = True                     # never a paid call
    opts = pipeline.RunOptions(skip_server=True, skip_ceiling=True)
    bundle = pipeline.run(inp, opts)
    m = item_metrics(bundle)
    (inp.work / "audio16k.wav").unlink(missing_ok=True)   # disk: keep run.json/phone.json only
    return m


def summarize(rows: list[dict]) -> dict:
    out = {}
    for k in ("vad_coverage", "vad_precision", "merged_turn_pct", "frag_pct", "turns_per_min", "wearer_recall",
              "purity", "self_time_recall", "self_time_precision"):
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        out[k] = round(sum(vals) / len(vals), 4) if vals else None
    out["phone_turns"] = sum(r.get("phone_turns") or 0 for r in rows)
    out["items"] = len(rows)
    return out


def rescore(exps: list[str]) -> None:
    for exp in exps:
        rows = []
        for rj in sorted((pipeline.RECORDINGS / "work_D" / exp).glob("*/run.json")):
            m = item_metrics(json.loads(rj.read_text()))
            m["name"] = rj.parent.name
            rows.append(m)
        s = summarize(rows)
        s["exp"] = exp
        print("[listen-sweep] RESCORE " + json.dumps(s), flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="*")
    ap.add_argument("--exp")
    ap.add_argument("--rescore", nargs="+")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the phone replay (tuning.ts)")
    a = ap.parse_args(argv)
    if a.rescore:
        rescore(a.rescore)
        return 0
    if not a.exp or not a.items:
        ap.error("--exp and items are required (or --rescore)")
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    os.environ.setdefault("MINDSHIFT_SPEND_LEDGER_ON", "1")
    exp_root = pipeline.RECORDINGS / "work_D" / a.exp
    if exp_root.exists():
        shutil.rmtree(exp_root)            # this experiment's own scratch only
    rows = []
    for n in a.items:
        try:
            m = run_item(n, exp_root)
        except Exception as exc:  # noqa: BLE001
            m = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        m["name"] = n
        rows.append(m)
        print("[listen-sweep] " + json.dumps(m), flush=True)
    s = summarize([r for r in rows if "error" not in r])
    s.update({"exp": a.exp, "env": a.env})
    (exp_root / "summary.json").write_text(json.dumps({"summary": s, "items": rows}, indent=1))
    print("[listen-sweep] SUMMARY " + json.dumps(s), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
