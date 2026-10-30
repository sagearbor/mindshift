#!/usr/bin/env python3
"""verifier_eval.py — run the speak verifier (server/speak_verifier.py) in
ISOLATION over exported coach lines (judge_export.py batches) and compare its
yes/no with Sonnet reference labels (``should_have_stayed_silent``).

  tmp/venv/bin/python scripts/recreplay/verifier_eval.py --variant v2 \
      --batch tmp/recordings/judge/dev-L001.jsonl \
      --labels tmp/recordings/judge/batches/dev-L001.recheck40.S.jsonl \
      [--labels-only] [--out verdicts.jsonl]

The verifier sees the grader_view's ``context_before`` turns (with their
roles) and the coach line — the same shape the live server builds from its
own turns. Spend goes through the shared LLM cache + spend ledger
(MINDSHIFT_SPEND_LEDGER_ON=1, MINDSHIFT_LLM_CACHE_DIR); a rerun is free.
Metrics vs Sonnet: agreement (verifier speak == not should_have_stayed_silent),
false_speak (verifier speaks where Sonnet says silent, share of those rows),
speak_recall (verifier speaks on Sonnet's should-speak rows).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[2] / "server")]

import speak_verifier as sv  # noqa: E402


def turns_from_view(gv: dict) -> list[sv.Turn]:
    """What the live server would have had when the coach answered: the
    context turns up to the answered turn's end. A truth turn still running
    then is replaced by the PHONE's answered-turn text (the live verifier
    never sees words spoken after the line was generated)."""
    ans = gv.get("answered_turn") or {}
    cut = float(ans.get("ended_at_s") or 1e9)
    out = []
    for t in gv.get("context_before") or []:
        start, end = float(t.get("start") or 0.0), float(t.get("end") or 0.0)
        if start > cut + 0.25:
            continue
        text = t.get("text") or ""
        if end > cut + 0.5 and ans.get("text"):
            text, end = ans["text"], cut
        out.append(sv.Turn(t.get("role") or "other", str(t.get("speaker") or "?"), start, end, text))
    return out


def user_for_row(row: dict) -> str:
    gv = row["grader_view"]
    return sv.build_user(turns_from_view(gv), gv["coach_line"], kind=gv.get("kind") or "response",
                         setting=gv.get("setting"), relationship=gv.get("relationship"))


def compare(verdicts: dict[str, bool], labels: dict[str, bool]) -> dict:
    """labels: id -> should_have_stayed_silent."""
    ids = [i for i in labels if i in verdicts]
    silent = [i for i in ids if labels[i]]
    speak = [i for i in ids if not labels[i]]
    agree = sum(verdicts[i] == (not labels[i]) for i in ids)
    return {
        "n": len(ids),
        "agreement": round(agree / len(ids), 3) if ids else None,
        "false_speak": round(sum(verdicts[i] for i in silent) / len(silent), 3) if silent else None,
        "speak_recall": round(sum(verdicts[i] for i in speak) / len(speak), 3) if speak else None,
        "n_should_speak": len(speak),
        "verifier_speak": sum(verdicts[i] for i in ids),
    }


def make_llm():
    from llm_cache import LLMResponseCache
    from llm_client import LLMClient
    cache = Path(os.environ.get("MINDSHIFT_LLM_CACHE_DIR")
                 or HERE.parents[2] / "tmp" / "recordings" / "llm_cache_shared")
    model = os.environ.get("MINDSHIFT_MODEL", "claude-haiku-4-5-20251001")
    return LLMResponseCache(LLMClient(model), cache_dir=cache)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=sv.DEFAULT_VARIANT, choices=sorted(sv.VARIANTS))
    ap.add_argument("--batch", action="append", required=True,
                    help="BATCH.jsonl or BATCH.jsonl:LABELS.jsonl (labels refer to that batch's ids)")
    ap.add_argument("--labels-only", action="store_true", help="only rows that have a reference label")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    # Line ids ("ami_TS3004a#12") repeat across batches, so key by batch too.
    rows: list[dict] = []
    labels: dict[str, bool] = {}
    for spec in a.batch:
        bpath, _, lpath = spec.partition(":")
        bname = Path(bpath).stem
        for l in Path(bpath).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                r["id"] = f"{bname}/{r['id']}"
                rows.append(r)
        if lpath:
            for l in Path(lpath).read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    if "should_have_stayed_silent" in r:
                        labels[f"{bname}/{r['id']}"] = bool(r["should_have_stayed_silent"])
    if a.labels_only:
        rows = [r for r in rows if r["id"] in labels]
    seen, uniq = set(), []
    for r in rows:
        if r["id"] not in seen and "grader_view" in r:
            seen.add(r["id"])
            uniq.append(r)

    llm = make_llm()

    def one(r):
        v = sv.verify_sync(llm, user_for_row(r), a.variant)
        return r["id"], v

    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(one, uniq))
    verdicts = {i: v.speak for i, v in results}
    errors = sum(1 for _, v in results if v.error)
    out = {"variant": a.variant, "rows": len(results), "verifier_speak_all": sum(verdicts.values()),
           "errors": errors, "vs_sonnet": compare(verdicts, labels), "cache": llm.stats}
    if a.out:
        with open(a.out, "w") as fh:
            for i, v in results:
                fh.write(json.dumps({"id": i, "speak": v.speak, "reason": v.reason, "error": v.error,
                                     "sonnet_silent": labels.get(i)}, ensure_ascii=False) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
