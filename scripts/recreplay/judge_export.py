#!/usr/bin/env python3
"""judge_export.py — one JSONL row per coach line, for LLM graders (Haiku)
scoring against tmp/recordings/judge/RUBRIC.md.

    tmp/venv/bin/python scripts/recreplay/judge_export.py tmp/recordings/work/ami_TS3004a ... --batch dev-L001
    tmp/venv/bin/python scripts/recreplay/judge_export.py --split dev --batch dev-L001
    tmp/venv/bin/python scripts/recreplay/judge_export.py --split clip30_t1 --batch clip30_t1-A-C001
    -> tmp/recordings/judge/<batch>.jsonl

Each row:
  id, batch, item, line_index
  grader_view   EVERYTHING the grader may see, nothing else:
                setting, relationship, the 6 conversation turns before the
                answered turn ends and the 2 after (role "wearer" | "other",
                speaker id, start/end s, text, backchannel), the answered
                turn, the coach line (+ its alternatives), when it arrived,
                whether it was spoken aloud, and its kind (nudge = about the
                wearer's own turn, response = what the wearer could say next)
  meta          for analysis only, NEVER shown to the grader (it would bias
                the grade): corpus, group (heated/calm), importance, latency,
                the scorer's nearest ground-truth moment, truth of who spoke
                the answered turn, the phone's is_self verdict

Turns come from the ground-truth annotation (consecutive segments of one
speaker merged into a turn), else Deepgram's turns (CONFER: Greek, no
speaker truth; the roles there are the replay's own wearer guess and are
flagged ``roles_basis``). Re-scores each run with the current scorer first.
Phone haptic buzzes have no words and are not exported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from recreplay import corpus_summary as cs  # noqa: E402
from recreplay import score as score_mod  # noqa: E402

RECORDINGS = HERE.parents[2] / "tmp" / "recordings"
OUT = RECORDINGS / "judge"
BEFORE, AFTER = 6, 2
MERGE_GAP_S = 1.0


def conversation_turns(bundle: dict) -> tuple[list[dict], str]:
    """[{role, speaker, start, end, text, backchannel}] in time order."""
    segs, src = score_mod.truth_segments(bundle)
    basis = "ground-truth annotation" if src.startswith("annotation") else "deepgram + replay wearer guess"
    turns: list[dict] = []
    for s in sorted(segs, key=lambda s: s["start"]):
        bc = bool(s.get("is_backchannel"))
        cur = turns[-1] if turns else None
        # merge only a direct continuation: same speaker, short gap, nobody else started in between
        if (cur and not bc and not cur["backchannel"] and cur["speaker"] == s["label"]
                and s["start"] - cur["end"] <= MERGE_GAP_S):
            cur["end"] = max(cur["end"], s["end"])
            cur["text"] = (cur["text"] + " " + (s.get("text") or "")).strip()
            continue
        turns.append({"role": "wearer" if s["wearer"] else "other", "speaker": s["label"],
                      "start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2),
                      "text": (s.get("text") or "").strip(), "backchannel": bc})
    for t in turns:
        t["start"], t["end"] = round(t["start"], 2), round(t["end"], 2)
    return turns, basis


def _context(turns: list[dict], anchor_end: float) -> tuple[list[dict], list[dict]]:
    """The last BEFORE turns that started before the answered turn ended
    (a turn still running then is included whole) and the next AFTER turns
    that start after it."""
    before = [t for t in turns if t["start"] < anchor_end]
    after = [t for t in turns if t["start"] >= anchor_end]
    return before[-BEFORE:], after[:AFTER]


def rows_for(bundle: dict, prov: dict | None, batch: str) -> list[dict]:
    cs.rescore(bundle)
    sc = bundle["score"]
    turns, basis = conversation_turns(bundle)
    mom_items = sc["moments"]["items"]
    sent = {t.get("turn_uid"): t for t in (bundle.get("phone") or {}).get("sent") or []}
    segs, _ = score_mod.truth_segments(bundle)
    out = []
    settings = bundle.get("settings") or {}
    n = 0
    for ln in sc["lines"]:
        if ln["source"] != "server" or ln["kind"] not in ("response", "nudge", "room_answer", "room_card") or not ln.get("text"):
            continue
        anchor_end = ln.get("turn_end_s") if ln.get("turn_end_s") is not None else ln["at_s"]
        before, after = _context(turns, float(anchor_end))
        phone_turn = sent.get(ln.get("turn_uid")) or {}
        truth_w = None
        if phone_turn:
            truth_w = score_mod.wearer_truth(segs, float(phone_turn["start_time"]), float(phone_turn["end_time"]))
        near = min(mom_items, key=lambda it: abs(ln["at_s"] - it["anchor_t"])) if mom_items else None
        out.append({
            "id": f"{bundle.get('name')}#{n}", "batch": batch, "item": bundle.get("name"), "line_index": n,
            "grader_view": {
                "setting": settings.get("session_context"),
                "relationship": settings.get("relationship"),
                "roles_basis": basis,
                "context_before": before,
                "answered_turn": {"text": ln.get("utterance_text"), "ended_at_s": anchor_end,
                                  "role_per_phone": "wearer" if ln.get("kind") == "nudge" else "other"},
                "context_after": after,
                "coach_line": ln["text"],
                "alternatives": [x for x in (ln.get("all") or [])[1:] if x],
                "kind": ln["kind"],
                "arrived_at_s": ln["at_s"],
                "seconds_after_turn_end": None if ln.get("latency_ms") is None else round(ln["latency_ms"] / 1000.0, 2),
                "spoken": bool(ln.get("fires")),
            },
            "meta": {
                "corpus": (prov or {}).get("corpus"), "group": (prov or {}).get("group"),
                "importance": ln.get("importance"), "latency_ms": ln.get("latency_ms"),
                "answered_turn_truth_is_wearer": truth_w, "phone_is_self": phone_turn.get("is_self"),
                "nearest_moment": None if near is None else {
                    "anchor_t": near["anchor_t"], "rule": near.get("rule"), "delta_s": round(ln["at_s"] - near["anchor_t"], 2),
                    "in_window": near["anchor_t"] - sc["moments"]["pre_s"] <= ln["at_s"] <= near["anchor_t"] + sc["moments"]["window_s"]},
                "scorer": {"anchor": sc["moments"]["anchor"], "window_s": sc["moments"]["window_s"]},
            },
        })
        n += 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path, help="run dirs (tmp/recordings/work/<name>) holding run.json")
    ap.add_argument("--split", help="dev | all | clip30_t1 | clip30_t2 | clip30_t3 | clip30_all (never held_out: graders tune too)")
    ap.add_argument("--batch", default=None, help="output name (default: export-<timestamp>)")
    a = ap.parse_args(argv)
    if a.split in ("held_out", "heldout", "held-out"):
        ap.error("held-out runs are not exported for judging during tuning")
    batch = a.batch or time.strftime("export-%Y%m%d-%H%M")
    pairs: list[tuple[dict, dict | None]] = []
    if a.split:
        pairs += cs.load_bundles(RECORDINGS / "work", RECORDINGS / "inbox", cs.split_names(a.split))
    for d in a.runs:
        run = Path(d) / "run.json"
        b = json.loads(run.read_text())
        prov_path = RECORDINGS / "inbox" / run.parent.name / f"{run.parent.name}.corpus.json"
        pairs.append((b, json.loads(prov_path.read_text()) if prov_path.exists() else None))
    if not pairs:
        ap.error("no runs: pass run dirs or --split")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{batch}.jsonl"
    rows = [r for b, prov in pairs for r in rows_for(b, prov, batch)]
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} coach lines from {len(pairs)} runs ({sum(1 for r in rows if r['grader_view']['spoken'])} spoken)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
