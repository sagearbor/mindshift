"""Build the FROZEN nudge-judge case set (held-out sessions only) + SHA-256 manifest + leakage check.

Real cases: coach lines from tmp/recordings/judge/final-H{0,1,2}.jsonl (held-out sessions),
balanced 24 per cell of group {calm, heated} x answered_turn_truth_is_wearer {True, False}.
Synthetic cases (labels known BY CONSTRUCTION from the owner's written rules in RUBRIC_v4):
  adv_internal_label, adv_invented_name, adv_speaks_as_wearer, adv_praise_calm, adv_engagement_calm (4 each)
  silent_laughter, silent_nospeech (4 each)          -> expected helpful 0, should_be_silent true
  pos_regulate_heated (8)                              -> expected helpful 1 (owner preference, report-only)
Hindsight removed (context_after, alternatives, meta dropped).
Leakage check: ids cited in rubric/owner files and rubric-development grades are excluded, and every
case's normalised coach line is searched in those tuning sources (exact-text hash match).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from env import MAIN, OUT

REC = MAIN / "tmp/recordings"
J = REC / "judge"
SEED = 20261105
WT = Path(__file__).resolve().parents[2]
PR215 = "origin/feat/nudge-eval-prompt-v3"
EXISTING17 = MAIN / ".claude/worktrees/agent-a90ab94643480b5c9/tmp/nudge-eval/judge-baseline-all.json"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def git_show(path: str) -> str:
    return subprocess.run(["git", "-C", str(WT), "show", f"{PR215}:{path}"], capture_output=True, text=True,
                          check=True, timeout=30).stdout


def tuning_sources() -> tuple[set, str]:
    """(excluded ids, concatenated normalised tuning text)."""
    texts = [git_show("scripts/recreplay/RUBRIC_v4.md")]
    for f in ["RUBRIC_v3.md", "RUBRIC.md", "RUBRIC-notes.md", "owner-labels-20261011.md", "line-quality-findings.md",
              "owner-label-sheet.html"]:
        p = J / f
        if p.exists():
            texts.append(p.read_text(errors="ignore"))
    ids = set()
    for t in texts:
        ids |= set(re.findall(r"[a-z0-9]+_[A-Za-z0-9_]+#\d+", t))
    for f in [J / "v3-grades.jsonl", *sorted((J / "v3dev").glob("*.jsonl"))]:
        for line in open(f):
            try:
                ids.add(json.loads(line)["id"])
            except Exception:  # noqa: BLE001
                pass
    if EXISTING17.exists():
        d = json.loads(EXISTING17.read_text())
        for k, v in d["views"].items():
            ids.add(k)
            texts.append(v.get("coach_line", ""))
    return ids, " | ".join(norm(t) for t in texts)


def clean(view: dict) -> dict:
    v = {k: view[k] for k in ("setting", "relationship", "roles_basis", "answered_turn", "coach_line", "kind",
                               "arrived_at_s", "seconds_after_turn_end", "spoken") if k in view}
    arr = view.get("arrived_at_s", 1e9)
    ctx = []
    for t in view.get("context_before", []):
        if t.get("start", 0) >= arr:
            continue
        t = dict(t)
        if t.get("end", 0) > arr:
            t["still_speaking_at_arrival"] = True
            t["text"] = t["text"] + " ... (still speaking)"
        ctx.append(t)
    v["context_before"] = ctx
    v["context_note"] = "Shows only what had been said by arrived_at_s; nothing after it."
    return v


def load_pool():
    rows = []
    for b in ("final-H0", "final-H1", "final-H2"):
        for line in open(J / f"{b}.jsonl"):
            r = json.loads(line)
            rows.append({"src": f"{b}:{r['id']}", "id": r["id"], "item": r["item"], "meta": r["meta"],
                         "view": r["grader_view"]})
    return rows


SYN_LINES = {
    "adv_internal_label": "Let Speaker B finish before you jump in.",
    "adv_invented_name": "Let Maria finish her point about the budget first.",
    "adv_speaks_as_wearer": "I think we should just agree on the price and move on.",
    "adv_praise_calm": "Good, hold that tone. Nice work.",
    "adv_engagement_calm": "Sound engaged: ask them a detail about what they just said.",
    "silent_laughter": "Slow down. Let them finish.",
    "silent_nospeech": "Take a breath before your next sentence.",
    "pos_regulate_heated": "Take a breath. Say what you need from them.",
}


def main() -> None:
    held = set(json.loads((REC / "landscape/splits.json").read_text())["held_out"])
    excl_ids, tune_text = tuning_sources()
    pool = [r for r in load_pool() if r["item"] in held]
    n_pool = len(pool)
    pool = [r for r in pool if r["id"] not in excl_ids]
    seen, uniq = set(), []
    for r in pool:
        key = (r["item"], norm(r["view"]["coach_line"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    rng = np.random.default_rng(SEED)
    cells = defaultdict(list)
    for r in uniq:
        cells[(r["meta"]["group"], bool(r["meta"]["answered_turn_truth_is_wearer"]))].append(r)
    cases, used = [], set()
    for cell in sorted(cells):
        lst = cells[cell]
        pick = rng.choice(len(lst), size=min(24, len(lst)), replace=False)
        for i in sorted(pick.tolist()):
            r = lst[i]
            used.add(r["src"])
            cases.append({"case_id": f"real:{r['src']}", "kind": "real", "cell": f"{cell[0]}|wearer={cell[1]}",
                          "corpus": r["meta"]["corpus"], "expected_helpful": None, "view": clean(r["view"])})
    # synthetic, built on rows NOT used as real cases
    rest = [r for r in uniq if r["src"] not in used]
    calm = [r for r in rest if r["meta"]["group"] == "calm"]
    heated_w = [r for r in rest if r["meta"]["group"] == "heated" and r["meta"]["answered_turn_truth_is_wearer"]]
    anyr = rest

    def pick(lst, n):
        return [lst[i] for i in sorted(rng.choice(len(lst), size=n, replace=False).tolist())]

    plan = [("adv_internal_label", anyr, 4, 0), ("adv_invented_name", anyr, 4, 0), ("adv_speaks_as_wearer", anyr, 4, 0),
            ("adv_praise_calm", calm, 4, 0), ("adv_engagement_calm", calm, 4, 0), ("silent_laughter", anyr, 4, 0),
            ("silent_nospeech", anyr, 4, 0), ("pos_regulate_heated", heated_w, 8, 1)]
    for cat, lst, n, exp in plan:
        for r in pick(lst, n):
            v = clean(r["view"])
            v["coach_line"] = SYN_LINES[cat]
            v["spoken"] = True
            if cat == "silent_laughter":
                t0 = v["arrived_at_s"] - 12
                v["context_before"] = [
                    {"role": "other", "speaker": "S2", "start": t0, "end": t0 + 3, "text": "[laughter]", "backchannel": True},
                    {"role": "wearer", "speaker": "S1", "start": t0 + 3.5, "end": t0 + 5, "text": "[laughter] mm-hm", "backchannel": True},
                    {"role": "other", "speaker": "S2", "start": t0 + 6, "end": t0 + 7, "text": "yeah", "backchannel": True}]
                v["answered_turn"] = {"text": "yeah", "ended_at_s": t0 + 7, "role_per_phone": "other"}
            if cat == "silent_nospeech":
                v["context_before"] = []
                v["answered_turn"] = {"text": "(no speech detected in the last 30 s)", "ended_at_s": v["arrived_at_s"] - 30,
                                      "role_per_phone": "unknown"}
            cases.append({"case_id": f"syn:{cat}:{r['src']}", "kind": cat, "cell": cat, "corpus": r["meta"]["corpus"],
                          "expected_helpful": exp, "view": v})
    # leakage check + hashes
    leaks = []
    for c in cases:
        c["sha256"] = h(json.dumps(c["view"], sort_keys=True, ensure_ascii=False))
        line = norm(c["view"]["coach_line"])
        if c["kind"] == "real" and line and len(line) > 12 and line in tune_text:
            leaks.append(c["case_id"])
    cases = [c for c in cases if c["case_id"] not in leaks]
    rubric = git_show("scripts/recreplay/RUBRIC_v4.md")
    (OUT / "RUBRIC_v4.md").write_text(rubric)
    manifest = {
        "what": "frozen nudge-judge case set (held-out sessions only)", "seed": SEED,
        "rubric": {"source": f"{PR215}:scripts/recreplay/RUBRIC_v4.md", "sha256": h(rubric)},
        "pool_rows_heldout": n_pool, "excluded_ids_count": len(excl_ids),
        "leakage_text_matches_dropped": leaks, "n": len(cases),
        "by_kind": {k: sum(1 for c in cases if c["kind"] == k) for k in sorted({c["kind"] for c in cases})},
        "by_cell": {k: sum(1 for c in cases if c["cell"] == k) for k in sorted({c["cell"] for c in cases})},
        "set_sha256": h("".join(sorted(c["sha256"] for c in cases))),
        "cases": cases,
    }
    (OUT / "nudge_cases_frozen.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in manifest.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()
