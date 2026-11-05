#!/usr/bin/env python3
"""Real-LLM measurement of the quiet coach's spoken lines (plan 2026-11-05, Steps 1-2).

    gen    replay each fixture's phone turns through the trigger gate (the same
           server functions the live pipeline runs), make the REAL coach LLM
           call for every triggered turn with the production (baseline) or the
           v3 prompt, apply the policy gates, and record the lines.
    judge  grade every SPOKEN line, blind, no hindsight (context clipped to the
           moment the line arrived), with two judges of different lineages
           under scripts/recreplay/RUBRIC_v4.md.
    score  U_2/h, helpful share, leaks, calls/h with session-bootstrap CIs
           (nudge_quality.py); paired against a baseline file for v3.

    tmp/venv/bin/python scripts/recreplay/quiet_coach_live_eval.py gen --prompt baseline --split dev \
        --out tmp/nudge-eval/gen-baseline-dev.json
    ... judge --lines tmp/nudge-eval/gen-baseline-dev.json --out tmp/nudge-eval/judge-baseline-dev.json
    ... score --split dev --lines tmp/nudge-eval/gen-baseline-dev.json --judgments tmp/nudge-eval/judge-baseline-dev.json

Honest limits (printed into every report): the session history the prompt
sees is rebuilt from the replay's phone turns (the coach's own earlier lines
are not fed back); phone-reported triggers (mid-turn floor cue) are not
replayed (tsx is not installed), so the trigger set is the server's own
(heat lexicon, wearer floor clock, direct question); the judges are text-only.
Every call goes through server/llm_client.py behind the on-disk
llm_cache (keyed by model + full prompt), so re-grades are free.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
WT = HERE.parents[2]
sys.path.insert(0, str(WT / "server"))
sys.path.insert(0, str(WT / "scripts"))
sys.path.insert(0, str(HERE.parent))

MAIN_REPO = Path(subprocess.run(
    ["git", "-C", str(WT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
    capture_output=True, text=True).stdout.strip() or str(WT / ".git")).parent

REC = Path(os.environ.get("MINDSHIFT_RECORDINGS") or (WT / "tmp" / "recordings"))
if not REC.exists():
    REC = MAIN_REPO / "tmp" / "recordings"
EVAL_DIR = WT / "tmp" / "nudge-eval"
CACHE_DIR = EVAL_DIR / "llm_cache"

ARRIVAL_AFTER_TURN_S = 1.5           # the replay's measured speech-end -> line (p50 ~1.5 s)
SPEAK_MIN, UNKNOWN_CAP, GAP_S, MIN_WORDS = 80, 72, 60.0, 3
NUDGE_MIN = 40
JUDGES = {"A": "claude-sonnet-5-5", "B": "gpt-5-mini"}
PRICE = {  # USD per token (in, out); pessimistic for reasoning models
    "claude-haiku-4-5-20251001": (1e-6, 5e-6),
    "claude-sonnet-5-5": (2e-6, 10e-6),
    "gpt-5-mini": (0.25e-6, 2e-6),
}
SPEND_CAP_USD = float(os.environ.get("NUDGE_EVAL_CAP_USD", "1.80"))


def load_env() -> None:
    envp = MAIN_REPO / ".env"
    if not envp.exists():
        return
    for line in envp.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


# ---------------------------------------------------------------------------
# metered, cached LLM client
# ---------------------------------------------------------------------------


class Metered:
    """LLMClient wrapper with a hard spend cap shared via a JSON ledger."""

    def __init__(self, model: str):
        import llm_client
        self._inner = llm_client.LLMClient(model)
        self.model = model

    def _ledger(self) -> Path:
        return EVAL_DIR / "spend.json"

    def complete(self, system, user, temperature=0.7, max_tokens=512, response_schema=None):
        pin, pout = PRICE.get(self.model, (2e-6, 10e-6))
        est_out = max_tokens if "gpt-5" in self.model else min(max_tokens, 400)
        est = (len(str(system)) + len(str(user))) / 3.0 * pin + est_out * pout
        p = self._ledger()
        led = json.loads(p.read_text()) if p.exists() else {"total_usd": 0.0, "calls": 0, "by_model": {}}
        if led["total_usd"] + est > SPEND_CAP_USD:
            raise RuntimeError(f"spend cap ${SPEND_CAP_USD} would be exceeded (at ${led['total_usd']:.3f})")
        if self.model.startswith("gpt-5"):
            # gpt-5 family: no temperature (llm_client only drops it for "-reasoning" names)
            out = self._inner._complete_openai_responses(system, user, None, max_tokens)
        else:
            out = self._inner.complete(system=system, user=user, temperature=temperature,
                                       max_tokens=max_tokens, response_schema=response_schema)
        led["total_usd"] += est
        led["calls"] += 1
        led["by_model"][self.model] = led["by_model"].get(self.model, 0.0) + est
        p.write_text(json.dumps(led, indent=1))
        return out

    def close(self):
        pass


def cached_client(model: str):
    import llm_cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return llm_cache.LLMResponseCache(Metered(model), cache_dir=CACHE_DIR)


# ---------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------


def words(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def group_of(it: str) -> str:
    try:
        return json.load(open(REC / "inbox" / it / f"{it}.corpus.json")).get("group") or "heated"
    except Exception:
        return "heated"


def load_items(split: str) -> list[dict]:
    from quiet_sim import simulate
    splits = json.load(open(REC / "landscape" / "splits.json"))
    names = splits["dev"] if split == "dev" else splits["held_out"] if split == "held_out" else splits["dev"] + splits["held_out"]
    out = []
    for it in names:
        h1 = REC / "ot" / "X" / "X-H1" / it / "run.json"
        h0 = REC / "work" / it / "run.json"
        p = h1 if (it in splits["held_out"] and h1.exists()) else h0
        if not p.exists():
            continue
        b = json.load(open(p))
        if not b.get("phone"):
            print(f"skip {it}: its recorded phone replay failed upstream", file=sys.stderr)
            continue
        sent = b["phone"]["sent"]
        out.append({"item": it, "split": "held_out" if it in splits["held_out"] else "dev", "group": group_of(it),
                    "bundle": b, "sent": sent, "dur_s": float(b["audio"]["duration_s"]),
                    "rows": simulate(sent, {})})
    return out


# ---------------------------------------------------------------------------
# gen
# ---------------------------------------------------------------------------


def build_history(it: dict, i: int) -> dict:
    rows, sent = it["rows"], it["sent"]
    row, ev = rows[i], sent[i]
    now = float(ev["end_time"])
    role = {"self": "You", "other": "Other", "unknown": "Unknown"}
    turns = []
    for j in range(i):
        if not rows[j]["text"].strip() or float(sent[j]["start_time"]) > now:
            continue
        turns.append({"t": float(sent[j]["end_time"]), "start": float(sent[j]["start_time"]),
                      "who": role[rows[j]["ident"]], "text": rows[j]["text"], "ongoing": False})
    turns = turns[-6:]
    return {"turns": turns, "coach": [], "self": row["ident"] == "self", "unknown": row["ident"] == "unknown",
            "now": now, "turn_start": float(ev["start_time"]), "wearer_floor_s": float(row["floor"]) if row["ident"] == "self" else 0.0}


async def generate_item(it: dict, prompt: str, llm) -> list[dict]:
    import audio_pipeline as ap
    import coach_gate
    import coach_prompt_v3 as v3
    from models.audio import Utterance

    settings = it["bundle"].get("settings") or {}
    recs = []
    for i, row in enumerate(it["rows"]):
        if not row["triggers"]:
            continue
        ev = it["sent"][i]
        utt = Utterance(session_id="replay", speaker=ev["speaker"], text=ev["text"],
                        start_time=float(ev["start_time"]), end_time=float(ev["end_time"]))
        hist = build_history(it, i)
        tone = ap._tone_context_from_parts(ev.get("text_tone"), ev.get("prosody"))
        ident = row["ident"]
        common = dict(history=hist, relationship=settings.get("relationship"),
                      session_context=settings.get("session_context"), library=None, coach_language=None)
        rec = {"item": it["item"], "split": it["split"], "turn_uid": row["uid"], "i": i, "end": row["end"],
               "turn": row["text"], "ident": ident, "triggers": list(row["triggers"]), "called": False,
               "line": "", "importance": 0, "kind": "nudge" if ident == "self" else "response"}
        if prompt == "v3":
            os.environ[v3.FLAG] = "1"
            if v3.line_type(row["triggers"], ident) is None:
                rec["skipped"] = "v3: no line type (unknown wearer / no usable trigger)"
                recs.append(rec)
                continue
            extra = dict(triggers=tuple(row["triggers"]), trigger_evidence=dict(row["evidence"]))
        else:
            os.environ.pop(v3.FLAG, None)
            extra = {}
        rec["called"] = True
        try:
            if ident == "self":
                nudge, imp = await ap._generate_nudge(llm, utt, 50, None, None, tone, None, **common, **extra)
                nudge, imp = (nudge, imp) if (nudge and imp >= NUDGE_MIN) else ("", 0)
                lines = [nudge] if nudge else []
            else:
                lines, imp = await ap._generate_suggestions(
                    llm, utt, 50, None, None, tone, wearer_unknown=ident == "unknown", **common,
                    **({**extra, "identity": ident} if extra else {}))
            lines = coach_gate.scrub_lines(list(lines), set())
            rec["line"] = lines[0] if lines else ""
            rec["importance"] = imp if rec["line"] else 0
        except Exception as exc:  # honest failure, never a fabricated line
            rec["error"] = f"{type(exc).__name__}: {exc}"
        recs.append(rec)
    os.environ.pop(v3.FLAG, None)
    return recs


def speak_gate(recs: list[dict]) -> None:
    """Re-run the live speak gate (importance >= 80, unknown capped at 72,
    turn >= 3 words, 60 s gap) in time order; sets rec['spoken']."""
    last = None
    for r in sorted(recs, key=lambda r: (r["item"], r["end"])):
        r["spoken"] = False
        if not r["line"]:
            continue
        imp = min(r["importance"], UNKNOWN_CAP) if r["ident"] == "unknown" else r["importance"]
        if imp < SPEAK_MIN or words(r["turn"]) < MIN_WORDS:
            continue
        if last is not None and last[0] == r["item"] and r["end"] - last[1] < GAP_S:
            continue
        last = (r["item"], r["end"])
        r["spoken"] = True


def cmd_gen(a) -> None:
    load_env()
    items = load_items(a.split)
    llm = cached_client(os.environ["MINDSHIFT_MODEL"])

    async def run():
        out = []
        for it in items:
            out += await generate_item(it, a.prompt, llm)
        return out

    recs = asyncio.run(run())
    speak_gate(recs)
    res = {"prompt": a.prompt, "split": a.split, "model": llm.model, "made_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "llm_cache": llm.stats, "items": {it["item"]: {"split": it["split"], "group": it["group"], "dur_s": it["dur_s"]} for it in items},
           "records": recs}
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    called = sum(1 for r in recs if r["called"])
    spoken = [r for r in recs if r["spoken"]]
    print(f"{a.prompt}/{a.split}: {len(items)} sessions, {sum(it['dur_s'] for it in items)/3600:.2f} h, "
          f"{called} LLM calls, {sum(1 for r in recs if r['line'])} non-silent, {len(spoken)} spoken, "
          f"{sum(1 for r in recs if r.get('error'))} errors; cache {llm.stats}")
    for r in spoken:
        print(f"  [{r['item']} @{r['end']:.0f}s {r['ident']} {','.join(r['triggers'])}] {r['line']}")


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------


def truth_turns(bundle: dict):
    import judge_export
    return judge_export.conversation_turns(bundle)[0]


def clip_turn(t: dict, a: float) -> dict:
    if t["end"] <= a:
        return dict(t)
    w = t["text"].split()
    k = int(len(w) * (a - t["start"]) / max(1e-6, t["end"] - t["start"]))
    out = dict(t)
    out["end"] = round(a, 2)
    out["still_speaking_at_arrival"] = True
    out["text"] = (" ".join(w[:k]) + " ..." if k else "...") + " (still speaking)"
    return out


def grader_view(it: dict, rec: dict) -> dict:
    a = rec["end"] + ARRIVAL_AFTER_TURN_S
    turns = [t for t in truth_turns(it["bundle"]) if t["start"] < a][-6:]
    settings = it["bundle"].get("settings") or {}
    return {
        "setting": settings.get("session_context"), "relationship": settings.get("relationship"),
        "roles_basis": "ground truth where annotated, else the replay's guess",
        "context_before": [clip_turn(t, a) for t in turns],
        "context_note": "Shows only what had been said by arrived_at_s; nothing after it.",
        "answered_turn": {"text": rec["turn"], "ended_at_s": rec["end"],
                          "role_per_phone": "wearer" if rec["ident"] == "self" else "other"},
        "coach_line": rec["line"], "kind": rec["kind"], "arrived_at_s": round(a, 2),
        "seconds_after_turn_end": ARRIVAL_AFTER_TURN_S, "spoken": True,
    }


def line_id(r: dict) -> str:
    return f"{r['item']}#{r['i']}"


def judge_one(client, view: dict, rubric: str, model: str) -> dict:
    from main import parse_llm_json
    system = rubric + ("\n\nGrade the single row below. Respond with ONLY one JSON object: "
                       '{"timing": 0-3, "specificity": 0-3, "helpful": 0 or 1, "should_be_silent": true|false, "why": "<=20 words"}')
    raw = client.complete(system=system, user=json.dumps({"grader_view": view}, ensure_ascii=False),
                          max_tokens=3000 if "gpt-5" in model else 400)
    d = parse_llm_json(raw)
    return {"timing": d.get("timing"), "specificity": d.get("specificity"),
            "helpful": bool(int(d.get("helpful", 0))), "should_be_silent": d.get("should_be_silent"), "why": d.get("why")}


def cmd_judge(a) -> None:
    load_env()
    gen = json.load(open(a.lines))
    items = {it["item"]: it for it in load_items("all")}
    rubric = (HERE.parent / "RUBRIC_v4.md").read_text()
    judges = {j: cached_client(m) for j, m in JUDGES.items() if j in a.judges}
    out = {"rubric": "RUBRIC_v4.md", "judges": {j: JUDGES[j] for j in judges}, "grades": {}, "views": {}}
    for r in gen["records"]:
        if not r["line"]:
            continue          # silence needs no grade (spoken lines are a subset of these)
        lid = line_id(r)
        view = grader_view(items[r["item"]], r)
        out["views"][lid] = view
        for j, c in judges.items():
            try:
                out["grades"].setdefault(j, {})[lid] = judge_one(c, view, rubric, JUDGES[j])
            except Exception as exc:
                out.setdefault("errors", []).append({"judge": j, "id": lid, "error": f"{type(exc).__name__}: {exc}"})
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"judged {len(out['views'])} spoken lines with {list(judges)}; errors {len(out.get('errors', []))}")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def leaks(r: dict, ctx_text: str) -> list[str]:
    import coach_gate
    import coach_prompt_v3 as v3
    found = []
    if coach_gate.has_label(r["line"]) or re.search(r"\bthe wearer\b", r["line"], re.I):
        found.append("label")
    if v3._foreign_names(r["line"], ctx_text):
        found.append("unseen_name")
    return found


def score_arm(gen: dict, judg: dict, mode: str, items_all: dict, scope: str = "spoken") -> dict:
    from recreplay import nudge_quality as nq
    labels = nq.combine_judges({j: {k: v["helpful"] for k, v in g.items()} for j, g in judg["grades"].items()}, mode)
    spoken, lk = [], []
    for r in gen["records"]:
        if not r["line"] or (scope == "spoken" and not r["spoken"]):
            continue
        lid = line_id(r)
        view = judg["views"][lid]
        ctx_text = " ".join([view["answered_turn"]["text"]] + [t["text"] for t in view["context_before"]])
        for k in leaks(r, ctx_text):
            lk.append({"id": lid, "line": r["line"], "leak": k})
        if lid in labels:
            spoken.append({"item": r["item"], "helpful": labels[lid], "id": lid, "line": r["line"],
                           "triggers": r["triggers"], "ident": r["ident"], "group": items_all[r["item"]]["group"]})
    return {"spoken": spoken, "leaks": lk}


def cmd_score(a) -> None:
    from recreplay import nudge_quality as nq
    gen = json.load(open(a.lines))
    judg = json.load(open(a.judgments))
    base = json.load(open(a.baseline_gen)) if a.baseline_gen else None
    base_j = json.load(open(a.baseline_judgments)) if a.baseline_judgments else None
    meta = gen["items"]
    hours_by_item = {k: v["dur_s"] / 3600 for k, v in meta.items()}
    hours = sum(hours_by_item.values())
    items_all = {k: {"group": v["group"]} for k, v in meta.items()}
    res = {"prompt": gen["prompt"], "split": gen["split"], "hours": round(hours, 4), "sessions": len(meta),
           "model": gen["model"], "judges": judg["judges"], "limits": __doc__.split("Honest limits")[1].split("\n\n")[0].strip()}
    calls = sum(1 for r in gen["records"] if r["called"])
    res["llm_calls"] = calls
    res["llm_calls_per_h"] = round(calls / hours, 2)
    res["nonsilent_lines"] = sum(1 for r in gen["records"] if r["line"])
    res["spoken_per_item"] = {k: sum(1 for r in gen["records"] if r["spoken"] and r["item"] == k) for k in meta}
    for scope, mode in [("spoken", m) for m in ("both", "any", *judg["judges"])] + [("generated", "both"), ("generated", "any")]:
        arm = score_arm(gen, judg, mode, items_all, scope)
        key = f"label_{mode}" if scope == "spoken" else f"generated_{mode}"
        s = nq.summarize(arm["spoken"], hours)
        boot = nq.bootstrap(arm["spoken"], None, hours_by_item, B=a.B)
        s["U_per_h_ci95"] = boot["a_ci95"]
        if mode == "both":
            s["lines"] = arm["spoken"]
            s["leaks"] = arm["leaks"]
            s["leaks_n"] = len(arm["leaks"])
            calm = [r for r in arm["spoken"] if r["group"] == "calm"]
            calm_h = sum(hours_by_item[k] for k, v in meta.items() if v["group"] == "calm")
            s["calm_spoken_per_h"] = round(len(calm) / calm_h, 2) if calm_h else None
            s["calm_hours"] = round(calm_h, 3)
        if base is not None and mode in ("both", "any"):
            barm = score_arm(base, base_j, mode, items_all, scope)
            d = nq.bootstrap(arm["spoken"], barm["spoken"], hours_by_item, B=a.B)
            s["baseline_U_per_h"] = nq.summarize(barm["spoken"], hours)["U_per_h"]
            s["baseline_helpful_share"] = nq.summarize(barm["spoken"], hours)["helpful_share"]
            s["delta_U_per_h"] = s["U_per_h"] - s["baseline_U_per_h"]
            s["delta_ci95"] = d["delta_ci95"]
        res[key] = s
    both = [j for j in judg["grades"]]
    if len(both) == 2:
        ids = sorted(set(judg["grades"][both[0]]) & set(judg["grades"][both[1]]))
        res["judge_kappa"] = nq.cohen_kappa([judg["grades"][both[0]][i]["helpful"] for i in ids],
                                            [judg["grades"][both[1]][i]["helpful"] for i in ids])
        res["judge_agree"] = sum(judg["grades"][both[0]][i]["helpful"] == judg["grades"][both[1]][i]["helpful"] for i in ids), len(ids)
    res["judge_errors"] = judg.get("errors", [])
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    b = res["label_both"]
    print(f"{gen['prompt']}/{gen['split']}: {res['hours']} h, {calls} calls ({res['llm_calls_per_h']}/h), "
          f"spoken {b['n_spoken']} (helpful {b['helpful']}, share {b['helpful_share']}), "
          f"U2/h {b['U_per_h']:.1f} CI {b['U_per_h_ci95']}, leaks {b['leaks_n']}, kappa {res.get('judge_kappa')}")
    if "delta_U_per_h" in b:
        print(f"  vs baseline: delta U2/h {b['delta_U_per_h']:.1f} CI {b['delta_ci95']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--prompt", choices=["baseline", "v3"], required=True)
    g.add_argument("--split", choices=["dev", "held_out", "all"], required=True)
    g.add_argument("--out", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--lines", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--judges", default="AB")
    s = sub.add_parser("score")
    s.add_argument("--lines", required=True)
    s.add_argument("--judgments", required=True)
    s.add_argument("--baseline-gen")
    s.add_argument("--baseline-judgments")
    s.add_argument("--out", required=True)
    s.add_argument("--split")
    s.add_argument("--B", type=int, default=4000)
    a = ap.parse_args()
    {"gen": cmd_gen, "judge": cmd_judge, "score": cmd_score}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
