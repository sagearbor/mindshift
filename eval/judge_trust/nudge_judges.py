"""Multi-family nudge-quality judge panel (rubric v4 + P(helpful) + atomic checklist, one call per case).

Judges (families): Anthropic claude-sonnet-5-5 / claude-haiku-5-5; OpenAI gpt-5-mini / gpt-6.1-terra
(evaluation only); Google gemini-3.8-flash / gemini-3.5-flash-lite; open-weight Qwen (local MLX,
see qwen_judge.py).  Every API call goes through ledger (cost log, cache, hard cap).
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import ledger
from env import MAIN, OUT, load_env

load_env()

ADDENDUM = """

## Extra output fields for this evaluation (add them to the same JSON object)
"p_helpful": your probability from 0 to 1 that the wearer is better off hearing this line than silence.
"checklist": answer each from the context shown only:
  "real_trigger": true if the wearer is talking over someone, has held the floor about 45 s or more,
      heat is rising (sarcasm, raised stakes, repeated overlap), a question to the wearer was dropped,
      or the other person asked for the floor;
  "internal_label": the line says an internal label aloud (Speaker A/B/C, S1, "the wearer");
  "invented_fact": the line asserts a name, fact or event that is not in the context;
  "wrong_person": "you" points at the wrong person, or the line speaks AS the wearer / scripts the
      wearer's own words, or treats another speaker's words as the wearer's;
  "praise_or_engagement_in_calm": praise/reinforcement, or a curiosity/engagement prompt in calm talk;
  "misreads_moment": laughter/banter read as conflict, or tells a wearer who is only listening to stop/wait.
Reply with ONE JSON object only, no prose:
{"id": "...", "timing": 0, "specificity": 0, "helpful": 0, "should_be_silent": false, "p_helpful": 0.0,
 "checklist": {"real_trigger": false, "internal_label": false, "invented_fact": false, "wrong_person": false,
 "praise_or_engagement_in_calm": false, "misreads_moment": false}, "why": "<=20 words"}
"""
FLAGS = ["internal_label", "invented_fact", "wrong_person", "praise_or_engagement_in_calm", "misreads_moment"]
FAMILY = {"claude-sonnet-5-5": "Anthropic", "claude-haiku-5-5": "Anthropic", "gpt-5-mini": "OpenAI",
          "gpt-6.1-terra": "OpenAI", "gemini-3.8-flash": "Google", "gemini-3.5-flash-lite": "Google",
          "qwen-local": "Qwen (open weights)"}
EST = {"claude-sonnet-5-5": 0.012, "claude-haiku-5-5": 0.001, "gpt-5-mini": 0.004, "gpt-6.1-terra": 0.008,
       "gemini-3.8-flash": 0.006, "gemini-3.5-flash-lite": 0.002}


def system_prompt() -> str:
    return (OUT / "RUBRIC_v4.md").read_text() + ADDENDUM


def parse(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "helpful" not in d:
        return None
    ck = d.get("checklist") or {}
    d["helpful"] = int(bool(d["helpful"]) if not isinstance(d["helpful"], (int, float)) else d["helpful"] >= 1)
    try:
        d["p_helpful"] = float(d.get("p_helpful"))
    except (TypeError, ValueError):
        d["p_helpful"] = float(d["helpful"])
    d["derived_helpful"] = int(bool(ck.get("real_trigger")) and not any(bool(ck.get(f)) for f in FLAGS))
    return d


_clients: dict = {}


def call_anthropic(model, system, user, task):
    import anthropic
    c = _clients.setdefault("a", anthropic.Anthropic())
    r = c.messages.create(model=model, max_tokens=4000, output_config={"effort": "low"},
                          system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                          messages=[{"role": "user", "content": user}])
    if r.stop_reason == "refusal":
        raise RuntimeError("refusal")
    u = r.usage
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    # cache reads billed at 0.1x, writes 1.25x: fold into an equivalent text_in count
    ledger.record(task, model, int(u.input_tokens + 0.1 * cr + 1.25 * cw), 0, u.output_tokens,
                  {"cache_read": cr, "cache_write": cw})
    return "".join(b.text for b in r.content if b.type == "text")


def call_openai(model, system, user, task):
    import openai
    c = _clients.setdefault("o", openai.OpenAI())
    r = c.chat.completions.create(model=model, reasoning_effort="low", max_completion_tokens=6000,
                                  response_format={"type": "json_object"},
                                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    u = r.usage
    cached = (u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0
    ledger.record(task, model, int(u.prompt_tokens - 0.9 * cached), 0, u.completion_tokens, {"cached": cached})
    return r.choices[0].message.content


def call_gemini(model, system, user, task):
    from google import genai
    from google.genai import types
    c = _clients.setdefault("g", genai.Client(vertexai=True, project="arborfam-hub", location="global"))
    cfg = types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                      max_output_tokens=4000, thinking_config=types.ThinkingConfig(thinking_level="low"))
    r = c.models.generate_content(model=model, contents=user, config=cfg)
    u = r.usage_metadata
    ledger.record(task, model, u.prompt_token_count or 0, 0, (u.candidates_token_count or 0) + (u.thoughts_token_count or 0))
    return r.text


def backend(model):
    if model.startswith("claude"):
        return call_anthropic
    if model.startswith("gpt"):
        return call_openai
    return call_gemini


def judge(model: str, case_id: str, view: dict, system: str) -> dict:
    user = json.dumps({"id": case_id, "grader_view": view}, ensure_ascii=False)
    key = ledger.cache_key("nudge-v1", model, system, user)
    hit = ledger.cache_get(key)
    if hit is not None:
        return hit
    ledger.check_budget(EST.get(model, 0.01))
    raw, d = None, None
    for _ in range(2):
        try:
            raw = backend(model)(model, system, user, "nudge-judge")
        except ledger.BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001
            raw = f"ERROR {type(e).__name__}: {str(e)[:200]}"
        d = parse(raw)
        if d is not None:
            break
    out = {"case_id": case_id, "model": model, "ok": d is not None, "raw": None if d else (raw or "")[:300], **(d or {})}
    out["case_id"] = case_id
    if d is not None:
        ledger.cache_put(key, out)
    return out


# Owner calibration cards 2026-10-11 (tmp/recordings/judge/owner-labels-20261011.md), helpful per owner.
# Card 4 ("moment": right moment, wrong words) is ambiguous for a binary verdict -> excluded.
# CONTAMINATED anchor: RUBRIC_v4's owner rules were written from these cards and cite several as examples.
OWNER_CARDS = {"final-H1:ami_TS3006a#37": 1, "final-H1:sbcsae_SBC042#5": 1, "final-H0:sbcsae_SBC007#17": 0,
               "final-H0:chime6_S02_t975#31": 0, "final-H0:chime6_S09_t3975#19": 1, "final-H0:sbcsae_SBC034#4": 0,
               "final-H0:chime6_S09_t3975#10": 0}


def load_owner_cards() -> list[tuple[str, dict]]:
    from nudge_cases import J, clean
    out = []
    for src in OWNER_CARDS:
        b, cid = src.split(":", 1)
        for line in open(J / f"{b}.jsonl"):
            r = json.loads(line)
            if r["id"] == cid:
                out.append((f"owner:{src}", clean(r["grader_view"])))
                break
    return out


def load_cases(which: str) -> list[tuple[str, dict]]:
    out = []
    if which in ("owner", "all"):
        out += load_owner_cards()
    if which in ("frozen", "all"):
        m = json.loads((OUT / "nudge_cases_frozen.json").read_text())
        out += [(c["case_id"], c["view"]) for c in m["cases"]]
    if which in ("existing17", "all"):
        p = MAIN / ".claude/worktrees/agent-a90ab94643480b5c9/tmp/nudge-eval/judge-baseline-all.json"
        d = json.loads(p.read_text())
        out += [(f"pr215:{k}", v) for k, v in d["views"].items()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--cases", default="all", choices=["frozen", "existing17", "owner", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    system = system_prompt()
    cases = load_cases(a.cases)[: a.limit]
    jobs = [(m, cid, v) for m in a.models for cid, v in cases]
    print(f"{len(jobs)} judge jobs; spent so far ${ledger.spent():.3f}", flush=True)
    done = 0
    with ThreadPoolExecutor(a.workers) as ex, open(OUT / "nudge_results.jsonl", "a") as f:
        futs = [ex.submit(judge, m, cid, v, system) for m, cid, v in jobs]
        for fu in as_completed(futs):
            try:
                r = fu.result()
            except ledger.BudgetExceeded as e:
                print("BUDGET STOP:", e, flush=True)
                ex.shutdown(cancel_futures=True)
                break
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} ${ledger.spent():.3f}", flush=True)
    print(f"finished {done}; total spend ${ledger.spent():.3f}")


if __name__ == "__main__":
    main()
