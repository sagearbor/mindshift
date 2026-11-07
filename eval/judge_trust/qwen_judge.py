"""Open-weight local nudge judge: Qwen3.5-9B (Apache-2.0) 4-bit via mlx-lm on this Mac, $0.

Same system prompt / output as nudge_judges.py; results appended to nudge_results_qwen.jsonl with
model "qwen-local".  Run with tmp/judge-trust-20261105/venv-mlx/bin/python qwen_judge.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ledger  # noqa: E402
from env import OUT  # noqa: E402
from nudge_judges import load_cases, parse, system_prompt  # noqa: E402

MODEL_ID = "mlx-community/Qwen3.5-9B-4bit"


def main() -> None:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(MODEL_ID)
    system = system_prompt()
    cases = load_cases(sys.argv[1] if len(sys.argv) > 1 else "all")
    sampler = make_sampler(temp=0.0)
    with open(OUT / "nudge_results_qwen.jsonl", "a") as f:
        for i, (cid, view) in enumerate(cases):
            user = json.dumps({"id": cid, "grader_view": view}, ensure_ascii=False)
            key = ledger.cache_key("nudge-v1", "qwen-local", MODEL_ID, system, user)
            hit = ledger.cache_get(key)
            if hit is None:
                msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                try:
                    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
                except TypeError:
                    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
                t0 = time.time()
                raw = generate(model, tok, prompt=prompt, max_tokens=600, sampler=sampler)
                d = parse(raw)
                hit = {"case_id": cid, "model": "qwen-local", "ok": d is not None,
                       "raw": None if d else raw[:300], "secs": round(time.time() - t0, 1), **(d or {})}
                hit["case_id"] = cid
                if d is not None:
                    ledger.cache_put(key, hit)
            f.write(json.dumps(hit, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print(i, len(cases), hit.get("secs"), flush=True)
    print("done")


if __name__ == "__main__":
    main()
