#!/usr/bin/env python
"""Bench the cloud-suggestion path against the REAL Anthropic API.

Two modes, both measuring what production's ``scripts/live_e2e.py`` reports
(``latency_summary`` stages, time-to-first-partial, parse failures):

``llm`` (default) — the LLM stage in isolation, N calls per prompt/request
variant, sequential like a session's turns. Per variant: p50/p95 of total,
time-to-first-token and time-to-first-suggestion (the moment the streaming
``partial`` preview would fire), output tokens, raw-parse failures (and
whether the one-shot repair recovered them), and the prompt-cache usage
fields (``cache_read_input_tokens`` > 0 = the cache engaged)::

    set -a; source .env; set +a
    python scripts/bench_suggestions.py --n 20
    python scripts/bench_suggestions.py --n 20 --variants legacy,lean+cached
    python scripts/bench_suggestions.py --n 20 --model claude-sonnet-5

``ws`` — the whole realtime pipeline in-process: a real uvicorn serving
``main.app`` on the loopback with ONLY the LLM real (transcriber/TTS are
nulls, auth is a keyless fake, exactly the seams the in-process e2e test
uses), driven by ``live_e2e.stream_live_session`` — the phone-shaped client
production is measured with. Prints the server's ``latency_summary``
(queue_wait / llm / llm_first_partial / total …) and the client-side
time-to-first-partial. ``--legacy`` flips every perf knob back to the
pre-change behaviour (REST prompt, 512 max_tokens, no cache, one worker, no
repair) so before/after come from the same code::

    python scripts/bench_suggestions.py --mode ws --scene scene_couple_escalation --speed 2
    python scripts/bench_suggestions.py --mode ws --legacy --scene scene_couple_escalation --speed 2

Spend is small (Haiku, ~100 short calls per full run). Never deploys.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import socket
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_MODEL = os.getenv("MINDSHIFT_MODEL", "claude-haiku-4-5-20251001")
SCENES = ("scene_couple_escalation", "scene_family3", "scene_meeting4")


# ---------------------------------------------------------------------------
# Sample turns (the scene packs the phone-shaped e2e client uses)
# ---------------------------------------------------------------------------

def sample_turns(n: int, *, self_only: bool = False, other_only: bool = False) -> list[dict]:
    """``n`` turns cycled from the three scene packs, each with the
    text-tone block the phone would attach (live_e2e.text_tone_for)."""
    import live_e2e

    turns: list[dict] = []
    for name in SCENES:
        scene = live_e2e.load_scene(name)
        for t in scene.turns:
            is_self = scene.is_self(t["speaker"])
            if (self_only and not is_self) or (other_only and is_self):
                continue
            turns.append({
                "text": t["text"], "speaker": t["speaker"], "is_self": is_self,
                "tone_context": {"text_tone": live_e2e.text_tone_for(t)},
            })
    if not turns:
        raise SystemExit("no scene turns found")
    return [turns[i % len(turns)] for i in range(n)]


def pct(values: list[float], p: int) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def fmt(v: float | None) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.0f}"


# ---------------------------------------------------------------------------
# Mode: llm — the LLM stage in isolation
# ---------------------------------------------------------------------------

SUGGESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestions": {"type": "array", "items": {"type": "string"}},
        "importance": {"type": "integer"},
    },
    "required": ["suggestions", "importance"],
    "additionalProperties": False,
}

VARIANTS: dict[str, dict] = {
    # production before this branch: REST contract (tone_score), 512 tokens,
    # plain-string system prompt
    "legacy": dict(live=False, cache=False, max_tokens=512, schema=False),
    "cached": dict(live=False, cache=True, max_tokens=512, schema=False),
    # what ships (audio_pipeline + llm_client defaults)
    "lean": dict(live=True, cache=False, max_tokens=200, schema=False),
    # the cache_control marker on top — evaluated, OFF by default (see
    # llm_client.PROMPT_CACHE_ENABLED for the measurement)
    "lean+cached": dict(live=True, cache=True, max_tokens=200, schema=False),
    # structured outputs (output_config.format) on top — evaluated, not shipped
    "lean+schema": dict(live=True, cache=False, max_tokens=200, schema=True),
}


def run_llm_variant(name: str, spec: dict, turns: list[dict], model: str) -> dict:
    import audio_pipeline
    from audio_pipeline import _first_suggestion_in, _parse_or_repair, _turn_prompt
    from llm_client import LLMClient
    from main import empathy_system_prompt, parse_llm_json
    from models.audio import Utterance

    llm = LLMClient(model=model, cache_system_prompt=spec["cache"])
    totals: list[float] = []
    ttfts: list[float] = []
    firsts: list[float] = []
    out_tokens: list[int] = []
    raw_fail = repaired = unrecovered = 0
    max_hit = 0
    per_call: list[dict] = []
    for turn in turns:
        u = Utterance(session_id="bench", speaker=turn["speaker"], text=turn["text"],
                      start_time=0.0, end_time=1.0)
        system = empathy_system_prompt(60, "Husband", None, live=spec["live"])
        user = _turn_prompt(u, turn["tone_context"])
        parts: list[str] = []
        t0 = time.monotonic()
        ttft = first = None
        for delta in llm.stream_complete(
            system=system, user=user, max_tokens=spec["max_tokens"],
            response_schema=SUGGESTION_SCHEMA if spec["schema"] else None,
        ):
            now = time.monotonic()
            if ttft is None:
                ttft = now
            parts.append(delta)
            if first is None and _first_suggestion_in("".join(parts)) is not None:
                first = now
        t1 = time.monotonic()
        raw = "".join(parts)
        usage = llm.last_usage or {}
        totals.append((t1 - t0) * 1000)
        ttfts.append(((ttft or t1) - t0) * 1000)
        if first is not None:
            firsts.append((first - t0) * 1000)
        out_tokens.append(usage.get("output_tokens", 0))
        if usage.get("output_tokens", 0) >= spec["max_tokens"]:
            max_hit += 1
        ok_raw = True
        try:
            data = parse_llm_json(raw)
            ok_raw = isinstance(data, dict) and isinstance(data.get("suggestions"), list)
        except Exception:  # noqa: BLE001
            ok_raw = False
        if not ok_raw:
            raw_fail += 1
            try:
                asyncio.run(_parse_or_repair(
                    llm, raw, keys='"suggestions" (list of strings), "importance" (integer)',
                    what="response", utterance_text=u.text,
                ))
                repaired += 1
            except audio_pipeline.SuggestionUnavailable:
                unrecovered += 1
        per_call.append({"ms": round(totals[-1]), "out": out_tokens[-1],
                         "cache_read": usage.get("cache_read_input_tokens", 0),
                         "ok": ok_raw})
    llm.close()
    u = llm.usage_totals
    return {
        "variant": name, "n": len(turns),
        "total_p50": pct(totals, 50), "total_p95": pct(totals, 95),
        "ttft_p50": pct(ttfts, 50), "ttft_p95": pct(ttfts, 95),
        "first_p50": pct(firsts, 50) if firsts else None,
        "first_p95": pct(firsts, 95) if firsts else None,
        "first_n": len(firsts),
        "out_tok_p50": pct(out_tokens, 50), "max_tokens_hit": max_hit,
        "raw_fail": raw_fail, "repaired": repaired, "unrecovered": unrecovered,
        "input_tokens": u["input_tokens"], "cache_write": u["cache_creation_input_tokens"],
        "cache_read": u["cache_read_input_tokens"],
        "per_call": per_call,
    }


def run_nudge_variant(name: str, max_tokens: int, turns: list[dict], model: str) -> dict:
    from audio_pipeline import _turn_prompt
    from llm_client import LLMClient
    from main import parse_llm_json, self_feedback_prompt
    from models.audio import Utterance

    llm = LLMClient(model=model)
    totals: list[float] = []
    out_tokens: list[int] = []
    fails = 0
    for turn in turns:
        u = Utterance(session_id="bench", speaker=turn["speaker"], text=turn["text"],
                      start_time=0.0, end_time=1.0)
        t0 = time.monotonic()
        raw = llm.complete(system=self_feedback_prompt(60, "Husband"),
                           user=_turn_prompt(u, turn["tone_context"]), max_tokens=max_tokens)
        totals.append((time.monotonic() - t0) * 1000)
        out_tokens.append((llm.last_usage or {}).get("output_tokens", 0))
        try:
            d = parse_llm_json(raw)
            assert isinstance(d.get("nudge"), str)
        except Exception:  # noqa: BLE001
            fails += 1
    llm.close()
    return {"variant": name, "n": len(turns), "total_p50": pct(totals, 50),
            "total_p95": pct(totals, 95), "out_tok_p50": pct(out_tokens, 50), "raw_fail": fails}


def print_llm_table(rows: list[dict], model: str) -> None:
    print(f"\nLLM stage, model={model} (ms; sequential calls; n per variant)")
    head = (f"{'variant':<13}{'n':>3}{'total p50':>10}{'p95':>7}{'ttft p50':>10}{'p95':>7}"
            f"{'1st-sugg p50':>13}{'p95':>7}{'out tok':>8}{'cap hit':>8}"
            f"{'parse fail':>11}{'repaired':>9}{'cache w/r':>12}")
    print(head)
    for r in rows:
        print(f"{r['variant']:<13}{r['n']:>3}{fmt(r['total_p50']):>10}{fmt(r['total_p95']):>7}"
              f"{fmt(r['ttft_p50']):>10}{fmt(r['ttft_p95']):>7}"
              f"{fmt(r['first_p50']):>13}{fmt(r['first_p95']):>7}{fmt(r['out_tok_p50']):>8}"
              f"{r['max_tokens_hit']:>8}{r['raw_fail']:>11}{r['repaired']:>9}"
              f"{str(r['cache_write']) + '/' + str(r['cache_read']):>12}")
    print("cache w/r = cache_creation_input_tokens / cache_read_input_tokens summed over the"
          " variant's calls (0/0 = the prefix is below the model's cacheable minimum).")


def run_llm_mode(args: argparse.Namespace) -> int:
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; choose from {list(VARIANTS)}")
    turns = sample_turns(args.n, other_only=True)
    rows = []
    for name in names:
        print(f"... {name} x{args.n} on {args.model}", flush=True)
        try:
            rows.append(run_llm_variant(name, VARIANTS[name], turns, args.model))
        except Exception as exc:  # noqa: BLE001 — a variant the API rejects is a finding
            print(f"    {name}: FAILED {type(exc).__name__}: {str(exc)[:200]}")
    print_llm_table(rows, args.model)
    if args.nudges:
        self_turns = sample_turns(args.n, self_only=True)
        nrows = []
        for name, mt in (("nudge legacy(512)", 512), ("nudge capped(60)", 60)):
            print(f"... {name} x{args.n}", flush=True)
            nrows.append(run_nudge_variant(name, mt, self_turns, args.model))
        print(f"\nNudge (self-turn) LLM stage, model={args.model}")
        print(f"{'variant':<20}{'n':>3}{'total p50':>10}{'p95':>7}{'out tok':>8}{'parse fail':>11}")
        for r in nrows:
            print(f"{r['variant']:<20}{r['n']:>3}{fmt(r['total_p50']):>10}{fmt(r['total_p95']):>7}"
                  f"{fmt(r['out_tok_p50']):>8}{r['raw_fail']:>11}")
    if args.json:
        print(json.dumps(rows, indent=1, default=str))
    return 0


# ---------------------------------------------------------------------------
# Mode: ws — the whole pipeline in-process, only the LLM real
# ---------------------------------------------------------------------------

class _NullTranscriber:
    async def connect(self) -> None: ...
    async def stream(self, audio_bytes: bytes): return []
    async def finish(self): return []
    async def close(self) -> None: ...


class _FakeTTS:
    async def synthesize(self, text: str): return "ZmFrZS1hdWRpbw=="


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def apply_legacy_knobs(legacy: bool) -> dict:
    """Flip the perf knobs (module attributes — same seam the tests use)."""
    import audio_pipeline
    import llm_client

    knobs = {
        "LIVE_PROMPT": not legacy,
        "PARSE_REPAIR": not legacy,
        "SUGGESTION_MAX_TOKENS": 512 if legacy else audio_pipeline.SUGGESTION_MAX_TOKENS,
        "NUDGE_MAX_TOKENS": 512 if legacy else audio_pipeline.NUDGE_MAX_TOKENS,
        "LOCAL_FIRST_CONCURRENCY": 1 if legacy else audio_pipeline.LOCAL_FIRST_CONCURRENCY,
    }
    for k, v in knobs.items():
        setattr(audio_pipeline, k, v)
    knobs["cache_system_prompt"] = False if legacy else llm_client.PROMPT_CACHE_ENABLED
    llm_client.LLMClient.cache_system_prompt = knobs["cache_system_prompt"]
    return knobs


async def run_ws_once(base_url: str, scene_name: str, speed: float, session_id: str) -> dict:
    import live_e2e

    scene = live_e2e.load_scene(scene_name)
    account = live_e2e.Account(email="bench@example.test", ws_token="bench-token",
                               headers={}, uid="bench-user")
    run = await live_e2e.stream_live_session(
        base_url, account, scene, session_id=session_id, speed=speed,
    )
    if run.error or run.session_complete is None:
        raise SystemExit(f"ws run failed: error={run.error} complete={run.session_complete}")
    timing = live_e2e.first_response_ms(run)
    suggestions = run.of_type("suggestion")
    errors = run.of_type("suggestion_error")
    return {
        "scene": scene_name, "turns": len(scene.turns),
        "latency_summary": run.session_complete.get("latency_summary", {}),
        "finals": len([s for s in suggestions if not s.get("partial")]),
        "partials": len([s for s in suggestions if s.get("partial")]),
        "errors": [e.get("reason") for e in errors],
        "timing": timing,
        "wall_s": round(run.wall_seconds, 1),
    }


def run_ws_mode(args: argparse.Namespace) -> int:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    os.environ["MINDSHIFT_DB_PATH"] = tmp.name
    tmp.close()
    os.environ["MINDSHIFT_MODEL"] = args.model
    import uvicorn

    import audio_pipeline
    import auth
    import main
    from llm_client import LLMClient

    knobs = apply_legacy_knobs(args.legacy)
    asyncio.run(main.init_db())
    auth.verify_id_token = lambda token: "bench-user"  # keyless, like the test suite
    audio_pipeline.watch_relay = None
    main.app.state.llm_client = LLMClient(model=args.model, cache_system_prompt=knobs["cache_system_prompt"])
    main.app.state.transcriber_factory = lambda: _NullTranscriber()
    main.app.state.tts_client = _FakeTTS()

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port,
                                           log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise SystemExit("uvicorn did not start")
    label = "LEGACY (pre-change knobs)" if args.legacy else "CURRENT (shipping knobs)"
    print(f"\nws mode — {label}: {knobs}; model={args.model}; speed={args.speed}x")
    results = []
    try:
        for scene_name in args.scene.split(","):
            for rep in range(args.repeat):
                sid = f"bench-{scene_name}-{rep}-{int(time.time() * 1000)}"
                print(f"... {scene_name} run {rep + 1}/{args.repeat}", flush=True)
                results.append(asyncio.run(run_ws_once(f"http://127.0.0.1:{port}", scene_name,
                                                       args.speed, sid)))
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        main.app.state.llm_client.close()

    # Merge the per-run summaries into one table (server percentiles per
    # run; the client-side time-to-first numbers pooled across runs).
    for r in results:
        lat = r["latency_summary"]
        print(f"\n{r['scene']}: {r['turns']} turns, {r['finals']} finals, {r['partials']} partials, "
              f"errors={r['errors'] or 0}, wall {r['wall_s']}s")
        print(f"    {'stage':<18}{'p50':>9}{'p95':>9}{'n':>5}")
        for stage in ("seg_to_enqueue", "queue_wait", "llm", "llm_first_partial", "tts", "total"):
            v = lat.get(stage)
            if v:
                print(f"    {stage:<18}{v['p50']:>9.1f}{v['p95']:>9.1f}{v['n']:>5}")
        t = r["timing"]
        print(f"    client time-to-first-partial p50 {t['partial_p50_ms']} ms; "
              f"first-response p50 {t['first_p50_ms']} (min {t['first_min_ms']}, max {t['first_max_ms']}) "
              f"over {t['turns_with_response']} turns")
    # Pooled client-side numbers across every run — per-turn samples, so the
    # percentiles are over all turns rather than a median of per-run medians.
    partial_all: list[float] = []
    first_all: list[float] = []
    first_other: list[float] = []
    first_self: list[float] = []
    for r in results:
        for t in r["timing"]["per_turn"]:
            fp, ff = t["first_partial_ms"], t["first_final_ms"]
            if fp is not None:
                partial_all.append(fp)
            first = min(x for x in (fp, ff) if x is not None) if (fp is not None or ff is not None) else None
            if first is None:
                continue
            first_all.append(first)
            (first_self if t["is_self"] else first_other).append(first)
    print(f"\nPOOLED over {len(results)} run(s) — client-side, ms from turn_local sent (n = turns with a response)")
    print(f"{'metric':<34}{'n':>4}{'p50':>8}{'p95':>8}{'min':>7}{'max':>7}")
    for label, vals in (("time-to-first-partial (other)", partial_all),
                        ("first response, any turn", first_all),
                        ("first response, OTHER turns", first_other),
                        ("first response, SELF (nudge) turns", first_self)):
        if vals:
            print(f"{label:<34}{len(vals):>4}{pct(vals, 50):>8.0f}{pct(vals, 95):>8.0f}"
                  f"{min(vals):>7.0f}{max(vals):>7.0f}")
    usage = main.app.state.llm_client.usage_totals
    print(f"\nLLM usage over the run(s): {usage}")
    if args.json:
        print(json.dumps(results, indent=1, default=str))
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["llm", "ws"], default="llm")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n", type=int, default=20, help="llm mode: calls per variant")
    p.add_argument("--variants", default=",".join(VARIANTS), help="llm mode: comma list")
    p.add_argument("--nudges", action="store_true", help="llm mode: also bench the self-turn nudge path")
    p.add_argument("--legacy", action="store_true", help="ws mode: pre-change knobs")
    p.add_argument("--scene", default="scene_couple_escalation", help="ws mode: comma list")
    p.add_argument("--speed", type=float, default=1.0, help="ws mode: stream N× real time")
    p.add_argument("--repeat", type=int, default=1, help="ws mode: runs per scene")
    p.add_argument("--json", action="store_true")
    return p


def load_env_file(path: Path) -> None:
    """Minimal ``.env`` loader (KEY=VALUE lines, # comments, optional
    quotes); never overrides a variable already in the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main_cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.add_argument("--env-file", default=str(ROOT / ".env"),
                        help="read ANTHROPIC_API_KEY / MINDSHIFT_MODEL from here when unset")
    args = parser.parse_args(argv)
    load_env_file(Path(args.env_file))
    if args.model == DEFAULT_MODEL and os.getenv("MINDSHIFT_MODEL"):
        args.model = os.environ["MINDSHIFT_MODEL"]
    if not os.getenv("ANTHROPIC_API_KEY") and args.model.startswith("claude-"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (set -a; source .env; set +a, or --env-file)")
    return run_ws_mode(args) if args.mode == "ws" else run_llm_mode(args)


if __name__ == "__main__":
    raise SystemExit(main_cli())
