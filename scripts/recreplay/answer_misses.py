#!/usr/bin/env python3
"""answer_misses.py — the $0 coach stand-in for offline replays.

An offline replay (``--offline``) never calls a paid model: a prompt the LLM
cache has not seen is an honest ``LLMCacheMiss`` (that turn gets a
``suggestion_error``), and server/llm_cache.py appends the exact prompt to a
miss log (``work/<item>/llm_misses.jsonl``). This tool closes the loop:

    # 1. batch the misses for Claude Code subagents (Max plan, $0)
    python scripts/recreplay/answer_misses.py export tmp/recordings/work/<item>/llm_misses.jsonl \\
        --out tmp/recordings/free_replay/batches/<run>
    # 2. in a Claude Code session: one subagent per batch_NN.md; each writes answers_NN.jsonl
    # 3. ingest the answers as cache entries
    python scripts/recreplay/answer_misses.py ingest tmp/recordings/free_replay/batches/<run> --answered-by sonnet
    # 4. re-run the replay --offline: every prompt now hits

Honesty rules (AGENTS.md: no fabricated success):

* every entry is labelled ``provenance: "cc-subagent:<model>"`` and
  ``model: "<coach-model>@cc-subagent:<model>"`` — it is a stand-in, never
  passed off as the coach model's (Haiku's) output. The cache key is still the
  coach model's, so the replay finds it; ``LLMResponseCache.stats["sources"]``
  and the run report count these answers separately.
* the latency of these entries is NOT real. Each one borrows the timing curve
  (total latency + chunk arrival times) of a RECORDED real call from the same
  cache dirs (picked deterministically by key), so a replay with recorded
  latency stays realistic; ``latency_simulated: true`` + ``latency_source``
  say so. Without recorded entries ingest refuses unless
  ``--latency-ms-fallback`` is given (labelled ``fallback:<ms>``).
* answers that break the output contract (not JSON when the prompt demands
  JSON), unknown ids, and prompts that already have a REAL entry are rejected
  and listed; a real entry is never overwritten.
* a stand-in is Sonnet/Opus-class, not Haiku: results are an upper bound on
  line quality; confirm a final candidate with one paid Haiku run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT / "server") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "server"))

from llm_cache import LLMResponseCache  # noqa: E402

MANIFEST = "manifest.jsonl"


class NoLatencySource(RuntimeError):
    """No recorded real call to borrow timing from, and no fallback given."""


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def read_manifest(batch_dir: Path) -> list[dict]:
    return _read_jsonl(Path(batch_dir) / MANIFEST)


def _answered(row: dict) -> bool:
    return (Path(row["cache_dir"]) / f"{row['key']}.json").exists()


def _batch_text(n: int, rows: list[dict], batch_dir: Path) -> str:
    systems: dict[str, str] = {}
    for r in rows:
        systems.setdefault(r["system"], f"S{len(systems) + 1}")
    answers = (batch_dir / f"answers_{n:02d}.jsonl").resolve()
    parts = [
        f"# Coach stand-in batch {n:02d} ({len(rows)} prompts)\n",
        "You are standing in for the live coach model. For EACH item below, write exactly the text the model "
        "should output for that system prompt + user prompt: follow the system prompt's output contract to the "
        "letter (it usually asks for ONE JSON object, e.g. {\"suggestions\": [...], \"importance\": n} or "
        "{\"nudge\": \"...\", \"importance\": n}). Judge each item on its own prompt only; do not look at other "
        "files.\n",
        f"Write one JSON line per item to `{answers}`:\n",
        '    {"id": "<item id>", "response": "<the raw model output, as a JSON string>"}\n',
        "Every item gets exactly one line. No commentary, no code fences inside `response`.\n",
    ]
    for text, sid in systems.items():
        parts.append(f"\n## System prompt {sid}\n\n````text\n{text}\n````\n")
    for r in rows:
        params = f"max_tokens {r.get('max_tokens')}, temperature {r.get('temperature')}"
        if r.get("response_schema"):
            params += f", response_schema {json.dumps(r['response_schema'])}"
        parts.append(f"\n## Item {r['id']}  (system {systems[r['system']]}; {params})\n\n````text\n{r['user']}\n````\n")
    return "".join(parts)


def export(miss_logs: list[Path], out_dir: Path, *, batch_size: int = 20) -> dict:
    """Miss log(s) -> ``out_dir/batch_NN.md`` + ``manifest.jsonl``. Prompts
    that already have a cache entry, and duplicates, are skipped."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows: list[dict] = []
    skipped = 0
    for log in miss_logs:
        for r in _read_jsonl(Path(log)):
            if r["key"] in seen:
                continue
            seen.add(r["key"])
            if _answered(r):
                skipped += 1
                continue
            rows.append({**r, "id": r["key"][:16]})
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    with (out_dir / MANIFEST).open("w") as f:
        for n, b in enumerate(batches):
            for r in b:
                f.write(json.dumps({**r, "batch": n}) + "\n")
    for n, b in enumerate(batches):
        (out_dir / f"batch_{n:02d}.md").write_text(_batch_text(n, b, out_dir))
    return {"prompts": len(rows), "batches": len(batches), "already_answered": skipped, "out": str(out_dir)}


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?(.*?)\n?\s*```\s*$", re.S)


def _clean(response: str) -> str:
    m = _FENCE.match(response)
    return (m.group(1) if m else response).strip()


def _wants_json(row: dict) -> bool:
    return bool(row.get("response_schema")) or "json" in str(row.get("system", "")).lower()


def _is_real(entry: dict) -> bool:
    return not str(entry.get("provenance") or "api").startswith("cc-subagent")


def _timing_pool(dirs: list[Path]) -> list[tuple[str, dict]]:
    pool = []
    for d in dirs:
        if not Path(d).is_dir():
            continue
        for p in sorted(Path(d).glob("*.json")):
            try:
                e = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(e, dict) and _is_real(e) and e.get("latency_ms"):
                pool.append((f"{Path(d).name}/{p.stem[:12]}", e))
    return pool


def _simulated_chunks(response: str, template: dict) -> list[list]:
    """``response`` re-chunked on the template's (char fraction -> arrival
    ms) curve: same number of chunks, same timing, our text."""
    tchunks = template.get("chunks") or [[template["latency_ms"], template.get("response") or "x"]]
    total_chars = sum(len(c[1]) for c in tchunks) or 1
    out, done, cum = [], 0, 0
    for i, (at_ms, text) in enumerate(tchunks):
        cum += len(text)
        upto = len(response) if i == len(tchunks) - 1 else round(len(response) * cum / total_chars)
        piece = response[done:upto]
        done = upto
        if piece or i == len(tchunks) - 1:
            out.append([float(at_ms), piece])
    if out:
        out[-1][0] = float(template["latency_ms"])
    return out


def ingest(batch_dir: Path, *, answered_by: str, latency_from: list[Path] | None = None,
           latency_ms_fallback: float | None = None) -> dict:
    batch_dir = Path(batch_dir)
    manifest = {r["id"]: r for r in read_manifest(batch_dir)}
    dirs = sorted({Path(r["cache_dir"]) for r in manifest.values()} | {Path(d) for d in latency_from or []})
    pool = _timing_pool(dirs)
    if not pool and latency_ms_fallback is None:
        raise NoLatencySource(
            "no recorded real LLM call in " + ", ".join(map(str, dirs))
            + " to borrow timing from; pass --latency-from <cache dir> or --latency-ms-fallback <ms>")
    answers: dict[str, str] = {}
    rejected: list[dict] = []
    for f in sorted(batch_dir.glob("answers_*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                a = json.loads(line)
                answers[str(a["id"])] = a["response"]
            except (ValueError, KeyError, TypeError) as exc:
                rejected.append({"id": None, "reason": f"unparseable answer line in {f.name}: {exc}"})
    written = 0
    for aid, response in answers.items():
        row = manifest.get(aid)
        if row is None:
            rejected.append({"id": aid, "reason": "unknown id (not in this batch's manifest)"})
            continue
        if not isinstance(response, str) or not response.strip():
            rejected.append({"id": aid, "reason": "empty response"})
            continue
        text = _clean(response)
        if _wants_json(row):
            try:
                if not isinstance(json.loads(text), dict):
                    raise ValueError("not an object")
            except ValueError as exc:
                rejected.append({"id": aid, "reason": f"the prompt demands a JSON object; answer is not one ({exc})"})
                continue
        if LLMResponseCache._cache_key(row["model"], row["system"], row["user"]) != row["key"]:
            rejected.append({"id": aid, "reason": "manifest key does not match its prompt"})
            continue
        path = Path(row["cache_dir"]) / f"{row['key']}.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except ValueError:
                existing = {}
            if _is_real(existing):
                rejected.append({"id": aid, "reason": "a real entry exists for this prompt; never overwritten"})
                continue
        if pool:
            src, tmpl = pool[int(hashlib.sha256(row["key"].encode()).hexdigest(), 16) % len(pool)]
            chunks = _simulated_chunks(text, tmpl)
            latency, source = float(tmpl["latency_ms"]), f"recorded:{src}"
        else:
            latency = float(latency_ms_fallback)
            chunks, source = [[latency, text]], f"fallback:{latency:g}ms"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "model": f"{row['model']}@cc-subagent:{answered_by}",
            "provenance": f"cc-subagent:{answered_by}",
            "system": row["system"], "user": row["user"], "response": text,
            "chunks": chunks, "latency_ms": latency, "first_chunk_ms": chunks[0][0],
            "latency_simulated": True, "latency_source": source,
            "answered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, indent=2))
        written += 1
    summary = {"written": written, "rejected": rejected,
               "unanswered": sum(1 for i in manifest if i not in answers),
               "answered_by": answered_by, "timing_pool": len(pool)}
    (batch_dir / "ingest_report.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# run-report line
# ---------------------------------------------------------------------------

def sources_line(llm_stats: dict | None, rules_stats: dict | None) -> str:
    """Where every coach answer of a run came from — one line for the log
    and the report."""
    src = (llm_stats or {}).get("sources") or {}
    cc = int(src.get("cc-subagent") or 0)
    models = ",".join(sorted(((llm_stats or {}).get("cc_models") or {}).keys())) or "?"
    rules = rules_stats or {}
    line = (f"coach answers: real API {int(src.get('api') or 0)} · cache {int(src.get('cache') or 0)} · "
            f"cc-subagent {cc}{f' ({models})' if cc else ''} · rules {int(rules.get('fires') or 0)} fires / "
            f"{int(rules.get('turns') or 0)} turns · offline misses {int((llm_stats or {}).get('offline_misses') or 0)}")
    if cc:
        line += " — cc-subagent entries are a stand-in (not the coach model) and their latency is simulated " \
                "from recorded real calls"
    return line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="miss log(s) -> batch files for CC subagents")
    e.add_argument("miss_logs", nargs="+", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--batch-size", type=int, default=20)
    i = sub.add_parser("ingest", help="answers_NN.jsonl -> labelled cache entries")
    i.add_argument("batch_dir", type=Path)
    i.add_argument("--answered-by", required=True, help="the subagent's model, e.g. sonnet")
    i.add_argument("--latency-from", type=Path, action="append", default=[],
                   help="extra cache dir(s) of recorded real calls to borrow timing from")
    i.add_argument("--latency-ms-fallback", type=float, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "export":
        print(json.dumps(export(args.miss_logs, args.out, batch_size=args.batch_size), indent=2))
        return 0
    s = ingest(args.batch_dir, answered_by=args.answered_by, latency_from=args.latency_from,
               latency_ms_fallback=args.latency_ms_fallback)
    print(json.dumps(s, indent=2))
    return 1 if s["rejected"] or s["unanswered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
