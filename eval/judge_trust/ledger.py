"""Cost ledger + response cache for every judge/labeller API call.

* Every real call appends one JSON line to OUT/calls.jsonl (model, task, tokens,
  estimated USD).  Cached replays are not billed and not re-logged.
* A hard cap (JT_CAP_USD, default 4.50, never above 5.00) is checked BEFORE each
  call; once the running estimate passes it, BudgetExceeded is raised and the
  run stops.
* Prices are per 1M tokens, taken from the public price pages on 2026-11-05
  (developers.openai.com/api/docs/pricing, aistudio.google.com/docs/pricing,
  Anthropic model table).  Where a model is missing from the page we use a
  deliberately HIGH estimate (marked "assumed") so the cap errs on the safe side.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

from env import OUT

# model -> (text_in, audio_in, out, note)
PRICES = {
    "claude-sonnet-5-5": (2.00, None, 10.00, "Anthropic table"),
    "claude-haiku-5-5": (0.10, None, 0.50, "Anthropic table"),
    "claude-opus-5-5": (4.00, None, 20.00, "Anthropic table"),
    "gpt-5-mini": (0.25, None, 2.00, "OpenAI page"),
    "gpt-6.1-terra": (1.00, None, 5.00, "OpenAI page"),
    "gpt-audio-1.5": (2.50, 32.00, 10.00, "OpenAI page"),
    "gemini-3.8-flash": (0.75, 0.75, 3.75, "Gemini page (2026 price; audio not listed separately)"),
    "gemini-3.5-flash": (1.50, 1.50, 7.50, "assumed (not on page)"),
    "gemini-3.5-flash-lite": (0.30, 0.30, 2.50, "Gemini page"),
    "gemini-3.1-flash-lite": (0.25, 0.50, 1.50, "Gemini page"),
    "gemini-3.1-pro-preview": (2.00, 2.00, 12.00, "Gemini page"),
}
CAP = min(float(os.environ.get("JT_CAP_USD", "4.50")), 5.00)
_lock = threading.Lock()
LOG = OUT / "calls.jsonl"
CACHE = OUT / "cache"
MAX_LOG_LINES = 200_000  # bounded log


class BudgetExceeded(RuntimeError):
    pass


def price(model: str, text_in: int, audio_in: int, out: int) -> float:
    p = PRICES.get(model)
    if p is None:
        return 0.0  # local models
    ti, ai, o, _ = p
    return (text_in * ti + audio_in * (ai if ai is not None else ti) + out * o) / 1e6


def spent() -> float:
    if not LOG.exists():
        return 0.0
    tot = 0.0
    with open(LOG) as f:
        for line in f:
            try:
                tot += json.loads(line).get("usd", 0.0)
            except json.JSONDecodeError:
                pass
    return tot


_spent_cache = {"v": None}


def _running() -> float:
    if _spent_cache["v"] is None:
        _spent_cache["v"] = spent()
    return _spent_cache["v"]


def check_budget(expected: float = 0.0) -> None:
    with _lock:
        # re-read the shared log every time so parallel runner processes see each other's spend
        _spent_cache["v"] = spent()
        if _running() + expected > CAP:
            raise BudgetExceeded(f"spend estimate ${_running():.3f} + ${expected:.3f} would pass cap ${CAP:.2f}")


def record(task: str, model: str, text_in: int, audio_in: int, out: int, extra: dict | None = None) -> float:
    usd = price(model, text_in, audio_in, out)
    with _lock:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        n = sum(1 for _ in open(LOG)) if LOG.exists() and LOG.stat().st_size < 50_000_000 else 0
        if n < MAX_LOG_LINES:
            with open(LOG, "a") as f:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": task, "model": model,
                                    "text_in": text_in, "audio_in": audio_in, "out": out, "usd": round(usd, 6),
                                    **(extra or {})}) + "\n")
        _spent_cache["v"] = _running() + usd
    return usd


def cache_key(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\x00")
    return h.hexdigest()


def cache_get(key: str):
    p = CACHE / key[:2] / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def cache_put(key: str, value) -> None:
    p = CACHE / key[:2] / f"{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False))
