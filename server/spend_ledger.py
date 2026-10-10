"""Shared spend ledger for paid API calls made by offline tooling (replays,
tuning runs). One JSON file, file-locked so parallel agents/processes share
one budget; a call that would push the total past the cap is REFUSED before
it is made (SpendCapExceeded), never made-then-regretted.

File: $MINDSHIFT_SPEND_LEDGER (default <repo-main>/tmp/recordings/api-spend.json)
Cap:  $MINDSHIFT_SPEND_CAP_USD (default 5.00), also stored in the file; the
      LOWER of the two wins, so nobody can raise it by env alone.
Only used when MINDSHIFT_SPEND_LEDGER_ON=1 (the replay tooling sets it); the
production server never touches it.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

# Prices in USD (2026-10). Haiku 4.5: $1 / M input, $5 / M output.
# Deepgram nova-3 pre-recorded: ~$0.0043 / audio minute.
PRICES = {
    "haiku_in_per_tok": 1.0 / 1_000_000,
    "haiku_out_per_tok": 5.0 / 1_000_000,
    "deepgram_per_min": 0.0043,
}
DEFAULT_CAP = 5.0


class SpendCapExceeded(RuntimeError):
    pass


def enabled() -> bool:
    return os.getenv("MINDSHIFT_SPEND_LEDGER_ON", "").strip() == "1"


def ledger_path() -> Path:
    env = os.getenv("MINDSHIFT_SPEND_LEDGER")
    if env:
        return Path(env)
    return Path("/Users/sagearbor/projects/githubs/mindshift/tmp/recordings/api-spend.json")


@contextmanager
def _locked():
    p = ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    lock = p.with_suffix(".lock")
    with open(lock, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            data = json.loads(p.read_text()) if p.exists() else {"cap_usd": DEFAULT_CAP, "total_usd": 0.0, "calls": []}
            yield data
            p.write_text(json.dumps(data, indent=1))
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _cap(data: dict) -> float:
    env = os.getenv("MINDSHIFT_SPEND_CAP_USD")
    caps = [float(data.get("cap_usd", DEFAULT_CAP))]
    if env:
        caps.append(float(env))
    return min(caps)


def reserve(service: str, est_usd: float, item: str = "") -> None:
    """Refuse (raise) if est_usd would exceed the cap. No-op when disabled."""
    if not enabled():
        return
    with _locked() as data:
        if data["total_usd"] + est_usd > _cap(data):
            raise SpendCapExceeded(
                f"{service}: est ${est_usd:.4f} would exceed cap ${_cap(data):.2f} "
                f"(spent ${data['total_usd']:.4f})")


def record(service: str, usd: float, item: str = "", **meta) -> None:
    if not enabled():
        return
    with _locked() as data:
        data["total_usd"] = round(data["total_usd"] + usd, 6)
        data["calls"].append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "service": service,
                              "usd": round(usd, 6), "item": item, **meta})


def llm_cost(system: str, user: str, out_text: str = "", max_tokens: int = 0) -> float:
    tin = (len(system) + len(user)) / 4.0
    tout = (len(out_text) / 4.0) if out_text else max_tokens
    return tin * PRICES["haiku_in_per_tok"] + tout * PRICES["haiku_out_per_tok"]


def deepgram_cost(n_bytes_pcm16_16k: int) -> float:
    minutes = n_bytes_pcm16_16k / (16000 * 2) / 60.0
    return minutes * PRICES["deepgram_per_min"]
