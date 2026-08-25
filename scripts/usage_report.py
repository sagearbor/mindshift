#!/usr/bin/env python3
"""usage_report.py — what MindShift is costing, per account, in dollars.

Reads the owner rollup (``GET /admin/usage``, allowlisted by
``MINDSHIFT_ADMIN_UIDS``) and prints a per-uid table plus a projected monthly
bill at the observed run rate. The unit prices live in ONE constant block
below — every number in it is a public list price with its source URL and the
date it was read; the derivation of the per-call and per-therapist figures is
in ``docs/plans/2026-08-25-cost-model.md``.

Usage
-----
  # against a running server (an admin Firebase ID token)
  python scripts/usage_report.py --id-token "$TOKEN" --since 2026-08-01

  # against a saved payload (no network; e.g. from curl or CI)
  python scripts/usage_report.py --json usage.json

  # machine-readable, for a spreadsheet
  python scripts/usage_report.py --id-token "$TOKEN" --csv

``--base-url`` (or ``MINDSHIFT_API_URL``) defaults to the production Cloud Run
URL. ``--model`` picks the price row to bill LLM tokens against; it defaults
to ``MINDSHIFT_MODEL`` when that is set, else Haiku 4.5 (what the live loop
runs on today).

Exit status: 0 when a table printed, 1 on an error (auth, network, bad JSON).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

DEFAULT_BASE_URL = os.environ.get(
    "MINDSHIFT_API_URL", "https://mindshift-api-e5zja2g5bq-uc.a.run.app",
)

# ===========================================================================
# PRICES — public list prices. Every entry: value, unit, source, date read.
# Nothing here is estimated or remembered; update the whole block together
# and move the date with it. Cross-checked in
# docs/plans/2026-08-25-cost-model.md, which shows the arithmetic.
# ===========================================================================

PRICES_READ_ON = "2026-08-25"

# --- Anthropic (https://docs.claude.com/en/docs/about-claude/pricing) ------
# USD per 1,000,000 tokens. Cache reads/writes are billed differently by the
# API (~0.1x read, ~1.25x write); the counters keep them separate so this
# stays honest when prompt caching is eventually switched on.
LLM_PRICES: dict[str, dict[str, float]] = {
    # model id            input   output  cache_read  cache_write
    "claude-haiku-4-5":  {"input": 1.00, "output": 5.00,
                          "cache_read": 0.10, "cache_write": 1.25},
    "claude-sonnet-5":   {"input": 3.00, "output": 15.00,
                          "cache_read": 0.30, "cache_write": 3.75},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00,
                          "cache_read": 0.30, "cache_write": 3.75},
    "claude-opus-5":     {"input": 5.00, "output": 25.00,
                          "cache_read": 0.50, "cache_write": 6.25},
    # The repo's historical default (llm_client.DEFAULT_MODEL). Legacy Haiku 3
    # list price, kept so a deployment still on it prices correctly.
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25,
                                "cache_read": 0.03, "cache_write": 0.30},
}
DEFAULT_LLM_MODEL = "claude-haiku-4-5"

# --- Deepgram (https://deepgram.com/pricing) -------------------------------
# Nova-3 Monolingual, pay-as-you-go. The streaming line currently shows a
# promotional $0.0048/min against a REGULAR price of $0.0077/min — we budget
# against the regular price, because a promo expiring is not a cost surprise
# anyone should have to discover from an invoice.
DEEPGRAM_STREAMING_USD_PER_MIN = 0.0077       # promo today: 0.0048
DEEPGRAM_PRERECORDED_USD_PER_MIN = 0.0043

# --- Google Cloud Storage (https://cloud.google.com/storage/pricing) -------
# us-central1 single-region, Standard class.
GCS_STORAGE_USD_PER_GB_MONTH = 0.020
GCS_CLASS_A_USD_PER_1K_OPS = 0.005            # writes/lists
GCS_CLASS_B_USD_PER_1K_OPS = 0.0004           # reads
# Egress to the internet, first 10 TiB/month tier.
GCS_EGRESS_USD_PER_GB = 0.12

# --- Google Cloud Run (https://cloud.google.com/run/pricing) ---------------
# Request-based billing, tier-1 region (us-central1), list price.
CLOUD_RUN_USD_PER_VCPU_SECOND = 0.000024
CLOUD_RUN_USD_PER_GIB_SECOND = 0.0000025
CLOUD_RUN_USD_PER_MILLION_REQUESTS = 0.40

PRICE_SOURCES = {
    "Anthropic": "https://docs.claude.com/en/docs/about-claude/pricing",
    "Deepgram": "https://deepgram.com/pricing",
    "Cloud Storage": "https://cloud.google.com/storage/pricing",
    "Cloud Run": "https://cloud.google.com/run/pricing",
}

# ===========================================================================


def llm_prices(model: str) -> dict[str, float]:
    if model in LLM_PRICES:
        return LLM_PRICES[model]
    print(
        f"warning: no price row for model {model!r} — billing at "
        f"{DEFAULT_LLM_MODEL} rates. Add it to LLM_PRICES.",
        file=sys.stderr,
    )
    return LLM_PRICES[DEFAULT_LLM_MODEL]


def cost_for(row: dict, model: str) -> dict[str, float]:
    """Dollar cost of one user's rollup row, broken out by vendor."""
    price = llm_prices(model)
    per_site = row.get("llm") or {}
    inp = out = cache_read = cache_write = 0.0
    for counters in per_site.values():
        # hedge_extra_input_tokens are the losing hedged request's prompt —
        # billed at the input rate, exactly like the winner's.
        inp += counters.get("input_tokens", 0) + counters.get(
            "hedge_extra_input_tokens", 0,
        )
        out += counters.get("output_tokens", 0)
        cache_read += counters.get("cache_read_input_tokens", 0)
        cache_write += counters.get("cache_creation_input_tokens", 0)
    llm = (
        inp * price["input"] + out * price["output"]
        + cache_read * price["cache_read"] + cache_write * price["cache_write"]
    ) / 1_000_000
    stt = (row.get("stt_seconds", 0) / 60.0) * DEEPGRAM_STREAMING_USD_PER_MIN
    egress = (row.get("model_bytes", 0) / (1024 ** 3)) * GCS_EGRESS_USD_PER_GB
    return {"llm": llm, "stt": stt, "egress": egress, "total": llm + stt + egress}


def fetch(base_url: str, since: str | None, id_token: str) -> dict:
    query = f"?since={urllib.parse.quote(since)}" if since else ""
    url = f"{base_url.rstrip('/')}/admin/usage{query}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {id_token}",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _fmt_usd(value: float) -> str:
    return f"${value:,.2f}" if value >= 0.01 or value == 0 else f"${value:.4f}"


def render_table(payload: dict, model: str, out=sys.stdout) -> float:
    users = payload.get("users") or []
    days = max(1, int(payload.get("days") or 1))
    headers = [
        "uid", "llm calls", "in tok", "out tok", "STT min", "live min",
        "model dl", "calls", "$ llm", "$ stt", "$ total",
    ]
    rows = []
    grand = 0.0
    for row in sorted(
        users, key=lambda r: -cost_for(r, model)["total"],
    ):
        cost = cost_for(row, model)
        grand += cost["total"]
        rows.append([
            row.get("uid", "?"),
            f"{row.get('llm_calls', 0):,.0f}",
            f"{row.get('llm_input_tokens', 0):,.0f}",
            f"{row.get('llm_output_tokens', 0):,.0f}",
            f"{row.get('stt_seconds', 0) / 60:,.1f}",
            f"{row.get('live_minutes', 0):,.1f}",
            f"{row.get('model_downloads', 0):,.0f}",
            f"{row.get('calls_started', 0):,.0f}",
            _fmt_usd(cost["llm"]),
            _fmt_usd(cost["stt"]),
            _fmt_usd(cost["total"]),
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells):
        return "  ".join(
            c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])
            for i, c in enumerate(cells)
        )

    print(
        f"MindShift usage {payload.get('since')} → {payload.get('until')} "
        f"({days} day{'s' if days != 1 else ''}, model={model})",
        file=out,
    )
    if payload.get("persistent") is False:
        print(
            "  NOTE: the server has no recordings bucket configured, so these "
            "counters are one process's memory only.",
            file=out,
        )
    print(file=out)
    print(line(headers), file=out)
    print("  ".join("-" * w for w in widths), file=out)
    for row in rows:
        print(line(row), file=out)
    if not rows:
        print("(no usage recorded in this window)", file=out)

    print(file=out)
    print(f"Window total:      {_fmt_usd(grand)} over {days} day(s)", file=out)
    per_day = grand / days
    print(f"Run rate:          {_fmt_usd(per_day)}/day", file=out)
    print(f"Projected month:   {_fmt_usd(per_day * 30)} (30 × the run rate)", file=out)
    if users:
        print(
            f"Per active account: {_fmt_usd(per_day * 30 / len(users))}/month "
            f"across {len(users)} account(s) seen in this window",
            file=out,
        )
    print(file=out)
    print(f"Prices read {PRICES_READ_ON}: " + ", ".join(
        f"{name} {url}" for name, url in PRICE_SOURCES.items()
    ), file=out)
    print(
        "Not included: GCS storage/ops for recordings, Cloud Run compute, TTS. "
        "See docs/plans/2026-08-25-cost-model.md.",
        file=out,
    )
    return grand


def render_csv(payload: dict, model: str, out=sys.stdout) -> None:
    writer = csv.writer(out)
    writer.writerow([
        "uid", "llm_calls", "llm_input_tokens", "llm_output_tokens",
        "stt_seconds", "live_minutes", "model_downloads", "model_bytes",
        "calls_started", "usd_llm", "usd_stt", "usd_egress", "usd_total",
    ])
    for row in payload.get("users") or []:
        cost = cost_for(row, model)
        writer.writerow([
            row.get("uid", ""),
            row.get("llm_calls", 0), row.get("llm_input_tokens", 0),
            row.get("llm_output_tokens", 0), row.get("stt_seconds", 0),
            row.get("live_minutes", 0), row.get("model_downloads", 0),
            row.get("model_bytes", 0), row.get("calls_started", 0),
            round(cost["llm"], 6), round(cost["stt"], 6),
            round(cost["egress"], 6), round(cost["total"], 6),
        ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-account MindShift usage and its dollar cost.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--id-token", default=os.environ.get("MINDSHIFT_ADMIN_ID_TOKEN", ""),
        help="Firebase ID token of a uid in MINDSHIFT_ADMIN_UIDS.",
    )
    parser.add_argument(
        "--since", default=None,
        help="First UTC day, YYYY-MM-DD (default: today, server-side).",
    )
    parser.add_argument(
        "--json", dest="json_path", default=None,
        help="Read a saved /admin/usage payload instead of calling the server.",
    )
    parser.add_argument(
        "--model", default=os.environ.get("MINDSHIFT_MODEL") or DEFAULT_LLM_MODEL,
        help="Price LLM tokens at this model's rates.",
    )
    parser.add_argument("--csv", action="store_true", help="CSV instead of a table.")
    args = parser.parse_args(argv)

    if args.json_path:
        try:
            with open(args.json_path) as fh:
                payload = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"error: could not read {args.json_path}: {exc}", file=sys.stderr)
            return 1
    else:
        if not args.id_token:
            print(
                "error: --id-token (or MINDSHIFT_ADMIN_ID_TOKEN) is required "
                "without --json",
                file=sys.stderr,
            )
            return 1
        try:
            payload = fetch(args.base_url, args.since, args.id_token)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:400]
            if exc.code == 404:
                print(
                    "error: 404 — this uid is not in MINDSHIFT_ADMIN_UIDS on "
                    f"{args.base_url} (the endpoint 404s rather than 403s).",
                    file=sys.stderr,
                )
            else:
                print(f"error: HTTP {exc.code} — {body}", file=sys.stderr)
            return 1
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.csv:
        render_csv(payload, args.model)
    else:
        render_table(payload, args.model)
    return 0


if __name__ == "__main__":
    _ = datetime.now(timezone.utc)  # fail fast on a broken clock/tz build
    sys.exit(main())
