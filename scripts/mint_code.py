#!/usr/bin/env python3
"""Mint / list / revoke MindShift redeem codes (server/plans.py).

Uses the same Firestore access as the other server scripts: Application
Default Credentials and the project in --project / MINDSHIFT_FIRESTORE_PROJECT
(no key files, nothing secret is read or printed).

  scripts/mint_code.py mint --plan pro --days 30 --uses 5 --note 'sales friend'
  scripts/mint_code.py list
  scripts/mint_code.py revoke ABCD-2345-EFGH
  scripts/mint_code.py grant --uid <firebase uid> --plan pro --days 90   # direct admin grant

`mint` prints each new code once; hand it to the person. Users enter it in
Settings -> Plan -> Redeem a code.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))

import asyncio  # noqa: E402

import plans  # noqa: E402


def _store(project):
    project = project or os.environ.get("MINDSHIFT_FIRESTORE_PROJECT")
    if not project:
        sys.exit("Set --project or MINDSHIFT_FIRESTORE_PROJECT (the Firestore project).")
    return plans.FirestorePlanStore(project)


def _level(value: str) -> str:
    if value not in plans.LEVELS or value == "free":
        raise argparse.ArgumentTypeError(
            f"plan must be one of {[l for l in plans.LEVELS if l != 'free']}")
    return value


async def cmd_mint(args) -> None:
    store = _store(args.project)
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=args.valid_days) if args.valid_days else None
    for _ in range(args.count):
        raw = None
        for _attempt in range(5):
            display = plans.generate_code()
            raw = plans.normalize_code(display)
            doc = {
                "code": raw, "plan": args.plan, "days": args.days,
                "max_uses": args.uses, "uses": 0, "expires_at": expires,
                "revoked": False, "note": args.note or "", "created_by": getpass.getuser(),
                "created_at": now,
            }
            try:
                await store.create_code(doc)
                break
            except ValueError:
                raw = None
        if raw is None:
            sys.exit("could not generate a unique code")
        print(f"{display}  plan={args.plan} days={args.days} uses={args.uses}"
              + (f" note={args.note!r}" if args.note else ""))


async def cmd_list(args) -> None:
    store = _store(args.project)
    rows = sorted(await store.list_codes(), key=lambda d: str(d.get("created_at")))
    for d in rows:
        c = d["code"]
        shown = "-".join(c[i:i + 4] for i in range(0, len(c), 4))
        state = "REVOKED" if d.get("revoked") else "active"
        print(f"{shown}  {state:7} {d.get('plan')} {d.get('days')}d "
              f"used {d.get('uses', 0)}/{d.get('max_uses') or 'unlimited'} "
              f"by={d.get('created_by', '?')} note={d.get('note', '')!r}")
    if not rows:
        print("(no codes)")


async def cmd_revoke(args) -> None:
    store = _store(args.project)
    ok = await store.revoke_code(plans.normalize_code(args.code))
    print("revoked" if ok else "no such code")
    if not ok:
        sys.exit(1)


async def cmd_grant(args) -> None:
    store = _store(args.project)
    now = datetime.now(timezone.utc)
    await store.set_fields(args.uid, {
        "plan": args.plan, "plan_source": "admin", "plan_updated_at": now,
        "plan_expires_at": now + timedelta(days=args.days) if args.days else None,
    })
    print(f"granted {args.plan} to {args.uid} "
          + (f"for {args.days} days" if args.days else "with no expiry"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", help="Firestore project (default: $MINDSHIFT_FIRESTORE_PROJECT)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mint", help="create code(s)")
    m.add_argument("--plan", type=_level, default="pro")
    m.add_argument("--days", type=int, required=True, help="days of plan each redemption grants")
    m.add_argument("--uses", type=int, default=1, help="how many accounts may redeem (default 1)")
    m.add_argument("--valid-days", type=int, default=0, help="code itself stops working after N days")
    m.add_argument("--count", type=int, default=1)
    m.add_argument("--note", default="")
    m.set_defaults(fn=cmd_mint)

    sub.add_parser("list", help="show all codes and usage").set_defaults(fn=cmd_list)

    r = sub.add_parser("revoke", help="stop a code working (existing grants stay)")
    r.add_argument("code")
    r.set_defaults(fn=cmd_revoke)

    g = sub.add_parser("grant", help="grant a plan directly to a uid (source=admin)")
    g.add_argument("--uid", required=True)
    g.add_argument("--plan", type=_level, default="pro")
    g.add_argument("--days", type=int, default=0, help="0 = no expiry")
    g.set_defaults(fn=cmd_grant)

    args = ap.parse_args()
    if getattr(args, "days", 1) is not None and getattr(args, "days", 1) < 0:
        sys.exit("--days must be positive")
    if args.cmd == "mint" and (args.days <= 0 or args.uses < 1 or not 1 <= args.count <= 100):
        sys.exit("--days > 0, --uses >= 1, --count 1..100")
    asyncio.run(args.fn(args))


if __name__ == "__main__":
    main()
