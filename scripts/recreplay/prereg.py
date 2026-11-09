#!/usr/bin/env python3
"""Append one pre-registration entry (JSON on argv[1] or stdin) to the eval
prereg logs. Written BEFORE the run it covers, never edited after.

    tmp/venv/bin/python scripts/recreplay/prereg.py entry.json
"""
import json
import sys
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
LOGS = [WT / "tmp" / "nudge-eval" / "prereg-log.jsonl",
        WT / "tmp" / "recordings" / "judge" / "validity_F" / "prereg-log.jsonl"]

entry = json.loads(Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else sys.stdin.read())
entry = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "agent": "nudge-sprint", **entry}
for p in LOGS:
    if p.parent.exists():
        with open(p, "a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
print("logged", entry["exp"])
