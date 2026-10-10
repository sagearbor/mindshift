#!/usr/bin/env python3
"""yt_prepare.py — make the open-web yt_* inbox items replay-ready at $0
(local faster-whisper transcript, speakers, wearer, held-out enrollment
tail). Logic and file layout: scripts/recreplay/yt_prep.py.

    tmp/venv/bin/python scripts/yt_prepare.py                 # every tmp/recordings/inbox/yt_*
    tmp/venv/bin/python scripts/yt_prepare.py yt_1DUmETzgOvM  # one item

Then: tmp/venv/bin/python scripts/recording_replay.py tmp/recordings/inbox/yt_<id> [--skip-server]
(open-web items default to --stt whisper; never Deepgram, never Gemini).
Summary: tmp/recordings/inbox/yt_prepared.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "server", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from recreplay import pipeline, yt_prep  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="*")
    a = ap.parse_args(argv)
    inbox = pipeline.RECORDINGS / "inbox"
    folders = [inbox / n for n in a.items] if a.items else sorted(p for p in inbox.glob("yt_*") if p.is_dir())
    summary_path = inbox / "yt_prepared.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for f in folders:
        t0 = time.monotonic()
        try:
            row = yt_prep.prepare_item(f, work_root=pipeline.RECORDINGS / "work")
            row["wall_s"] = round(time.monotonic() - t0, 1)
        except Exception as exc:  # noqa: BLE001 — one bad item never stops the batch
            row = {"name": f.name, "error": f"{type(exc).__name__}: {exc}"[:400]}
        summary[f.name] = row
        summary_path.write_text(json.dumps(summary, indent=1))
        print(f"[yt-prepare] {json.dumps(row)}", flush=True)
    ok = [r for r in summary.values() if not r.get("error")]
    print(f"[yt-prepare] ready: {len(ok)}/{len(summary)}; replay hours "
          f"{sum(r['replay_s'] for r in ok) / 3600:.2f}; with held-out print: {sum(1 for r in ok if r['voiceprint'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
