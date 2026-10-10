#!/usr/bin/env python3
"""recording_replay.py — what WOULD the live coach have done in this real
conversation, and how fast? One recording in, a timeline report + a
re-runnable regression fixture out. See tmp/recordings/README.md (owner
guide) and docs/recording-annotation-format.md (the sidecar contract).

    tmp/venv/bin/python scripts/recording_replay.py tmp/recordings/inbox/<name>
    tmp/venv/bin/python scripts/recording_replay.py --all
    tmp/venv/bin/python scripts/recording_replay.py --app-latest --email sagearbor@gmail.com
    tmp/venv/bin/python scripts/recording_replay.py --fixture <name>      # offline regression re-run
    tmp/venv/bin/python scripts/recording_replay.py --fixtures            # all of them
    tmp/venv/bin/python scripts/recording_replay.py --set clip30_t1 --jobs 4   # a 30 s clip set (recreplay/clips.py)

Per recording: ffmpeg -> 16 kHz mono; Deepgram reference transcript
(cached); annotations re-timed to Deepgram's words; which voice is the
owner (notes / annotation / voiceprint); the phone's REAL on-device loop
(apps/mobile replay, Silero + segmenter + ECAPA) -> turn_local stream; the
SERVER in real time (local uvicorn by default, ``--url`` for a deployed one)
-> every event with its arrival time; score; report at
tmp/recordings/reports/<name>.html; freeze tmp/recordings/fixtures/<name>/.

Exit status: 0 ok; 1 a recording failed or (--fixture) regressed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT / "server", REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from recreplay import stt  # noqa: E402

# Before anything imports server/main.py: the repo .env (the main checkout's
# when this runs from a worktree), never overriding real environment.
stt.load_dotenv_into_environ()

from recreplay import app_pull, fixture, pipeline, report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*", type=Path, help="drop folder(s): tmp/recordings/inbox/<name>")
    ap.add_argument("--all", action="store_true", help="every folder in tmp/recordings/inbox")
    ap.add_argument("--app-latest", action="store_true", help="pull the newest app recording (GCS) into the inbox, then run it")
    ap.add_argument("--recording-id", help="with --app-latest: that recording instead of the newest")
    ap.add_argument("--uid", help="with --app-latest: the account uid (skips the email lookup)")
    ap.add_argument("--fixture", action="append", default=[], help="re-run a frozen fixture OFFLINE and compare to its baseline")
    ap.add_argument("--fixtures", action="store_true", help="re-run every frozen fixture")
    ap.add_argument("--set", action="append", default=[], metavar="NAME",
                    help="run a clip set from tmp/recordings/inbox/clip30_manifest.json (clip30_t1|t2|t3|all); "
                         "implies --no-freeze")
    ap.add_argument("--jobs", type=int, default=1,
                    help="with --set/--all/folders: replay this many items at once (each its own local server)")
    g = ap.add_argument_group("session")
    g.add_argument("--mode", choices=["earpiece", "speaker", "therapist", "room"])
    g.add_argument("--relationship", choices=["child", "partner", "parent", "coworker", "friend", "other"])
    g.add_argument("--library", help="comma-separated library item ids (needs --url: the local server has no library)")
    g.add_argument("--context", help="session_context (default: the notes' setting: line)")
    g.add_argument("--wearer", help="force the owner's Deepgram label, e.g. 'Speaker B'")
    g.add_argument("--phone-tone", choices=["annotation", "neutral"], help="phone tone stand-in (default: annotation if present)")
    g.add_argument("--enroll", choices=["profile", "same", "none"], help="phone voiceprint (default: profile if owner_profile.json exists)")
    g.add_argument("--profile", type=Path, help="owner voiceprint JSON (default tmp/private_fixtures/owner_profile.json)")
    g.add_argument("--window", type=float, default=None, help="seconds after a moment a coach line still counts (default 6)")
    s = ap.add_argument_group("server")
    s.add_argument("--url", help="a deployed server (e.g. the Cloud Run URL) instead of a local uvicorn: true latency")
    s.add_argument("--id-token", help="Firebase ID token for --url")
    s.add_argument("--email", help="account email (--url sign-in with --password; --app-latest lookup)")
    s.add_argument("--password", help="account password for --url")
    s.add_argument("--speed", type=float, default=1.0, help="stream N x faster than real time (latency then means less)")
    o = ap.add_argument_group("output")
    o.add_argument("--no-freeze", action="store_true", help="don't write/refresh tmp/recordings/fixtures/<name>")
    o.add_argument("--rebaseline", action="store_true", help="overwrite the fixture's baseline with this run")
    o.add_argument("--skip-server", action="store_true", help="phone side only")
    o.add_argument("--corpus-summary", action="store_true",
                   help="write tmp/recordings/reports/corpus-summary.html over every corpus item run so far (scripts/corpus_to_inbox.py)")
    return ap


def _options(args) -> pipeline.RunOptions:
    opts = pipeline.RunOptions(
        mode=args.mode, relationship=args.relationship,
        library_item_ids=[x.strip() for x in args.library.split(",") if x.strip()] if args.library else None,
        session_context=args.context, url=args.url, id_token=args.id_token, email=args.email if args.url else None,
        password=args.password, speed=args.speed, phone_tone=args.phone_tone, enroll=args.enroll, wearer=args.wearer,
        skip_server=args.skip_server,
    )
    if args.window is not None:
        opts.moment_window_s = args.window
    return opts


def run_folder(folder: Path, args) -> bool:
    inp = pipeline.inputs_from_inbox(folder, profile=args.profile)
    opts = _options(args)
    bundle = pipeline.run(inp, opts)
    out = report.write_report(bundle, pipeline.RECORDINGS / "reports" / f"{inp.name}.html")
    print(f"[recording-replay] report: {out}")
    if not args.no_freeze and bundle.get("server") and not args.url:
        fx = fixture.freeze(bundle, inp, rebaseline=args.rebaseline)
        print(f"[recording-replay] fixture: {fx}")
    elif args.url:
        print("[recording-replay] --url run: not frozen as a fixture (the deployed LLM is not cached)")
    return not bundle.get("server_error") and not bundle.get("phone_error")


def run_fixture(fx: Path) -> bool:
    with tempfile.TemporaryDirectory(prefix=f"recreplay-{fx.name}-") as tmp:
        inp, baseline = fixture.inputs_from_fixture(fx, Path(tmp))
        bundle = pipeline.run(inp, fixture.options_from_baseline(baseline))
        cur = fixture.metrics(bundle)
        reg, notes = fixture.compare(baseline, cur)
        out = report.write_report(bundle, pipeline.RECORDINGS / "reports" / f"{inp.name}.rerun.html")
    print(f"[recording-replay] {fx.name}: re-run report {out}")
    for n in notes:
        print(f"[recording-replay]   note: {n}")
    for r in reg:
        print(f"[recording-replay]   REGRESSION: {r}")
    print(f"[recording-replay] {fx.name}: {'OK' if not reg else f'{len(reg)} regression(s)'}")
    return not reg


CLIP_PREFIX = "clip30_"


def clip_set_folders(name: str) -> list[Path]:
    """Inbox folders of a clip set (scripts/recreplay/clips.py's manifest)."""
    inbox = pipeline.RECORDINGS / "inbox"
    man = json.loads((inbox / "clip30_manifest.json").read_text())
    sets = man["sets"]
    if name not in sets and name != "clip30_all":
        raise SystemExit(f"unknown clip set {name!r}; have {sorted(sets)} (or clip30_all)")
    names = [n for k, v in sorted(sets.items()) if k == name or name == "clip30_all" for n in v]
    return [inbox / n for n in names]


def _child_argv(argv: list[str]) -> list[str]:
    """argv minus the item selection (folders, --set, --all, --jobs)."""
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
            continue
        if x in ("--set", "--jobs"):
            skip = True
            continue
        if x.startswith(("--set=", "--jobs=")) or x == "--all" or not x.startswith("-") and Path(x).is_dir():
            continue
        out.append(x)
    return out


def run_parallel(folders: list[Path], jobs: int, argv: list[str]) -> bool:
    """Each folder in its own child process (own local server on a free
    port), ``jobs`` at a time; logs to tmp/recordings/logs/<name>.log."""
    logs = pipeline.RECORDINGS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, str(Path(__file__).resolve())] + _child_argv(argv) + ["--no-freeze"]
    pending, running, ok = list(folders), [], True
    while pending or running:
        while pending and len(running) < jobs:
            f = pending.pop(0)
            fh = open(logs / f"{f.name}.log", "w")
            running.append((f, subprocess.Popen(base + [str(f)], stdout=fh, stderr=subprocess.STDOUT, env=os.environ.copy()), fh))
        for item in list(running):
            f, proc, fh = item
            if proc.poll() is not None:
                fh.close()
                running.remove(item)
                ok = ok and proc.returncode == 0
                print(f"[recording-replay] {f.name}: {'ok' if proc.returncode == 0 else f'FAILED (exit {proc.returncode})'}"
                      f" — log {logs / (f.name + '.log')}", flush=True)
        if running:
            time.sleep(0.5)
    return ok


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ok = True
    folders = list(args.folders)
    ran_parallel = False
    if args.app_latest:
        try:
            dest = app_pull.pull_latest(pipeline.RECORDINGS / "inbox", email=args.email, uid=args.uid,
                                        recording_id=args.recording_id)
        except app_pull.AppPullError as exc:
            print(f"[recording-replay] app pull failed: {exc}")
            return 1
        print(f"[recording-replay] pulled {dest} — edit {dest.name}.notes.txt (who:) for a better identity check")
        folders.append(dest)
    if args.all:
        inbox = pipeline.RECORDINGS / "inbox"
        # the 30 s curriculum clips are a set of their own (--set), never part of --all
        folders += sorted(p for p in inbox.iterdir() if p.is_dir() and not p.name.startswith(CLIP_PREFIX)) \
            if inbox.is_dir() else []
    for name in args.set:
        folders += clip_set_folders(name)
        args.no_freeze = True
    if args.jobs > 1 and len(folders) > 1:
        ok = run_parallel(folders, args.jobs, argv if argv is not None else sys.argv[1:]) and ok
        ran_parallel = True
        folders = []
    fixtures = [fixture.FIXTURES / n for n in args.fixture]
    if args.fixtures:
        fixtures += fixture.list_fixtures()
    if not folders and not fixtures and not args.corpus_summary and not ran_parallel:
        build_parser().print_help()
        return 1
    for f in folders:
        try:
            ok = run_folder(f, args) and ok
        except Exception as exc:  # noqa: BLE001 — one bad recording never stops --all
            ok = False
            print(f"[recording-replay] {f.name}: FAILED {type(exc).__name__}: {exc}")
            traceback.print_exc()
    for fx in fixtures:
        if not (fx / "baseline.json").exists():
            print(f"[recording-replay] no fixture at {fx}")
            ok = False
            continue
        ok = run_fixture(fx) and ok
    if args.corpus_summary:
        from recreplay import corpus_summary
        out = corpus_summary.write(pipeline.RECORDINGS / "reports" / "corpus-summary.html",
                                   pipeline.RECORDINGS / "work", pipeline.RECORDINGS / "inbox")
        print(f"[recording-replay] corpus summary: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
