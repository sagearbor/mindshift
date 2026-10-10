#!/usr/bin/env python3
"""annotate_audio.py — annotate conversation audio with Gemini on Vertex AI.

Claude cannot hear audio; this produces the ``mindshift-annotation/v1``
sidecar (docs/recording-annotation-format.md) that the recording-replay
pipeline uses as ground truth, automatically.

    tmp/venv-annot/bin/python scripts/annotate_audio.py tmp/recordings/inbox/<name>
    tmp/venv-annot/bin/python scripts/annotate_audio.py some/file.wav --model gemini-3.8-flash
    tmp/venv-annot/bin/python scripts/annotate_audio.py tmp/recordings/inbox --all

Writes ``<name>.annotation.<model>.json`` next to the audio (raw replies go
to tmp/recordings/work/annotator-raw/). Long audio is cut into overlapping
windows (default 480 s, 30 s overlap) and stitched. Auth is Application
Default Credentials on project arborfam-hub (``gcloud auth
application-default login``), never an API key.

COST: every call is priced from token usage and recorded in a shared ledger
(default tmp/recordings/annotation-spend.json). The run hard-stops before
any call that could take the ledger past ``--cap`` (default $25), across
runs. Prices are conservative estimates (scripts/annotator/core.py).

Exit status: 0 all ok; 1 some file failed; 2 the spend cap stopped the run.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from annotator import core  # noqa: E402
from recreplay.annotation import loads_lenient  # noqa: E402

AUDIO_EXT = (".m4a", ".mp3", ".wav", ".ogg", ".webm", ".aac", ".flac", ".opus")
DEFAULT_LEDGER = REPO / "tmp/recordings/annotation-spend.json"
RAW_DIR = REPO / "tmp/recordings/work/annotator-raw"


def find_inputs(target: Path, all_items: bool) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise SystemExit(f"no such file or directory: {target}")
    own = [p for p in sorted(target.iterdir()) if p.suffix.lower() in AUDIO_EXT]
    named = [p for p in own if p.stem == target.name]
    if named or own:
        return named[:1] or own[:1]
    items = []
    for sub in sorted(p for p in target.iterdir() if p.is_dir()):
        a = [p for p in sorted(sub.iterdir()) if p.suffix.lower() in AUDIO_EXT]
        if a:
            items.append(next((p for p in a if p.stem == sub.name), a[0]))
    if items and not all_items:
        raise SystemExit(f"{target} holds {len(items)} items; pass --all to annotate them all")
    return items


def out_path(audio: Path, model: str) -> Path:
    return audio.with_name(f"{audio.stem}.annotation.{model}.json")


def validate(obj: dict) -> list[str]:
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema not installed; strict validation skipped"]
    schema = json.loads((REPO / "server/tests/fixtures/annotation/mindshift-annotation-v1.schema.json").read_text())
    return [f"{'/'.join(map(str, e.absolute_path)) or '(root)'}: {e.message[:140]}"
            for e in jsonschema.Draft202012Validator(schema).iter_errors(obj)]


def annotate_file(gem, audio: Path, model: str, *, window_s: float, overlap_s: float, log=print) -> dict:
    from annotator import gemini as g
    dur = g.duration_s(audio)
    windows = core.plan_windows(dur, window_s=window_s, overlap_s=overlap_s)
    log(f"  {audio.name}: {dur:.1f}s, {len(windows)} window(s), model {model}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    objs, usd, fixes, notes = [], 0.0, [], []
    with tempfile.TemporaryDirectory(prefix="annot-") as td:
        for i, (a, b) in enumerate(windows):
            if len(windows) == 1:
                clip = g.encode_window(audio, Path(td) / f"w{i}.mp3")
                prompt = core.PROMPT
            else:
                clip = g.encode_window(audio, Path(td) / f"w{i}.mp3", a, b - a)
                known = core.stitch(objs, windows[:i])["speakers"] if objs else []
                prompt = core.PROMPT + "\n" + core.window_note(a, b, dur, known)
            res = gem.annotate(model, clip, prompt, b - a, item=f"{audio.name}#w{i}")
            usd += res.usd
            stamp = time.strftime("%Y%m%d-%H%M%S")
            (RAW_DIR / f"{audio.stem}.{model}.w{i}.{stamp}.txt").write_text("\n\n=====PART=====\n\n".join(res.parts))
            obj, repairs, problems = loads_lenient(core.repair_text(core.join_parts(res.parts)))
            if len(res.parts) > 1:
                notes.append(f"window {i}: {len(res.parts) - 1} continuation(s)")
            if obj is None:
                raise RuntimeError(f"window {i}: unparseable reply ({'; '.join(problems)})")
            fixes += [f"w{i}: {r}" for r in repairs]
            # sanitise per window so stitching sees clean ids/numbers
            wf: list[str] = []
            objs.append(core.sanitize(obj, model=model, duration_s=b - a, fixes=wf))
            fixes += [f"w{i}: {x}" for x in wf]
    merged = objs[0] if len(objs) == 1 else core.stitch(objs, windows)
    final_fixes: list[str] = []
    out = core.sanitize(merged, model=model, duration_s=dur, fixes=final_fixes)
    fixes += final_fixes
    base = out["annotator"].get("notes") or ""
    tail = (f"[auto: scripts/annotate_audio.py, Vertex AI {model}, {len(windows)} window(s)"
            f"{', ' + '; '.join(notes) if notes else ''}; {len(fixes)} sanitiser fix(es); est. ${usd:.4f}]")
    out["annotator"]["notes"] = (base + " " + tail).strip()
    problems = validate(out)
    return {"annotation": out, "usd": usd, "fixes": fixes, "schema_problems": problems, "duration_s": dur,
            "windows": len(windows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path, help="audio file, inbox item folder, or inbox folder (with --all)")
    ap.add_argument("--model", action="append", help="Vertex Gemini model id (repeatable). Default gemini-3.8-flash (best in validation)")
    ap.add_argument("--all", action="store_true", help="annotate every item in an inbox folder")
    ap.add_argument("--force", action="store_true", help="re-annotate even if the output exists")
    ap.add_argument("--window-s", type=float, default=480.0)
    ap.add_argument("--overlap-s", type=float, default=30.0)
    ap.add_argument("--cap", type=float, default=25.0, help="hard spend cap in USD across runs (default 25)")
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--schema", action="store_true",
                    help="also constrain decoding with the JSON schema (off by default: it caused runaway output)")
    args = ap.parse_args(argv)
    models = args.model or ["gemini-3.8-flash"]

    from annotator.gemini import Gemini
    ledger = core.Ledger(args.ledger, cap_usd=args.cap)
    print(f"spend so far ${ledger.total:.4f} of ${ledger.cap:.2f} cap (ledger {args.ledger})")
    gem = Gemini(ledger, use_schema=args.schema)
    inputs = find_inputs(args.target, args.all)
    failed = 0
    for audio in inputs:
        for model in models:
            dst = out_path(audio, model)
            if dst.exists() and not args.force:
                print(f"  skip {dst.name} (exists; --force to redo)")
                continue
            try:
                r = annotate_file(gem, audio, model, window_s=args.window_s, overlap_s=args.overlap_s)
            except core.BudgetExceeded as exc:
                print(f"STOP: {exc}")
                return 2
            except Exception as exc:  # noqa: BLE001 — report and continue with the next file
                failed += 1
                print(f"  FAILED {audio.name} [{model}]: {type(exc).__name__}: {str(exc)[:300]}")
                continue
            dst.write_text(json.dumps(r["annotation"], indent=1, ensure_ascii=False))
            a = r["annotation"]
            print(f"  wrote {dst}  segments={len(a['segments'])} speakers={len(a['speakers'])} "
                  f"heat={a['summary']['overall_heat']} fixes={len(r['fixes'])} "
                  f"schema_problems={len(r['schema_problems'])}  file cost ${r['usd']:.4f}  "
                  f"running total ${ledger.total:.4f}")
            for p in r["schema_problems"][:5]:
                print(f"    schema: {p}")
    print(f"done: total spend ${ledger.total:.4f} of ${ledger.cap:.2f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
