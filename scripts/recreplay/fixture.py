"""Freeze a run as a permanent, re-runnable regression fixture, and compare
a re-run against its recorded baseline.

``tmp/recordings/fixtures/<name>/`` (gitignored — private family audio)::

    audio.wav              normalised 16 kHz mono PCM16
    notes.txt              the owner's notes as given
    annotations/           the annotation files as given (+ .partN)
    annotation.aligned.json  the primary annotation re-timed to Deepgram
    deepgram.json          the raw reference transcript (STT cache)
    llm_cache/             every LLM reply of the recorded run (+ its latency)
    phone_meta.json        the script the phone loop was replayed with
    phone.json             the phone loop's output (turn_local stream etc.)
    baseline.json          metrics + settings of the recorded run
    run.json               the full bundle of the recorded run

A re-run (``recording_replay.py --fixture <name>`` or the pytest) is fully
offline: STT from deepgram.json, the LLM from llm_cache (with recorded
latency; a cache miss is an honest error, never a call), the phone loop
re-executed, the server re-run locally in real time.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import annotation as ann
from .pipeline import PRIVATE, RECORDINGS, REPO_ROOT, RunInputs

FIXTURES = RECORDINGS / "fixtures"

# A latency budget no live coach line should exceed (speech end -> line).
LATENCY_P90_BUDGET_MS = 4000.0


def metrics(bundle: dict) -> dict:
    sc = bundle.get("score") or {}
    ph = (sc.get("identity") or {}).get("phone") or {}
    lat = sc.get("latency") or {}
    mom = sc.get("moments") or {}
    att = (bundle.get("phone") or {}).get("attribution") or {}
    return {
        "phone_turns": ph.get("turns"),
        "identity_phone_accuracy": ph.get("accuracy"),
        "identity_wearer_recall": ph.get("wearer_recall"),
        "identity_false_self": ph.get("false_self"),
        "phone_first_confirmed_s": ph.get("first_confirmed_s"),
        "phone_self_correct": att.get("selfCorrect"),
        "latency_p50_ms": lat.get("p50_ms"),
        "latency_p90_ms": lat.get("p90_ms"),
        "server_lines": lat.get("n"),
        "moments_hits": mom.get("hits"),
        "moments_total": mom.get("total"),
        "fires": mom.get("fires"),
        "violations": len(sc.get("violations") or []),
        "errors": len(sc.get("errors") or []),
        "llm_offline_misses": (lat.get("llm_cache") or {}).get("offline_misses"),
    }


def compare(baseline: dict, current: dict) -> tuple[list[str], list[str]]:
    """(regressions, notes). Phone-side numbers are deterministic, so they
    must not get worse at all; latency gets headroom for scheduler jitter."""
    base = baseline.get("metrics", baseline)
    reg: list[str] = []
    notes: list[str] = []

    def worse_if_lower(key: str, tol: float = 1e-9):
        b, c = base.get(key), current.get(key)
        if b is not None and (c is None or c < b - tol):
            reg.append(f"{key}: {c} < baseline {b}")

    def worse_if_higher(key: str):
        b, c = base.get(key), current.get(key)
        if b is not None and c is not None and c > b:
            reg.append(f"{key}: {c} > baseline {b}")

    worse_if_lower("identity_phone_accuracy")
    worse_if_lower("identity_wearer_recall")
    worse_if_lower("phone_self_correct")
    worse_if_higher("identity_false_self")
    worse_if_lower("moments_hits")
    worse_if_higher("violations")
    worse_if_higher("errors")
    b_first, c_first = base.get("phone_first_confirmed_s"), current.get("phone_first_confirmed_s")
    if b_first is not None and (c_first is None or c_first > b_first + 0.5):
        reg.append(f"phone_first_confirmed_s: {c_first} later than baseline {b_first}")
    b90, c90 = base.get("latency_p90_ms"), current.get("latency_p90_ms")
    if c90 is not None:
        limit = max(LATENCY_P90_BUDGET_MS, (b90 or 0) * 1.5, (b90 or 0) + 750.0)
        if c90 > limit:
            reg.append(f"latency_p90_ms: {c90:.0f} over the limit {limit:.0f} (baseline {b90})")
    if (current.get("llm_offline_misses") or 0) > 0:
        notes.append(f"{current['llm_offline_misses']} LLM prompt(s) changed since recording (offline cache miss): "
                     "those turns got no line. Re-record with --rebaseline if the prompt change is intended.")
    if current.get("server_lines") != base.get("server_lines"):
        notes.append(f"server_lines {current.get('server_lines')} vs baseline {base.get('server_lines')}")
    return reg, notes


def freeze(bundle: dict, inp: RunInputs, *, dest: Path | None = None, rebaseline: bool = False) -> Path:
    dest = Path(dest or FIXTURES / inp.name)
    dest.mkdir(parents=True, exist_ok=True)
    work = inp.work
    shutil.copyfile(work / "audio16k.wav", dest / "audio.wav")
    (dest / "notes.txt").write_text(inp.notes_text)
    adir = dest / "annotations"
    adir.mkdir(exist_ok=True)
    for f in inp.annotation_files:
        if f.path.parent.resolve() != adir.resolve():
            shutil.copyfile(f.path, adir / f.path.name.replace(f.path.name.split(".annotation")[0], inp.name, 1))
            for c in f.continuations:
                shutil.copyfile(c, adir / c.name.replace(c.name.split(".annotation")[0], inp.name, 1))
    primary = next((a for a in bundle.get("annotations") or [] if a["label"] == bundle.get("primary_annotation")), None)
    if primary and primary.get("aligned"):
        (dest / "annotation.aligned.json").write_text(json.dumps(primary["aligned"], indent=1))
    if inp.deepgram_cache.exists() and inp.deepgram_cache.resolve() != (dest / "deepgram.json").resolve():
        shutil.copyfile(inp.deepgram_cache, dest / "deepgram.json")
    if inp.llm_cache_dir.exists() and inp.llm_cache_dir.resolve() != (dest / "llm_cache").resolve():
        shutil.copytree(inp.llm_cache_dir, dest / "llm_cache", dirs_exist_ok=True)
    if inp.app_meta:
        (dest / "app_meta.json").write_text(json.dumps(inp.app_meta, indent=1))
    for name in ("phone_meta.json", "phone.json"):
        if (work / name).exists() and (work / name).resolve() != (dest / name).resolve():
            shutil.copyfile(work / name, dest / name)
    base_path = dest / "baseline.json"
    if rebaseline or not base_path.exists():
        settings = dict(bundle.get("settings") or {})
        settings["llm_model"] = ((bundle.get("server") or {}).get("llm") or {}).get("model")
        prof = bundle.get("voiceprint_profile")
        if prof:
            try:  # portable: relative to the checkout when it lives inside it
                prof = str(Path(prof).resolve().relative_to(REPO_ROOT.resolve()))
            except ValueError:
                pass
        settings["profile"] = prof
        settings["wearer"] = (bundle.get("identity") or {}).get("wearer_label")
        base_path.write_text(json.dumps({"name": inp.name, "recorded_at": bundle.get("generated_at"),
                                         "settings": settings, "metrics": metrics(bundle)}, indent=1))
    (dest / "run.json").write_text(json.dumps(bundle, indent=1, default=str))
    return dest


def list_fixtures(root: Path | None = None) -> list[Path]:
    root = Path(root or FIXTURES)
    return sorted(p for p in root.iterdir() if (p / "baseline.json").exists() and (p / "audio.wav").exists()) if root.is_dir() else []


def inputs_from_fixture(fx: Path, work: Path) -> tuple[RunInputs, dict]:
    fx = Path(fx)
    baseline = json.loads((fx / "baseline.json").read_text())
    name = baseline.get("name") or fx.name
    prof = (baseline.get("settings") or {}).get("profile")
    prof_path = (Path(prof) if Path(prof).is_absolute() else REPO_ROOT / prof) if prof else None
    if prof_path is not None and not prof_path.exists():
        # the recording machine's absolute path; fall back to this checkout's private fixtures
        alt = PRIVATE / prof_path.name
        prof_path = alt if alt.exists() else None
    inp = RunInputs(
        name=name, audio=fx / "audio.wav",
        notes_text=(fx / "notes.txt").read_text() if (fx / "notes.txt").exists() else "",
        annotation_files=ann.discover(fx / "annotations", name) if (fx / "annotations").is_dir() else [],
        work=work, deepgram_cache=fx / "deepgram.json", llm_cache_dir=fx / "llm_cache",
        profile_path=prof_path, offline=True,
        app_meta=json.loads((fx / "app_meta.json").read_text()) if (fx / "app_meta.json").exists() else None,
    )
    return inp, baseline


def options_from_baseline(baseline: dict):
    """RunOptions that reproduce the recorded run's settings."""
    from .pipeline import RunOptions

    st = baseline.get("settings") or {}
    return RunOptions(
        mode=st.get("mode"), relationship=st.get("relationship"), session_context=st.get("session_context"),
        speed=float(st.get("speed") or 1.0), phone_tone=st.get("phone_tone"), enroll=st.get("enroll"),
        moment_window_s=float(st.get("moment_window_s") or 6.0),
        llm_model=st.get("llm_model"), replay_latency=True,
    )
