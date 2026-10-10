"""One recording, end to end: normalise -> Deepgram -> annotations aligned
-> owner identity -> phone loop -> server in real time -> score -> report
(-> freeze as a regression fixture). Used by scripts/recording_replay.py
and by the offline fixture test (server/tests/test_recording_replay_fixtures.py).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import annotation as ann
from . import audio as audio_mod
from . import identity as identity_mod
from . import notes as notes_mod
from . import phone as phone_mod
from . import score as score_mod
from . import server_run
from . import stt

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDINGS = Path(os.getenv("MINDSHIFT_RECORDINGS_DIR") or (REPO_ROOT / "tmp" / "recordings"))
PRIVATE = Path(os.getenv("MINDSHIFT_PRIVATE_FIXTURES") or (REPO_ROOT / "tmp" / "private_fixtures"))
DEFAULT_PROFILE = PRIVATE / "owner_profile.json"


@dataclass
class RunInputs:
    name: str
    audio: Path                                  # any format (normalised into work/)
    notes_text: str
    annotation_files: list[ann.AnnotationFile]
    work: Path                                   # outputs + caches for this run
    deepgram_cache: Path
    llm_cache_dir: Path
    profile_path: Path | None = None
    offline: bool = False
    app_meta: dict | None = None                 # <name>.app_meta.json (an app recording pulled from GCS)
    stt_language: str | None = None              # Deepgram language override (annotation audio.language)


@dataclass
class RunOptions:
    mode: str | None = None                      # None = notes' mode or earpiece
    relationship: str | None = None
    library_item_ids: list[str] | None = None
    session_context: str | None = None
    url: str | None = None
    id_token: str | None = None
    email: str | None = None
    password: str | None = None
    speed: float = 1.0
    phone_tone: str | None = None                # annotation | neutral (None = annotation if present)
    enroll: str | None = None                    # profile | same | none (None = profile if present)
    wearer: str | None = None                    # Deepgram label override
    moment_window_s: float = score_mod.MOMENT_WINDOW_S
    replay_latency: bool = True
    llm_model: str | None = None                 # pin the model (offline fixture re-runs)
    skip_phone: bool = False
    skip_server: bool = False
    log: list[str] = field(default_factory=list)


def inputs_from_inbox(folder: Path, *, work_root: Path | None = None, profile: Path | None = None) -> RunInputs:
    folder = Path(folder)
    name = folder.name
    work = (work_root or RECORDINGS / "work") / name
    notes_path = next((p for p in (folder / f"{name}.notes.txt", folder / "notes.txt") if p.exists()), None)
    prof = profile if profile is not None else (DEFAULT_PROFILE if DEFAULT_PROFILE.exists() else None)
    # A corpus item (scripts/corpus_to_inbox.py) brings its wearer's own print,
    # enrolled from a held-out part of the same session.
    # Never the owner's print on a stranger's corpus conversation.
    item_vp = folder / f"{name}.voiceprint.json"
    if item_vp.exists():
        prof = item_vp
    elif notes_path is not None and "source: corpus ground truth" in notes_path.read_text(errors="replace"):
        prof = None
    app_meta_path = folder / f"{name}.app_meta.json"
    app_meta = json.loads(app_meta_path.read_text()) if app_meta_path.exists() else None
    return RunInputs(
        name=name, audio=audio_mod.find_audio(folder, name),
        notes_text=notes_path.read_text(errors="replace") if notes_path else "",
        annotation_files=ann.discover(folder, name), work=work,
        deepgram_cache=work / "deepgram.json", llm_cache_dir=work / "llm_cache", profile_path=prof,
        app_meta=app_meta, stt_language=_annotation_language(folder, name),
    )


def _annotation_language(folder: Path, name: str) -> str | None:
    """``audio.language`` of the folder's first annotation that states one
    (a corpus item in another language, e.g. CONFER's Greek debates)."""
    for f in ann.discover(folder, name):
        try:
            lang = (json.loads(f.path.read_text(errors="replace")).get("audio") or {}).get("language")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(lang, str) and lang.strip():
            return lang.strip()
    return None


def _say(opts: RunOptions, msg: str) -> None:
    opts.log.append(msg)
    print(f"[recording-replay] {msg}", flush=True)


def run(inp: RunInputs, opts: RunOptions) -> dict:
    t_start = time.monotonic()
    server_run.ensure_ecapa_cache()
    inp.work.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    # 1. notes
    notes = notes_mod.parse_notes(inp.notes_text)
    problems += notes.problems

    # 2. audio
    wav = inp.work / "audio16k.wav"
    if inp.audio.suffix.lower() == ".wav" and _is_16k_mono(inp.audio):
        if inp.audio.resolve() != wav.resolve():
            shutil.copyfile(inp.audio, wav)
        audio_info = {"path": str(wav), "duration_s": round(len(audio_mod.read_wav16(wav)) / 16000, 3), "source": str(inp.audio)}
    else:
        audio_info = audio_mod.normalize(inp.audio, wav)
    pcm = audio_mod.read_wav16(wav)
    _say(opts, f"{inp.name}: {audio_info['duration_s']:.1f} s audio")

    # 3. reference transcript
    raw, stt_source = stt.transcribe(wav.read_bytes(), inp.deepgram_cache, offline=inp.offline,
                                     language=inp.stt_language)
    words = stt.words_from_raw(raw)
    dg_turns = stt.turns_from_words(words)
    _say(opts, f"Deepgram ({stt_source}): {len(words)} words, {len(dg_turns)} turns, "
               f"{len({t['speaker'] for t in dg_turns})} speakers")

    # 4. annotations
    annotations = []
    for f in inp.annotation_files:
        a = ann.load_file(f)
        al = ann.align(a, words) if a.ok else None
        annotations.append((a, al))
        _say(opts, f"annotation {f.label}: ok={a.ok} segments={len(a.segments)} "
                   + (f"aligned {al.quality['words_matched_pct']}% words, shift {al.quality['median_shift_s']} s" if al else "")
                   + (f"; {len(a.problems)} problems" if a.problems else ""))
    primary = next(((a, al) for a, al in annotations if a.ok), (None, None))
    a0, al0 = primary

    # 5. who is the owner
    profile = identity_mod.load_profile(inp.profile_path) if inp.profile_path else None
    vp = identity_mod.voiceprint_scores(pcm, dg_turns, profile) if profile else None
    dg_speakers = sorted({t["speaker"] for t in dg_turns})
    wearer = identity_mod.resolve(notes, a0, al0, voiceprint_scores=vp, dg_speakers=dg_speakers,
                                  talk_seconds=stt.talk_seconds(dg_turns), override=opts.wearer)
    _say(opts, f"owner = {wearer.wearer_label} / {wearer.wearer_ann_id} ({wearer.method})"
               + (f" — {'; '.join(wearer.warnings)}" if wearer.warnings else ""))

    # A print enrolled FROM this very recording makes the identity numbers optimistic.
    rec_id = (inp.app_meta or {}).get("recording_id") or (inp.app_meta or {}).get("id")
    if profile and rec_id and any(s.get("recording_id") == rec_id for s in profile.get("samples") or []):
        wearer.warnings.append(
            f"the owner's voiceprint was partly enrolled from THIS recording ({rec_id[:8]}…): "
            "voice-ID results here are optimistic, not what a new conversation gets")

    mode = opts.mode or notes.mode or "earpiece"
    phone_tone = opts.phone_tone or ("annotation" if a0 is not None else "neutral")
    enroll = opts.enroll or ("profile" if profile else "same")

    # 6. phone loop
    meta, seg_source = phone_mod.build_meta(
        inp.name, dg_turns, words, wearer_label=wearer.wearer_label, wearer_ann_id=wearer.wearer_ann_id,
        alignment=al0, annotation=a0, phone_tone=phone_tone,
    )
    meta_path = inp.work / "phone_meta.json"
    meta_path.write_text(json.dumps(meta, indent=1))
    phone_out, phone_ceiling, phone_error = None, None, None
    if not opts.skip_phone:
        try:
            phone_out = phone_mod.run_phone(wav, meta_path, inp.work / "phone.json", mode=mode, enroll=enroll,
                                            profile=inp.profile_path)
            _say(opts, f"phone ({enroll}): {phone_out.get('_stdout', '')}")
            if enroll != "same":
                phone_ceiling = phone_mod.run_phone(wav, meta_path, inp.work / "phone_same.json", mode=mode,
                                                    enroll="same")
        except phone_mod.PhoneReplayUnavailable as exc:
            phone_error = str(exc)
            problems.append(f"phone replay unavailable: {exc}")
            _say(opts, f"phone replay unavailable: {exc}")

    # 7. server in real time
    server_out, server_error = None, None
    turn_locals, releases = phone_mod.turn_locals_for_server(phone_out or {"sent": []}, "pending")
    if not opts.skip_server and turn_locals:
        cfg = server_run.ServerConfig(
            url=opts.url, mode=mode, session_context=opts.session_context or notes.setting,
            relationship=opts.relationship or notes.relationship,
            library_item_ids=opts.library_item_ids if opts.library_item_ids is not None else notes.library_item_ids,
            speed=opts.speed, llm_cache_dir=inp.llm_cache_dir, offline=inp.offline,
            replay_latency=opts.replay_latency, profile=profile,
            id_token=opts.id_token, email=opts.email, password=opts.password, llm_model=opts.llm_model,
        )
        _say(opts, f"server: streaming {audio_info['duration_s']:.0f} s in real time to "
                   f"{opts.url or 'a local uvicorn'} ({len(turn_locals)} turn_local)…")
        try:
            srv = server_run.run_server(pcm, turn_locals, releases, cfg)
            server_out = srv.as_dict()
            server_error = srv.error
            n_sug = sum(1 for e in srv.events if e["event"].get("type") == "suggestion" and not e["event"].get("partial"))
            _say(opts, f"server: {len(srv.events)} events, {n_sug} suggestions, llm {srv.llm.get('hits', '-')} hits / "
                       f"{srv.llm.get('misses', '-')} misses" + (f", ERROR {srv.error}" if srv.error else ""))
        except Exception as exc:  # noqa: BLE001 — reported in the run, never a crash past this point
            server_error = f"{type(exc).__name__}: {exc}"
            problems.append(f"server run failed: {server_error}")
            _say(opts, f"server run failed: {server_error}")
    elif not turn_locals:
        problems.append("no turn_locals from the phone loop: server not run")

    bundle = {
        "name": inp.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "offline": inp.offline,
        "audio": audio_info,
        "notes": {"who": notes.who, "self_clause": notes.self_clause, "setting": notes.setting, "phone": notes.phone,
                  "relationship": notes.relationship, "mode": notes.mode, "library_item_ids": notes.library_item_ids,
                  "moments": [m.__dict__ for m in notes.moments], "problems": notes.problems},
        "settings": {"mode": mode, "phone_tone": phone_tone, "enroll": enroll, "speed": opts.speed,
                     "url": opts.url, "moment_window_s": opts.moment_window_s,
                     "session_context": opts.session_context or notes.setting,
                     "relationship": opts.relationship or notes.relationship},
        "stt": {"source": stt_source, "words": words, "turns": [{k: v for k, v in t.items() if k != "words"} for t in dg_turns]},
        "annotations": [
            {"label": a.label, "ok": a.ok, "model": a.model, "problems": a.problems, "repairs": a.repairs,
             "quality": al.quality if al else None, "aligned": ann.aligned_to_dict(al, a) if al else None,
             "source": a.source}
            for a, al in annotations
        ],
        "primary_annotation": a0.label if a0 else None,
        "identity": wearer.as_dict(),
        "voiceprint_profile": str(inp.profile_path) if profile else None,
        "app_meta": inp.app_meta,
        "phone_meta": {"segmentation": seg_source, "path": str(meta_path), "turns": len(meta["turns"])},
        "phone": _slim_phone(phone_out),
        "phone_ceiling": _slim_phone(phone_ceiling, ceiling=True),
        "phone_error": phone_error,
        "server": server_out,
        "server_error": server_error,
        "problems": problems,
        "log": opts.log,
    }
    bundle["score"] = score_mod.score(bundle, moment_window_s=opts.moment_window_s)
    bundle["wall_s"] = round(time.monotonic() - t_start, 1)
    (inp.work / "run.json").write_text(json.dumps(bundle, indent=1, default=str))
    return bundle


def _slim_phone(p: dict | None, *, ceiling: bool = False) -> dict | None:
    if p is None:
        return None
    if ceiling:
        return {"enrollment": p.get("enrollment"), "attribution": p.get("attribution"),
                "sent": [{k: e.get(k) for k in ("speaker", "is_self", "speaker_match_score", "start_time", "end_time",
                                                "sent_at_audio_s", "speaker_match_basis")} for e in p.get("sent", [])]}
    return {k: v for k, v in p.items() if not k.startswith("_")}


def _is_16k_mono(path: Path) -> bool:
    try:
        audio_mod.read_wav16(path)
        return True
    except Exception:  # noqa: BLE001
        return False
