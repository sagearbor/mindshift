"""Get the open-web ``yt_*`` inbox items (tmp/recordings/inbox/yt_<id>/,
filed by the open-audio intake) ready for phone-only and full replays at $0:

1. normalised 16 kHz audio (work/<name>/audio16k.full.wav);
2. a local faster-whisper reference transcript with speakers (the item's
   Gemini annotation segments when it has one, else the server's local ECAPA
   diarizer) — work/<name>/whisper.full.json;
3. a WEARER: the notes' ``I'm S<n>`` (the intake's most-involved pick) or,
   unannotated, the voice that talks most -> <name>.wearer.json;
4. a HELD-OUT enrollment slice, the corpus_to_inbox approach (solo speech
   only, outside the tested window, with a guard) adapted to a clip with no
   audio outside it: the wearer's solo speech at the clip's END is enrolled
   (<name>.voiceprint.json) and NOT replayed — <name>.replay_window.json
   says replay [0, end_s), and work/<name>/whisper.json is the reference cut
   to that window.

Never calls Gemini or any paid STT.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .annotation import dg_label

SR = 16000
TARGET_ENROLL_S = 30.0       # pooled solo speech to enroll from (corpus items pool up to 60 s)
MIN_ENROLL_S = 12.0          # below this, no held-out print (replay with --enroll same/none)
GUARD_S = 15.0               # gap between the replayed window and the enrollment slice
MIN_REPLAY_FRAC = 0.6        # never give up more than 40% of the clip to enrollment
MIN_PIECE_S = 1.0


def pick_wearer(words: list[dict], *, explicit_id: str | None, labels: dict[str, int] | None) -> tuple[str, str]:
    if explicit_id and labels and explicit_id in labels:
        return dg_label(labels[explicit_id]), f"notes: {explicit_id} (intake's most-involved pick)"
    talk: dict[int, float] = {}
    for w in words:
        talk[w["speaker"]] = talk.get(w["speaker"], 0.0) + (w["end"] - w["start"])
    best = max(talk, key=talk.get) if talk else 0
    return dg_label(best), f"most talk ({talk.get(best, 0.0):.0f} s of words)"


def _solo_pieces(turns: list[dict], wearer: str) -> list[tuple[float, float]]:
    others = sorted((t["start_time"], t["end_time"]) for t in turns if t["speaker"] != wearer)
    out = []
    for t in turns:
        if t["speaker"] != wearer:
            continue
        pieces = [(t["start_time"], t["end_time"])]
        for os_, oe in others:
            nxt = []
            for ps, pe in pieces:
                if oe <= ps or os_ >= pe:
                    nxt.append((ps, pe))
                    continue
                if os_ > ps:
                    nxt.append((ps, os_))
                if oe < pe:
                    nxt.append((oe, pe))
            pieces = nxt
        out += [(a, b) for a, b in pieces if b - a >= MIN_PIECE_S]
    return sorted(out)


def choose_enroll_tail(turns: list[dict], wearer: str, duration_s: float, *, target_s: float = TARGET_ENROLL_S,
                       min_s: float = MIN_ENROLL_S, guard_s: float = GUARD_S,
                       min_replay_frac: float = MIN_REPLAY_FRAC) -> dict | None:
    """Walk the wearer's solo pieces back from the clip's end until
    ``target_s`` is pooled, never past ``(1 - min_replay_frac)`` of the
    clip. None when that yields under ``min_s``."""
    limit = duration_s * min_replay_frac + guard_s
    chosen, total = [], 0.0
    for a, b in reversed(_solo_pieces(turns, wearer)):
        if a < limit:
            break
        chosen.append((a, b))
        total += b - a
        if total >= target_s:
            break
    if total < min_s:
        return None
    start = min(a for a, _ in chosen)
    return {"enroll": sorted(chosen), "enroll_s": round(total, 1), "enroll_from_s": start,
            "end_s": round(start - guard_s, 3), "guard_s": guard_s}


def truncate_raw(raw: dict, end_s: float) -> dict:
    out = json.loads(json.dumps(raw))
    alt = out["results"]["channels"][0]["alternatives"][0]
    alt["words"] = [w for w in alt.get("words") or [] if float(w["end"]) <= end_s]
    out.setdefault("metadata", {})["replay_end_s"] = end_s
    return out


def truncate_doc(doc: dict, end_s: float) -> dict:
    out = dict(doc)
    segs = []
    for s in doc.get("segments") or []:
        ws = [w for w in s.get("words") or [] if float(w["end"]) <= end_s]
        if ws:
            segs.append({**s, "end": min(float(s["end"]), end_s), "words": ws})
    out["segments"] = segs
    out["replay_end_s"] = end_s
    return out


def enroll_print(pcm: np.ndarray, pieces: list[tuple[float, float]], *, name: str, wearer: str, plan: dict) -> dict | None:
    import speaker_id

    chunks = [pcm[int(a * SR):int(b * SR)].astype(np.float32) / 32768.0 for a, b in pieces]
    chunks = [c for c in chunks if c.size]
    if not chunks:
        return None
    pooled = np.concatenate(chunks)
    if speaker_id.speech_seconds(pooled, SR) < speaker_id.MIN_ENROLL_SECONDS:
        return None
    emb = speaker_id.embed_pcm(pooled, SR)
    return speaker_id.new_profile(
        np.asarray(emb, dtype=np.float32), None, recording_id=f"openweb:{name}", speaker=wearer,
        now_iso=datetime.now(timezone.utc).isoformat(), seconds=plan["enroll_s"],
        note=(f"held-out open-web enrollment: {plan['enroll_s']:.0f} s of {wearer}'s solo speech from the clip's "
              f"tail ({plan['enroll_from_s']:.0f} s on), not replayed (replay window 0-{plan['end_s']:.0f} s, "
              f"{plan['guard_s']:.0f} s guard)"),
    )


def prepare_item(folder: Path, *, work_root: Path) -> dict:
    """Prepare one yt_* item (idempotent; whisper decode cached)."""
    from . import audio as audio_mod
    from . import notes as notes_mod
    from . import pipeline, stt

    folder = Path(folder)
    name = folder.name
    work = Path(work_root) / name
    work.mkdir(parents=True, exist_ok=True)
    full_wav = work / "audio16k.full.wav"
    if not full_wav.exists():
        audio_mod.normalize(audio_mod.find_audio(folder, name), full_wav)
    pcm = audio_mod.read_wav16(full_wav)
    duration = len(pcm) / SR
    inp = pipeline.inputs_from_inbox(folder, work_root=Path(work_root), stt_engine="whisper")
    full_json = work / "whisper.full.json"
    if full_json.exists():
        raw = json.loads(full_json.read_text())
    else:
        raw = pipeline.build_whisper_reference(inp, full_wav, pcm, raw_cache=work / "whisper.full.raw.json",
                                               out=full_json)
    words = stt.words_from_raw(raw)
    turns = stt.turns_from_words(words)
    notes = notes_mod.parse_notes(inp.notes_text)
    diar = (raw.get("metadata") or {}).get("diarization") or {}
    wearer, how = pick_wearer(words, explicit_id=notes.self_speaker_id, labels=diar.get("labels"))
    ann_id = next((k for k, v in (diar.get("labels") or {}).items() if dg_label(v) == wearer), None)
    (folder / f"{name}.wearer.json").write_text(json.dumps(
        {"label": wearer, "ann_id": ann_id, "method": how, "speakers_from": diar.get("source"),
         "written_by": "scripts/yt_prepare.py"}, indent=1))
    plan = choose_enroll_tail(turns, wearer, duration)
    vp = enroll_print(pcm, plan["enroll"], name=name, wearer=wearer, plan=plan) if plan else None
    doc = json.loads((work / "whisper.full.raw.json").read_text())
    if plan and vp:
        (folder / f"{name}.voiceprint.json").write_text(json.dumps(vp))
        (folder / f"{name}.replay_window.json").write_text(json.dumps(
            {"end_s": plan["end_s"], "enroll": plan["enroll"], "enroll_s": plan["enroll_s"],
             "guard_s": plan["guard_s"], "clip_s": round(duration, 3), "wearer": wearer,
             "written_by": "scripts/yt_prepare.py"}, indent=1))
        (work / "whisper.json").write_text(json.dumps(truncate_raw(raw, plan["end_s"])))
        (work / "whisper.raw.json").write_text(json.dumps(truncate_doc(doc, plan["end_s"])))
    else:
        for p in (folder / f"{name}.replay_window.json",):
            if p.exists():
                p.unlink()
        (work / "whisper.json").write_text(json.dumps(raw))
        (work / "whisper.raw.json").write_text(json.dumps(doc))
    return {"name": name, "clip_s": round(duration, 1), "words": len(words),
            "speakers": len({w["speaker"] for w in words}), "speakers_from": diar.get("source"),
            "wearer": wearer, "wearer_how": how,
            "replay_s": round(plan["end_s"], 1) if plan and vp else round(duration, 1),
            "enroll_s": plan["enroll_s"] if plan and vp else 0.0,
            "voiceprint": bool(vp)}
