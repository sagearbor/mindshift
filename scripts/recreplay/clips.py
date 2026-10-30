#!/usr/bin/env python3
"""clips.py — the 30 s CURRICULUM: short replay items cut from the DEV corpus
sessions, so a tuning loop can replay a focused set in ~1 min instead of
5-min windows.

    tmp/venv/bin/python scripts/recreplay/clips.py                # write every tier (DEV only)
    tmp/venv/bin/python scripts/recreplay/clips.py --dry-run      # list what would be cut
    tmp/venv/bin/python scripts/recording_replay.py --set clip30_t1 --jobs 4

Each MOMENT clip = one ground-truth coach moment +-15 s (shifted right when
needed so the turn-end anchor + 6 s scoring window still fits), and each one
gets a MATCHED CALM clip: 30 s of the same session with no moment within
6 s of it, picked to match the moment clip's speech density and loudness.

Tiers (exclusive, first match wins):
  t3  quiet: the clip's speech-active loudness is >= 2 dB below its own
      session's median 30 s window (relative, because the corpora differ by
      ~10 dB in recording level; they hold no true far-field audio, so quiet
      stretches are the stand-in)
  t2  overlap-dense: >= 25 % of the clip's speech time has two or more voices
  t1  clean single moment: exactly one moment in the clip
  (clips with several moments that are neither quiet nor overlap-dense are not cut)

Nothing new is transcribed: the parent's cached Deepgram JSON is sliced by
time into tmp/recordings/work/<clip>/deepgram.json (the pipeline's cache
path), and the parent's ground-truth annotation is sliced the same way
(segments clipped to the window, events and coach_moments kept when inside;
speaker ids, the who: line and the held-out voiceprint carry over unchanged).

Writes tmp/recordings/inbox/clip30_<tier>_<parent>_<start>[c]/ (c = the calm
match) and tmp/recordings/inbox/clip30_manifest.json. DEV sessions only
(tmp/recordings/landscape/splits.json); held-out sessions are never cut.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "server"))       # recreplay.stt imports server modules

from recreplay import score as score_mod  # noqa: E402

RECORDINGS = HERE.parents[2] / "tmp" / "recordings"
SR = 16000
CLIP_S = 30.0
HALF_S = 15.0
OVERLAP_DENSE = 0.25
QUIET_DB = 2.0
PREFIX = "clip30_"


# ---------------------------------------------------------------------------
# Parent session
# ---------------------------------------------------------------------------

def _wearer_id(notes: str) -> str | None:
    m = re.search(r"^who:\s*I'?m\s+(S\d+)", notes, re.M | re.I)
    return m.group(1) if m else None


def load_parent(name: str, inbox: Path, work: Path) -> dict:
    folder = inbox / name
    ann_path = next(folder.glob(f"{name}.annotation.groundtruth.json"))
    notes = (folder / f"{name}.notes.txt").read_text()
    with wave.open(str(folder / f"{name}.wav")) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return {"name": name, "folder": folder, "ann": json.loads(ann_path.read_text()), "ann_path": ann_path,
            "notes": notes, "prov": json.loads((folder / f"{name}.corpus.json").read_text()),
            "pcm": pcm, "dur": len(pcm) / SR, "deepgram": json.loads((work / name / "deepgram.json").read_text()),
            "wid": _wearer_id(notes)}


def _dg_words(raw: dict) -> list[dict]:
    try:
        return raw["results"]["channels"][0]["alternatives"][0].get("words") or []
    except (KeyError, IndexError, TypeError):
        return []


def truth_segs(p: dict) -> tuple[list[dict], str]:
    """Scorer-shaped segments: the annotation's, else Deepgram turns (CONFER)."""
    segs = [{"start": float(s["start"]), "end": float(s["end"]), "label": s["speaker"],
             "wearer": s["speaker"] == p["wid"], "is_backchannel": bool(s.get("is_backchannel"))}
            for s in p["ann"].get("segments") or []]
    if segs:
        return segs, "segment"
    from recreplay import stt
    words = [{"word": w.get("punctuated_word") or w.get("word"), "start": float(w["start"]), "end": float(w["end"]),
              "speaker": int(w.get("speaker", 0) or 0)} for w in _dg_words(p["deepgram"])]
    return ([{"start": t["start_time"], "end": t["end_time"], "label": t["speaker"], "wearer": False,
              "is_backchannel": False} for t in stt.turns_from_words(words)], "deepgram-turn")


# ---------------------------------------------------------------------------
# Window features
# ---------------------------------------------------------------------------

def features(p: dict, segs: list[dict], a: float, b: float) -> dict:
    grid = 0.05
    n = int((b - a) / grid)
    cnt = np.zeros(n, dtype=np.int16)
    for s in segs:
        if s.get("is_backchannel"):
            continue
        i, j = int((max(s["start"], a) - a) / grid), int((min(s["end"], b) - a) / grid)
        if j > i:
            cnt[max(i, 0):min(j, n)] += 1
    speech = cnt > 0
    x = p["pcm"][int(a * SR):int(b * SR)].astype(np.float32) / 32768.0
    # loudness over speech-active 50 ms frames only (silence would make every sparse clip "quiet")
    fr = int(grid * SR)
    frames = x[: (len(x) // fr) * fr].reshape(-1, fr)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    m = min(len(rms), n)
    act = rms[:m][speech[:m]] if speech[:m].any() else rms[:m]
    return {"speech_frac": round(float(speech.mean()), 3),
            "overlap_frac": round(float((cnt >= 2).sum() / max(speech.sum(), 1)), 3),
            "speech_dbfs": round(float(20 * np.log10(np.median(act) + 1e-9)), 1) if len(act) else None}


def moment_anchor(p: dict, segs: list[dict], basis: str, cm: dict) -> float:
    it = {"t": float(cm["t"]), "text": cm.get("ideal_nudge") or "", "rule": cm.get("rule"), "source": "annotation"}
    return score_mod.anchor_moment(it, segs, basis)[0]


def clip_start(t: float, anchor: float, dur: float) -> float:
    s = t - HALF_S
    if anchor + score_mod.MOMENT_WINDOW_S + 1.0 > s + CLIP_S:
        s = min(anchor + score_mod.MOMENT_WINDOW_S + 1.0 - CLIP_S, t - 1.0)
    return float(round(min(max(s, 0.0), max(dur - CLIP_S, 0.0)), 1))


def plan(names: list[str], inbox: Path, work: Path) -> list[dict]:
    cands = []
    parents = {}
    for name in names:
        p = load_parent(name, inbox, work)
        parents[name] = p
        if p["dur"] < CLIP_S + 1:
            continue
        segs, basis = truth_segs(p)
        lv = [features(p, segs, x, x + CLIP_S)["speech_dbfs"] for x in np.arange(0.0, p["dur"] - CLIP_S + 1e-6, 5.0)]
        lv = [v for v in lv if v is not None]
        p["median_dbfs"] = float(np.median(lv)) if lv else None
        p["used_calm"] = []
        cms = p["ann"].get("coach_moments") or []
        anchors = [moment_anchor(p, segs, basis, cm) for cm in cms]
        for cm, anc in zip(cms, anchors):
            s = clip_start(float(cm["t"]), anc, p["dur"])
            inside = [c for c in cms if s <= float(c["t"]) < s + CLIP_S]
            f = features(p, segs, s, s + CLIP_S)
            cands.append({"parent": name, "start": s, "t": float(cm["t"]), "anchor": round(anc, 3),
                          "rule": cm.get("rule") or score_mod.RULE_BY_NUDGE.get(cm.get("ideal_nudge") or ""),
                          "n_moments": len(inside), "features": f, "segs": segs, "anchors": anchors, "cms": cms})
    out, taken = [], {}
    for c in cands:
        f = c["features"]
        med = parents[c["parent"]]["median_dbfs"]
        c["quiet_at_dbfs"] = None if med is None else round(med - QUIET_DB, 1)
        if f["speech_dbfs"] is not None and med is not None and f["speech_dbfs"] <= med - QUIET_DB:
            tier = "t3"
        elif f["overlap_frac"] >= OVERLAP_DENSE:
            tier = "t2"
        elif c["n_moments"] == 1:
            tier = "t1"
        else:
            continue
        prev = taken.setdefault((tier, c["parent"]), [])
        if any(abs(c["start"] - s0) < CLIP_S / 2 for s0 in prev):
            continue                                  # half the clip would repeat one already cut
        prev.append(c["start"])
        c["tier"] = tier
        c["calm"] = calm_match(parents[c["parent"]], c)
        out.append(c)
    return out


def calm_match(p: dict, c: dict) -> dict | None:
    """The 30 s window of the same session with no moment (t or anchor)
    within 6 s of it that best matches the moment clip's speech density and
    loudness; None when the session has no such stretch. A calm window is
    used once per session (no two calm clips overlap by more than half), so
    calm hours are never double-counted."""
    marks = [float(m["t"]) for m in c["cms"]] + list(c["anchors"])
    best, best_cost = None, None
    s = 0.0
    while s + CLIP_S <= p["dur"] + 1e-6:
        if (not any(s - score_mod.MOMENT_WINDOW_S <= t <= s + CLIP_S + score_mod.MOMENT_WINDOW_S for t in marks)
                and not any(abs(s - u) < CLIP_S / 2 for u in p["used_calm"])):
            f = features(p, c["segs"], s, s + CLIP_S)
            if f["speech_frac"] >= 0.3:
                cost = abs(f["speech_frac"] - c["features"]["speech_frac"]) + (
                    abs((f["speech_dbfs"] or 0) - (c["features"]["speech_dbfs"] or 0)) / 20.0)
                if best_cost is None or cost < best_cost:
                    best, best_cost = {"start": round(s, 1), "features": f, "match_cost": round(cost, 3)}, cost
        s += 2.0
    if best:
        p["used_calm"].append(best["start"])
    return best


# ---------------------------------------------------------------------------
# Writing one clip item
# ---------------------------------------------------------------------------

def slice_annotation(ann: dict, a: float, b: float, *, keep_moments: bool, clip_name: str, parent: str) -> dict:
    out = json.loads(json.dumps(ann))
    segs = []
    for s in ann.get("segments") or []:
        st, en = max(float(s["start"]), a), min(float(s["end"]), b)
        if en - st >= 0.15:
            segs.append({**s, "start": round(st - a, 3), "end": round(en - a, 3)})
    out["segments"] = segs
    out["events"] = [{**e, "t": round(float(e["t"]) - a, 3)} for e in ann.get("events") or [] if a <= float(e["t"]) < b]
    out["coach_moments"] = ([{**m, "t": round(float(m["t"]) - a, 3)} for m in ann.get("coach_moments") or []
                             if a <= float(m["t"]) < b] if keep_moments else [])
    out["audio"] = {**(ann.get("audio") or {}), "duration_s": round(b - a, 3)}
    src = dict(ann.get("source") or {})
    w0 = (src.get("window_s") or [0.0, 0.0])[0]
    src.update({"window_s": [round(w0 + a, 3), round(w0 + b, 3)], "clip_of": parent, "clip_window_in_parent_s": [a, b],
                "clip": clip_name})
    out["source"] = src
    out["annotator"] = {**(ann.get("annotator") or {}),
                        "notes": (ann.get("annotator") or {}).get("notes", "") + f" CLIP: {clip_name} = {parent} {a:.1f}-{b:.1f} s, "
                                 "sliced by time by scripts/recreplay/clips.py."}
    for key in ("summary",):
        if out.get(key):
            out[key] = {**out[key], "peak_heat_t": None}
    return out


def slice_deepgram(raw: dict, a: float, b: float, *, parent: str) -> dict:
    words = []
    for w in _dg_words(raw):
        if float(w["start"]) >= a and float(w["end"]) <= b:
            words.append({**w, "start": round(float(w["start"]) - a, 3), "end": round(float(w["end"]) - a, 3)})
    text = " ".join(w.get("punctuated_word") or w.get("word") or "" for w in words)
    return {"metadata": {"sliced_from": parent, "window_s": [a, b], "by": "scripts/recreplay/clips.py",
                         "note": "a time slice of the parent's cached Deepgram response; no new STT call"},
            "results": {"channels": [{"alternatives": [{"transcript": text, "words": words}]}]}}


def write_clip(p: dict, name: str, a: float, *, keep_moments: bool, group: str, meta: dict, inbox: Path, work: Path) -> Path:
    b = a + CLIP_S
    folder = inbox / name
    folder.mkdir(parents=True, exist_ok=True)
    pcm = p["pcm"][int(a * SR):int(b * SR)]
    with wave.open(str(folder / f"{name}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
    (folder / f"{name}.annotation.groundtruth.json").write_text(
        json.dumps(slice_annotation(p["ann"], a, b, keep_moments=keep_moments, clip_name=name, parent=p["name"]), indent=1))
    notes = re.sub(r"^# group:.*$", f"# group: {group}; 30 s clip {a:.1f}-{b:.1f} s of {p['name']} (clips.py)",
                   p["notes"], count=1, flags=re.M)
    (folder / f"{name}.notes.txt").write_text(notes)
    vp = p["folder"] / f"{p['name']}.voiceprint.json"
    if vp.exists():
        shutil.copyfile(vp, folder / f"{name}.voiceprint.json")     # enrolled outside the parent window
    prov = dict(p["prov"])
    w0 = (prov.get("window_s") or [0.0, 0.0])[0]
    prov.update({"name": name, "window_s": [round(w0 + a, 3), round(w0 + b, 3)], "duration_s": CLIP_S, "group": group,
                 "clip": {**meta, "parent": p["name"], "window_in_parent_s": [a, b]}})
    (folder / f"{name}.corpus.json").write_text(json.dumps(prov, indent=1))
    wdir = work / name
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "deepgram.json").write_text(json.dumps(slice_deepgram(p["deepgram"], a, b, parent=p["name"])))
    return folder


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inbox", type=Path, default=RECORDINGS / "inbox")
    ap.add_argument("--work", type=Path, default=RECORDINGS / "work")
    ap.add_argument("--splits", type=Path, default=RECORDINGS / "landscape" / "splits.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    dev = json.loads(a.splits.read_text())["dev"]
    held = set(json.loads(a.splits.read_text())["held_out"])
    assert not held & set(dev)
    clips = plan(dev, a.inbox, a.work)
    parents = {n: load_parent(n, a.inbox, a.work) for n in {c["parent"] for c in clips}}
    manifest = {"by": "scripts/recreplay/clips.py", "clip_s": CLIP_S, "split": "dev", "sets": {}, "clips": []}
    for c in clips:
        base = f"{PREFIX}{c['tier']}_{c['parent']}_{int(round(c['start'])):04d}"
        meta = {"tier": c["tier"], "kind": "moment", "moment_t_in_parent": c["t"], "anchor_in_parent": c["anchor"],
                "rule": c["rule"], "n_moments": c["n_moments"], "features": c["features"], "quiet_at_dbfs": c["quiet_at_dbfs"]}
        row = {"name": base, **meta, "parent": c["parent"], "start_in_parent": c["start"]}
        manifest["clips"].append(row)
        manifest["sets"].setdefault(f"{PREFIX}{c['tier']}", []).append(base)
        if c["calm"]:
            cm = {"tier": c["tier"], "kind": "calm-match", "paired_with": base, "features": c["calm"]["features"],
                  "match_cost": c["calm"]["match_cost"]}
            manifest["clips"].append({"name": base + "c", **cm, "parent": c["parent"], "start_in_parent": c["calm"]["start"]})
            manifest["sets"][f"{PREFIX}{c['tier']}"].append(base + "c")
        print(f"{c['tier']} {base:48} moment {c['t']:7.1f} anchor {c['anchor']:7.1f} {c['rule'] or '?':18} "
              f"{json.dumps(c['features'])}  calm@{c['calm']['start'] if c['calm'] else None}")
        if not a.dry_run:
            write_clip(parents[c["parent"]], base, c["start"], keep_moments=True, group="heated", meta=meta,
                       inbox=a.inbox, work=a.work)
            if c["calm"]:
                write_clip(parents[c["parent"]], base + "c", c["calm"]["start"], keep_moments=False, group="calm",
                           meta=cm, inbox=a.inbox, work=a.work)
    for k, v in sorted(manifest["sets"].items()):
        print(f"{k}: {len(v)} items ({sum(1 for n in v if not n.endswith('c'))} moment + "
              f"{sum(1 for n in v if n.endswith('c'))} calm)")
    if not a.dry_run:
        (a.inbox / "clip30_manifest.json").write_text(json.dumps(manifest, indent=1))
        print(a.inbox / "clip30_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
