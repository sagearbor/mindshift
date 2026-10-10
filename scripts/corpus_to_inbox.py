#!/usr/bin/env python3
"""corpus_to_inbox.py — turn REAL conversation corpora (already on disk in
gitignored tmp/, with human ground truth) into recording-replay inbox items,
so scripts/recording_replay.py can be exercised without the owner recording
anything. Logic lives in scripts/recreplay/corpora.py (see its docstring for
the per-corpus ground truth, licences and derivation rules).

    tmp/venv/bin/python scripts/corpus_to_inbox.py --plan          # the default heated-leaning batch
    tmp/venv/bin/python scripts/corpus_to_inbox.py --plan --dry-run
    tmp/venv/bin/python scripts/corpus_to_inbox.py --item sbcsae:SBC033:heated --length 300
    tmp/venv/bin/python scripts/corpus_to_inbox.py --item confer:20111031_seq5:heated

Each item lands in tmp/recordings/inbox/<corpus>_<id>/ (audio window, notes,
annotation.groundtruth.json, held-out voiceprint, corpus.json provenance) and
a batch manifest is written to tmp/recordings/inbox/corpus_batch.json.

Sources (all in tmp/, never committed):
  SBCSAE   tmp/corpora/sbcsae/audio16k/SBCnnn.wav + tmp/corpora/sbcsae/cha/CHAT/SBCnnn.cha  (CC BY-ND 3.0)
  AMI      tmp/ami-corpus/<m>.mix16k.wav + tmp/corpora/ami_annotations/{words,dialogueActs,corpusResources} (CC BY 4.0)
  CHiME-6  tmp/corpora/chime6/segments16k/<S>_segNN.wav + transcriptions/dev/<S>.json        (CHiME-6 licence)
  CONFER   tmp/corpora/confer/clips16k/<clip>.wav + heatmap_manifest.json (rated conflict)   (research only)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "server", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from recreplay import corpora as C  # noqa: E402

TMP = REPO / "tmp"
INBOX = TMP / "recordings" / "inbox"
SBC = TMP / "corpora" / "sbcsae"
AMI_AUDIO = TMP / "ami-corpus"
AMI_ANN = TMP / "corpora" / "ami_annotations"
CHIME = TMP / "corpora" / "chime6"
CONFER = TMP / "corpora" / "confer"

AMI_ROLES = {"PM": "project manager", "ID": "industrial designer", "UI": "user-interface designer", "ME": "marketing expert"}

# The owner's priority (2026-10-09): lean HEATED, keep a few labelled calm controls.
SBC_HEATED = ["SBC042", "SBC033", "SBC035", "SBC010"]   # family arguments / a tense business talk (corpus descriptions + voice marks)
SBC_CALM = ["SBC007", "SBC034"]                         # late-night sisters / a couple winding down
# What "heated" rests on, per corpus (reported next to every heated number).
HEAT_BASIS = {
    "CONFER": "rated conflict (10 human raters, frame level)",
    "SBCSAE": "argument per the corpus description + transcriber voice marks + overlap",
    "AMI": "overlap/interruption density + loudness rise only (no conflict rating): lively, not necessarily angry",
    "CHiME-6": "overlap/interruption density + loudness rise only (no conflict rating): lively, not necessarily angry",
}
AMI_MEETINGS = ["ES2002a", "ES2003a", "ES2004a", "ES2005a", "IS1000a", "IS1001a", "IS1003a", "IS1006a",
                "TS3003a", "TS3004a", "TS3005a", "TS3006a"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_sbcsae(sid: str) -> C.Session:
    s = C.parse_cha_text((SBC / "cha" / "CHAT" / f"{sid}.cha").read_text(encoding="utf-8"), sid)
    src = C.WavSource([(SBC / "audio16k" / f"{sid}.wav", 0.0)])
    s.read_audio, s.duration_s = src.read, src.duration_s
    if len(s.setting) > 320:
        s.setting = s.setting[:317].rsplit(" ", 1)[0] + "..."
    return s


def _ami_speakers(meeting: str) -> dict[str, tuple[str, str]]:
    """nxt agent letter -> (global name, description)."""
    xml = (AMI_ANN / "corpusResources" / "meetings.xml").read_text(encoding="iso-8859-1")
    m = re.search(rf'<meeting [^>]*observation="{meeting}".*?</meeting>', xml, re.S)
    out = {}
    for sp in re.finditer(r'<speaker [^>]*/>', m.group(0) if m else ""):
        tag = sp.group(0)
        agent = re.search(r'nxt_agent="(\w)"', tag).group(1)
        gname = re.search(r'global_name="(\w+)"', tag).group(1)
        role = (re.search(r'role="(\w+)"', tag) or [None, ""])[1]
        sex = "female" if gname.startswith("F") else "male" if gname.startswith("M") else "adult"
        out[agent] = (gname, f"corpus speaker {gname}: adult {sex}, the team's {AMI_ROLES.get(role, role)}")
    return out


def load_ami(meeting: str) -> C.Session:
    spk = _ami_speakers(meeting)
    turns, events = [], []
    for agent, (gname, _) in sorted(spk.items()):
        wf = AMI_ANN / "words" / f"{meeting}.{agent}.words.xml"
        df = AMI_ANN / "dialogueActs" / f"{meeting}.{agent}.dialog-act.xml"
        if not wf.exists():
            continue
        t, e = C.parse_ami_speaker(wf.read_bytes().decode("iso-8859-1").replace('encoding="ISO-8859-1"', ""),
                                   df.read_bytes().decode("iso-8859-1").replace('encoding="ISO-8859-1"', "") if df.exists() else None,
                                   speaker=gname)
        turns += t
        events += e
    turns.sort(key=lambda t: t.start)
    src = C.WavSource([(AMI_AUDIO / f"{meeting}.mix16k.wav", 0.0)])
    s = C.Session(
        corpus="AMI", sid=meeting, turns=turns, events=sorted(events, key=lambda e: e["t"]), duration_s=src.duration_s,
        setting=("AMI scenario meeting: a four-person product-design team's kickoff meeting in an office meeting "
                 "room (project manager, industrial designer, user-interface designer, marketing expert); coworkers"),
        speakers={g: d for g, d in spk.values()}, licence="CC BY 4.0 (AMI Meeting Corpus)",
        phone="corpus: the four headset mics summed into one channel (stands in for a phone on the table; no earpiece)",
        relationship="coworker", read_audio=src.read,
    )
    s.loudness = src.loudness_per_second()
    return s


def load_chime(session: str) -> C.Session:
    utts = json.loads((CHIME / "transcriptions" / "transcriptions" / "dev" / f"{session}.json").read_text())
    turns, events, locs = C.parse_chime(utts)
    parts = sorted((CHIME / "segments16k").glob(f"{session}_seg*.wav"))
    src = C.WavSource([(p, int(re.search(r"seg(\d+)", p.stem).group(1)) * 900.0) for p in parts])
    s = C.Session(
        corpus="CHiME-6", sid=session, turns=[t for t in turns if t.start < src.duration_s],
        events=[e for e in events if e["t"] < src.duration_s], duration_s=src.duration_s,
        setting="CHiME-6 dinner party at a real home: four friends cooking, eating and chatting (kitchen, dining room, living room)",
        speakers={sp: f"corpus speaker {sp}" for sp in sorted({t.speaker for t in turns})},
        licence="CHiME-6 data licence (research)",
        phone="corpus: the four participants' headset mics summed into one channel (stands in for a phone in the room; no earpiece)",
        relationship="friend", read_audio=src.read,
    )
    s.loudness = src.loudness_per_second()
    s._locations = (utts, locs)  # type: ignore[attr-defined]
    return s


def _chime_location(s: C.Session, window: tuple[float, float]) -> str | None:
    utts, _ = getattr(s, "_locations", ([], {}))
    from collections import Counter
    c = Counter(u.get("location") for u in utts if window[0] <= C._hms(u["start_time"]) < window[1] and u.get("location"))
    return c.most_common(1)[0][0] if c else None


def confer_manifest() -> dict[str, dict]:
    return {e["id"].removeprefix("confer_"): e for e in json.loads((CONFER / "heatmap_manifest.json").read_text())}


def load_confer(clip: str) -> C.Session:
    e = confer_manifest()[clip]
    src = C.WavSource([(CONFER / e["audio"], 0.0)])
    n_people = "three" if (CONFER / "ann" / "Annotations" / "three" / clip).exists() else "two"
    return C.Session(
        corpus="CONFER", sid=clip, turns=[], events=[], duration_s=src.duration_s,
        setting=f"Greek televised political debate, {n_people} people arguing on air (CONFER clip {clip}; not English)",
        speakers={}, licence="CONFER: research use with citation, no redistribution",
        phone="corpus: broadcast studio mix of the debaters' microphones (no phone, no earpiece)",
        relationship="other", conflict=list(e["conflict_per_second"]), read_audio=src.read,
    )


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

def make_item(s: C.Session, window: tuple[float, float], group: str, *, name: str | None = None,
              dry_run: bool = False) -> dict:
    heated = group == "heated"
    if s.turns:
        wearer, why = C.choose_wearer(s.turns, window[0], window[1], heated=heated, exclude=s.exclude)
    else:
        wearer, why = None, "the corpus has no speaker labels (heat ratings only); the pipeline falls back to the voice that talks most"
    if s.corpus == "CHiME-6":
        loc = _chime_location(s, window)
        if loc:
            base = getattr(s, "_base_setting", None) or s.setting
            s._base_setting = base  # type: ignore[attr-defined]
            s.setting = base.split(" (")[0] + f"; this stretch mostly in the {loc.replace('_', ' ')}"
    name = name or f"{re.sub(r'[^a-z0-9]', '', s.corpus.lower())}_{s.sid}"
    if dry_run:
        f = C.window_features(s.turns, *window, exclude=s.exclude, loudness=s.loudness, laughs=s.events) if s.turns else {}
        return {"name": name, "group": group, "window_s": list(window), "wearer": wearer, "why": why,
                "heat_score": round(C.heat_score(f, window[1] - window[0]), 3) if f else None}
    prov = C.write_item(INBOX, name, s, window, group=group, wearer=wearer, why=why)
    prov["heat_basis"] = HEAT_BASIS[s.corpus] if heated else "calm control (calmest qualifying window)"
    (INBOX / name / f"{name}.corpus.json").write_text(json.dumps(prov, indent=1, default=float))
    print(f"[corpus-to-inbox] {name}: {group}, {window[0]:.0f}-{window[1]:.0f} s, wearer {wearer} "
          f"({prov['wearer_id']}), {prov['coach_moments']} moments, voiceprint: {prov['voiceprint']}", flush=True)
    return prov


def plan(length: float, dry_run: bool) -> list[dict]:
    out: list[dict] = []
    # CONFER: every human-rated clip on disk (all carry high-conflict spans), whole clip.
    # The longer clips the raters scored LOW (never >= 400) are same-domain calm controls.
    for clip, e in sorted(confer_manifest().items()):
        if e.get("heated_spans"):
            group = "heated"
        elif e["duration_s"] >= 100 and e["conflict_max"] < 400:
            group = "calm"
        else:
            continue
        s = load_confer(clip)
        out.append(make_item(s, (0.0, round(s.duration_s, 3)), group, dry_run=dry_run))
    # SBCSAE: genuine arguments + two calm controls.
    for sid, group in [(x, "heated") for x in SBC_HEATED] + [(x, "calm") for x in SBC_CALM]:
        s = load_sbcsae(sid)
        win = C.pick_window(s.turns, s.duration_s, length=length, heated=group == "heated", exclude=s.exclude)
        out.append(make_item(s, win, group, dry_run=dry_run))
    # AMI: the three hottest meetings (overlap + interruptions + loudness rise), one calm control.
    ami = []
    for m in AMI_MEETINGS:
        s = load_ami(m)
        win = C.pick_window(s.turns, s.duration_s, length=length, heated=True, exclude=s.exclude, loudness=s.loudness)
        f = C.window_features(s.turns, *win, exclude=s.exclude, loudness=s.loudness)
        ami.append((C.heat_score(f, length), m, s, win))
    ami.sort(key=lambda x: x[0], reverse=True)
    for _, m, s, win in ami[:3]:
        out.append(make_item(s, win, "heated", dry_run=dry_run))
    calm_s = ami[-1][2]
    win = C.pick_window(calm_s.turns, calm_s.duration_s, length=length, heated=False, exclude=calm_s.exclude,
                        loudness=calm_s.loudness)
    out.append(make_item(calm_s, win, "calm", dry_run=dry_run))
    # CHiME-6: the three hottest stretches across both dinner parties, one calm control.
    cands = []
    for sess in ("S02", "S09"):
        s = load_chime(sess)
        taken: list[tuple[float, float]] = []
        for _ in range(3):
            win = C.pick_window(s.turns, s.duration_s, length=length, heated=True, exclude=s.exclude,
                                loudness=s.loudness, avoid=taken)
            taken.append(win)
            f = C.window_features(s.turns, *win, exclude=s.exclude, loudness=s.loudness)
            cands.append((C.heat_score(f, length), sess, s, win))
    cands.sort(key=lambda x: x[0], reverse=True)
    used: dict[str, list] = {}
    for _, sess, s, win in cands[:3]:
        used.setdefault(sess, []).append(win)
        out.append(make_item(s, win, "heated", name=f"chime6_{sess}_t{int(win[0])}", dry_run=dry_run))
    calm_sess = min(("S02", "S09"), key=lambda x: len(used.get(x, [])))
    s = next(c[2] for c in cands if c[1] == calm_sess)
    win = C.pick_window(s.turns, s.duration_s, length=length, heated=False, exclude=s.exclude, loudness=s.loudness,
                        avoid=[w for c in cands if c[1] == calm_sess for w in [c[3]]])
    out.append(make_item(s, win, "calm", name=f"chime6_{calm_sess}_t{int(win[0])}", dry_run=dry_run))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="the default heated-leaning batch")
    ap.add_argument("--item", action="append", default=[], help="corpus:id:group (group heated|calm), e.g. sbcsae:SBC033:heated")
    ap.add_argument("--length", type=float, default=300.0, help="window seconds (3-10 min; CONFER clips are used whole)")
    ap.add_argument("--dry-run", action="store_true", help="print the chosen windows/wearers, write nothing")
    args = ap.parse_args(argv)
    if not 180 <= args.length <= 600:
        ap.error("--length must be 180-600 s")
    rows: list[dict] = []
    if args.plan:
        rows += plan(args.length, args.dry_run)
    loaders = {"sbcsae": load_sbcsae, "ami": load_ami, "chime6": load_chime, "confer": load_confer}
    for spec in args.item:
        corpus, sid, group = (spec.split(":") + ["heated"])[:3]
        s = loaders[corpus](sid)
        if corpus == "confer":
            win = (0.0, round(s.duration_s, 3))
        else:
            win = C.pick_window(s.turns, s.duration_s, length=args.length, heated=group == "heated",
                                exclude=s.exclude, loudness=s.loudness)
        rows.append(make_item(s, win, group, dry_run=args.dry_run))
    if not rows:
        ap.print_help()
        return 1
    if args.dry_run:
        for r in rows:
            print(json.dumps(r))
    else:
        INBOX.mkdir(parents=True, exist_ok=True)
        (INBOX / "corpus_batch.json").write_text(json.dumps(rows, indent=1, default=float))
        hours = sum(r["duration_s"] for r in rows) / 3600
        print(f"[corpus-to-inbox] {len(rows)} items, {hours:.2f} h -> {INBOX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
