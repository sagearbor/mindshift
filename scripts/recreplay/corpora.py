"""Real conversation corpora -> recording-replay inbox items.

The owner cannot collect test recordings, so the pipeline is fed with public
research corpora that already sit in gitignored ``tmp/`` WITH HUMAN GROUND
TRUTH. Each item is a 3-10 minute window of one session:

* ``<name>.wav``                          the window, 16 kHz mono
* ``<name>.notes.txt``                    synthesized notes, marked ``source: corpus ground truth``
* ``<name>.annotation.groundtruth.json``  ``mindshift-annotation/v1`` built from the HUMAN transcript
* ``<name>.voiceprint.json``              the wearer's print, enrolled from a HELD-OUT part of the session
* ``<name>.corpus.json``                  provenance: corpus, window, wearer, group, licence, features

Corpora (all stay in tmp/, never committed):

========  =================================================  ===================  ==================
corpus    ground truth used                                  licence              speaker truth
========  =================================================  ===================  ==================
SBCSAE    TalkBank CHAT transcripts (per-IU times, overlap,   CC BY-ND 3.0         transcript
          laughter, forte/yell voice marks)
AMI       NXT manual words + dialogue acts (backchannel DA)   CC BY 4.0            transcript
CHiME-6   per-utterance JSON transcripts ([laughs] marks)     CHiME-6 licence      transcript
CONFER    10 raters' frame-level conflict intensity (0-1000)  research only        NONE (heat only)
========  =================================================  ===================  ==================

Everything an item claims is derived by a written rule from that ground
truth; the rules are listed in ``annotator.notes`` (``DERIVATION``). Emotion
fields stay null: none of these corpora rate emotion per utterance (CONFER
rates conflict per frame, which becomes moments + summary heat, not a
per-segment emotion). ``volume`` is set only where a transcriber marked a
raised/yelled voice (SBCSAE).
"""

from __future__ import annotations

import bisect
import json
import re
import wave
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

SR = 16000
NITE = "{http://nite.sourceforge.net/}"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    text: str
    backchannel: bool = False
    loud: str | None = None          # "raised" | "shouting" (transcriber-marked only)
    laughing: bool = False           # speech marked as laughed-through


@dataclass
class Session:
    corpus: str
    sid: str
    turns: list[Turn]
    events: list[dict]
    duration_s: float
    setting: str
    speakers: dict[str, str]         # corpus id -> description
    licence: str
    phone: str
    exclude: set[str] = field(default_factory=set)   # never the wearer / not a person (ENV, MANY, X)
    relationship: str | None = None
    conflict: list[float] | None = None              # CONFER: raters' mean conflict per second
    title: str = ""
    description: str = ""
    read_audio: Callable[[float, float], np.ndarray] | None = None
    loudness: np.ndarray | None = None               # dB per second (whole session)
    language: str | None = None                      # not English (CONFER: "el"); goes to annotation audio.language


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_BC_TOKENS = {"mhm", "mm", "mmm", "hm", "hmm", "uh-huh", "uhhuh", "mm-hm", "mmhm", "yeah", "yep", "yup", "yes",
              "right", "okay", "ok", "uh", "um", "oh", "wow", "sure", "huh", "ah", "aha", "i see"}


def is_backchannel_text(text: str) -> bool:
    """A short listener token ("mhm", "yeah right"). A transcript-token
    heuristic for corpora that don't mark backchannels (SBCSAE, CHiME-6)."""
    toks = [t for t in re.findall(r"[a-z][a-z\-']*", (text or "").lower())]
    return 0 < len(toks) <= 2 and all(t in _BC_TOKENS for t in toks)


def _tidy(text: str) -> str:
    s = re.sub(r"\s+", " ", text).strip()
    s = re.sub(r"\s+([.?!,;:])", r"\1", s)
    s = re.sub(r"([.?!,])(?:\s*[.?!,])+", r"\1", s)
    s = re.sub(r"(?:\[inaudible\]\s*){2,}", "[inaudible] ", s).strip()
    return s.strip(" ,")


# ---------------------------------------------------------------------------
# SBCSAE (TalkBank CHAT, CA conventions)
# ---------------------------------------------------------------------------

_BULLET = re.compile(r"\x15(\d+)_(\d+)\x15")
_LOUD = {"F": "raised", "FF": "shouting", "YELL": "shouting", "SHOUT": "shouting"}
_LAUGH_CODES = {"laugh", "laughter", "giggle", "chuckle"}


def _clean_cha(text: str) -> tuple[str, int]:
    """CHAT/CA line -> plain words, plus how many laughter codes it held."""
    laughs = len([m for m in re.findall(r"&=(\w+)", text) if m.lower() in _LAUGH_CODES])
    s = _BULLET.sub(" ", text)
    s = re.sub(r"&[{}]l=[A-Z@]+", " ", s)             # voice-quality brackets (kept text inside)
    s = re.sub(r"&=\w+(?::\w+)?", " ", s)               # &=laugh &=in &=tsk ...
    s = re.sub(r"\[[%^][^\]]*\]", " ", s)               # [% laugh] comments
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"[⌈⌉⌊⌋]\d*", "", s)                     # overlap brackets (may sit mid-word: li⌉⌈2ttle)
    s = re.sub(r"\(\s*\.+\s*\)|\(\d+(?:\.\d+)?\)", " ", s)  # pauses
    s = re.sub(r"\+[^\s]*", " ", s)                     # +... +/. terminators
    s = re.sub(r"(?<!\w)X{1,3}(?!\w)", " [inaudible] ", s)
    s = re.sub(r"(?<=\w):(?=\w|\s|$)", "", s)           # n:othing (lengthening)
    s = re.sub(r"[ʔ↑↓°∙@=~≈≋∬‡„]", "", s)
    s = re.sub(r"&\S*", " ", s)
    s = re.sub(r"\s+\.\s*$", ".", s)
    return _tidy(s), laughs


def parse_cha_text(text: str, sid: str) -> Session:
    lines = text.splitlines()
    comments: list[str] = []
    participants: dict[str, str] = {}
    utts: list[tuple[str, list[str]]] = []
    cur_header: list[str] | None = None
    for ln in lines:
        if ln.startswith("@Comment:"):
            comments.append(ln.split("\t", 1)[-1].strip())
            cur_header = comments
            continue
        if ln.startswith("@Participants:"):
            for part in ln.split("\t", 1)[-1].split(","):
                bits = part.split()
                if bits:
                    participants[bits[0]] = " ".join(bits[1:-1]).title() if len(bits) > 2 else bits[0]
            cur_header = None
            continue
        if ln.startswith("@"):
            cur_header = None
            continue
        if ln.startswith("*"):
            m = re.match(r"\*([A-Z0-9_]+):\t?(.*)", ln)
            if m:
                utts.append((m.group(1), [m.group(2)]))
            cur_header = None
            continue
        if ln.startswith("%"):
            cur_header = None
            utts.append(("%", []))    # dependent tier: ends the utterance's continuation
            continue
        if ln.startswith("\t"):
            if cur_header is comments and comments:
                comments[-1] += " " + ln.strip()
            elif utts and utts[-1][0] != "%":
                utts[-1][1].append(ln.strip())
    turns: list[Turn] = []
    events: list[dict] = []
    for spk, pieces in utts:
        if spk == "%" or not pieces:
            continue
        raw = " ".join(pieces)
        times = [(int(a) / 1000.0, int(b) / 1000.0) for a, b in _BULLET.findall(raw)]
        if not times:
            continue
        start, end = times[0][0], max(b for _, b in times)
        labels = set(re.findall(r"&\{l=([A-Z@]+)", raw))
        words, laughs = _clean_cha(raw)
        if laughs:
            events.append({"t": start, "type": "laughter", "speakers": [spk], "note": "corpus: &=laugh in transcript"})
        if not re.search(r"[A-Za-z]", words.replace("[inaudible]", "")) and "[inaudible]" not in words:
            continue
        loud = None
        for lab in labels:
            if lab in _LOUD and (loud is None or _LOUD[lab] == "shouting"):
                loud = _LOUD[lab]
        turns.append(Turn(speaker=spk, start=start, end=max(end, start + 0.05), text=words,
                          backchannel=is_backchannel_text(words), loud=loud, laughing="@" in labels))
    turns.sort(key=lambda t: t.start)
    title = comments[0] if comments else sid
    desc = " ".join(comments[1:2]).strip()
    dur = max((t.end for t in turns), default=0.0)
    return Session(corpus="SBCSAE", sid=sid, turns=turns, events=sorted(events, key=lambda e: e["t"]), duration_s=dur,
                   setting=f"{title}: {desc}".strip(": "), speakers={k: f"corpus speaker {k} ({v})" for k, v in participants.items()},
                   licence="CC BY-ND 3.0 (SBCSAE, OpenSLR 155)", phone="corpus: one field recording of the room (no phone, no earpiece)",
                   exclude={"ENV", "MANY", "X", "AUD"} | {k for k in participants if k.startswith("X")}, title=title, description=desc)


# ---------------------------------------------------------------------------
# AMI (NXT words + dialogue acts)
# ---------------------------------------------------------------------------

_AMI_BC_DA = "ami_da_1"


def _href_ids(href: str) -> tuple[str, str]:
    ids = re.findall(r"id\(([^)]+)\)", href)
    return (ids[0], ids[-1]) if ids else ("", "")


def parse_ami_speaker(words_xml: str, dacts_xml: str | None, speaker: str) -> tuple[list[Turn], list[dict]]:
    root = ET.fromstring(words_xml.encode("utf-8") if isinstance(words_xml, str) else words_xml)
    elems = []
    for el in root:
        tag = el.tag.split("}")[-1]
        st, en = el.get("starttime"), el.get("endtime")
        elems.append({"id": el.get(f"{NITE}id"), "tag": tag, "start": float(st) if st else None,
                      "end": float(en) if en else None, "text": (el.text or "").strip(),
                      "punc": el.get("punc") == "true", "type": el.get("type")})
    index = {e["id"]: i for i, e in enumerate(elems)}
    events = [{"t": e["start"], "type": "laughter", "speakers": [speaker], "note": "corpus: vocalsound laugh"}
              for e in elems if e["tag"] == "vocalsound" and e["type"] == "laugh" and e["start"] is not None]
    turns: list[Turn] = []
    if dacts_xml:
        droot = ET.fromstring(dacts_xml.encode("utf-8") if isinstance(dacts_xml, str) else dacts_xml)
        for da in droot:
            ptr = da.find(f"{NITE}pointer")
            child = da.find(f"{NITE}child")
            if child is None:
                continue
            a, b = _href_ids(child.get("href", ""))
            if a not in index or b not in index:
                continue
            ws = [e for e in elems[index[a]:index[b] + 1] if e["tag"] == "w" and e["start"] is not None]
            if not ws:
                continue
            text = ""
            for w in ws:
                text += w["text"] if (w["punc"] or not text) else " " + w["text"]
            real = [w for w in ws if not w["punc"]] or ws
            da_type = _href_ids(ptr.get("href", ""))[0] if ptr is not None else ""
            turns.append(Turn(speaker=speaker, start=real[0]["start"], end=max(real[-1]["end"], real[0]["start"] + 0.05),
                              text=_tidy(text), backchannel=da_type == _AMI_BC_DA))
    turns.sort(key=lambda t: t.start)
    return turns, events


# ---------------------------------------------------------------------------
# CHiME-6
# ---------------------------------------------------------------------------

def _hms(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def parse_chime(utts: list[dict]) -> tuple[list[Turn], list[dict], dict[str, int]]:
    turns: list[Turn] = []
    events: list[dict] = []
    locs: Counter = Counter()
    for u in utts:
        try:
            a, b = _hms(u["start_time"]), _hms(u["end_time"])
        except (KeyError, ValueError):
            continue
        words = u.get("words") or ""
        if re.search(r"\[laugh", words, re.I):
            events.append({"t": a, "type": "laughter", "speakers": [u["speaker"]], "note": "corpus: [laughs] in transcript"})
        s = re.sub(r"\[inaudible[^\]]*\]", " [inaudible] ", words, flags=re.I)
        s = re.sub(r"\[(?!inaudible\])[^\]]*\]", " ", s)
        s = _tidy(s)
        if u.get("location"):
            locs[u["location"]] += 1
        if not re.search(r"[A-Za-z]", s):
            continue
        turns.append(Turn(speaker=u["speaker"], start=a, end=max(b, a + 0.05), text=s, backchannel=is_backchannel_text(s)))
    turns.sort(key=lambda t: t.start)
    return turns, sorted(events, key=lambda e: e["t"]), dict(locs)


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def _in_window(turns: list[Turn], a: float, b: float) -> list[Turn]:
    return [t for t in turns if t.end > a and t.start < b]


def _overlap_seconds(turns: list[Turn], a: float, b: float) -> float:
    pts = []
    for t in turns:
        if t.backchannel:
            continue
        s, e = max(t.start, a), min(t.end, b)
        if e > s:
            pts.append((s, 1, t.speaker))
            pts.append((e, -1, t.speaker))
    pts.sort(key=lambda p: (p[0], p[1]))
    active: Counter = Counter()
    last, total = None, 0.0
    for x, d, sp in pts:
        if last is not None and sum(1 for v in active.values() if v > 0) >= 2:
            total += x - last
        active[sp] += d
        last = x
    return total


def _interrupters(turns: list[Turn], a: float, b: float, *, remaining_s: float = 0.5, min_len: float = 0.0) -> list[Turn]:
    """Non-backchannel turns that START while another speaker's
    non-backchannel turn still has >= ``remaining_s`` to go."""
    real = [t for t in turns if not t.backchannel]
    out = []
    for x in real:
        if not (a <= x.start < b) or x.end - x.start < min_len:
            continue
        if any(y.speaker != x.speaker and y.start < x.start and y.end >= x.start + remaining_s for y in real):
            out.append(x)
    return out


def window_features(turns: list[Turn], a: float, b: float, *, exclude: set[str] | None = None,
                    loudness: np.ndarray | None = None, laughs: list[dict] | None = None) -> dict:
    ex = exclude or set()
    ts = [t for t in _in_window(turns, a, b) if t.speaker not in ex]
    talk: Counter = Counter()
    loud: Counter = Counter()
    for t in ts:
        if t.loud:
            loud[t.speaker] += 1
        if not t.backchannel:
            talk[t.speaker] += min(t.end, b) - max(t.start, a)
    ints = Counter(t.speaker for t in _interrupters(ts, a, b))
    # speech coverage: union of all turns
    iv = sorted((max(t.start, a), min(t.end, b)) for t in ts)
    cov, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                cov += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        cov += cur_e - cur_s
    rise = None
    if loudness is not None and len(loudness):
        seg = loudness[int(a):int(b)]
        voiced = loudness[loudness > -60]
        if len(seg) and len(voiced):
            rise = float(np.percentile(seg, 90) - np.median(voiced))
    return {"overlap_s": _overlap_seconds(ts, a, b), "interruptions": dict(ints), "talk_s": dict(talk),
            "loud": dict(loud), "coverage": cov / max(b - a, 1e-6),
            "laughs": sum(1 for e in (laughs or []) if a <= e["t"] < b), "loudness_rise_db": rise}


def heat_score(f: dict, length: float) -> float:
    """How heated a window looks from ground truth alone: overlap share,
    interruption and raised-voice rates, and the loudness rise over the
    session's median (where audio levels are comparable, AMI/CHiME)."""
    mins = max(length / 60.0, 1e-6)
    return (f["overlap_s"] / max(length, 1e-6)
            + 0.10 * sum(f["interruptions"].values()) / mins
            + 0.30 * sum(f["loud"].values()) / mins
            + 0.05 * max(0.0, f.get("loudness_rise_db") or 0.0))


def choose_wearer(turns: list[Turn], a: float, b: float, *, heated: bool, exclude: set[str]) -> tuple[str, str]:
    f = window_features(turns, a, b, exclude=exclude)
    talk = f["talk_s"]
    cands = [s for s in talk if s not in exclude]
    if not cands:
        raise ValueError("no speaker talks in this window")
    if heated:
        def sc(s):
            return 2.0 * f["interruptions"].get(s, 0) + 3.0 * f["loud"].get(s, 0) + talk.get(s, 0.0) / 60.0
        w = max(cands, key=sc)
        return w, (f"most involved in conflict: {f['interruptions'].get(w, 0)} interruptions, "
                   f"{f['loud'].get(w, 0)} raised-voice marks, {talk.get(w, 0.0):.0f} s of talk")
    w = max(cands, key=lambda s: talk.get(s, 0.0))
    return w, f"most talk ({talk.get(w, 0.0):.0f} s of {b - a:.0f} s)"


def pick_window(turns: list[Turn], total_s: float, *, length: float = 300.0, heated: bool, step: float = 15.0,
                avoid: list[tuple[float, float]] | None = None, exclude: set[str] | None = None,
                loudness: np.ndarray | None = None, min_coverage: float = 0.5) -> tuple[float, float]:
    """The hottest (or, for a calm control, the calmest) ``length`` window
    with real two-way talk in it."""
    avoid = avoid or []
    best, best_sc = None, None
    a = 0.0
    turns = sorted(turns, key=lambda t: t.start)
    starts = [t.start for t in turns]
    while a + length <= total_s + 1e-6:
        b = a + length
        if not any(a < y and b > x for x, y in avoid):
            lo = bisect.bisect_left(starts, a - 120.0)
            hi = bisect.bisect_right(starts, b)
            sub = turns[lo:hi]
            f = window_features(sub, a, b, exclude=exclude, loudness=loudness)
            talkers = [s for s, v in f["talk_s"].items() if v >= 0.05 * length]
            if f["coverage"] >= min_coverage and len(talkers) >= 2:
                sc = heat_score(f, length)
                if best_sc is None or (sc > best_sc if heated else sc < best_sc):
                    best, best_sc = (round(a, 3), round(b, 3)), sc
        a += step
    if best is None:
        raise ValueError("no window qualifies (too little two-way talk)")
    return best


# ---------------------------------------------------------------------------
# Coach moments (ground truth only)
# ---------------------------------------------------------------------------

MERGE_S = 8.0
MONOLOGUE_S = 30.0
DERIVATION = (
    "Built by scripts/corpus_to_inbox.py from the corpus's HUMAN transcript; times are the transcript's own "
    "(retime=false, not re-timed to Deepgram). Segments = transcript utterances/intonation units (AMI: dialogue "
    "acts). overlaps_with = time overlap with another speaker's segment. Backchannels: AMI = the corpus's "
    "Backchannel dialogue act; SBCSAE/CHiME-6 = a short listener token (mhm/yeah/right..., <=2 words), a "
    "heuristic. Laughter events = corpus laughter marks (&=laugh, [laughs], vocalsound laugh). volume = "
    "transcriber voice marks only (SBCSAE F -> raised, FF/YELL/SHOUT -> shouting). Emotion fields are null: the "
    "corpus does not rate them. coach_moments (all 'warning', for the wearer): talks-over = the wearer starts a "
    ">=1.5 s turn while another speaker still has >=1 s to go; raised-voice = a transcriber-marked raised/yelled "
    "wearer turn; monologue-cuts-off = the wearer holds the floor >=30 s (gaps <2 s) and someone's attempt to come "
    "in (<=2 s, overlapping) dies; conflict-peak (CONFER) = the 10 raters' mean conflict >=400/1000 for >=3 s. "
    "Moments closer than 8 s merge (higher priority wins)."
)

_NUDGE = {
    "talks-over": "Let them finish first.",
    "raised-voice": "Lower your voice, slow down.",
    "monologue-cuts-off": "Pause, they're trying to speak.",
    "conflict-peak": "It's heating up: slow down, lower voice.",
}


def _merge_moments(ms: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in sorted(ms, key=lambda m: m["t"]):
        if out and m["t"] - out[-1]["t"] < MERGE_S:
            if (m.get("priority") or 9) < (out[-1].get("priority") or 9):
                out[-1] = m
            continue
        out.append(m)
    return out


def derive_moments(segs: list[dict], wearer: str,
                   rules: tuple[str, ...] = ("talks-over", "raised-voice", "monologue-cuts-off")) -> list[dict]:
    real = [s for s in segs if not s.get("is_backchannel")]
    ms: list[dict] = []
    for x in real:
        if x["speaker"] != wearer:
            continue
        if "talks-over" in rules and x["end"] - x["start"] >= 1.5:
            over = [y for y in real if y["speaker"] != wearer and y["start"] < x["start"] and y["end"] >= x["start"] + 1.0]
            if over:
                y = over[0]
                ms.append({"t": round(x["start"], 3), "for_speaker": wearer, "rule": "talks-over", "kind": "warning",
                           "priority": 2, "ideal_nudge": _NUDGE["talks-over"],
                           "what_happened": f"started talking while {y['speaker']} still had "
                                            f"{y['end'] - x['start']:.1f} s to go"})
        if "raised-voice" in rules and x.get("loud") in ("raised", "shouting"):
            ms.append({"t": round(x["start"], 3), "for_speaker": wearer, "rule": "raised-voice", "kind": "warning",
                       "priority": 1 if x["loud"] == "shouting" else 2, "ideal_nudge": _NUDGE["raised-voice"],
                       "what_happened": f"transcriber marked this turn {x['loud']}: {x['text'][:60]!r}"})
    if "monologue-cuts-off" in rules:
        mine = sorted((s for s in real if s["speaker"] == wearer), key=lambda s: s["start"])
        runs: list[list[float]] = []
        for s in mine:
            if runs and s["start"] - runs[-1][1] < 2.0:
                runs[-1][1] = max(runs[-1][1], s["end"])
            else:
                runs.append([s["start"], s["end"]])
        for r0, r1 in runs:
            if r1 - r0 < MONOLOGUE_S:
                continue
            tries = [y for y in real if y["speaker"] != wearer and r0 + 5.0 <= y["start"] < r1
                     and y["end"] - y["start"] <= 2.0]
            if tries:
                y = tries[0]
                ms.append({"t": round(y["end"], 3), "for_speaker": wearer, "rule": "monologue-cuts-off", "kind": "warning",
                           "priority": 3, "ideal_nudge": _NUDGE["monologue-cuts-off"],
                           "what_happened": f"{r1 - r0:.0f} s floor-hold; {y['speaker']} tried to come in ({y['text'][:40]!r})"})
    return _merge_moments(ms)


HEATED_AT = 400.0


def heat_level(v: float) -> int:
    return 0 if v < 200 else 1 if v < HEATED_AT else 2 if v < 600 else 3


def confer_moments(series: list[float], *, threshold: float = HEATED_AT, min_run: int = 3) -> list[dict]:
    ms = []
    start = None
    for i, v in enumerate(list(series) + [-1.0]):
        if v >= threshold and start is None:
            start = i
        elif v < threshold and start is not None:
            if i - start >= min_run:
                peak = max(series[start:i])
                ms.append({"t": float(start), "for_speaker": None, "rule": "conflict-peak", "kind": "warning",
                           "priority": 1 if peak >= 600 else 2, "ideal_nudge": _NUDGE["conflict-peak"],
                           "what_happened": f"raters' mean conflict >= {threshold:.0f}/1000 for {i - start} s (peak {peak:.0f})"})
            start = None
    return _merge_moments(ms)


# ---------------------------------------------------------------------------
# Writing an item
# ---------------------------------------------------------------------------

def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def window_segments(s: Session, window: tuple[float, float]) -> list[Turn]:
    a, b = window
    out = []
    for t in _in_window(s.turns, a, b):
        if t.speaker in s.exclude:
            continue
        st, en = max(t.start, a), min(t.end, b)
        if en - st >= 0.15:
            out.append(Turn(t.speaker, st, en, t.text, t.backchannel, t.loud, t.laughing))
    return out


def build_annotation(s: Session, window: tuple[float, float], *, wearer: str | None, group: str,
                     derivation: str = DERIVATION) -> tuple[dict, dict[str, str]]:
    a, b = window
    segs = window_segments(s, window)
    idmap: dict[str, str] = {}
    for t in segs:
        idmap.setdefault(t.speaker, f"S{len(idmap) + 1}")
    rel = []
    for t in segs:
        rel.append({"start": round(t.start - a, 3), "end": round(t.end - a, 3), "speaker": idmap[t.speaker],
                    "text": t.text, "is_backchannel": bool(t.backchannel), "overlaps_with": [],
                    "vocal": {"emotion": None, "intensity": None, "volume": t.loud, "pace": None},
                    "text_emotion": None, "confidence": 1.0, "_loud": t.loud})
    for x in rel:
        x["overlaps_with"] = sorted({y["speaker"] for y in rel if y["speaker"] != x["speaker"]
                                     and min(x["end"], y["end"]) - max(x["start"], y["start"]) > 0.05})
    wid = idmap.get(wearer) if wearer else None
    moments = derive_moments([{**x, "loud": x["_loud"]} for x in rel], wid) if wid else []
    summary = {"tone_arc": None, "overall_heat": None, "peak_heat_t": None}
    if s.conflict is not None:
        ser = list(s.conflict[int(a):int(np.ceil(b))])
        moments = confer_moments(ser)
        if ser:
            pk = int(np.argmax(ser))
            summary = {"tone_arc": f"CONFER raters' mean conflict: mean {np.mean(ser):.0f}, peak {max(ser):.0f}/1000 at {pk} s",
                       "overall_heat": heat_level(float(np.mean(ser))), "peak_heat_t": float(pk)}
    for x in rel:
        x.pop("_loud")
    events = []
    for e in s.events:
        if a <= e["t"] < b:
            events.append({"t": round(e["t"] - a, 3), "type": e["type"],
                           "speakers": [idmap[sp] for sp in e.get("speakers", []) if sp in idmap], "note": e.get("note")})
    talk: Counter = Counter()
    for x in rel:
        talk[x["speaker"]] += x["end"] - x["start"]
    tot = sum(talk.values()) or 1.0
    inv = {v: k for k, v in idmap.items()}
    speakers = [{"id": sid, "voice_description": s.speakers.get(inv[sid], f"corpus speaker {inv[sid]}")
                 + (" (the wearer)" if inv[sid] == wearer else ""),
                 "approx_age": None, "talk_share_pct": round(100.0 * talk[sid] / tot, 1)} for sid in idmap.values()]
    obj = {
        "format": "mindshift-annotation/v1",
        "annotator": {"model": f"corpus-ground-truth:{s.corpus}", "notes": derivation},
        "audio": {"duration_s": round(b - a, 3), "quality": "ok", "environment": s.setting[:120],
                  **({"language": s.language} if s.language else {})},
        "retime": False,
        "source": {"kind": "corpus ground truth", "corpus": s.corpus, "session": s.sid, "window_s": [a, b],
                   "licence": s.licence, "group": group, "speaker_ids": idmap},
        "speakers": speakers,
        "segments": rel,
        "events": events,
        "coach_moments": moments,
        "summary": summary,
    }
    return obj, idmap


def build_notes(s: Session, window: tuple[float, float], *, wearer: str | None, idmap: dict[str, str], why: str,
                group: str) -> str:
    a, b = window
    lines = [
        f"# source: corpus ground truth — synthesized by scripts/corpus_to_inbox.py from {s.corpus} {s.sid} "
        f"({s.licence}); NOT the owner's notes.",
        f"# group: {group}; window {_mmss(a)}-{_mmss(b)} of the session ({a:.0f}-{b:.0f} s)",
    ]
    if wearer and wearer in idmap:
        others = ", ".join(f"{v} = corpus speaker {k}" for k, v in idmap.items() if k != wearer)
        lines.append(f"who: I'm {idmap[wearer]} (corpus speaker {wearer}, picked as the wearer: {why}); "
                     f"other voices: {others or 'none'}")
    else:
        lines.append(f"who: unknown, {why}")
    lines.append(f"setting: {s.setting}")
    lines.append(f"phone: {s.phone}")
    if s.relationship:
        lines.append(f"relationship: {s.relationship}")
    return "\n".join(lines) + "\n"


def heldout_turns(turns: list[Turn], speaker: str, window: tuple[float, float], *, guard_s: float = 30.0,
                  min_piece_s: float = 0.5, exclude: set[str] | None = None) -> list[Turn]:
    """The wearer's SOLO speech outside the tested window (+- guard): what a
    voiceprint may be enrolled from without seeing the window it is scored on."""
    a, b = window[0] - guard_s, window[1] + guard_s
    ex = exclude or set()
    others = sorted((t.start, t.end) for t in turns if t.speaker != speaker and t.speaker not in ex)
    out: list[Turn] = []
    for t in turns:
        if t.speaker != speaker or t.backchannel or (t.end > a and t.start < b):
            continue
        pieces = [(t.start, t.end)]
        for os_, oe in others:
            if oe <= t.start or os_ >= t.end:
                continue
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
        out += [Turn(speaker, ps, pe, t.text) for ps, pe in pieces if pe - ps >= min_piece_s]
    return out


def voiceprint_document(embedding: np.ndarray, *, recording_id: str, speaker: str, seconds: float,
                        window: tuple[float, float]) -> dict:
    import speaker_id
    return speaker_id.new_profile(
        np.asarray(embedding, dtype=np.float32), None, recording_id=recording_id, speaker=speaker,
        now_iso=datetime.now(timezone.utc).isoformat(), seconds=seconds,
        note=(f"held-out corpus enrollment: {seconds:.0f} s of {speaker}'s solo speech from the same session, "
              f"outside the tested window {window[0]:.0f}-{window[1]:.0f} s (+-30 s guard)"),
    )


def enroll_heldout(s: Session, speaker: str, window: tuple[float, float], *, max_seconds: float = 60.0) -> tuple[dict | None, str]:
    """(voiceprint document | None, why). Uses the server's own enrollment
    path: speaker_id.embed_pcm on pooled turns + speaker_id.new_profile."""
    import speaker_id
    if s.read_audio is None:
        return None, "no audio reader"
    held = sorted(heldout_turns(s.turns, speaker, window, exclude=s.exclude), key=lambda t: t.end - t.start, reverse=True)
    chunks, total = [], 0.0
    for t in held:
        if t.end - t.start < 1.0:
            continue
        pcm = s.read_audio(t.start, t.end).astype(np.float32) / 32768.0
        if pcm.size:
            chunks.append(pcm)
            total += pcm.size / SR
        if total >= max_seconds:
            break
    if not chunks:
        return None, "no held-out solo speech"
    pooled = np.concatenate(chunks)
    speech = speaker_id.speech_seconds(pooled, SR)
    if speech < speaker_id.MIN_ENROLL_SECONDS:
        return None, f"only {speech:.1f} s of held-out speech"
    emb = speaker_id.embed_pcm(pooled, SR)
    doc = voiceprint_document(emb, recording_id=f"corpus:{s.corpus}:{s.sid}", speaker=speaker, seconds=total, window=window)
    return doc, f"{total:.0f} s pooled from {len(chunks)} held-out solo turns ({speech:.0f} s voiced)"


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

class WavSource:
    """One session's audio, possibly split over consecutive files."""

    def __init__(self, parts: list[tuple[Path, float]]):
        self.parts = []
        for p, off in parts:
            with wave.open(str(p)) as w:
                assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, p
                self.parts.append((Path(p), off, w.getnframes() / SR))
        self.duration_s = max(off + d for _, off, d in self.parts) if self.parts else 0.0

    def read(self, a: float, b: float) -> np.ndarray:
        out = []
        for p, off, d in self.parts:
            s, e = max(a, off), min(b, off + d)
            if e <= s:
                continue
            with wave.open(str(p)) as w:
                w.setpos(int((s - off) * SR))
                out.append(np.frombuffer(w.readframes(int((e - s) * SR)), dtype="<i2"))
        return np.concatenate(out) if out else np.zeros(0, dtype="<i2")

    def loudness_per_second(self) -> np.ndarray:
        vals = []
        for p, off, d in self.parts:
            with wave.open(str(p)) as w:
                n = w.getnframes()
                for _ in range(int(n // SR)):
                    x = np.frombuffer(w.readframes(SR), dtype="<i2").astype(np.float32) / 32768.0
                    vals.append(20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9))
        return np.asarray(vals, dtype=np.float32)


def write_wav(path: Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.asarray(pcm, dtype="<i2").tobytes())


def write_item(inbox: Path, name: str, s: Session, window: tuple[float, float], *, group: str,
               wearer: str | None, why: str, enroll: bool = True) -> dict:
    """Write one inbox item; returns its provenance record."""
    d = Path(inbox) / name
    d.mkdir(parents=True, exist_ok=True)
    a, b = window
    write_wav(d / f"{name}.wav", s.read_audio(a, b))
    obj, idmap = build_annotation(s, window, wearer=wearer, group=group)
    (d / f"{name}.annotation.groundtruth.json").write_text(json.dumps(obj, indent=1))
    (d / f"{name}.notes.txt").write_text(build_notes(s, window, wearer=wearer, idmap=idmap, why=why, group=group))
    vp_why = "not enrolled"
    if enroll and wearer:
        doc, vp_why = enroll_heldout(s, wearer, window)
        if doc is not None:
            (d / f"{name}.voiceprint.json").write_text(json.dumps(doc))
    feats = window_features(s.turns, a, b, exclude=s.exclude, loudness=s.loudness, laughs=s.events) if s.turns else {}
    prov = {"name": name, "corpus": s.corpus, "session": s.sid, "window_s": [a, b], "duration_s": round(b - a, 3),
            "group": group, "wearer": wearer, "wearer_id": idmap.get(wearer) if wearer else None, "wearer_why": why,
            "voiceprint": vp_why, "licence": s.licence, "features": feats,
            "heat_score": round(heat_score(feats, b - a), 4) if feats else None,
            "coach_moments": len(obj["coach_moments"]), "segments": len(obj["segments"])}
    (d / f"{name}.corpus.json").write_text(json.dumps(prov, indent=1))
    return prov
