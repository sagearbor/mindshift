"""``mindshift-annotation/v1`` — an external audio LLM's reading of a
recording (contract: docs/recording-annotation-format.md + the JSON Schema at
server/tests/fixtures/annotation/mindshift-annotation-v1.schema.json).

LLM output is messy, so everything here is LENIENT: code fences and prose
are stripped, ``.partN`` continuations concatenated, a truncated tail closed,
numeric strings coerced, missing keys filled. Every fix is recorded in
``repairs`` and every doubt in ``problems``; nothing raises on bad input.

:func:`align` re-times the segments against Deepgram's word timings (LLM
timestamps drift by seconds) and maps annotation voices onto Deepgram labels.
"""

from __future__ import annotations

import bisect
import difflib
import json
import re
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

FORMAT = "mindshift-annotation/v1"
EMOTIONS = ("neutral", "calm", "happy", "excited", "amused", "frustrated", "angry", "sad", "anxious", "tired", "sarcastic")
AGES = ("child", "teen", "adult", "older_adult")
EVENT_TYPES = ("interruption", "long_pause", "laughter", "escalation", "de_escalation", "repair", "apology",
               "question", "topic_change", "praise")
# The coarse emotion the phone's tone stand-in and the heat lane use.
COARSE_BY_EMOTION = {
    "angry": "angry", "frustrated": "angry", "sarcastic": "angry",
    "sad": "sad", "tired": "sad", "anxious": "sad",
    "happy": "happy", "excited": "happy", "amused": "happy",
    "neutral": "neutral", "calm": "neutral",
}


@dataclass
class Speaker:
    id: str
    voice_description: str | None = None
    approx_age: str | None = None
    talk_share_pct: float | None = None


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str
    is_backchannel: bool = False
    overlaps_with: list[str] = field(default_factory=list)
    vocal_emotion: str | None = None
    vocal_intensity: int | None = None
    volume: str | None = None
    pace: str | None = None
    text_emotion: str | None = None
    confidence: float | None = None
    # set by align()
    orig_start: float | None = None
    orig_end: float | None = None
    match: float | None = None
    aligned: bool | None = None

    @property
    def coarse(self) -> str | None:
        return COARSE_BY_EMOTION.get(self.vocal_emotion or "")


@dataclass
class Event:
    t: float
    type: str
    speakers: list[str] = field(default_factory=list)
    note: str | None = None
    orig_t: float | None = None


@dataclass
class CoachMoment:
    t: float
    for_speaker: str | None
    what_happened: str | None = None
    ideal_nudge: str | None = None
    kind: str | None = None
    priority: int | None = None
    orig_t: float | None = None


@dataclass
class Annotation:
    ok: bool
    raw: dict | None = None
    model: str | None = None
    annotator_notes: str | None = None
    duration_s: float | None = None
    quality: str | None = None
    environment: str | None = None
    speakers: list[Speaker] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    coach_moments: list[CoachMoment] = field(default_factory=list)
    tone_arc: str | None = None
    overall_heat: int | None = None
    peak_heat_t: float | None = None
    problems: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    label: str = "default"
    source: str | None = None

    def speaker(self, sid: str) -> Speaker | None:
        return next((s for s in self.speakers if s.id == sid), None)


# ---------------------------------------------------------------------------
# Text -> JSON (fences, prose, truncation)
# ---------------------------------------------------------------------------

def _strip_wrappers(text: str, repairs: list[str]) -> str:
    s = text.strip().lstrip("﻿")
    if "```" in s:
        m = re.search(r"```(?:json|JSON)?\s*(.*?)(?:```|$)", s, flags=re.S)
        if m:
            s = m.group(1).strip()
            repairs.append("stripped markdown code fences")
    start = s.find("{")
    if start == -1:
        return s
    if start > 0:
        repairs.append(f"stripped {start} chars of prose before the JSON")
        s = s[start:]
    return s


def _scan_json(s: str) -> tuple[int | None, list[str], bool, int]:
    """Walk ``s``: (index just past the first complete top-level object or
    None, the open-bracket stack at the end, whether we ended inside a string,
    the index of the last position that was safely between values)."""
    stack: list[str] = []
    in_str = False
    esc = False
    last_safe = 0
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                return i + 1, [], False, i + 1
            last_safe = i + 1
        elif ch == ",":
            last_safe = i
    return None, stack, in_str, last_safe


def _close_truncated(s: str, repairs: list[str]) -> str | None:
    """Best-effort close of a cut-off object: back up to the last point that
    was between values, drop a dangling key/comma, then close every open
    bracket. Returns None when nothing usable remains."""
    for _ in range(200):
        end, stack, in_str, last_safe = _scan_json(s)
        if end is not None:
            return s[:end]
        cut = s[:last_safe].rstrip()
        cut = re.sub(r",\s*$", "", cut)
        cut = re.sub(r'[,{]\s*"[^"]*"\s*:?\s*$', lambda m: "{" if m.group(0).lstrip().startswith("{") else "", cut)
        _, stack2, in_str2, _ = _scan_json(cut)
        if in_str2:
            s = cut[:-1]
            continue
        closing = "".join("}" if c == "{" else "]" for c in reversed(stack2))
        candidate = cut + closing
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            if len(cut) < 2 or cut == s:
                return None
            s = cut
            continue
        repairs.append(f"repaired a truncated reply: dropped {len(s) - len(cut)} trailing chars, closed {len(stack2)} bracket(s)")
        return candidate
    return None


def loads_lenient(text: str, continuations: list[str] | None = None) -> tuple[dict | None, list[str], list[str]]:
    repairs: list[str] = []
    problems: list[str] = []
    joined = text or ""
    for part in continuations or []:
        joined = joined.rstrip() + part.lstrip()
        repairs.append("concatenated a continuation part")
    s = _strip_wrappers(joined, repairs)
    if not s.startswith("{"):
        problems.append("no JSON object found in the annotation reply")
        return None, repairs, problems
    end, _, _, _ = _scan_json(s)
    body = s[:end] if end is not None else None
    if end is not None and s[end:].strip():
        repairs.append(f"ignored {len(s[end:].strip())} chars after the JSON object")
    if body is None:
        body = _close_truncated(s, repairs)
        if body is None:
            problems.append("annotation JSON is truncated beyond repair")
            return None, repairs, problems
    # Trailing commas are a classic LLM slip.
    fixed = re.sub(r",\s*([}\]])", r"\1", body)
    if fixed != body:
        repairs.append("removed trailing commas")
    try:
        obj = json.loads(fixed)
    except json.JSONDecodeError as exc:
        problems.append(f"annotation JSON does not parse: {exc}")
        return None, repairs, problems
    if not isinstance(obj, dict):
        problems.append("annotation JSON is not an object")
        return None, repairs, problems
    return obj, repairs, problems


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------

def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().rstrip("s")
        m = re.fullmatch(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", s)
        if m:
            h, mi, se = m.groups()
            return (int(h) * 3600 if h else 0) + int(mi) * 60 + float(se)
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v if v.strip() and v.strip().lower() != "null" else None
    return str(v)


def _enum(v: Any, allowed: tuple[str, ...], where: str, problems: list[str]) -> str | None:
    s = _str(v)
    if s is None:
        return None
    low = s.strip().lower()
    if low in allowed:
        return low
    # "angry|frustrated" (the LLM echoed the menu) -> first allowed option
    for piece in re.split(r"[|/, ]+", low):
        if piece in allowed:
            problems.append(f"{where}: {s!r} reduced to {piece!r}")
            return piece
    problems.append(f"{where}: {s!r} is not one of {'|'.join(allowed)} (ignored)")
    return None


def parse_annotation_obj(obj: dict, *, problems: list[str] | None = None, repairs: list[str] | None = None) -> Annotation:
    problems = list(problems or [])
    repairs = list(repairs or [])
    a = Annotation(ok=True, raw=obj, problems=problems, repairs=repairs)
    fmt = obj.get("format")
    if fmt != FORMAT:
        problems.append(f"format is {fmt!r}, expected {FORMAT!r} (parsed anyway)")
    annotator = obj.get("annotator") if isinstance(obj.get("annotator"), dict) else {}
    a.model = _str(annotator.get("model"))
    a.annotator_notes = _str(annotator.get("notes"))
    audio = obj.get("audio") if isinstance(obj.get("audio"), dict) else {}
    a.duration_s = _num(audio.get("duration_s"))
    a.quality = _enum(audio.get("quality"), ("clean", "ok", "noisy", "very_noisy"), "audio.quality", problems)
    a.environment = _str(audio.get("environment"))

    for i, sp in enumerate(obj.get("speakers") or []):
        if not isinstance(sp, dict) or not _str(sp.get("id")):
            problems.append(f"speakers[{i}]: no id (dropped)")
            continue
        age = sp.get("approx_age")
        a.speakers.append(Speaker(
            id=str(sp["id"]).strip(),
            voice_description=_str(sp.get("voice_description")),
            approx_age=None if _str(age) in (None, "null") else _enum(age, AGES, f"speakers[{i}].approx_age", problems),
            talk_share_pct=_num(sp.get("talk_share_pct")),
        ))
    if not a.speakers:
        problems.append("no speakers listed")

    raw_segs = obj.get("segments")
    if not isinstance(raw_segs, list):
        problems.append("no segments array")
        raw_segs = []
    for i, sg in enumerate(raw_segs):
        if not isinstance(sg, dict):
            problems.append(f"segments[{i}]: not an object (dropped)")
            continue
        start, end = _num(sg.get("start")), _num(sg.get("end"))
        text = _str(sg.get("text"))
        if isinstance(sg.get("start"), str) and start is not None:
            repairs.append(f"segments[{i}].start: coerced {sg['start']!r} to a number")
        if start is None and text is None:
            problems.append(f"segments[{i}]: no start and no text (dropped)")
            continue
        if start is None:
            start = a.segments[-1].end if a.segments else 0.0
            problems.append(f"segments[{i}]: missing start, placed after the previous segment")
        if end is None or end < start:
            words = len((text or "").split())
            end = start + max(0.6, 0.35 * words)
            problems.append(f"segments[{i}]: missing/invalid end, estimated {end:.1f}s from word count")
        vocal = sg.get("vocal") if isinstance(sg.get("vocal"), dict) else {}
        intensity = _num(vocal.get("intensity"))
        speaker = _str(sg.get("speaker")) or "S?"
        if a.speakers and not a.speaker(speaker):
            problems.append(f"segments[{i}]: speaker {speaker!r} not in speakers[]")
        a.segments.append(Segment(
            start=start, end=end, speaker=speaker, text=text or "",
            is_backchannel=bool(sg.get("is_backchannel")),
            overlaps_with=[str(x) for x in (sg.get("overlaps_with") or []) if x],
            vocal_emotion=_enum(vocal.get("emotion"), EMOTIONS, f"segments[{i}].vocal.emotion", problems),
            vocal_intensity=None if intensity is None else int(max(0, min(3, round(intensity)))),
            volume=_enum(vocal.get("volume"), ("quiet", "normal", "raised", "shouting"), f"segments[{i}].vocal.volume", problems),
            pace=_enum(vocal.get("pace"), ("slow", "normal", "fast"), f"segments[{i}].vocal.pace", problems),
            text_emotion=_enum(sg.get("text_emotion"), ("neutral", "positive", "negative", "hostile", "supportive", "questioning"),
                               f"segments[{i}].text_emotion", problems),
            confidence=_num(sg.get("confidence")),
        ))
    a.segments.sort(key=lambda s: s.start)
    if not a.segments:
        problems.append("no usable segments")

    for i, ev in enumerate(obj.get("events") or []):
        if not isinstance(ev, dict) or _num(ev.get("t")) is None:
            problems.append(f"events[{i}]: no time (dropped)")
            continue
        a.events.append(Event(
            t=_num(ev["t"]), type=_enum(ev.get("type"), EVENT_TYPES, f"events[{i}].type", problems) or str(ev.get("type")),
            speakers=[str(x) for x in (ev.get("speakers") or []) if x], note=_str(ev.get("note")),
        ))
    for i, cm in enumerate(obj.get("coach_moments") or []):
        t = _num(cm.get("t")) if isinstance(cm, dict) else None
        if t is None:
            problems.append(f"coach_moments[{i}]: no time (dropped)")
            continue
        if isinstance(cm.get("t"), str):
            repairs.append(f"coach_moments[{i}].t: coerced {cm['t']!r} to {t:g}s")
        pr = _num(cm.get("priority"))
        a.coach_moments.append(CoachMoment(
            t=t, for_speaker=_str(cm.get("for_speaker")), what_happened=_str(cm.get("what_happened")),
            ideal_nudge=_str(cm.get("ideal_nudge")),
            kind=_enum(cm.get("kind"), ("warning", "encouragement"), f"coach_moments[{i}].kind", problems),
            priority=None if pr is None else int(pr),
        ))
    summary = obj.get("summary") if isinstance(obj.get("summary"), dict) else {}
    a.tone_arc = _str(summary.get("tone_arc"))
    oh = _num(summary.get("overall_heat"))
    a.overall_heat = None if oh is None else int(oh)
    a.peak_heat_t = _num(summary.get("peak_heat_t"))
    # A heat-only annotation (corpus ground truth with rated conflict but no
    # speaker turns, e.g. CONFER) still carries moments worth scoring.
    a.ok = bool(a.segments) or (obj.get("retime") is False and bool(a.coach_moments))
    return a


def parse_annotation_text(text: str, continuations: list[str] | None = None) -> Annotation:
    obj, repairs, problems = loads_lenient(text, continuations)
    if obj is None:
        return Annotation(ok=False, problems=problems, repairs=repairs)
    return parse_annotation_obj(obj, problems=problems, repairs=repairs)


def schema_problems(obj: dict | None, schema: dict) -> list[str]:
    """Strict JSON Schema findings (informational; the parse above is what
    the pipeline uses). Empty when ``jsonschema`` is not installed."""
    if obj is None:
        return ["no JSON object"]
    try:
        import jsonschema
    except ImportError:
        return []
    v = jsonschema.Draft202012Validator(schema)
    out = []
    for err in sorted(v.iter_errors(obj), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in err.absolute_path) or "(root)"
        out.append(f"schema: {where}: {err.message[:160]}")
    return out


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

@dataclass
class AnnotationFile:
    label: str
    path: Path
    continuations: list[Path]


def discover(folder: Path, name: str) -> list[AnnotationFile]:
    """``<name>.annotation.json`` (label "default") then
    ``<name>.annotation.<label>.json`` alphabetically, each with its
    ``.partN`` continuations in order."""
    folder = Path(folder)
    out: list[AnnotationFile] = []
    for p in sorted(folder.glob(f"{name}.annotation*.json")):
        middle = p.name[len(name) + len(".annotation"):-len(".json")]
        label = middle.lstrip(".") or "default"
        parts = sorted(folder.glob(p.name + ".part*"), key=lambda q: int(re.sub(r"\D", "", q.suffix) or 0))
        out.append(AnnotationFile(label=label, path=p, continuations=parts))
    out.sort(key=lambda f: (f.label != "default", f.label))
    return out


def load_file(f: AnnotationFile) -> Annotation:
    try:
        text = f.path.read_text(errors="replace")
        conts = [c.read_text(errors="replace") for c in f.continuations]
    except OSError as exc:
        return Annotation(ok=False, problems=[f"{f.path.name}: unreadable: {exc}"], label=f.label, source=str(f.path))
    a = parse_annotation_text(text, conts)
    a.label = f.label
    a.source = str(f.path)
    return a


# ---------------------------------------------------------------------------
# Alignment to Deepgram words
# ---------------------------------------------------------------------------

def norm_token(w: str) -> str:
    w = w.lower().replace("’", "'")
    w = re.sub(r"[^a-z0-9']+", "", w)
    return {"woah": "whoa", "gonna": "going", "wanna": "want", "ok": "okay", "yeah": "yes", "yep": "yes"}.get(w, w)


def tokens(text: str) -> list[str]:
    out = []
    for w in re.split(r"\s+", text or ""):
        if w.startswith("[") or w.endswith("]"):
            continue  # [inaudible]
        t = norm_token(w)
        if t:
            out.append(t)
    return out


def dg_label(speaker: int | str | None) -> str:
    """Deepgram's integer speaker -> the pipeline's "Speaker A" label."""
    if isinstance(speaker, str):
        return speaker
    i = int(speaker or 0)
    return f"Speaker {chr(ord('A') + i)}" if i < 26 else f"Speaker {i + 1}"


@dataclass
class Alignment:
    segments: list[Segment]
    events: list[Event]
    coach_moments: list[CoachMoment]
    speaker_map: dict[str, str]
    quality: dict[str, Any]
    anchors: list[tuple[float, float]]

    def retime(self, t: float) -> float:
        return _piecewise(self.anchors, t)


def _piecewise(anchors: list[tuple[float, float]], t: float) -> float:
    """Map an annotation time onto the audio timeline: linear between the
    aligned anchors, constant shift beyond either end; identity without any."""
    if not anchors:
        return t
    xs = [a for a, _ in anchors]
    i = bisect.bisect_left(xs, t)
    if i == 0:
        return t + (anchors[0][1] - anchors[0][0])
    if i >= len(anchors):
        return t + (anchors[-1][1] - anchors[-1][0])
    (x0, y0), (x1, y1) = anchors[i - 1], anchors[i]
    if x1 == x0:
        return t + (y0 - x0)
    return y0 + (t - x0) * (y1 - y0) / (x1 - x0)


ALIGNED_MIN_MATCH = 0.3


def align(a: Annotation, words: list[dict]) -> Alignment:
    """Re-time every segment from the Deepgram words its tokens matched.

    One global token alignment (difflib on the full token streams) keeps the
    order monotonic and survives drift of tens of seconds. A segment with at
    least ``ALIGNED_MIN_MATCH`` of its tokens matched takes the first/last
    matched word's times; the rest are moved by the piecewise shift of their
    aligned neighbours and flagged ``aligned=False``."""
    dg_tokens: list[str] = []
    dg_idx: list[int] = []
    for i, w in enumerate(words):
        t = norm_token(str(w.get("punctuated_word") or w.get("word") or ""))
        if t:
            dg_tokens.append(t)
            dg_idx.append(i)
    ann_tokens: list[str] = []
    ann_seg: list[int] = []
    for si, s in enumerate(a.segments):
        for t in tokens(s.text):
            ann_tokens.append(t)
            ann_seg.append(si)

    matched_word: dict[int, int] = {}
    if ann_tokens and dg_tokens:
        sm = difflib.SequenceMatcher(None, ann_tokens, dg_tokens, autojunk=False)
        for blk in sm.get_matching_blocks():
            for k in range(blk.size):
                matched_word[blk.a + k] = dg_idx[blk.b + k]

    per_seg: dict[int, list[int]] = {}
    seg_total: dict[int, int] = {}
    for ti, si in enumerate(ann_seg):
        seg_total[si] = seg_total.get(si, 0) + 1
        if ti in matched_word:
            per_seg.setdefault(si, []).append(matched_word[ti])

    out_segs: list[Segment] = []
    anchors: list[tuple[float, float]] = []
    shifts: list[float] = []
    # Human ground truth (corpus transcripts, ``"retime": false``) is already
    # on the audio clock: keep its times, only measure how well it matches.
    keep_times = isinstance(a.raw, dict) and a.raw.get("retime") is False
    for si, s in enumerate(a.segments):
        total = seg_total.get(si, 0)
        hits = per_seg.get(si, [])
        match = (len(hits) / total) if total else 0.0
        ns = replace(s, orig_start=s.start, orig_end=s.end, match=round(match, 3))
        if keep_times:
            ns.aligned = True
            if hits:
                shifts.append(float(words[min(hits)]["start"]) - s.start)
        elif hits and match >= ALIGNED_MIN_MATCH:
            w0, w1 = words[min(hits)], words[max(hits)]
            ns.start, ns.end = float(w0["start"]), float(w1["end"])
            ns.aligned = True
            # Starts only: Deepgram's word END times disagree with an LLM's
            # segment ends by design (trailing silence), so ends would
            # stretch the map; starts are where both put the first word.
            anchors.append((s.start, ns.start))
            shifts.append(ns.start - s.start)
        else:
            ns.aligned = False
        out_segs.append(ns)
    anchors = [] if keep_times else _monotonic(sorted(anchors))   # no anchors = identity map
    for ns in out_segs:
        if not ns.aligned:
            dur = ns.end - ns.start
            ns.start = round(_piecewise(anchors, ns.orig_start), 3)
            ns.end = round(ns.start + dur, 3)

    events = [replace(e, orig_t=e.t, t=round(_piecewise(anchors, e.t), 3)) for e in a.events]
    moments = [replace(m, orig_t=m.t, t=round(_piecewise(anchors, m.t), 3)) for m in a.coach_moments]

    # Annotation voice -> Deepgram label by time overlap of the words.
    overlap: dict[str, dict[str, float]] = {}
    for ns in out_segs:
        for w in words:
            ov = min(ns.end, float(w["end"])) - max(ns.start, float(w["start"]))
            if ov > 0 and "speaker" in w:
                lab = dg_label(w.get("speaker"))
                overlap.setdefault(ns.speaker, {}).setdefault(lab, 0.0)
                overlap[ns.speaker][lab] += ov
    speaker_map = {sid: max(d, key=d.get) for sid, d in overlap.items() if d}

    n_tok = len(ann_tokens)
    quality = {
        "segments": len(out_segs),
        "segments_aligned": sum(1 for s in out_segs if s.aligned),
        "words_matched_pct": round(100.0 * len(matched_word) / n_tok, 1) if n_tok else 0.0,
        "dg_words_covered_pct": round(100.0 * len(set(matched_word.values())) / len(dg_tokens), 1) if dg_tokens else 0.0,
        "median_shift_s": round(statistics.median(shifts), 3) if shifts else None,
        "max_abs_shift_s": round(max(abs(x) for x in shifts), 3) if shifts else None,
        "speaker_overlap_s": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in overlap.items()},
        "retimed": not keep_times,
    }
    return Alignment(out_segs, events, moments, speaker_map, quality, anchors)


def _monotonic(anchors: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop anchors that would make the time map go backwards."""
    out: list[tuple[float, float]] = []
    for x, y in anchors:
        if out and (y < out[-1][1] or x == out[-1][0]):
            continue
        out.append((x, y))
    return out


def aligned_to_dict(al: Alignment, a: Annotation) -> dict:
    """The aligned annotation as JSON (frozen into the fixture)."""
    from dataclasses import asdict
    return {
        "format": FORMAT, "aligned": True, "label": a.label, "model": a.model,
        "speakers": [asdict(s) for s in a.speakers],
        "segments": [asdict(s) for s in al.segments],
        "events": [asdict(e) for e in al.events],
        "coach_moments": [asdict(m) for m in al.coach_moments],
        "speaker_map": al.speaker_map, "quality": al.quality,
        "summary": {"tone_arc": a.tone_arc, "overall_heat": a.overall_heat,
                    "peak_heat_t": None if a.peak_heat_t is None else al.retime(a.peak_heat_t)},
        "problems": a.problems, "repairs": a.repairs,
    }
