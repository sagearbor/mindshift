"""Pure parts of the Gemini audio annotator: the prompt, cost estimation with
a persistent hard-capped ledger, windowing long audio, stitching window
annotations back together, and sanitising model output into
``mindshift-annotation/v1`` (schema:
server/tests/fixtures/annotation/mindshift-annotation-v1.schema.json).

No network here; :mod:`annotator.gemini` does the calls.
"""

from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompt: verbatim from docs/recording-annotation-format.md /
# tmp/annotation-prompt.html (a test checks it stays in sync with the doc).
# ---------------------------------------------------------------------------

PROMPT = """You are annotating an audio recording of a real conversation for a research tool that coaches people to communicate better. Listen to the WHOLE file carefully, then output ONE JSON object and nothing else (no prose, no markdown fences).

RULES
- Timestamps are seconds from the start of the file, with 1 decimal place. Be as accurate as you can; it is fine to be approximate, but never make segments up.
- Never invent words. If you can't make something out, write [inaudible]; if you're unsure, use your best guess and set "confidence" low.
- Do NOT guess who people are. Label voices S1, S2, S3… in order of first appearance and describe each voice so a human can match it.
- A new segment starts whenever the speaker changes or a speaker pauses more than ~1 second. Keep segments under ~15 seconds.
- Judge emotion from the VOICE (tone, loudness, pace) and from the WORDS separately. They often disagree, e.g. excited vs angry are both loud; record both.
- Mark overlapping speech (two people talking at once) and short listener sounds ("mm-hm", "yeah").
- Also mark "coach_moments": points where a helpful, private coach in ONE speaker's ear could have helped that speaker be a more positive presence (e.g. they interrupted, escalated, didn't acknowledge, went silent, or did something well worth reinforcing). Give a nudge of 10 words or fewer. Never script what the other person should say.

OUTPUT FORMAT (exactly these keys; use null where unknown)
{
  "format": "mindshift-annotation/v1",
  "annotator": {"model": "<your model name>", "notes": "<anything about audio quality or your confidence>"},
  "audio": {"duration_s": 0.0, "quality": "clean|ok|noisy|very_noisy", "environment": "<e.g. quiet room, train, restaurant>"},
  "speakers": [
    {"id": "S1", "voice_description": "<e.g. adult male, low voice, fast talker>", "approx_age": "child|teen|adult|older_adult|null", "talk_share_pct": 0}
  ],
  "segments": [
    {
      "start": 0.0, "end": 0.0, "speaker": "S1",
      "text": "<verbatim words>",
      "is_backchannel": false,
      "overlaps_with": [],
      "vocal": {"emotion": "neutral|calm|happy|excited|amused|frustrated|angry|sad|anxious|tired|sarcastic", "intensity": 0, "volume": "quiet|normal|raised|shouting", "pace": "slow|normal|fast"},
      "text_emotion": "neutral|positive|negative|hostile|supportive|questioning",
      "confidence": 0.0
    }
  ],
  "events": [
    {"t": 0.0, "type": "interruption|long_pause|laughter|escalation|de_escalation|repair|apology|question|topic_change|praise", "speakers": ["S1"], "note": "<short>"}
  ],
  "coach_moments": [
    {"t": 0.0, "for_speaker": "S1", "what_happened": "<short>", "ideal_nudge": "<=10 words", "kind": "warning|encouragement", "priority": 1}
  ],
  "summary": {"tone_arc": "<one or two sentences on how the mood moved>", "overall_heat": 0, "peak_heat_t": null}
}

SCALES: intensity and overall_heat are 0 = none, 1 = mild, 2 = clear, 3 = strong. priority 1 = most important. talk_share_pct across speakers should sum to about 100."""

CONTINUE_PROMPT = ("Your JSON was cut off. Continue EXACTLY where it stopped, with no repetition and no "
                   "commentary, so the two parts concatenate into valid JSON.")


def window_note(start: float, end: float, total: float, known: list[dict]) -> str:
    """Appended to PROMPT when the audio is one window of a longer file."""
    lines = [
        "",
        f"NOTE: this audio is ONE WINDOW of a longer recording: {start:.1f}s to {end:.1f}s of a {total:.1f}s file. "
        "Give every timestamp relative to the start of THIS audio (0.0 = the start of this window); the tool adds the offset.",
    ]
    if known:
        lines.append("Voices already identified in earlier windows (REUSE these ids when the same voice speaks; "
                     "number any new voice after the last one):")
        for s in known:
            lines.append(f"- {s.get('id')}: {s.get('voice_description') or 'no description'}"
                         f" ({s.get('approx_age') or 'age unknown'})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Raw-reply repairs (before the lenient JSON parse)
# ---------------------------------------------------------------------------

_CLOCK = re.compile(r'("(?:start|end|t|peak_heat_t)"\s*:\s*)(\d+):(\d{1,2}(?:\.\d+)?)(?::(\d{1,2}(?:\.\d+)?))?')


def repair_text(text: str) -> str:
    """Models sometimes write clock times (``"start": 1:52.4``), which is not
    JSON. Convert them to seconds."""
    def sub(m):
        a, b, c = m.group(2), m.group(3), m.group(4)
        secs = int(a) * 3600 + int(float(b)) * 60 + float(c) if c is not None else int(a) * 60 + float(b)
        return f"{m.group(1)}{secs:.2f}"
    return _CLOCK.sub(sub, text)


def _strip_fence(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    return re.sub(r"\s*```\s*$", "", s)


def _open_stack(s: str) -> list[tuple[str, int]]:
    stack: list[tuple[str, int]] = []
    in_str = esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append((ch, i))
        elif ch in "]}" and stack:
            stack.pop()
    return stack


def join_parts(parts: list[str]) -> str:
    """Concatenate a cut-off reply and its continuation(s). The follow-up asks
    for an exact continuation, but models often restart the half-written
    array element instead (``{"start": ...``). Then the half element is cut
    from the first part before joining."""
    out = _strip_fence(parts[0]) if parts else ""
    for cont in parts[1:]:
        c = _strip_fence(cont)
        if c.startswith("{") and not out.rstrip().endswith((",", "[")):
            stack = _open_stack(out)
            cut = None
            for k in range(len(stack) - 1, 0, -1):
                if stack[k][0] == "{" and stack[k - 1][0] == "[":
                    cut = stack[k][1]
                    break
            if cut is not None:
                out = out[:cut]
        out = out + c
    return out


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

# USD per 1M tokens. Deliberately CONSERVATIVE (the >200k-context tier for the
# pro models; the newer flash models, whose list prices we could not confirm,
# at roughly twice gemini-3-flash). Thinking tokens bill as output.
PRICES: dict[str, dict[str, float]] = {
    "gemini-2.5-pro": {"text_in": 2.50, "audio_in": 2.50, "out": 15.0},
    "gemini-2.5-flash": {"text_in": 0.30, "audio_in": 1.00, "out": 2.50},
    "gemini-2.5-flash-lite": {"text_in": 0.10, "audio_in": 0.30, "out": 0.40},
    "gemini-3-pro-preview": {"text_in": 4.00, "audio_in": 4.00, "out": 18.0},
    "gemini-3.1-pro-preview": {"text_in": 4.00, "audio_in": 4.00, "out": 18.0},
    "gemini-3-flash-preview": {"text_in": 0.50, "audio_in": 1.00, "out": 3.00},
    "gemini-3.5-flash": {"text_in": 1.00, "audio_in": 2.00, "out": 6.00},
    "gemini-3.6-flash": {"text_in": 1.00, "audio_in": 2.00, "out": 6.00},
    "gemini-3.7-flash": {"text_in": 1.00, "audio_in": 2.00, "out": 6.00},
    "gemini-3.8-flash": {"text_in": 1.00, "audio_in": 2.00, "out": 6.00},
}
AUDIO_TOKENS_PER_S = 32  # Gemini's documented audio tokenisation


def price_for(model: str) -> dict[str, float]:
    if model in PRICES:
        return PRICES[model]
    return {k: max(p[k] for p in PRICES.values()) for k in ("text_in", "audio_in", "out")}


def estimate_cost(model: str, usage: dict) -> float:
    p = price_for(model)
    return (usage.get("text_in", 0) * p["text_in"] + usage.get("audio_in", 0) * p["audio_in"]
            + (usage.get("out", 0) + usage.get("thoughts", 0)) * p["out"]) / 1e6


def projected_cost(model: str, audio_s: float, max_out_tokens: int) -> float:
    """Worst case for one call: the audio + prompt in, every output token used."""
    return estimate_cost(model, {"audio_in": audio_s * AUDIO_TOKENS_PER_S, "text_in": 3000,
                                 "out": max_out_tokens})


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Spend across the WHOLE effort, persisted to JSON so separate runs share
    one cap. :meth:`check` raises before a call that could cross the cap."""

    def __init__(self, path: Path, cap_usd: float = 25.0):
        self.path = Path(path)
        self.cap = float(cap_usd)
        self.calls: list[dict] = []
        if self.path.exists():
            try:
                self.calls = json.loads(self.path.read_text()).get("calls", [])
            except (OSError, json.JSONDecodeError):
                raise RuntimeError(f"spend ledger {self.path} is unreadable; refusing to run without it")

    @property
    def total(self) -> float:
        return float(sum(c.get("usd", 0.0) for c in self.calls))

    def check(self, projected_usd: float) -> None:
        if self.total >= self.cap or self.total + projected_usd > self.cap:
            raise BudgetExceeded(f"spend cap ${self.cap:.2f}: spent ${self.total:.4f}, "
                                 f"next call could cost up to ${projected_usd:.4f}")

    def record(self, model: str, item: str, usd: float, usage: dict) -> None:
        self.calls.append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model, "item": item,
                           "usd": round(float(usd), 6), "usage": usage})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"cap_usd": self.cap, "total_usd": round(self.total, 6),
                                   "calls": self.calls}, indent=1))
        tmp.replace(self.path)


# ---------------------------------------------------------------------------
# Windows + stitching
# ---------------------------------------------------------------------------

def plan_windows(duration: float, window_s: float = 600.0, overlap_s: float = 30.0) -> list[tuple[float, float]]:
    if duration <= window_s:
        return [(0.0, float(duration))]
    out = []
    start = 0.0
    while True:
        end = min(duration, start + window_s)
        out.append((round(start, 3), round(end, 3)))
        if end >= duration:
            break
        start = end - overlap_s
    return out


def offset(obj: dict, dt: float) -> dict:
    o = copy.deepcopy(obj)
    for s in o.get("segments") or []:
        for k in ("start", "end"):
            if isinstance(s.get(k), (int, float)):
                s[k] = round(s[k] + dt, 3)
    for key in ("events", "coach_moments"):
        for e in o.get(key) or []:
            if isinstance(e.get("t"), (int, float)):
                e["t"] = round(e["t"] + dt, 3)
    summ = o.get("summary") or {}
    if isinstance(summ.get("peak_heat_t"), (int, float)):
        summ["peak_heat_t"] = round(summ["peak_heat_t"] + dt, 3)
    return o


def _span_overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _next_id(ids) -> str:
    n = max([int(i[1:]) for i in ids if re.fullmatch(r"S\d+", i)] or [0])
    return f"S{n + 1}"


def stitch(window_objs: list[dict], windows: list[tuple[float, float]]) -> dict:
    """Merge per-window annotations (times relative to each window) into one.

    Window k owns [cut_{k-1}, cut_k) where a cut is the middle of the overlap
    between neighbouring windows. Window speaker ids are mapped to the running
    global ids by time overlap of their segments inside the shared overlap
    region; a window id with no overlap evidence keeps its own id if the
    global set has it and nobody claimed it (the prompt asked the model to
    reuse ids), otherwise it becomes a new global id."""
    shifted = [offset(o, w[0]) for o, w in zip(window_objs, windows)]
    cuts = [(windows[i][1] + windows[i + 1][0]) / 2.0 for i in range(len(windows) - 1)]
    out = {"speakers": [], "segments": [], "events": [], "coach_moments": []}
    speakers: dict[str, dict] = {}
    best_heat, peak_t, arcs, notes = None, None, [], []
    audio = None
    prev_tail: list[dict] = []
    for k, o in enumerate(shifted):
        lo = cuts[k - 1] if k > 0 else float("-inf")
        hi = cuts[k] if k < len(cuts) else float("inf")
        mapping: dict[str, str] = {}
        if k == 0:
            for s in o.get("speakers") or []:
                mapping[s["id"]] = s["id"]
        else:
            ov0, ov1 = windows[k][0], windows[k - 1][1]
            votes: dict[tuple[str, str], float] = {}
            for s in o.get("segments") or []:
                a, b = max(s.get("start", 0), ov0), min(s.get("end", 0), ov1)
                if b <= a:
                    continue
                for p in prev_tail:
                    d = _span_overlap(a, b, p["start"], p["end"])
                    if d > 0:
                        key = (s["speaker"], p["speaker"])
                        votes[key] = votes.get(key, 0.0) + d
            claimed = set()
            for (w, g), _ in sorted(votes.items(), key=lambda kv: -kv[1]):
                if w not in mapping and g not in claimed:
                    mapping[w] = g
                    claimed.add(g)
            for s in o.get("speakers") or []:
                if s["id"] in mapping:
                    continue
                if s["id"] in speakers and s["id"] not in claimed:
                    mapping[s["id"]] = s["id"]
                    claimed.add(s["id"])
                else:
                    mapping[s["id"]] = _next_id(list(speakers) + list(mapping.values()))
        for s in o.get("speakers") or []:
            g = mapping.get(s["id"], s["id"])
            if g not in speakers:
                speakers[g] = dict(s, id=g)
        # full window segments (all of them) are kept aside as the overlap
        # reference for the next window
        prev_tail = [dict(s, speaker=mapping.get(s.get("speaker"), s.get("speaker"))) for s in o.get("segments") or []]
        for s in o.get("segments") or []:
            if lo <= s.get("start", 0) < hi:
                s = dict(s, speaker=mapping.get(s.get("speaker"), s.get("speaker")))
                if s.get("overlaps_with"):
                    s["overlaps_with"] = [mapping.get(x, x) for x in s["overlaps_with"]]
                out["segments"].append(s)
        for e in o.get("events") or []:
            if lo <= e.get("t", 0) < hi:
                out["events"].append(dict(e, speakers=[mapping.get(x, x) for x in e.get("speakers") or []]))
        for c in o.get("coach_moments") or []:
            if lo <= c.get("t", 0) < hi:
                out["coach_moments"].append(dict(c, for_speaker=mapping.get(c.get("for_speaker"), c.get("for_speaker"))))
        summ = o.get("summary") or {}
        h = summ.get("overall_heat")
        if isinstance(h, (int, float)) and (best_heat is None or h > best_heat):
            best_heat, peak_t = h, summ.get("peak_heat_t")
        if summ.get("tone_arc"):
            arcs.append(f"[{windows[k][0]:.0f}-{windows[k][1]:.0f}s] {summ['tone_arc']}")
        ann = o.get("annotator") or {}
        if ann.get("notes"):
            notes.append(ann["notes"])
        audio = audio or o.get("audio")
    out["speakers"] = [speakers[k] for k in sorted(speakers, key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)]
    out["segments"].sort(key=lambda s: s["start"])
    out["events"].sort(key=lambda e: e["t"])
    out["coach_moments"].sort(key=lambda c: c["t"])
    out["summary"] = {"tone_arc": " ".join(arcs) or None, "overall_heat": best_heat, "peak_heat_t": peak_t}
    out["annotator"] = {"notes": " | ".join(notes) or None}
    out["audio"] = dict(audio or {})
    return out


# ---------------------------------------------------------------------------
# Sanitise
# ---------------------------------------------------------------------------

EMOTIONS = ("neutral", "calm", "happy", "excited", "amused", "frustrated", "angry", "sad", "anxious", "tired", "sarcastic")
VOLUMES = ("quiet", "normal", "raised", "shouting")
PACES = ("slow", "normal", "fast")
TEXT_EMOTIONS = ("neutral", "positive", "negative", "hostile", "supportive", "questioning")
EVENT_TYPES = ("interruption", "long_pause", "laughter", "escalation", "de_escalation", "repair", "apology",
               "question", "topic_change", "praise")
AGES = ("child", "teen", "adult", "older_adult")
QUALITIES = ("clean", "ok", "noisy", "very_noisy")


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def _enum(v, allowed, where, fixes):
    if v is None:
        return None
    s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
    if s in allowed:
        return s
    if s not in ("", "null", "none", "unknown"):
        fixes.append(f"{where}: {v!r} not in the allowed set, set to null")
    return None


def _int_clamp(v, lo, hi, where, fixes):
    n = _num(v)
    if n is None:
        return None
    r = int(round(n))
    c = min(hi, max(lo, r)) if hi is not None else max(lo, r)
    if c != n:
        fixes.append(f"{where}: {v!r} -> {c}")
    return c


def sanitize(obj: dict, *, model: str, duration_s: float | None, fixes: list[str]) -> dict:
    o = obj if isinstance(obj, dict) else {}
    # speaker ids -> S<n>
    idmap: dict[str, str] = {}
    raw_speakers = [s for s in (o.get("speakers") or []) if isinstance(s, dict)]
    for s in raw_speakers:
        sid = str(s.get("id") or "")
        if re.fullmatch(r"S\d+", sid):
            idmap[sid] = sid
    for s in raw_speakers:
        sid = str(s.get("id") or "")
        if sid in idmap:
            continue
        m = re.search(r"(\d+)", sid)
        cand = f"S{m.group(1)}" if m and f"S{m.group(1)}" not in idmap.values() else _next_id(idmap.values())
        idmap[sid] = cand
        fixes.append(f"speaker id {sid!r} -> {cand}")

    def sp(x):
        x = str(x) if x is not None else ""
        if x in idmap:
            return idmap[x]
        if re.fullmatch(r"S\d+", x):
            return x
        m = re.search(r"(\d+)", x)
        return f"S{m.group(1)}" if m else (x or "S0")

    segs = []
    for i, s in enumerate(o.get("segments") or []):
        if not isinstance(s, dict):
            continue
        st, en = _num(s.get("start")), _num(s.get("end"))
        if st is None:
            fixes.append(f"segments[{i}] dropped: no start")
            continue
        st = max(0.0, st)
        if en is None or en < st:
            fixes.append(f"segments[{i}]: end {s.get('end')!r} < start; set end = start")
            en = st
        v = s.get("vocal") if isinstance(s.get("vocal"), dict) else {}
        conf = _num(s.get("confidence"))
        if conf is not None and not 0 <= conf <= 1:
            fixes.append(f"segments[{i}].confidence {conf} clamped")
            conf = min(1.0, max(0.0, conf))
        ow = s.get("overlaps_with")
        segs.append({
            "start": round(st, 2), "end": round(en, 2), "speaker": sp(s.get("speaker")),
            "text": str(s.get("text")) if s.get("text") is not None else "",
            "is_backchannel": s.get("is_backchannel") if isinstance(s.get("is_backchannel"), bool) else None,
            "overlaps_with": [sp(x) for x in ow] if isinstance(ow, list) else [],
            "vocal": {
                "emotion": _enum(v.get("emotion"), EMOTIONS, f"segments[{i}].vocal.emotion", fixes),
                "intensity": _int_clamp(v.get("intensity"), 0, 3, f"segments[{i}].vocal.intensity", fixes),
                "volume": _enum(v.get("volume"), VOLUMES, f"segments[{i}].vocal.volume", fixes),
                "pace": _enum(v.get("pace"), PACES, f"segments[{i}].vocal.pace", fixes),
            },
            "text_emotion": _enum(s.get("text_emotion"), TEXT_EMOTIONS, f"segments[{i}].text_emotion", fixes),
            "confidence": conf,
        })
    segs.sort(key=lambda s: s["start"])

    talk: dict[str, float] = {}
    for s in segs:
        talk[s["speaker"]] = talk.get(s["speaker"], 0.0) + (s["end"] - s["start"])
    tot = sum(talk.values()) or 1.0
    speakers = []
    seen = set()
    for s in raw_speakers:
        sid = idmap[str(s.get("id") or "")]
        if sid in seen:
            continue
        seen.add(sid)
        share = _num(s.get("talk_share_pct"))
        if share is None or not 0 <= share <= 100:
            fixes.append(f"speaker {sid} talk_share_pct {s.get('talk_share_pct')!r} recomputed from segments")
            share = round(100.0 * talk.get(sid, 0.0) / tot, 1)
        speakers.append({"id": sid, "voice_description": s.get("voice_description") if s.get("voice_description") is not None else None,
                         "approx_age": _enum(s.get("approx_age"), AGES, f"speaker {sid}.approx_age", fixes),
                         "talk_share_pct": share})
    for sid in talk:
        if sid not in seen and re.fullmatch(r"S\d+", sid):
            fixes.append(f"speaker {sid} used in segments but not listed; added")
            speakers.append({"id": sid, "voice_description": None, "approx_age": None,
                             "talk_share_pct": round(100.0 * talk[sid] / tot, 1)})
            seen.add(sid)
    speakers.sort(key=lambda s: int(s["id"][1:]))
    for s in segs:  # speaker must be a listed S<n>
        if s["speaker"] not in seen:
            s["speaker"] = speakers[0]["id"] if speakers else "S1"

    events = []
    for i, e in enumerate(o.get("events") or []):
        if not isinstance(e, dict):
            continue
        t = _num(e.get("t"))
        typ = _enum(e.get("type"), EVENT_TYPES, f"events[{i}].type", fixes)
        if t is None or typ is None:
            fixes.append(f"events[{i}] dropped (t={e.get('t')!r}, type={e.get('type')!r})")
            continue
        events.append({"t": round(max(0.0, t), 2), "type": typ,
                       "speakers": [sp(x) for x in e.get("speakers") or []] if isinstance(e.get("speakers"), list) else [],
                       "note": str(e["note"]) if e.get("note") is not None else None})
    moments = []
    for i, c in enumerate(o.get("coach_moments") or []):
        if not isinstance(c, dict):
            continue
        t = _num(c.get("t"))
        if t is None or c.get("for_speaker") is None:
            fixes.append(f"coach_moments[{i}] dropped (no t / for_speaker)")
            continue
        moments.append({"t": round(max(0.0, t), 2), "for_speaker": sp(c.get("for_speaker")),
                        "what_happened": str(c["what_happened"]) if c.get("what_happened") is not None else None,
                        "ideal_nudge": str(c["ideal_nudge"]) if c.get("ideal_nudge") is not None else None,
                        "kind": _enum(c.get("kind"), ("warning", "encouragement"), f"coach_moments[{i}].kind", fixes),
                        "priority": _int_clamp(c.get("priority"), 1, None, f"coach_moments[{i}].priority", fixes)})
    summ = o.get("summary") if isinstance(o.get("summary"), dict) else {}
    audio = o.get("audio") if isinstance(o.get("audio"), dict) else {}
    ann = o.get("annotator") if isinstance(o.get("annotator"), dict) else {}
    return {
        "format": "mindshift-annotation/v1",
        "annotator": {"model": model, "notes": str(ann["notes"]) if ann.get("notes") is not None else None},
        "audio": {"duration_s": round(duration_s, 2) if duration_s is not None else _num(audio.get("duration_s")),
                  "quality": _enum(audio.get("quality"), QUALITIES, "audio.quality", fixes),
                  "environment": str(audio["environment"]) if audio.get("environment") is not None else None},
        "speakers": speakers,
        "segments": segs,
        "events": events,
        "coach_moments": moments,
        "summary": {"tone_arc": str(summ["tone_arc"]) if summ.get("tone_arc") is not None else None,
                    "overall_heat": _int_clamp(summ.get("overall_heat"), 0, 3, "summary.overall_heat", fixes),
                    "peak_heat_t": _num(summ.get("peak_heat_t"))},
    }
