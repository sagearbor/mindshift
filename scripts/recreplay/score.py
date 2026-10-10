"""Score one run bundle (recreplay.pipeline): what the coach said and when,
against the owner's moments, the annotation's coach_moments and the
identity truth.

All times are on the AUDIO clock (seconds from the start of the recording).
A server event's ``at_s`` is measured from the first PCM frame on the wall
clock, so with ``speed`` 1 it IS the audio clock (real time), +-100 ms
frame granularity.

* **lines** — every coaching output: server suggestions (``response`` = what
  to say to someone else, ``nudge`` = about the wearer's own turn), room
  cards/answers, and the phone's alert haptics. ``latency_ms`` = arrival
  minus the END of the turn it answers (speech stopped -> coach heard);
  ``from_sent_ms`` = arrival minus when the phone sent the turn;
  ``first_words_ms`` = the first partial preview (or the final).
* **fires** — lines the wearer would actually notice: a server suggestion
  with ``speak`` true (its importance cleared the interject threshold) or a
  phone alert haptic (level >= 1).
* **moments** — the owner's ``mm:ss`` lines and the primary annotation's
  coach_moments FOR THE WEARER; a moment is hit when a fire lands within
  ``[anchor - pre_s, anchor + window]``. A rule-derived corpus moment is
  anchored at the END of the turn it is about (MOMENT_ANCHOR; the coach
  cannot answer before the turn closes); others at their own ``t``. The
  headline is the chance-adjusted ``lift`` = hit rate minus the hit rate
  of the same fire train at random circular offsets (chance_hits).
* **turn_coverage** — the phone's turns vs ground-truth segment TIMES
  (VAD coverage / precision, segment recall, how late turns close),
  independent of any text.
* **identity** — the phone's per-turn ``is_self`` and the server's
  ``speaker_identity`` verdicts vs who was actually speaking (annotation
  segments, else Deepgram + the resolved owner label).
* **violations** — heuristics, flagged as "possible": a proper noun or
  number the conversation never contained (invented fact), a first-person
  claim not grounded in anything said, and lines that script what someone
  else should say (words in mouth).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

MOMENT_WINDOW_S = 6.0          # a fire counts up to this long AFTER the moment's anchor
MOMENT_PRE_S = 1.5             # ... and from this long BEFORE it
# Where a rule-derived moment's window is anchored. The corpus rules
# (corpora.derive_moments) stamp talks-over / raised-voice at the wearer
# turn's START, but the coach can only answer once the phone closes the
# turn: fires land ~2 s after the turn's END, so a start-anchored 6 s window
# misses every reply to a turn longer than ~4 s. "turn_end" (default) moves
# the anchor to the end of the turn the rule is about (ground-truth segment
# times; Deepgram turns when there is no segment truth, e.g. CONFER);
# "turn_start" reproduces the legacy scoring. The owner's own mm:ss moments
# and free-text annotator moments are never moved (a human / the annotator
# chose that instant).
MOMENT_ANCHOR = "turn_end"
ANCHOR_MAX_SHIFT_S = 15.0      # never move an anchor further than this past the rule's own t
RULE_BY_NUDGE = {
    "Let them finish first.": "talks-over",
    "Lower your voice, slow down.": "raised-voice",
    "Pause, they're trying to speak.": "monologue-cuts-off",
    "It's heating up: slow down, lower voice.": "conflict-peak",
}
CHANCE_SHIFTS = 200            # circular shifts of the fire train used for the chance hit rate


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


# ---------------------------------------------------------------------------
# Truth: who was speaking when
# ---------------------------------------------------------------------------

def truth_segments(bundle: dict) -> tuple[list[dict], str]:
    """[{start, end, wearer: bool, label}] and the source ("annotation:<label>"
    | "deepgram")."""
    ident = bundle.get("identity") or {}
    primary = bundle.get("primary_annotation")
    for a in bundle.get("annotations") or []:
        if a.get("label") == primary and a.get("aligned"):
            wid = ident.get("wearer_ann_id")
            segs = [{"start": float(s["start"]), "end": float(s["end"]), "label": s["speaker"],
                     "wearer": s["speaker"] == wid, "intensity": s.get("vocal_intensity"),
                     "emotion": s.get("vocal_emotion"), "text": s.get("text", ""),
                     "is_backchannel": bool(s.get("is_backchannel"))}
                    for s in a["aligned"].get("segments", [])]
            if segs:
                return segs, f"annotation:{primary}"
    wl = ident.get("wearer_label")
    turns = (bundle.get("stt") or {}).get("turns") or []
    return ([{"start": float(t["start_time"]), "end": float(t["end_time"]), "label": t["speaker"],
              "wearer": t["speaker"] == wl, "intensity": None, "emotion": None, "text": t.get("text", ""),
              "is_backchannel": False}
             for t in turns], "deepgram")


def wearer_truth(segs: list[dict], start: float, end: float) -> bool | None:
    w = o = 0.0
    for s in segs:
        ov = min(end, s["end"]) - max(start, s["start"])
        if ov > 0:
            if s["wearer"]:
                w += ov
            else:
                o += ov
    if w == 0 and o == 0:
        return None
    return w > o


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------

def _sent_turns(bundle: dict) -> list[dict]:
    """The phone's turns with their send time (audio clock)."""
    phone = bundle.get("phone") or {}
    out = []
    for e in phone.get("sent") or []:
        out.append({**e, "sent_at": float(e.get("sent_at_audio_s", e.get("end_time", 0.0)))})
    return out


def _match_turn(turns: list[dict], text: str | None, at: float) -> dict | None:
    cands = [t for t in turns if t.get("text") == text and t["sent_at"] <= at + 0.05]
    return max(cands, key=lambda t: t["sent_at"]) if cands else None


def coach_lines(bundle: dict) -> tuple[list[dict], list[dict]]:
    speed = float((bundle.get("settings") or {}).get("speed") or 1.0)
    turns = _sent_turns(bundle)
    lines: list[dict] = []
    errors: list[dict] = []
    server = bundle.get("server") or {}
    events = server.get("events") or []
    first_partial: dict[str, float] = {}
    for e in events:
        ev, at = e["event"], float(e["at_s"]) * speed
        kind = ev.get("type")
        if kind == "suggestion" and ev.get("partial"):
            first_partial.setdefault(ev.get("utterance_text", ""), at)
            continue
        if kind == "suggestion":
            turn = _match_turn(turns, ev.get("utterance_text"), at)
            end = float(turn["end_time"]) if turn else None
            fp = first_partial.get(ev.get("utterance_text", ""))
            lines.append({
                "source": "server", "kind": ev.get("kind") or "response", "at_s": round(at, 3),
                "text": (ev.get("suggestions") or [""])[0], "all": ev.get("suggestions") or [],
                "speak": bool(ev.get("speak", True)), "importance": ev.get("importance"),
                "fires": bool(ev.get("speak", True)), "utterance_text": ev.get("utterance_text"),
                "turn_speaker": ev.get("speaker"), "turn_uid": turn.get("turn_uid") if turn else None,
                "turn_end_s": end,
                "latency_ms": None if end is None else round((at - end) / speed * 1000.0, 1),
                "from_sent_ms": None if turn is None else round((at - turn["sent_at"]) / speed * 1000.0, 1),
                "first_words_ms": None if end is None else round(((fp if fp is not None else at) - end) / speed * 1000.0, 1),
            })
        elif kind in ("room_card", "room_answer"):
            text = ev.get("fact") if kind == "room_card" else ev.get("text")
            lines.append({"source": "server", "kind": kind, "at_s": round(at, 3), "text": text or "",
                          "all": [text or ""], "speak": kind == "room_answer", "fires": True,
                          "utterance_text": ev.get("question"), "turn_speaker": None, "turn_uid": None,
                          "turn_end_s": ev.get("t"), "latency_ms": None if ev.get("t") is None else round((at - float(ev["t"])) * 1000.0, 1),
                          "from_sent_ms": None, "first_words_ms": None, "importance": None})
        elif kind in ("suggestion_error", "room_answer_error"):
            errors.append({"at_s": round(at, 3), "reason": ev.get("reason"), "utterance_text": ev.get("utterance_text") or ev.get("question")})
    phone = bundle.get("phone") or {}
    for h in phone.get("haptics") or []:
        at = float(h.get("atSec", 0.0))
        positive = (h.get("code") or "") in ("D", "E", "R", "K")
        prior = [t for t in turns if float(t["end_time"]) <= at + 0.01]
        turn = max(prior, key=lambda t: float(t["end_time"])) if prior else None
        end = float(turn["end_time"]) if turn else None
        lines.append({
            "source": "phone", "kind": "positive" if positive else "haptic", "at_s": round(at, 3),
            "text": f"buzz L{h.get('level')}" + (f" ({h['code']})" if h.get("code") else ""), "all": [],
            "speak": False, "fires": (not positive) and int(h.get("level") or 0) >= 1, "importance": None,
            "utterance_text": turn.get("text") if turn else None, "turn_speaker": turn.get("speaker") if turn else None,
            "turn_uid": turn.get("turn_uid") if turn else None, "turn_end_s": end,
            "latency_ms": None if end is None else round((at - end) * 1000.0, 1),
            "from_sent_ms": None, "first_words_ms": None,
        })
    lines.sort(key=lambda ln: ln["at_s"])
    return lines, errors


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def moment_rule(it: dict) -> str | None:
    """The corpus rule a moment came from (by its fixed nudge text), else None."""
    if it.get("rule"):
        return it["rule"]
    return RULE_BY_NUDGE.get(it.get("text") or it.get("ideal_nudge") or "")


def anchor_moment(it: dict, segs: list[dict], basis: str = "segment",
                  anchor: str = MOMENT_ANCHOR, max_shift_s: float = ANCHOR_MAX_SHIFT_S) -> tuple[float, str]:
    """(anchor time, basis) for one moment item. ``segs`` = truth_segments()
    (``basis`` "segment" for annotation truth, "deepgram-turn" when they are
    Deepgram's turns). See MOMENT_ANCHOR."""
    t = float(it["t"])
    rule = moment_rule(it)
    if anchor != "turn_end" or it.get("source") == "owner" or rule is None:
        return t, "as-marked"
    real = [s for s in segs if not s.get("is_backchannel")]
    if not real:
        return t, "no-turns"
    wearer_only = rule in ("talks-over", "raised-voice", "monologue-cuts-off") and any(s.get("wearer") for s in real)
    pool = [s for s in real if s.get("wearer")] if wearer_only else real
    cand = None
    if rule in ("talks-over", "raised-voice"):
        exact = [s for s in pool if abs(s["start"] - t) <= 0.05]
        if exact:
            cand = max(exact, key=lambda s: s["end"])
    if cand is None:
        inside = [s for s in pool if s["start"] - 0.05 <= t < s["end"]]
        if inside:
            cand = min(inside, key=lambda s: s["end"])         # the first moment the floor can change
        else:
            nxt = [s for s in pool if t <= s["start"] <= t + 3.0]
            cand = min(nxt, key=lambda s: s["start"]) if nxt else None
    if cand is None:
        return t, "no-turn-at-t"
    end = float(cand["end"])
    if end - t > max_shift_s:
        return t + max_shift_s, f"{basis}-end-capped"
    return max(end, t), f"{basis}-end"


def chance_hits(anchors: list[float], fire_times: list[float], dur: float, pre_s: float, post_s: float,
                shifts: int = CHANCE_SHIFTS) -> float | None:
    """Expected number of moments a fire train with the SAME count and the
    same internal spacing would hit at a random offset: the mean over
    ``shifts`` evenly spaced circular shifts of the fire times (mod dur).
    Keeping the train's clustering makes this harder to beat than a Poisson
    stream at the same rate."""
    if not anchors:
        return 0.0
    if not dur or dur <= 0:
        return None
    if not fire_times:
        return 0.0
    f = np.asarray(fire_times, dtype=float)
    a = np.asarray(anchors, dtype=float)
    total = 0
    for k in range(shifts):
        g = np.sort((f + dur * k / shifts) % dur)
        lo = np.searchsorted(g, a - pre_s, side="left")
        hi = np.searchsorted(g, a + post_s, side="right")
        total += int(np.count_nonzero(hi > lo))
    return total / shifts


def moments(bundle: dict, lines: list[dict], window_s: float, *, pre_s: float = MOMENT_PRE_S,
            anchor: str = MOMENT_ANCHOR, segs: list[dict] | None = None) -> dict:
    ident = bundle.get("identity") or {}
    wid = ident.get("wearer_ann_id")
    if segs is None:
        segs, src = truth_segments(bundle)
    else:
        src = "given"
    items: list[dict] = []
    for m in (bundle.get("notes") or {}).get("moments") or []:
        items.append({"source": "owner", "t": float(m["t"]), "text": m.get("text", ""), "kind": None})
    others: list[dict] = []
    primary = bundle.get("primary_annotation")
    for a in bundle.get("annotations") or []:
        if a.get("label") != primary or not a.get("aligned"):
            continue
        for cm in a["aligned"].get("coach_moments", []):
            row = {"source": "annotation", "t": float(cm["t"]), "text": cm.get("ideal_nudge") or cm.get("what_happened") or "",
                   "kind": cm.get("kind"), "priority": cm.get("priority"), "for_speaker": cm.get("for_speaker"),
                   "what_happened": cm.get("what_happened")}
            if wid is None or cm.get("for_speaker") in (None, wid):
                items.append(row)
            else:
                others.append(row)
    fires = [ln for ln in lines if ln["fires"]]
    seg_basis = "deepgram-turn" if src == "deepgram" else "segment"
    for it in items:
        it["anchor_t"], it["anchor_basis"] = anchor_moment(it, segs, seg_basis, anchor)
        it["anchor_t"] = round(it["anchor_t"], 3)
        it["rule"] = moment_rule(it)
        a = it["anchor_t"]
        near = [ln for ln in fires if a - pre_s <= ln["at_s"] <= a + window_s]
        it["hit"] = bool(near)
        best = min(near, key=lambda ln: abs(ln["at_s"] - a)) if near else (
            min(fires, key=lambda ln: abs(ln["at_s"] - a)) if fires else None)
        it["nearest"] = None if best is None else {"at_s": best["at_s"], "kind": best["kind"], "text": best["text"],
                                                   "delta_s": round(best["at_s"] - a, 2),
                                                   "delta_from_t_s": round(best["at_s"] - it["t"], 2)}
    items.sort(key=lambda x: x["t"])
    unmatched = [ln for ln in fires
                 if not any(it["anchor_t"] - pre_s <= ln["at_s"] <= it["anchor_t"] + window_s for it in items)]
    dur = float((bundle.get("audio") or {}).get("duration_s") or 0.0)
    hits = sum(1 for it in items if it["hit"])
    ch = chance_hits([it["anchor_t"] for it in items], [ln["at_s"] for ln in fires], dur, pre_s, window_s)
    return {
        "window_s": window_s, "pre_s": pre_s, "anchor": anchor, "items": items, "for_others": others,
        "hits": hits, "total": len(items),
        "owner_hits": sum(1 for it in items if it["hit"] and it["source"] == "owner"),
        "owner_total": sum(1 for it in items if it["source"] == "owner"),
        "unmatched_lines": [{"at_s": ln["at_s"], "kind": ln["kind"], "text": ln["text"], "source": ln["source"]} for ln in unmatched],
        "fires": len(fires),
        # chance-adjusted: the same fire train circularly shifted (see chance_hits)
        "chance_hits": ch,
        "hit_rate": (hits / len(items)) if items else None,
        "chance_hit_rate": (ch / len(items)) if items and ch is not None else None,
        "lift": ((hits - ch) / len(items)) if items and ch is not None else None,
    }


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def identity(bundle: dict, lines: list[dict], segs: list[dict]) -> dict:
    turns = _sent_turns(bundle)
    rows = []
    timeline = []
    for t in turns:
        truth = wearer_truth(segs, float(t["start_time"]), float(t["end_time"]))
        pred = t.get("is_self")
        ok = None if pred is None or truth is None else (pred == truth)
        rows.append({"uid": t.get("turn_uid"), "speaker": t.get("speaker"), "start": t["start_time"], "end": t["end_time"],
                     "sent_at": t["sent_at"], "pred": pred, "truth": truth, "ok": ok,
                     "score": t.get("speaker_match_score"), "basis": t.get("speaker_match_basis")})
        if pred is not None:
            timeline.append({"t": t["sent_at"], "who": "phone", "label": t.get("speaker"), "is_self": pred, "ok": ok,
                             "score": t.get("speaker_match_score")})
    decided = [r for r in rows if r["pred"] is not None and r["truth"] is not None]
    wearer_rows = [r for r in rows if r["truth"] is True]
    first = next((r["sent_at"] for r in sorted(rows, key=lambda r: r["sent_at"]) if r["pred"] is True and r["truth"] is True), None)
    false_self = [r for r in rows if r["pred"] is True and r["truth"] is False]
    phone = {
        "turns": len(rows), "decided": len(decided), "correct": sum(1 for r in decided if r["ok"]),
        "accuracy": (sum(1 for r in decided if r["ok"]) / len(decided)) if decided else None,
        "coverage": (len(decided) / len(rows)) if rows else None,
        "wearer_turns": len(wearer_rows),
        "wearer_recall": (sum(1 for r in wearer_rows if r["pred"] is True) / len(wearer_rows)) if wearer_rows else None,
        "false_self": len(false_self), "first_confirmed_s": first, "rows": rows,
    }
    # server voiceprint verdicts per phone label
    speed = float((bundle.get("settings") or {}).get("speed") or 1.0)
    srv_rows = []
    for e in (bundle.get("server") or {}).get("events") or []:
        ev = e["event"]
        if ev.get("type") != "speaker_identity":
            continue
        at = float(e["at_s"]) * speed
        label_turns = [r for r in rows if r["speaker"] == ev.get("speaker") and r["sent_at"] <= at + 0.05]
        truths = [r["truth"] for r in label_turns if r["truth"] is not None]
        truth = (sum(truths) > len(truths) / 2) if truths else None
        ok = None if truth is None else (bool(ev.get("is_self")) == truth)
        srv_rows.append({"t": round(at, 3), "label": ev.get("speaker"), "is_self": bool(ev.get("is_self")),
                         "person_id": ev.get("person_id"), "score": ev.get("score"), "truth": truth, "ok": ok})
        timeline.append({"t": round(at, 3), "who": "server", "label": ev.get("speaker"), "is_self": bool(ev.get("is_self")),
                         "ok": ok, "score": ev.get("score")})
    srv_first = next((r["t"] for r in srv_rows if r["is_self"] and r["ok"]), None)
    server = {"events": len(srv_rows), "correct": sum(1 for r in srv_rows if r["ok"]),
              "decided": sum(1 for r in srv_rows if r["ok"] is not None),
              "first_confirmed_s": srv_first, "rows": srv_rows}
    # coaching direction: nudge = treated as the wearer's turn, response = someone else's
    direction = []
    seen = set()
    for ln in lines:
        if ln["source"] != "server" or ln["kind"] not in ("nudge", "response") or not ln.get("turn_uid"):
            continue
        if ln["turn_uid"] in seen:
            continue
        seen.add(ln["turn_uid"])
        row = next((r for r in rows if r["uid"] == ln["turn_uid"]), None)
        if row is None or row["truth"] is None:
            continue
        direction.append({"uid": ln["turn_uid"], "kind": ln["kind"], "truth_wearer": row["truth"],
                          "ok": (ln["kind"] == "nudge") == row["truth"]})
    timeline.sort(key=lambda x: x["t"])
    return {"phone": phone, "server": server, "timeline": timeline,
            "coaching_direction": {"correct": sum(1 for d in direction if d["ok"]), "total": len(direction), "rows": direction}}


# ---------------------------------------------------------------------------
# Violations (heuristics)
# ---------------------------------------------------------------------------

_STOP = set("""a an the and or but if so to of in on at for with from by about as is are was were be been being am
i i'm i've i'd i'll me my mine you your yours you're he him his she her they them their we us our it its this that
these those there here what why how when where who which do does did done not no yes just now then than too very
can could would should will shall may might must have has had having get got let lets let's ok okay oh well really
more most less some any all each every one two up down out over again back still even also maybe""".split())
_EMPATHY = set("""hear heard understand understood see feel felt sorry want wonder love care appreciate think know
imagine sounds sound seem seems get notice noticed glad happy sad hard tough mean meant guess hope wish""".split())
_PAST_FACT = re.compile(
    r"\bI(?:'ve| have| had)?\s+(?:\w+\s+){0,2}?(was|did|went|told|said|made|bought|got|saw|called|finished|fixed|took|"
    r"left|lost|spent|met|worked|paid|drove|cooked|cleaned|forgot|missed|booked|ordered|sent|picked)\b", re.I)
_WORDS_IN_MOUTH = [
    re.compile(r"\b(?:tell|ask|make|have|get)\s+(?:him|her|them|your\s+\w+)\s+(?:to\s+)?(?:say|apologi[sz]e)\b", re.I),
    re.compile(r"\b(?:he|she|they|your\s+(?:son|daughter|wife|husband|partner|kid|child|mom|dad))\s+(?:should|needs?\s+to|has\s+to|must|ought\s+to)\s+(?:say|apologi[sz]e|admit)\b", re.I),
    re.compile(r"\btell\s+(?:him|her|them)\s+(?:he|she|they)\s+(?:should|needs?\s+to|has\s+to|must)\b", re.I),
]


def _vocab(bundle: dict, segs: list[dict]) -> set[str]:
    texts = [s.get("text", "") for s in segs]
    texts += [w.get("word", "") for w in (bundle.get("stt") or {}).get("words") or []]
    texts += [t.get("text", "") for t in (bundle.get("stt") or {}).get("turns") or []]
    texts.append((bundle.get("settings") or {}).get("session_context") or "")
    texts.append((bundle.get("notes") or {}).get("setting") or "")
    toks = set()
    for t in texts:
        toks |= {w.lower().strip("'") for w in re.findall(r"[A-Za-z0-9']+", t or "")}
    return toks


def violations(bundle: dict, lines: list[dict], segs: list[dict]) -> list[dict]:
    vocab = _vocab(bundle, segs)
    out = []
    for ln in lines:
        if ln["source"] != "server" or ln["kind"] not in ("response", "nudge", "room_answer"):
            continue
        for idx, text in enumerate(ln.get("all") or [ln["text"]]):
            if not text:
                continue
            for sent in re.split(r"(?<=[.!?])\s+", text):
                toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’]*", sent)
                novel_names = []
                for i, tok in enumerate(toks):
                    low = tok.lower().replace("’", "'")
                    base = re.sub(r"'s$", "", low)
                    if tok[0].isdigit() and low not in vocab:
                        novel_names.append(tok)
                    elif i > 0 and tok[0].isupper() and base not in _STOP and base not in vocab and low not in ("i", "i'm", "i've", "i'd", "i'll"):
                        novel_names.append(tok)
                if novel_names:
                    out.append({"kind": "invented-fact", "at_s": ln["at_s"], "line_kind": ln["kind"], "index": idx,
                                "text": text, "evidence": f"not said by anyone: {', '.join(novel_names)}"})
                m = _PAST_FACT.search(sent)
                if m:
                    content = [w.lower() for w in toks if len(w) > 3 and w.lower() not in _STOP and w.lower() not in _EMPATHY]
                    ungrounded = [w for w in content if w not in vocab and w != m.group(1).lower()]
                    if ungrounded:
                        out.append({"kind": "ungrounded-first-person", "at_s": ln["at_s"], "line_kind": ln["kind"],
                                    "index": idx, "text": text,
                                    "evidence": f"first-person claim with words nobody said: {', '.join(ungrounded[:5])}"})
            for rx in _WORDS_IN_MOUTH:
                m = rx.search(text)
                if m:
                    out.append({"kind": "words-in-mouth", "at_s": ln["at_s"], "line_kind": ln["kind"], "index": idx,
                                "text": text, "evidence": f"scripts someone else: {m.group(0)!r}"})
                    break
    return out


# ---------------------------------------------------------------------------
# Heat lane (annotation intensity)
# ---------------------------------------------------------------------------

def turn_coverage(bundle: dict, segs: list[dict], truth_source: str, *, grid_s: float = 0.05,
                  min_seg_s: float = 0.5) -> dict:
    """How well the phone's turn segmentation (VAD + segmenter) covers the
    ground-truth speech, from segment TIMES only (never the words):

    * ``vad_coverage`` — share of ground-truth speech time (union of
      non-backchannel segments) inside some phone turn's [start, end];
    * ``vad_precision`` — share of phone-turn time that is ground-truth speech;
    * ``segment_recall`` — share of ground-truth segments >= ``min_seg_s``
      at least half covered by phone turns (``wearer_segment_recall``: the
      wearer's segments only);
    * ``end_offset_p50_s`` — median (phone turn end - ground-truth segment
      end) over ground-truth segments, each matched to
      the phone turn holding most of the segment — how late the phone
      closes a turn (negative: it cut the segment short).

    None when the truth is Deepgram's own turns (no independent timing)."""
    if not truth_source.startswith("annotation"):
        return {"basis": "none (no ground-truth segment times)", "vad_coverage": None, "vad_precision": None,
                "segment_recall": None, "wearer_segment_recall": None, "end_offset_p50_s": None, "segments": 0}
    real = [s for s in segs if not s.get("is_backchannel") and s["end"] > s["start"]]
    turns = [(float(t["start_time"]), float(t["end_time"])) for t in (bundle.get("phone") or {}).get("sent") or []]
    dur = max([float((bundle.get("audio") or {}).get("duration_s") or 0.0)] + [s["end"] for s in real] + [b for _, b in turns])
    n = int(dur / grid_s) + 2
    g = np.zeros(n, dtype=bool)
    p = np.zeros(n, dtype=bool)
    for s in real:
        g[int(s["start"] / grid_s):int(s["end"] / grid_s)] = True
    for a, b in turns:
        p[int(a / grid_s):int(b / grid_s)] = True
    long = [s for s in real if s["end"] - s["start"] >= min_seg_s]

    def covered(s):
        a, b = int(s["start"] / grid_s), int(s["end"] / grid_s)
        return b > a and p[a:b].mean() >= 0.5

    offs = []
    for s in long:
        ov = [(min(b, s["end"]) - max(a, s["start"]), b) for a, b in turns]
        ov = [x for x in ov if x[0] > 0]
        if ov:
            offs.append(max(ov)[1] - s["end"])       # the phone turn holding most of this segment
    wl = [s for s in long if s.get("wearer")]
    return {
        "basis": truth_source, "segments": len(long), "phone_turns": len(turns),
        "vad_coverage": float((g & p).sum() / g.sum()) if g.any() else None,
        "vad_precision": float((g & p).sum() / p.sum()) if p.any() else None,
        "segment_recall": (sum(1 for s in long if covered(s)) / len(long)) if long else None,
        "wearer_segment_recall": (sum(1 for s in wl if covered(s)) / len(wl)) if wl else None,
        "end_offset_p50_s": percentile(offs, 50),
    }


def heat(segs: list[dict]) -> list[dict]:
    return [{"start": s["start"], "end": s["end"], "intensity": s["intensity"], "emotion": s["emotion"], "label": s["label"]}
            for s in segs if s.get("intensity") is not None or s.get("emotion")]


def score(bundle: dict, moment_window_s: float = MOMENT_WINDOW_S, *, moment_pre_s: float = MOMENT_PRE_S,
          moment_anchor: str = MOMENT_ANCHOR) -> dict[str, Any]:
    segs, truth_source = truth_segments(bundle)
    lines, errors = coach_lines(bundle)
    srv_final = [ln for ln in lines if ln["source"] == "server" and ln["kind"] in ("response", "nudge") and ln["latency_ms"] is not None]
    lat = [ln["latency_ms"] for ln in srv_final]
    first = [ln["first_words_ms"] for ln in srv_final if ln["first_words_ms"] is not None]
    sent_lat = [ln["from_sent_ms"] for ln in srv_final if ln["from_sent_ms"] is not None]
    phone_lat = [ln["latency_ms"] for ln in lines if ln["source"] == "phone" and ln["latency_ms"] is not None]
    llm = (bundle.get("server") or {}).get("llm") or {}
    return {
        "truth_source": truth_source,
        "lines": lines,
        "errors": errors,
        "latency": {
            "n": len(lat), "p50_ms": percentile(lat, 50), "p90_ms": percentile(lat, 90), "max_ms": max(lat) if lat else None,
            "first_words_p50_ms": percentile(first, 50), "first_words_p90_ms": percentile(first, 90),
            "from_sent_p50_ms": percentile(sent_lat, 50), "from_sent_p90_ms": percentile(sent_lat, 90),
            "phone_haptic_p50_ms": percentile(phone_lat, 50),
            "llm_cache": {k: llm.get(k) for k in ("hits", "misses", "offline_misses", "model") if k in llm},
            "server_stage_summary": ((bundle.get("server") or {}).get("session_complete") or {}).get("latency_summary"),
        },
        "moments": moments(bundle, lines, moment_window_s, pre_s=moment_pre_s, anchor=moment_anchor),
        "turn_coverage": turn_coverage(bundle, segs, truth_source),
        "identity": identity(bundle, lines, segs),
        "violations": violations(bundle, lines, segs),
        "heat": heat(segs),
    }
