"""Shared trigger simulation for the quiet-coach replay tools.

``simulate`` replays a recording's phone turns through the SAME server
functions the live pipeline runs (``coach_gate.skip_reason`` then
``coach_trigger.evaluate``) and says, per turn, who the wearer identity was
and which triggers fired. Used by quiet_coach_eval.py (the $0 before/after
report) and quiet_coach_live_eval.py (real LLM lines + judging).
"""
import os
import sys
from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, f"{WT}/server")
import coach_gate  # noqa: E402
import coach_trigger  # noqa: E402


def tone_ctx(ev):
    tt = ev.get("text_tone") or {}
    pr = ev.get("prosody") or {}
    if not any(v is not None for v in tt.values()) and not pr:
        return None
    return {**{k: v for k, v in tt.items() if v is not None}, **({"prosody": pr} if pr else {})}


def simulate(sent, after_by_uid):
    """Per sent turn (in order): skipped? identity, floor, triggers."""
    self_labels, other_labels = set(), set()
    utts = []
    rows = []
    for ev in sent:
        text = ev.get("text") or ""
        start, end = float(ev["start_time"]), float(ev["end_time"])
        label, is_self = ev["speaker"], ev.get("is_self")
        if is_self is True:
            self_labels.add(label)
            other_labels.discard(label)
            ident = "self"
        elif is_self is False:
            other_labels.add(label)
            self_labels.discard(label)
            ident = "other"
        elif label in self_labels:
            ident = "self"
        elif label in other_labels or self_labels:
            ident = "other"
        else:
            ident = "unknown"
        row = {"uid": ev.get("turn_uid"), "end": end, "text": text, "ident": ident, "skip": None, "triggers": (),
               "floor": 0.0, "evidence": ()}
        if text.strip():
            recent = [(u[0], u[1], u[2]) for u in utts[-6:]]
            row["skip"] = coach_gate.skip_reason(text, start, end, recent, tone_ctx(ev))
            if row["skip"] is None:
                floor = 0.0
                if ident == "self":
                    fs = start
                    for u in reversed(utts):
                        if not u[2].strip():
                            continue
                        if u[3] not in self_labels or u[0] > end:
                            break
                        fs = min(fs, u[0])
                    floor = end - fs
                a = after_by_uid.get(ev.get("turn_uid")) or {}
                v = coach_trigger.evaluate(text, is_wearer=ident == "self", wearer_floor_s=floor,
                                           wearer_floor_before_s=max(0.0, floor - (end - start)),
                                           phone_triggers=a.get("triggers"))
                row["triggers"] = v.triggers
                row["floor"] = floor
                row["evidence"] = v.evidence
        utts.append((start, end, text, label))
        rows.append(row)
    return rows
