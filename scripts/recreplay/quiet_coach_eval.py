"""Quiet coach before/after over the replay corpus ($0: files only).

    # 1. phone half with the production flags ($0, local):
    MINDSHIFT_LIVE_DEFAULTS=1 tsx apps/mobile/src/live/replay/recordingReplay.ts \
        --wav <item>/audio16k.wav --meta <item>/phone_meta.json --out <ph_after>/<item>.json ...
    # 2. this report:
    tmp/venv/bin/python scripts/recreplay/quiet_coach_eval.py <ph_after_dir> <out.json> [--rec tmp/recordings]

before (H0) = tmp/recordings/work/<item>/run.json   (the coach as shipped, 10 Oct)
before (H1) = tmp/recordings/ot/X/X-H1/<item>/run.json (#195's speak gate; same phone turns)
after       = the H1 phone turns + the quiet-coach phone replay (ph_after/<item>.json,
              MINDSHIFT_LIVE_DEFAULTS=1) with the server's trigger gate applied post hoc
              (server/coach_trigger.py + coach_gate.skip_reason, the same functions the
              live pipeline runs), then the speak gate re-run in time order.
"""
import json
import os
import re
import statistics
import sys
from collections import Counter

from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
REC = (sys.argv[sys.argv.index("--rec") + 1] if "--rec" in sys.argv
       else os.environ.get("MINDSHIFT_RECORDINGS") or f"{WT}/tmp/recordings")
sys.path.insert(0, f"{WT}/server")
import coach_gate  # noqa: E402
import coach_trigger  # noqa: E402

ph_dir, out_path = [a for a in sys.argv[1:] if a != "--rec" and a != REC][:2]
splits = json.load(open(f"{REC}/landscape/splits.json"))
SPEAK_MIN, UNKNOWN_CAP, GAP_S, MIN_WORDS = 80, 72, 60.0, 3
FLOOR_TEXT = "You've had the floor a while. Check in with them."


def group_of(it):
    try:
        return json.load(open(f"{REC}/inbox/{it}/{it}.corpus.json")).get("group") or "heated"
    except Exception:
        return "heated"


def llm_calls(bundle):
    llm = (bundle.get("server") or {}).get("llm") or {}
    return int(llm.get("hits") or 0) + int(llm.get("misses") or 0) + int(llm.get("offline_misses") or 0)


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
        row = {"uid": ev.get("turn_uid"), "end": end, "text": text, "ident": ident, "skip": None, "triggers": ()}
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
        utts.append((start, end, text, label))
        rows.append(row)
    return rows


def regate(lines, rows_by_uid, triggered_only):
    spoken, last = [], None
    for ln in sorted(lines, key=lambda ln: ln["at_s"]):
        if ln.get("source") != "server" or ln.get("kind") not in ("response", "nudge"):
            continue
        r = rows_by_uid.get(ln.get("turn_uid"))
        if r is None:
            continue
        if triggered_only and not r["triggers"]:
            continue
        imp = int(ln.get("importance") or 0)
        if r["ident"] == "unknown":
            imp = min(imp, UNKNOWN_CAP)
        words = len(re.findall(r"\w+", ln.get("utterance_text") or ""))
        if imp < SPEAK_MIN or words < MIN_WORDS:
            continue
        t = float(ln.get("turn_end_s") or ln["at_s"])
        if last is not None and t - last < GAP_S:
            continue
        last = t
        spoken.append({"t": round(t, 1), "text": ln["text"], "turn": (ln.get("utterance_text") or "")[:120],
                       "importance": ln.get("importance"), "triggers": list(r["triggers"])})
    return spoken


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


items = []
for split in ("held_out", "dev"):
    for it in splits[split]:
        h1p = f"{REC}/ot/X/X-H1/{it}/run.json"
        h0p = f"{REC}/work/{it}/run.json"
        ap = f"{ph_dir}/{it}.json"
        if not os.path.exists(ap) or not os.path.exists(h0p):
            continue
        base = json.load(open(h1p)) if (split == "held_out" and os.path.exists(h1p)) else json.load(open(h0p))
        h0 = json.load(open(h0p))
        after = json.load(open(ap))
        dur = float(base["audio"]["duration_s"])
        sent = base["phone"]["sent"]
        a_sent = after["sent"]
        a_by_uid = {s.get("turn_uid"): s for s in a_sent}
        same_turns = [s.get("turn_uid") for s in sent] == [s.get("turn_uid") for s in a_sent] and \
            [s.get("text") for s in sent] == [s.get("text") for s in a_sent]
        rows = simulate(sent, a_by_uid)
        rows_by_uid = {r["uid"]: r for r in rows}
        lines = base["score"]["lines"]
        coached = [r for r in rows if r["text"].strip() and r["skip"] is None]
        trig = [r for r in coached if r["triggers"]]
        # spoken server lines
        h0_spoken = sum(1 for ln in h0["score"]["lines"] if ln.get("source") == "server" and ln.get("fires"))
        h1_spoken = sum(1 for ln in base["score"]["lines"] if ln.get("source") == "server" and ln.get("fires")) if split == "held_out" else None
        sim_h1 = regate(lines, rows_by_uid, triggered_only=False)
        sim_after = regate(lines, rows_by_uid, triggered_only=True)
        # phone side
        b_local = len(base["phone"].get("spoken") or [])
        a_spoken = after.get("spoken") or []
        a_floor_lines = sum(1 for s in a_spoken if s.get("text") == FLOOR_TEXT)
        a_local = len(a_spoken) - a_floor_lines
        heat = [h for h in after.get("haptics") or [] if h.get("code") == "H"]
        floor = after.get("floorCues") or []
        # latency: speech end -> turn_local sent
        lat_b = [s["sent_at_audio_s"] - s["end_time"] for s in sent if (s.get("text") or "").strip()]
        lat_a = [s["sent_at_audio_s"] - s["end_time"] for s in a_sent if (s.get("text") or "").strip()]
        from_sent = [ln["from_sent_ms"] for ln in lines if ln.get("source") == "server" and ln.get("from_sent_ms") is not None
                     and not ln.get("partial")]
        # A cloud line is only voiced when the phone did not answer the turn
        # itself (useAudioStream cloudAnswersOpenMoment + phoneAnsweredTurn):
        # with an on-device LLM, a heat/question turn got the local line and
        # a floor turn got the fixed cue line.
        def phone_answered(uid):
            a = a_by_uid.get(uid) or {}
            return bool(set(a.get("triggers") or []) & {"heat", "question", "floor"})
        uid_of = {round(float(ln.get("turn_end_s") or ln["at_s"]), 1): ln.get("turn_uid") for ln in lines}
        cloud_voiced_with_local = [x for x in sim_after if not phone_answered(uid_of.get(x["t"]))]
        floor_t = {round(f["t"]) for f in (after.get("floorCues") or [])}
        other_haptics_before = sum(1 for h in base["phone"].get("haptics") or [] if int(h.get("level") or 0) >= 1)
        other_haptics_after = sum(1 for h in after.get("haptics") or [] if int(h.get("level") or 0) >= 1
                                  and h.get("code") != "H" and not (h.get("code") == "A" and round(h["atSec"] - 0.14) in floor_t))
        heat_lat, heat_rows = [], []
        for h in heat:
            t = h["atSec"]
            turn = max((tt for tt in after["turns"] if tt["endTime"] <= t + 0.01), key=lambda tt: tt["endTime"], default=None)
            if turn:
                heat_lat.append(t - turn["endTime"])
                heat_rows.append({"t": round(t, 1), "text": turn["text"][:160], "isSelf": turn.get("isSelf")})
        items.append({
            "item": it, "split": split, "group": group_of(it), "dur_s": dur, "same_turns": same_turns,
            "sent": len(sent),
            "llm_h0": llm_calls(h0), "llm_h1": llm_calls(base) if split == "held_out" else None,
            "sim_before_calls": len(coached), "sim_after_calls": len(trig),
            "trigger_counts": dict(Counter(t for r in trig for t in r["triggers"])),
            "server_spoken_h0": h0_spoken, "server_spoken_h1": h1_spoken,
            "sim_h1_spoken": len(sim_h1), "sim_after_spoken": len(sim_after), "after_lines": sim_after,
            "local_voiced_before": b_local, "local_voiced_after": a_local,
            "cloud_voiced_after_with_local": len(cloud_voiced_with_local),
            "other_haptics_before": other_haptics_before, "other_haptics_after": other_haptics_after,
            "floor_cues": [round(f["t"], 1) for f in floor], "floor_lines_spoken": a_floor_lines,
            "heat_buzzes": heat_rows,
            "lat_sent_before": lat_b, "lat_sent_after": lat_a, "from_sent_ms": from_sent, "heat_lat": heat_lat,
            "triggered_turns": [{"t": round(r["end"], 1), "triggers": list(r["triggers"]), "text": r["text"][:140]} for r in trig],
        })


def agg(rows, key, hours):
    return round(sum(r[key] or 0 for r in rows) / hours, 1) if hours else None


out = {"items": items, "groups": {}}
for split in ("held_out", "dev"):
    for grp in ("calm", "heated", "all"):
        rs = [r for r in items if r["split"] == split and (grp == "all" or r["group"] == grp)]
        if not rs:
            continue
        hours = sum(r["dur_s"] for r in rs) / 3600
        g = {"n": len(rs), "minutes": round(hours * 60, 1)}
        for k in ("llm_h0", "llm_h1", "sim_before_calls", "sim_after_calls", "server_spoken_h0", "server_spoken_h1",
                  "sim_h1_spoken", "sim_after_spoken", "local_voiced_before", "local_voiced_after", "floor_lines_spoken",
                  "cloud_voiced_after_with_local", "other_haptics_before", "other_haptics_after"):
            if all(r[k] is not None for r in rs):
                g[k + "_per_h"] = agg(rs, k, hours)
                g[k + "_total"] = sum(r[k] for r in rs)
        g["heat_buzz_per_h"] = round(sum(len(r["heat_buzzes"]) for r in rs) / hours, 1)
        g["heat_buzz_total"] = sum(len(r["heat_buzzes"]) for r in rs)
        g["floor_cues_total"] = sum(len(r["floor_cues"]) for r in rs)
        g["floor_cues_per_h"] = round(g["floor_cues_total"] / hours, 1)
        lb = [x for r in rs for x in r["lat_sent_before"]]
        la = [x for r in rs for x in r["lat_sent_after"]]
        fs = [x / 1000 for r in rs for x in r["from_sent_ms"]]
        hl = [x for r in rs for x in r["heat_lat"]]
        g["end_to_sent_p50_before"] = round(statistics.median(lb), 2) if lb else None
        g["end_to_sent_p50_after"] = round(statistics.median(la), 2) if la else None
        g["end_to_sent_p90_before"] = round(pct(lb, 90), 2) if lb else None
        g["end_to_sent_p90_after"] = round(pct(la, 90), 2) if la else None
        g["sent_to_final_p50"] = round(statistics.median(fs), 2) if fs else None
        g["heat_buzz_after_turn_end_p50"] = round(statistics.median(hl), 2) if hl else None
        g["all_same_turns"] = all(r["same_turns"] for r in rs)
        g["triggers"] = dict(sum((Counter(r["trigger_counts"]) for r in rs), Counter()))
        out["groups"][f"{split}/{grp}"] = g
json.dump(out, open(out_path, "w"), indent=1)
for k, g in out["groups"].items():
    print(k, json.dumps(g))
