"""Aggregate report over a batch of corpus replays (scripts/corpus_to_inbox.py
items, plus any open-web ``yt_*`` items): moment hit rate, false fires per
hour, identity accuracy and time-to-confirm, latency p50/p90, invented-fact
and words-in-mouth flags — per group (heated / calm control / open-web) and
per corpus, never pooled across groups — with automatic defect detectors and
the worst examples linked to each item's own report.

    tmp/venv/bin/python scripts/recording_replay.py --corpus-summary
    -> tmp/recordings/reports/corpus-summary.html (+ corpus-summary.json)
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .report import CSS

GROUPS = [("heated", "Heated"), ("calm", "Calm control"), ("open-web", "Open-web (Gemini-annotated)")]
NO_SPEAKER_TRUTH = ("CONFER",)
RULE_BY_NUDGE = {
    "Let them finish first.": "talks-over",
    "Lower your voice, slow down.": "raised-voice",
    "Pause, they're trying to speak.": "monologue-cuts-off",
    "It's heating up: slow down, lower voice.": "conflict-peak",
}
LAUGH_PRE_S, LAUGH_POST_S = 1.0, 4.0
LEAK_RE = re.compile(r"\bSpeaker [A-Z]\b")
GATES = (50, 70)
# Defects are ordered by what they cost a wearer (judged, written down here),
# not by the detectors' raw magnitudes, which are not comparable.
RANK = ("every-line", "false-fires", "no-alert-buzz", "identity", "turn-merging", "dropped-speech", "loud-happy",
        "label-leak", "fragments", "template", "missed-moments", "flags")        # interject-slider values to report besides the app default (0)


def _opening(text: str) -> str:
    """A coach line's first two words, lowercased ("pause let", "slow down")."""
    return " ".join(re.findall(r"[a-z']+", text.lower())[:2])


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _pct(v) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def _ms(v) -> str:
    return "–" if v is None else f"{v / 1000:.1f} s"


def _num(v, fmt="{:.1f}") -> str:
    return "–" if v is None else fmt.format(v)


def _corpus_of(name: str, prov: dict | None) -> str:
    if prov and prov.get("corpus"):
        return prov["corpus"]
    return {"sbcsae": "SBCSAE", "ami": "AMI", "chime6": "CHiME-6", "confer": "CONFER", "yt": "open-web"}.get(name.split("_")[0], "other")


def _gt(bundle: dict) -> dict:
    primary = bundle.get("primary_annotation")
    for a in bundle.get("annotations") or []:
        if a.get("label") == primary and a.get("aligned"):
            return a["aligned"]
    return {}


def item_metrics(bundle: dict, prov: dict | None) -> dict:
    prov = prov or {}
    name = bundle.get("name", "?")
    corpus = _corpus_of(name, prov)
    sc = bundle.get("score") or {}
    dur = float((bundle.get("audio") or {}).get("duration_s") or 0.0)
    hours = dur / 3600.0
    lines = sc.get("lines") or []
    mom = sc.get("moments") or {}
    fires = [ln for ln in lines if ln.get("fires")]
    unmatched = mom.get("unmatched_lines") or []
    gt = _gt(bundle)
    segs = [s for s in gt.get("segments") or [] if not s.get("is_backchannel")]
    laughs = [e["t"] for e in gt.get("events") or [] if e.get("type") == "laughter"]
    un_t = {round(float(u["at_s"]), 3) for u in unmatched}
    near_laugh = [ln for ln in fires if round(float(ln["at_s"]), 3) in un_t
                  and any(t - LAUGH_PRE_S <= ln["at_s"] <= t + LAUGH_POST_S for t in laughs)]
    # turn merging: one phone turn covering >= 2 ground-truth voices (>= 0.5 s each)
    merged, merged_ex = 0, []
    sent = (bundle.get("phone") or {}).get("sent") or []
    for t in sent:
        a, b = float(t["start_time"]), float(t["end_time"])
        per = Counter()
        for s in segs:
            ov = min(b, s["end"]) - max(a, s["start"])
            if ov > 0:
                per[s["speaker"]] += ov
        voices = [k for k, v in per.items() if v >= 0.5]
        if len(voices) >= 2:
            merged += 1
            if len(merged_ex) < 3:
                merged_ex.append({"t": a, "end": b, "voices": sorted(voices), "text": (t.get("text") or "")[:160]})
    # how much ground-truth speech the phone turned into turns at all (0.1 s grid)
    coverage = None
    if segs and dur:
        g = np.zeros(int(dur * 10) + 2, dtype=bool)
        p = np.zeros_like(g)
        for s in segs:
            g[int(s["start"] * 10):int(s["end"] * 10)] = True
        for t in sent:
            p[int(float(t["start_time"]) * 10):int(float(t["end_time"]) * 10)] = True
        coverage = float((g & p).sum() / max(g.sum(), 1))
    haps = (bundle.get("phone") or {}).get("haptics") or []
    alerts = [h for h in haps if (h.get("code") or "") not in ("D", "E", "R", "K") and int(h.get("level") or 0) >= 1]
    hit_by, miss_by = Counter(), Counter()
    missed = []
    for it in mom.get("items") or []:
        rule = RULE_BY_NUDGE.get(it.get("text") or "", "annotation")
        (hit_by if it.get("hit") else miss_by)[rule] += 1
        if not it.get("hit"):
            missed.append({"t": it.get("t"), "rule": rule, "what": it.get("what_happened"), "priority": it.get("priority"),
                           "nearest": it.get("nearest")})
    # The app's interject slider defaults to 0 (every coach line is spoken).
    # What a raised slider would have done: server lines gated on importance.
    gated = {}
    moments_t = [float(it["t"]) for it in mom.get("items") or []]
    pre, post = 1.5, float(mom.get("window_s") or 6.0)
    for th in GATES:
        fz = [ln for ln in lines if (ln.get("source") == "phone" and ln.get("fires"))
              or (ln.get("source") == "server" and ln.get("kind") in ("response", "nudge")
                  and (ln.get("importance") or 0) >= th)]
        near = [ln for ln in fz if any(t - pre <= ln["at_s"] <= t + post for t in moments_t)]
        gated[th] = {"fires": len(fz), "false_fires": len(fz) - len(near),
                     "hits": sum(1 for t in moments_t if any(t - pre <= ln["at_s"] <= t + post for ln in fz))}
    imps = [float(ln["importance"]) for ln in lines
            if ln.get("source") == "server" and ln.get("kind") in ("response", "nudge") and ln.get("importance") is not None]
    srv = [ln for ln in lines if ln.get("source") == "server" and ln.get("kind") in ("response", "nudge")]
    frag = [ln for ln in srv if len(re.findall(r"[A-Za-z']+", ln.get("utterance_text") or "")) <= 2]
    openings = Counter(_opening(ln.get("text") or "") for ln in srv)
    leak = [ln for ln in srv if LEAK_RE.search(" ".join(ln.get("all") or [ln.get("text") or ""]))]
    ph = (sc.get("identity") or {}).get("phone") or {}
    srv_id = (sc.get("identity") or {}).get("server") or {}
    has_truth = corpus not in NO_SPEAKER_TRUTH and bool(segs)
    viol = sc.get("violations") or []
    lat = [float(ln["latency_ms"]) for ln in lines
           if ln.get("source") == "server" and ln.get("kind") in ("response", "nudge") and ln.get("latency_ms") is not None]
    return {
        "name": name, "corpus": corpus, "group": prov.get("group") or ("open-web" if corpus == "open-web" else "heated"),
        "heat_basis": prov.get("heat_basis"), "duration_s": dur, "hours": hours,
        "wearer": prov.get("wearer"), "window_s": prov.get("window_s"),
        "moments_hits": int(mom.get("hits") or 0), "moments_total": int(mom.get("total") or 0),
        "fires": len(fires), "false_fires": len(unmatched),
        # What the same number of fires dropped at RANDOM times would catch:
        # P(at least one fire in the 7.5 s window) for a Poisson stream.
        "chance_hit_rate": (1.0 - float(np.exp(-len(fires) / dur * (pre + post)))) if dur else None,
        "fires_per_h": len(fires) / hours if hours else None,
        "false_fires_per_h": len(unmatched) / hours if hours else None,
        "server_lines": sum(1 for ln in lines if ln.get("source") == "server" and ln.get("kind") in ("response", "nudge")),
        "server_spoken": sum(1 for ln in fires if ln.get("source") == "server"),
        "phone_buzzes": sum(1 for ln in fires if ln.get("source") == "phone"),
        "identity_accuracy": ph.get("accuracy") if has_truth else None,
        "identity_recall": ph.get("wearer_recall") if has_truth else None,
        "identity_false_self": ph.get("false_self") if has_truth else None,
        "identity_decided": ph.get("decided") if has_truth else None,
        "identity_correct": ph.get("correct") if has_truth else None,
        "time_to_confirm_s": ph.get("first_confirmed_s") if has_truth else None,
        "server_time_to_confirm_s": srv_id.get("first_confirmed_s") if has_truth else None,
        "latencies_ms": lat,
        "violations": dict(Counter(v["kind"] for v in viol)),
        "violation_examples": [{"kind": v["kind"], "at_s": v["at_s"], "text": v.get("text"), "evidence": v.get("evidence")} for v in viol[:6]],
        "errors": len(sc.get("errors") or []),
        "merged_turns": merged, "phone_turns": len(sent), "merged_examples": merged_ex,
        "fires_near_laughter": len(near_laugh),
        "laughter_examples": [{"at_s": ln["at_s"], "text": ln.get("text"), "source": ln.get("source")} for ln in near_laugh[:3]],
        "laughs": len(laughs),
        "hit_by_rule": dict(hit_by), "missed_by_rule": dict(miss_by), "missed": missed[:8],
        "gated": gated, "importances": imps,
        "phone_speech_coverage": coverage, "phone_alerts": len(alerts), "phone_positive": len(haps) - len(alerts),
        "fragment_lines": len(frag),
        "fragment_examples": [{"at_s": ln["at_s"], "text": f"“{ln.get('utterance_text')}” -> {ln.get('text')}"} for ln in frag[:3]],
        "openings": dict(openings),
        "label_leak_lines": len(leak),
        "label_leaks": [{"at_s": ln["at_s"], "text": next((x for x in (ln.get("all") or [ln.get("text")]) if x and LEAK_RE.search(x)),
                                                          ln.get("text"))} for ln in leak[:2]],
        "false_fire_examples": [{"at_s": u["at_s"], "text": u.get("text"), "source": u.get("source"), "kind": u.get("kind")} for u in unmatched[:6]],
    }


def _pctl(xs: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if xs else None


def _agg(items: list[dict]) -> dict:
    hours = sum(i["hours"] for i in items)
    hits = sum(i["moments_hits"] for i in items)
    tot = sum(i["moments_total"] for i in items)
    idd = [i for i in items if i["identity_decided"]]
    dec = sum(i["identity_decided"] for i in idd)
    cor = sum(i["identity_correct"] for i in idd)
    ttc = [i["time_to_confirm_s"] for i in items if i["time_to_confirm_s"] is not None]
    never = sum(1 for i in items if i["identity_decided"] is not None and i["time_to_confirm_s"] is None)
    lat = [x for i in items for x in i["latencies_ms"]]
    viol = Counter()
    for i in items:
        viol.update(i["violations"])
    rec = [i["identity_recall"] for i in items if i["identity_recall"] is not None]
    lines = sum(i["server_lines"] for i in items)
    gate = {}
    for th in GATES:
        f = sum(i["gated"][th]["fires"] for i in items)
        ff = sum(i["gated"][th]["false_fires"] for i in items)
        h = sum(i["gated"][th]["hits"] for i in items)
        gate[th] = {"fires_per_h": f / hours if hours else None, "false_fires_per_h": ff / hours if hours else None,
                    "moment_hit_rate": h / tot if tot else None, "hits": h}
    imps = [x for i in items for x in i["importances"]]
    return {
        "items": len(items), "hours": hours, "gated": gate,
        "importance_p50": _pctl(imps, 50), "importance_p90": _pctl(imps, 90),
        "moments_hits": hits, "moments_total": tot, "moment_hit_rate": hits / tot if tot else None,
        "chance_hit_rate": (sum(i["chance_hit_rate"] * i["moments_total"] for i in items if i["chance_hit_rate"] is not None) / tot)
        if tot else None,
        "fires": sum(i["fires"] for i in items), "false_fires": sum(i["false_fires"] for i in items),
        "fires_per_h": sum(i["fires"] for i in items) / hours if hours else None,
        "false_fires_per_h": sum(i["false_fires"] for i in items) / hours if hours else None,
        "identity_accuracy": cor / dec if dec else None, "identity_items": len(idd),
        "identity_recall_mean": float(np.mean(rec)) if rec else None,
        "time_to_confirm_median_s": float(np.median(ttc)) if ttc else None, "never_confirmed": never,
        "latency_p50_ms": _pctl(lat, 50), "latency_p90_ms": _pctl(lat, 90), "latency_n": len(lat),
        "server_lines": lines, "invented_fact": viol.get("invented-fact", 0), "words_in_mouth": viol.get("words-in-mouth", 0),
        "ungrounded_first_person": viol.get("ungrounded-first-person", 0),
        "flags_per_100_lines": (100.0 * (viol.get("invented-fact", 0) + viol.get("words-in-mouth", 0)) / lines) if lines else None,
        "merged_turns": sum(i["merged_turns"] for i in items), "phone_turns": sum(i["phone_turns"] for i in items),
        "fires_near_laughter": sum(i["fires_near_laughter"] for i in items),
        "errors": sum(i["errors"] for i in items),
    }


def aggregate(items: list[dict]) -> dict[tuple[str, str], dict]:
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for i in items:
        by[(i["group"], i["corpus"])].append(i)
        by[(i["group"], "ALL")].append(i)
    return {k: _agg(v) for k, v in by.items()}


# ---------------------------------------------------------------------------
# Defects (detectors with evidence; ranked by how much they would cost a wearer)
# ---------------------------------------------------------------------------

def defects(items: list[dict]) -> list[dict]:
    out = []
    agg = aggregate(items)

    def g(group, key):
        return (agg.get((group, "ALL")) or {}).get(key)

    spoken = sum(i["server_spoken"] for i in items)
    srv_lines = sum(i["server_lines"] for i in items)
    if srv_lines and spoken >= 0.9 * srv_lines:
        cg = (agg.get(("calm", "ALL")) or {}).get("gated") or {}
        hg = (agg.get(("heated", "ALL")) or {}).get("gated") or {}
        out.append({"id": "every-line", "title": "The default interject setting (0) speaks after nearly every turn",
                    "severity": (g("calm", "fires_per_h") or 0) / 10,
                    "evidence": f"{spoken} of {srv_lines} coach lines spoken; calm controls {_num(g('calm', 'fires_per_h'))} fires/h, "
                                f"heated {_num(g('heated', 'fires_per_h'))}/h. At interject 50: calm "
                                f"{_num((cg.get(50) or {}).get('false_fires_per_h'))}/h, heated moments caught "
                                f"{_pct((hg.get(50) or {}).get('moment_hit_rate'))}; at 70: calm "
                                f"{_num((cg.get(70) or {}).get('false_fires_per_h'))}/h, heated "
                                f"{_pct((hg.get(70) or {}).get('moment_hit_rate'))}. Importance p50 calm "
                                f"{_num(g('calm', 'importance_p50'), '{:.0f}')} vs heated {_num(g('heated', 'importance_p50'), '{:.0f}')}",
                    "items": []})
    heated_items = [i for i in items if i["group"] == "heated"]
    if heated_items:
        al = sum(i["phone_alerts"] for i in heated_items)
        mins = sum(i["hours"] for i in heated_items) * 60
        if al <= 0.05 * mins:
            out.append({"id": "no-alert-buzz", "title": "The phone's own alert buzz almost never fires, even in rated conflict and shouting",
                        "severity": 40.0,
                        "evidence": f"{al} alert buzzes in {mins:.0f} min of heated audio (all {sum(i['phone_positive'] for i in items)} "
                                    "phone haptics in the batch were positive-reinforcement codes)",
                        "items": [(i["name"], f"{i['moments_total']} moments, {i['phone_alerts']} alert buzzes", [])
                                  for i in sorted(heated_items, key=lambda i: i["moments_total"], reverse=True)[:3]]})
    cov = [i for i in items if i["phone_speech_coverage"] is not None]
    if cov:
        low = sorted(cov, key=lambda i: i["phone_speech_coverage"])[:3]
        mean = float(np.mean([i["phone_speech_coverage"] for i in cov]))
        out.append({"id": "dropped-speech", "title": "Quiet talk never becomes a phone turn (dropped speech)",
                    "severity": 40.0 * (1 - low[0]["phone_speech_coverage"]) * (1 if low[0]["phone_speech_coverage"] < 0.5 else 0.3),
                    "evidence": f"mean {100 * mean:.0f}% of ground-truth speech covered by a phone turn; worst "
                                + ", ".join(f"{i['name']} {100 * i['phone_speech_coverage']:.0f}%" for i in low),
                    "items": [(i["name"], f"{100 * i['phone_speech_coverage']:.0f}% of speech covered, {i['phone_turns']} turns, "
                                          f"{i['moments_hits']}/{i['moments_total']} moments", []) for i in low]})
    if srv_lines:
        fr = sum(i["fragment_lines"] for i in items)
        worst = sorted(items, key=lambda i: i["fragment_lines"], reverse=True)[:3]
        out.append({"id": "fragments", "title": "Coaching on fragments and backchannels (turns of two words or fewer)",
                    "severity": 25.0 * fr / srv_lines,
                    "evidence": f"{fr} of {srv_lines} coach lines ({100.0 * fr / srv_lines:.0f}%) answer a turn of <= 2 words",
                    "items": [(i["name"], f"{i['fragment_lines']} lines", i["fragment_examples"][:2]) for i in worst if i["fragment_lines"]]})
        op = Counter()
        for i in items:
            op.update(i["openings"])
        top = op.most_common(4)
        share = sum(n for _, n in top) / srv_lines
        out.append({"id": "template", "title": "Template collapse: the same few openings everywhere, calm or heated",
                    "severity": 15.0 * share,
                    "evidence": f"{100 * share:.0f}% of {srv_lines} lines open with one of: "
                                + ", ".join(f"“{k}…” ({n})" for k, n in top),
                    "items": []})
    calm_ff = g("calm", "false_fires_per_h")
    lively = [i for i in items if i["corpus"] in ("AMI", "CHiME-6") and i["group"] == "heated"]
    lively_ff = _agg(lively)["false_fires_per_h"] if lively else None
    if calm_ff is not None or lively_ff is not None:
        worst = sorted((i for i in items if i["group"] == "calm" or i in lively), key=lambda i: i["false_fires_per_h"] or 0, reverse=True)[:3]
        out.append({"id": "false-fires", "title": "Fires with no ground-truth moment, even in calm talk",
                    "severity": (calm_ff or 0) + 0.5 * (lively_ff or 0),
                    "evidence": f"calm controls: {_num(calm_ff)} unmatched fires/h; lively (overlap-dense AMI/CHiME, no conflict): {_num(lively_ff)}/h",
                    "items": [(i["name"], f"{_num(i['false_fires_per_h'])}/h", i["false_fire_examples"][:2]) for i in worst]})
    laugh = sum(i["fires_near_laughter"] for i in items)
    laughs = sum(i["laughs"] for i in items)
    if laugh:
        worst = sorted(items, key=lambda i: i["fires_near_laughter"], reverse=True)[:3]
        out.append({"id": "loud-happy", "title": "Loud-but-happy treated as trouble: fires right after laughter",
                    "severity": 60.0 * laugh / max(sum(i["hours"] for i in items), 1e-6) / 10,
                    "evidence": f"{laugh} unmatched fires within {LAUGH_PRE_S:.0f} s before / {LAUGH_POST_S:.0f} s after a corpus laughter mark "
                                f"({laughs} laughter marks in the batch)",
                    "items": [(i["name"], f"{i['fires_near_laughter']} after laughter", i["laughter_examples"][:2]) for i in worst if i["fires_near_laughter"]]})
    mt, pt = sum(i["merged_turns"] for i in items), sum(i["phone_turns"] for i in items)
    if pt:
        worst = sorted(items, key=lambda i: (i["merged_turns"] / i["phone_turns"]) if i["phone_turns"] else 0, reverse=True)[:3]
        out.append({"id": "turn-merging", "title": "Turn merging: one phone turn spans two or more real voices",
                    "severity": 30.0 * mt / pt,
                    "evidence": f"{mt} of {pt} phone turns ({100.0 * mt / pt:.0f}%) cover >= 2 ground-truth speakers (>= 0.5 s each)",
                    "items": [(i["name"], f"{i['merged_turns']}/{i['phone_turns']}", i["merged_examples"][:2]) for i in worst]})
    idi = [i for i in items if i["identity_decided"]]
    if idi:
        acc = _agg(idi)
        worst = sorted(idi, key=lambda i: (i["identity_recall"] if i["identity_recall"] is not None else 1.0))[:3]
        out.append({"id": "identity", "title": "Identity misses: the wearer's own turns not recognised (held-out voiceprint)",
                    "severity": 20.0 * (1 - (acc["identity_recall_mean"] or 0)) + 2.0 * acc["never_confirmed"],
                    "evidence": f"phone accuracy {_pct(acc['identity_accuracy'])}, mean wearer recall {_pct(acc['identity_recall_mean'])}, "
                                f"never confirmed in {acc['never_confirmed']} of {len(idi)} items, median time-to-confirm "
                                f"{_num(acc['time_to_confirm_median_s'])} s",
                    "items": [(i["name"], f"recall {_pct(i['identity_recall'])}, false-self {i['identity_false_self']}, first ok "
                                          f"{_num(i['time_to_confirm_s'])} s", []) for i in worst]})
    heated = [i for i in items if i["group"] == "heated"]
    miss = Counter()
    hit = Counter()
    for i in heated:
        miss.update(i["missed_by_rule"])
        hit.update(i["hit_by_rule"])
    if miss:
        rate = {r: hit[r] / (hit[r] + miss[r]) for r in set(hit) | set(miss)}
        worst = sorted(heated, key=lambda i: (i["moments_total"] - i["moments_hits"]), reverse=True)[:3]
        out.append({"id": "missed-moments", "title": "Heated moments missed",
                    "severity": 10.0 * sum(miss.values()) / max(sum(miss.values()) + sum(hit.values()), 1),
                    "evidence": f"overall {_pct(_agg(heated)['moment_hit_rate'])} vs {_pct(_agg(heated)['chance_hit_rate'])} for random fires at the same rate; by rule: " + ", ".join(f"{r} {hit[r]}/{hit[r] + miss[r]} ({100 * v:.0f}%)" for r, v in sorted(rate.items())),
                    "items": [(i["name"], f"{i['moments_hits']}/{i['moments_total']} caught",
                               [{"at_s": m["t"], "text": f"{m['rule']}: {m['what']}"} for m in i["missed"][:2]]) for i in worst]})
    flags = sum(i["violations"].get("invented-fact", 0) + i["violations"].get("words-in-mouth", 0) for i in items)
    if flags:
        worst = sorted(items, key=lambda i: i["violations"].get("invented-fact", 0) + i["violations"].get("words-in-mouth", 0), reverse=True)[:3]
        lines = sum(i["server_lines"] for i in items)
        out.append({"id": "flags", "title": "Possible invented facts / words in someone's mouth (heuristic flags)",
                    "severity": 100.0 * flags / max(lines, 1) / 5,
                    "evidence": f"{flags} flags over {lines} coach lines",
                    "items": [(i["name"], json.dumps(i["violations"]), [{"at_s": v["at_s"], "text": f"{v['text']} [{v['evidence']}]"}
                                                                       for v in i["violation_examples"][:2]]) for i in worst]})
    leaks = [(i, e) for i in items for e in i["label_leaks"]]
    if leaks:
        n = sum(i["label_leak_lines"] for i in items)
        out.append({"id": "label-leak", "title": "The coach says internal diarization labels aloud (“Let Speaker E finish”)",
                    "severity": 20.0 * n / max(srv_lines, 1),
                    "evidence": f"{n} of {srv_lines} coach lines name a “Speaker X” label the wearer has never heard "
                                "(most of the heuristic invented-fact flags are these)",
                    "items": [(i["name"], "", [e]) for i, e in leaks[:3]]})
    order = {k: n for n, k in enumerate(RANK)}
    return sorted(out, key=lambda d: (order.get(d["id"], 99), -d["severity"]))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

EXTRA_CSS = """
.grp{margin-top:22px}.grp>h2{font-size:1.15rem}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.scroll table{min-width:640px}
.defect{border-left:4px solid var(--bad);padding-left:10px;margin:10px 0}
.defect h3{font-size:.98rem;margin:.1rem 0}
.ex{font-size:.82rem;color:var(--muted);margin:.15rem 0 .15rem 1rem}
.basis{font-size:.78rem;color:var(--muted)}
"""


def _tiles(a: dict) -> str:
    def tile(v, label, cls=""):
        return f'<div class="tile"><b class="{cls}">{v}</b><span>{esc(label)}</span></div>'
    return '<div class="tiles">' + "".join([
        tile(f"{a['moments_hits']}/{a['moments_total']}",
             f"moments caught ({_pct(a['moment_hit_rate'])}; random fires at the same rate: {_pct(a['chance_hit_rate'])})"),
        tile(_num(a["false_fires_per_h"]), "fires with no moment / hour", "bad" if (a["false_fires_per_h"] or 0) > 30 else ""),
        tile(_pct(a["identity_accuracy"]), f"phone voice-ID accuracy ({a['identity_items']} items)"),
        tile(_num(a["time_to_confirm_median_s"]), "median s to first correct “that's you”"),
        tile(f"{_ms(a['latency_p50_ms'])} / {_ms(a['latency_p90_ms'])}", f"coach latency p50 / p90 (n={a['latency_n']})"),
        tile(f"{a['invented_fact']} / {a['words_in_mouth']}", f"invented-fact / words-in-mouth flags ({a['server_lines']} lines)"),
    ]) + "</div>" + '<div class="sub" style="margin-top:6px">Above: the app default (interject slider 0, every coach line spoken). '\
        'If the slider were raised, server lines below that importance stay silent:</div><div class="tiles">' + "".join(
        tile(f"{_pct(a['gated'][th]['moment_hit_rate'])} · {_num(a['gated'][th]['false_fires_per_h'])}/h",
             f"interject {th}: moments caught · no-moment fires per hour") for th in GATES) + tile(
        f"{_num(a['importance_p50'], '{:.0f}')} / {_num(a['importance_p90'], '{:.0f}')}", "coach importance p50 / p90") + "</div>"


def _corpus_table(agg: dict, group: str) -> str:
    rows = sorted((k[1], v) for k, v in agg.items() if k[0] == group and k[1] != "ALL")
    if not rows:
        return ""
    body = "".join(
        f"<tr><td>{esc(c)}</td><td class=num>{a['items']}</td><td class=num>{a['hours'] * 60:.0f}</td>"
        f"<td class=num>{a['moments_hits']}/{a['moments_total']}</td><td class=num>{_num(a['false_fires_per_h'])}</td>"
        f"<td class=num>{_num(a['fires_per_h'])}</td><td class=num>{_pct(a['identity_accuracy'])}</td>"
        f"<td class=num>{_num(a['time_to_confirm_median_s'])}</td><td class=num>{_ms(a['latency_p50_ms'])}</td>"
        f"<td class=num>{_ms(a['latency_p90_ms'])}</td><td class=num>{a['invented_fact']}/{a['words_in_mouth']}</td>"
        f"<td class=num>{a['merged_turns']}/{a['phone_turns']}</td><td class=num>{a['fires_near_laughter']}</td></tr>"
        for c, a in rows)
    return ('<div class="scroll"><table><tr><th>corpus</th><th class=num>items</th><th class=num>min</th><th class=num>moments</th>'
            '<th class=num>no-moment fires/h</th><th class=num>all fires/h</th><th class=num>voice-ID</th><th class=num>confirm s</th>'
            '<th class=num>p50</th><th class=num>p90</th><th class=num>fact/mouth</th><th class=num>merged turns</th>'
            f'<th class=num>fires after laughter</th></tr>{body}</table></div>')


def _item_table(items: list[dict]) -> str:
    body = "".join(
        f"<tr><td><a href=\"{esc(i['name'])}.html\">{esc(i['name'])}</a><div class=basis>{esc(i['heat_basis'] or '')}</div></td>"
        f"<td class=num>{i['duration_s'] / 60:.1f}</td><td class=num>{i['moments_hits']}/{i['moments_total']}</td>"
        f"<td class=num>{i['false_fires']} ({_num(i['false_fires_per_h'], '{:.0f}')}/h)</td>"
        f"<td class=num>{i['server_spoken']}/{i['server_lines']}</td><td class=num>{i['phone_buzzes']}</td>"
        f"<td class=num>{_pct(i['identity_accuracy'])} / {_pct(i['identity_recall'])}</td>"
        f"<td class=num>{_num(i['time_to_confirm_s'])}</td>"
        f"<td class=num>{_ms(_pctl(i['latencies_ms'], 50))}/{_ms(_pctl(i['latencies_ms'], 90))}</td>"
        f"<td class=num>{i['violations'].get('invented-fact', 0)}/{i['violations'].get('words-in-mouth', 0)}</td></tr>"
        for i in sorted(items, key=lambda i: (i["corpus"], i["name"])))
    return ('<div class="scroll"><table><tr><th>item</th><th class=num>min</th><th class=num>moments</th><th class=num>no-moment fires</th>'
            '<th class=num>spoken/lines</th><th class=num>buzzes</th><th class=num>ID acc / recall</th><th class=num>confirm s</th>'
            f'<th class=num>p50/p90</th><th class=num>fact/mouth</th></tr>{body}</table></div>')


def _ex(e: dict) -> str:
    t = e.get("at_s", e.get("t"))
    head = f"{float(t):.1f} s: " if isinstance(t, (int, float)) else ""
    if "voices" in e:
        return f'<div class="ex">{head}{esc(", ".join(e["voices"]))} in one turn: “{esc(e["text"])}”</div>'
    return f'<div class="ex">{head}{esc(e.get("text"))}</div>'


def render(items: list[dict], agg: dict, defs: list[dict], *, notes: list[str] | None = None) -> str:
    total_h = sum(i["hours"] for i in items)
    corp = Counter(i["corpus"] for i in items)
    parts = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Corpus Replay Summary</title>'
        f"<style>{CSS}{EXTRA_CSS}</style></head><body><main>",
        "<h1>Corpus replay summary</h1>",
        f'<div class="sub">{len(items)} items, {total_h:.2f} h of real conversation ('
        + ", ".join(f"{esc(c)} {n}" for c, n in sorted(corp.items()))
        + f'). Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}. Heated and calm results are never pooled.</div>',
    ]
    if notes:
        parts.append('<div class="banner"><ul class="plain">' + "".join(f"<li>{n}</li>" for n in notes) + "</ul></div>")
    if defs:
        parts.append('<div class="card"><h2>Biggest defects (ranked)</h2>')
        for n, d in enumerate(defs, 1):
            parts.append(f'<div class="defect"><h3>{n}. {esc(d["title"])}</h3><div>{esc(d["evidence"])}</div>')
            for name, summ, exs in d["items"]:
                parts.append(f'<div><a href="{esc(name)}.html">{esc(name)}</a> <span class="muted">{esc(summ)}</span></div>')
                parts += [_ex(e) for e in exs]
            parts.append("</div>")
        parts.append("</div>")
    for gkey, glabel in GROUPS:
        gitems = [i for i in items if i["group"] == gkey]
        if not gitems:
            continue
        a = agg[(gkey, "ALL")]
        parts.append(f'<section class="grp"><h2>{esc(glabel)}: {a["items"]} items, {a["hours"] * 60:.0f} min</h2>')
        parts.append(_tiles(a))
        parts.append(f'<div class="card"><h2>Per corpus</h2>{_corpus_table(agg, gkey)}</div>')
        parts.append(f'<div class="card"><details><summary>Per item ({len(gitems)})</summary>{_item_table(gitems)}</details></div>')
        parts.append("</section>")
    parts.append(
        '<div class="card"><h2>How to read this</h2><ul class="plain">'
        "<li><b>Moments</b> come only from ground truth by written rules (talks-over, raised-voice, monologue-cuts-off, "
        "CONFER's rated conflict peaks); a moment is caught when a fire lands 1.5 s before to 6 s after it.</li>"
        "<li><b>Fires</b> = a spoken server line (importance cleared the interject threshold) or a phone alert buzz. "
        "<b>No-moment fires</b> are fires with no ground-truth moment nearby: in a calm control every fire is one.</li>"
        "<li><b>Voice-ID</b> = the phone's per-turn “is this the wearer” against transcript truth, with a print enrolled "
        "from a HELD-OUT part of the same session (not the window). CONFER has no speaker truth: not scored.</li>"
        "<li><b>Latency</b> = speech end to the coach line arriving, local server + real Claude (Haiku via the LLM cache), "
        "three items replayed in parallel.</li>"
        "<li><b>Flags</b> are heuristics (score.violations): a capitalised word or number nobody said, or a line scripting "
        "someone else.</li>"
        "<li>AMI and CHiME-6 “heated” windows are the most overlap-dense/loud stretches; no one rated them as conflict.</li>"
        "</ul></div></main></body></html>")
    return "".join(parts)


def collect(work_root: Path, inbox: Path, names: list[str] | None = None) -> list[dict]:
    items = []
    for run in sorted(Path(work_root).glob("*/run.json")):
        name = run.parent.name
        if names is not None and name not in names:
            continue
        prov_path = Path(inbox) / name / f"{name}.corpus.json"
        prov = json.loads(prov_path.read_text()) if prov_path.exists() else None
        if prov is None and not name.startswith("yt_"):
            continue                     # the owner's own recordings are not part of the corpus batch
        bundle = json.loads(run.read_text())
        if not bundle.get("server"):
            continue
        items.append(item_metrics(bundle, prov))
    return items


def write(out: Path, work_root: Path, inbox: Path, *, notes: list[str] | None = None) -> Path:
    items = collect(work_root, inbox)
    done = {i["name"] for i in items}
    missing = sorted(p.parent.name for p in Path(inbox).glob("*/*.corpus.json") if p.parent.name not in done)
    notes = list(notes or [])
    if missing:
        notes.append("Not in these numbers (no completed server run): " + ", ".join(esc(m) for m in missing)
                     + ". Typical cause: Deepgram returned no words, so the phone loop had no script.")
    if any(i["corpus"] == "CONFER" for i in items):
        notes.append("CONFER is Greek: Deepgram ran with language=el and the coach read Greek text. Its numbers test "
                     "conflict detection and pacing, not English coaching; it has no speaker truth, so no voice-ID numbers.")
    agg = aggregate(items)
    defs = defects(items)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(items, agg, defs, notes=notes))
    out.with_suffix(".json").write_text(json.dumps(
        {"items": items, "aggregate": {f"{g}|{c}": v for (g, c), v in agg.items()}, "defects": defs}, indent=1, default=float))
    return out
