#!/usr/bin/env python3
"""signal_gate.py — post-hoc speak gates driven by MEASURED conversation
signals instead of (or blended with) the LLM's self-rated importance, at $0.

regate.py showed that raising the importance bar silences lines at random
(lift ~0.00): Haiku's importance is flat (median ~72). This module computes,
for every recorded server line, signals a LIVE system could know at the time
the line would be spoken (phone turns that ended before it, their prosody,
the phone's speaker labels, the transcript text, the coach line's own text),
plus a few corpus-truth-only signals marked UPPER BOUND, and gates on a
small linear score of them.

    tmp/venv/bin/python scripts/recreplay/signal_gate.py --split dev --gate '{"score":"turn_switch","thresh":1,"min_gap_s":60}'
    tmp/venv/bin/python scripts/recreplay/signal_gate.py --split dev --features   # per-signal moment-proximity table

Gate keys (all optional):
  min_words / skip_bc          same as regate (the B gate's turn filters)
  importance / unknown_cap     same as regate (applied BEFORE the signal score when set)
  score                        name of a feature, or {feature: weight, ...}; a line passes when
                               sum(weight * feature) >= thresh
  thresh                       threshold on that score
  rank_window / rank_top       session-relative: the line's score must be in the top
                               ``rank_top`` fraction of the last ``rank_window`` server lines' scores
  min_gap_s                    greedy spacing between spoken server lines (as regate)

Live-available features (computed causally: only phone turns whose end_time
<= the line's arrival, and the line itself):
  imp            the LLM's importance
  self / other   answered turn is the wearer's / someone else's (phone is_self)
  turn_switch    the answered turn has a different phone speaker label than the turn before it
  gap_short      that switch came with a gap <= 0.4 s (latched turn-taking; the segmenter's own floor is ~0.32 s)
  alt30          speaker changes among phone turns in the 30 s before the line
  loud_z         answered turn's rms_dbfs vs the session's running median/MAD (prior turns only)
  rate_z, pitch_z  same for speech rate and pitch
  floor_req      floor-request phrase in the answered turn ("let me finish", "can I answer", Greek too)
  heat_lex       lexical heat count (you always/never, insults, "shut up", Greek equivalents)
  question       the answered turn ends with a question mark
  coach_interrupt the coach LINE itself is about interrupting / letting someone finish
  wearer_run     seconds of consecutive wearer turns ending with the answered turn
Extra gate keys from the line-quality judge (agent J):
  listen_rule    drop "let them finish / pause / listen" lines unless the answered turn is the wearer's
                 AND (wearer_run >= listen_run_s, loud_z >= listen_loud_z, lexical heat, or a latched switch)
  max_dur_s      drop lines answering phone turns longer than this (merged-voice turns)
Truth-only (UPPER BOUND, needs the annotation): gt_overlap = a ground-truth
speaker started while another still had >= 1 s left inside [turn start - 1, line arrival].

Same CAVEAT as regate.py: post-hoc gating ignores cascade effects.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from recreplay import corpus_summary as cs  # noqa: E402
from recreplay import regate  # noqa: E402
from recreplay import score as score_mod  # noqa: E402
from recreplay.corpora import is_backchannel_text  # noqa: E402

# Data lives in the MAIN checkout (worktrees have no tmp/recordings).
MAIN_RECORDINGS = Path("/Users/sagearbor/projects/githubs/mindshift/tmp/recordings")
RECORDINGS = MAIN_RECORDINGS if MAIN_RECORDINGS.exists() else regate.RECORDINGS

B_GATE = {"min_words": 3, "skip_bc": True, "importance": 78, "unknown_cap": 72, "min_gap_s": 60.0}

FLOOR_RE = re.compile(
    r"\b(let me (finish|talk|speak|answer|explain|say)|can i (finish|answer|talk|speak|say)|"
    r"(would|will) you let me|hold on|wait a (second|minute)|you'?re not listening|stop interrupting|"
    r"let me get a word)\b"
    r"|αφήστε με|άσε με|να ολοκληρώσω|να τελειώσω|μη με διακόπτ|δεν με αφήνετε|ένα λεπτό|μισό λεπτό|ακούστε με",
    re.I)
HEAT_RE = re.compile(
    r"\byou (always|never)\b|\bshut up\b|\bstupid\b|\bidiot\b|\bridiculous\b|\bliar\b|\bbullshit\b|\bshit\b|\bfuck"
    r"|\bthat'?s not true\b|\bno no\b|\bwhatever\b|\bhow dare\b"
    r"|ψέματα|ντροπή|γελοί|ανοησίες|πάντα εσείς|ποτέ δεν|δεν ντρέπεστε|ψεύτ|απαράδεκτ",
    re.I)
COACH_INT_RE = re.compile(r"\b(interrupt|finish|let (them|her|him) (complete|speak|talk)|cut (them|her|him) off|"
                          r"talking over|wait your turn|pause\b|listen\b|hold (back|off)|let (them|her|him) (finish|talk|speak))", re.I)

LIVE_FEATURES = ["imp", "imp_rel", "dur", "self", "other", "turn_switch", "gap_short", "alt30", "loud_z", "rate_z", "pitch_z",
                 "floor_req", "heat_lex", "question", "coach_interrupt", "wearer_run"]
TRUTH_FEATURES = ["gt_overlap"]


def _robust_z(x: float | None, hist: list[float]) -> float:
    if x is None or len(hist) < 4:
        return 0.0
    med = statistics.median(hist)
    mad = statistics.median(abs(h - med) for h in hist) or 1.0
    return (x - med) / (1.4826 * mad)


def line_features(bundle: dict) -> dict:
    """{(turn_uid, at_s): features} for every server response/nudge line."""
    sent = sorted((bundle.get("phone") or {}).get("sent") or [], key=lambda t: float(t["start_time"]))
    by_uid = {t.get("turn_uid"): i for i, t in enumerate(sent)}
    gt = cs._gt(bundle)
    segs = sorted((s for s in gt.get("segments") or [] if not s.get("is_backchannel")), key=lambda s: s["start"])
    out = {}
    for ln in bundle["score"]["lines"]:
        if ln.get("source") != "server" or ln.get("kind") not in ("response", "nudge"):
            continue
        at = float(ln["at_s"])
        i = by_uid.get(ln.get("turn_uid"))
        turn = sent[i] if i is not None else {}
        prev = sent[i - 1] if i else {}
        prior = [t for t in sent if float(t["end_time"]) <= at and t is not turn]
        f: dict = {"imp": float(ln["importance"]) if ln.get("importance") is not None else 100.0}
        f["self"] = 1.0 if turn.get("is_self") is True else 0.0
        f["other"] = 1.0 if turn.get("is_self") is False else 0.0
        f["unknown"] = 1.0 if turn.get("is_self") is None else 0.0
        sw = bool(prev) and bool(turn) and prev.get("speaker") != turn.get("speaker")
        f["turn_switch"] = 1.0 if sw else 0.0
        gap = (float(turn["start_time"]) - float(prev["end_time"])) if (prev and turn) else 9.0
        f["gap_short"] = 1.0 if sw and gap <= 0.4 else 0.0     # segmenter silence-close is 0.3 s, so 0.32 is the floor
        win = [t for t in sent if at - 30.0 <= float(t["end_time"]) <= at]
        f["alt30"] = float(sum(1 for a, b in zip(win, win[1:]) if a.get("speaker") != b.get("speaker")))
        pr = turn.get("prosody") or {}
        for key, name in (("rms_dbfs", "loud_z"), ("speech_rate", "rate_z"), ("pitch_hz", "pitch_z")):
            hist = [float((t.get("prosody") or {})[key]) for t in prior if (t.get("prosody") or {}).get(key) is not None]
            f[name] = round(_robust_z(pr.get(key), hist), 3)
        txt = ln.get("utterance_text") or turn.get("text") or ""
        f["floor_req"] = 1.0 if FLOOR_RE.search(txt) else 0.0
        f["heat_lex"] = float(len(HEAT_RE.findall(txt)))
        f["question"] = 1.0 if txt.strip().endswith(("?", ";")) else 0.0     # Greek question mark is ';'
        f["coach_interrupt"] = 1.0 if COACH_INT_RE.search(" ".join(ln.get("all") or [ln.get("text") or ""])) else 0.0
        # UPPER BOUND (truth): someone started while another speaker still had >= 1 s left
        lo = (float(turn["start_time"]) - 1.0) if turn else at - 8.0
        ov = 0.0
        for s in segs:
            if lo <= s["start"] <= at:
                if any(o["speaker"] != s["speaker"] and o["start"] < s["start"] and o["end"] - s["start"] >= 1.0 for o in segs
                       if o["start"] < s["start"]):
                    ov = 1.0
                    break
        f["gt_overlap"] = ov
        f["dur"] = (float(turn["end_time"]) - float(turn["start_time"])) if turn else 0.0
        # wearer talk run: seconds of consecutive is_self turns ending with the answered turn
        run = 0.0
        if turn.get("is_self") is True and i is not None:
            j = i
            while j >= 0 and sent[j].get("is_self") is True:
                run += float(sent[j]["end_time"]) - float(sent[j]["start_time"])
                j -= 1
        f["wearer_run"] = run
        out[(ln.get("turn_uid"), round(at, 3))] = f
    # session-relative importance: the line's importance minus the median of the
    # session's EARLIER lines (live: the server has every line it produced)
    hist: list[float] = []
    for key in sorted(out, key=lambda k: k[1]):
        f = out[key]
        f["imp_rel"] = (f["imp"] - statistics.median(hist)) if len(hist) >= 3 else 0.0
        hist.append(f["imp"])
    return out


def _score(f: dict, spec) -> float:
    if spec is None:
        return 0.0
    if isinstance(spec, str):
        return float(f.get(spec, 0.0))
    return sum(float(w) * float(f.get(k, 0.0)) for k, w in spec.items())


def apply_gate(lines: list[dict], facts: dict, gate: dict) -> list[dict]:
    g = {"min_words": 0, "skip_bc": False, "importance": 0, "unknown_cap": None, "score": None, "thresh": None,
         "rank_window": 0, "rank_top": 1.0, "min_gap_s": 0.0,
         "listen_rule": False, "listen_run_s": 45.0, "listen_loud_z": 1.0, "max_dur_s": None, **gate}
    out = copy.deepcopy(lines)
    spoken: list[float] = []
    hist: list[float] = []
    for ln in out:
        if ln.get("source") != "server" or ln.get("kind") not in ("response", "nudge"):
            continue
        f = facts["feat"].get((ln.get("turn_uid"), round(float(ln["at_s"]), 3)), {})
        s = _score(f, g["score"])
        rank_ok = True
        if g["rank_window"]:
            w = hist[-int(g["rank_window"]):]
            if w:
                rank_ok = sum(1 for x in w if x < s) / len(w) >= 1.0 - float(g["rank_top"])
        hist.append(s)
        if not ln.get("fires"):
            continue
        utt = ln.get("utterance_text") or ""
        keep = True
        if g["min_words"] and regate._words(utt) < g["min_words"]:
            keep = False
        elif g["skip_bc"] and is_backchannel_text(utt):
            keep = False
        else:
            imp = float(ln.get("importance") if ln.get("importance") is not None else 100)
            if g["unknown_cap"] is not None and f.get("unknown"):
                imp = min(imp, float(g["unknown_cap"]))
            if imp < g["importance"]:
                keep = False
            elif g["thresh"] is not None and s < float(g["thresh"]):
                keep = False
            elif g["max_dur_s"] is not None and f.get("dur", 0.0) > float(g["max_dur_s"]):
                keep = False             # long phone turns are often merged voices: the line may coach the wrong person
            elif g["listen_rule"] and f.get("coach_interrupt") and not (
                    f.get("self") and (f.get("wearer_run", 0.0) >= float(g["listen_run_s"])
                                       or f.get("loud_z", 0.0) >= float(g["listen_loud_z"])
                                       or f.get("heat_lex", 0.0) >= 1 or f.get("gap_short"))):
                keep = False             # judge J: listen/pause cues only fit a wearer who is overlapping, monologuing or heated
            elif not rank_ok:
                keep = False
            elif g["min_gap_s"] and spoken and ln["at_s"] - spoken[-1] < g["min_gap_s"]:
                keep = False
        ln["fires"] = keep
        ln["speak"] = keep
        if keep:
            spoken.append(ln["at_s"])
    return out


def load(split: str, names: list[str] | None = None) -> list[tuple[dict, dict | None, dict]]:
    if names is None:
        if split == "yt":
            names = sorted(p.parent.name for p in (RECORDINGS / "work").glob("yt_*/run.json"))
        else:
            names = cs.split_names(split, RECORDINGS / "landscape" / "splits.json")
    pairs = cs.load_bundles(RECORDINGS / "work", RECORDINGS / "inbox", names)
    out = []
    for b, prov in pairs:
        b = cs.rescore(b)
        out.append((b, prov, {"feat": line_features(b)}))
    return out


def evaluate(prepared, gate: dict) -> dict:
    items = []
    for bundle, prov, facts in prepared:
        sc = bundle["score"]
        lines = apply_gate(sc["lines"], facts, gate)
        mom = score_mod.moments(bundle, lines, sc["moments"]["window_s"], pre_s=sc["moments"]["pre_s"],
                                anchor=sc["moments"]["anchor"])
        items.append(cs.item_metrics({**bundle, "score": {**sc, "lines": lines, "moments": mom}}, prov))
    m = cs.landscape_metrics(items)
    m["spoken_lines"] = sum(i["server_spoken"] for i in items)
    return m


def feature_table(prepared) -> list[dict]:
    """Per-feature: mean value on server lines that land inside a moment window vs outside."""
    rows = []
    near_all, far_all = [], []
    for bundle, _prov, facts in prepared:
        mom = bundle["score"]["moments"]
        anchors = [float(it.get("anchor_t", it["t"])) for it in mom.get("items") or []]
        pre, post = float(mom["pre_s"]), float(mom["window_s"])
        for ln in bundle["score"]["lines"]:
            f = facts["feat"].get((ln.get("turn_uid"), round(float(ln["at_s"]), 3)))
            if f is None or regate._words(ln.get("utterance_text")) < 3:
                continue
            near = any(a - pre <= ln["at_s"] <= a + post for a in anchors)
            (near_all if near else far_all).append(f)
    for k in LIVE_FEATURES + TRUTH_FEATURES:
        a = [f[k] for f in near_all]
        b = [f[k] for f in far_all]
        rows.append({"feature": k, "near_mean": round(statistics.mean(a), 3) if a else None,
                     "far_mean": round(statistics.mean(b), 3) if b else None, "n_near": len(a), "n_far": len(b),
                     "truth_only": k in TRUTH_FEATURES})
    return rows


def _auc(pairs: list[tuple[bool, float]]) -> tuple[float | None, int]:
    pos = [s for y, s in pairs if y]
    neg = [s for y, s in pairs if not y]
    if not pos or not neg:
        return None, 0
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return c / (len(pos) * len(neg)), len(pos) * len(neg)


def auc_table(prepared, specs: dict) -> dict:
    """{name: (pooled AUC, within-session AUC)} of each score spec for "this
    server line (answering a >= 3-word turn) lands inside a moment window".
    Threshold-free, so far less noisy than a gate's hit count on 42 moments;
    within-session pools only same-session pairs (removes the confound that
    heated sessions have both more moments and different baselines)."""
    rows = []
    for bundle, _prov, facts in prepared:
        mom = bundle["score"]["moments"]
        anchors = [float(it.get("anchor_t", it["t"])) for it in mom.get("items") or []]
        pre, post = float(mom["pre_s"]), float(mom["window_s"])
        for ln in bundle["score"]["lines"]:
            f = facts["feat"].get((ln.get("turn_uid"), round(float(ln["at_s"]), 3)))
            if f is None or regate._words(ln.get("utterance_text")) < 3:
                continue
            rows.append((bundle["name"], any(a - pre <= ln["at_s"] <= a + post for a in anchors), f))
    out = {}
    for name, spec in specs.items():
        pooled = _auc([(y, _score(f, spec)) for _, y, f in rows])[0]
        num = den = 0.0
        for n in {r[0] for r in rows}:
            a, w = _auc([(y, _score(f, spec)) for nm, y, f in rows if nm == n])
            if a is not None:
                num += a * w
                den += w
        out[name] = (round(pooled, 3) if pooled is not None else None, round(num / den, 3) if den else None)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "held_out", "yt"])
    ap.add_argument("--gate", action="append", help="gate JSON (repeatable); 'B' = the shipped B gate")
    ap.add_argument("--features", action="store_true")
    a = ap.parse_args(argv)
    prepared = load(a.split)
    if a.features:
        for r in feature_table(prepared):
            print(json.dumps(r))
    for gs in a.gate or []:
        g = B_GATE if gs == "B" else json.loads(gs)
        print(json.dumps({"gate": g, "metrics": evaluate(prepared, g)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
