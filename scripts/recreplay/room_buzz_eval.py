"""Offline evaluation of identity-free / identity-soft buzz designs over phone
replay outputs (worktree recordingReplay.ts with the per-second heat log).

Everything a design reads is live-available on the phone: the instant-tier
heat windows (fastLoop.tickHeat, 1 Hz, 2 s windows), the window's level
(rms dBFS of the same 2 s of PCM), and per-turn identity (cluster label,
isSelf, labeler selfScore). Truth is only used for scoring.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from bisect import insort
from pathlib import Path

import numpy as np
import soundfile as sf

R = Path("/Users/sagearbor/projects/githubs/mindshift/tmp/recordings")
WIN_PRE, WIN_POST = 1.5, 6.0


def src_dir(n):
    w = R / "work" / n
    return w if (w / "audio16k.wav").exists() else R / "ot/X/X-Y0" / n


_cache: dict = {}


def load(n, run_dir: Path):
    key = (n, str(run_dir))
    if key in _cache:
        return _cache[key]
    ph = json.loads((run_dir / f"{n}.json").read_text())
    pcm, sr = sf.read(str(src_dir(n) / "audio16k.wav"), dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    secs = []
    for w in ph.get("heat") or []:
        t = float(w["t"])
        a, b = int(max(0, (t - 2) * sr)), int(t * sr)
        seg = pcm[a:b]
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if len(seg) else 0.0
        lvl = 20 * math.log10(rms) if rms > 1e-9 else -120.0
        secs.append({"t": t, "heat": float(w["score"]), "voiced": int(w["voicedFrames"]), "lvl": lvl})
    turns = [t for t in ph["turns"] if t.get("kind", "primary") == "primary"]
    ll = ph.get("labelLog") or []
    if len(ll) == len(turns):
        for t, l in zip(turns, ll):
            t["selfScore"] = l.get("selfScore")
    else:
        for t in turns:
            t["selfScore"] = 0.9 if t.get("isSelf") else None
    ann_p = R / "inbox" / n / f"{n}.annotation.groundtruth.json"
    if not ann_p.exists():
        ann_p = R / "inbox" / n / f"{n}.annotation.gemini-2.5-flash.json"
    ann =json.loads(ann_p.read_text()) if ann_p.exists() else {"segments": [], "coach_moments": []}
    corp_p = R / "inbox" / n / f"{n}.corpus.json"
    corp = json.loads(corp_p.read_text()) if corp_p.exists() else {}
    dur = float(ph["durationSec"])
    group = corp.get("group")
    if group is None:
        group = (json.loads((R / "inbox" / n / f"{n}.yt.json").read_text()).get("group")
                 if (R / "inbox" / n / f"{n}.yt.json").exists() else "open-web")
    wid = corp.get("wearer_id")
    segs = ann.get("segments") or []
    # Targets: heated moments that need no identity to define.
    tg_any, tg_w, lively = [], [], []
    for m in ann.get("coach_moments") or []:
        t = float(m["t"])
        rule = m.get("rule")
        if rule is None and m.get("kind") == "warning":
            # open-web (Gemini) moments carry no rule: every warning counts
            iv = (t - WIN_PRE, t + WIN_POST)
            tg_any.append(iv); tg_w.append(iv)
        elif rule == "conflict-peak":
            import re
            mm = re.search(r"for (\d+) s", m.get("what_happened", ""))
            d = float(mm.group(1)) if mm else 0.0
            iv = (t - WIN_PRE, t + max(WIN_POST, d))
            tg_any.append(iv); tg_w.append(iv)
        elif rule == "raised-voice":
            cand = [s for s in segs if s["start"] - 0.05 <= t < s["end"] + 0.05 and s["speaker"] == m.get("for_speaker")]
            e = max((s["end"] for s in cand), default=t)
            iv = (e - WIN_PRE, e + WIN_POST)
            tg_any.append(iv); tg_w.append(iv)
        elif rule == "talks-over":
            cand = [s for s in segs if abs(s["start"] - t) <= 0.05 and s["speaker"] == m.get("for_speaker")]
            e = max((s["end"] for s in cand), default=t)
            lively.append((e - WIN_PRE, e + WIN_POST))
    gt = ann_p.name.endswith("groundtruth.json")
    # Gemini marks ~19% of open-web segments "raised"; only its "shouting" counts there.
    loud = ("raised", "shouting") if gt else ("shouting",)
    for s in segs:
        if (s.get("vocal") or {}).get("volume") in loud:
            tg_any.append((s["end"] - WIN_PRE, s["end"] + WIN_POST))
    tg_any = merge(tg_any)
    tg_w = merge(tg_w)
    # per-turn truth: wearer?
    for t in turns:
        best, ov = None, 0.0
        for s in segs:
            o = min(t["endTime"], s["end"]) - max(t["startTime"], s["start"])
            if o > ov:
                best, ov = s["speaker"], o
        t["truth_wearer"] = None if (best is None or wid is None) else (best == wid)
    item = {"name": n, "dur": dur, "group": group, "secs": secs, "turns": turns,
            "tg_any": tg_any, "tg_w": tg_w, "lively": lively, "haptics": ph.get("haptics") or []}
    _cache[key] = item
    return item


def merge(ivs, gap=8.0):
    ivs = sorted(ivs)
    out = []
    for a, b in ivs:
        if out and a - out[-1][0] < gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


class RunMed:
    def __init__(self):
        self.v = []

    def add(self, x):
        insort(self.v, x)

    def med(self):
        return self.v[len(self.v) // 2] if self.v else None

    def __len__(self):
        return len(self.v)


def psoft(score, c, k):
    if score is None:
        return 0.0
    return 1 / (1 + math.exp(-(score - c) / k))


def design(item, p) -> list[float]:
    """Return buzz times. p['kind'] picks the family."""
    kind = p["kind"]
    out: list[float] = []
    last = -1e9
    C = p.get("cooldown", 60)
    if kind == "baseline":
        return [h["atSec"] for h in item["haptics"] if h.get("code") in (None, "H", "C", "A")]
    if kind in ("room", "room_soft"):
        med = RunMed()
        k = p.get("k", 3)
        hist = []
        run = 0
        turns = item["turns"]
        for s in item["secs"]:
            ok_speech = s["voiced"] >= p.get("vmin", 10)
            base = med.med() if len(med) >= p.get("warm", 20) else None
            over = (s["lvl"] - base) if base is not None else None
            if ok_speech:
                med.add(s["lvl"])
            hist.append((s["heat"] if ok_speech else 0.0, over if over is not None else -99))
            hk = [h for h, _ in hist[-k:]]
            ok = [o for _, o in hist[-k:]]
            e = sum(hk) / len(hk)
            o = sum(ok) / len(ok)
            cond = e >= p["h"] and (p.get("x") is None or o >= p["x"]) and (p.get("absmin") is None or s["lvl"] >= p["absmin"])
            if kind == "room_soft":
                # weight by P(self) of the turn(s) overlapping the last k s (closed turns only: causal)
                tt = s["t"]
                ws = [psoft(t["selfScore"], p["c"], p["ks"]) for t in turns if t["endTime"] <= tt + 1.0 and t["endTime"] >= tt - k - 1]
                w = max(ws) if ws else p.get("w0", 0.0)
                cond = cond and w >= p["pmin"]
            run = run + 1 if cond else 0
            if run >= p.get("sustain", 1) and s["t"] - last >= C:
                out.append(s["t"]); last = s["t"]
        return out
    if kind in ("turn", "turn_soft", "prox"):
        cl = {}
        allmed = RunMed()
        lvl_by_cluster = {}
        for t in item["turns"]:
            rms = (t.get("prosody") or {}).get("rms_dbfs")
            heat = t.get("instantHeat")
            spk = t["speaker"]
            dur = t["endTime"] - t["startTime"]
            if rms is None or heat is None:
                continue
            ref = cl.get(spk) if p.get("ref") == "cluster" else allmed
            if ref is None:
                ref = cl.setdefault(spk, RunMed())
            base = ref.med() if len(ref) >= p.get("warm", 3) else None
            over = rms - base if base is not None else None
            ref.add(rms)
            if p.get("ref") == "cluster":
                pass
            else:
                allmed.add(rms)
            lvl_by_cluster.setdefault(spk, []).append(rms)
            if kind == "turn_soft":
                gate = psoft(t.get("selfScore"), p["c"], p["ks"]) >= p["pmin"]
            elif kind == "prox":
                # loudest cluster so far (median of its turn levels, >= 3 turns) stands in for the wearer
                cands = {k2: statistics.median(v) for k2, v in lvl_by_cluster.items() if len(v) >= 3 and k2 != "Unknown"}
                gate = bool(cands) and spk == max(cands, key=cands.get) and cands[spk] - statistics.median(
                    [x for v in lvl_by_cluster.values() for x in v]) >= p.get("margin", 2.0)
            else:
                gate = True
            cond = gate and heat >= p["h"] and dur >= p.get("dmin", 1.0) and (
                p.get("x") is None or (over is not None and over >= p["x"]))
            at = t["endTime"] + 0.5
            if cond and at - last >= C:
                out.append(at); last = at
        return out
    raise ValueError(kind)


def hits(buzz, targets):
    return sum(1 for a, b in targets if any(a <= z <= b for z in buzz))


def evaluate(items, p, n_shift=40):
    agg = {"calm_s": 0, "heat_s": 0, "calm_b": 0, "heat_b": 0, "tg_any": 0, "hit_any": 0, "tg_w": 0,
           "hit_w": 0, "ch_any": 0.0, "ch_w": 0.0, "lively": 0, "hit_lively": 0, "raised_w": 0, "hit_raised_w": 0}
    per = []
    for it in items:
        bz = design(it, p)
        d = it["dur"]
        if it["group"] == "calm":
            agg["calm_s"] += d; agg["calm_b"] += len(bz)
        else:
            agg["heat_s"] += d; agg["heat_b"] += len(bz)
        ha, hw = hits(bz, it["tg_any"]), hits(bz, it["tg_w"])
        agg["tg_any"] += len(it["tg_any"]); agg["hit_any"] += ha
        agg["tg_w"] += len(it["tg_w"]); agg["hit_w"] += hw
        agg["lively"] += len(it["lively"]); agg["hit_lively"] += hits(bz, it["lively"])
        # chance: same buzz train circularly shifted
        if bz and (it["tg_any"] or it["tg_w"]):
            ca = cw = 0.0
            for i in range(1, n_shift + 1):
                off = d * i / (n_shift + 1)
                sh = [(z + off) % d for z in bz]
                ca += hits(sh, it["tg_any"]); cw += hits(sh, it["tg_w"])
            agg["ch_any"] += ca / n_shift; agg["ch_w"] += cw / n_shift
        per.append((it["name"], it["group"], len(bz), ha, len(it["tg_any"])))
    m = {
        "calm_buzz_h": round(agg["calm_b"] / (agg["calm_s"] / 3600), 1) if agg["calm_s"] else None,
        "heated_buzz_h": round(agg["heat_b"] / (agg["heat_s"] / 3600), 1) if agg["heat_s"] else None,
        "heated_targets": agg["tg_any"], "heated_hits": agg["hit_any"],
        "buzz_hit_rate": round(agg["hit_any"] / agg["tg_any"], 3) if agg["tg_any"] else None,
        "chance_hit_rate": round(agg["ch_any"] / agg["tg_any"], 3) if agg["tg_any"] else None,
        "wearer_heated_hits": agg["hit_w"], "wearer_heated_targets": agg["tg_w"],
        "lively_hits": agg["hit_lively"], "lively_targets": agg["lively"],
        "buzzes": agg["calm_b"] + agg["heat_b"],
    }
    if m["buzz_hit_rate"] is not None:
        m["buzz_lift"] = round(m["buzz_hit_rate"] - m["chance_hit_rate"], 3)
    return m, per


def load_split(names, run_dir):
    out = []
    for n in names:
        if (run_dir / f"{n}.json").exists():
            out.append(load(n, run_dir))
        else:
            print("missing", n, file=sys.stderr)
    return out
