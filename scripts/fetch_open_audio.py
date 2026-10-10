#!/usr/bin/env python3
"""fetch_open_audio.py — openly licensed, genuinely heated, multi-speaker
conversation audio from YouTube, turned into recording-replay inbox items.

The corpora on disk (AMI, CHiME-6 dinners, most of SBCSAE) are calm; this
fills the heated end. LICENCE FIRST: a video is considered only when its
yt-dlp metadata says Creative Commons, or it comes from a PROVEN official
House/Senate channel (US federal work; scripts/annotator/licence.py).
Everything else is rejected before download and the reason logged.

Stages (each resumable; state in tmp/open_audio/manifest.json):

    tmp/venv-annot/bin/python scripts/fetch_open_audio.py search      # metadata + licence gate
    tmp/venv-annot/bin/python scripts/fetch_open_audio.py download    # audio only, accepted items
    tmp/venv-annot/bin/python scripts/fetch_open_audio.py screen      # loudness -> hottest window(s), trim
    tmp/venv-annot/bin/python scripts/fetch_open_audio.py flash       # cheap Gemini flash heat/overlap screen
    tmp/venv/bin/python       scripts/fetch_open_audio.py evidence    # server tone model arousal + loudness vs speaker baseline
    tmp/venv-annot/bin/python scripts/fetch_open_audio.py finalize --model gemini-3.1-pro-preview --keep 15

finalize writes tmp/recordings/inbox/yt_<id>/{yt_<id>.m4a, yt_<id>.notes.txt,
yt_<id>.annotation.<model>.json} for the hottest licence-clean items and the
full attribution + heat evidence into the manifest. Audio and annotations
live only under gitignored tmp/ and are never committed. Gemini spend shares
the annotator's ledger and its hard cap.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "server"))

from annotator import licence  # noqa: E402

ROOT = REPO / "tmp/open_audio"
RAW = ROOT / "raw"
CLIPS = ROOT / "clips"
MANIFEST = ROOT / "manifest.json"
INBOX = REPO / "tmp/recordings/inbox"
YTDLP = str(Path(sys.executable).parent / "yt-dlp")
FFMPEG = "/opt/homebrew/bin/ffmpeg"
CC_FILTER = "&sp=EgIwAQ%253D%253D"   # YouTube search filter: Creative Commons only

# Conversational heat first (people talking over each other), speeches last.
CC_QUERIES = [
    "city council meeting heated argument", "city council meeting gets heated", "council meeting shouting match",
    "school board meeting heated", "school board meeting parents angry", "county commission meeting heated",
    "town hall meeting heated exchange", "public comment heated city council", "zoning board meeting argument",
    "council member argue mayor", "heated debate podcast", "debate gets heated", "argument caught on camera",
    "roommate argument", "family argument vlog", "heated discussion panel", "parliamentary debate final round",
    "british parliamentary debate grand final", "debate tournament final round", "town meeting argument residents",
    "HOA meeting heated", "board of supervisors heated", "city commission meeting heated exchange",
    "first amendment audit council meeting heated",
]
FED_QUERIES = [
    "house committee hearing heated exchange", "senate hearing heated exchange", "house oversight hearing heated",
    "house judiciary committee heated exchange", "senate judiciary committee heated", "house hearing shouting",
]
# Official committee channels searched directly (still licence-gated per video:
# of ~25 House/Senate handles probed on 2026-10-10, only HASC's channel text
# carries the .gov link the gate requires).
FED_CHANNEL_SEARCHES = [("HouseArmedServices", "heated exchange"), ("HouseArmedServices", "hearing")]
MIN_S, MAX_S = 180, 4 * 3600


def log(*a):
    print(*a, flush=True)


def load() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "candidates": {}, "rejected": {}, "channels": {}}


def save(m: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=1, ensure_ascii=False))
    tmp.replace(MANIFEST)


def ytdlp_json(args: list[str], timeout: int = 180) -> dict | None:
    r = subprocess.run([YTDLP, "-J", "--no-warnings", *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def channel_info(m: dict, channel_id: str) -> dict:
    if channel_id in m["channels"]:
        return m["channels"][channel_id]
    d = ytdlp_json(["--flat-playlist", "--playlist-items", "0", f"https://www.youtube.com/channel/{channel_id}"]) or {}
    info = {"channel": d.get("channel"), "description": (d.get("description") or "")[:3000],
            "uploader_id": d.get("uploader_id")}
    m["channels"][channel_id] = info
    return info


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def cmd_search(args) -> int:
    m = load()
    seen = set(m["candidates"]) | set(m["rejected"])
    queries = [(q, True) for q in CC_QUERIES] + [(q, False) for q in FED_QUERIES]
    queries += [(f"@{h}/search?query={q}", False) for h, q in FED_CHANNEL_SEARCHES]
    for q, cc in queries:
        if q.startswith("@"):
            url = "https://www.youtube.com/" + q.replace(" ", "+")
        else:
            url = "https://www.youtube.com/results?search_query=" + q.replace(" ", "+") + (CC_FILTER if cc else "")
        d = ytdlp_json(["--flat-playlist", "--playlist-end", str(args.per_query), url]) or {}
        ents = d.get("entries") or []
        log(f"[{'CC' if cc else 'fed'}] {q!r}: {len(ents)} results")
        for e in ents:
            vid = e.get("id")
            if not vid or vid in seen:
                continue
            seen.add(vid)
            dur = e.get("duration") or 0
            if dur and not (MIN_S <= dur <= MAX_S):
                m["rejected"][vid] = {"title": e.get("title"), "channel": e.get("channel"),
                                      "why": f"duration {dur}s outside {MIN_S}-{MAX_S}s", "query": q}
                continue
            if not cc and not q.startswith("@") and not any(w in (e.get("channel") or "").lower() for w in ("house", "senate", "committee")):
                m["rejected"][vid] = {"title": e.get("title"), "channel": e.get("channel"), "query": q,
                                      "why": "not CC-filtered and channel name is not House/Senate/committee"}
                continue
            info = ytdlp_json(["--skip-download", f"https://www.youtube.com/watch?v={vid}"])
            if not info:
                m["rejected"][vid] = {"title": e.get("title"), "why": "metadata unavailable", "query": q}
                continue
            meta = {k: info.get(k) for k in ("id", "title", "channel", "channel_id", "channel_url", "uploader",
                                              "uploader_id", "uploader_url", "upload_date", "duration", "license",
                                              "webpage_url", "language")}
            if not licence.is_creative_commons(meta) and meta.get("channel_id"):
                ch = channel_info(m, meta["channel_id"])
                meta["channel_description"] = ch.get("description")
            keep, kind, why = licence.decide(meta)
            meta.pop("channel_description", None)
            meta["query"] = q
            if not keep:
                m["rejected"][vid] = {"title": meta["title"], "channel": meta["channel"], "license": meta["license"],
                                      "url": meta["webpage_url"], "why": why, "query": q}
                continue
            if not (MIN_S <= (meta.get("duration") or 0) <= MAX_S):
                m["rejected"][vid] = {"title": meta["title"], "why": f"duration {meta.get('duration')}s", "query": q}
                continue
            meta.update({"licence_kind": kind, "licence_reason": why, "stage": "accepted"})
            m["candidates"][vid] = meta
            log(f"   + {vid} [{kind}] {meta['duration']}s {meta['channel']} | {meta['title']}")
        save(m)
    log(f"candidates {len(m['candidates'])}, rejected {len(m['rejected'])}")
    return 0


# ---------------------------------------------------------------------------
# download + screen
# ---------------------------------------------------------------------------

_HOT_WORDS = re.compile(r"heated|argu|shout|yell|scream|fight|clash|angry|anger|furious|lit\b|meltdown|freaks? out|"
                        r"rant|confront|heckl|outburst|chaos|explod|blow[s]? up|walks? out|kicked out|removed|"
                        r"tense|spar|feud|debate|vs\.?\b|versus|interrupt|parking lot", re.I)


def title_heat(c: dict) -> float:
    """Download priority (before any audio): heat words in the title, and a
    preference for 5-60 minute videos (short enough to be about the clash)."""
    hits = len(_HOT_WORDS.findall(c.get("title") or ""))
    d = c.get("duration") or 0
    return hits * 2 + (1 if 300 <= d <= 3600 else 0) - (1 if d > 3 * 3600 else 0)


# Channels whose CC flag we do not trust after reading their video lists:
# they re-post other people's debates/streams/news footage.
SKIP_CHANNELS = {
    "Hope In Christ": "re-posts other creators' debate streams",
    "Triggered Daily": "re-posts other creators' debate streams",
    "KEWL V1C": "re-cuts of other creators' livestreams",
    "Dum Dum News Channel": "news-footage re-uploads",
    "The Humanist Report": "commentary over others' footage (host voice-over)",
    "We See You!": "re-uploaded news footage",
    "Patriot Fire": "re-uploaded meeting footage with commentary",
    "Etaya TV": "re-uploaded parliament broadcast",
    "3News": "broadcaster re-post",
    "Bodycam Footage": "police bodycam compilations",
    "Real Police Stories": "police bodycam compilations",
    "Lovely News Network": "news-footage re-uploads",
    "JUST ZEKI": "news re-telling over others' footage",
}


def regate(m: dict) -> None:
    """Re-apply the licence gate (it may have tightened) + the channel skip list."""
    for vid, c in list(m["candidates"].items()):
        why = None
        if c.get("channel") in SKIP_CHANNELS:
            why = f"CC label, but skipped: channel {c['channel']!r} {SKIP_CHANNELS[c['channel']]}"
        elif c.get("licence_kind", "").startswith("CC"):
            keep, _, reason = licence.decide(c)
            if not keep:
                why = reason
        if why:
            m["rejected"][vid] = {"title": c.get("title"), "channel": c.get("channel"), "license": c.get("license"),
                                  "url": c.get("webpage_url"), "why": why, "query": c.get("query")}
            del m["candidates"][vid]


def cmd_download(args) -> int:
    m = load()
    regate(m)
    save(m)
    RAW.mkdir(parents=True, exist_ok=True)
    todo = [c for c in m["candidates"].values() if c.get("stage") == "accepted"]
    todo.sort(key=lambda c: -title_heat(c))
    if args.limit:
        todo = todo[:args.limit]
    for c in todo:
        out = RAW / f"{c['id']}.m4a"
        if not out.exists():
            # lowest-bitrate audio-only stream: plenty for speech, small on disk
            r = subprocess.run([YTDLP, "--no-warnings", "-f", "worstaudio[ext=m4a]/worstaudio/bestaudio",
                                "--no-playlist", "-o", str(RAW / f"{c['id']}.%(ext)s"),
                                c["webpage_url"]], capture_output=True, text=True, timeout=1800)
            got = sorted(RAW.glob(f"{c['id']}.*"))
            if r.returncode != 0 or not got:
                c["stage"] = "download_failed"
                c["error"] = (r.stderr or "")[-300:]
                save(m)
                continue
            if got[0] != out:
                out = got[0]
        c["raw"] = str(out.relative_to(REPO))
        c["stage"] = "downloaded"
        log(f"  downloaded {c['id']} ({out.stat().st_size / 1e6:.1f} MB)")
        save(m)
    return 0


def pcm16k(path: Path) -> np.ndarray:
    r = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                       capture_output=True, timeout=1800, check=True)
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def loudness_per_second(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    from prosody import rms_energy, rms_to_dbfs
    n = len(x) // sr
    out = np.full(n, -100.0)
    for i in range(n):
        db = rms_to_dbfs(rms_energy(x[i * sr:(i + 1) * sr]))
        out[i] = db if db is not None else -100.0
    return out


def hot_windows(db: np.ndarray, win: int, k: int = 1) -> list[dict]:
    """Rank windows by a loudness heat proxy: share of seconds at least 6 dB
    above the file's own median speech loudness (raised voices, relative so
    mic gain does not matter) + half the speech density (back-and-forth with
    few gaps). Returns up to k non-overlapping windows."""
    floor = np.percentile(db, 5)
    speech = db > floor + 6
    if speech.sum() < 10:
        return []
    med = np.median(db[speech])
    raised = (db >= med + 6) & speech
    win = min(win, len(db))
    cs_r = np.concatenate([[0], np.cumsum(raised)])
    cs_s = np.concatenate([[0], np.cumsum(speech)])
    scores = []
    for a in range(0, len(db) - win + 1, 10):
        r = (cs_r[a + win] - cs_r[a]) / win
        s = (cs_s[a + win] - cs_s[a]) / win
        scores.append((r + 0.5 * s, a, r, s))
    scores.sort(reverse=True)
    out = []
    for sc, a, r, s in scores:
        if all(abs(a - o["start"]) >= win for o in out):
            out.append({"start": int(a), "end": int(a + win), "score": round(float(sc), 4),
                        "raised_frac": round(float(r), 4), "speech_frac": round(float(s), 4)})
        if len(out) >= k:
            break
    return out


def cmd_screen(args) -> int:
    m = load()
    CLIPS.mkdir(parents=True, exist_ok=True)
    for c in list(m["candidates"].values()):
        if c.get("stage") != "downloaded":
            continue
        try:
            x = pcm16k(REPO / c["raw"])
        except Exception as exc:  # noqa: BLE001
            c["stage"], c["error"] = "decode_failed", str(exc)[:200]
            save(m)
            continue
        db = loudness_per_second(x)
        k = 2 if len(db) > 1800 else 1
        wins = hot_windows(db, args.window_s, k=k)
        if not wins:
            c["stage"] = "no_speech"
            save(m)
            continue
        c["windows"] = []
        for j, w in enumerate(wins):
            clip = CLIPS / f"{c['id']}_w{j}.m4a"
            subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", str(w["start"]), "-i", str(REPO / c["raw"]),
                            "-t", str(w["end"] - w["start"]), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac",
                            "-b:a", "64k", str(clip)], check=True, timeout=600)
            w["clip"] = str(clip.relative_to(REPO))
            c["windows"].append(w)
        c["stage"] = "screened"
        log(f"  screened {c['id']}: " + ", ".join(f"{w['start']}-{w['end']}s score {w['score']}" for w in wins))
        save(m)
    return 0


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def overlap_stats(ann: dict) -> dict:
    """Overlap % = seconds where >= 2 different annotated speakers talk at once
    / seconds with any speech (from the annotation's segment times)."""
    dur = int(np.ceil(ann["audio"].get("duration_s") or max([s["end"] for s in ann["segments"]] or [0]))) + 1
    hop = 10  # 0.1 s
    n = dur * hop
    by = {}
    for s in ann["segments"]:
        a, b = int(s["start"] * hop), int(s["end"] * hop)
        by.setdefault(s["speaker"], np.zeros(n, bool))[a:b] = True
    if not by:
        return {"overlap_pct": 0.0, "speech_s": 0.0, "speakers_active": 0}
    stack = np.stack(list(by.values()))
    cnt = stack.sum(0)
    speech = float((cnt >= 1).sum()) / hop
    ov = float((cnt >= 2).sum()) / hop
    talk = {k: float(v.sum()) / hop for k, v in by.items()}
    active = sum(1 for v in talk.values() if v >= 10)
    # Annotators often write overlapping talk as back-to-back segments, so also
    # count segments the annotator explicitly marked as overlapping someone.
    marked = sum(1 for s in ann["segments"] if s.get("overlaps_with"))
    return {"overlap_pct": round(100 * ov / speech, 1) if speech else 0.0, "speech_s": speech,
            "overlap_marked_pct": round(100 * marked / len(ann["segments"]), 1) if ann["segments"] else 0.0,
            "speakers_active": active, "talk_s": talk}


def heat_stats(ann: dict) -> dict:
    segs = ann["segments"]
    tot = sum(s["end"] - s["start"] for s in segs) or 1.0
    hot = sum(s["end"] - s["start"] for s in segs
              if (s.get("vocal") or {}).get("intensity") is not None and s["vocal"]["intensity"] >= 2
              and s["vocal"].get("emotion") in ("angry", "frustrated", "sarcastic", "excited", "anxious"))
    neg = sum(1 for s in segs if s.get("text_emotion") in ("hostile", "negative"))
    return {"overall_heat": ann["summary"].get("overall_heat"), "hot_speech_pct": round(100 * hot / tot, 1),
            "hostile_or_negative_segments": neg, "segments": len(segs),
            "interruptions": sum(1 for e in ann.get("events") or [] if e["type"] == "interruption"),
            "escalations": sum(1 for e in ann.get("events") or [] if e["type"] == "escalation")}


def cmd_flash(args) -> int:
    import annotate_audio as aa
    from annotator import core
    from annotator.gemini import Gemini
    m = load()
    ledger = core.Ledger(aa.DEFAULT_LEDGER, cap_usd=args.cap)
    gem = Gemini(ledger)
    for c in list(m["candidates"].values()):
        if c.get("stage") != "screened":
            continue
        for w in c["windows"]:
            if w.get("flash"):
                continue
            clip = REPO / w["clip"]
            try:
                r = aa.annotate_file(gem, clip, args.model, window_s=600, overlap_s=30)
            except core.BudgetExceeded as exc:
                log(f"STOP: {exc}")
                save(m)
                return 2
            except Exception as exc:  # noqa: BLE001
                w["flash"] = {"error": str(exc)[:200]}
                continue
            dst = aa.out_path(clip, args.model)
            dst.write_text(json.dumps(r["annotation"], indent=1, ensure_ascii=False))
            w["flash"] = {"model": args.model, "usd": round(r["usd"], 4), **heat_stats(r["annotation"]),
                          **{k: v for k, v in overlap_stats(r["annotation"]).items() if k != "talk_s"}}
            log(f"  {c['id']} w{c['windows'].index(w)}: heat {w['flash']['overall_heat']} "
                f"hot {w['flash']['hot_speech_pct']}% overlap {w['flash']['overlap_pct']}% "
                f"speakers {w['flash']['speakers_active']}  ${r['usd']:.3f} (total ${ledger.total:.2f})")
            save(m)
        c["stage"] = "flashed"
        save(m)
    return 0


def hot_rank(w: dict) -> float:
    f = w.get("flash") or {}
    if "overall_heat" not in f:
        return -1
    ov = max(f.get("overlap_pct", 0), f.get("overlap_marked_pct", 0))
    return ((f.get("overall_heat") or 0) * 10 + f.get("hot_speech_pct", 0) * 0.3 + min(ov, 30) * 0.5
            + min(f.get("interruptions", 0), 15) * 0.5 + (0 if f.get("speakers_active", 0) >= 2 else -100))


# ---------------------------------------------------------------------------
# evidence (server tone model + loudness vs each speaker's own baseline)
# ---------------------------------------------------------------------------

def measured_evidence(clip: Path, ann: dict, use_tone: bool) -> dict:
    x = pcm16k(clip)
    sr = 16000
    from prosody import rms_energy, rms_to_dbfs
    per = {}
    rows = []
    for s in ann["segments"]:
        a, b = int(s["start"] * sr), int(s["end"] * sr)
        if b - a < sr // 2:
            continue
        db = rms_to_dbfs(rms_energy(x[a:b]))
        if db is None:
            continue
        rows.append((s["speaker"], s["start"], s["end"], db, a, b))
        per.setdefault(s["speaker"], []).append(db)
    base = {k: float(np.median(v)) for k, v in per.items()}
    raised = [r for r in rows if r[3] - base[r[0]] >= 6.0]
    out = {"loud_segments": len(rows),
           "raised_vs_own_baseline_pct": round(100 * len(raised) / len(rows), 1) if rows else 0.0,
           "max_raise_db": round(max([r[3] - base[r[0]] for r in rows] or [0]), 1)}
    if use_tone:
        import tone_id
        thr = tone_id.escalation_threshold()
        ar = {}
        seq = []
        for spk, s0, s1, _, a, b in rows:
            try:
                res = tone_id.classify_pcm(x[a:min(b, a + 8 * sr)], sr)
            except Exception as exc:  # noqa: BLE001
                out["tone_error"] = str(exc)[:200]
                break
            ar.setdefault(spk, []).append(res["arousal"])
            seq.append((spk, res["arousal"]))
        if seq:
            bl = {k: float(np.median(v)) for k, v in ar.items()}
            esc = [1 for spk, a in seq if a - bl[spk] >= thr]
            out.update({"tone_backend": tone_id.backend(), "arousal_mean": round(float(np.mean([a for _, a in seq])), 3),
                        "escalating_segments_pct": round(100 * len(esc) / len(seq), 1),
                        "escalation_threshold": thr})
    return out


def cmd_evidence(args) -> int:
    import os
    os.environ.setdefault("MINDSHIFT_TONE_CACHE", str(REPO.parent.parent.parent / "server/.tone_cache")
                          if (REPO.parent.parent.parent / "server/.tone_cache").exists()
                          else str(REPO / "server/.tone_cache"))
    import annotate_audio as aa
    m = load()
    for c in m["candidates"].values():
        for w in c.get("windows") or []:
            f = w.get("flash") or {}
            if w.get("evidence") or not f.get("model") or (f.get("overall_heat") or 0) < args.min_heat:
                continue
            ann_p = aa.out_path(REPO / w["clip"], f["model"])
            if not ann_p.exists():
                continue
            ann = json.loads(ann_p.read_text())
            try:
                w["evidence"] = measured_evidence(REPO / w["clip"], ann, use_tone=not args.no_tone)
                w["evidence"]["segments_from"] = f["model"]
            except Exception as exc:  # noqa: BLE001
                w["evidence"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            log(f"  {c['id']}: {w['evidence']}")
            save(m)
    return 0


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

# Server tone model (odyssey_dim) mean arousal on reference clips, measured
# 2026-10-10 with the same code: AMI meeting 0.34, CHiME-6 dinner 0.48,
# low-conflict CONFER TV debate 0.63, heated CONFER debate 0.83.
AROUSAL_HEATED = 0.55
RAISED_PCT_HEATED = 10.0


def measured_heated(ev: dict) -> tuple[bool, str]:
    if not ev or "error" in ev:
        return False, f"no measured evidence ({(ev or {}).get('error', 'not computed')})"
    ar, raised = ev.get("arousal_mean"), ev.get("raised_vs_own_baseline_pct", 0)
    bits = []
    if ar is not None:
        bits.append(f"tone-model arousal {ar:.2f} (calm refs 0.34-0.48, heated debate 0.83)")
    bits.append(f"{raised}% of segments >=6 dB above the speaker's own median (max +{ev.get('max_raise_db')} dB)")
    ok = (ar is not None and ar >= AROUSAL_HEATED) or raised >= RAISED_PCT_HEATED
    return ok, ("" if ok else "NOT heated by measurement: ") + "; ".join(bits)


def wearer(ann: dict) -> tuple[str, str]:
    """Most-involved speaker: talk time + 2x time spent in overlap + 5 s per
    interruption/escalation event they are part of."""
    ov = overlap_stats(ann)
    hop = 10
    dur = int(np.ceil(ann["audio"].get("duration_s") or 0)) + 1
    masks = {}
    for s in ann["segments"]:
        masks.setdefault(s["speaker"], np.zeros(dur * hop, bool))[int(s["start"] * hop):int(s["end"] * hop)] = True
    cnt = np.sum(list(masks.values()), axis=0) if masks else np.zeros(1)
    score = {}
    for k, mk in masks.items():
        score[k] = ov["talk_s"].get(k, 0) + 2 * float((mk & (cnt >= 2)).sum()) / hop
    for e in ann.get("events") or []:
        if e["type"] in ("interruption", "escalation"):
            for k in e.get("speakers") or []:
                score[k] = score.get(k, 0) + 5
    best = max(score, key=score.get)
    desc = next((s.get("voice_description") for s in ann["speakers"] if s["id"] == best), None) or ""
    return best, desc


def cmd_finalize(args) -> int:
    import annotate_audio as aa
    from annotator import core
    from annotator.gemini import Gemini
    m = load()
    ledger = core.Ledger(aa.DEFAULT_LEDGER, cap_usd=args.cap)
    gem = Gemini(ledger)
    pool = []
    for c in m["candidates"].values():
        for j, w in enumerate(c.get("windows") or []):
            f = w.get("flash") or {}
            fp = aa.out_path(REPO / w["clip"], f["model"]) if f.get("model") else None
            if fp and fp.exists():  # recompute with the current stats code
                fa = json.loads(fp.read_text())
                f.update(heat_stats(fa))
                f.update({k: v for k, v in overlap_stats(fa).items() if k != "talk_s"})
                f["timing_untrustworthy"] = "TIMING UNTRUSTWORTHY" in (fa["annotator"].get("notes") or "")
            if hot_rank(w) >= 0:
                pool.append((hot_rank(w), c, j, w))
    pool.sort(key=lambda t: -t[0])
    chosen, seen = [], set()
    for rank, c, j, w in pool:
        if c["id"] in seen:
            continue
        f = w["flash"]
        if (f.get("overall_heat") or 0) < args.min_heat and not args.allow_calm:
            continue
        if f.get("timing_untrustworthy"):
            continue
        ok_ev, why_ev = measured_heated(w.get("evidence") or {})
        if not ok_ev and not args.allow_calm:
            log(f"  skip {c['id']} w{j}: flash heat {f.get('overall_heat')} but {why_ev}")
            continue
        w["why_heated"] = (f"Gemini flash overall_heat {f.get('overall_heat')}, {f.get('hot_speech_pct')}% of speech "
                           f"angry/frustrated/excited at intensity>=2, {f.get('interruptions')} interruptions; "
                           f"measured: {why_ev}")
        seen.add(c["id"])
        chosen.append((c, j, w))
        if len(chosen) >= args.keep:
            break
    log(f"finalising {len(chosen)} item(s)")
    for c, j, w in chosen:
        name = f"yt_{c['id']}"
        d = INBOX / name
        d.mkdir(parents=True, exist_ok=True)
        audio = d / f"{name}.m4a"
        if not audio.exists():
            subprocess.run(["cp", str(REPO / w["clip"]), str(audio)], check=True)
        dst = aa.out_path(audio, args.model)
        if dst.exists():
            ann = json.loads(dst.read_text())
            usd = (w.get("final") or {}).get("usd", 0.0)
        else:
            try:
                r = aa.annotate_file(gem, audio, args.model, window_s=600, overlap_s=30)
            except core.BudgetExceeded as exc:
                log(f"STOP: {exc}")
                save(m)
                return 2
            except Exception as exc:  # noqa: BLE001
                log(f"  FAILED {name}: {exc}")
                continue
            ann, usd = r["annotation"], r["usd"]
            dst.write_text(json.dumps(ann, indent=1, ensure_ascii=False))
        # keep the flash screen annotation too (a second annotator for the pipeline to compare)
        flash_src = aa.out_path(REPO / w["clip"], w["flash"]["model"])
        flash_dst = aa.out_path(audio, w["flash"]["model"])
        if flash_src.exists() and not flash_dst.exists():
            flash_dst.write_text(flash_src.read_text())
        spk, desc = wearer(ann)
        lic = c["licence_kind"]
        ymd = c.get("upload_date") or ""
        date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}" if len(ymd) == 8 else ymd
        (d / f"{name}.notes.txt").write_text(
            f"who: I'm {spk} — {desc} (chosen as the most-involved speaker; not a real wearer)\n"
            f"setting: {c['title']} — {c['channel']}; open-web clip {w['start'] // 60}:{w['start'] % 60:02d}-"
            f"{w['end'] // 60}:{w['end'] % 60:02d} of the original\n"
            f"phone: none — meeting/broadcast/online video audio, not a phone recording\n"
            f"source: open web, licence {lic}\n"
            f"url: {c['webpage_url']}\n"
            f"attribution: \"{c['title']}\" by {c['channel']} ({date}), {lic}, {c['webpage_url']}\n")
        hs, ov = heat_stats(ann), overlap_stats(ann)
        w["final"] = {"inbox": str(d.relative_to(REPO)), "audio": str(audio.relative_to(REPO)),
                      "annotation": str(dst.relative_to(REPO)), "model": args.model, "usd": round(usd, 4),
                      "wearer": spk, **hs, "overlap_pct": ov["overlap_pct"],
                      "overlap_marked_pct": ov["overlap_marked_pct"], "speakers_active": ov["speakers_active"],
                      "why_heated": w.get("why_heated")}
        c["stage"] = "final"
        c["chosen_window"] = j
        log(f"  {name}: heat {hs['overall_heat']} overlap {ov['overlap_pct']}% wearer {spk}  ${usd:.3f} "
            f"(total ${ledger.total:.2f})")
        save(m)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--per-query", type=int, default=12)
    d = sub.add_parser("download")
    d.add_argument("--limit", type=int, default=0)
    sc = sub.add_parser("screen")
    sc.add_argument("--window-s", type=int, default=480)
    f = sub.add_parser("flash")
    f.add_argument("--model", default="gemini-2.5-flash")
    f.add_argument("--cap", type=float, default=25.0)
    e = sub.add_parser("evidence")
    e.add_argument("--no-tone", action="store_true")
    e.add_argument("--min-heat", type=int, default=2)
    fi = sub.add_parser("finalize")
    fi.add_argument("--model", default="gemini-2.5-pro")
    fi.add_argument("--keep", type=int, default=15)
    fi.add_argument("--min-heat", type=int, default=2)
    fi.add_argument("--allow-calm", action="store_true")
    fi.add_argument("--cap", type=float, default=25.0)
    args = ap.parse_args(argv)
    return {"search": cmd_search, "download": cmd_download, "screen": cmd_screen, "flash": cmd_flash,
            "evidence": cmd_evidence, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
