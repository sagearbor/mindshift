#!/usr/bin/env python3
"""annotator_validate.py — can we trust Gemini's annotations? Score them
against HUMAN ground truth before anyone uses them as truth.

    tmp/venv-annot/bin/python scripts/annotator_validate.py build
    tmp/venv-annot/bin/python scripts/annotator_validate.py run --model gemini-3.1-pro-preview --model gemini-3.8-flash
    tmp/venv-annot/bin/python scripts/annotator_validate.py score   # -> tmp/recordings/reports/annotator-validation.html

Clips (tmp/annotator-validation/clips/, gitignored):
* AMI meetings (5 min windows): who-spoke-when from the MANUAL word
  transcripts' word timings, words from the same transcripts.
* CHiME-6 dinner party (5 min): human utterance segmentation + transcript.
* SBCSAE (5 min): human CHAT transcript with utterance timestamps.
* CONFER Greek TV debates (whole clips, 25-140 s): ten raters' continuous
  conflict intensity, for heat.

Metrics: DER-style speaker error (optimal mapping, no collar; raw and after
the best constant shift), median distance from each true speaker change to
the nearest annotated change, speaker-count error, WER, and heat vs human
conflict (Spearman across clips + per-second within clips).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from annotator import core, scoring  # noqa: E402

TMP = REPO / "tmp"
CLIPS = TMP / "annotator-validation/clips"
REPORT = TMP / "recordings/reports/annotator-validation.html"
FFMPEG = "/opt/homebrew/bin/ffmpeg"

AMI = [("ES2002a", 600.0), ("IS1003a", 300.0), ("TS3005a", 600.0)]
CLIP_S = 300.0
CHIME = [("S02", 1200.0)]          # absolute session seconds
SBC = [("SBC013", 300.0)]
CONFER = ["20111031_seq10", "20111111_seq4", "20111031_seq2", "20120326_seq10", "20120123_seq4",
          "20111205_seq1", "20111003_seq10", "20120213_seq2"]


def cut(src: Path, dst: Path, start: float | None, length: float | None) -> None:
    cmd = [FFMPEG, "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if length is not None:
        cmd += ["-t", f"{length:.3f}"]
    cmd += ["-ac", "1", "-ar", "16000", str(dst)]
    subprocess.run(cmd, check=True)


def merge_spans(spans, gap=0.3):
    out = []
    for s, e in sorted(spans):
        if out and s - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def clip_spans(spk: dict, a: float, b: float) -> dict:
    out = {}
    for k, v in spk.items():
        c = [[max(s, a) - a, min(e, b) - a] for s, e in v if e > a and s < b]
        c = [x for x in c if x[1] > x[0]]
        if c:
            out[k] = c
    return out


def build_ami(meeting: str, a: float) -> dict:
    words = []
    for f in sorted((TMP / "corpora/ami_annotations/words").glob(f"{meeting}.*.words.xml")):
        spk = f.name.split(".")[1]
        for el in ET.parse(f).getroot():
            if el.tag != "w" or el.get("punc") == "true" or el.get("starttime") is None:
                continue
            words.append((float(el.get("starttime")), float(el.get("endtime")), spk, el.text or ""))
    b = a + CLIP_S
    win = sorted(w for w in words if a <= w[0] < b)
    spans = defaultdict(list)
    for s, e, k, _ in words:
        spans[k].append([s, max(e, s + 0.05)])
    spk = clip_spans({k: merge_spans(v) for k, v in spans.items()}, a, b)
    cut(TMP / f"ami-corpus/{meeting}.mix16k.wav", CLIPS / f"ami_{meeting}_{int(a)}.wav", a, CLIP_S)
    return {"id": f"ami_{meeting}_{int(a)}", "corpus": "AMI", "duration_s": CLIP_S, "speakers": spk,
            "ref_text": " ".join(w[3] for w in win), "truth": "AMI manual transcript word timings"}


def hms(s: str) -> float:
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def build_chime(sess: str, a: float) -> dict:
    utts = json.loads((TMP / f"corpora/chime6/transcriptions/transcriptions/dev/{sess}.json").read_text())
    b = a + CLIP_S
    seg_idx = int(a // 900)
    src = TMP / f"corpora/chime6/segments16k/{sess}_seg{seg_idx:02d}.wav"
    spans = defaultdict(list)
    texts = []
    for u in utts:
        s, e = hms(u["start_time"]), hms(u["end_time"])
        spans[u["speaker"]].append([s, e])
        if a <= s < b:
            texts.append((s, u["words"]))
    cut(src, CLIPS / f"chime6_{sess}_{int(a)}.wav", a - seg_idx * 900, CLIP_S)
    return {"id": f"chime6_{sess}_{int(a)}", "corpus": "CHiME-6", "duration_s": CLIP_S,
            "speakers": clip_spans({k: merge_spans(v) for k, v in spans.items()}, a, b),
            "ref_text": " ".join(t for _, t in sorted(texts)), "truth": "CHiME-6 human utterance segmentation"}


BULLET = re.compile(r"\x15(\d+)_(\d+)\x15")


def build_sbc(name: str, a: float) -> dict:
    lines = (TMP / f"corpora/sbcsae/cha/CHAT/{name}.cha").read_text(encoding="utf-8", errors="replace").splitlines()
    b = a + CLIP_S
    spans = defaultdict(list)
    texts = []
    cur, buf = None, []

    def flush():
        if cur is None:
            return
        joined = " ".join(buf)
        for m in BULLET.finditer(joined):
            s, e = int(m.group(1)) / 1000, int(m.group(2)) / 1000
            if e > s:
                spans[cur].append([s, e])
        bm = BULLET.search(joined)
        if bm:
            s = int(bm.group(1)) / 1000
            if a <= s < b:
                t = BULLET.sub(" ", joined)
                t = re.sub(r"(?<!\w)\d(?!\w)|&=\S+|&\S+|\bxxx\b|\byyy\b|@\S*|\[[^\]]*\]|\(\.+\)|[‹›⌈⌉⌊⌋∙↗↘→=~^]", " ", t)
                texts.append((s, t))

    for ln in lines:
        if ln.startswith("*"):
            flush()
            cur = ln[1:ln.index(":")].strip() if ":" in ln else None
            buf = [ln[ln.index(":") + 1:]] if ":" in ln else []
        elif ln.startswith("\t") and cur is not None:
            buf.append(ln)
        else:
            flush()
            cur, buf = None, []
    flush()
    src = TMP / f"corpora/sbcsae/audio16k/{name}.wav"
    cut(src, CLIPS / f"sbcsae_{name}_{int(a)}.wav", a, CLIP_S)
    return {"id": f"sbcsae_{name}_{int(a)}", "corpus": "SBCSAE", "duration_s": CLIP_S,
            "speakers": clip_spans({k: merge_spans(v) for k, v in spans.items()}, a, b),
            "ref_text": " ".join(t for _, t in sorted(texts)), "truth": "SBCSAE human CHAT transcript"}


def build_confer(seq: str) -> dict:
    man = {e["id"]: e for e in json.loads((TMP / "corpora/confer/heatmap_manifest.json").read_text())}
    e = man[f"confer_{seq}"]
    cut(TMP / "corpora/confer" / e["audio"], CLIPS / f"confer_{seq}.wav", None, None)
    return {"id": f"confer_{seq}", "corpus": "CONFER", "duration_s": e["duration_s"], "speakers": None,
            "ref_text": None, "conflict_mean": e["conflict_mean"], "conflict_max": e["conflict_max"],
            "conflict_per_second": e["conflict_per_second"], "truth": "CONFER 10-rater conflict intensity (0-1000)"}


def cmd_build(_args) -> int:
    CLIPS.mkdir(parents=True, exist_ok=True)
    truths = []
    truths += [build_ami(m, a) for m, a in AMI]
    truths += [build_chime(s, a) for s, a in CHIME]
    truths += [build_sbc(s, a) for s, a in SBC]
    truths += [build_confer(s) for s in CONFER]
    for t in truths:
        (CLIPS / f"{t['id']}.truth.json").write_text(json.dumps(t, indent=1))
        n = len(t["speakers"] or {})
        print(f"  {t['id']}: {t['duration_s']:.0f}s speakers={n} words={len((t['ref_text'] or '').split())}")
    return 0


def truths() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(CLIPS.glob("*.truth.json"))]


def cmd_run(args) -> int:
    import annotate_audio as aa
    from annotator.gemini import Gemini
    ledger = core.Ledger(aa.DEFAULT_LEDGER, cap_usd=args.cap)
    gem = Gemini(ledger)
    for t in truths():
        if args.only and not any(o in t["id"] for o in args.only):
            continue
        audio = CLIPS / f"{t['id']}.wav"
        for model in args.model:
            dst = aa.out_path(audio, model)
            if dst.exists() and not args.force:
                continue
            try:
                r = aa.annotate_file(gem, audio, model, window_s=480, overlap_s=30)
            except core.BudgetExceeded as exc:
                print(f"STOP: {exc}")
                return 2
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED {t['id']} [{model}]: {exc}")
                continue
            dst.write_text(json.dumps(r["annotation"], indent=1, ensure_ascii=False))
            (dst.with_suffix(".meta.json")).write_text(json.dumps(
                {"usd": r["usd"], "fixes": r["fixes"], "schema_problems": r["schema_problems"]}, indent=1))
            print(f"  {t['id']} [{model}] ${r['usd']:.4f} total ${ledger.total:.4f}")
    return 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def hyp_spans(ann: dict, include_backchannel: bool = True) -> dict:
    out = defaultdict(list)
    for s in ann["segments"]:
        if not include_backchannel and s.get("is_backchannel"):
            continue
        if s["end"] > s["start"]:
            out[s["speaker"]].append([s["start"], s["end"]])
    return {k: merge_spans(v, 0.0) for k, v in out.items()}


def heat_series(ann: dict, dur: float) -> np.ndarray:
    """Per-second annotated heat: max vocal intensity of segments covering the
    second, bumped +1 for angry/frustrated/sarcastic (capped at 3), 0 in
    silence."""
    n = int(np.ceil(dur))
    x = np.zeros(n)
    for s in ann["segments"]:
        v = s.get("vocal") or {}
        i = v.get("intensity") or 0
        if v.get("emotion") in ("angry", "frustrated", "sarcastic"):
            i = min(3, i + 1)
        a, b = int(s["start"]), int(np.ceil(s["end"]))
        x[a:min(b, n)] = np.maximum(x[a:min(b, n)], i)
    return x


def score_one(t: dict, ann: dict) -> dict:
    r = {"segments": len(ann["segments"]), "speakers_hyp": len(ann["speakers"]),
         "overall_heat": ann["summary"].get("overall_heat")}
    if t.get("speakers"):
        hyp = hyp_spans(ann)
        dur = t["duration_s"]
        d0 = scoring.der(t["speakers"], hyp, dur)
        sh = scoring.best_shift(t["speakers"], hyp, dur, max_shift=5.0, step=0.25)
        d1 = scoring.der(t["speakers"], hyp, dur, shift=sh)
        be = scoring.boundary_error(t["speakers"], {k: [[s + sh, e + sh] for s, e in v] for k, v in hyp.items()})
        r.update({"speakers_true": len(t["speakers"]), "der": d0["der"], "der_shift": d1["der"], "shift": sh,
                  "miss": d1["miss"], "fa": d1["false_alarm"], "conf": d1["confusion"],
                  "boundary_med": be["median_s"], "boundary_1s": be["within_1s"]})
    if t.get("ref_text"):
        hyp_text = " ".join(s["text"] for s in sorted(ann["segments"], key=lambda s: s["start"]))
        r["wer"] = scoring.wer(t["ref_text"], hyp_text)
    if t.get("conflict_per_second"):
        hs = heat_series(ann, t["duration_s"])
        c = np.array(t["conflict_per_second"][:len(hs)])
        hs = hs[:len(c)]
        r["heat_mean"] = float(hs.mean())
        r["heat_r_within"] = scoring.spearman(hs, c)
    return r


def cmd_score(_args) -> int:
    rows = []
    models = set()
    for t in truths():
        for p in sorted(CLIPS.glob(f"{t['id']}.annotation.*.json")):
            if p.name.endswith(".meta.json"):
                continue
            model = p.name[len(t["id"]) + len(".annotation."):-len(".json")]
            ann = json.loads(p.read_text())
            meta_p = p.with_suffix(".meta.json")
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            r = score_one(t, ann)
            r.update({"id": t["id"], "corpus": t["corpus"], "model": model, "usd": meta.get("usd"),
                      "conflict_mean": t.get("conflict_mean")})
            rows.append(r)
            models.add(model)
    out = TMP / "annotator-validation/scores.json"
    summary = {}
    for m in sorted(models):
        mr = [r for r in rows if r["model"] == m]

        def avg(k, rs=mr):
            v = [r[k] for r in rs if r.get(k) is not None]
            return float(np.mean(v)) if v else None
        conf = [r for r in mr if r["corpus"] == "CONFER"]
        summary[m] = {
            "n": len(mr), "der": avg("der"), "der_shift": avg("der_shift"), "boundary_med": avg("boundary_med"),
            "boundary_1s": avg("boundary_1s"), "wer": avg("wer"),
            "wer_ami": avg("wer", [r for r in mr if r["corpus"] == "AMI"]),
            "spk_count_err": avg("spk_err", [dict(r, spk_err=abs(r["speakers_hyp"] - r["speakers_true"]))
                                              for r in mr if r.get("speakers_true")]),
            "heat_rho_clips": scoring.spearman([r["heat_mean"] for r in conf], [r["conflict_mean"] for r in conf])
            if len(conf) >= 3 else None,
            "overall_rho_clips": scoring.spearman([r["overall_heat"] or 0 for r in conf],
                                                  [r["conflict_mean"] for r in conf]) if len(conf) >= 3 else None,
            "heat_r_within": avg("heat_r_within", conf),
            "usd": sum(r["usd"] or 0 for r in mr),
            "audio_s": sum(t["duration_s"] for t in truths() if any(r["id"] == t["id"] for r in mr)),
        }
    out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--model", action="append", required=True)
    r.add_argument("--only", action="append")
    r.add_argument("--force", action="store_true")
    r.add_argument("--cap", type=float, default=25.0)
    sub.add_parser("score")
    args = ap.parse_args(argv)
    return {"build": cmd_build, "run": cmd_run, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
