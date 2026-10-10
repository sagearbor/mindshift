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
import time
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
        sc, sh2 = scoring.best_affine(t["speakers"], hyp, dur)
        warped = scoring.warp(hyp, sc, sh2)
        d2 = scoring.der(t["speakers"], warped, dur)
        be = scoring.boundary_error(t["speakers"], warped)
        r.update({"speakers_true": len(t["speakers"]), "der": d0["der"], "der_shift": d1["der"], "shift": sh,
                  "der_affine": d2["der"], "scale": sc, "affine_shift": sh2,
                  "miss": d2["miss"], "fa": d2["false_alarm"], "conf": d2["confusion"],
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
            "n": len(mr), "der": avg("der"), "der_shift": avg("der_shift"), "der_affine": avg("der_affine"),
            "boundary_med": avg("boundary_med"),
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
    notes_p = TMP / "annotator-validation/findings.json"
    findings = json.loads(notes_p.read_text()) if notes_p.exists() else {}
    write_html(summary, rows, findings)
    print(f"report: {REPORT}")
    return 0


def _f(v, pct=False, nd=2):
    if v is None:
        return "—"
    return f"{100 * v:.0f}%" if pct else f"{v:.{nd}f}"


def write_html(summary: dict, rows: list[dict], findings: dict) -> None:
    models = sorted(summary, key=lambda m: (summary[m]["der_shift"] is None, summary[m]["der_shift"] or 9))
    head = ("<tr><th>model</th><th>clips</th><th>DER</th><th>DER after best shift</th><th>DER after shift + clock-rate fit</th><th>turn change: median miss (after fit)</th>"
            "<th>changes within 1 s</th><th>speaker-count error</th><th>WER (AMI)</th><th>WER (all)</th>"
            "<th>heat vs conflict, across clips (ρ)</th><th>overall_heat vs conflict (ρ)</th>"
            "<th>heat vs conflict, per second (ρ)</th><th>est. $ / audio min</th></tr>")
    body = []
    for m in models:
        s = summary[m]
        per_min = s["usd"] / (s["audio_s"] / 60) if s["audio_s"] else None
        body.append(f"<tr><td><b>{html.escape(m)}</b></td><td>{s['n']}</td><td>{_f(s['der'], True)}</td>"
                    f"<td>{_f(s['der_shift'], True)}</td><td>{_f(s['der_affine'], True)}</td><td>{_f(s['boundary_med'], nd=1)} s</td>"
                    f"<td>{_f(s['boundary_1s'], True)}</td><td>{_f(s['spk_count_err'], nd=1)}</td>"
                    f"<td>{_f(s['wer_ami'], True)}</td><td>{_f(s['wer'], True)}</td>"
                    f"<td>{_f(s['heat_rho_clips'])}</td><td>{_f(s['overall_rho_clips'])}</td>"
                    f"<td>{_f(s['heat_r_within'])}</td><td>{_f(per_min, nd=3)}</td></tr>")
    clip_rows = []
    for r in sorted(rows, key=lambda r: (r["id"], r["model"])):
        clip_rows.append(
            f"<tr><td>{html.escape(r['id'])}</td><td>{html.escape(r['model'])}</td><td>{r['segments']}</td>"
            f"<td>{r.get('speakers_hyp')}/{r.get('speakers_true', '—')}</td><td>{_f(r.get('der'), True)}</td>"
            f"<td>{_f(r.get('der_shift'), True)} ({_f(r.get('shift'), nd=2)} s)</td>"
            f"<td>{_f(r.get('der_affine'), True)} (x{_f(r.get('scale'), nd=2)}, {_f(r.get('affine_shift'), nd=1)} s)</td>"
            f"<td>{_f(r.get('miss'), True)} / {_f(r.get('fa'), True)} / {_f(r.get('conf'), True)}</td>"
            f"<td>{_f(r.get('boundary_med'), nd=1)}</td><td>{_f(r.get('wer'), True)}</td>"
            f"<td>{r.get('overall_heat', '—')}</td><td>{_f(r.get('heat_mean'))}</td>"
            f"<td>{_f(r.get('conflict_mean'), nd=0)}</td><td>{_f(r.get('heat_r_within'))}</td></tr>")
    verdict = "".join(f"<li>{v}</li>" for v in findings.get("verdict", []))
    caveats = "".join(f"<li>{v}</li>" for v in findings.get("caveats", []))
    failures = "".join(f"<li>{v}</li>" for v in findings.get("failures", []))
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Annotator Validation</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--line:#e2e0da;--card:#fff;--acc:#2f5d8a}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#161615;--fg:#ecebe6;--mut:#a3a29c;--line:#34332f;--card:#1f1f1d;--acc:#8db7e0}}}}
:root[data-theme=dark]{{--bg:#161615;--fg:#ecebe6;--mut:#a3a29c;--line:#34332f;--card:#1f1f1d;--acc:#8db7e0}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:16px}}
main{{max-width:1100px;margin:auto}} h1{{font-size:1.5em;margin:.2em 0}} h2{{font-size:1.15em;margin-top:1.6em}}
.mut{{color:var(--mut)}} .card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;width:100%}}
th,td{{border-bottom:1px solid var(--line);padding:5px 7px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child,td:nth-child(2){{text-align:left}} th{{font-weight:600;color:var(--mut);white-space:normal}}
li{{margin:.3em 0}}
</style></head><body><main>
<h1>Gemini annotator validation</h1>
<p class="mut">Gemini's annotations scored against human ground truth on {len({r['id'] for r in rows})} corpus clips.
Generated {time.strftime('%Y-%m-%d %H:%M')} by scripts/annotator_validate.py. Lower is better for DER, WER and turn-change error;
higher is better for ρ.</p>
<div class="card"><b>Verdict</b><ul>{verdict}</ul></div>
<h2>Per model</h2><div class="tw"><table>{head}{''.join(body)}</table></div>
<h2>What the ground truth is</h2><ul>
<li><b>AMI</b> (3 meetings, 5 min each): who-spoke-when from the manual transcripts' word timings (words merged into turns
across gaps under 0.3 s); WER against the same manual words. The cleanest speaker-timing truth we hold.</li>
<li><b>CHiME-6</b> S02 dinner party (5 min): human utterance segmentation (utterance boundaries are generous, so silence inside
an utterance counts as speech) and the official transcript.</li>
<li><b>SBCSAE</b> SBC013 (5 min, 5 speakers): human CHAT transcript; CHAT markup is stripped, but its conventions (fragments,
laughter notation) still add some WER that is not Gemini's fault.</li>
<li><b>CONFER</b> (8 Greek TV-debate clips, 37-114 s): ten raters' continuous conflict intensity. Annotated heat per second =
max vocal intensity of segments covering it, +1 for angry/frustrated/sarcastic.</li>
<li>DER: 10 ms frames, optimal one-to-one speaker mapping, no collar, overlap counted per speaker. "After best shift" removes one
constant offset (±5 s) before scoring; "after fit" also fits a clock rate (0.70-1.40) and offset (±10 s), because
model timelines drift. The recording-replay pipeline re-times every segment against Deepgram's word timings, so the
after-fit number is the one that matters for the pipeline; the raw one is what you get using Gemini's times as is. A missing annotation for a clip (failed call) is left out of that model's averages.</li>
</ul>
<h2>Failures and fixes seen while running</h2><ul>{failures}</ul>
<h2>Caveats</h2><ul>{caveats}</ul>
<h2>Per clip</h2><div class="tw"><table><tr><th>clip</th><th>model</th><th>segments</th><th>speakers hyp/true</th>
<th>DER</th><th>DER shifted (shift)</th><th>DER after fit (rate, shift)</th><th>miss / FA / confusion (after fit)</th><th>turn-change miss (s)</th><th>WER</th>
<th>overall_heat</th><th>mean heat</th><th>conflict mean</th><th>per-second ρ</th></tr>{''.join(clip_rows)}</table></div>
</main></body></html>"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(doc)


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
