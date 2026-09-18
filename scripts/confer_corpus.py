"""CONFER -> heat-map recordings: real televised debates with CONTINUOUS conflict intensity.

The heated end of the axis that everything else we hold lacks. CONFER (Imperial
iBUG) is 120 clips of real Greek political debates — 2 or 3 people genuinely
arguing on air — with conflict intensity rated frame-by-frame (25 fps) by 10
annotators on a 0..1000 scale. That is a human answer to "how heated is this
moment", second by second, on spontaneous speech.

Two honest limits carried into every number: it is Greek (irrelevant to a
loudness chain and to an acoustic tone model, relevant to any text tone), and it
is broadcast studio audio with two or three microphones mixed down, not one
phone on a table. It has no per-speaker channel, so it contributes heat ground
truth but not attribution ground truth.

Each clip becomes ONE recording: audio extracted at 16 kHz mono, the ten
raters' mean conflict resampled to 1 s, heated_spans = runs where that mean is
at or above HEATED_AT (400/1000 — the upper end of the corpus's own mean of
250 ± 170), and the full per-second series kept for finer analysis.

    python scripts/confer_corpus.py
    -> tmp/corpora/confer/clips16k/*.wav + tmp/corpora/confer/heatmap_manifest.json

Licence: free for research with citation of the CONFER paper (Georgakis et
al.); redistribution is not permitted, so nothing here leaves gitignored tmp/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import wave
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "tmp/corpora/confer"
SR = 16000
FPS = 25.0
HEATED_AT = 400.0          # of 1000
MIN_HEATED_RUN_S = 2.0
MEDIA_EXT = (".mp4", ".avi", ".wmv", ".mov", ".mkv", ".mpg", ".mpeg", ".webm", ".flv")


def extract_folds(root: Path) -> None:
    for z in sorted(root.glob("Clips_*.zip")):
        marker = z.with_suffix(".extracted")
        if marker.exists():
            continue
        if not zipfile.is_zipfile(z):
            print(f"  {z.name}: not complete yet — skipped")
            continue
        try:
            with zipfile.ZipFile(z) as zf:
                zf.testzip()
                zf.extractall(root / "clips")
        except zipfile.BadZipFile:
            print(f"  {z.name}: still downloading — skipped")
            continue
        marker.write_text("ok")
        print(f"  extracted {z.name}")


def media_index(root: Path) -> dict[str, Path]:
    """clip id (e.g. 20111003_seq11) -> media file. The zip layout was not
    documented, so match by the annotation folder name appearing in the path."""
    idx: dict[str, Path] = {}
    for p in (root / "clips").rglob("*"):
        if p.suffix.lower() in MEDIA_EXT and p.is_file():
            for part in reversed(p.parts):
                stem = Path(part).stem
                if "_seq" in stem:
                    idx.setdefault(stem, p)
                    break
    return idx


def to_wav16k(src: Path, dst: Path) -> bool:
    if dst.exists():
        return True
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR), str(dst)],
                       capture_output=True)
    return r.returncode == 0 and dst.exists()


def conflict_per_second(mat: Path) -> np.ndarray:
    a = sio.loadmat(str(mat))["annotations"]        # raters x frames
    mean = a.mean(axis=0)
    secs = int(np.ceil(len(mean) / FPS))
    out = np.zeros(secs)
    for s in range(secs):
        seg = mean[int(s * FPS):int((s + 1) * FPS)]
        out[s] = seg.mean() if len(seg) else 0.0
    return out


def heated_spans(series: np.ndarray) -> list[list[float]]:
    hot = series >= HEATED_AT
    spans, start = [], None
    for i, h in enumerate(list(hot) + [False]):
        if h and start is None:
            start = i
        elif not h and start is not None:
            if i - start >= MIN_HEATED_RUN_S:
                spans.append([float(start), float(i)])
            start = None
    return spans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    extract_folds(root)
    media = media_index(root)
    print(f"  {len(media)} media clips found")
    out_dir = root / "clips16k"
    out_dir.mkdir(exist_ok=True)
    entries = []
    for mat in sorted(root.glob("ann/Annotations/*/*/annotations.mat")):
        clip_id = mat.parent.name
        group = mat.parent.parent.name                  # two / three
        src = media.get(clip_id)
        if src is None:
            continue
        wav = out_dir / f"{clip_id}.wav"
        if not to_wav16k(src, wav):
            print(f"  {clip_id}: ffmpeg failed")
            continue
        series = conflict_per_second(mat)
        with wave.open(str(wav)) as w:
            dur = w.getnframes() / w.getframerate()
        spans = heated_spans(series)
        level = float(series.mean())
        entries.append({
            "id": f"confer_{clip_id}", "corpus": "CONFER",
            "setting": f"TV debate, {group} people — " + ("heated" if level >= HEATED_AT else "tense" if level >= 200 else "calm"),
            "audio": f"clips16k/{wav.name}", "speakers": None, "self": None,
            "heated_spans": spans or None, "duration_s": round(dur, 1),
            "conflict_mean": round(level, 1), "conflict_max": round(float(series.max()), 1),
            "conflict_per_second": [round(float(x), 1) for x in series],
        })
    (root / "heatmap_manifest.json").write_text(json.dumps(entries, indent=1))
    hot = sum(1 for e in entries if e["heated_spans"])
    print(f"\n{len(entries)} clips ({hot} with heated spans), "
          f"{sum(e['duration_s'] for e in entries)/60:.1f} min -> {root/'heatmap_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
