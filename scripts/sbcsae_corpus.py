"""SBCSAE -> heat-map recordings: everyday American conversation, the widest natural range we hold.

The Santa Barbara Corpus of Spoken American English is ~60 recordings of real
people in real settings — family dinners, arguments, phone calls, a tutoring
session, a town meeting — with every utterance transcribed and timestamped per
speaker in TalkBank CHAT format. It is spontaneous, informal and emotionally
varied, which is exactly what AMI's office meetings are not.

Speaker truth comes from the CHAT timestamps (`\\x15start_end\\x15` in ms), so
each recording contributes attribution truth. It carries no heat labels; the
tone-model reference (heat_reference.py) supplies that axis.

    python scripts/sbcsae_corpus.py
    -> tmp/corpora/sbcsae/audio16k/*.wav + tmp/corpora/sbcsae/heatmap_manifest.json

Licence: CC BY-ND 3.0 (OpenSLR 155 mirror). Kept in gitignored tmp/, never
redistributed.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tarfile
import wave
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "tmp/corpora/sbcsae"
SR = 16000
AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
BULLET = re.compile(r"\x15(\d+)_(\d+)\x15")


def extract(root: Path) -> None:
    tgz = root / "SBCSAE.tar.gz"
    marker = root / "SBCSAE.extracted"
    if marker.exists() or not tgz.exists():
        return
    try:
        with tarfile.open(tgz) as t:
            t.extractall(root / "raw", filter="data")
        marker.write_text("ok")
        print("  extracted SBCSAE.tar.gz")
    except (tarfile.ReadError, EOFError):
        print("  SBCSAE.tar.gz incomplete — extraction skipped")


def parse_cha(path: Path) -> dict[str, list[list[float]]]:
    """speaker -> [[start_s, end_s], ...] from CHAT main tiers (*SPK: ... bullet)."""
    spans: dict[str, list[list[float]]] = defaultdict(list)
    cur = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("*"):
            cur = line[1:line.index(":")].strip() if ":" in line else None
        elif not line.startswith("\t"):
            cur = None
        if cur is None:
            continue
        for a, b in BULLET.findall(line):
            s, e = int(a) / 1000.0, int(b) / 1000.0
            if e > s:
                spans[cur].append([round(s, 3), round(e, 3)])
    # merge touching spans per speaker so a turn is a turn, not a bullet
    for spk, v in spans.items():
        v.sort()
        merged = []
        for s, e in v:
            if merged and s - merged[-1][1] < 0.3:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        spans[spk] = merged
    return dict(spans)


def to_wav16k(src: Path, dst: Path) -> bool:
    if dst.exists():
        return True
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR), str(dst)],
                       capture_output=True)
    return r.returncode == 0 and dst.exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    extract(root)
    raw = root / "raw"
    audio = {p.stem.upper(): p for p in raw.rglob("*") if p.suffix.lower() in AUDIO_EXT}
    out_dir = root / "audio16k"
    out_dir.mkdir(exist_ok=True)
    entries = []
    for cha in sorted(raw.rglob("*.cha")):
        rec = cha.stem.upper()
        src = audio.get(rec)
        if src is None:
            continue
        spans = parse_cha(cha)
        if len(spans) < 2:
            continue
        wav = out_dir / f"{rec}.wav"
        if not to_wav16k(src, wav):
            print(f"  {rec}: ffmpeg failed")
            continue
        with wave.open(str(wav)) as w:
            dur = w.getnframes() / w.getframerate()
        talk = {k: round(sum(e - s for s, e in v), 1) for k, v in spans.items()}
        self_spk = max(talk, key=talk.get)
        entries.append({
            "id": f"sbcsae_{rec}", "corpus": "SBCSAE",
            "setting": f"everyday conversation, {len(spans)} speakers",
            "audio": f"audio16k/{wav.name}", "speakers": spans, "self": self_spk,
            "heated_spans": None, "duration_s": round(dur, 1), "speaking_seconds": talk,
        })
        print(f"  {rec}: {len(spans)} speakers, {dur/60:.1f} min", flush=True)
    (root / "heatmap_manifest.json").write_text(json.dumps(entries, indent=1))
    print(f"\n{len(entries)} recordings, {sum(e['duration_s'] for e in entries)/3600:.1f} h -> {root/'heatmap_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
