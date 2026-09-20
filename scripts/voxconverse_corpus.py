"""VoxConverse -> heat-map recordings: debates and panel shows with official diarization.

Multi-speaker, often adversarial broadcast conversation — panel debates, talk
shows, interviews — released for research under CC BY 4.0 with per-speaker RTTM
ground truth. It sits between AMI's calm meetings and CONFER's open arguments
on the heat axis, and it is far-field broadcast audio rather than one phone on
a table, which the setting label says.

Speaker truth from RTTM; no heat labels (the tone-model reference supplies that
axis). Recordings under MIN_S are skipped — a 20 s clip cannot establish the
speaker baseline the loudness chain needs.

    python scripts/voxconverse_corpus.py
    -> tmp/corpora/voxconverse/audio16k/*.wav + tmp/corpora/voxconverse/heatmap_manifest.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import wave
import zipfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "tmp/corpora/voxconverse"
SR = 16000
MIN_S = 60.0
MAX_RECORDINGS = 80


def extract(root: Path) -> None:
    z = root / "voxconverse_dev_wav.zip"
    marker = root / "dev_wav.extracted"
    if marker.exists() or not z.exists():
        return
    if not zipfile.is_zipfile(z):
        print("  dev wav zip incomplete — skipped")
        return
    with zipfile.ZipFile(z) as zf:
        zf.extractall(root / "dev_wav")
    marker.write_text("ok")
    print("  extracted voxconverse_dev_wav.zip")


def parse_rttm(path: Path) -> dict[str, list[list[float]]]:
    spans: dict[str, list[list[float]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            s, d = float(f[3]), float(f[4])
            spans[f[7]].append([round(s, 3), round(s + d, 3)])
    for v in spans.values():
        v.sort()
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
    ap.add_argument("--max", type=int, default=MAX_RECORDINGS)
    args = ap.parse_args()
    root = Path(args.root)
    extract(root)
    rttm_dir = root / "voxconverse" / "dev"
    wavs = {p.stem: p for p in (root / "dev_wav").rglob("*.wav")}
    if not rttm_dir.exists() or not wavs:
        raise SystemExit("need both the RTTM repo (voxconverse/dev) and the extracted dev wavs")
    out_dir = root / "audio16k"
    out_dir.mkdir(exist_ok=True)
    entries = []
    # Longest first: more turns per recording, and the baseline has time to settle.
    def dur_of(p: Path) -> float:
        with wave.open(str(p)) as w:
            return w.getnframes() / w.getframerate()
    for rttm in sorted(rttm_dir.glob("*.rttm"), key=lambda r: -dur_of(wavs[r.stem]) if r.stem in wavs else 0):
        if len(entries) >= args.max:
            break
        src = wavs.get(rttm.stem)
        if src is None:
            continue
        spans = parse_rttm(rttm)
        if len(spans) < 2:
            continue
        wav = out_dir / f"{rttm.stem}.wav"
        if not to_wav16k(src, wav):
            continue
        dur = dur_of(wav)
        if dur < MIN_S:
            continue
        talk = {k: round(sum(e - s for s, e in v), 1) for k, v in spans.items()}
        entries.append({
            "id": f"vox_{rttm.stem}", "corpus": "VoxConverse",
            "setting": f"broadcast debate/panel, {len(spans)} speakers",
            "audio": f"audio16k/{wav.name}", "speakers": spans, "self": max(talk, key=talk.get),
            "heated_spans": None, "duration_s": round(dur, 1), "speaking_seconds": talk,
        })
    (root / "heatmap_manifest.json").write_text(json.dumps(entries, indent=1))
    print(f"{len(entries)} recordings, {sum(e['duration_s'] for e in entries)/3600:.1f} h -> {root/'heatmap_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
