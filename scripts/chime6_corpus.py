"""CHiME-6 -> heat-map recordings: a real dinner party at home, with per-person headsets.

The home counterpart to AMI's office. Four friends cooking and eating dinner in
a real house, recorded on each participant's own binaural headset — so, exactly
like AMI, the per-person channel gives ground truth for who spoke when (and for
overlap) with no annotation, and summing the channels gives a faithful
one-device-in-the-room mix. Informal, spontaneous, and emotionally alive in a
way a meeting is not. CC BY-SA 4.0.

Each session is long (~2 h). It is split into SEGMENT_S chunks so a recording
is comparable in length to everything else on the graph, and so the loudness
chain's baseline re-seeds the way it would on a real session start.

Layout (CHiME-6 dev release): audio/dev/<S>_<P>.wav per participant, and
transcriptions/dev/<S>.json with speaker / start_time / end_time per utterance.

    python scripts/chime6_corpus.py
    -> tmp/corpora/chime6/segments16k/*.wav + tmp/corpora/chime6/heatmap_manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
ROOT = REPO / "tmp/corpora/chime6"
SR = 16000
SEGMENT_S = 15 * 60
MIN_SEGMENT_S = 5 * 60


def extract(root: Path) -> None:
    tgz = root / "CHiME6_dev.tar.gz"
    marker = root / "CHiME6_dev.extracted"
    if marker.exists() or not tgz.exists():
        return
    try:
        with tarfile.open(tgz) as t:
            t.extractall(root / "raw", filter="data")
        marker.write_text("ok")
        print("  extracted CHiME6_dev.tar.gz")
    except (tarfile.ReadError, EOFError):
        print("  CHiME6_dev.tar.gz incomplete — extraction skipped")


def hms(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x.astype(np.float64), sr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    extract(root)
    raw = root / "raw"
    # The OpenSLR dev package ships AUDIO ONLY: per-participant binaural
    # headsets (S02_P08.wav ...) and the six Kinect arrays (S02_U01.CH1.wav
    # ...). The per-utterance transcript JSONs are a separate download. That is
    # fine — the headsets ARE the ground truth, exactly as with AMI: energy VAD
    # on each person's own channel says who spoke when, with no annotation.
    # Transcripts are used when present, headsets otherwise.
    transcripts = {p.stem: p for p in raw.rglob("*.json") if re.fullmatch(r"S\d+", p.stem)}
    sessions = sorted({m.group(1) for p in raw.rglob("S*_P*.wav") for m in [re.match(r"(S\d+)_P\d+", p.stem)] if m})
    if not sessions:
        raise SystemExit("no per-participant headsets (S*_P*.wav) under tmp/corpora/chime6/raw")
    out_dir = root / "segments16k"
    out_dir.mkdir(exist_ok=True)
    entries = []
    for sess in sessions:
        tpath = transcripts.get(sess)
        heads = sorted(p for p in raw.rglob(f"{sess}_P*.wav"))
        if len(heads) < 2:
            print(f"  {sess}: fewer than 2 headsets — skipped")
            continue
        chans, sr = [], None
        for h in heads:
            x, sr = read_wav(h)
            chans.append(x)
        n = min(len(c) for c in chans)
        mix = np.sum([c[:n] for c in chans], axis=0)
        peak = np.max(np.abs(mix))
        if peak > 0:
            mix = mix / peak * 10 ** (-3.0 / 20.0)
        if sr != SR:
            from math import gcd

            from scipy.signal import resample_poly
            g = gcd(sr, SR)
            mix = resample_poly(mix, SR // g, sr // g)
        pcm = np.clip(mix * 32767.0, -32768, 32767).astype("<i2")
        total_s = len(pcm) / SR

        spans: dict[str, list[list[float]]] = defaultdict(list)
        if tpath is not None:
            for u in json.loads(tpath.read_text()):
                try:
                    a, b = hms(u["start_time"]), hms(u["end_time"])
                except (KeyError, ValueError):
                    continue
                if b > a:
                    spans[u["speaker"]].append([a, b])
            for v in spans.values():
                v.sort()
        else:
            # Headset-derived truth — the AMI method, same constants, so the two
            # corpora' speaker labels are comparable by construction. A binaural
            # headset is stereo; read_wav already averaged it to mono.
            from ami_corpus import intervals, speaking_mask
            active, _frames = speaking_mask(chans, sr)
            for k, h in enumerate(heads):
                spans[h.stem.split("_")[1]] = intervals(active[k])

        nseg = int(total_s // SEGMENT_S) + (1 if total_s % SEGMENT_S >= MIN_SEGMENT_S else 0)
        for k in range(max(nseg, 1)):
            a0, b0 = k * SEGMENT_S, min((k + 1) * SEGMENT_S, total_s)
            if b0 - a0 < MIN_SEGMENT_S:
                continue
            seg = pcm[int(a0 * SR):int(b0 * SR)]
            wav = out_dir / f"{sess}_seg{k:02d}.wav"
            if not wav.exists():
                with wave.open(str(wav), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SR)
                    w.writeframes(seg.tobytes())
            local = {
                spk: [[round(max(s, a0) - a0, 3), round(min(e, b0) - a0, 3)] for s, e in v if e > a0 and s < b0]
                for spk, v in spans.items()
            }
            local = {spk: v for spk, v in local.items() if v}
            talk = {spk: round(sum(e - s for s, e in v), 1) for spk, v in local.items()}
            entries.append({
                "id": f"chime6_{sess}_seg{k:02d}", "corpus": "CHiME-6",
                "setting": f"dinner party at home, {len(local)} people",
                "audio": f"segments16k/{wav.name}", "speakers": local,
                "self": max(talk, key=talk.get) if talk else None,
                "heated_spans": None, "duration_s": round(b0 - a0, 1), "speaking_seconds": talk,
            })
        print(f"  {sess}: {total_s/60:.0f} min, {len(heads)} headsets, truth from {'transcripts' if tpath else 'headsets'} -> {nseg} segments", flush=True)
    (root / "heatmap_manifest.json").write_text(json.dumps(entries, indent=1))
    print(f"\n{len(entries)} segments, {sum(e['duration_s'] for e in entries)/3600:.1f} h -> {root/'heatmap_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
