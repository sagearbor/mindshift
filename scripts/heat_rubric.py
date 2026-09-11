"""WHAT SHOULD FIRE? — a rubric built from labelled corpora, not from intuition.

The nudge ladder (+6 / +10 / +14 dB over the speaker's own baseline) was never
calibrated against a corpus of people actually being angry. This builds the
calibration set: for every clip, every candidate signal we ship or could ship,
beside the corpus's own emotion label — so a threshold can be chosen from data
and its cost stated in false alarms rather than argued about.

CREMA-D is the reference corpus because of its shape, not its size: 91 speakers
each perform the SAME sentences neutral AND angry, so "louder than your own
baseline" is directly measurable per speaker instead of approximated. That is
the exact quantity the shipped ladder claims to use.

  ANG / NEU / HAP / SAD / FEA / DIS, plus LO/MD/HI intensity on a subset.
  Licence: Open Database License (ODbL) — the audio lives in tmp/ and is never
  committed; only the numbers this prints are.

Usage:
  python scripts/heat_rubric.py [--corpus cremad] [--limit N] [--out DIR]

Writes <out>/heat_rubric.csv (one row per clip) and prints the calibration
table. Needs requirements-voice-free deps only: numpy, scipy, soundfile.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "server"))
import prosody as P  # noqa: E402

TARGET_SR = 16000

#: CREMA-D codes -> whether a coach should nudge. Anger is the only one that
#: should: a coach that buzzes for SADNESS or FEAR is punishing someone for
#: being upset, which is the opposite of the product. HAP is included as a
#: HARD negative on purpose — excited happy speech is loud, and a loudness
#: detector's most embarrassing failure is buzzing someone for laughing.
CREMAD_EMOTIONS = {
    "ANG": ("angry", True),
    "NEU": ("neutral", False),
    "HAP": ("happy", False),
    "SAD": ("sad", False),
    "FEA": ("fear", False),
    "DIS": ("disgust", False),
}
CREMAD_INTENSITY = {"LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified"}

#: RAVDESS (CC BY-NC-SA 4.0 — audio stays in tmp/, derivatives are never
#: committed). 24 actors, and unlike CREMA-D it carries an explicit
#: normal/strong intensity on every emotion, which is the closest thing any
#: free corpus has to "mild vs shouting".
#: `calm` is mapped as a NEGATIVE deliberately: it is the one label that means
#: "deliberately relaxed speech", so a detector that fires on it is broken in
#: the most obvious way available.
RAVDESS_EMOTIONS = {
    "01": ("neutral", False), "02": ("calm", False), "03": ("happy", False),
    "04": ("sad", False), "05": ("angry", True), "06": ("fear", False),
    "07": ("disgust", False), "08": ("surprised", False),
}
RAVDESS_INTENSITY = {"01": "normal", "02": "strong"}


def to_mono16k(samples: np.ndarray, sr: int):
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32)
    if sr == TARGET_SR:
        return samples, sr
    g = gcd(sr, TARGET_SR)
    return resample_poly(samples, TARGET_SR // g, sr // g).astype(np.float32), TARGET_SR


def features(samples: np.ndarray, sr: int) -> dict | None:
    """Frame-level prosody via server/prosody.py — the SAME estimator the phone
    ports, so a threshold chosen here transfers to the device rather than to a
    lookalike."""
    frame_len = max(1, int(sr * P.FRAME_MS / 1000.0))
    hop = max(1, int(sr * P.HOP_MS / 1000.0))
    if samples.size < frame_len:
        return None
    f0s, energies_db, voiced = [], [], 0
    n = 0
    for start in range(0, samples.size - frame_len + 1, hop):
        frame = samples[start:start + frame_len]
        n += 1
        energies_db.append(20.0 * np.log10(max(P.rms_energy(frame), 1e-6)))
        f0 = P._frame_f0(frame, sr)
        if f0 is not None:
            f0s.append(f0)
            voiced += 1
    if n == 0:
        return None
    e = np.asarray(energies_db)
    f = np.asarray(f0s) if f0s else np.zeros(1)
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    return {
        # What the ladder actually reads.
        "rms_dbfs": 20.0 * np.log10(max(rms, 1e-9)),
        # Gain- and length-invariant shape features (activation v2's inputs).
        "f0_mean": float(np.mean(f)),
        "f0_sd": float(np.std(f)),
        "f0_range_ratio": float((np.max(f) - np.min(f)) / max(np.mean(f), 1e-6)) if f0s else 0.0,
        "energy_db_sd": float(np.std(e)),
        "energy_dynamic_range": float(np.max(e) - np.min(e)),
        "voiced_fraction": voiced / n,
        "duration_s": samples.size / sr,
    }


def scan_cremad(root: str, limit: int | None):
    """CREMA-D filenames: <actor>_<sentence>_<EMO>_<INTENSITY>.wav"""
    paths = sorted(glob.glob(os.path.join(root, "*.wav")))
    if limit:
        paths = paths[:limit]
    for p in paths:
        parts = os.path.basename(p)[:-4].split("_")
        if len(parts) != 4:
            continue
        actor, sentence, emo, intensity = parts
        if emo not in CREMAD_EMOTIONS:
            continue
        name, should_nudge = CREMAD_EMOTIONS[emo]
        yield {
            "path": p, "speaker": actor, "sentence": sentence, "emotion": name,
            "should_nudge": should_nudge,
            "intensity": CREMAD_INTENSITY.get(intensity, intensity),
        }


def scan_ravdess(root: str, limit: int | None):
    """RAVDESS filenames: modality-channel-EMOTION-INTENSITY-statement-rep-ACTOR.wav"""
    paths = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    if limit:
        paths = paths[:limit]
    for p in paths:
        parts = os.path.basename(p)[:-4].split("-")
        if len(parts) != 7:
            continue
        _, _, emo, intensity, statement, _rep, actor = parts
        if emo not in RAVDESS_EMOTIONS:
            continue
        name, should_nudge = RAVDESS_EMOTIONS[emo]
        yield {
            "path": p, "speaker": actor, "sentence": statement, "emotion": name,
            "should_nudge": should_nudge,
            "intensity": RAVDESS_INTENSITY.get(intensity, intensity),
        }


CORPORA = {
    "cremad": (scan_cremad, "tmp/corpora/cremad/AudioWAV"),
    "ravdess": (scan_ravdess, "tmp/ravdess/audio"),
}


def auc(pos, neg) -> float:
    """Mann-Whitney AUC. Ties count a half, so a feature that cannot separate
    at all scores exactly 0.5 rather than something flattering."""
    if not len(pos) or not len(neg):
        return float("nan")
    pos, neg = np.asarray(pos), np.asarray(neg)
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="cremad", choices=sorted(CORPORA))
    ap.add_argument("--root", default=None, help="override the corpus's default audio root")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(REPO, "tmp/corpora"))
    args = ap.parse_args()

    scan, default_root = CORPORA[args.corpus]
    root = args.root or os.path.join(REPO, default_root)
    rows = []
    clips = list(scan(root, args.limit))
    if not clips:
        raise SystemExit(f"no clips under {root} — see the module docstring for the download")
    print(f"extracting features from {len(clips)} clips ...", flush=True)
    for i, c in enumerate(clips):
        if i and i % 500 == 0:
            print(f"  {i}/{len(clips)}", flush=True)
        try:
            s, sr = sf.read(c["path"])
            s, sr = to_mono16k(s, sr)
            f = features(s, sr)
        except Exception as exc:                       # a corrupt clip must not stop the scan
            print(f"  skipped {os.path.basename(c['path'])}: {exc}")
            continue
        if f is None:
            continue
        rows.append({**c, **f})

    # Per-speaker baseline from that speaker's OWN neutral clips — the quantity
    # the device approximates with a running median, measured exactly here.
    by_speaker: dict[str, list[float]] = {}
    for r in rows:
        if r["emotion"] == "neutral":
            by_speaker.setdefault(r["speaker"], []).append(r["rms_dbfs"])
    baseline = {k: float(np.median(v)) for k, v in by_speaker.items() if v}
    for r in rows:
        b = baseline.get(r["speaker"])
        r["db_over_own_neutral"] = None if b is None else r["rms_dbfs"] - b

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, f"heat_rubric_{args.corpus}.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}  ({len(rows)} clips, {len(baseline)} speakers with a baseline)")
    report(rows, args.corpus)


def report(rows, corpus="cremad"):
    scored = [r for r in rows if r.get("db_over_own_neutral") is not None]
    ang = [r for r in scored if r["emotion"] == "angry"]
    neu = [r for r in scored if r["emotion"] == "neutral"]
    hap = [r for r in scored if r["emotion"] == "happy"]
    other_neg = [r for r in scored if not r["should_nudge"]]

    speakers = len({r["speaker"] for r in scored})
    print(f"\n{'='*72}\nTHE SHIPPED LADDER — {corpus}, {speakers} real speakers being angry\n{'='*72}")
    d = np.array([r["db_over_own_neutral"] for r in ang])
    print(f"angry clips: {len(d)}   median {np.median(d):+.1f} dB over that speaker's own neutral")
    for rung, lvl in ((6.0, 1), (10.0, 2), (14.0, 3)):
        caught = (d >= rung).mean() * 100
        fp = np.array([r["db_over_own_neutral"] for r in other_neg])
        print(f"  rung +{rung:<4.0f} (level {lvl}): catches {caught:5.1f}% of angry clips"
              f"   |  fires on {(fp >= rung).mean()*100:5.1f}% of NON-angry clips")

    print(f"\n{'='*72}\nWHICH SIGNAL ACTUALLY SEPARATES ANGRY FROM EVERYTHING ELSE?\n{'='*72}")
    print(f"{'feature':24} {'AUC ang-vs-all':>15} {'AUC ang-vs-neutral':>20} {'AUC ang-vs-HAPPY':>18}")
    feats = ["db_over_own_neutral", "rms_dbfs", "f0_mean", "f0_sd", "f0_range_ratio",
             "energy_db_sd", "energy_dynamic_range", "voiced_fraction"]
    best = []
    for k in feats:
        a = [r[k] for r in ang]
        v_all = auc(a, [r[k] for r in other_neg])
        v_neu = auc(a, [r[k] for r in neu])
        v_hap = auc(a, [r[k] for r in hap])
        best.append((v_all, k))
        print(f"{k:24} {v_all:>15.3f} {v_neu:>20.3f} {v_hap:>18.3f}")

    print(f"\n{'='*72}\nIF WE KEEP LOUDNESS, WHAT THRESHOLD SHOULD IT BE?\n{'='*72}")
    fp_all = np.array([r["db_over_own_neutral"] for r in other_neg])
    print(f"{'threshold':>10} {'catches angry':>15} {'false alarms':>14}   (on {len(fp_all)} non-angry clips)")
    for t in (2, 3, 4, 5, 6, 8, 10, 14):
        print(f"{t:>+10} {(d >= t).mean()*100:>14.1f}% {(fp_all >= t).mean()*100:>13.1f}%")

    print(f"\n{'='*72}\nTHE EMBARRASSING CASE: would it buzz someone for being HAPPY?\n{'='*72}")
    h = np.array([r["db_over_own_neutral"] for r in hap])
    print(f"happy clips median {np.median(h):+.1f} dB vs angry {np.median(d):+.1f} dB "
          f"-> at +6 dB it fires on {(h >= 6).mean()*100:.1f}% of happy clips "
          f"and {(d >= 6).mean()*100:.1f}% of angry ones")

    by_int = {}
    for r in ang:
        by_int.setdefault(r["intensity"], []).append(r["db_over_own_neutral"])
    print(f"\nangry clips by the corpus's own INTENSITY label:")
    for k in ("low", "medium", "high", "unspecified"):
        if k in by_int:
            v = np.array(by_int[k])
            print(f"  {k:12} n={len(v):<4} median {np.median(v):+5.1f} dB   over +6: {(v>=6).mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
