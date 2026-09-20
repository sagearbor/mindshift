"""Build a REAL-CONVERSATION test corpus from AMI, with ground truth and no hand labelling.

Why AMI, and why this shape:

Every other corpus this repo uses is single-utterance and acted — CREMA-D and
RAVDESS answer "does this clip sound angry", which is a question about clips,
not about conversations. The loop's real job is conversational: who is
speaking, are they the coached user, are they talking over someone, and — the
one that actually bit us — HOW OFTEN does any of that make the wrist buzz. None
of that can be measured on isolated clips, and our own TTS scenes turned out to
be acoustically flat (see docs/plans/2026-09-10-heat-rubric-and-buzz-dose.md).

AMI publishes each participant's OWN headset recording alongside the meeting,
sample-aligned. That gives:

  * ground-truth "who spoke when" with NO annotation — energy VAD per headset;
  * ground-truth OVERLAP, for the same reason;
  * a faithful single-microphone mix, by summing the headsets — which is
    exactly what one phone or one watch in the room actually hears.

Licence: AMI is CC BY 4.0. Audio is written to a gitignored tmp/ directory and
never committed; only derived numbers are.

    python scripts/ami_corpus.py                 # default meeting set
    python scripts/ami_corpus.py --meetings ES2002a IS1000a --out tmp/ami
"""
from __future__ import annotations

import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"

#: A spread across AMI's three meeting series (ES/IS/TS) rather than four
#: sessions of one meeting: different rooms, different participants, different
#: recording conditions. Speaker diversity is the point — a detector tuned to
#: four voices in one room is tuned to nothing.
DEFAULT_MEETINGS = [
    "ES2002a", "ES2003a", "ES2004a", "ES2005a",
    "IS1000a", "IS1001a", "IS1003a", "IS1006a",
    "TS3003a", "TS3004a", "TS3005a", "TS3006a",
]

#: Ground-truth VAD, mirrored from tmp/ami-overlap/fetch.py (which validated
#: the overlap probe) so the two agree by construction.
SPEECH_OVER_FLOOR_DB = 12.0
BLEED_REJECT_DB = 10.0        # headsets hear the others faintly; real simultaneous speech is not 10 dB down
FRAME_MS, HOP_MS = 25.0, 10.0
MIN_RUN_FRAMES = 10           # drop blips under 100 ms
BRIDGE_FRAMES = 10            # bridge gaps under 100 ms


def fetch_headsets(meeting: str, out_dir: Path, max_channels: int = 4) -> list[Path]:
    """Download a meeting's per-speaker headsets. AMI meetings usually have 4;
    a missing channel is normal for some sessions and is skipped rather than
    treated as an error."""
    out_dir.mkdir(parents=True, exist_ok=True)
    got = []
    for i in range(max_channels):
        dest = out_dir / f"{meeting}.Headset-{i}.wav"
        if dest.exists() and dest.stat().st_size > 1_000_000:
            got.append(dest)
            continue
        url = f"{MIRROR}/{meeting}/audio/{meeting}.Headset-{i}.wav"
        print(f"  → {dest.name}", flush=True)
        r = subprocess.run(["curl", "-fsS", "--max-time", "900", "-o", str(dest), url])
        if r.returncode != 0 or not dest.exists() or dest.stat().st_size < 1_000_000:
            dest.unlink(missing_ok=True)
            print(f"    (no channel {i} for {meeting} — skipped)")
            break
        got.append(dest)
    return got


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path.name}: expected mono int16")
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return data.astype(np.float64) / 32768.0, sr


def speaking_mask(chans: list[np.ndarray], sr: int) -> tuple[np.ndarray, int]:
    """Per-channel boolean "this wearer is speaking", frame by frame."""
    hop, win = int(sr * HOP_MS / 1000), int(sr * FRAME_MS / 1000)
    n = min(len(c) for c in chans)
    frames = (n - win) // hop
    idx = np.arange(frames)[:, None] * hop + np.arange(win)[None, :]
    db = np.empty((len(chans), frames))
    for i, c in enumerate(chans):
        seg = c[idx]
        db[i] = 20 * np.log10(np.maximum(np.sqrt((seg * seg).mean(axis=1)), 1e-9))
    floors = np.percentile(db, 20, axis=1)[:, None]
    return (db > floors + SPEECH_OVER_FLOOR_DB) & (db > db.max(axis=0, keepdims=True) - BLEED_REJECT_DB), frames


def intervals(mask: np.ndarray) -> list[list[float]]:
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    out: list[list[int]] = []
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s < MIN_RUN_FRAMES:
            continue
        if out and s - out[-1][1] < BRIDGE_FRAMES:
            out[-1][1] = int(e)
        else:
            out.append([int(s), int(e)])
    return [[round(s * HOP_MS / 1000, 2), round(e * HOP_MS / 1000, 2)] for s, e in out]


def write_mix(chans: list[np.ndarray], sr: int, dest: Path, target_sr: int = 16000) -> None:
    """The single-microphone mix: what ONE device in the room actually hears.

    Summed then peak-normalised to -3 dBFS. Normalising matters — the loop's
    whole ladder is relative to the speaker's own running baseline, so an
    arbitrary absolute level would be measuring the mixdown, not the meeting
    (the lesson from activation v1, which had learned "louder microphone =
    louder person").
    """
    n = min(len(c) for c in chans)
    mix = np.sum([c[:n] for c in chans], axis=0)
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 10 ** (-3.0 / 20.0)
    if sr != target_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, target_sr)
        mix = resample_poly(mix, target_sr // g, sr // g)
    pcm = np.clip(mix * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(target_sr)
        w.writeframes(pcm.tobytes())


def build(meeting: str, out_dir: Path, keep_headsets: bool) -> dict | None:
    print(f"{meeting}:", flush=True)
    heads = fetch_headsets(meeting, out_dir / "headsets")
    if len(heads) < 2:
        print("  skipped (needs at least 2 channels)")
        return None
    chans, sr = [], None
    for p in heads:
        c, sr = read_mono(p)
        chans.append(c)
    active, frames = speaking_mask(chans, sr)
    mix_path = out_dir / f"{meeting}.mix16k.wav"
    write_mix(chans, sr, mix_path)
    if not keep_headsets:
        for p in heads:
            p.unlink(missing_ok=True)

    speakers = {f"S{i}": intervals(active[i]) for i in range(len(chans))}
    dur = frames * HOP_MS / 1000
    overlap = float((active.sum(axis=0) >= 2).mean())
    meta = {
        "meeting": meeting,
        "mix": mix_path.name,
        "duration_s": round(dur, 1),
        "n_speakers": len(chans),
        "overlap_fraction": round(overlap, 4),
        "speakers": speakers,
        "speaking_seconds": {k: round(sum(e - s for s, e in v), 1) for k, v in speakers.items()},
    }
    print(f"  {dur/60:.1f} min · {len(chans)} speakers · overlap {overlap*100:.1f}%")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*", default=DEFAULT_MEETINGS)
    ap.add_argument("--out", default=str(REPO / "tmp/ami-corpus"))
    ap.add_argument("--keep-headsets", action="store_true",
                    help="keep the per-speaker files (~160 MB/meeting); off by default since "
                         "the ground truth derived from them is what the harness consumes")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"meetings": []}
    done = {m["meeting"] for m in manifest["meetings"]}

    for m in args.meetings:
        if m in done:
            print(f"{m}: ✓ already built")
            continue
        try:
            meta = build(m, out_dir, args.keep_headsets)
        except Exception as exc:                      # one bad meeting must not lose the rest
            print(f"  FAILED: {exc}")
            continue
        if meta:
            manifest["meetings"].append(meta)
            manifest_path.write_text(json.dumps(manifest, indent=1))

    total = sum(m["duration_s"] for m in manifest["meetings"])
    print(f"\n{len(manifest['meetings'])} meetings · {total/3600:.1f} h of real conversation")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
