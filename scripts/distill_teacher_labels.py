"""Teacher-label our real-conversation audio for the watch-sized distilled "heat" model.

Step 6 of docs/decisions/2026-09-20-heat-judge-plan.md's roadmap: before a
72 K-param student can be trained, every 2 s window of speech across the real
corpora needs a teacher label. The teacher is the two MIT/Apache models
already in ``server/tone_id.py`` — never audeering's NC-licensed one:

    classify_pcm(pcm, 16000)["scores"]  -> {arousal, dominance, valence}   (odyssey_dim, WavLM)
    angry_vote(pcm, sr)                 -> P(angry) - P(happy) | None      (IEMOCAP, SpeechBrain)
    stacked_heat_score(dims, angry_p)   -> 0..1                            (fixed logistic fit)

Windows are cut 2 s / 1 s hop, gated on speech: a corpus manifest that carries
per-speaker intervals (AMI, SBCSAE, CHiME-6) uses them directly; CONFER (no
speaker truth) and CREMA-D (single-utterance clips, no manifest at all) fall
back to the same -45 dBFS silence floor ``conversation_audit.py`` ships.

Resumable at RECORDING granularity: one parquet per recording
(``tmp/distill/labels/<corpus>/<recording>.parquet``), written atomically
(tmp file + rename) and skipped on a re-run the moment it exists — exactly the
pattern ``ami_corpus.py`` / ``sbcsae_corpus.py`` already use, so a labelling
job surviving only a few foreground minutes at a time still makes monotonic
progress.

**Breadth over depth.** The brief's priority order is "SBCSAE + CHiME-6 +
CONFER + CREMA-D, AMI last if time allows" — but SBCSAE alone is ~23 h of
audio, enough to consume the entire 3 CPU-hour budget on its own (at ~190 ms/
window: dims ~130 ms + angry_vote ~59 ms, measured in
docs/decisions/2026-09-20-heat-judge-plan.md) and leave CONFER (the one
corpus with human-rated ground truth) and CREMA-D (the one corpus with true
angry/happy labels) untouched. So recordings are scheduled ROUND-ROBIN across
corpora, one recording at a time, in the stated priority order — every
corpus gets some coverage before any one corpus can exhaust the clock. AMI is
scheduled last, one recording at a time, same as the others.

    python scripts/distill_teacher_labels.py --max-seconds 540
    -> tmp/distill/labels/<corpus>/<recording>.parquet (+ tmp/distill/labels/_progress.json)
"""
from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
import pandas as pd

import os
import sys

#: Code (this script, server/tone_id.py) lives in the worktree; DATA — corpora,
#: the tone-model cache, and every distillation output — lives in the MAIN repo,
#: because the worktree's gitignored tmp/ is a separate, empty directory (a
#: worktree does not inherit another worktree's ignored files). Never conflate
#: the two: importing tone_id from the wrong repo would silently run stale code,
#: and writing data into the worktree's tmp/ would silently lose it.
WORKTREE_REPO = Path(__file__).resolve().parent.parent
MAIN_REPO = Path("/Users/sagearbor/projects/githubs/mindshift")

os.environ.setdefault("MINDSHIFT_TONE_CACHE", str(MAIN_REPO / "server/.tone_cache"))
os.environ.setdefault("MINDSHIFT_TONE_AUDIO", "dark")   # compute + log only; never surfaces
sys.path.insert(0, str(WORKTREE_REPO / "server"))
import tone_id  # noqa: E402

OUT = MAIN_REPO / "tmp/distill/labels"
WINDOW_S = 2.0
HOP_S = 1.0
SR = 16000
SILENCE_FLOOR_DBFS = -45.0     # mirrors scripts/conversation_audit.py SentinelDetector
MIN_SPEECH_FRAC = 0.5          # of the window's duration, when speaker intervals exist

#: The brief's stated priority; AMI explicitly de-prioritised (round 2 of
#: docs/decisions/2026-09-20-heat-judge-plan.md already treats it as the
#: office-meeting corpus with the least emotional range on the graph).
CORPUS_PRIORITY = ["sbcsae", "chime6", "confer", "cremad", "ami"]

COLUMNS = ["window_start", "arousal", "valence", "dominance", "angry_p", "heat", "rms_db", "audio_path"]


# --------------------------------------------------------------------- audio --

def read_wave_mono16(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if sr != SR or sw != 2:
            raise ValueError(f"{path}: expected {SR} Hz / 16-bit, got {sr} Hz / {sw*8}-bit")
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x


def rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return 20.0 * np.log10(max(rms, 1e-12))


def speaking_fraction(speakers: dict | None, t0: float, t1: float) -> float | None:
    if not speakers:
        return None
    covered = 0.0
    for spans in speakers.values():
        for a, b in spans:
            covered += max(0.0, min(b, t1) - max(a, t0))
    return min(1.0, covered / (t1 - t0))


def is_speech_window(speakers: dict | None, t0: float, t1: float, sl: np.ndarray) -> bool:
    frac = speaking_fraction(speakers, t0, t1)
    if frac is not None:
        return frac >= MIN_SPEECH_FRAC
    return rms_dbfs(sl) > SILENCE_FLOOR_DBFS


def iter_windows(duration_s: float):
    t = 0.0
    while t + WINDOW_S <= duration_s:
        yield t
        t += HOP_S


# ----------------------------------------------------------------- manifests --

def _from_heatmap(mpath: Path) -> list[dict]:
    if not mpath.exists():
        return []
    base = mpath.parent
    out = []
    for e in json.loads(mpath.read_text()):
        audio = base / e["audio"]
        out.append({"id": e["id"], "audio": audio, "speakers": e.get("speakers"),
                    "duration_s": e.get("duration_s")})
    return out


def load_manifests() -> dict[str, list[dict]]:
    entries: dict[str, list[dict]] = {c: [] for c in CORPUS_PRIORITY}
    entries["sbcsae"] = _from_heatmap(MAIN_REPO / "tmp/corpora/sbcsae/heatmap_manifest.json")
    entries["chime6"] = _from_heatmap(MAIN_REPO / "tmp/corpora/chime6/heatmap_manifest.json")
    entries["confer"] = _from_heatmap(MAIN_REPO / "tmp/corpora/confer/heatmap_manifest.json")

    ami_mp = MAIN_REPO / "tmp/ami-corpus/manifest.json"
    if ami_mp.exists():
        m = json.loads(ami_mp.read_text())
        entries["ami"] = [{
            "id": x["meeting"], "audio": MAIN_REPO / "tmp/ami-corpus" / x["mix"],
            "speakers": x["speakers"], "duration_s": x["duration_s"],
        } for x in m["meetings"]]

    cremad_dir = MAIN_REPO / "tmp/corpora/cremad/AudioWAV"
    if cremad_dir.exists():
        for p in sorted(cremad_dir.glob("*.wav")):
            entries["cremad"].append({"id": p.stem, "audio": p, "speakers": None, "duration_s": None})
    return entries


# ------------------------------------------------------------------ labelling --

def process_recording(entry: dict, out_path: Path) -> tuple[str, int, float]:
    """Returns (status, n_windows, seconds_spent). Never partially writes —
    a tmp file is renamed into place only once every window is scored, so a
    killed run leaves no corrupt parquet for the next run to trust."""
    audio_path = Path(entry["audio"])
    if not audio_path.exists():
        return "missing-audio", 0, 0.0
    t_start = time.time()
    try:
        pcm = read_wave_mono16(audio_path)
    except Exception as exc:                                   # one bad file must not kill the run
        return f"read-error: {exc}", 0, time.time() - t_start
    duration = len(pcm) / SR
    speakers = entry.get("speakers")
    rows = []
    for t0 in iter_windows(duration):
        a0, a1 = int(round(t0 * SR)), int(round(t0 * SR)) + int(WINDOW_S * SR)
        sl = pcm[a0:a1]
        if len(sl) < int(WINDOW_S * SR):
            continue
        if not is_speech_window(speakers, t0, t0 + WINDOW_S, sl):
            continue
        try:
            dims = tone_id.classify_pcm(sl, SR)["scores"]
            angry_p = tone_id.angry_vote(sl, SR)
        except tone_id.ToneUnavailable:
            continue
        heat = tone_id.stacked_heat_score(dims, angry_p)
        rows.append({
            "window_start": round(t0, 2),
            "arousal": float(dims["arousal"]), "valence": float(dims["valence"]),
            "dominance": float(dims["dominance"]),
            "angry_p": None if angry_p is None else float(angry_p),
            "heat": float(heat), "rms_db": round(rms_dbfs(sl), 2),
            "audio_path": str(audio_path),
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp)
    tmp.replace(out_path)
    return ("ok" if rows else "no-speech"), len(rows), time.time() - t_start


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=540.0,
                     help="wall-clock budget for THIS invocation (repeat the command to keep going)")
    ap.add_argument("--corpora", nargs="*", default=CORPUS_PRIORITY)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(4)

    manifests = load_manifests()
    for c in args.corpora:
        if not manifests.get(c):
            print(f"  ({c}: no manifest / audio found, skipping)")

    # Round-robin queues per corpus, already-labelled recordings dropped up front.
    queues: dict[str, list[dict]] = {}
    for c in args.corpora:
        out_dir = OUT / c
        queues[c] = [e for e in manifests.get(c, []) if not (out_dir / f"{e['id']}.parquet").exists()]
    total_pending = sum(len(q) for q in queues.values())
    total_all = sum(len(manifests.get(c, [])) for c in args.corpora)
    print(f"{total_all - total_pending}/{total_all} recordings already labelled; "
          f"{total_pending} pending across {', '.join(args.corpora)}")

    t_start = time.time()
    n_done = n_windows = 0
    per_corpus = {c: {"done": 0, "windows": 0} for c in args.corpora}
    stop_reason = None
    while True:
        active = [c for c in args.corpora if queues[c]]
        if not active:
            stop_reason = "all corpora exhausted"
            break
        elapsed = time.time() - t_start
        if elapsed > args.max_seconds:
            stop_reason = f"time budget reached ({elapsed:.0f}s)"
            break
        progressed = False
        for c in active:
            elapsed = time.time() - t_start
            if elapsed > args.max_seconds:
                break
            entry = queues[c].pop(0)
            out_path = OUT / c / f"{entry['id']}.parquet"
            status, n, dt = process_recording(entry, out_path)
            progressed = True
            if status == "ok":
                n_done += 1
                n_windows += n
                per_corpus[c]["done"] += 1
                per_corpus[c]["windows"] += n
                rate = n_windows / max(time.time() - t_start, 1e-6)
                remaining_est = sum(len(q) for q in queues.values())
                eta_s = remaining_est / max(n_done / max(time.time() - t_start, 1e-6), 1e-6)
                print(f"  [{c}] {entry['id']}: {n} windows in {dt:.1f}s "
                      f"({rate:.1f} win/s overall; ~{remaining_est} recordings left, "
                      f"ETA {eta_s/60:.1f} min at this pace)", flush=True)
            elif status == "no-speech":
                per_corpus[c]["done"] += 1
                print(f"  [{c}] {entry['id']}: no speech windows above floor ({dt:.1f}s)")
            else:
                print(f"  [{c}] {entry['id']}: {status}")
        if not progressed:
            stop_reason = "all corpora exhausted"
            break

    elapsed_total = time.time() - t_start
    remaining = {c: len(queues[c]) for c in args.corpora if queues[c]}
    print(f"\n{elapsed_total:.0f}s this invocation ({elapsed_total/3600:.2f} CPU-h) — "
          f"{n_done} recordings, {n_windows} windows labelled")
    print(f"stopped: {stop_reason}")
    if remaining:
        print(f"still pending: {remaining} — rerun this command to continue (resumable, per-recording)")
    else:
        print("all requested corpora fully labelled")

    progress_path = OUT / "_progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    prior = json.loads(progress_path.read_text()) if progress_path.exists() else {"runs": []}
    prior["runs"].append({
        "seconds": round(elapsed_total, 1), "recordings_done": n_done, "windows": n_windows,
        "per_corpus_this_run": per_corpus, "stop_reason": stop_reason,
    })
    prior["cumulative_seconds"] = round(sum(r["seconds"] for r in prior["runs"]), 1)
    prior["remaining_recordings"] = remaining
    progress_path.write_text(json.dumps(prior, indent=1))
    print(f"progress log: {progress_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
