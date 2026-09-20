"""Approach A — hand-crafted acoustic features for speaker separation.

Pipeline (per fixture):
  1. Decode to 16 kHz mono.  Energy VAD (frame RMS vs. a noise-floor
     percentile) marks speech frames.
  2. Frame-level tracks computed ONCE on the whole file: Praat F0 (10 ms),
     Praat formants F1-F3 (Burg), librosa MFCC-13, spectral centroid /
     rolloff / tilt, RMS.
  3. Sliding windows (WIN s, HOP s) over speech; each window aggregates the
     tracks (median F0, IQR F0, voiced fraction, mean centroid/tilt/rolloff,
     mean+std MFCC, median formants, RMS).  Windows with < MIN_VOICED voiced
     frames are dropped for the pitch-based variants.
  4. Feature variants (ablation) → standardize → agglomerative clustering
     (Ward) at k = k_true (oracle) and k = auto (silhouette over 2..8).
  5. Temporal smoothing: majority vote over a ±SMOOTH-window neighbourhood,
     then merge runs shorter than MIN_RUN s into their neighbours.
  6. Windows → (start, end, label) segments → shared scorer.

Outputs (this directory): features_<fixture>.csv, pred_<fixture>.json (the
`full` variant at auto-k), pred_<fixture>_<variant>_<mode>.json for every run,
results.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import parselmouth
import librosa
from scipy.io import wavfile
from scipy.signal import medfilt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
ROOT = RESEARCH.parents[2]
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(ROOT / "server"))

import score as scorer  # noqa: E402

SR = 16000
FRAME = 0.01          # 10 ms analysis grid (matches Praat pitch time step)
WIN = 0.75            # window seconds
HOP = 0.20            # hop seconds
MIN_VOICED = 0.15     # min voiced fraction for a window to carry a trusted pitch
MIN_SPEECH = 0.4      # min fraction of VAD-speech frames inside a window
VAD_FRAC = 0.22       # VAD threshold = floor + VAD_FRAC * (peak - floor)  [dB]
GAP_FILL = 1.0        # fill unlabelled gaps shorter than this between segments (s)
WINSOR = 2.0          # winsorize features at this percentile / 100-this
SIL_TOL = 0.03        # auto-k: smallest k whose silhouette is within SIL_TOL of the best
SMOOTH = 2            # ± windows for majority-vote smoothing
MIN_RUN = 0.6         # merge label runs shorter than this (s)
KMAX = 8

VARIANTS = {
    "pitch":      ["f0_med", "f0_iqr"],
    "pitch_spec": ["f0_med", "f0_iqr", "centroid", "tilt", "rolloff"],
    "pitch_form": ["f0_med", "f0_iqr", "f1", "f2", "f3"],
    "mfcc":       [f"mfcc{i}_m" for i in range(13)] + [f"mfcc{i}_s" for i in range(13)],
    "full":       ["f0_med", "f0_iqr", "centroid", "tilt", "rolloff", "f1", "f2", "f3", "rms"]
                  + [f"mfcc{i}_m" for i in range(13)] + [f"mfcc{i}_s" for i in range(13)],
}


# ----------------------------------------------------------------- audio
def load_audio(fx: dict) -> np.ndarray:
    p = Path(fx["audio_path"])
    if p.suffix.lower() == ".wav":
        sr, y = wavfile.read(p)
        if y.dtype != np.float32:
            y = y.astype(np.float32) / np.iinfo(y.dtype).max
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        return y.astype(np.float32)
    import audio_ingest  # server/
    y, sr = audio_ingest.decode_to_pcm_16k(p.read_bytes(), "audio" + p.suffix)
    assert sr == SR
    return np.asarray(y, dtype=np.float32)


# ------------------------------------------------------------ frame tracks
def frame_tracks(y: np.ndarray) -> dict[str, np.ndarray]:
    """Every track is on the 10 ms grid with the same length n."""
    hop = int(SR * FRAME)
    n_fft = 512
    n = 1 + len(y) // hop
    # RMS / VAD
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop, center=True)[0]
    # Spectral
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    centroid = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=SR)[0]
    rolloff = librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=SR, roll_percent=0.85)[0]
    # Spectral tilt: slope of log-power vs. frequency (dB/kHz) over 100 Hz-4 kHz
    band = (freqs >= 100) & (freqs <= 4000)
    logS = 10 * np.log10(S[band] + 1e-10)
    f_khz = freqs[band] / 1000.0
    f_c = f_khz - f_khz.mean()
    tilt = (f_c[:, None] * (logS - logS.mean(axis=0))).sum(axis=0) / (f_c ** 2).sum()
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, n_fft=n_fft, hop_length=hop, center=True)
    # Praat pitch + formants
    snd = parselmouth.Sound(y.astype(np.float64), SR)
    pitch = snd.to_pitch(time_step=FRAME, pitch_floor=70.0, pitch_ceiling=500.0)
    f0_raw = pitch.selected_array["frequency"]           # 0 = unvoiced
    t_p = pitch.xs()
    formant = snd.to_formant_burg(time_step=FRAME, max_number_of_formants=5,
                                  maximum_formant=5500.0)
    t_f = np.arange(formant.get_number_of_frames()) * formant.get_time_step() + formant.get_time_from_frame_number(1)
    fm = np.array([[formant.get_value_at_time(k, t) for k in (1, 2, 3)] for t in t_f])
    # re-grid Praat tracks onto the 10 ms grid (nearest)
    grid = np.arange(n) * FRAME
    f0 = np.zeros(n)
    idx = np.clip(np.round((grid - t_p[0]) / FRAME).astype(int), 0, len(f0_raw) - 1)
    f0 = f0_raw[idx]
    idf = np.clip(np.round((grid - t_f[0]) / FRAME).astype(int), 0, len(fm) - 1)
    f123 = fm[idf]

    def fit(a):
        a = np.asarray(a)
        if a.ndim == 1:
            out = np.zeros(n); m = min(n, len(a)); out[:m] = a[:m]; return out
        out = np.zeros((a.shape[0], n)); m = min(n, a.shape[1]); out[:, :m] = a[:, :m]; return out

    return {
        "rms": fit(rms), "centroid": fit(centroid), "rolloff": fit(rolloff),
        "tilt": fit(tilt), "mfcc": fit(mfcc), "f0": f0,
        "f1": f123[:, 0], "f2": f123[:, 1], "f3": f123[:, 2],
    }


def energy_vad(rms: np.ndarray) -> np.ndarray:
    db = 20 * np.log10(rms + 1e-8)
    floor = np.percentile(db, 10)
    peak = np.percentile(db, 95)
    thr = floor + VAD_FRAC * (peak - floor)
    speech = db > thr
    # smooth: close 100 ms gaps, drop blips < 100 ms
    speech = medfilt(speech.astype(int), kernel_size=11).astype(bool)
    return speech


# ------------------------------------------------------------ windows
def window_features(tr: dict, speech: np.ndarray, gt) -> tuple[list[dict], np.ndarray]:
    n = len(speech)
    w, h = int(WIN / FRAME), int(HOP / FRAME)
    gt_frames = scorer._frames(gt, n)
    rows = []
    for s in range(0, n - w + 1, h):
        e = s + w
        sp = speech[s:e]
        if sp.mean() < MIN_SPEECH:
            continue
        f0 = tr["f0"][s:e]
        f0 = f0[(f0 > 0) & sp]
        voiced = len(f0) / w
        sel = sp
        row = {
            "time": round((s + w / 2) * FRAME, 3), "start": round(s * FRAME, 3), "end": round(e * FRAME, 3),
            "speaker_gt": max(set(gt_frames[s:e]) - {None}, key=gt_frames[s:e].count) if any(g is not None for g in gt_frames[s:e]) else "",
            "voiced": round(voiced, 3),
            "f0": round(float(np.median(f0)), 1) if len(f0) else float("nan"),
            # semitones re 100 Hz; IQR clipped at one octave so octave errors can't dominate
            "f0_med": float(12 * np.log2(np.median(f0) / 100.0)) if len(f0) else float("nan"),
            "f0_iqr": float(min(12.0, 12 * np.log2(np.percentile(f0, 75) / max(np.percentile(f0, 25), 1.0)))) if len(f0) else float("nan"),
            "centroid": float(tr["centroid"][s:e][sel].mean()),
            "tilt": float(tr["tilt"][s:e][sel].mean()),
            "rolloff": float(tr["rolloff"][s:e][sel].mean()),
            "rms": float(20 * np.log10(tr["rms"][s:e][sel].mean() + 1e-8)),
        }
        for k in ("f1", "f2", "f3"):
            v = tr[k][s:e]
            v = v[sel & (tr["f0"][s:e] > 0) & np.isfinite(v)]
            row[k] = float(np.median(v)) if len(v) else float("nan")
        m = tr["mfcc"][:, s:e][:, sel]
        for i in range(13):
            row[f"mfcc{i}_m"] = float(m[i].mean())
            row[f"mfcc{i}_s"] = float(m[i].std())
        rows.append(row)
    return rows, gt_frames


def write_csv(rows: list[dict], path: Path):
    cols = ["time", "speaker_gt", "f0", "centroid", "tilt", "rms"]
    with path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if (isinstance(r[c], float) and np.isnan(r[c])) else str(r[c]) for c in cols) + "\n")


# ------------------------------------------------------------ clustering
def impute(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col) if np.isfinite(col).any() else 0.0
        lo, hi = np.percentile(col, [WINSOR, 100 - WINSOR])
        X[:, j] = np.clip(col, lo, hi)
    return X


def choose_k(Z: np.ndarray, kmax: int) -> tuple[int, dict]:
    scores = {}
    for k in range(2, min(kmax, len(Z) - 1) + 1):
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
        if len(set(lab)) < 2:
            continue
        scores[k] = float(silhouette_score(Z, lab))
    if not scores:
        return 1, scores
    best = max(scores.values())
    return min(k for k, v in scores.items() if v >= best - SIL_TOL), scores


def smooth_labels(lab: np.ndarray, times: np.ndarray) -> np.ndarray:
    lab = lab.copy()
    n = len(lab)
    # majority vote in ±SMOOTH windows, only across temporally-contiguous windows
    out = lab.copy()
    for i in range(n):
        lo, hi = max(0, i - SMOOTH), min(n, i + SMOOTH + 1)
        neigh = [lab[j] for j in range(lo, hi) if abs(times[j] - times[i]) <= (SMOOTH + 0.5) * HOP]
        vals, cnt = np.unique(neigh, return_counts=True)
        out[i] = vals[np.argmax(cnt)] if cnt.max() > len(neigh) / 2 else lab[i]
    lab = out
    # min-run merge
    changed = True
    while changed:
        changed = False
        runs = []
        s = 0
        for i in range(1, n + 1):
            if i == n or lab[i] != lab[s] or times[i] - times[i - 1] > HOP * 1.5:
                runs.append((s, i)); s = i
        for ri, (s, e) in enumerate(runs):
            dur = times[e - 1] - times[s] + WIN
            if dur < MIN_RUN and len(runs) > 1:
                # take neighbour with the longer run
                cands = []
                if ri > 0: cands.append(runs[ri - 1])
                if ri < len(runs) - 1: cands.append(runs[ri + 1])
                nb = max(cands, key=lambda r: r[1] - r[0])
                if lab[nb[0]] != lab[s]:
                    lab[s:e] = lab[nb[0]]; changed = True; break
    return lab


def to_segments(rows: list[dict], lab: np.ndarray) -> list[tuple[float, float, str]]:
    segs = []
    for r, l in zip(rows, lab):
        s, e, L = r["start"], r["end"], f"S{int(l)}"
        if segs and segs[-1][2] == L and s <= segs[-1][1] + 1e-6:
            segs[-1] = (segs[-1][0], e, L)
        else:
            # trim overlap with previous different-label segment: split at midpoint
            if segs and s < segs[-1][1]:
                mid = round((s + segs[-1][1]) / 2, 3)
                segs[-1] = (segs[-1][0], mid, segs[-1][2]); s = mid
            segs.append((s, e, L))
    # fill short unlabelled gaps: same label both sides -> that label; else split at midpoint
    filled = []
    for seg in segs:
        if filled and 0 < seg[0] - filled[-1][1] <= GAP_FILL:
            ps, pe, pl = filled[-1]
            if pl == seg[2]:
                filled[-1] = (ps, seg[1], pl); continue
            mid = round((pe + seg[0]) / 2, 3)
            filled[-1] = (ps, mid, pl); seg = (mid, seg[1], seg[2])
        filled.append(seg)
    return filled


def cluster_variant(rows: list[dict], cols: list[str], k: int | None) -> tuple[np.ndarray, dict]:
    times = np.array([r["time"] for r in rows])
    uses_pitch = any(c.startswith("f0") or c in ("f1", "f2", "f3") for c in cols)
    ok = np.array([(r["voiced"] >= MIN_VOICED) if uses_pitch else True for r in rows])
    if ok.sum() < 4:
        ok[:] = True
    X = impute(np.array([[r[c] for c in cols] for r in rows], dtype=float)[ok])
    Z = StandardScaler().fit_transform(X)
    info = {"n_clustered": int(ok.sum())}
    if k is None:
        k, sil = choose_k(Z, KMAX)
        info["silhouette"] = {str(a): round(b, 3) for a, b in sil.items()}
    lab_ok = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z) if k > 1 else np.zeros(len(Z), int)
    # propagate to windows that had no trusted pitch: nearest-in-time clustered window
    lab = np.zeros(len(rows), int)
    idx_ok = np.where(ok)[0]
    for i in range(len(rows)):
        lab[i] = lab_ok[np.argmin(np.abs(times[idx_ok] - times[i]))]
    lab = smooth_labels(lab, times)
    return lab, info


# ------------------------------------------------------------ main
def run_fixture(name: str) -> dict:
    fx = scorer.load_fixture(name)
    t0 = time.perf_counter()
    y = load_audio(fx)
    t_dec = time.perf_counter() - t0
    t0 = time.perf_counter()
    tr = frame_tracks(y)
    speech = energy_vad(tr["rms"])
    rows, _ = window_features(tr, speech, fx["gt"])
    t_feat = time.perf_counter() - t0
    write_csv(rows, HERE / f"features_{name}.csv")
    out = {"fixture": name, "k_true": fx["k_true"], "n_windows": len(rows),
           "duration_s": round(len(y) / SR, 2), "speech_frac": round(float(speech.mean()), 3),
           "runtime_s": {"decode": round(t_dec, 2), "features": round(t_feat, 2)}, "variants": {}}
    best = None
    for vname, cols in VARIANTS.items():
        # pitch variants: windows with too little voicing are unreliable — keep them but
        # they are imputed to the median (documented limitation).
        for mode in ("auto", "oracle"):
            t0 = time.perf_counter()
            lab, info = cluster_variant(rows, cols, None if mode == "auto" else fx["k_true"])
            pred = to_segments(rows, lab)
            sc = scorer.score_fixture(name, pred)
            sc["runtime_s"] = round(time.perf_counter() - t0, 3)
            sc.update(info)
            out["variants"][f"{vname}_{mode}"] = sc
            (HERE / f"pred_{name}_{vname}_{mode}.json").write_text(json.dumps([list(p) for p in pred]))
            if mode == "auto" and vname == "full":
                (HERE / f"pred_{name}.json").write_text(json.dumps([list(p) for p in pred]))
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or scorer.all_fixtures()
    results = {}
    for n in names:
        r = run_fixture(n)
        results[n] = r
        print(f"\n== {n}  k_true={r['k_true']} windows={r['n_windows']} speech={r['speech_frac']} "
              f"feat={r['runtime_s']['features']}s")
        for v, sc in r["variants"].items():
            print(f"  {v:18s} acc={sc['frame_accuracy']:.3f} k={sc['k_pred']}/{sc['k_true']} "
                  f"own={sc['owner_purity']} unl={sc['unlabelled_frac']:.2f} t={sc['runtime_s']}s")
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
