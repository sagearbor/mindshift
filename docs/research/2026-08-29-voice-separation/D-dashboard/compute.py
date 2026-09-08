"""D-dashboard: per-fixture voice-feature computation for the separability
dashboard (see README.md and build_html.py).

For every fixture this writes ``D-dashboard/data_<fixture>.json`` with:

* ``frames``  — 100 ms hop acoustic features (F0, spectral centroid, spectral
  tilt, RMS dB, F1/F2/F3) for every NON-silent frame, each tagged with its
  ground-truth speaker (or null outside GT speech);
* ``embed``   — pinned-ECAPA embeddings of 1.5 s windows (hop 0.25 s) projected
  to 2-D by PCA and t-SNE, tagged with the GT speaker at the window centre;
* ``pooled``  — per-GT-speaker POOLED embeddings' cosine-similarity matrix;
* ``prod``    — what ``diarize_local.diarize_turns`` (production today) says
  when handed the GT intervals as turns (plus, for maggiano3, the two stored
  Deepgram transcripts), each scored with the shared scorer;
* ``sep``     — per-feature separability (between-speaker spread of medians vs
  within-speaker IQR, plus a nearest-median single-feature frame accuracy).

Run:  tmp/venv-voice/bin/python docs/research/2026-08-29-voice-separation/D-dashboard/compute.py [fixture ...]
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
ROOT = RESEARCH.parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(RESEARCH))

import librosa  # noqa: E402
import parselmouth  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

import audio_ingest  # noqa: E402
import diarize_local  # noqa: E402
import speaker_id  # noqa: E402
import score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("D-dashboard")

SR = 16000
HOP_S = 0.1
HOP = int(SR * HOP_S)
N_FFT = 1024
WIN_S, WIN_HOP_S = 1.5, 0.25

FEATURES = {
    "f0": {"label": "Pitch (F0)", "unit": "Hz"},
    "centroid": {"label": "Spectral centroid", "unit": "Hz"},
    "tilt": {"label": "Spectral tilt", "unit": "dB/kHz"},
    "energy": {"label": "Energy (RMS)", "unit": "dB"},
    "f1": {"label": "Formant F1", "unit": "Hz"},
    "f2": {"label": "Formant F2", "unit": "Hz"},
    "f3": {"label": "Formant F3", "unit": "Hz"},
}

DEFAULT_FIXTURES = ["poker6", "family_real", "maggiano3", "scene_family3",
                    "scene_meeting4", "openai", "gptaudio", "scene_couple"]


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

def load_audio(fx: dict) -> np.ndarray:
    path = Path(fx["audio_path"])
    data = path.read_bytes()
    pcm, sr = audio_ingest.decode_to_pcm_16k(data, path.name)
    assert sr == SR, sr
    return np.ascontiguousarray(pcm, dtype=np.float32)


OVERLAP = "overlap"


def is_overlap(label) -> bool:
    return isinstance(label, (tuple, list))


def label_at(gt: list, t: float) -> str | None:
    """GT speaker at time ``t``; an overlap segment (several speakers at once)
    collapses to the single pseudo-label :data:`OVERLAP` so it gets its own
    colour and stays OUT of every per-speaker statistic."""
    for s, e, l in gt:
        if s <= t < e:
            return OVERLAP if is_overlap(l) else l
    return None


def label_str(label) -> str:
    return "+".join(label) if is_overlap(label) else str(label)


# ---------------------------------------------------------------------------
# frame features
# ---------------------------------------------------------------------------

def frame_features(pcm: np.ndarray, gt: list) -> tuple[list[dict], dict]:
    n_frames = 1 + pcm.size // HOP
    times = np.arange(n_frames) * HOP_S

    rms = librosa.feature.rms(y=pcm, frame_length=N_FFT, hop_length=HOP, center=True)[0]
    rms_db = 20 * np.log10(rms + 1e-9)
    S = np.abs(librosa.stft(pcm, n_fft=N_FFT, hop_length=HOP, center=True))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    centroid = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    # spectral tilt: least-squares slope of the dB spectrum vs frequency (kHz)
    # over 100 Hz .. 4 kHz — the classic "bright vs dark" timbre number.
    band = (freqs >= 100) & (freqs <= 4000)
    fk = freqs[band] / 1000.0
    P = 20 * np.log10(S[band, :] + 1e-9)
    fk_c = fk - fk.mean()
    tilt = (fk_c[:, None] * (P - P.mean(axis=0))).sum(axis=0) / (fk_c ** 2).sum()

    snd = parselmouth.Sound(pcm.astype(np.float64), sampling_frequency=SR)
    pitch = snd.to_pitch(time_step=HOP_S, pitch_floor=60, pitch_ceiling=600)
    formant = snd.to_formant_burg(time_step=HOP_S, max_number_of_formants=5,
                                  maximum_formant=5500)

    n = min(n_frames, rms_db.size, centroid.size, tilt.size)
    # energy VAD: a frame is speech if it sits clearly above the recording's
    # noise floor (10th percentile) — 6 dB keeps quiet, far-from-mic voices
    # (poker6's Player1 sits ~20 dB under the loudest player) while still
    # dropping the pauses.
    floor = float(np.percentile(rms_db[:n], 10))
    peak = float(np.percentile(rms_db[:n], 98))
    thr = max(floor + 6.0, peak - 42.0)
    frames = []
    for i in range(n):
        t = float(times[i])
        if rms_db[i] < thr:
            continue
        f0 = pitch.get_value_at_time(t)
        row = {
            "t": round(t, 2),
            "spk": label_at(gt, t + HOP_S / 2),
            "f0": None if (f0 is None or np.isnan(f0)) else round(float(f0), 1),
            "centroid": round(float(centroid[i]), 0),
            "tilt": round(float(tilt[i]), 2),
            "energy": round(float(rms_db[i]), 1),
        }
        for k, fn in ((1, "f1"), (2, "f2"), (3, "f3")):
            v = formant.get_value_at_time(k, t)
            row[fn] = None if (v is None or np.isnan(v)) else round(float(v), 0)
        # formants are only meaningful on voiced frames
        if row["f0"] is None:
            row["f1"] = row["f2"] = row["f3"] = None
        frames.append(row)
    vad = {"threshold_db": round(thr, 1), "floor_db": round(floor, 1),
           "frames_total": int(n), "frames_speech": len(frames)}
    return frames, vad


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------

def window_embeddings(pcm: np.ndarray, gt: list, name: str) -> dict:
    win, hop = int(WIN_S * SR), int(WIN_HOP_S * SR)
    starts = list(range(0, max(1, pcm.size - win + 1), hop))
    # The ECAPA pass is the slow part (~1 min per fixture); cache the raw
    # 192-d window embeddings so re-runs that only change downstream maths
    # (projection, labels, VAD) are instant. Cache key = fixture + window grid.
    cache = HERE / f"emb_{name}_{WIN_S}s_{WIN_HOP_S}s.npy"
    if cache.exists():
        X = np.load(cache)
        if X.shape[0] != len(starts):
            X = None
    else:
        X = None
    if X is None:
        chunks = [pcm[s:s + win] for s in starts]
        t0 = time.time()
        embs = speaker_id.embed_pcm_batch(chunks, SR)
        log.info("embedded %d windows in %.1fs", len(chunks), time.time() - t0)
        X = np.stack(embs).astype(np.float32)
        np.save(cache, X)
    centers = [(s + win / 2) / SR for s in starts]
    labels = [label_at(gt, c) for c in centers]

    pca = PCA(n_components=3, random_state=0).fit(X)
    P = pca.transform(X)
    perp = max(5, min(30, (len(X) - 1) // 4))
    T = TSNE(n_components=2, perplexity=perp, random_state=0, init="pca",
             metric="cosine").fit_transform(X)
    lab_idx = [i for i, l in enumerate(labels) if l is not None and l != OVERLAP]
    sil = None
    if len({labels[i] for i in lab_idx}) >= 2 and len(lab_idx) > 3:
        sil = float(silhouette_score(X[lab_idx], [labels[i] for i in lab_idx], metric="cosine"))
    # window-level nearest-pooled-centroid accuracy is computed by the caller
    return {
        "window_s": WIN_S, "hop_s": WIN_HOP_S,
        "points": [
            {"t": round(c, 2), "spk": l,
             "pca": [round(float(P[i, 0]), 4), round(float(P[i, 1]), 4), round(float(P[i, 2]), 4)],
             "tsne": [round(float(T[i, 0]), 3), round(float(T[i, 1]), 3)]}
            for i, (c, l) in enumerate(zip(centers, labels))
        ],
        "pca_explained": [round(float(v), 3) for v in pca.explained_variance_ratio_],
        "silhouette_cosine": None if sil is None else round(sil, 3),
        "_X": X, "_labels": labels,
    }


def pooled_matrix(pcm: np.ndarray, gt: list, speakers: list[str]) -> dict:
    # overlap segments carry a joined label that matches no single speaker,
    # so they are excluded from every pooled voiceprint
    turns = [{"speaker": label_str(l), "start_time": s, "end_time": e} for s, e, l in gt]
    vecs = {}
    for spk in speakers:
        pooled = speaker_id.pool_speaker_pcm(pcm, SR, turns, spk)
        vecs[spk] = speaker_id.embed_pcm(pooled, SR)
    mat = [[round(float(speaker_id.cosine(vecs[a], vecs[b])), 3) for b in speakers]
           for a in speakers]
    return {"speakers": speakers, "cosine": mat, "_vecs": vecs}


# ---------------------------------------------------------------------------
# production diarizer
# ---------------------------------------------------------------------------

def run_production(pcm: np.ndarray, gt: list, owner: str | None, turns: list[dict],
                   name: str) -> dict:
    t0 = time.time()
    res = diarize_local.diarize_turns(pcm, SR, turns)
    dt = time.time() - t0
    if res is None:
        pred = [(float(t["start_time"]), float(t["end_time"]), "Speaker A") for t in turns]
        out = {"name": name, "k_pred": 1, "returned_none": True, "seconds": round(dt, 1),
               "segments": [[s, e, l] for s, e, l in pred]}
    else:
        pred = [(float(t["start_time"]), float(t["end_time"]), t["speaker"]) for t in res["turns"]]
        out = {"name": name, "k_pred": res["num_speakers"], "returned_none": False,
               "seconds": round(dt, 1), "pooled_cosine": res["pooled_cosine"],
               "split_utterances": res["split_utterances"],
               "k_evaluated": res["k_evaluated"],
               "segments": [[s, e, l] for s, e, l in pred]}
    sc = score.score_segments(gt, pred, owner)
    out["score"] = sc
    return out


def gt_as_turns(gt: list) -> list[dict]:
    return [{"speaker": label_str(l), "text": "…", "start_time": s, "end_time": e} for s, e, l in gt]


# ---------------------------------------------------------------------------
# separability
# ---------------------------------------------------------------------------

def separability(frames: list[dict], speakers: list[str]) -> dict:
    out = {}
    for feat in FEATURES:
        per = {}
        for spk in speakers:
            vals = np.array([f[feat] for f in frames if f["spk"] == spk and f[feat] is not None], float)
            if vals.size < 3:
                continue
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            p10, p90 = np.percentile(vals, [10, 90])
            per[spk] = {"n": int(vals.size), "median": round(float(med), 2),
                        "q1": round(float(q1), 2), "q3": round(float(q3), 2),
                        "p10": round(float(p10), 2), "p90": round(float(p90), 2)}
        if len(per) < 2:
            out[feat] = {"per_speaker": per, "ratio": None, "accuracy": None}
            continue
        meds = np.array([p["median"] for p in per.values()])
        iqrs = np.array([p["q3"] - p["q1"] for p in per.values()])
        between = float(meds.std(ddof=1)) if len(meds) > 1 else 0.0
        within = float(np.mean(iqrs) / 1.349) or 1e-9  # IQR → sigma-equivalent
        ratio = between / within
        # nearest-median single-feature classifier, frame accuracy over labelled frames
        spks = list(per)
        correct = total = 0
        for f in frames:
            if f["spk"] not in per or f[feat] is None:
                continue
            total += 1
            guess = min(spks, key=lambda s: abs(f[feat] - per[s]["median"]))
            correct += guess == f["spk"]
        # smallest gap between any two speaker medians, in within-sigma units
        srt = np.sort(meds)
        min_gap = float(np.min(np.diff(srt))) / within if len(srt) > 1 else None
        out[feat] = {"per_speaker": per, "ratio": round(ratio, 2),
                     "min_gap_sigma": None if min_gap is None else round(min_gap, 2),
                     "accuracy": round(correct / total, 3) if total else None,
                     "chance": round(1 / len(spks), 3)}
    return out


def embedding_accuracy(emb: dict, pooled: dict) -> float | None:
    X, labels = emb["_X"], emb["_labels"]
    vecs = pooled["_vecs"]
    spks = pooled["speakers"]
    C = np.stack([vecs[s] for s in spks])
    correct = total = 0
    for x, l in zip(X, labels):
        if l not in vecs:
            continue
        total += 1
        correct += spks[int(np.argmax(C @ x))] == l
    return round(correct / total, 3) if total else None


# ---------------------------------------------------------------------------

def process(name: str) -> dict:
    fx = score.load_fixture(name)
    gt = [(float(s), float(e), tuple(l) if is_overlap(l) else str(l)) for s, e, l in fx["gt"]]
    speakers = []
    for _, _, l in gt:
        if not is_overlap(l) and l not in speakers:
            speakers.append(l)
    n_overlap = sum(1 for _, _, l in gt if is_overlap(l))
    pcm = load_audio(fx)
    log.info("%s: %.1fs audio, %d GT intervals, %d speakers", name, pcm.size / SR, len(gt), len(speakers))

    frames, vad = frame_features(pcm, gt)
    log.info("%s: %d speech frames (thr %.1f dB)", name, len(frames), vad["threshold_db"])
    emb = window_embeddings(pcm, gt, name)
    pooled = pooled_matrix(pcm, gt, speakers)
    emb_acc = embedding_accuracy(emb, pooled)

    prods = [run_production(pcm, gt, fx["owner_label"], gt_as_turns(gt), "GT intervals as turns")]
    for tp in fx.get("transcripts", []):
        turns = json.loads(Path(tp).read_text())
        prods.append(run_production(pcm, gt, fx["owner_label"], turns,
                                    f"stored transcript {Path(tp).stem.replace('transcript_', '')}"))
    for p in prods:
        log.info("%s / %s: k=%d acc=%.3f (%.1fs)", name, p["name"], p["k_pred"],
                 p["score"]["frame_accuracy"], p["seconds"])

    sep = separability(frames, speakers)
    off = [[s, e, l] for s, e, l in pooled_cosine_offdiag(pooled)]
    data = {
        "fixture": name, "k_true": fx["k_true"], "owner": fx["owner_label"],
        "duration_s": round(pcm.size / SR, 2), "speakers": speakers,
        "gt": [[s, e, list(l) if is_overlap(l) else l] for s, e, l in gt],
        "overlap_label": OVERLAP, "overlap_segments": n_overlap,
        "vad": vad, "hop_s": HOP_S,
        "features": FEATURES, "frames": frames,
        "embed": {k: v for k, v in emb.items() if not k.startswith("_")},
        "pooled": {"speakers": speakers, "cosine": pooled["cosine"], "pairs": off,
                   "nearest_centroid_window_accuracy": emb_acc},
        "prod": prods, "sep": sep,
        "private": name == "maggiano3",
    }
    return data


def pooled_cosine_offdiag(pooled: dict):
    spks, mat = pooled["speakers"], pooled["cosine"]
    for i in range(len(spks)):
        for j in range(i + 1, len(spks)):
            yield spks[i], spks[j], mat[i][j]


def main(argv: list[str]) -> None:
    names = argv or [n for n in DEFAULT_FIXTURES if n in score.all_fixtures()]
    for name in names:
        t0 = time.time()
        data = process(name)
        out = HERE / f"data_{name}.json"
        out.write_text(json.dumps(data, separators=(",", ":")))
        log.info("wrote %s (%.0f KB) in %.0fs", out.name, out.stat().st_size / 1024, time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1:])
