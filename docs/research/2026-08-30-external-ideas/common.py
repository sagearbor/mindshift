"""Shared helpers for the 2026-08-30 external-ideas experiments (AS-Norm,
overlap masking, denoising). Runs under tmp/venv-voice (torch/speechbrain).

Reuses, unchanged: the shared scorer (../2026-08-29-voice-separation/score.py),
approach B's window/VAD/clustering code (../2026-08-29-voice-separation/
B-sliding-window/run_b.py, with its embedding cache redirected to ./cache/)
and the production modules (server/speaker_id.py, server/diarize_local.py).
Nothing here touches server/ or apps/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent / "2026-08-29-voice-separation"
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(RESEARCH / "B-sliding-window"))

import score  # noqa: E402
import speaker_id  # noqa: E402
import run_b  # noqa: E402

CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
run_b.CACHE = CACHE           # B's window-embedding cache lives HERE (gitignored)
PRIVATE = ROOT / "tmp" / "private_fixtures"
WINDOW, HOP = run_b.GRIDS[0]  # 1.5 s / 0.25 s — the production window grid


def torch_threads(n: int = 4) -> None:
    import torch
    torch.set_num_threads(n)


def load_audio(name: str) -> tuple[np.ndarray, int]:
    """Fixture PCM through the production decoder (16 kHz float32)."""
    fx = score.load_fixture(name)
    return run_b.load_audio(fx["audio_path"])


def gt_single(name: str) -> list[tuple[float, float, str]]:
    """GT intervals with a SINGLE speaker (rubric overlap segments dropped)."""
    return [(s, e, l) for s, e, l in score.load_fixture(name)["gt"] if isinstance(l, str)]


def speaker_pcm(name: str, pcm: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Each GT speaker's concatenated PCM (single-speaker intervals only)."""
    out: dict[str, list[np.ndarray]] = {}
    for s, e, l in gt_single(name):
        out.setdefault(l, []).append(pcm[int(s * sr):int(e * sr)])
    return {l: np.concatenate(v) for l, v in out.items()}


def embed_pooled(pcm: np.ndarray, sr: int) -> np.ndarray:
    """Production-style pooled voiceprint (speaker_id.embed_pcm, capped at
    MAX_POOL_SECONDS like the diarizer's pooling)."""
    cap = int(speaker_id.MAX_POOL_SECONDS * sr)
    return speaker_id.embed_pcm(np.ascontiguousarray(pcm[:cap]), sr)


def speaker_windows(pcm: np.ndarray, sr: int, window: float = WINDOW, hop: float = WINDOW) -> list[np.ndarray]:
    """Speech-gated ``window``-second chunks at ``hop`` inside one speaker's
    pooled PCM (B's noise-floor-relative VAD, >= 30 % speech)."""
    if pcm.size < int(window * sr):
        return []
    mask, _, frame_s = run_b.energy_vad(pcm, sr)
    starts, chunks = run_b._window_slices(pcm, sr, window, hop)
    return [c for s, c in zip(starts, chunks)
            if run_b.speech_frac(mask, frame_s, s, s + window) >= run_b.VAD_MIN_SPEECH_FRAC]


def embed_many(chunks: list[np.ndarray], sr: int, batch: int = 64) -> np.ndarray:
    out: list[np.ndarray] = []
    for i in range(0, len(chunks), batch):
        out.extend(speaker_id.embed_pcm_batch(chunks[i:i + batch], sr))
    return np.stack(out).astype(np.float32) if out else np.zeros((0, 192), np.float32)


# ---------------------------------------------------------------------------
# Approach B (spectral, eigengap, p=0.80) on a window grid, with an optional
# window mask — the exact bake-off pipeline, one global setting.
# ---------------------------------------------------------------------------

def b_windows(tag: str, pcm: np.ndarray, sr: int, use_cache: bool = True) -> dict:
    return run_b.window_embeddings(tag, pcm, sr, WINDOW, HOP, use_cache)


def b_cluster(starts: np.ndarray, embs: np.ndarray, duration: float, keep: np.ndarray | None = None,
              p: float = 0.80, k: int | None = None) -> tuple[list[list], int]:
    """B's headline variant: spectral + eigengap at p=0.80, mode filter, run
    absorption. ``keep`` (bool per window) drops windows from the grid BEFORE
    clustering; the dropped span inherits the nearest kept window's label
    (labels_to_segments' nearest-centre timeline)."""
    if keep is not None:
        starts, embs = starts[keep], embs[keep]
    if len(starts) < 3:
        return [[0.0, duration, 0]], 1
    lab, k_hat, _ = run_b.spectral(embs, p=p, k=k)
    sm = run_b.mode_filter(lab, starts, hop=HOP)
    return run_b.labels_to_segments(sm, starts, WINDOW, duration), k_hat


def score_segments(name: str, segs: list[list]) -> dict:
    fx = score.load_fixture(name)
    sc = run_b.safe_score(name, fx["k_true"], segs)
    return {k: sc.get(k) for k in ("k_pred", "frame_accuracy", "owner_purity", "per_gt_recall")}


# ---------------------------------------------------------------------------
# Window-level ceiling: each speech window -> nearest TRUE voiceprint
# ---------------------------------------------------------------------------

def window_gt_labels(name: str, starts: np.ndarray) -> list[str | None]:
    """GT label under each window CENTRE (None = gap; 'a/b' = overlap)."""
    fx = score.load_fixture(name)
    out = []
    for c in starts + WINDOW / 2:
        g = next((l for s, e, l in fx["gt"] if s <= c < e), None)
        out.append(None if g is None else (g if isinstance(g, str) else "/".join(sorted(g))))
    return out


def window_ceiling(name: str, starts: np.ndarray, embs: np.ndarray, duration: float,
                   pooled_prints: dict[str, np.ndarray] | None = None) -> dict:
    """(a) B's oracle: nearest mean-of-windows GT centroid (separability.py);
    (b) nearest POOLED-audio voiceprint (what production stores). Both as
    window accuracy and as a scored timeline; plus within/cross window cosine."""
    lab = window_gt_labels(name, starts)
    labs = sorted({l for l in lab if l and "/" not in l})
    idx = [i for i, l in enumerate(lab) if l in labs]
    S = embs @ embs.T
    within, cross = {}, {}
    for a in labs:
        ia = [i for i in idx if lab[i] == a]
        if len(ia) > 1:
            sub = S[np.ix_(ia, ia)][np.triu_indices(len(ia), 1)]
            within[a] = round(float(sub.mean()), 3)
        for b in labs:
            if a < b:
                ib = [i for i in idx if lab[i] == b]
                if ia and ib:
                    cross[f"{a}|{b}"] = round(float(S[np.ix_(ia, ib)].mean()), 3)
    out = {"n_windows": int(len(starts)), "n_labelled": len(idx), "within": within, "cross": cross,
           "within_mean": round(float(np.mean(list(within.values()))), 3) if within else None,
           "cross_max": max(cross.values()) if cross else None}
    for kind, prints in (("centroid", None), ("pooled", pooled_prints)):
        if kind == "centroid":
            cents = {a: speaker_id.l2_normalize(embs[[i for i in idx if lab[i] == a]].mean(0)) for a in labs}
        else:
            if not prints:
                continue
            cents = {a: prints[a] for a in labs if a in prints}
        names = list(cents)
        C = np.stack([cents[a] for a in names])
        near = (embs @ C.T).argmax(axis=1)
        ok = sum(1 for i in idx if names[near[i]] == lab[i])
        seg = run_b.labels_to_segments(near, starts, WINDOW, duration)
        segs = [[s, e, names[l]] for s, e, l in seg]
        sc = score_segments(name, segs)
        out[kind] = {"window_acc": round(ok / max(1, len(idx)), 3), "frame_accuracy": sc["frame_accuracy"],
                     "owner_purity": sc["owner_purity"]}
    return out


# ---------------------------------------------------------------------------
# Production path (baseline/run.py, unchanged logic) on arbitrary PCM
# ---------------------------------------------------------------------------

def production_variants(name: str) -> dict[str, list[dict]]:
    fx = score.load_fixture(name)
    v = {"gt_boundaries": [{"speaker": f"U{i}", "text": "…", "start_time": s, "end_time": e}
                           for i, (s, e, _) in enumerate(fx["gt"])]}
    for t in fx.get("transcripts", []):
        v[Path(t).stem] = json.loads(Path(t).read_text())
    return v


def run_production(name: str, pcm: np.ndarray, sr: int, variants: list[str] | None = None) -> dict:
    import diarize_local
    out = {}
    for vname, turns in production_variants(name).items():
        if variants and vname not in variants:
            continue
        t0 = time.time()
        res = diarize_local.diarize_turns(pcm, sr, [dict(t) for t in turns])
        dt = time.time() - t0
        if res is None:
            pred = [(t["start_time"], t["end_time"], "one") for t in turns]
        else:
            pred = [(t["start_time"], t["end_time"], t["speaker"]) for t in res["turns"]]
        r = score.score_fixture(name, pred)
        out[vname] = {k: r[k] for k in ("k_pred", "frame_accuracy", "owner_purity", "per_gt_recall")}
        out[vname]["runtime_s"] = round(dt, 1)
    return out


def wav_write(path: Path, pcm: np.ndarray, sr: int) -> None:
    from scipy.io import wavfile
    wavfile.write(path, sr, np.clip(pcm, -1, 1).astype(np.float32))


def wav_read(path: Path) -> tuple[np.ndarray, int]:
    from scipy.io import wavfile
    sr, x = wavfile.read(path)
    if x.dtype != np.float32:
        x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)


def merge_results(key: str, value: dict) -> None:
    p = HERE / "results.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[key] = value
    p.write_text(json.dumps(d, indent=1, default=str))
