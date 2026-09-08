"""Approach B — transcript-free sliding-window diarization (2026-08-29 bake-off).

Pipeline (see README.md for results):

  decode -> energy VAD -> uniform windows (1.5 s / 0.25 s hop, also 1.0 / 0.5)
  -> ECAPA window embeddings in ONE batched call (speaker_id.embed_pcm_batch)
  -> cosine affinity -> {agglomerative @ cosine-distance threshold,
                         spectral + eigengap k (Wang et al. 2018 refinement)}
     (+ oracle-k variants as an upper bound)
  -> mode filter over window labels, absorb runs < MIN_RUN_S
  -> (start, end, label) intervals
  -> HYBRID: pool each cluster's PCM, embed the pooled audio (what production
     trusts), re-assign every SEGMENT to its nearest pooled centroid (one pass)
  -> within-cluster coherence = mean pairwise cosine of a cluster's windows.

Window embeddings are cached under ``cache/`` (.npz, regenerable — deleted after the run) so
clustering sweeps are instant; the model is only re-run for the segment /
pooled re-embed step.  All paths are relative to this file.

Usage:  python run_b.py [fixture ...]        (default: every fixture score.py finds)
        python run_b.py --no-cache ...        (force re-embedding)
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
ROOT = RESEARCH.parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(RESEARCH))

import score  # noqa: E402  (shared scorer)
import speaker_id  # noqa: E402
from diarize_sliding_window import _window_slices  # noqa: E402

CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

# ---- knobs ------------------------------------------------------------------
GRIDS = [(1.5, 0.25), (1.0, 0.5)]          # (window_s, hop_s)
VAD_FRAME_MS = 30.0
VAD_MIN_SPEECH_FRAC = 0.3                  # a window must be >= this much speech
VAD_ABS_FLOOR = 0.003
VAD_FLOOR_MULT = 1.5
AGG_THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]  # cosine DISTANCE
AGG_DEFAULT = 0.85                          # the single global pick (see README)
SPEC_PS = [0.95, 0.90, 0.80, 0.70]         # row-wise percentile sweep
POOLED_MERGE_COSINE = 0.45                  # = diarize_local.MAX_POOLED_COSINE
POOLED_MERGE_ROUNDS = 3
SPEC_P = 0.95                               # row-wise percentile threshold
SPEC_MAX_K = 8
SMOOTH_WIN = 5                              # windows, odd (mode filter)
MIN_RUN_S = 0.5                             # absorb shorter runs into neighbours
TORCH_THREADS = 4                           # approximate Cloud Run 4 vCPU
# -----------------------------------------------------------------------------


def _torch_threads():
    try:
        import torch
        torch.set_num_threads(TORCH_THREADS)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Audio + VAD
# ---------------------------------------------------------------------------

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Always through the production decoder: the openai/gptaudio WAVs are
    natively 24 kHz and must be resampled to the ECAPA contract rate."""
    import audio_ingest
    pcm, sr = audio_ingest.decode_to_pcm_16k(Path(path).read_bytes(), "audio" + Path(path).suffix)
    pcm = np.asarray(pcm, dtype=np.float32)
    assert sr == speaker_id.TARGET_SR, sr
    return pcm, sr


def energy_vad(pcm: np.ndarray, sr: int) -> tuple[np.ndarray, float, float]:
    """Per-frame speech mask (30 ms frames). Threshold is NOISE-FLOOR
    relative: max(0.003, 1.5 x the 10th-percentile frame RMS). Measured
    2026-08-29: poker6's quietest player has median RMS 0.0036 against a
    floor of 0.0032, so speaker_id's absolute 0.01 gate (or anything
    peak-relative) silently drops him; the TTS fixtures' gaps are digital
    silence (RMS ~0) and family_real's gaps sit at 0.0035-0.005 vs the
    child's 0.012 median, so 1.5 x floor still separates them."""
    frame = int(sr * VAD_FRAME_MS / 1000)
    n = pcm.size // frame
    rms = np.sqrt(np.mean(pcm[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1))
    thr = max(VAD_ABS_FLOOR, VAD_FLOOR_MULT * float(np.percentile(rms, 10)))
    mask = rms >= thr
    return mask, thr, VAD_FRAME_MS / 1000.0


def speech_frac(mask: np.ndarray, frame_s: float, start: float, end: float) -> float:
    a, b = int(start / frame_s), max(int(start / frame_s) + 1, int(end / frame_s))
    seg = mask[a:b]
    return float(seg.mean()) if seg.size else 0.0


# ---------------------------------------------------------------------------
# Window embeddings (cached)
# ---------------------------------------------------------------------------

def window_embeddings(name: str, pcm: np.ndarray, sr: int, window: float, hop: float,
                      use_cache: bool = True) -> dict:
    tag = f"{name}_w{window}_h{hop}"
    f = CACHE / f"{tag}.npz"
    if use_cache and f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    t0 = time.perf_counter()
    mask, thr, frame_s = energy_vad(pcm, sr)
    starts, chunks = _window_slices(pcm, sr, window, hop)
    keep = [i for i, s in enumerate(starts) if speech_frac(mask, frame_s, s, s + window) >= VAD_MIN_SPEECH_FRAC]
    t_vad = time.perf_counter() - t0
    t0 = time.perf_counter()
    c0 = time.process_time()
    embs = []
    kept_chunks = [chunks[i] for i in keep]
    for i in range(0, len(kept_chunks), 128):
        embs.extend(speaker_id.embed_pcm_batch(kept_chunks[i:i + 128], sr))
    t_emb = time.perf_counter() - t0
    c_emb = time.process_time() - c0
    out = {
        "starts": np.array([starts[i] for i in keep], dtype=np.float32),
        "embs": np.stack(embs).astype(np.float32) if embs else np.zeros((0, 192), np.float32),
        "n_total_windows": np.array(len(starts)),
        "vad_thr": np.array(thr),
        "vad_speech_s": np.array(float(mask.sum()) * frame_s),
        "duration_s": np.array(pcm.size / sr),
        "t_vad": np.array(t_vad), "t_embed": np.array(t_emb), "t_embed_cpu": np.array(c_emb),
        "window": np.array(window), "hop": np.array(hop),
    }
    np.savez(f, **out)
    return out


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def agglomerative(embs: np.ndarray, *, threshold: float | None = None, k: int | None = None) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    if embs.shape[0] < 2:
        return np.zeros(embs.shape[0], dtype=int)
    if k is not None:
        k = min(k, embs.shape[0])
        m = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    else:
        m = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                    metric="cosine", linkage="average")
    return m.fit_predict(embs)


def refine_affinity(embs: np.ndarray, p: float = SPEC_P) -> np.ndarray:
    """Wang et al. 2018 (Google 'LSTM diarization') affinity refinement."""
    a = embs @ embs.T
    np.fill_diagonal(a, 0.0)
    np.fill_diagonal(a, a.max(axis=1))            # diag = row max
    # (Gaussian blur skipped: N is small and windows overlap heavily already)
    thr = np.percentile(a, p * 100, axis=1, keepdims=True)
    a = np.where(a >= thr, a, a * 0.01)           # row-wise thresholding
    a = np.maximum(a, a.T)                        # symmetrize
    a = a @ a.T                                   # diffusion
    a = a / np.maximum(a.max(axis=1, keepdims=True), 1e-9)  # row-max normalise
    return a


def eigengap_k(a: np.ndarray, max_k: int = SPEC_MAX_K) -> tuple[int, list[float]]:
    w = np.linalg.eigvalsh((a + a.T) / 2)[::-1]
    w = np.clip(w, 1e-9, None)
    max_k = min(max_k, len(w) - 1)
    ratios = [float(w[i] / w[i + 1]) for i in range(max_k)]   # ratio for k = i+1
    return int(np.argmax(ratios)) + 1, [round(float(x), 4) for x in w[: max_k + 1]]


def spectral(embs: np.ndarray, *, k: int | None = None, p: float = SPEC_P) -> tuple[np.ndarray, int, list[float]]:
    from sklearn.cluster import KMeans
    n = embs.shape[0]
    if n < 3:
        return np.zeros(n, dtype=int), 1, []
    a = refine_affinity(embs, p)
    k_hat, eig = eigengap_k(a)
    kk = min(k if k is not None else k_hat, n)
    if kk == 1:
        return np.zeros(n, dtype=int), k_hat, eig
    w, v = np.linalg.eigh((a + a.T) / 2)
    feats = v[:, ::-1][:, :kk]
    feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-9)
    lab = KMeans(n_clusters=kk, n_init=10, random_state=0).fit_predict(feats)
    return lab, k_hat, eig


# ---------------------------------------------------------------------------
# Temporal smoothing + intervals
# ---------------------------------------------------------------------------

def mode_filter(labels: np.ndarray, starts: np.ndarray, win: int = SMOOTH_WIN, hop: float = 0.25) -> np.ndarray:
    """Mode over the +/- win//2 TEMPORAL neighbours (only windows within
    win//2 hops in time count — a VAD gap breaks the neighbourhood)."""
    out = labels.copy()
    r = win // 2
    for i in range(len(labels)):
        nb = [labels[j] for j in range(len(labels))
              if abs(starts[j] - starts[i]) <= r * hop + 1e-6]
        vals, cnt = np.unique(nb, return_counts=True)
        best = vals[cnt == cnt.max()]
        out[i] = labels[i] if labels[i] in best else best[0]
    return out


def timeline(labels: np.ndarray, starts: np.ndarray, window: float, duration: float,
             step: float = 0.01) -> np.ndarray:
    """Label every ``step`` frame of the clip by its nearest window CENTRE
    (gaps inherit the nearest window, so nothing is left unlabelled)."""
    n = int(duration / step) + 1
    if len(starts) == 0:
        return np.zeros(n, dtype=int)
    centres = starts + window / 2
    t = np.arange(n) * step
    idx = np.abs(t[:, None] - centres[None, :]).argmin(axis=1)
    return labels[idx]


def runs(tl: np.ndarray, step: float = 0.01) -> list[list]:
    out, s = [], 0
    for i in range(1, len(tl) + 1):
        if i == len(tl) or tl[i] != tl[s]:
            out.append([s * step, i * step, int(tl[s])])
            s = i
    return out


def absorb_short_runs(segs: list[list], min_s: float = MIN_RUN_S) -> list[list]:
    segs = [list(x) for x in segs]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        lens = [e - s for s, e, _ in segs]
        i = int(np.argmin(lens))
        if lens[i] < min_s:
            # give it to the longer neighbour
            cand = [j for j in (i - 1, i + 1) if 0 <= j < len(segs)]
            j = max(cand, key=lambda j: segs[j][1] - segs[j][0])
            segs[i][2] = segs[j][2]
            changed = True
            segs = merge_adjacent(segs)
    return segs


def merge_adjacent(segs: list[list]) -> list[list]:
    out: list[list] = []
    for s, e, l in segs:
        if out and out[-1][2] == l and abs(out[-1][1] - s) < 1e-6:
            out[-1][1] = e
        else:
            out.append([s, e, l])
    return out


def labels_to_segments(labels: np.ndarray, starts: np.ndarray, window: float, duration: float) -> list[list]:
    tl = timeline(labels, starts, window, duration)
    return absorb_short_runs(merge_adjacent(runs(tl)))


# ---------------------------------------------------------------------------
# Hybrid refinement: pooled-centroid re-assignment of SEGMENTS
# ---------------------------------------------------------------------------

def pooled_refine(segs: list[list], pcm: np.ndarray, sr: int, mask: np.ndarray, frame_s: float) -> tuple[list[list], dict]:
    """Embed each segment (raw PCM) and each cluster's POOLED PCM in one batch;
    re-assign every segment to the nearest pooled centroid; report the pooled
    centroid cosine matrix (the production acceptance quantity)."""
    t0 = time.perf_counter()
    labs = sorted({l for _, _, l in segs})
    seg_pcm = [pcm[int(s * sr):int(e * sr)] for s, e, _ in segs]
    pooled = {l: np.concatenate([seg_pcm[i] for i, (_, _, ll) in enumerate(segs) if ll == l]) for l in labs}
    pad = lambda c: c if c.size >= sr // 4 else np.pad(c, (0, sr // 4 - c.size))  # noqa: E731
    # two batches: segments are padded to the longest SEGMENT, pooled chunks
    # to the longest CLUSTER (one batch would pad every segment to ~30 s)
    seg_e = np.stack(speaker_id.embed_pcm_batch([pad(c) for c in seg_pcm], sr))
    cen = np.stack(speaker_id.embed_pcm_batch([pad(pooled[l]) for l in labs], sr))
    sims = seg_e @ cen.T
    new = [[s, e, labs[int(np.argmax(sims[i]))]] for i, (s, e, _) in enumerate(segs)]
    new = merge_adjacent(new)
    cm = (cen @ cen.T)
    off = [round(float(cm[i, j]), 3) for i in range(len(labs)) for j in range(i + 1, len(labs))]
    info = {
        "pooled_cosine_matrix": [[round(float(x), 3) for x in row] for row in cm],
        "pooled_max_offdiag": max(off) if off else None,
        "n_segments_in": len(segs), "n_segments_out": len(new),
        "n_reassigned": int(sum(1 for a, b in zip(segs, [x for x in
                                [[s, e, labs[int(np.argmax(sims[i]))]] for i, (s, e, _) in enumerate(segs)]]) if a[2] != b[2])),
        "t_refine": round(time.perf_counter() - t0, 2),
    }
    return new, info


def pooled_merge(segs: list[list], pcm: np.ndarray, sr: int, thr: float = POOLED_MERGE_COSINE,
                 rounds: int = POOLED_MERGE_ROUNDS) -> tuple[list[list], dict]:
    """PRODUCTION-SHAPED hybrid: start from an OVER-clustered partition, embed
    each cluster's POOLED PCM, average-linkage-merge clusters whose pooled
    centroids are >= ``thr`` cosine (diarize_local's MAX_POOLED_COSINE: same
    voice pooled ~0.73, different ~0.19), re-pool, repeat; then one
    nearest-pooled-centroid re-assignment of every segment."""
    from sklearn.cluster import AgglomerativeClustering
    t0 = time.perf_counter()
    segs = [list(x) for x in segs]
    history = []
    pad = lambda c: c if c.size >= sr // 4 else np.pad(c, (0, sr // 4 - c.size))  # noqa: E731
    for _ in range(rounds):
        labs = sorted({l for _, _, l in segs})
        if len(labs) < 2:
            break
        pooled = [np.concatenate([pcm[int(s * sr):int(e * sr)] for s, e, l in segs if l == lab]) for lab in labs]
        cen = np.stack(speaker_id.embed_pcm_batch([pad(c) for c in pooled], sr))
        cm = cen @ cen.T
        off = cm[np.triu_indices(len(labs), 1)]
        history.append({"k": len(labs), "pooled_max_offdiag": round(float(off.max()), 3)})
        if off.max() < thr:
            break
        m = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - thr, metric="cosine",
                                    linkage="average").fit_predict(cen)
        remap = {lab: int(m[i]) for i, lab in enumerate(labs)}
        segs = merge_adjacent([[s, e, remap[l]] for s, e, l in segs])
    new, info = pooled_refine(segs, pcm, sr, None, None)
    info["merge_history"] = history
    info["t_merge_total"] = round(time.perf_counter() - t0, 2)
    return new, info


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------

def coherence(labels: np.ndarray, embs: np.ndarray) -> dict[int, dict]:
    out = {}
    for l in sorted(set(labels.tolist())):
        idx = np.where(labels == l)[0]
        if len(idx) < 2:
            out[int(l)] = {"n": int(len(idx)), "mean_pairwise_cos": None}
            continue
        sub = embs[idx] @ embs[idx].T
        iu = np.triu_indices(len(idx), 1)
        out[int(l)] = {"n": int(len(idx)), "mean_pairwise_cos": round(float(sub[iu].mean()), 3),
                       "p10_pairwise_cos": round(float(np.percentile(sub[iu], 10)), 3)}
    return out


def cluster_gt_profile(labels: np.ndarray, starts: np.ndarray, window: float, gt: list) -> dict[int, dict]:
    """Which GT speaker(s) each cluster's window CENTRES fall on (phantom = a
    cluster whose majority GT speaker is also another cluster's majority)."""
    centres = starts + window / 2
    prof: dict[int, dict] = {}
    for l in sorted(set(labels.tolist())):
        cnt: dict[str, int] = {}
        for c in centres[labels == l]:
            g = next((lab for s, e, lab in gt if s <= c < e), "unlabelled")
            if not isinstance(g, str):          # rubric overlap segment: (a, b)
                g = "/".join(sorted(g))
            cnt[g] = cnt.get(g, 0) + 1
        tot = sum(cnt.values())
        maj = max(cnt, key=cnt.get)
        prof[int(l)] = {"majority": maj, "purity": round(cnt[maj] / tot, 2), "counts": cnt}
    majs = [p["majority"] for p in prof.values()]
    for p in prof.values():
        p["phantom"] = p["majority"] == "unlabelled" or majs.count(p["majority"]) > 1
    return prof


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

MAX_SCORE_PERMS = 50_000   # score.py's best mapping is permutations(k_pred, k_true) in pure Python


def safe_score(name: str, k_true: int, segs: list[list]) -> dict:
    """The shared scorer, unchanged — but a grossly over-clustered variant
    (e.g. k=26 vs k_true=6 -> 1.7e8 permutations) is reported by k only."""
    import math
    k = len({l for _, _, l in segs})
    a, b = (k, k_true) if k <= k_true else (k_true, k)
    if math.perm(b, a) > MAX_SCORE_PERMS:
        return {"k_pred": k, "frame_accuracy": None, "owner_purity": None, "per_gt_recall": None,
                "note": f"not scored: {math.perm(b, a)} label permutations"}
    return score.score_fixture(name, [tuple(x) for x in segs])

def run_fixture(name: str, use_cache: bool, embed_only: bool = False) -> dict:
    fx = score.load_fixture(name)
    t0 = time.perf_counter()
    pcm, sr = load_audio(fx["audio_path"])
    t_decode = time.perf_counter() - t0
    mask, thr, frame_s = energy_vad(pcm, sr)
    res: dict = {"fixture": name, "k_true": fx["k_true"], "duration_s": round(pcm.size / sr, 2),
                 "vad_speech_s": round(float(mask.sum()) * frame_s, 2), "vad_thr": round(thr, 4),
                 "t_decode": round(t_decode, 2), "variants": {}, "coherence": {}}
    for window, hop in GRIDS:
        we = window_embeddings(name, pcm, sr, window, hop, use_cache)
        starts, embs = we["starts"], we["embs"]
        g = f"w{window}_h{hop}"
        res[f"windows_{g}"] = {"kept": int(len(starts)), "total": int(we["n_total_windows"]),
                               "t_vad": round(float(we["t_vad"]), 2), "t_embed": round(float(we["t_embed"]), 2),
                               "t_embed_cpu": round(float(we["t_embed_cpu"]), 2) if "t_embed_cpu" in we else None}
        if len(starts) < 3 or embed_only:
            continue
        variants: dict[str, np.ndarray] = {}
        extra: dict[str, dict] = {}
        tc = time.perf_counter()
        for T in AGG_THRESHOLDS:
            variants[f"agg_t{T:.2f}"] = agglomerative(embs, threshold=T)
        variants["agg_oracle"] = agglomerative(embs, k=fx["k_true"])
        for pp in SPEC_PS:
            lab, k_hat, eig = spectral(embs, p=pp)
            vn = "spec_eigengap" if pp == SPEC_P else f"spec_eigengap_p{pp:.2f}"
            variants[vn] = lab
            extra[vn] = {"k_eigengap": k_hat, "eigs": eig}
        variants["spec_oracle"] = spectral(embs, k=fx["k_true"])[0]
        t_cluster = time.perf_counter() - tc
        for vname, lab in variants.items():
            ts = time.perf_counter()
            sm = mode_filter(lab, starts, hop=hop)
            segs = labels_to_segments(sm, starts, window, pcm.size / sr)
            t_smooth = time.perf_counter() - ts
            sc = safe_score(name, fx["k_true"], segs)
            entry = {"raw_k": int(len(set(lab.tolist()))),
                     "smoothed": {k: sc[k] for k in ("k_pred", "frame_accuracy", "owner_purity", "per_gt_recall")},
                     "t_cluster_all_variants": round(t_cluster, 2), "t_smooth": round(t_smooth, 2),
                     "segments": segs}
            entry.update(extra.get(vname, {}))
            # hybrid pooled refinement for the headline variants only (model calls)
            if not embed_only and (window, hop) == GRIDS[0] and vname in (
                    "agg_t0.80", "agg_t0.85", "agg_oracle", "spec_eigengap_p0.80"):
                new, info = pooled_refine(segs, pcm, sr, mask, frame_s)
                sc2 = safe_score(name, fx["k_true"], new)
                entry["refined"] = {k: sc2[k] for k in ("k_pred", "frame_accuracy", "owner_purity", "per_gt_recall")}
                entry["refined"].update(info)
                entry["refined_segments"] = new
                if "oracle" not in vname:
                    new2, info2 = pooled_merge(segs, pcm, sr)
                    sc3 = safe_score(name, fx["k_true"], new2)
                    entry["merged"] = {k: sc3[k] for k in ("k_pred", "frame_accuracy", "owner_purity", "per_gt_recall")}
                    entry["merged"].update(info2)
                    entry["merged_segments"] = new2
            res["variants"][f"{g}/{vname}"] = entry
        # coherence on the (smoothed) window labels of the headline variants
        for vname in ("agg_t0.80", "agg_t0.85", "spec_eigengap_p0.80", "agg_oracle"):
            lab = mode_filter(variants[vname], starts, hop=hop)
            coh = coherence(lab, embs)
            prof = cluster_gt_profile(lab, starts, window, fx["gt"])
            res["coherence"][f"{g}/{vname}"] = {str(l): {**coh[l], **prof[l]} for l in coh}
    return res


def main(argv: list[str]) -> None:
    _torch_threads()
    use_cache = "--no-cache" not in argv
    embed_only = "--embed-only" in argv
    names = [a for a in argv if not a.startswith("--")] or score.all_fixtures()
    t0 = time.perf_counter()
    speaker_id._load_model()
    t_model = time.perf_counter() - t0
    out_path = HERE / "results.json"
    all_res = json.loads(out_path.read_text()) if out_path.exists() else {"fixtures": {}}
    all_res.update({"model_load_s": round(t_model, 2), "torch_threads": TORCH_THREADS})
    for n in names:
        t0 = time.perf_counter()
        r = run_fixture(n, use_cache, embed_only)
        r["t_total_wall"] = round(time.perf_counter() - t0, 2)
        all_res["fixtures"][n] = r
        hl = {k: (v["smoothed"]["frame_accuracy"], v["smoothed"]["k_pred"],
                  v.get("merged", {}).get("frame_accuracy"), v.get("merged", {}).get("k_pred"))
              for k, v in r["variants"].items() if "merged" in v or "oracle" in k}
        print(n, "k_true", r["k_true"], "wall", r["t_total_wall"], json.dumps(hl), flush=True)
        if embed_only:
            continue
        # per-fixture file (parallel runs must not clobber one another);
        # make_report.py merges results_*.json into results.json
        (HERE / f"results_{n}.json").write_text(json.dumps(
            {"model_load_s": all_res["model_load_s"], "torch_threads": TORCH_THREADS, "fixture": r}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
