"""Re-clustering on top of pyannote's cached intermediates (step 3 helpers).
Runs under tmp/venv-pyannote (needs pyannote for reconstruct/to_annotation)."""
from __future__ import annotations

import json

import numpy as np
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.audio.pipelines.clustering import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster, linkage

from common import CACHE, annotation_to_pred, score

PY_THRESHOLD = 0.7045654963945799  # pyannote 3.1's tuned clustering threshold
PY_MIN_CLUSTER_SIZE = 12


class Cached:
    """One fixture's cached pipeline intermediates + both embedding sets."""

    def __init__(self, name: str):
        self.name = name
        z = np.load(CACHE / f"{name}_intermediates.npz")
        sw = SlidingWindow(start=float(z["seg_start"]), duration=float(z["seg_duration"]), step=float(z["seg_step"]))
        self.seg = SlidingWindowFeature(z["seg"], sw)
        self.bin = SlidingWindowFeature(z["binarized"], sw)
        self.count = SlidingWindowFeature(z["count"], SlidingWindow(
            start=float(z["count_start"]), duration=float(z["count_duration"]), step=float(z["count_step"])))
        self.wespeaker = z["wespeaker"]                    # (chunks, local, 256)
        self.gt_wespeaker = z["gt_wespeaker"]              # (n_gt, 256)
        self.units = json.loads((CACHE / f"{name}_units.json").read_text())
        e = np.load(CACHE / f"{name}_ecapa.npz")
        C, L, _ = self.wespeaker.shape
        self.ecapa = np.full((C, L, 192), np.nan, dtype=np.float32)
        for u in self.units:
            self.ecapa[u["chunk"], u["spk"]] = e[f"unit_u{u['chunk']}_{u['spk']}"]
        fx = score.load_fixture(name)
        self.gt = fx["gt"]
        self.k_true = fx["k_true"]
        self.gt_ecapa = np.stack([e[f"gt_g{i}"] for i in range(len(self.gt))])

    def emb(self, which: str) -> np.ndarray:
        return (self.wespeaker if which == "wespeaker" else self.ecapa).copy()

    def gt_emb(self, which: str) -> np.ndarray:
        return (self.gt_wespeaker if which == "wespeaker" else self.gt_ecapa).copy()

    # ---- pyannote's tail: hard clusters -> annotation -> pred ---------------
    def finish(self, pipeline, hard: np.ndarray, max_speakers: int = 20) -> list:
        hard = hard.copy()
        count = SlidingWindowFeature(np.minimum(self.count.data, max_speakers).astype(np.int8), self.count.sliding_window)
        inactive = np.sum(self.bin.data, axis=1) == 0
        hard[inactive] = -2
        disc = pipeline.reconstruct(self.seg, hard, count)
        ann = pipeline.to_annotation(disc, min_duration_on=0.0, min_duration_off=pipeline.segmentation.min_duration_off)
        return annotation_to_pred(ann)

    # ---- unit -> GT majority label (for the segmentation ceiling) ----------
    def unit_gt_labels(self) -> np.ndarray:
        C, F, L = self.bin.data.shape
        frame_len = self.bin.sliding_window.duration / F
        def allowed(l):
            return tuple(l) if isinstance(l, (tuple, list)) else (l,)
        labels = sorted({l for g in self.gt for l in allowed(g[2])})
        out = np.full((C, L), -2, dtype=int)
        for u in self.units:
            c, s = u["chunk"], u["spk"]
            start = u["chunk_start"]
            act = np.nan_to_num(self.bin.data[c, :, s]) > 0.5
            votes = {l: 0.0 for l in labels}
            for f in np.where(act)[0]:
                t = start + (f + 0.5) * frame_len
                for gs, ge, gl in self.gt:
                    if gs <= t < ge:
                        for l in allowed(gl):  # overlap: credit every allowed speaker
                            votes[l] += 1
                        break
            best = max(votes, key=votes.get)
            if votes[best] > 0:
                out[c, s] = labels.index(best)
        return out


def make_clusterer(threshold: float = PY_THRESHOLD, min_cluster_size: int = PY_MIN_CLUSTER_SIZE, method: str = "centroid"):
    c = AgglomerativeClustering(metric="cosine")
    c.instantiate({"method": method, "threshold": threshold, "min_cluster_size": min_cluster_size})
    return c


def pyannote_cluster(cached: Cached, emb: np.ndarray, *, threshold=PY_THRESHOLD, min_cluster_size=PY_MIN_CLUSTER_SIZE,
                     num_clusters=None, min_clusters=None, max_clusters=None) -> np.ndarray:
    """pyannote's own AgglomerativeClustering (filter -> centroid linkage ->
    threshold / min_cluster_size / bounds -> assign every unit to nearest
    centroid), on whichever embeddings we hand it."""
    c = make_clusterer(threshold, min_cluster_size)
    hard, _, _ = c(embeddings=emb, segmentations=cached.bin, num_clusters=num_clusters,
                   min_clusters=min_clusters, max_clusters=max_clusters)
    return hard


def ours_cluster(cached: Cached, emb: np.ndarray, k: int) -> np.ndarray:
    """OUR production recipe (diarize_local): average-linkage on cosine
    distance merged to exactly k, then everything (incl. units too short to
    embed) assigned to the nearest centroid via pyannote's assign step."""
    c = make_clusterer()
    train, ci, si = c.filter_embeddings(emb, segmentations=cached.bin)
    if len(train) <= k:
        clusters = np.arange(len(train))
    else:
        Z = linkage(train, method="average", metric="cosine")
        clusters = fcluster(Z, k, criterion="maxclust") - 1
    _, clusters = np.unique(clusters, return_inverse=True)
    hard, _, _ = c.assign_embeddings(emb, ci, si, clusters, constrained=False)
    return hard


def cluster_gt_segments(cached: Cached, emb: np.ndarray, *, k: int | None = None,
                        threshold: float | None = None, ours: bool = False) -> list:
    """Clustering-only arm: GT intervals are the segments; relabel them."""
    e = emb.copy()
    ok = ~np.isnan(e).any(axis=1)
    idx = np.where(ok)[0]
    if ours:
        Z = linkage(e[idx], method="average", metric="cosine")
        lab = fcluster(Z, k, criterion="maxclust") - 1
    else:
        c = make_clusterer(threshold if threshold is not None else PY_THRESHOLD)
        lab = c.cluster(e[idx], min_clusters=1, max_clusters=len(idx), num_clusters=k)
    labels = np.full(len(e), -1)
    labels[idx] = lab
    # anything un-embeddable -> nearest centroid
    if (labels == -1).any():
        cents = np.stack([np.nanmean(e[labels == j], axis=0) for j in range(lab.max() + 1)])
        for i in np.where(labels == -1)[0]:
            labels[i] = 0
    return [[s, en, f"C{labels[i]}"] for i, (s, en, _) in enumerate(cached.gt)]
