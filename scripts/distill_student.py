"""Train, measure, and export the watch-sized "heat" student — step 6 of
docs/decisions/2026-09-20-heat-judge-plan.md.

A Wav2Small-class network (paper arXiv 2408.13920: a tiny conv front-end on
raw 16 kHz waveform, a few depthwise-separable conv blocks, attention
pooling, regression/classification heads) distilled from the two teacher
signals in ``server/tone_id.py`` — ``scripts/distill_teacher_labels.py``'s
per-window parquets (``arousal``, ``valence``, ``dominance``, ``heat``) — on
CPU. Speaker-disjoint by SPEAKER (a whole recording goes to train or val,
never split across the boundary — except CREMA-D, where the split unit is
the ACTOR, since its many "recordings" per actor would otherwise leak the
same voice into both sides), mixup on both inputs and targets, MSE on the
three dimensions + BCE-with-logits on ``heat > 0.5``.

Resumable exactly like the labeller: a checkpoint
(``tmp/distill/checkpoints/student.pt``) is written atomically after every
epoch and reloaded on the next invocation, so ``--max-seconds`` (default
3300 s ≈ 55 min, under the 60-minute-per-invocation budget) can be hit
repeatedly without losing progress.

Stages (``--stages``, comma list, default all — each can be re-run alone
once its inputs exist):

  train    fit the student on tmp/distill/labels/*/*.parquet
  eval     CCC per dim (held-out recordings) + RAVDESS angry-vs-happy AUC
           (true labels, never trained on) + CONFER human-agreement Spearman
           (heat_map.py's within-clip / across-clip protocol), teacher vs
           student
  export   ONNX + int8-quantised ONNX, file sizes, CPU latency/window
  fixture  20-window parity fixture (base64 PCM + expected outputs) for the
           future Kotlin onnxruntime port

Every stage writes into tmp/distill/ (gitignored, MAIN repo) EXCEPT the
final metrics JSON and the parity fixture, which are committed —
``server/tests/fixtures/distill_student_metrics.json`` and
``distill_student_parity.json`` — per the "commit code + a small metrics
JSON, never audio/tmp/" rule.

    python scripts/distill_student.py --max-seconds 3300
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

WORKTREE_REPO = Path(__file__).resolve().parent.parent
MAIN_REPO = Path("/Users/sagearbor/projects/githubs/mindshift")   # see distill_teacher_labels.py

LABELS_DIR = MAIN_REPO / "tmp/distill/labels"
CKPT_DIR = MAIN_REPO / "tmp/distill/checkpoints"
ONNX_DIR = MAIN_REPO / "tmp/distill/onnx"
RAVDESS_DIR = MAIN_REPO / "tmp/ravdess/audio"
CONFER_MANIFEST = MAIN_REPO / "tmp/corpora/confer/heatmap_manifest.json"

FIXTURES_DIR = WORKTREE_REPO / "server/tests/fixtures"
METRICS_JSON = FIXTURES_DIR / "distill_student_metrics.json"
PARITY_JSON = FIXTURES_DIR / "distill_student_parity.json"

SR = 16000
WINDOW_S = 2.0
WINDOW_N = int(WINDOW_S * SR)

# Reference numbers this student is measured against (docs/decisions/2026-09-20-heat-judge-plan.md
# and tmp/feature-bank/bench.json's "loudness-over-own-baseline (shipped)" row) — never re-derived
# here, only compared against.
TEACHER_RAVDESS_AUC = 0.897      # odyssey_dim (WavLM dims) alone, cross-corpus CREMA-D->RAVDESS angry-vs-happy
LOUDNESS_RAVDESS_AUC = 0.733     # shipped dB-over-baseline signal, same protocol
TEACHER_CONFER_WITHIN = 0.51     # brief's stated reference; heat-judge-plan.md measured +0.55 on arousal alone
TEACHER_CONFER_ACROSS = 0.85     # brief's stated reference (not independently reproduced in this repo's docs)


# =============================================================================
# Model — Wav2Small-class student (paper arXiv 2408.13920)
# =============================================================================

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int = 5, stride: int = 2):
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv1d(cin, cin, kernel, stride=stride, padding=pad, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.pw(self.dw(x))))


class Wav2SmallStudent(nn.Module):
    """Raw-waveform conv front-end (downsamples 16 kHz PCM ~32x) -> three
    depthwise-separable blocks (each halves the time axis again) -> attention
    pooling over whatever time length remains -> a shared trunk -> four heads
    (arousal, valence, dominance regression; heat classification logit).
    Global pooling means ANY input length works, which the parity fixture
    below relies on to stay small."""

    def __init__(self, c1: int = 48, c2: int = 72, c3: int = 104, c4: int = 152, hidden: int = 192):
        # (48, 72, 104, 152, 192) measures at 72,077 params — matches the
        # paper's ~72 K target (arXiv 2408.13920) to within 0.1%.
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=80, stride=16, padding=32, bias=False),
            nn.BatchNorm1d(c1), nn.ReLU(inplace=True),
            nn.Conv1d(c1, c1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(c1), nn.ReLU(inplace=True),
        )
        self.block1 = DepthwiseSeparableBlock(c1, c2)
        self.block2 = DepthwiseSeparableBlock(c2, c3)
        self.block3 = DepthwiseSeparableBlock(c3, c4)
        self.attn = nn.Conv1d(c4, 1, kernel_size=1)
        self.trunk = nn.Sequential(nn.Linear(c4, hidden), nn.ReLU(inplace=True))
        self.head_dims = nn.Linear(hidden, 3)     # arousal, valence, dominance
        self.head_heat = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) raw waveform in [-1, 1]. Returns (B, 4): arousal, valence,
        dominance, heat_logit — concatenated (not a tuple) so the ONNX export
        has one output tensor and the parity fixture stays simple."""
        h = x.unsqueeze(1)
        h = self.frontend(h)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)                        # (B, C4, T')
        w = torch.softmax(self.attn(h), dim=-1)    # (B, 1, T')
        pooled = (h * w).sum(dim=-1)                # (B, C4)
        t = self.trunk(pooled)
        dims = self.head_dims(t)
        heat_logit = self.head_heat(t)
        return torch.cat([dims, heat_logit], dim=-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# Data
# =============================================================================

def build_index() -> pd.DataFrame:
    rows = []
    if not LABELS_DIR.exists():
        raise SystemExit(f"no labels at {LABELS_DIR} — run scripts/distill_teacher_labels.py first")
    for corpus_dir in sorted(LABELS_DIR.iterdir()):
        if not corpus_dir.is_dir() or corpus_dir.name.startswith("_"):
            continue
        for pq in sorted(corpus_dir.glob("*.parquet")):
            df = pd.read_parquet(pq)
            if df.empty:
                continue
            df = df.copy()
            df["corpus"] = corpus_dir.name
            df["recording"] = pq.stem
            rows.append(df)
    if not rows:
        raise SystemExit(f"no non-empty label parquets under {LABELS_DIR} — run the labeller first")
    idx = pd.concat(rows, ignore_index=True)
    idx["heat_label"] = (idx["heat"] > 0.5).astype(np.float32)
    # The split unit is the SPEAKER, not the recording filename. For every
    # corpus here except CREMA-D, one recording IS one speaker-set (a
    # meeting/dinner/debate's participants never reappear in another
    # recording), so recording == speaker. CREMA-D is the exception: its 91
    # actors each recorded MANY clips ("recordings" in this pipeline's
    # sense — filename e.g. "1001_DFA_ANG_XX"), so splitting by recording
    # there would let the same actor's voice appear in both train and val —
    # exactly the leak "speaker-disjoint" is supposed to prevent. The actor
    # id is the filename's leading token.
    idx["speaker_key"] = np.where(idx["corpus"] == "cremad",
                                   idx["recording"].str.split("_").str[0],
                                   idx["recording"])
    return idx


def split_recordings(idx: pd.DataFrame, val_frac: float = 0.15, seed: int = 17) -> tuple[set, set]:
    """Speaker-disjoint by construction: a whole SPEAKER (see `speaker_key`
    in build_index — a recording for every corpus except CREMA-D, an actor
    id for CREMA-D) goes to one side, never split across the boundary.
    Stratified per corpus so a small corpus still contributes at least one
    val speaker. Returns (train_keys, val_keys) as {(corpus, speaker_key)}
    sets — callers must key on `speaker_key`, not `recording`."""
    train_ids, val_ids = set(), set()
    recs = idx[["corpus", "speaker_key"]].drop_duplicates()
    rng = np.random.RandomState(seed)
    for corpus, g in recs.groupby("corpus"):
        ids = g["speaker_key"].tolist()
        rng.shuffle(ids)
        n_val = 1 if len(ids) > 1 else 0
        n_val = max(n_val, int(round(len(ids) * val_frac)))
        n_val = min(n_val, len(ids) - 1) if len(ids) > 1 else 0
        val_ids.update((corpus, r) for r in ids[:n_val])
        train_ids.update((corpus, r) for r in ids[n_val:])
    return train_ids, val_ids


def read_window(path: str, t0: float, window_n: int = WINDOW_N, sr: int = SR) -> np.ndarray:
    start = int(round(t0 * sr))
    data, _ = sf.read(path, start=start, frames=window_n, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if len(data) < window_n:
        data = np.pad(data, (0, window_n - len(data)))
    return data.astype(np.float32)


class WindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        x = read_window(r["audio_path"], float(r["window_start"]))
        y = np.array([r["arousal"], r["valence"], r["dominance"], r["heat_label"]], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0 or x.size(0) < 2:
        return x, y
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]


def compute_loss(pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    dim_loss = F.mse_loss(pred[:, :3], y[:, :3])
    heat_loss = F.binary_cross_entropy_with_logits(pred[:, 3], y[:, 3])
    return dim_loss + heat_loss, float(dim_loss.item()), float(heat_loss.item())


# =============================================================================
# Train
# =============================================================================

def train(args) -> tuple[Wav2SmallStudent, pd.DataFrame, pd.DataFrame, dict]:
    idx = build_index()
    train_ids, val_ids = split_recordings(idx, seed=args.seed)
    key = list(zip(idx["corpus"], idx["speaker_key"]))
    train_df = idx[[k in train_ids for k in key]]
    val_df = idx[[k in val_ids for k in key]]
    print(f"windows: {len(idx)} total, {len(train_df)} train / {len(val_df)} val "
          f"({len(train_ids)} train recordings / {len(val_ids)} val recordings)")
    for corpus, g in idx.groupby("corpus"):
        print(f"  {corpus:10} {len(g):6} windows, {g['recording'].nunique()} recordings")

    model = Wav2SmallStudent()
    n_params = count_params(model)
    print(f"student params: {n_params}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(WindowDataset(train_df), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(WindowDataset(val_df), batch_size=args.batch_size, shuffle=False, num_workers=0)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "student.pt"
    start_epoch, history = 0, []
    if ckpt_path.exists() and not args.fresh:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"]
        history = ck.get("history", [])
        print(f"resumed checkpoint at epoch {start_epoch} ({len(history)} epochs of history)")

    t_start = time.time()
    epoch = start_epoch
    stop_reason = None
    while epoch < args.max_epochs:
        if time.time() - t_start > args.max_seconds:
            stop_reason = "time budget (before epoch start)"
            break
        model.train()
        tr_loss = tr_dim = tr_heat = 0.0
        n_batches = 0
        for x, y in train_loader:
            if time.time() - t_start > args.max_seconds:
                stop_reason = "time budget (mid-epoch)"
                break
            x, y = mixup(x, y, args.mixup)
            opt.zero_grad()
            pred = model(x)
            loss, dloss, hloss = compute_loss(pred, y)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())
            tr_dim += dloss
            tr_heat += hloss
            n_batches += 1
        if n_batches == 0:
            break
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                loss, _, _ = compute_loss(model(x), y)
                vl += float(loss.item())
                vn += 1
        val_loss = vl / max(vn, 1)
        train_loss = tr_loss / n_batches
        print(f"epoch {epoch}: train_loss={train_loss:.4f} (dim {tr_dim/n_batches:.4f} "
              f"heat {tr_heat/n_batches:.4f}) val_loss={val_loss:.4f} "
              f"[{time.time()-t_start:.0f}s elapsed]", flush=True)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5)})
        epoch += 1
        tmp = ckpt_path.with_suffix(".tmp.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "history": history, "n_params": n_params}, tmp)
        tmp.replace(ckpt_path)
        if stop_reason:
            break
    if stop_reason is None:
        stop_reason = "max_epochs" if epoch >= args.max_epochs else "loader exhausted"
    elapsed = time.time() - t_start
    print(f"train stopped: {stop_reason}; epoch={epoch}; {elapsed:.0f}s this invocation")
    meta = {"n_params": n_params, "epoch": epoch, "history": history, "stop_reason": stop_reason,
            "train_recordings": len(train_ids), "val_recordings": len(val_ids),
            "train_windows": len(train_df), "val_windows": len(val_df),
            "labelled_windows_per_corpus": {c: int(len(g)) for c, g in idx.groupby("corpus")}}
    return model, train_df, val_df, meta


def load_model_from_checkpoint() -> Wav2SmallStudent:
    ckpt_path = CKPT_DIR / "student.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path} — run the train stage first")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Wav2SmallStudent()
    model.load_state_dict(ck["model"])
    model.eval()
    return model


# =============================================================================
# Eval
# =============================================================================

def ccc(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = float(np.mean((x - mx) * (y - my)))
    denom = vx + vy + (mx - my) ** 2
    return round(2 * cov / denom, 4) if denom > 1e-12 else 0.0


def eval_ccc(model: Wav2SmallStudent, val_df: pd.DataFrame) -> dict:
    loader = DataLoader(WindowDataset(val_df), batch_size=64, shuffle=False, num_workers=0)
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x)[:, :3].numpy())
            trues.append(y[:, :3].numpy())
    p, t = np.concatenate(preds), np.concatenate(trues)
    return {"arousal": ccc(p[:, 0], t[:, 0]), "valence": ccc(p[:, 1], t[:, 1]),
            "dominance": ccc(p[:, 2], t[:, 2]), "n_windows": int(len(p))}


def _resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr, SR)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def eval_ravdess_auc(model: Wav2SmallStudent) -> dict | None:
    """Cross-corpus angry-vs-happy AUC on RAVDESS TRUE labels (never trained
    on — the student only ever saw RAVDESS through teacher-labelled windows
    it wasn't trained on, and this loop trains on none of them). Filename
    convention: 03-01-EMOTION-INTENSITY-STATEMENT-REPETITION-ACTOR.wav,
    emotion 05=angry, 03=happy."""
    if not RAVDESS_DIR.exists():
        return None
    ys, scores = [], []
    model.eval()
    with torch.no_grad():
        for wav in sorted(RAVDESS_DIR.glob("Actor_*/*.wav")):
            parts = wav.stem.split("-")
            if len(parts) != 7:
                continue
            emo = parts[2]
            if emo not in ("05", "03"):
                continue
            x, sr = sf.read(str(wav), dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            x = _resample_to_16k(x, sr)
            if len(x) < WINDOW_N:
                x = np.pad(x, (0, WINDOW_N - len(x)))
            else:
                x = x[:WINDOW_N]
            out = model(torch.from_numpy(x).unsqueeze(0))
            score = float(torch.sigmoid(out[0, 3]))
            scores.append(score)
            ys.append(1 if emo == "05" else 0)
    if len(set(ys)) < 2:
        return {"n": len(ys), "auc": None, "note": "not enough of both classes scored"}
    auc = float(roc_auc_score(ys, scores))
    return {"n": len(ys), "n_angry": int(sum(ys)), "n_happy": int(len(ys) - sum(ys)), "auc": round(auc, 4)}


def eval_confer_agreement(model: Wav2SmallStudent, idx: pd.DataFrame) -> dict | None:
    """heat_map.py::human_conflict_agreement's protocol (Spearman rank
    correlation against CONFER's ten-rater conflict_per_second), split two
    ways: WITHIN-clip (each recording's own time series, Spearman per clip,
    averaged) and ACROSS-clip (every scored window from every recording
    pooled into one correlation — mixes between-recording level differences
    into the signal, which is why it usually reads higher). Computed for both
    the teacher's cached `heat` score and the student's predicted heat
    probability on the SAME windows, so the two are directly comparable."""
    if not CONFER_MANIFEST.exists():
        return None
    manifest = {e["id"]: e for e in json.loads(CONFER_MANIFEST.read_text())}
    sub = idx[idx.corpus == "confer"]
    if sub.empty:
        return None
    within_t, within_s = [], []
    pooled_t, pooled_s, pooled_h = [], [], []
    model.eval()
    for rec, g in sub.groupby("recording"):
        entry = manifest.get(rec)
        if not entry or not entry.get("conflict_per_second"):
            continue
        human = np.asarray(entry["conflict_per_second"], dtype=float)
        g = g.sort_values("window_start")
        starts = g["window_start"].to_numpy()
        bin_idx = np.round(starts).astype(int)
        keep = bin_idx < len(human)
        if keep.sum() < 10:
            continue
        g = g[keep]
        bin_idx = bin_idx[keep]
        # Pool human conflict over the WHOLE 2 s window each score covers
        # (matching heat_map.py's own reference_vs_human_conflict pooling),
        # not just the single second the window starts on.
        human_pooled = np.array([human[i:min(i + int(WINDOW_S), len(human))].mean() for i in bin_idx])
        teacher_score = g["heat"].to_numpy()
        with torch.no_grad():
            xs = np.stack([read_window(p, float(t0)) for p, t0 in zip(g["audio_path"], g["window_start"])])
            out = model(torch.from_numpy(xs))
            student_score = torch.sigmoid(out[:, 3]).numpy()
        if np.std(teacher_score) > 0 and np.std(human_pooled) > 0:
            within_t.append(float(spearmanr(teacher_score, human_pooled).statistic))
        if np.std(student_score) > 0 and np.std(human_pooled) > 0:
            within_s.append(float(spearmanr(student_score, human_pooled).statistic))
        pooled_t.append(teacher_score)
        pooled_s.append(student_score)
        pooled_h.append(human_pooled)
    if not pooled_t:
        return {"n_clips": 0, "note": "no CONFER recordings scored yet (labeller hasn't reached them)"}
    pt, ps, ph = np.concatenate(pooled_t), np.concatenate(pooled_s), np.concatenate(pooled_h)
    return {
        "n_clips": len(pooled_t),
        "teacher_within_clip_mean": round(float(np.mean(within_t)), 3) if within_t else None,
        "teacher_across_clip": round(float(spearmanr(pt, ph).statistic), 3),
        "student_within_clip_mean": round(float(np.mean(within_s)), 3) if within_s else None,
        "student_across_clip": round(float(spearmanr(ps, ph).statistic), 3),
        "reference_teacher_within_clip": TEACHER_CONFER_WITHIN,
        "reference_teacher_across_clip": TEACHER_CONFER_ACROSS,
    }


# =============================================================================
# Export
# =============================================================================

def export_onnx(model: Wav2SmallStudent) -> dict:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, WINDOW_N)
    fp32_path = ONNX_DIR / "student_fp32.onnx"
    # dynamo=False: the legacy TorchScript-tracing exporter. The new dynamo
    # exporter (torch 2.x default) mishandles this model's dynamic_axes (a
    # shape-inference conflict between the batch and channel dims surfaced
    # only downstream, in onnxruntime's quantizer) — measured, not guessed;
    # the legacy path exports and quantizes cleanly on the first try.
    torch.onnx.export(model, dummy, str(fp32_path), input_names=["pcm"], output_names=["dims_heat"],
                       dynamic_axes={"pcm": {0: "batch", 1: "time"}, "dims_heat": {0: "batch"}},
                       opset_version=17, dynamo=False)
    onnx.checker.check_model(str(fp32_path))

    int8_path = ONNX_DIR / "student_int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    fp32_ms = bench_latency(fp32_path)
    int8_ms = bench_latency(int8_path)
    sizes = {
        "fp32_path": str(fp32_path), "int8_path": str(int8_path),
        "fp32_bytes": fp32_path.stat().st_size, "int8_bytes": int8_path.stat().st_size,
        "fp32_latency_ms_per_2s_window": fp32_ms, "int8_latency_ms_per_2s_window": int8_ms,
    }
    print(f"ONNX fp32: {sizes['fp32_bytes']/1024:.1f} KB, {fp32_ms:.2f} ms/window")
    print(f"ONNX int8: {sizes['int8_bytes']/1024:.1f} KB, {int8_ms:.2f} ms/window")
    return sizes


def bench_latency(onnx_path: Path, n: int = 100) -> float:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, WINDOW_N).astype(np.float32)
    for _ in range(5):
        sess.run(None, {"pcm": x})
    t0 = time.time()
    for _ in range(n):
        sess.run(None, {"pcm": x})
    return round((time.time() - t0) / n * 1000, 3)


# =============================================================================
# Parity fixture
# =============================================================================

def make_parity_fixture(val_df: pd.DataFrame, n_fixtures: int = 20, snippet_s: float = 0.3, seed: int = 7) -> dict:
    """20 SHORT (0.3 s, not the production 2 s) real speech snippets, base64
    int16 PCM + the int8 ONNX model's own outputs on them, byte-for-byte
    reproducible — the gate a future Kotlin onnxruntime port checks against.
    Deliberately short to stay under the 300 KB budget: the model pools
    globally over time so any length exercises every op identically; the
    Kotlin port's real 2 s ring buffer is validated in the app, not here."""
    import onnxruntime as ort

    int8_path = ONNX_DIR / "student_int8.onnx"
    if not int8_path.exists():
        raise SystemExit(f"no {int8_path} — run the export stage first")
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    n = int(snippet_s * SR)
    rows = val_df.drop_duplicates(subset=["audio_path", "window_start"]).sample(
        n=min(n_fixtures, len(val_df)), random_state=seed)
    fixtures = []
    for _, r in rows.iterrows():
        x = read_window(r["audio_path"], float(r["window_start"]), window_n=n)
        pcm16 = np.clip(x * 32768.0, -32768, 32767).astype("<i2")
        out = sess.run(None, {"pcm": x.reshape(1, -1).astype(np.float32)})[0].reshape(-1)
        heat_prob = 1.0 / (1.0 + math.exp(-float(out[3])))
        fixtures.append({
            "corpus": r["corpus"], "recording": r["recording"], "window_start": float(r["window_start"]),
            "sample_rate": SR, "n_samples": n,
            "pcm_i16_base64": base64.b64encode(pcm16.tobytes()).decode("ascii"),
            "expected": {"arousal": float(out[0]), "valence": float(out[1]),
                         "dominance": float(out[2]), "heat_prob": heat_prob},
        })
    payload = {
        "note": ("Parity gate for a future Kotlin onnxruntime-android port of student_int8.onnx. "
                 "Windows here are 0.3 s (not the production 2 s) purely to keep this file under "
                 "300 KB; the model's global time-pooling makes any length numerically valid. "
                 "Feed pcm_i16_base64 (little-endian int16) -> float32/32768.0 -> the model and "
                 "expect these four outputs (arousal, valence, dominance, heat_prob = "
                 "sigmoid(heat_logit)) within ~1e-3, the int8 quantisation tolerance."),
        "model": "tmp/distill/onnx/student_int8.onnx (not committed — re-export via "
                 "scripts/distill_student.py --stages export)",
        "snippet_seconds": snippet_s, "fixtures": fixtures,
    }
    return payload


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="train,eval,export,fixture")
    ap.add_argument("--max-seconds", type=float, default=3300.0, help="per-invocation training budget (<=60 min)")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mixup", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    args = ap.parse_args()
    stages = set(args.stages.split(","))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)

    idx = build_index()
    train_ids, val_ids = split_recordings(idx, seed=args.seed)
    key = list(zip(idx["corpus"], idx["speaker_key"]))
    val_df = idx[[k in val_ids for k in key]]

    # Merge onto whatever metrics.json already holds — running a subset of
    # stages (e.g. --stages export,fixture to redo just the ONNX export)
    # must not erase fields an earlier full run already measured.
    metrics = json.loads(METRICS_JSON.read_text()) if METRICS_JSON.exists() else {}
    metrics["generated"] = datetime.now(timezone.utc).isoformat()

    meta = {}
    if "train" in stages:
        model, _, val_df, meta = train(args)
        metrics.update(meta)
    else:
        model = load_model_from_checkpoint()

    if "eval" in stages:
        print("\n-- eval --")
        metrics["ccc"] = eval_ccc(model, val_df)
        print("CCC (held-out recordings):", metrics["ccc"])
        ravdess = eval_ravdess_auc(model)
        metrics["ravdess_auc"] = {
            **(ravdess or {"auc": None, "note": "RAVDESS not found locally"}),
            "reference_teacher_auc": TEACHER_RAVDESS_AUC,
            "reference_loudness_auc": LOUDNESS_RAVDESS_AUC,
        }
        print("RAVDESS angry-vs-happy AUC:", metrics["ravdess_auc"])
        metrics["confer_agreement"] = eval_confer_agreement(model, idx)
        print("CONFER agreement:", metrics["confer_agreement"])

    if "export" in stages:
        print("\n-- export --")
        metrics["onnx"] = export_onnx(model)

    if "fixture" in stages:
        print("\n-- fixture --")
        payload = make_parity_fixture(val_df)
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        PARITY_JSON.write_text(json.dumps(payload, indent=1))
        size_kb = PARITY_JSON.stat().st_size / 1024
        print(f"parity fixture: {PARITY_JSON} ({size_kb:.1f} KB, {len(payload['fixtures'])} windows)")
        metrics["parity_fixture"] = {"path": str(PARITY_JSON.relative_to(WORKTREE_REPO)),
                                      "n_windows": len(payload["fixtures"]), "size_kb": round(size_kb, 1)}

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=1))
    print(f"\nmetrics: {METRICS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
