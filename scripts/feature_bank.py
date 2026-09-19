"""FEATURE BANK — every candidate signal, extracted once per labelled clip, cached to disk.

The bench (feature_bench.py) then compares any combination of groups without
re-touching audio. Groups are independent columns-with-a-prefix so a new
candidate is one function; each group is cached in its own parquet so an
expensive neural group can be filled in for a subset while the cheap ones
cover everything.

Corpora: CREMA-D (91 speakers) and RAVDESS (24) — every clip carries a
speaker id, an emotion, and (where the corpus has it) an intensity. Labels are
NOT features; they travel in the index so the bench can never leak them.

Groups:
  prosody   server/prosody.py's own frame estimator (what the phone ships) —
            f0/energy statistics, plus dB over the speaker's own NEUTRAL median
  egemaps   openSMILE eGeMAPSv02 functionals (88) — the research standard
  tone      WavLM-large arousal / valence / dominance (server/tone_id.py)
  w2v2er    superb/wav2vec2-base-superb-er class logits (small, 95M)
  e2v       emotion2vec+ base logits/embedding via funasr (if installed)

    python scripts/feature_bank.py --group egemaps
    python scripts/feature_bank.py --group tone --max-per-class 400
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "server"))
sys.path.insert(0, str(REPO / "scripts"))
BANK = REPO / "tmp/feature-bank"
SR = 16000

CREMAD = {"ANG": "angry", "NEU": "neutral", "HAP": "happy", "SAD": "sad", "FEA": "fear", "DIS": "disgust"}
RAVDESS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fear", "07": "disgust", "08": "surprised"}


# ------------------------------------------------------------------- index --

def index() -> pd.DataFrame:
    rows = []
    for p in sorted((REPO / "tmp/corpora/cremad/AudioWAV").glob("*.wav")):
        a = p.stem.split("_")
        if len(a) == 4 and a[2] in CREMAD:
            rows.append(dict(path=str(p), corpus="cremad", speaker=f"c{a[0]}", emotion=CREMAD[a[2]],
                             intensity={"LO": 1, "MD": 2, "HI": 3}.get(a[3], 0)))
    for p in sorted((REPO / "tmp/ravdess/audio").rglob("*.wav")):
        a = p.stem.split("-")
        if len(a) == 7 and a[2] in RAVDESS:
            rows.append(dict(path=str(p), corpus="ravdess", speaker=f"r{a[6]}", emotion=RAVDESS[a[2]],
                             intensity={"01": 1, "02": 3}.get(a[3], 0)))
    df = pd.DataFrame(rows).set_index("path")
    df["is_angry"] = df.emotion == "angry"
    return df


def load16k(path: str) -> np.ndarray:
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    x = x.astype(np.float32)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


# ------------------------------------------------------------------ groups --

def g_prosody(paths: list[str], idx: pd.DataFrame) -> pd.DataFrame:
    from heat_rubric import features
    out = {}
    for p in paths:
        f = features(load16k(p), SR)
        if f:
            out[p] = {f"prosody_{k}": v for k, v in f.items() if k != "duration_s"}
    df = pd.DataFrame.from_dict(out, orient="index")
    # dB over the speaker's OWN neutral median — the quantity the ladder uses
    neu = idx[idx.emotion == "neutral"]
    base = {}
    for spk, g in neu.groupby("speaker"):
        v = df.loc[df.index.intersection(g.index), "prosody_rms_dbfs"]
        if len(v):
            base[spk] = float(v.median())
    df["prosody_db_over_own_neutral"] = [df.at[p, "prosody_rms_dbfs"] - base.get(idx.at[p, "speaker"], np.nan) for p in df.index]
    return df


def g_egemaps(paths: list[str], idx: pd.DataFrame) -> pd.DataFrame:
    import opensmile
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    out = {}
    for p in paths:
        d = smile.process_signal(load16k(p), SR)
        out[p] = {f"egemaps_{c}": float(d.iloc[0][c]) for c in d.columns}
    return pd.DataFrame.from_dict(out, orient="index")


def g_tone(paths: list[str], idx: pd.DataFrame) -> pd.DataFrame:
    os.environ.setdefault("MINDSHIFT_TONE_AUDIO", "dark")
    import tone_id
    out = {}
    for p in paths:
        try:
            r = tone_id.classify_pcm(load16k(p), SR)
        except Exception:
            continue
        sc = r.get("scores") or {}
        if "valence" in sc:
            out[p] = {"tone_arousal": float(r["arousal"]), "tone_valence": float(sc["valence"]),
                      "tone_dominance": float(sc["dominance"])}
    return pd.DataFrame.from_dict(out, orient="index")


def g_w2v2er(paths: list[str], idx: pd.DataFrame) -> pd.DataFrame:
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
    mid = "superb/wav2vec2-base-superb-er"
    fe = AutoFeatureExtractor.from_pretrained(mid)
    m = Wav2Vec2ForSequenceClassification.from_pretrained(mid).eval()
    labels = [m.config.id2label[i] for i in range(m.config.num_labels)]
    out = {}
    with torch.no_grad():
        for p in paths:
            x = load16k(p)
            inp = fe(x, sampling_rate=SR, return_tensors="pt")
            logits = m(**inp).logits[0].numpy()
            out[p] = {f"w2v2er_{lab}": float(v) for lab, v in zip(labels, logits)}
    return pd.DataFrame.from_dict(out, orient="index")


def g_e2v(paths: list[str], idx: pd.DataFrame) -> pd.DataFrame:
    from funasr import AutoModel
    m = AutoModel(model="iic/emotion2vec_plus_base", disable_update=True)
    out = {}
    for p in paths:
        try:
            r = m.generate(p, granularity="utterance", extract_embedding=True)[0]
        except Exception:
            continue
        row = {f"e2v_{lab.split('/')[-1]}": float(s) for lab, s in zip(r["labels"], r["scores"])}
        emb = np.asarray(r.get("feats", []), dtype=float).ravel()
        for i, v in enumerate(emb[:64]):        # first 64 dims of the 768-d embedding, enough for a linear probe
            row[f"e2v_emb{i:02d}"] = float(v)
        out[p] = row
    return pd.DataFrame.from_dict(out, orient="index")


def g_sbiemocap(paths: list[str], idx: pd.DataFrame) -> pd.DataFrame:
    """speechbrain/emotion-recognition-wav2vec2-IEMOCAP — Apache-2.0, the
    highest verified accuracy (78.7% IEMOCAP) among cleanly-licensed options.
    Keeps the 4 class probabilities AND the 256-d pooled embedding's first 64
    dims, so the bench can probe the representation, not just the head."""
    import torch
    import torchaudio  # noqa: F401  (speechbrain needs it importable)
    from speechbrain.inference.interfaces import foreign_class
    clf = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                        pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier",
                        savedir=str(REPO / "tmp/models/sb-iemocap"), run_opts={"device": "cpu"})
    labels = clf.hparams.label_encoder.decode_ndim(list(range(4)))
    out = {}
    with torch.no_grad():
        for p in paths:
            wav = torch.tensor(load16k(p)).unsqueeze(0)
            probs = clf.classify_batch(wav)[0][0].exp().numpy() if hasattr(clf.classify_batch(wav)[0], "exp") else clf.classify_batch(wav)[0][0].numpy()
            emb = clf.encode_batch(wav)[0].squeeze().numpy()
            row = {f"sbiemocap_{lab}": float(v) for lab, v in zip(labels, probs)}
            for i, v in enumerate(emb[:64]):
                row[f"sbiemocap_emb{i:02d}"] = float(v)
            out[p] = row
    return pd.DataFrame.from_dict(out, orient="index")


GROUPS = {"prosody": g_prosody, "egemaps": g_egemaps, "tone": g_tone, "w2v2er": g_w2v2er,
          "e2v": g_e2v, "sbiemocap": g_sbiemocap}


# -------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="stratified cap per (corpus, emotion) — for expensive groups")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    BANK.mkdir(parents=True, exist_ok=True)
    idx = index()
    idx.to_parquet(BANK / "index.parquet")
    cache = BANK / f"{args.group}.parquet"
    have = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()
    todo = idx.drop(index=have.index, errors="ignore")
    if args.max_per_class:
        todo = todo.groupby(["corpus", "emotion"], group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.max_per_class), random_state=args.seed))
    paths = list(todo.index)
    print(f"{args.group}: {len(have)} cached, {len(paths)} to extract", flush=True)
    if not paths:
        return 0
    t0 = time.time()
    fn = GROUPS[args.group]
    chunk = 200
    for i in range(0, len(paths), chunk):
        part = fn(paths[i:i + chunk], idx)
        have = pd.concat([have, part]) if len(have) else part
        have.to_parquet(cache)
        done = min(i + chunk, len(paths))
        rate = (time.time() - t0) / done
        print(f"  {done}/{len(paths)}  ({rate*1000:.0f} ms/clip, ~{rate*(len(paths)-done)/60:.0f} min left)", flush=True)
    print(f"{args.group}: {len(have)} rows -> {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
