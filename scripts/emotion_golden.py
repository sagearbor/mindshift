"""Regenerate the golden fixtures for apps/mobile/src/emotion/logMel.ts.

Reference frontend = HF WhisperFeatureExtractor (80 mel, 16 kHz), called the
way eval/codex/model.py prepare_batch does (padding='max_length',
max_length=96000, no gain normalisation). One MESS demo clip (CC BY 4.0, see
ATTRIBUTION.md next to the fixtures) is resampled to 16 kHz, quantised to
int16 (what the phone's mic path hands the app), and the reference log-mel is
computed from exactly that quantised signal.

    python scripts/emotion_golden.py CLIP.wav OUT_DIR [--models DIR]

With --models DIR (a directory holding tiny-finetuned-v1/export/ and
small-finetuned-v1/export/ emotion-int8.onnx; PRIVATE, never committed) the
reference probabilities of both models are written too, so
emotionModelParity.test.ts can compare onnxruntime-node + the TS frontend
against Python when the files are present on the machine.
Needs: numpy scipy soundfile transformers onnxruntime
"""
import argparse
import json
import math
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor

LABELS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]
PAD = 96000


def frontend(x16):
    fe = WhisperFeatureExtractor(feature_size=80, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400)
    feats = fe(
        x16, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=PAD, truncation=False
    )["input_features"].astype("float32")
    vf = min(300, math.ceil(len(x16) / 320))
    return feats[:, :, :600], vf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--models")
    a = ap.parse_args()
    x, sr = sf.read(a.clip, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        g = gcd(16000, sr)
        x = resample_poly(x, 16000 // g, sr // g).astype("float32")
    x = x[:PAD]
    q = np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")
    x16 = q.astype("float32") / 32768.0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    q.tofile(out / "clip04-16k-int16.bin")
    feats, vf = frontend(x16)
    assert feats.shape == (1, 80, 600), feats.shape
    feats[0].astype("<f4").tofile(out / "clip04-logmel.f32")
    meta = {"samples": int(len(q)), "valid_frames": int(vf), "shape": [80, 600], "labels": LABELS}
    if a.models:
        import onnxruntime as ort

        probs = {}
        for name in ("tiny", "small"):
            p = Path(a.models) / f"{name}-finetuned-v1" / "export" / "emotion-int8.onnx"
            s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            t = {i.name: i for i in s.get_inputs()}["valid_frames"].type
            r = s.run(
                None,
                {"log_mel": feats, "valid_frames": np.array([vf], dtype="int32" if "int32" in t else "int64")},
            )[0][0]
            probs[name] = [float(v) for v in r]
        meta["reference_probs"] = probs
    (out / "clip04-meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)


main()
