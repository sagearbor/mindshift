"""Step 1: pyannote/speaker-diarization-3.1 end-to-end on every fixture,
three ways (default / oracle k / bounds 2..6), CPU, timed.  Also caches the
pipeline's intermediates (segmentation, binarised segmentation, speaker
count, wespeaker embeddings) and the per-unit audio so that step 2
(ECAPA in the OTHER venv) and step 3 (re-clustering) never re-run the nets.

    tmp/venv-pyannote/bin/python docs/research/2026-08-29-voice-separation/C-pyannote/run_pipeline.py [fixture ...]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from common import CACHE, HERE, Timer, annotation_to_pred, fixture_audio, load_pipeline, score, scored, write_pred

torch.set_num_threads(4)  # ~ a 4 vCPU Cloud Run instance
PIPE = "pyannote/speaker-diarization-3.1"


def cache_intermediates(pipeline, name: str, path: str, gt: list) -> dict:
    """Run segmentation + counting + wespeaker embedding ONCE and cache them
    together with the audio of every (chunk, local-speaker) unit, so ECAPA
    can embed exactly the same units in the other venv."""
    from pyannote.audio.utils.signal import binarize
    file = {"audio": path, "uri": name}
    with Timer() as t_seg:
        segmentations = pipeline.get_segmentations(file)
    if pipeline._segmentation.model.specifications.powerset:
        binarized = segmentations
    else:
        binarized = binarize(segmentations, onset=pipeline.segmentation.threshold, initial_state=False)
    count = pipeline.speaker_count(binarized, pipeline._segmentation.model.receptive_field, warm_up=(0.0, 0.0))
    with Timer() as t_emb:
        emb = pipeline.get_embeddings(file, binarized, exclude_overlap=pipeline.embedding_exclude_overlap)

    sw = segmentations.sliding_window
    num_chunks, num_frames, local = binarized.data.shape
    duration = sw.duration
    wav_sr = pipeline._embedding.sample_rate
    # per-unit audio (same clean-mask rule as pipeline.get_embeddings)
    min_num_samples = pipeline._embedding.min_num_samples
    min_num_frames = int(np.ceil(num_frames * min_num_samples / (duration * wav_sr)))
    clean = binarized.data * (np.sum(binarized.data, axis=2, keepdims=True) < 2)
    units_audio, units_idx, units_meta = {}, [], []
    frame_len = duration / num_frames
    for c, (chunk, masks) in enumerate(binarized):
        waveform, _ = pipeline._audio.crop(file, chunk, duration=duration, mode="pad")
        wav = waveform[0].numpy()
        masks = np.nan_to_num(masks, nan=0.0)
        for s in range(local):
            if masks[:, s].sum() == 0:
                continue
            cm = np.nan_to_num(clean[c, :, s], nan=0.0)
            used = cm if cm.sum() > min_num_frames else masks[:, s]
            # frame -> samples
            sel = np.repeat(used > 0.5, int(round(frame_len * wav_sr)))[: len(wav)]
            if len(sel) < len(wav):
                sel = np.pad(sel, (0, len(wav) - len(sel)))
            pcm = wav[sel]
            key = f"u{c}_{s}"
            units_audio[key] = pcm.astype(np.float32)
            units_idx.append((c, s))
            units_meta.append({"chunk": c, "spk": s, "chunk_start": float(chunk.start),
                               "active_sec": float(used.sum() * frame_len),
                               "clean": bool(cm.sum() > min_num_frames)})
    # GT-interval audio + wespeaker embeddings (for the clustering-only arm)
    from scipy.io import wavfile
    sr, x = wavfile.read(path)
    x = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
    gt_audio, gt_emb = {}, []
    for i, (s, e, lab) in enumerate(gt):
        seg = x[int(s * sr): int(e * sr)]
        gt_audio[f"g{i}"] = seg
        w = torch.from_numpy(seg)[None, None, :]
        gt_emb.append(pipeline._embedding(w)[0])
    np.savez(CACHE / f"{name}_intermediates.npz",
             seg=segmentations.data, seg_start=sw.start, seg_duration=sw.duration, seg_step=sw.step,
             binarized=binarized.data, count=count.data, count_start=count.sliding_window.start,
             count_duration=count.sliding_window.duration, count_step=count.sliding_window.step,
             wespeaker=emb, gt_wespeaker=np.stack(gt_emb))
    np.savez(CACHE / f"{name}_units.npz", **units_audio)
    np.savez(CACHE / f"{name}_gt_units.npz", **gt_audio)
    (CACHE / f"{name}_units.json").write_text(json.dumps(units_meta))
    # human-readable speech-turn segments from segmentation only (per unit)
    return {"t_segmentation": round(t_seg.s, 2), "t_embedding": round(t_emb.s, 2),
            "num_chunks": int(num_chunks), "num_units": len(units_idx),
            "max_count": int(np.nanmax(count.data))}


def main(names):
    pipeline = load_pipeline(PIPE)
    results_path = HERE / "results_pipeline.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name in names:
        fx = score.load_fixture(name)
        path = fixture_audio(name)
        k = fx["k_true"]
        res = results.setdefault(name, {})
        variants = {
            "p31_default": {},
            "p31_oracle_k": {"num_speakers": k},
            "p31_bounds_2_6": {"min_speakers": 2, "max_speakers": 6},
        }
        for vname, kw in variants.items():
            with Timer() as t:
                ann = pipeline(path, **kw)
            pred = annotation_to_pred(ann)
            write_pred(name, vname, pred)
            res[vname] = scored(name, pred, wall_s=round(t.s, 2), device="cpu", kwargs=kw)
            print(f"{name:14s} {vname:16s} acc={res[vname]['frame_accuracy']:.3f} "
                  f"k={res[vname]['k_pred']}/{k} own={res[vname]['owner_purity']} t={t.s:.1f}s", flush=True)
        res["intermediates"] = cache_intermediates(pipeline, name, path, fx["gt"])
        print(f"{name:14s} intermediates {res['intermediates']}", flush=True)
        results_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or score.all_fixtures())
