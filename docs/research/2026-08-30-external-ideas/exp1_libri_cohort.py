"""Experiment 1, optional cohort extension — public background speakers from
LibriSpeech dev-clean (40 read-speech speakers, CC BY 4.0), downloaded to
tmp/external-ideas/libri/ (openslr.org/resources/12/dev-clean.tar.gz).

Per speaker: the first ~30 s of utterances concatenated -> one POOLED
voiceprint (speaker_id.embed_pcm), plus up to WIN_PER_SPK evenly spaced
non-overlapping 1.5 s speech windows. Written to cache/libri_emb.npz in the
same layout as exp1_asnorm.py's fixture embeddings. Runs under tmp/venv-voice.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

import common as C
from common import speaker_id

LIBRI = C.ROOT / "tmp" / "external-ideas" / "libri" / "LibriSpeech" / "dev-clean"
POOL_S, WIN_PER_SPK = 30.0, 6


def main() -> None:
    C.torch_threads()
    speaker_id._load_model()
    pooled, pooled_v, windows, windows_v = [], [], [], []
    for spk_dir in sorted(LIBRI.iterdir(), key=lambda p: int(p.name)):
        parts, total = [], 0
        for f in sorted(spk_dir.rglob("*.flac")):
            x, sr = sf.read(f, dtype="float32")
            assert sr == 16000, sr
            parts.append(x); total += x.size
            if total >= POOL_S * sr:
                break
        pcm = np.concatenate(parts)[: int(POOL_S * 16000)].astype(np.float32)
        voice = f"libri_{spk_dir.name}"
        pooled.append(("libri", spk_dir.name, voice, "pooled")); pooled_v.append(C.embed_pooled(pcm, 16000))
        chunks = C.speaker_windows(pcm, 16000)
        step = max(1, len(chunks) // WIN_PER_SPK)
        chunks = chunks[::step][:WIN_PER_SPK]
        for v in C.embed_many(chunks, 16000):
            windows.append(("libri", spk_dir.name, voice, "window")); windows_v.append(v)
        print(f"  {voice:12s} {pcm.size / 16000:5.1f}s pooled, {len(chunks)} windows", flush=True)
    np.savez(C.CACHE / "libri_emb.npz", pooled_meta=np.array(pooled, dtype=object), pooled=np.stack(pooled_v),
             window_meta=np.array(windows, dtype=object), windows=np.stack(windows_v))
    print(f"{len(pooled)} speakers, {len(windows)} windows")


if __name__ == "__main__":
    main()
