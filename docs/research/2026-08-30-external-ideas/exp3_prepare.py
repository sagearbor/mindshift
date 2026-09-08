"""Experiment 3, step 0 — write the production-decoded 16 kHz PCM of the two
noisy real fixtures to cache/<name>_src.wav so the DeepFilterNet venv
(tmp/venv-dfn, Python 3.11) enhances EXACTLY the samples production sees;
the enhanced files keep this sample count, so transcript timings still
align. Runs under tmp/venv-voice. cache/ is gitignored (maggiano3 is private).
"""
import common as C

for name in ("maggiano3", "poker6"):
    pcm, sr = C.load_audio(name)
    C.wav_write(C.CACHE / f"{name}_src.wav", pcm, sr)
    print(name, pcm.size, sr, f"{pcm.size / sr:.2f}s")
