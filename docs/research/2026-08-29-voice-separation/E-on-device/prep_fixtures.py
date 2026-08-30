"""Step 1 of E: dump the bake-off fixtures in the shape the TS replay needs.

Writes (all under tmp/e-on-device/, gitignored — the private clip and its
rubric-derived timings never land in docs/):

  fixtures.json         {name: {wav, gt: [[s, e, label|[labels]]], k_true, owner}}
  maggiano3.wav         the private .m4a decoded to 16 kHz mono int16
  maggiano3_meta.json   a replay-harness meta (turns[{speaker,start_time,end_time}])
                        so apps/mobile's sceneReplay can drive the real loop over it

Run: tmp/venv-voice/bin/python docs/research/2026-08-29-voice-separation/E-on-device/prep_fixtures.py
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "docs/research/2026-08-29-voice-separation"))

import score  # noqa: E402
from audio_ingest import decode_to_pcm_16k  # noqa: E402

OUT = ROOT / "tmp" / "e-on-device"
OUT.mkdir(parents=True, exist_ok=True)

# score.py fixture name -> replay-harness scene name (test_recording_<scene>.wav)
SCENE = {
    "family_real": "family_real",
    "poker6": "poker6_real",
    "openai": "openai",
    "gptaudio": "gptaudio",
    "scene_couple": "scene_couple_escalation",
    "scene_family3": "scene_family3",
    "scene_meeting4": "scene_meeting4",
}


def write_wav16(path: Path, pcm: np.ndarray) -> None:
    i16 = np.clip(np.round(pcm * 32767), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(i16.tobytes())


out: dict = {}
for name in score.all_fixtures():
    fx = score.load_fixture(name)
    gt = [[float(s), float(e), list(l) if isinstance(l, (tuple, list)) else l] for s, e, l in fx["gt"]]
    if name == "maggiano3":
        data = Path(fx["audio_path"]).read_bytes()
        pcm, sr = decode_to_pcm_16k(data, "audio.m4a")
        assert sr == 16000
        wav = OUT / "maggiano3.wav"
        write_wav16(wav, pcm)
        meta = {
            "_note": "E-on-device replay meta derived from the private rubric (overlap -> first listed speaker)",
            "sample_rate": 16000,
            "turns": [
                {"speaker": (l[0] if isinstance(l, list) else l), "start_time": s, "end_time": e}
                for s, e, l in gt
            ],
        }
        (OUT / "maggiano3_meta.json").write_text(json.dumps(meta, indent=1))
        scene = str(wav)
        seconds = len(pcm) / 16000
        meta_path = str(OUT / "maggiano3_meta.json")
    else:
        meta_path = fx["audio_path"].replace(".wav", "_meta.json")
        with wave.open(fx["audio_path"]) as w:
            native = (w.getframerate(), w.getnchannels())
        if native == (16000, 1):
            scene = SCENE[name]
            with wave.open(fx["audio_path"]) as w:
                seconds = w.getnframes() / 16000
        else:
            # openai/gptaudio are 24 kHz TTS renders: resample through the
            # server's own path (ffmpeg) so the replay sees the 16 kHz PCM the
            # phone would have recorded.
            pcm, sr = decode_to_pcm_16k(Path(fx["audio_path"]).read_bytes(), Path(fx["audio_path"]).name)
            assert sr == 16000
            wav = OUT / f"{name}.wav"
            write_wav16(wav, pcm)
            scene = str(wav)
            seconds = len(pcm) / 16000
            print(f"  {name}: resampled {native[0]} Hz -> 16 kHz at {wav}")
    out[name] = {
        "scene": scene,
        "meta": meta_path,
        "gt": gt,
        "k_true": fx["k_true"],
        "owner": fx["owner_label"],
        "seconds": round(seconds, 3),
    }
    print(f"{name:15s} {seconds:6.1f} s  {len(gt):3d} GT segs  k={fx['k_true']}  owner={fx['owner_label']}")

(OUT / "fixtures.json").write_text(json.dumps(out, indent=1))
print("wrote", OUT / "fixtures.json")
