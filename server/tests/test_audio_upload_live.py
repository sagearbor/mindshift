"""Live-gated integration test for prerecorded transcription + prosody.

SKIPPED unless DEEPGRAM_API_KEY is present (mirrors test_deepgram_live.py). The
fixture is synthesized by scripts/make_test_recording.py, which builds physical
emotional ground truth by modulating each turn's signal (shouting = gain x4 +
faster; cold = quieter + slower). This test then:

  1. Transcribes the ORIGINAL WAV bytes via the real Deepgram pre-recorded API
     and asserts diarization produced >=2 speakers, >=6 turns, non-empty text.
  2. Runs the prosody pipeline on the fixture's KNOWN turn boundaries and
     asserts each modulated turn's labels match the metadata's physically-forced
     expectations — so if prosody calls the shouted turn "quiet", this fails.

Step 1 validates the real Deepgram integration; step 2 closes the loop on the
prosody math against real (modulated) speech. The key-free synthetic-signal unit
tests live in test_prosody.py.
"""

import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

import prosody

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# scripts/ is not on the test pythonpath — add it so the generator is importable.
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

pytestmark = pytest.mark.skipif(
    not os.getenv("DEEPGRAM_API_KEY"),
    reason="DEEPGRAM_API_KEY not set — live prerecorded/prosody test skipped",
)


def _load_or_generate() -> tuple[Path, dict]:
    """Reuse tmp/test_recording.wav + meta if present, else synthesize them."""
    tmp = _REPO_ROOT / "tmp"
    wav_path = tmp / "test_recording.wav"
    meta_path = tmp / "test_recording_meta.json"
    if not (wav_path.exists() and meta_path.exists()):
        import make_test_recording

        wav_path, meta_path = make_test_recording.generate(tmp)
    return wav_path, json.loads(meta_path.read_text())


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, sr


def test_live_prerecorded_transcription_and_prosody():
    from audio_ingest import decode_to_pcm, transcribe_prerecorded

    wav_path, meta = _load_or_generate()
    raw = wav_path.read_bytes()

    # 1) Real Deepgram pre-recorded transcription of the ORIGINAL bytes.
    turns = transcribe_prerecorded(raw, "audio/wav")
    assert len(turns) >= 6, f"expected >=6 utterances, got {len(turns)}"
    assert len({t["speaker"] for t in turns}) >= 2, "expected >=2 diarized speakers"
    assert all(t["text"].strip() for t in turns), "every turn should carry text"

    # 2) Prosody on the fixture's KNOWN turn boundaries (decoupled from
    #    Deepgram's segmentation so the ground-truth assertion is exact).
    pcm, sr = decode_to_pcm(raw, str(wav_path))
    meta_turns = meta["turns"]
    features = [
        prosody.turn_features(pcm, sr, t["start_time"], t["end_time"])
        for t in meta_turns
    ]
    labels = prosody.label_turns(features, meta_turns)

    for turn, label in zip(meta_turns, labels):
        expected = turn["expected"]
        for dim, want in expected.items():
            assert label[dim] == want, (
                f"turn '{turn['scripted_emotion']}' ({turn['text']!r}): "
                f"expected {dim}={want!r}, got {label[dim]!r} "
                f"(rms={label['rms']}, rate={label['speech_rate']})"
            )
