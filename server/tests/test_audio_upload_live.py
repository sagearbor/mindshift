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


def _nova2_speaker_count(raw: bytes) -> int:
    """Control probe: diarize the same bytes with nova-2 (see caller)."""
    import httpx

    resp = httpx.post(
        "https://api.deepgram.com/v1/listen",
        params={"model": "nova-2", "diarize": "true", "utterances": "true"},
        headers={
            "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY', '').strip()}",
            "Content-Type": "audio/wav",
        },
        content=raw,
        timeout=120,
    )
    resp.raise_for_status()
    utts = resp.json().get("results", {}).get("utterances", [])
    return len({u.get("speaker") for u in utts})


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

    # 3) Diarization — LAST, because the xfail below must not mask the
    #    transcription/prosody coverage above.
    speakers = {t["speaker"] for t in turns}
    if len(speakers) < 2:
        # nova-3 model 2025-07-31 started collapsing SYNTHETIC (Aura TTS)
        # voices into one speaker — even a clean, unmodulated female+male pair.
        # Disambiguate a Deepgram-side synthetic-voice limitation from a
        # regression in OUR audio/params: nova-2 on the SAME bytes is the
        # control. If nova-2 also hears one speaker, our fixture/params broke.
        if _nova2_speaker_count(raw) >= 2:
            pytest.xfail(
                "nova-3 (>=2025-07-31) no longer diarizes synthetic TTS "
                "voices (nova-2 control separates the same bytes) — "
                "Deepgram-side limitation, not a repo regression. Real-voice "
                "diarization should be spot-checked separately."
            )
        raise AssertionError(
            f"expected >=2 diarized speakers, got {sorted(speakers)} — and the "
            "nova-2 control ALSO collapsed them, so our audio/params are the "
            "likely culprit"
        )
