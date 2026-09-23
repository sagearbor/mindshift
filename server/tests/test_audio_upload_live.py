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


# The real-voice control. tmp/ is gitignored, so this may be absent; fetch with
#     python scripts/ami_corpus.py --meetings ES2002a --out tmp/ami
# and copy the Mix-Headset wav here, or point the env var at any real recording
# with two or more human speakers.
_REAL_VOICE_FIXTURE = Path(
    os.getenv("MINDSHIFT_DIARIZATION_CONTROL_WAV")
    or _REPO_ROOT / "tmp" / "testaudio" / "ES2002a.Mix-Headset.wav"
)


def _speaker_count(raw: bytes, model: str) -> int:
    """Diarize `raw` with `model` and count distinct speakers."""
    import httpx

    resp = httpx.post(
        "https://api.deepgram.com/v1/listen",
        params={"model": model, "diarize": "true", "utterances": "true"},
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
        # nova-3 (>=2025-07-31) collapses SYNTHETIC (Aura TTS) voices into one
        # speaker, even a clean unmodulated female+male pair. The question this
        # branch answers is whether that is Deepgram's limitation or OUR bug.
        #
        # The control used to be nova-2 on the same bytes. That stopped working
        # on 2026-09-22: nova-2 now collapses synthetic voices too, so a
        # same-audio/different-model probe can no longer tell the two causes
        # apart, and this test failed looking like a regression it was not.
        #
        # The control is now REAL HUMAN VOICES, which is the actual hypothesis:
        # if Deepgram separates real speakers but not synthetic ones, the
        # limitation is synthetic-voice-specific and nothing here is broken.
        # Measured 2026-09-22 on AMI ES2002a: nova-3 and nova-2 both found 3
        # speakers in real audio, and both found 1 in our TTS fixture.
        if not _REAL_VOICE_FIXTURE.exists():
            pytest.xfail(
                f"synthetic-voice diarization collapsed and the real-voice "
                f"control is absent ({_REAL_VOICE_FIXTURE}), so this run cannot "
                f"tell a Deepgram limitation from a regression. Fetch it with "
                f"`python scripts/ami_corpus.py --meetings ES2002a --out tmp/ami` "
                f"or set MINDSHIFT_DIARIZATION_CONTROL_WAV."
            )
        # A slice is enough to count speakers and keeps the probe quick.
        control = _REAL_VOICE_FIXTURE.read_bytes()[:8_000_000]
        real_speakers = _speaker_count(control, "nova-3")
        if real_speakers >= 2:
            pytest.xfail(
                f"Deepgram no longer diarizes synthetic TTS voices, but "
                f"separates {real_speakers} real speakers in "
                f"{_REAL_VOICE_FIXTURE.name} on the same model — a "
                f"synthetic-voice limitation, not a repo regression. Real-voice "
                f"diarization is what ships, and it works."
            )
        raise AssertionError(
            f"expected >=2 diarized speakers, got {sorted(speakers)} — and the "
            f"REAL-VOICE control ({_REAL_VOICE_FIXTURE.name}) also collapsed to "
            f"{real_speakers}. That is not a synthetic-voice limitation: "
            f"diarization is broken for real audio, which is what ships."
        )
