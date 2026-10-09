"""Recording-replay pipeline: audio normalisation, the Deepgram reference
transcript (cache + word -> turn grouping), and resolving which voice is the
owner from the notes, the annotation and the voiceprint. No network: the
Deepgram call is exercised only through its cache."""

from __future__ import annotations

import json
import shutil
import wave

import numpy as np
import pytest

from recreplay import annotation as ann
from recreplay import audio as audio_mod
from recreplay import identity
from recreplay import notes as notes_mod
from recreplay import stt


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

def test_find_ffmpeg_checks_homebrew_too(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    exe = audio_mod.find_ffmpeg()
    # Either Homebrew's copy is found despite PATH, or None (CI without ffmpeg).
    assert exe is None or exe.endswith("ffmpeg")


@pytest.mark.skipif(audio_mod.find_ffmpeg() is None, reason="ffmpeg not installed")
def test_normalize_any_wav_to_16k_mono(tmp_path):
    src = tmp_path / "in.wav"
    sr = 44100
    t = np.arange(int(sr * 1.5)) / sr
    tone = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    stereo = np.stack([tone, tone], axis=1).reshape(-1)
    with wave.open(str(src), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(stereo.tobytes())
    out = tmp_path / "out.wav"
    info = audio_mod.normalize(src, out)
    assert info["sample_rate"] == 16000 and info["channels"] == 1
    assert info["duration_s"] == pytest.approx(1.5, abs=0.05)
    pcm = audio_mod.read_wav16(out)
    assert pcm.dtype == np.int16 and abs(len(pcm) - 24000) < 400


def test_find_audio_picks_supported_extension(tmp_path):
    (tmp_path / "x.notes.txt").write_text("")
    (tmp_path / "x.m4a").write_bytes(b"\0")
    assert audio_mod.find_audio(tmp_path, "x").name == "x.m4a"
    # a voice-memo app picks its own file name: the folder's ONLY audio file is used
    assert audio_mod.find_audio(tmp_path, "y").name == "x.m4a"
    (tmp_path / "other.mp3").write_bytes(b"\0")
    with pytest.raises(FileNotFoundError):
        audio_mod.find_audio(tmp_path, "y")


# ---------------------------------------------------------------------------
# Deepgram reference transcript
# ---------------------------------------------------------------------------

RAW = {
    "results": {
        "channels": [{"alternatives": [{"words": [
            {"word": "how", "punctuated_word": "How", "start": 1.0, "end": 1.2, "speaker": 0},
            {"word": "was", "punctuated_word": "was", "start": 1.2, "end": 1.4, "speaker": 0},
            {"word": "it", "punctuated_word": "it?", "start": 1.4, "end": 1.6, "speaker": 0},
            {"word": "fine", "punctuated_word": "Fine.", "start": 1.9, "end": 2.3, "speaker": 1},
            {"word": "so", "punctuated_word": "So", "start": 4.0, "end": 4.2, "speaker": 1},
            {"word": "good", "punctuated_word": "good.", "start": 4.2, "end": 4.6, "speaker": 0},
        ]}]}],
    },
}


def test_words_and_turns_from_raw():
    words = stt.words_from_raw(RAW)
    assert len(words) == 6 and words[0]["word"] == "How" and words[3]["speaker"] == 1
    turns = stt.turns_from_words(words, max_gap_s=1.0)
    assert [(t["speaker"], t["text"]) for t in turns] == [
        ("Speaker A", "How was it?"), ("Speaker B", "Fine."), ("Speaker B", "So"), ("Speaker A", "good."),
    ]
    assert turns[0]["start_time"] == 1.0 and turns[0]["end_time"] == 1.6
    assert turns[0]["words"][2]["word"] == "it?"


def test_transcribe_uses_cache_without_network(tmp_path, monkeypatch):
    cache = tmp_path / "deepgram.json"
    cache.write_text(json.dumps(RAW))
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)

    def boom(*a, **k):
        raise AssertionError("network must not be touched on a cache hit")

    monkeypatch.setattr(stt, "_post_deepgram", boom)
    raw, source = stt.transcribe(b"RIFF....", cache)
    assert source == "cache" and raw == RAW


def test_transcribe_without_key_or_cache_is_an_honest_error(tmp_path, monkeypatch):
    monkeypatch.setattr(stt, "deepgram_key", lambda: None)
    with pytest.raises(stt.SttUnavailable):
        stt.transcribe(b"RIFF....", tmp_path / "none.json", offline=False)
    with pytest.raises(stt.SttUnavailable):
        stt.transcribe(b"RIFF....", tmp_path / "none.json", offline=True)


# ---------------------------------------------------------------------------
# Who is the owner?
# ---------------------------------------------------------------------------

ANN = {
    "format": "mindshift-annotation/v1",
    "speakers": [
        {"id": "S1", "voice_description": "boy, high voice", "approx_age": "child"},
        {"id": "S2", "voice_description": "adult male, deep voice", "approx_age": "adult"},
    ],
    "segments": [
        {"start": 1.0, "end": 1.6, "speaker": "S2", "text": "how was it"},
        {"start": 1.9, "end": 2.3, "speaker": "S1", "text": "fine"},
    ],
}


def _aligned():
    a = ann.parse_annotation_text(json.dumps(ANN))
    return a, ann.align(a, stt.words_from_raw(RAW))


def test_descriptor_vote_picks_the_low_voiced_man():
    a, al = _aligned()
    n = notes_mod.parse_notes("who: I'm the man with the low voice; other voice is my son (12)")
    res = identity.resolve(n, a, al, voiceprint_scores=None, dg_speakers=["Speaker A", "Speaker B"])
    assert res.wearer_ann_id == "S2"
    assert res.wearer_label == "Speaker A"        # S2's words are Deepgram speaker 0
    assert res.method == "notes-description"
    assert res.votes["description"]["S2"] > res.votes["description"]["S1"]


def test_explicit_speaker_id_wins():
    a, al = _aligned()
    n = notes_mod.parse_notes("who: I'm S1")
    res = identity.resolve(n, a, al, voiceprint_scores=None, dg_speakers=["Speaker A", "Speaker B"])
    assert res.wearer_ann_id == "S1" and res.wearer_label == "Speaker B" and res.method == "notes-explicit"


def test_voiceprint_vote_and_disagreement_reported():
    a, al = _aligned()
    n = notes_mod.parse_notes("who: I'm the man with the low voice")
    res = identity.resolve(n, a, al, voiceprint_scores={"Speaker A": 0.21, "Speaker B": 0.55},
                           dg_speakers=["Speaker A", "Speaker B"])
    assert res.wearer_label == "Speaker A"      # the owner's own words outrank the print
    assert res.votes["voiceprint_label"] == "Speaker B"
    assert res.agree is False
    assert any("disagree" in w for w in res.warnings)


def test_voiceprint_alone_without_annotation():
    n = notes_mod.parse_notes("who: me")
    res = identity.resolve(n, None, None, voiceprint_scores={"Speaker A": 0.62, "Speaker B": 0.18},
                           dg_speakers=["Speaker A", "Speaker B"])
    assert res.wearer_label == "Speaker A" and res.method == "voiceprint"


def test_unresolved_falls_back_loudly():
    n = notes_mod.parse_notes("")
    res = identity.resolve(n, None, None, voiceprint_scores=None, dg_speakers=["Speaker B", "Speaker A"],
                           talk_seconds={"Speaker A": 3.0, "Speaker B": 9.0})
    assert res.wearer_label == "Speaker B" and res.method == "fallback-most-talk"
    assert res.warnings


def test_label_override():
    n = notes_mod.parse_notes("who: I'm the man with the low voice")
    a, al = _aligned()
    res = identity.resolve(n, a, al, voiceprint_scores=None, dg_speakers=["Speaker A", "Speaker B"],
                           override="Speaker B")
    assert res.wearer_label == "Speaker B" and res.method == "override"

