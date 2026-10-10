"""Recording-replay: the FREE local reference transcript (faster-whisper in
its own venv, run as a subprocess) and how its speakerless words get
speakers (annotation segments, else the server's local diarizer). No model
runs here: the decode is exercised through its cache and a fake runner."""

from __future__ import annotations

import json

import numpy as np
import pytest

from recreplay import stt

DECODE = {
    "engine": "faster-whisper", "model": "small", "language": "en", "duration_s": 6.0,
    "segments": [
        {"start": 0.0, "end": 2.0, "text": "hello there", "words": [
            {"word": "hello", "start": 0.1, "end": 0.6, "probability": 0.9},
            {"word": "there", "start": 0.7, "end": 1.2, "probability": 0.8}]},
        {"start": 3.0, "end": 5.0, "text": "hi back", "words": [
            {"word": "hi", "start": 3.1, "end": 3.4, "probability": 0.95},
            {"word": " ", "start": 3.4, "end": 3.5, "probability": 0.1},
            {"word": "back", "start": 3.6, "end": 4.0, "probability": 0.7}]},
    ],
}


def test_whisper_decode_becomes_deepgram_shaped_raw():
    raw = stt.raw_from_whisper(DECODE, speakers=[0, 0, 1, 1])
    words = stt.words_from_raw(raw)
    assert [w["word"] for w in words] == ["hello", "there", "hi", "back"]
    assert [w["speaker"] for w in words] == [0, 0, 1, 1]
    assert words[2]["start"] == 3.1 and words[2]["confidence"] == 0.95
    assert raw["metadata"]["engine"] == "faster-whisper"


def test_whisper_words_without_speakers_are_one_label():
    words = stt.words_from_raw(stt.raw_from_whisper(DECODE))
    assert {w["speaker"] for w in words} == {0}


def test_speakers_from_segments_by_overlap_then_nearest():
    words = [{"start": 0.1, "end": 0.6}, {"start": 2.2, "end": 2.5}, {"start": 3.1, "end": 3.4},
             {"start": 9.0, "end": 9.3}]
    segs = [{"start": 0.0, "end": 2.4, "speaker": "S2"}, {"start": 2.4, "end": 5.0, "speaker": "S1"}]
    # S2 first seen -> 0; word 2 overlaps S2 more (0.2 vs 0.1); word 4 nearest = S1
    assert stt.speakers_from_segments(words, segs) == [0, 0, 1, 1]


def test_speakers_from_segments_no_segments_is_zero():
    assert stt.speakers_from_segments([{"start": 0, "end": 1}], []) == [0]


def test_transcribe_whisper_offline_needs_cache(tmp_path):
    with pytest.raises(stt.SttUnavailable):
        stt.transcribe_whisper(tmp_path / "a.wav", tmp_path / "whisper.raw.json", offline=True)


def test_transcribe_whisper_uses_cache_without_running(tmp_path):
    cache = tmp_path / "whisper.raw.json"
    cache.write_text(json.dumps(DECODE))

    def boom(*a, **k):
        raise AssertionError("decoder must not run on a cache hit")

    doc, src = stt.transcribe_whisper(tmp_path / "a.wav", cache, runner=boom)
    assert src == "whisper-cache" and doc["segments"][0]["text"] == "hello there"


def test_transcribe_whisper_runs_decoder_and_caches(tmp_path):
    cache = tmp_path / "whisper.raw.json"
    calls = []

    def runner(wav, out, model, language):
        calls.append((wav, model, language))
        out.write_text(json.dumps(DECODE))

    doc, src = stt.transcribe_whisper(tmp_path / "a.wav", cache, runner=runner, model="tiny", language="en")
    assert src == "whisper" and cache.exists() and calls[0][1:] == ("tiny", "en")


def test_diarize_whisper_falls_back_to_one_speaker_when_diarizer_says_nothing():
    pcm = np.zeros(16000 * 6, dtype=np.int16)
    speakers, info = stt.diarize_whisper(pcm, DECODE, diarize_fn=lambda pcm, sr, turns: None)
    assert speakers == [0, 0, 0, 0] and info["source"] == "none"


def test_diarize_whisper_maps_diarizer_turns_back_to_words():
    pcm = np.zeros(16000 * 6, dtype=np.int16)
    seen = {}

    def fake(pcm, sr, turns):
        seen["turns"] = turns
        return {"turns": [{"start_time": 0.0, "end_time": 2.0, "speaker": "Speaker B"},
                          {"start_time": 3.0, "end_time": 5.0, "speaker": "Speaker A"}],
                "num_speakers": 2, "source": "local-ecapa"}

    speakers, info = stt.diarize_whisper(pcm, DECODE, diarize_fn=fake)
    assert speakers == [0, 0, 1, 1] and info["num_speakers"] == 2 and info["source"] == "local-ecapa"
    assert len(seen["turns"]) == 2 and seen["turns"][0]["words"][0]["start_time"] == 0.1


def _inbox(tmp_path, *, annotation: dict | None):
    import wave
    d = tmp_path / "inbox" / "yt_x"
    d.mkdir(parents=True)
    with wave.open(str(d / "yt_x.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(16000 * 6, dtype="<i2").tobytes())
    (d / "yt_x.notes.txt").write_text("who: I'm S1\nsetting: test\nphone: none\n")
    if annotation:
        (d / "yt_x.annotation.json").write_text(json.dumps(annotation))
    return d


def test_inputs_from_inbox_whisper_caches_beside_deepgram(tmp_path):
    from recreplay import pipeline
    inp = pipeline.inputs_from_inbox(_inbox(tmp_path, annotation=None), work_root=tmp_path / "work",
                                     stt_engine="whisper")
    assert inp.deepgram_cache.name == "whisper.json" and inp.stt_engine == "whisper"


def test_build_whisper_reference_speakers_from_annotation(tmp_path, monkeypatch):
    from recreplay import pipeline
    annotation = {"format": "mindshift-annotation/v1", "speakers": [{"id": "S1"}, {"id": "S2"}],
                  "segments": [{"start": 0.0, "end": 2.0, "speaker": "S1", "text": "hello there"},
                               {"start": 3.0, "end": 5.0, "speaker": "S2", "text": "hi back"}]}
    inp = pipeline.inputs_from_inbox(_inbox(tmp_path, annotation=annotation), work_root=tmp_path / "work",
                                     stt_engine="whisper")
    inp.work.mkdir(parents=True)
    (inp.work / "whisper.raw.json").write_text(json.dumps(DECODE))

    def no_diarizer(*a, **k):
        raise AssertionError("annotation segments must be used, not the diarizer")

    monkeypatch.setattr(stt, "diarize_whisper", no_diarizer)
    raw = pipeline.build_whisper_reference(inp, inp.audio, np.zeros(16000 * 6, dtype=np.int16))
    words = stt.words_from_raw(json.loads(inp.deepgram_cache.read_text()))
    assert [w["speaker"] for w in words] == [0, 0, 1, 1]
    assert raw["metadata"]["diarization"]["source"].startswith("annotation")


def test_build_whisper_reference_falls_back_to_local_diarizer(tmp_path, monkeypatch):
    from recreplay import pipeline
    inp = pipeline.inputs_from_inbox(_inbox(tmp_path, annotation=None), work_root=tmp_path / "work",
                                     stt_engine="whisper")
    inp.work.mkdir(parents=True)
    (inp.work / "whisper.raw.json").write_text(json.dumps(DECODE))
    monkeypatch.setattr(stt, "diarize_whisper",
                        lambda pcm, doc: ([1, 1, 0, 0], {"source": "local-ecapa", "num_speakers": 2}))
    raw = pipeline.build_whisper_reference(inp, inp.audio, np.zeros(16000 * 6, dtype=np.int16))
    assert [w["speaker"] for w in stt.words_from_raw(raw)] == [1, 1, 0, 0]
