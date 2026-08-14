"""transcribe_prerecorded threads Deepgram's per-word timings into each turn.

Why: Deepgram sometimes welds a speaker handoff into ONE utterance. Fixing that
requires knowing where the words are, so the local diarizer can split a mixed
utterance at a word boundary. The ``words`` key is INTERNAL plumbing: it rides
on the turn dicts, is ignored by the pydantic request/response models, and never
reaches the public /analyze contract.

Deepgram HTTP is fully mocked (no network), mirroring test_transcribe_downmix.
"""

import numpy as np

import audio_ingest


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _transcribe(monkeypatch, payload):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    pcm = (0.2 * np.sin(np.linspace(0, 40, 16000))).astype(np.float32)
    monkeypatch.setattr(
        audio_ingest, "_decode_via_ffmpeg", lambda data, filename="": (pcm, 16000),
    )
    monkeypatch.setattr(
        audio_ingest.httpx, "post",
        lambda url, *, params, headers, content, timeout: _FakeResp(payload),
    )
    return audio_ingest.transcribe_prerecorded(b"bytes", "audio/wav")


def test_words_threaded_through_with_punctuation(monkeypatch):
    payload = {
        "results": {
            "utterances": [
                {
                    "speaker": 0,
                    "transcript": "You wanted the cat.",
                    "start": 0.0, "end": 2.0,
                    "words": [
                        {"word": "you", "punctuated_word": "You",
                         "start": 0.0, "end": 0.4},
                        {"word": "wanted", "punctuated_word": "wanted",
                         "start": 0.4, "end": 1.0},
                        {"word": "the", "punctuated_word": "the",
                         "start": 1.0, "end": 1.3},
                        {"word": "cat", "punctuated_word": "cat.",
                         "start": 1.3, "end": 2.0},
                    ],
                },
            ]
        }
    }
    turns = _transcribe(monkeypatch, payload)
    assert turns[0]["words"] == [
        {"word": "You", "start_time": 0.0, "end_time": 0.4},
        {"word": "wanted", "start_time": 0.4, "end_time": 1.0},
        {"word": "the", "start_time": 1.0, "end_time": 1.3},
        {"word": "cat.", "start_time": 1.3, "end_time": 2.0},
    ]


def test_no_words_in_payload_means_no_words_key(monkeypatch):
    """An utterance without word timings keeps the pre-existing turn shape."""
    payload = {
        "results": {
            "utterances": [
                {"speaker": 0, "transcript": "Hello there.",
                 "start": 0.0, "end": 1.0},
            ]
        }
    }
    turns = _transcribe(monkeypatch, payload)
    assert "words" not in turns[0]


def test_malformed_words_are_dropped_not_fabricated(monkeypatch):
    """Garbage word entries are skipped; only well-timed words survive."""
    payload = {
        "results": {
            "utterances": [
                {
                    "speaker": 0,
                    "transcript": "ok then",
                    "start": 0.0, "end": 1.0,
                    "words": [
                        "not-a-dict",
                        {"word": "", "start": 0.0, "end": 0.2},
                        {"word": "ok", "start": None, "end": 0.4},
                        {"word": "then", "punctuated_word": "then",
                         "start": 0.5, "end": 1.0},
                    ],
                },
            ]
        }
    }
    turns = _transcribe(monkeypatch, payload)
    assert turns[0]["words"] == [
        {"word": "then", "start_time": 0.5, "end_time": 1.0},
    ]
