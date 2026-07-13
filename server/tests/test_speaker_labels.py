"""Tests for §1 auto-title + §2 speaker display-label ladder.

Two layers:
  * Pure-function unit tests for the parsers and the precedence ladder
    (_clean_title / _clean_speaker_names / _resolve_speaker_labels) — key-free,
    covering the missing / low-confidence / tie / one-speaker / no-pitch paths.
  * Endpoint tests over /analyze/upload (LLM + Deepgram mocked, stdlib-WAV
    fixture) proving the title and speaker_labels reach the response, that the
    title addendum is gated on the user NOT having named the recording, and that
    a user title is never overridden.
"""

import io
import json
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import main
from main import (
    LABEL_SOURCE_GENERIC,
    LABEL_SOURCE_NAME,
    LABEL_SOURCE_VOICE,
    _clean_speaker_names,
    _clean_title,
    _resolve_speaker_labels,
    app,
    init_db,
)

SR = 16000


# ---------------------------------------------------------------------------
# _clean_title (§1)
# ---------------------------------------------------------------------------

def test_clean_title_keeps_trimmed_string():
    assert _clean_title("  Argument about the cat  ") == "Argument about the cat"


def test_clean_title_rejects_blank_and_non_string():
    assert _clean_title("") is None
    assert _clean_title("   ") is None
    assert _clean_title(None) is None
    assert _clean_title(42) is None


def test_clean_title_truncates_to_cap():
    long = "x" * (main.RECORDING_TITLE_MAX + 50)
    assert len(_clean_title(long)) == main.RECORDING_TITLE_MAX


# ---------------------------------------------------------------------------
# _clean_speaker_names (§2a) — only high-confidence, evidence-based names
# ---------------------------------------------------------------------------

def test_clean_speaker_names_keeps_only_high_confidence():
    out = _clean_speaker_names({
        "Speaker A": {"name": "Joe", "confidence": "high"},
        "Speaker B": {"name": "Mia", "confidence": "medium"},
        "Speaker C": {"name": "Sam", "confidence": "low"},
    })
    assert out == {"Speaker A": "Joe"}


def test_clean_speaker_names_drops_empty_and_malformed():
    out = _clean_speaker_names({
        "Speaker A": {"name": "   ", "confidence": "high"},  # blank name
        "Speaker B": {"name": 5, "confidence": "high"},       # non-string
        "Speaker C": {"confidence": "high"},                  # missing name
        "Speaker D": "nope",                                  # not a dict
        "Speaker E": {"name": "Ada", "confidence": "high"},   # the one keeper
    })
    assert out == {"Speaker E": "Ada"}


def test_clean_speaker_names_non_dict_is_empty():
    assert _clean_speaker_names(None) == {}
    assert _clean_speaker_names([]) == {}


def test_clean_speaker_names_truncates_name():
    long = "N" * (main.SPEAKER_NAME_MAX + 20)
    out = _clean_speaker_names({"A": {"name": long, "confidence": "high"}})
    assert len(out["A"]) == main.SPEAKER_NAME_MAX


# ---------------------------------------------------------------------------
# _resolve_speaker_labels — the precedence ladder (§2a > §2b > §2c)
# ---------------------------------------------------------------------------

def _vl(f0):
    return {"f0_median": f0}


def test_ladder_all_generic_without_names_or_voice():
    labels = _resolve_speaker_labels(["Speaker A", "Speaker B"], {}, None)
    assert labels["Speaker A"].display_label == "Speaker A"
    assert labels["Speaker A"].label_source == LABEL_SOURCE_GENERIC
    assert labels["Speaker B"].label_source == LABEL_SOURCE_GENERIC


def test_ladder_name_overrides_generic():
    labels = _resolve_speaker_labels(
        ["Speaker A", "Speaker B"], {"Speaker A": "Joe"}, None,
    )
    assert labels["Speaker A"].display_label == "Joe"
    assert labels["Speaker A"].label_source == LABEL_SOURCE_NAME
    # Unnamed speaker stays generic (no voice data here).
    assert labels["Speaker B"].label_source == LABEL_SOURCE_GENERIC


def test_ladder_voice_path_when_two_unnamed_speakers():
    speakers = ["Speaker A", "Speaker B", "Speaker A", "Speaker B"]
    voice = [_vl(110.0), _vl(210.0), _vl(115.0), _vl(205.0)]
    labels = _resolve_speaker_labels(speakers, {}, voice)
    assert labels["Speaker A"].display_label == "Deeper voice"
    assert labels["Speaker A"].label_source == LABEL_SOURCE_VOICE
    assert labels["Speaker B"].display_label == "Higher voice"
    assert labels["Speaker B"].label_source == LABEL_SOURCE_VOICE


def test_ladder_name_suppresses_voice_labeling():
    """A relative pair label is meaningless mixed with a real name — if EITHER
    speaker is named, neither gets a voice label."""
    speakers = ["Speaker A", "Speaker B"]
    voice = [_vl(110.0), _vl(210.0)]
    labels = _resolve_speaker_labels(speakers, {"Speaker A": "Joe"}, voice)
    assert labels["Speaker A"].label_source == LABEL_SOURCE_NAME
    assert labels["Speaker B"].label_source == LABEL_SOURCE_GENERIC


def test_ladder_voice_not_applied_for_three_speakers():
    speakers = ["A", "B", "C"]
    voice = [_vl(110.0), _vl(180.0), _vl(250.0)]
    labels = _resolve_speaker_labels(speakers, {}, voice)
    assert all(v.label_source == LABEL_SOURCE_GENERIC for v in labels.values())


def test_ladder_voice_declines_on_near_tie():
    speakers = ["A", "B"]
    voice = [_vl(200.0), _vl(205.0)]  # within 15%
    labels = _resolve_speaker_labels(speakers, {}, voice)
    assert all(v.label_source == LABEL_SOURCE_GENERIC for v in labels.values())


def test_ladder_ignores_name_for_absent_speaker():
    labels = _resolve_speaker_labels(["A", "B"], {"Ghost": "Nobody"}, None)
    assert set(labels) == {"A", "B"}
    assert all(v.label_source == LABEL_SOURCE_GENERIC for v in labels.values())


# ---------------------------------------------------------------------------
# Endpoint layer: title + speaker_labels over /analyze/upload
# ---------------------------------------------------------------------------

@pytest.fixture
async def client():
    await init_db()
    main._rate_limiter.reset()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


def _wav_bytes(pcm: np.ndarray, sr: int = SR) -> bytes:
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _sine(freq: float, seconds: float, amp: float) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# Alice speaks low (120 Hz), Bob speaks high (230 Hz) — a real prosody split so
# the §2b voice ladder has honest medians to work with when no names are given.
_TURN_SPEC = [
    ("Speaker A", 120.0),
    ("Speaker B", 230.0),
    ("Speaker A", 120.0),
    ("Speaker B", 230.0),
    ("Speaker A", 120.0),
    ("Speaker B", 230.0),
]
FIXTURE_PCM = np.concatenate(
    [_sine(f, 1.0, 0.4) for _, f in _TURN_SPEC]
).astype(np.float32)
FIXTURE_WAV = _wav_bytes(FIXTURE_PCM)
MOCK_TURNS = [
    {"speaker": sp, "text": f"Line {i} spoken here.",
     "start_time": float(i), "end_time": float(i + 1)}
    for i, (sp, _f) in enumerate(_TURN_SPEC)
]
_SPEAKERS = ["Speaker A", "Speaker B"]


def _analyze_llm_json(*, names=None, title=None) -> str:
    payload = {
        "per_turn": [
            {"heat": 20, "markers": [], "trigger_phrase": None}
            for _ in MOCK_TURNS
        ],
        "requests": [],
        "narrative": "You both keep showing up and trying to reconnect.",
        "report_cards": {
            sp: {
                "score": 70,
                "headline": f"{sp} stayed engaged",
                "did_well": "Kept trying.",
                "work_on": "Pause before answering.",
            }
            for sp in _SPEAKERS
        },
    }
    if names is not None:
        payload["speaker_names"] = names
    if title is not None:
        payload["title"] = title
    return json.dumps(payload)


def _mock_llm(payload: str) -> MagicMock:
    m = MagicMock()
    m.complete.return_value = payload
    return m


@pytest.mark.anyio
async def test_upload_autotitle_used_when_user_gave_none(client):
    mock = _mock_llm(_analyze_llm_json(title="Argument about the cat"))
    with patch("main.transcribe_prerecorded", return_value=MOCK_TURNS), \
         patch("main.get_llm_client", return_value=mock):
        resp = await client.post(
            "/analyze/upload",
            files={"file": ("clip.wav", FIXTURE_WAV, "audio/wav")},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["title"] == "Argument about the cat"
    # The title addendum WAS added to the system prompt (no user title).
    assert "title" in mock.complete.call_args.kwargs["system"].lower()


@pytest.mark.anyio
async def test_upload_user_title_not_overridden_and_addendum_absent(client):
    mock = _mock_llm(_analyze_llm_json(title="LLM Suggested"))
    with patch("main.transcribe_prerecorded", return_value=MOCK_TURNS), \
         patch("main.get_llm_client", return_value=mock):
        resp = await client.post(
            "/analyze/upload",
            files={"file": ("clip.wav", FIXTURE_WAV, "audio/wav")},
            data={"title": "My Own Title"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["title"] == "My Own Title"
    # No title was requested, so the addendum must be absent from the prompt.
    system = mock.complete.call_args.kwargs["system"]
    assert "3-6 words" not in system


@pytest.mark.anyio
async def test_upload_title_falls_back_when_llm_omits(client):
    """No user title AND the LLM omits one → response title is None (the stored
    recording falls back to its filename, never a fabricated title)."""
    mock = _mock_llm(_analyze_llm_json())  # no title field
    with patch("main.transcribe_prerecorded", return_value=MOCK_TURNS), \
         patch("main.get_llm_client", return_value=mock):
        resp = await client.post(
            "/analyze/upload",
            files={"file": ("clip.wav", FIXTURE_WAV, "audio/wav")},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["title"] is None


@pytest.mark.anyio
async def test_upload_speaker_labels_name_path(client):
    mock = _mock_llm(_analyze_llm_json(names={
        "Speaker A": {"name": "Joe", "confidence": "high"},
        "Speaker B": {"name": "Mia", "confidence": "low"},  # dropped
    }))
    with patch("main.transcribe_prerecorded", return_value=MOCK_TURNS), \
         patch("main.get_llm_client", return_value=mock):
        resp = await client.post(
            "/analyze/upload",
            files={"file": ("clip.wav", FIXTURE_WAV, "audio/wav")},
        )
    assert resp.status_code == 200, resp.text
    labels = resp.json()["speaker_labels"]
    assert labels["Speaker A"] == {"display_label": "Joe", "label_source": "name"}
    # Low-confidence name dropped; Speaker B stays generic (a name on the OTHER
    # speaker suppresses the voice ladder).
    assert labels["Speaker B"]["label_source"] == "generic"


@pytest.mark.anyio
async def test_upload_speaker_labels_voice_path(client):
    """No names + two speakers with a real pitch split → deeper/higher voice."""
    mock = _mock_llm(_analyze_llm_json())
    with patch("main.transcribe_prerecorded", return_value=MOCK_TURNS), \
         patch("main.get_llm_client", return_value=mock):
        resp = await client.post(
            "/analyze/upload",
            files={"file": ("clip.wav", FIXTURE_WAV, "audio/wav")},
        )
    assert resp.status_code == 200, resp.text
    labels = resp.json()["speaker_labels"]
    assert labels["Speaker A"]["display_label"] == "Deeper voice"
    assert labels["Speaker A"]["label_source"] == "voice"
    assert labels["Speaker B"]["display_label"] == "Higher voice"
