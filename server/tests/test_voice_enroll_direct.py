"""POST /voice/enroll-direct — guided voice enrollment from an uploaded clip.

The guided "Train my voice" flow records a few prompted phrases in-app and
uploads ONE short wav that is (by client promise) only the enrolling user's
voice. No diarization, no stored recording: decode → measure actual speech →
embed the whole clip → append a v2 sample with note "guided enrollment".

Torch-free like the rest of the voice suite: ``speaker_id.embed_pcm`` and
``speaker_id.is_available`` are monkeypatched with deterministic doubles; the
wav fixtures are built with the stdlib so decode runs the REAL 16 kHz stdlib
path (no ffmpeg needed).

Coverage: the pure speech measure (speech_seconds); happy path incl. stored
provenance (recording_id null, note "guided enrollment") and blending with the
This-is-me path; honest failures (silence/short → 422, undecodable → 422,
deps absent → 503, storage disabled → 503, oversized → 413, unauthenticated →
401); uid scoping.
"""

import io
import wave

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import main
import speaker_id
from auth import get_current_uid
from main import app, init_db

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def _fresh_rate_limit():
    """Isolate the in-process rate limiter per test (same pattern as the other
    rate-limited endpoint suites)."""
    main._rate_limiter.reset()
    yield
    main._rate_limiter.reset()


# ---------------------------------------------------------------------------
# Fixture audio — stdlib wav builder (16-bit PCM mono)
# ---------------------------------------------------------------------------

def _wav_bytes(
    seconds: float, *, sr: int = 16000, amplitude: float = 0.3, freq: float = 220.0,
) -> bytes:
    """A mono 16-bit wav of a sine tone (or silence when amplitude=0)."""
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    pcm = (amplitude * np.sin(2 * np.pi * freq * t) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# speech_seconds — the pure energy-based speech measure (no torch)
# ---------------------------------------------------------------------------

def test_speech_seconds_counts_a_tone_and_ignores_silence():
    sr = 16000
    t = np.arange(sr * 4, dtype=np.float32) / sr
    tone = 0.3 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    silence = np.zeros(sr * 4, dtype=np.float32)
    assert speaker_id.speech_seconds(tone, sr) == pytest.approx(4.0, abs=0.1)
    assert speaker_id.speech_seconds(silence, sr) == 0.0


def test_speech_seconds_counts_only_the_loud_span():
    # 2s speech-level signal inside 6s of silence → ~2s of speech, not 6.
    sr = 16000
    pcm = np.zeros(sr * 6, dtype=np.float32)
    t = np.arange(sr * 2, dtype=np.float32) / sr
    pcm[sr * 2 : sr * 4] = 0.3 * np.sin(2 * np.pi * 220.0 * t)
    assert speaker_id.speech_seconds(pcm, sr) == pytest.approx(2.0, abs=0.1)


def test_speech_seconds_empty_and_bad_sr_are_zero():
    assert speaker_id.speech_seconds(np.zeros(0, dtype=np.float32), 16000) == 0.0
    assert speaker_id.speech_seconds(np.zeros(100, dtype=np.float32), 0) == 0.0


# ---------------------------------------------------------------------------
# Router — in-memory fake store, mocked embedder
# ---------------------------------------------------------------------------

class FakeVoiceStore:
    """Only what the direct-enroll path touches: the per-uid voiceprint doc."""

    def __init__(self):
        self._voiceprints: dict[str, dict] = {}

    async def read_voiceprint(self, uid):
        return self._voiceprints.get(uid)

    async def write_voiceprint(self, uid, profile):
        self._voiceprints[uid] = profile

    async def delete_voiceprint(self, uid):
        return self._voiceprints.pop(uid, None) is not None


@pytest.fixture
async def client():
    await init_db()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def voice_store():
    fake = FakeVoiceStore()
    app.state.recordings_store = fake
    yield fake
    del app.state.recordings_store


@pytest.fixture
def embed_ready(monkeypatch):
    """Voice deps 'installed' + a deterministic embedder (records its input)."""
    calls: list[tuple[int, int]] = []  # (n_samples, sr) per embed call

    def _embed(pcm, sr=16000):
        calls.append((int(np.asarray(pcm).size), sr))
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    monkeypatch.setattr(speaker_id, "embed_pcm", _embed)
    return calls


def _upload(wav: bytes, name: str = "guided-enrollment.wav"):
    return {"file": (name, wav, "audio/wav")}


async def test_enroll_direct_happy_path_stores_guided_sample(
    client, voice_store, embed_ready,
):
    res = await client.post(
        "/voice/enroll-direct",
        files=_upload(_wav_bytes(5.0)),
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["enrolled"] is True
    assert body["enroll_count"] == 1
    assert "not your audio" in body["stored"]
    # The whole clip was embedded at 16 kHz (no diarization pooling).
    assert embed_ready == [(16000 * 5, 16000)]

    # Provenance: a v2 sample with NO source recording and the guided note.
    prof = await client.get("/voice/profile", headers={"X-Test-Uid": "u1"})
    assert prof.status_code == 200
    p = prof.json()
    assert p["enrolled"] is True
    assert p["enroll_count"] == 1
    (sample,) = p["samples"]
    assert sample["recording_id"] is None
    assert sample["speaker"] is None
    assert sample["note"] == "guided enrollment"
    assert sample["at"]  # timestamped

    # The stored doc never leaks through the API but must hold the embedding.
    stored = voice_store._voiceprints["u1"]
    assert stored["samples"][0]["note"] == "guided enrollment"
    assert stored["samples"][0]["embedding"] == [1.0, 0.0, 0.0]


async def test_enroll_direct_counts_add_up_with_repeat_enrollment(
    client, voice_store, embed_ready,
):
    for expected in (1, 2):
        res = await client.post(
            "/voice/enroll-direct",
            files=_upload(_wav_bytes(5.0)),
            headers={"X-Test-Uid": "u1"},
        )
        assert res.status_code == 200
        assert res.json()["enroll_count"] == expected
    # Two independent samples, both guided.
    stored = voice_store._voiceprints["u1"]
    assert [s["note"] for s in stored["samples"]] == ["guided enrollment"] * 2


async def test_enroll_direct_long_clip_is_capped_before_embedding(
    client, voice_store, embed_ready,
):
    # A 70s clip embeds at most MAX_POOL_SECONDS (60s) of audio — the embed
    # call stays bounded exactly like the pooled This-is-me path.
    res = await client.post(
        "/voice/enroll-direct",
        files=_upload(_wav_bytes(70.0)),
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 200
    assert embed_ready == [(int(16000 * speaker_id.MAX_POOL_SECONDS), 16000)]


async def test_enroll_direct_silence_422_not_enough_speech(
    client, voice_store, embed_ready,
):
    # 10 seconds LONG but silent: duration is not speech — honest 422.
    res = await client.post(
        "/voice/enroll-direct",
        files=_upload(_wav_bytes(10.0, amplitude=0.0)),
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 422
    assert "not enough speech" in res.json()["detail"]
    assert embed_ready == []  # never embedded, never stored
    assert "u1" not in voice_store._voiceprints


async def test_enroll_direct_too_short_422(client, voice_store, embed_ready):
    res = await client.post(
        "/voice/enroll-direct",
        files=_upload(_wav_bytes(1.0)),
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 422
    assert "not enough speech" in res.json()["detail"]


async def test_enroll_direct_undecodable_422(client, voice_store, embed_ready):
    res = await client.post(
        "/voice/enroll-direct",
        files={"file": ("clip.wav", b"RIFFgarbageWAVEgarbage", "audio/wav")},
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 422


async def test_enroll_direct_deps_absent_503(client, voice_store, monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: False)
    res = await client.post(
        "/voice/enroll-direct",
        files=_upload(_wav_bytes(5.0)),
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 503
    assert "not available" in res.json()["detail"]


async def test_enroll_direct_storage_disabled_503(client, monkeypatch):
    # No app.state.recordings_store → nowhere to keep a voiceprint.
    monkeypatch.setattr(speaker_id, "is_available", lambda: True)
    res = await client.post(
        "/voice/enroll-direct",
        files=_upload(_wav_bytes(5.0)),
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 503


async def test_enroll_direct_oversized_413(client, voice_store, embed_ready):
    from routers import voice as voice_router

    big = b"\x00" * (voice_router.MAX_DIRECT_ENROLL_BYTES + 1)
    res = await client.post(
        "/voice/enroll-direct",
        files={"file": ("big.wav", big, "audio/wav")},
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 413
    assert embed_ready == []


async def test_enroll_direct_requires_auth_401(
    client, voice_store, embed_ready, monkeypatch,
):
    # Drop the test-uid override → the real bearer-token dependency runs.
    monkeypatch.delitem(app.dependency_overrides, get_current_uid)
    res = await client.post(
        "/voice/enroll-direct", files=_upload(_wav_bytes(5.0)),
    )
    assert res.status_code == 401


async def test_enroll_direct_is_uid_scoped(client, voice_store, embed_ready):
    res = await client.post(
        "/voice/enroll-direct",
        files=_upload(_wav_bytes(5.0)),
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 200
    # Another user sees no enrollment.
    prof = await client.get("/voice/profile", headers={"X-Test-Uid": "u2"})
    assert prof.json()["enrolled"] is False


async def test_enroll_direct_is_rate_limited_429(
    client, voice_store, embed_ready, monkeypatch,
):
    monkeypatch.setattr(main._rate_limiter, "limit", 2)
    main._rate_limiter.reset()
    wav = _wav_bytes(5.0)
    for _ in range(2):
        res = await client.post(
            "/voice/enroll-direct",
            files=_upload(wav),
            headers={"X-Test-Uid": "u1"},
        )
        assert res.status_code == 200
    res = await client.post(
        "/voice/enroll-direct",
        files=_upload(wav),
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 429


async def test_guided_and_this_is_me_samples_blend(
    client, voice_store, embed_ready, monkeypatch,
):
    """Both enrollment paths append samples to the SAME v2 profile."""
    import routers.voice as voice_router

    # First: a guided enrollment.
    res = await client.post(
        "/voice/enroll-direct",
        files=_upload(_wav_bytes(5.0)),
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 200

    # Then: a classic This-is-me enrollment on a stored recording.
    turns = [
        {"speaker": "Speaker A", "text": "hi", "start_time": 0.0, "end_time": 4.0},
    ]
    rid = "0f0e0d0c-0b0a-4a4b-8c8d-0e0f10111213"
    recs = {("u1", rid): {"turns": turns, "audio": b"AUDIO"}}

    async def get_recording(uid, recording_id):
        r = recs.get((uid, recording_id))
        return None if r is None else {"id": recording_id, "turns": r["turns"]}

    async def get_audio_bytes(uid, recording_id):
        r = recs.get((uid, recording_id))
        return None if r is None else r["audio"]

    voice_store.get_recording = get_recording
    voice_store.get_audio_bytes = get_audio_bytes
    monkeypatch.setattr(
        voice_router, "decode_to_pcm",
        lambda data, name: (np.zeros(16000 * 5, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(
        speaker_id, "embed_speaker",
        lambda *a, **k: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    res = await client.post(
        "/voice/enroll",
        json={"recording_id": rid, "speaker": "Speaker A"},
        headers={"X-Test-Uid": "u1"},
    )
    assert res.status_code == 200, res.text
    assert res.json()["enroll_count"] == 2

    prof = await client.get("/voice/profile", headers={"X-Test-Uid": "u1"})
    notes = [s.get("note") for s in prof.json()["samples"]]
    assert notes == ["guided enrollment", None]
