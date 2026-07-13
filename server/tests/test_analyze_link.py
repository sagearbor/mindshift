"""Tests for POST /analyze/link and the link_fetch download helper.

link_fetch is exercised directly with an injected DNS resolver and an httpx
MockTransport (no real network, no real DNS) so the SSRF guard, size cap, HTML
rejection, and Drive URL rewrite are tested against the real logic. The endpoint
itself is tested by patching main.link_fetch.fetch_link (the fetch is covered by
the unit tests) with the analysis LLM/Deepgram mocked, so the wiring — pipeline
run, source provenance, error-status propagation — is verified end to end.
"""

import io
import json
import wave
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import audio_ingest
import link_fetch
import main
from main import app, init_db

SR = 16000


# ---------------------------------------------------------------------------
# Fixtures (mirror test_analyze_upload)
# ---------------------------------------------------------------------------

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


_AMPS = [0.1, 0.2, 0.5, 0.15, 0.3, 0.08]
FIXTURE_WAV = _wav_bytes(
    np.concatenate([_sine(180.0, 1.0, a) for a in _AMPS]).astype(np.float32)
)

MOCK_TURNS = [
    {"speaker": "Speaker A", "text": "Hey, can we talk about the schedule?",
     "start_time": 0.0, "end_time": 1.0},
    {"speaker": "Speaker B", "text": "Sure, what about it.",
     "start_time": 1.0, "end_time": 2.0},
    {"speaker": "Speaker A", "text": "You never stick to what we agree.",
     "start_time": 2.0, "end_time": 3.0},
    {"speaker": "Speaker B", "text": "That is not fair and you know it.",
     "start_time": 3.0, "end_time": 4.0},
    {"speaker": "Speaker A", "text": "Okay. I hear you. Let me try again.",
     "start_time": 4.0, "end_time": 5.0},
    {"speaker": "Speaker B", "text": "Thanks. I appreciate that.",
     "start_time": 5.0, "end_time": 6.0},
]
_SPEAKERS = ["Speaker A", "Speaker B"]
FAKE_AUDIO_M4A = b"FAKE-M4A-AUDIO-DERIVATIVE-" * 20


def _analyze_llm_json(n_turns: int) -> str:
    return json.dumps({
        "per_turn": [
            {"heat": 20 + i * 3, "markers": [], "trigger_phrase": None}
            for i in range(n_turns)
        ],
        "requests": [],
        "narrative": "You both keep showing up and trying to reconnect.",
        "report_cards": {
            sp: {"score": 70, "headline": f"{sp} stayed engaged",
                 "did_well": "Kept trying to reconnect.",
                 "work_on": "Pause before answering criticism."}
            for sp in _SPEAKERS
        },
    })


def _mock_llm(payload: str) -> MagicMock:
    m = MagicMock()
    m.complete.return_value = payload
    return m


@pytest.fixture
async def client():
    await init_db()
    main._rate_limiter.reset()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ===========================================================================
# link_fetch unit tests — real logic, injected resolver + MockTransport
# ===========================================================================

def _public_resolver(host):
    return ["93.184.216.34"]  # example.com, a public address


def test_rewrite_drive_file_share_url():
    out = link_fetch.rewrite_url(
        "https://drive.google.com/file/d/ABC123xyz/view?usp=sharing"
    )
    assert out == "https://drive.google.com/uc?export=download&id=ABC123xyz"


def test_rewrite_drive_open_id_url():
    out = link_fetch.rewrite_url(
        "https://drive.google.com/open?id=ZZZ999"
    )
    assert out == "https://drive.google.com/uc?export=download&id=ZZZ999"


def test_rewrite_leaves_non_drive_url_untouched():
    url = "https://example.com/path/clip.mp4"
    assert link_fetch.rewrite_url(url) == url


def test_fetch_happy_path_returns_bytes_and_filename():
    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "audio/wav"}, content=FIXTURE_WAV,
        )

    data, filename, ct = link_fetch.fetch_link(
        "https://example.com/dir/clip.wav",
        resolver=_public_resolver,
        transport=httpx.MockTransport(handler),
    )
    assert data == FIXTURE_WAV
    assert filename == "clip.wav"
    assert ct == "audio/wav"


def test_fetch_reports_download_progress():
    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "video/mp4"}, content=FIXTURE_WAV,
        )

    calls = []
    data, _filename, _ct = link_fetch.fetch_link(
        "https://example.com/dir/clip.mp4",
        resolver=_public_resolver,
        transport=httpx.MockTransport(handler),
        progress_cb=lambda done, total: calls.append((done, total)),
    )
    assert data == FIXTURE_WAV
    # The hook fired, and the final report is the full size with a known total
    # (MockTransport sets Content-Length), so a caller can render real progress.
    assert calls, "progress_cb was never called"
    assert calls[-1] == (len(FIXTURE_WAV), len(FIXTURE_WAV))


def test_fetch_progress_hook_error_never_breaks_download():
    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "video/mp4"}, content=FIXTURE_WAV,
        )

    def _boom(done, total):
        raise RuntimeError("hook blew up")

    # A misbehaving progress hook must not sink the download it only observes.
    data, _filename, ct = link_fetch.fetch_link(
        "https://example.com/dir/clip.mp4",
        resolver=_public_resolver,
        transport=httpx.MockTransport(handler),
        progress_cb=_boom,
    )
    assert data == FIXTURE_WAV
    assert ct == "video/mp4"


def test_fetch_filename_from_content_disposition():
    def handler(request):
        return httpx.Response(
            200,
            headers={
                "content-type": "video/mp4",
                "content-disposition": 'attachment; filename="meeting.mp4"',
            },
            content=b"hello",
        )

    _data, filename, _ct = link_fetch.fetch_link(
        "https://drive.google.com/uc?export=download&id=X",
        resolver=_public_resolver,
        transport=httpx.MockTransport(handler),
    )
    assert filename == "meeting.mp4"


def test_fetch_html_response_is_422():
    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "text/html; charset=utf-8"},
            content=b"<html>share page</html>",
        )

    with pytest.raises(link_fetch.LinkError) as ei:
        link_fetch.fetch_link(
            "https://example.com/share",
            resolver=_public_resolver,
            transport=httpx.MockTransport(handler),
        )
    assert ei.value.status_code == 422
    assert "google photos" in ei.value.detail.lower()


def test_fetch_oversize_stream_is_413():
    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "video/mp4"}, content=b"x" * 500,
        )

    with pytest.raises(link_fetch.LinkError) as ei:
        link_fetch.fetch_link(
            "https://example.com/big.mp4",
            resolver=_public_resolver,
            transport=httpx.MockTransport(handler),
            max_bytes=100,  # tiny cap so 500 bytes trips it
        )
    assert ei.value.status_code == 413
    assert "too large" in ei.value.detail.lower()


@pytest.mark.parametrize("private_ip", [
    "10.0.0.5", "172.16.3.4", "192.168.1.10", "127.0.0.1", "169.254.169.254",
    "::1",
])
def test_fetch_ssrf_private_address_blocked_422(private_ip):
    def handler(request):  # pragma: no cover — must never be reached
        return httpx.Response(200, content=b"should not fetch")

    with pytest.raises(link_fetch.LinkError) as ei:
        link_fetch.fetch_link(
            "https://internal.example.com/x",
            resolver=lambda host: [private_ip],
            transport=httpx.MockTransport(handler),
        )
    assert ei.value.status_code == 422
    assert "private" in ei.value.detail.lower()


def test_fetch_rejects_non_http_scheme():
    with pytest.raises(link_fetch.LinkError) as ei:
        link_fetch.fetch_link(
            "file:///etc/passwd", resolver=_public_resolver,
        )
    assert ei.value.status_code == 422


def test_fetch_ssrf_guards_redirect_hops():
    """A public URL that 302-redirects to an internal address is blocked at the
    redirect hop, not followed."""
    def handler(request):
        if request.url.host == "example.com":
            return httpx.Response(
                302, headers={"location": "http://169.254.169.254/latest/meta"},
            )
        return httpx.Response(200, content=b"secret")  # pragma: no cover

    def resolver(host):
        return ["93.184.216.34"] if host == "example.com" else ["169.254.169.254"]

    with pytest.raises(link_fetch.LinkError) as ei:
        link_fetch.fetch_link(
            "https://example.com/redir",
            resolver=resolver,
            transport=httpx.MockTransport(handler),
        )
    assert ei.value.status_code == 422


# ===========================================================================
# Google Photos share-link tests — real _fetch_photos logic, MockTransport
# routes the short-link → share-page → media-variant hops (never real Google).
# ===========================================================================

PHOTOS_SHORT = "https://photos.app.goo.gl/abc123XYZ"
PHOTOS_SHARE = "https://photos.google.com/share/LONGSHAREID"
PHOTOS_BASE = "https://lh3.googleusercontent.com/pw/AP1GczABC_base-URL"
PHOTOS_VIDEO_BYTES = b"FAKE-MP4-VIDEO-BYTES-" * 8


def _photos_page(base_urls):
    """A minimal share page embedding the given base URLs, as Google's does."""
    embeds = ", ".join(f'"{u}"' for u in base_urls)
    return (
        f"<html><body><script>var media=[{embeds}];</script></body></html>"
    ).encode()


def _photos_handler(
    *, base_urls, video_ok=True, video_ct="video/mp4", d_ct="image/jpeg",
):
    """MockTransport handler for the three Photos hops. ``=dv`` returns a video
    (unless ``video_ok`` is False → 404); ``=d`` returns ``d_ct``."""
    def handler(request):
        host = request.url.host
        raw = str(request.url)
        if host == "photos.app.goo.gl":
            return httpx.Response(302, headers={"location": PHOTOS_SHARE})
        if host == "photos.google.com":
            return httpx.Response(
                200, headers={"content-type": "text/html; charset=utf-8"},
                content=_photos_page(base_urls),
            )
        if host == "lh3.googleusercontent.com":
            if raw.endswith("=dv"):
                if not video_ok:
                    return httpx.Response(404)
                return httpx.Response(
                    200, headers={"content-type": video_ct},
                    content=PHOTOS_VIDEO_BYTES,
                )
            if raw.endswith("=d"):
                return httpx.Response(
                    200, headers={"content-type": d_ct}, content=b"original",
                )
        return httpx.Response(404)  # pragma: no cover — unexpected hop

    return handler


def test_fetch_photos_share_link_downloads_video():
    """Short-link → share page → single base URL → =dv video, downloaded through
    the same capped/streaming path as any direct link."""
    data, filename, ct = link_fetch.fetch_link(
        PHOTOS_SHORT,
        resolver=_public_resolver,
        transport=httpx.MockTransport(_photos_handler(base_urls=[PHOTOS_BASE])),
    )
    assert data == PHOTOS_VIDEO_BYTES
    assert filename == "photos_share.mp4"
    assert ct == "video/mp4"


def test_fetch_photos_multiple_items_422():
    with pytest.raises(link_fetch.LinkError) as ei:
        link_fetch.fetch_link(
            PHOTOS_SHORT,
            resolver=_public_resolver,
            transport=httpx.MockTransport(
                _photos_handler(base_urls=[PHOTOS_BASE, PHOTOS_BASE + "2nd"]),
            ),
        )
    assert ei.value.status_code == 422
    assert "multiple" in ei.value.detail.lower()


def test_fetch_photos_zero_candidates_422():
    with pytest.raises(link_fetch.LinkError) as ei:
        link_fetch.fetch_link(
            PHOTOS_SHORT,
            resolver=_public_resolver,
            transport=httpx.MockTransport(_photos_handler(base_urls=[])),
        )
    assert ei.value.status_code == 422
    assert "couldn't find media" in ei.value.detail.lower()


def test_fetch_photos_image_only_422():
    """=dv unavailable, =d is an image → the honest 'that link is a photo' 422."""
    with pytest.raises(link_fetch.LinkError) as ei:
        link_fetch.fetch_link(
            PHOTOS_SHORT,
            resolver=_public_resolver,
            transport=httpx.MockTransport(
                _photos_handler(base_urls=[PHOTOS_BASE], video_ok=False),
            ),
        )
    assert ei.value.status_code == 422
    assert "photo" in ei.value.detail.lower()


def test_fetch_photos_page_over_cap_413(monkeypatch):
    """A share page larger than the (small, patched) HTML cap is aborted 413,
    never buffered — a hostile 'photos' host can't stream unbounded 'html'."""
    monkeypatch.setattr(link_fetch, "PHOTOS_PAGE_MAX_BYTES", 100)
    big_page = b"<html>" + b"x" * 500 + b"</html>"

    def handler(request):
        if request.url.host == "photos.app.goo.gl":
            return httpx.Response(302, headers={"location": PHOTOS_SHARE})
        if request.url.host == "photos.google.com":
            return httpx.Response(
                200, headers={"content-type": "text/html"}, content=big_page,
            )
        return httpx.Response(404)  # pragma: no cover

    with pytest.raises(link_fetch.LinkError) as ei:
        link_fetch.fetch_link(
            PHOTOS_SHORT,
            resolver=_public_resolver,
            transport=httpx.MockTransport(handler),
        )
    assert ei.value.status_code == 413


# ===========================================================================
# /analyze/link endpoint tests — fetch patched, analysis mocked
# ===========================================================================

class _LinkFakeStore:
    """Minimal fake — just enough for the /analyze/link store path + detail read
    (importlib test-collection mode forbids importing the other suite's fake)."""

    def __init__(self):
        self._by_uid: dict[str, dict[str, dict]] = {}
        self.save_calls: list[dict] = []

    async def save_recording(
        self, uid, *, audio_m4a, video_360p, original_filename,
        original_content_type, original_bytes, duration_seconds, turns,
        analysis, source=None, title=None,
    ):
        import uuid
        from datetime import datetime, timezone
        rid = str(uuid.uuid4())
        meta = {
            "id": rid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "filename": original_filename or "recording",
            "media_type": "video" if video_360p is not None else "audio",
            "duration_seconds": duration_seconds,
            "source": source,
        }
        self._by_uid.setdefault(uid, {})[rid] = {
            "meta": meta, "turns": turns, "analysis": analysis,
        }
        self.save_calls.append({"uid": uid, "recording_id": rid,
                                "audio_m4a": audio_m4a, "source": source})
        return rid

    async def get_recording(self, uid, recording_id):
        r = self._by_uid.get(uid, {}).get(recording_id)
        if r is None:
            return None
        return {**r["meta"], "turns": r["turns"], "analysis": r["analysis"]}


@pytest.fixture
def link_store(monkeypatch):
    """Storage-enabled fake + deterministic derivatives, like the other suites."""
    monkeypatch.setattr(
        main, "build_derivatives",
        lambda data, **kw: audio_ingest.Derivatives(
            audio_m4a=FAKE_AUDIO_M4A, video_360p=None, has_video=False,
            video_note=None,
        ),
    )
    fake = _LinkFakeStore()
    app.state.recordings_store = fake
    yield fake
    del app.state.recordings_store


@pytest.mark.anyio
async def test_analyze_link_happy_path_stores_original_url(client, link_store):
    original_url = "https://drive.google.com/file/d/ABC123/view?usp=sharing"

    def _fake_fetch(url, **kw):
        # Endpoint passes the ORIGINAL url; the helper would rewrite internally.
        assert url == original_url
        return FIXTURE_WAV, "clip.wav", "audio/wav"

    with patch("main.link_fetch.fetch_link", _fake_fetch), \
         patch("main.transcribe_prerecorded", return_value=MOCK_TURNS), \
         patch("main.get_llm_client",
               return_value=_mock_llm(_analyze_llm_json(len(MOCK_TURNS)))):
        resp = await client.post(
            "/analyze/link",
            json={"url": original_url, "consent": True, "store": True},
            headers={"X-Test-Uid": "test-user"},
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["per_turn"]) == len(MOCK_TURNS)
    assert data["stored"] is True

    # Provenance stores the ORIGINAL pasted URL (pre-Drive-rewrite).
    call = link_store.save_calls[0]
    assert call["source"] == {
        "type": "link", "url": original_url, "original_filename": "clip.wav",
    }
    rid = data["recording_id"]
    detail = await client.get(
        f"/recordings/{rid}", headers={"X-Test-Uid": "test-user"},
    )
    assert detail.json()["source"]["type"] == "link"
    assert detail.json()["source"]["url"] == original_url


@pytest.mark.anyio
async def test_analyze_link_consent_false_not_stored(client, link_store):
    with patch("main.link_fetch.fetch_link",
               return_value=(FIXTURE_WAV, "clip.wav", "audio/wav")), \
         patch("main.transcribe_prerecorded", return_value=MOCK_TURNS), \
         patch("main.get_llm_client",
               return_value=_mock_llm(_analyze_llm_json(len(MOCK_TURNS)))):
        resp = await client.post(
            "/analyze/link",
            json={"url": "https://example.com/clip.wav", "consent": False},
            headers={"X-Test-Uid": "test-user"},
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["stored"] is False
    assert data["storage_note"] == "consent not given"
    assert link_store.save_calls == []


@pytest.mark.anyio
async def test_analyze_link_propagates_html_422(client):
    def _boom(url, **kw):
        raise link_fetch.LinkError(422, link_fetch._HTML_HINT)

    with patch("main.link_fetch.fetch_link", _boom):
        resp = await client.post(
            "/analyze/link",
            json={"url": "https://example.com/share"},
            headers={"X-Test-Uid": "test-user"},
        )
    assert resp.status_code == 422
    assert "google photos" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_analyze_link_propagates_oversize_413(client):
    def _boom(url, **kw):
        raise link_fetch.LinkError(413, "linked file too large: exceeds the 200MB limit")

    with patch("main.link_fetch.fetch_link", _boom):
        resp = await client.post(
            "/analyze/link",
            json={"url": "https://example.com/big.mp4"},
            headers={"X-Test-Uid": "test-user"},
        )
    assert resp.status_code == 413


@pytest.mark.anyio
async def test_analyze_link_propagates_ssrf_422(client):
    def _boom(url, **kw):
        raise link_fetch.LinkError(
            422, "link resolves to a private/internal address — not allowed",
        )

    with patch("main.link_fetch.fetch_link", _boom):
        resp = await client.post(
            "/analyze/link",
            json={"url": "https://internal.example.com/x"},
            headers={"X-Test-Uid": "test-user"},
        )
    assert resp.status_code == 422
    assert "private" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_analyze_link_requires_auth_401(client, monkeypatch):
    from auth import get_current_uid
    monkeypatch.delitem(app.dependency_overrides, get_current_uid)
    resp = await client.post(
        "/analyze/link", json={"url": "https://example.com/clip.wav"},
    )
    assert resp.status_code == 401
