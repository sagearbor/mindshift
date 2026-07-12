"""Unit tests for ``link_fetch.resolve_media_url`` — the resolve-ONLY (no
download) path behind GET /recordings/{id}/source_url (HD replay).

Unlike ``fetch_link`` this returns the CURRENT direct media URL for the client to
stream itself; it must never download the media bytes. Exercised with an injected
DNS resolver + an httpx MockTransport (no real network / DNS), mirroring
test_analyze_link's harness so the SSRF guard, Drive rewrite, passthrough hint,
and the Google Photos page re-parse are all covered for real.
"""

import httpx
import pytest

import link_fetch


def _public_resolver(host):
    return ["93.184.216.34"]  # a routable public address


# ---------------------------------------------------------------------------
# Drive rewrite / direct passthrough (no network — resolver only)
# ---------------------------------------------------------------------------

def test_resolve_drive_share_returns_direct_download():
    url, ct = link_fetch.resolve_media_url(
        "https://drive.google.com/file/d/ABC123xyz/view?usp=sharing",
        resolver=_public_resolver,
    )
    assert url == "https://drive.google.com/uc?export=download&id=ABC123xyz"
    # No bytes fetched → no content-type known for a Drive uc? url.
    assert ct is None


def test_resolve_direct_url_passthrough_with_extension_hint():
    url, ct = link_fetch.resolve_media_url(
        "https://example.com/dir/clip.mp4", resolver=_public_resolver,
    )
    assert url == "https://example.com/dir/clip.mp4"
    assert ct == "video/mp4"  # best-effort hint from the extension


def test_resolve_blocks_private_address():
    with pytest.raises(link_fetch.LinkError) as ei:
        link_fetch.resolve_media_url(
            "https://internal.example/clip.mp4",
            resolver=lambda host: ["169.254.169.254"],
        )
    assert ei.value.status_code == 422


def test_resolve_rejects_non_http_scheme():
    with pytest.raises(link_fetch.LinkError) as ei:
        link_fetch.resolve_media_url(
            "file:///etc/passwd", resolver=_public_resolver,
        )
    assert ei.value.status_code == 422


# ---------------------------------------------------------------------------
# Google Photos share pages — re-parse to the =dv URL WITHOUT downloading media
# ---------------------------------------------------------------------------

PHOTOS_SHORT = "https://photos.app.goo.gl/abc123XYZ"
PHOTOS_SHARE = "https://photos.google.com/share/LONGSHAREID"
PHOTOS_BASE = "https://lh3.googleusercontent.com/pw/AP1GczABC_base-URL"


def _photos_page(base_urls):
    embeds = ", ".join(f'"{u}"' for u in base_urls)
    return (
        f"<html><body><script>var media=[{embeds}];</script></body></html>"
    ).encode()


def _page_only_handler(base_urls):
    """Serves the short-link redirect + share page, and FAILS the test if the
    media host (lh3) is ever fetched — resolve must not download the bytes."""
    def handler(request):
        host = request.url.host
        if host == "photos.app.goo.gl":
            return httpx.Response(302, headers={"location": PHOTOS_SHARE})
        if host == "photos.google.com":
            return httpx.Response(
                200, headers={"content-type": "text/html; charset=utf-8"},
                content=_photos_page(base_urls),
            )
        raise AssertionError(
            f"resolve_media_url must not fetch media; hit {host}"
        )
    return handler


def test_resolve_photos_returns_dv_url_without_downloading():
    url, ct = link_fetch.resolve_media_url(
        PHOTOS_SHORT,
        resolver=_public_resolver,
        transport=httpx.MockTransport(_page_only_handler([PHOTOS_BASE])),
    )
    assert url == PHOTOS_BASE + "=dv"
    assert ct == "video/mp4"


def test_resolve_photos_multiple_items_422():
    with pytest.raises(link_fetch.LinkError) as ei:
        link_fetch.resolve_media_url(
            PHOTOS_SHORT,
            resolver=_public_resolver,
            transport=httpx.MockTransport(
                _page_only_handler([PHOTOS_BASE, PHOTOS_BASE + "2nd"]),
            ),
        )
    assert ei.value.status_code == 422


def test_resolve_photos_no_media_422():
    with pytest.raises(link_fetch.LinkError) as ei:
        link_fetch.resolve_media_url(
            PHOTOS_SHORT,
            resolver=_public_resolver,
            transport=httpx.MockTransport(_page_only_handler([])),
        )
    assert ei.value.status_code == 422
