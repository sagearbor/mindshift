"""Server-side download of a user-provided media link for POST /analyze/link.

The user pastes a share URL (e.g. a Google Drive link); the server fetches the
bytes itself and feeds them to the SAME analyze+derivative-store pipeline as an
upload. Three honest guardrails, each mapping to one HTTP status at the endpoint:

* SSRF guard (:func:`_guard_ssrf`) — the host is resolved and any private,
  loopback, link-local, reserved, or multicast address is rejected (422). We
  must never let a user make the server fetch an internal endpoint. Redirects
  are followed MANUALLY so every hop's host is re-checked (auto-following would
  let a public URL 302 to ``169.254.169.254``).
* Size cap — the body is streamed and aborted past 200MB (413), so a huge link
  can never exhaust memory.
* HTML detection — a ``text/html`` response is a share/preview page, not a file
  (422 with actionable guidance). Google Drive ``/file/d/<ID>/`` and ``open?id=``
  share forms are rewritten to the direct-download URL first.

Google Photos share links (``photos.app.goo.gl`` / ``photos.google.com``) are a
special case handled BEFORE the HTML rejection: the share page IS html, but it
embeds the media's base URL, from which we can fetch the actual video bytes. See
:func:`_fetch_photos`.

Everything here is synchronous (httpx.Client) — the caller runs it in a worker
thread, matching the Deepgram/ffmpeg house pattern.
"""

from __future__ import annotations

import ipaddress
import logging
import mimetypes
import re
import socket
from urllib.parse import parse_qs, urlparse

import httpx

logger = logging.getLogger(__name__)

# Hard cap on downloaded bytes — matches the chunked-upload ceiling so a link and
# an upload can carry the same maximum recording.
MAX_LINK_BYTES = 200 * 1024 * 1024
# Generous: a 200MB download over a slow link can take minutes.
LINK_TIMEOUT_S = 600.0
# Bounded redirect chain — each hop is SSRF-checked before it is followed.
MAX_REDIRECTS = 5

# The Google Photos share page is HTML, never media — cap its fetch far below the
# media ceiling so a hostile "photos" host can't stream 200MB of "html" at us.
PHOTOS_PAGE_MAX_BYTES = 5 * 1024 * 1024
# Hosts whose links are Google Photos share pages (short-link + expanded form).
PHOTOS_HOSTS = ("photos.app.goo.gl", "photos.google.com")
# The media base URLs the share page embeds. Appending "=dv" yields the video
# bytes, "=d" the original photo. Empirically verified against a real link.
_PHOTOS_MEDIA_RE = re.compile(r"https://lh3\.googleusercontent\.com/pw/[A-Za-z0-9_\-]+")
# Some Google endpoints (the Photos share page in particular) only serve the
# real HTML to a browser-shaped User-Agent — send a desktop one on every request
# (harmless for direct/Drive links, required for the Photos probe).
_DESKTOP_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_HTML_HINT = (
    "That link isn't a direct file link — use a direct file URL, a Google Drive "
    "share link, or a Google Photos share link of a single video."
)

# Honest failures for the (unofficial) Google Photos share-page parse. If Google
# changes the page format the regex simply matches nothing and we surface this
# 422 — we never crash on a format we no longer recognise.
_PHOTOS_NO_MEDIA = (
    "couldn't find media in that Google Photos link — make sure the link shares "
    "a single photo or video"
)
_PHOTOS_MULTIPLE = (
    "that Google Photos link contains multiple items — share a single video "
    "instead"
)
_PHOTOS_IS_PHOTO = (
    "that link is a photo — MindShift analyzes conversations, share a video or "
    "audio file"
)


class LinkError(Exception):
    """A link could not be fetched. ``status_code`` is the honest HTTP status the
    endpoint should return (422 for a bad/blocked/HTML link, 413 for oversize)."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def rewrite_url(url: str) -> str:
    """Rewrite a Google Drive SHARE url to its direct-download form.

    ``drive.google.com/file/d/<ID>/view`` and ``drive.google.com/open?id=<ID>``
    (and ``uc?id=<ID>`` without export) become
    ``https://drive.google.com/uc?export=download&id=<ID>``. Any other URL is
    returned unchanged. Purely syntactic — no network.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in ("drive.google.com", "www.drive.google.com"):
        return url
    m = re.search(r"/file/d/([^/]+)", parsed.path)
    file_id = m.group(1) if m else None
    if file_id is None:
        qs = parse_qs(parsed.query)
        ids = qs.get("id")
        file_id = ids[0] if ids else None
    if not file_id:
        return url
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _default_resolver(host: str) -> list[str]:
    """Resolve ``host`` to a list of IP strings via the system resolver."""
    infos = socket.getaddrinfo(host, None)
    return [info[4][0] for info in infos]


def _guard_ssrf(host: str, resolver) -> None:
    """Raise :class:`LinkError` (422) unless every address ``host`` resolves to
    is a routable public address. Blocks private/loopback/link-local/reserved/
    multicast/unspecified — the SSRF surface (10.x, 172.16-31, 192.168, 127.,
    169.254., ::1, …)."""
    if not host:
        raise LinkError(422, "invalid link: no host")
    try:
        ips = resolver(host)
    except Exception as exc:  # noqa: BLE001 — DNS failure → treat as unfetchable
        raise LinkError(422, f"could not resolve link host: {host}") from exc
    if not ips:
        raise LinkError(422, f"could not resolve link host: {host}")
    for ip in ips:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            raise LinkError(422, f"link host resolved to an invalid address: {ip}")
        if (
            addr.is_private or addr.is_loopback or addr.is_link_local
            or addr.is_reserved or addr.is_multicast or addr.is_unspecified
        ):
            raise LinkError(
                422,
                "link resolves to a private/internal address — not allowed",
            )


def _filename_from(resp: httpx.Response, parsed_url) -> str | None:
    """Best-effort filename: Content-Disposition ``filename=`` first, then the
    URL path's basename. None when neither yields anything usable."""
    disp = resp.headers.get("content-disposition", "")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', disp)
    if m:
        name = m.group(1).strip()
        if name:
            return name
    tail = (parsed_url.path or "").rsplit("/", 1)[-1]
    return tail or None


def _content_type(resp: httpx.Response) -> str:
    """The bare, lower-cased content-type (no ``; charset=…`` suffix)."""
    return resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

# Report download progress at most this often (bytes) so a large body fires a
# manageable number of callbacks — enough to keep a job's heartbeat fresh without
# a write per network chunk.
_PROGRESS_STEP_BYTES = 1024 * 1024


def _run_download(
    client: httpx.Client,
    start_url: str,
    *,
    resolver,
    max_bytes: int,
    progress_cb=None,
) -> tuple[bytes, httpx.Response, "object"]:
    """Follow redirects (SSRF-guarding every hop), stream the terminal body with a
    hard ``max_bytes`` cap, and return ``(data, terminal_response, parsed_url)``.

    Content-type policy (html rejection, video/image checks) is intentionally
    left to the caller — this is the raw, guarded fetch shared by the direct/Drive
    path and the Google Photos media path. Raises :class:`LinkError` on a bad
    scheme, blocked host, HTTP error, oversize body (413), or too many redirects.
    The returned response is already closed with its body fully read into
    ``data``; only its headers should be inspected afterward.

    ``progress_cb(bytes_downloaded, bytes_total_or_None)`` — when given — is called
    as the body streams (throttled to ~1MB steps; ``bytes_total`` from
    Content-Length, or None when the server omits it). A callback exception is
    swallowed so a progress hook can never break the download."""
    current = start_url
    for _ in range(MAX_REDIRECTS + 1):
        parsed = urlparse(current)
        if parsed.scheme not in ("http", "https"):
            raise LinkError(422, "only http(s) links are supported")
        _guard_ssrf(parsed.hostname or "", resolver)

        request = client.build_request("GET", current)
        try:
            resp = client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise LinkError(422, f"could not fetch link: {exc}") from exc

        if resp.is_redirect:
            location = resp.headers.get("location")
            resp.close()
            if not location:
                raise LinkError(422, "link redirected without a destination")
            # Resolve relative redirects against the current URL.
            current = str(httpx.URL(current).join(location))
            continue

        # Terminal response — read it (bounded), then hand it back.
        try:
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise LinkError(
                    422, f"could not fetch link (HTTP {resp.status_code})",
                ) from exc
            total = 0
            chunks: list[bytes] = []
            cl = resp.headers.get("content-length")
            bytes_total = int(cl) if cl and cl.isdigit() else None
            reported = 0
            if progress_cb is not None:
                _safe_progress(progress_cb, 0, bytes_total)
            try:
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise LinkError(
                            413,
                            "linked file too large: exceeds the "
                            f"{max_bytes // (1024 * 1024)}MB limit",
                        )
                    chunks.append(chunk)
                    if (
                        progress_cb is not None
                        and total - reported >= _PROGRESS_STEP_BYTES
                    ):
                        reported = total
                        _safe_progress(progress_cb, total, bytes_total)
            except httpx.HTTPError as exc:
                raise LinkError(
                    422, f"could not read link body: {exc}",
                ) from exc
            if progress_cb is not None:
                _safe_progress(progress_cb, total, bytes_total or total)
            return b"".join(chunks), resp, parsed
        finally:
            resp.close()

    raise LinkError(422, "too many redirects")


def _safe_progress(progress_cb, done: int, total: "int | None") -> None:
    """Invoke a download progress hook, swallowing any error — a progress callback
    must never break the download it is only observing."""
    try:
        progress_cb(done, total)
    except Exception:  # noqa: BLE001 — progress reporting is best-effort
        logger.debug("download progress callback failed", exc_info=True)


def fetch_link(
    url: str,
    *,
    resolver=None,
    transport: "httpx.BaseTransport | None" = None,
    timeout: float = LINK_TIMEOUT_S,
    max_bytes: int = MAX_LINK_BYTES,
    progress_cb=None,
) -> tuple[bytes, str | None, str | None]:
    """Download the media at ``url`` and return ``(data, filename, content_type)``.

    Rewrites Drive share links, resolves Google Photos share links to their video
    bytes, enforces http(s), SSRF-guards every redirect hop, streams the body with
    a hard size cap, and rejects a bare HTML response. Raises :class:`LinkError`
    with the appropriate status on any of these. ``resolver`` and ``transport``
    are injectable for tests (real DNS / real sockets are used by default).

    ``progress_cb(bytes_downloaded, bytes_total_or_None)`` — when given — is called
    as the MEDIA body streams so a caller (the async job runner) can surface a live
    download size and keep its heartbeat fresh. For a Google Photos link it fires
    only for the actual media fetch, not the small HTML share-page probe."""
    resolver = resolver or _default_resolver

    client = httpx.Client(
        follow_redirects=False,
        timeout=timeout,
        transport=transport,
        headers={"User-Agent": _DESKTOP_UA},
    )
    try:
        host = (urlparse(url).hostname or "").lower()
        if host in PHOTOS_HOSTS:
            # Handled BEFORE the generic html rejection: the share page IS html.
            return _fetch_photos(
                client, url, resolver=resolver, max_bytes=max_bytes,
                progress_cb=progress_cb,
            )

        data, resp, parsed = _run_download(
            client, rewrite_url(url), resolver=resolver, max_bytes=max_bytes,
            progress_cb=progress_cb,
        )
        content_type = _content_type(resp)
        if content_type == "text/html":
            raise LinkError(422, _HTML_HINT)
        filename = _filename_from(resp, parsed)
        return data, filename, content_type or None
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Google Photos share links
# ---------------------------------------------------------------------------

def _photos_filename(content_type: str | None) -> str:
    """A stable share filename for the downloaded Photos media, extension chosen
    from the served content-type (``photos_share.mp4`` by default)."""
    ext = mimetypes.guess_extension(content_type) if content_type else None
    return f"photos_share{ext or '.mp4'}"


def _photos_media_download(
    client: httpx.Client,
    media_url: str,
    *,
    resolver,
    max_bytes: int,
    progress_cb=None,
) -> tuple[bytes | None, str | None]:
    """Fetch one Photos media variant URL. Returns ``(data, content_type)`` on a
    successful fetch, or ``(None, None)`` if the variant is unavailable (an HTTP
    error) so the caller can fall back to the other variant. An oversize body
    (413) is a real, actionable error and is re-raised, never swallowed."""
    try:
        data, resp, _parsed = _run_download(
            client, media_url, resolver=resolver, max_bytes=max_bytes,
            progress_cb=progress_cb,
        )
    except LinkError as exc:
        if exc.status_code == 413:
            raise
        return None, None
    return data, _content_type(resp) or None



def _photos_page_url(url: str) -> str:
    """Short photos.app.goo.gl links serve a Firebase Dynamic-Links interstitial
    ("open in the app?") to plain HTTP clients instead of redirecting — the
    share page (and its embedded media URLs) never loads. The interstitial's
    own "continue in browser" parameter, ``_imcp=1``, deterministically skips
    it (verified against a real link: without it, a 34KB JS shell; with it, a
    302 to photos.google.com/share/… whose page embeds the media). Long-form
    photos.google.com URLs don't need it and are returned unchanged."""
    parsed = urlparse(url)
    if (parsed.hostname or "").lower() != "photos.app.goo.gl":
        return url
    sep = "&" if parsed.query else "?"
    return f"{url}{sep}_imcp=1"

def _fetch_photos(
    client: httpx.Client,
    original_url: str,
    *,
    resolver,
    max_bytes: int,
    progress_cb=None,
) -> tuple[bytes, str | None, str | None]:
    """Resolve a Google Photos share link to a single video's bytes.

    Fetches the share page (html, capped well below the media ceiling), extracts
    the embedded ``lh3.googleusercontent.com/pw/…`` base URL(s), and requires
    exactly one. For that one, tries ``=dv`` (video) then ``=d`` (original) — a
    video is downloaded and returned; a photo is a clear 422. This parses an
    UNOFFICIAL page format: if Google changes it the regex matches nothing and we
    return a 422 (:data:`_PHOTOS_NO_MEDIA`) rather than crash.

    ``progress_cb`` is passed only to the MEDIA fetch (not this small HTML page
    probe) so a caller sees real video-download progress."""
    page, _resp, _parsed = _run_download(
        client, _photos_page_url(original_url), resolver=resolver,
        max_bytes=PHOTOS_PAGE_MAX_BYTES,
    )
    html = page.decode("utf-8", "replace")

    # De-duplicate matches preserving first-seen order (dict keys are ordered).
    candidates = list(dict.fromkeys(_PHOTOS_MEDIA_RE.findall(html)))
    if not candidates:
        raise LinkError(422, _PHOTOS_NO_MEDIA)
    if len(candidates) > 1:
        raise LinkError(422, _PHOTOS_MULTIPLE)
    base = candidates[0]

    # Prefer the video variant.
    data, ct = _photos_media_download(
        client, base + "=dv", resolver=resolver, max_bytes=max_bytes,
        progress_cb=progress_cb,
    )
    if data is not None and ct and ct.startswith("video/"):
        return data, _photos_filename(ct), ct

    # Fall back to the original variant — mainly to give an honest, specific
    # error when the shared item is a photo (or a video reachable only via =d).
    data, ct = _photos_media_download(
        client, base + "=d", resolver=resolver, max_bytes=max_bytes,
        progress_cb=progress_cb,
    )
    if ct and ct.startswith("image/"):
        raise LinkError(422, _PHOTOS_IS_PHOTO)
    if data is not None and ct and ct.startswith("video/"):
        return data, _photos_filename(ct), ct

    # Neither variant yielded usable media — treat as an extraction failure.
    raise LinkError(422, _PHOTOS_NO_MEDIA)


# ---------------------------------------------------------------------------
# Resolve-only (no download) — for HD replay from the user's own source
# ---------------------------------------------------------------------------
# The replay feature hands the CLIENT a direct media URL to stream itself from
# the source CDN; the server must NOT proxy the bytes. So this path re-derives
# the CURRENT direct URL (Photos pages change; a share link may have been
# revoked) WITHOUT downloading the media — reusing the same SSRF-guarded fetch
# and Photos regex as the download path so the two can never drift.

def _resolve_photos_media_url(client: httpx.Client, original_url: str, *, resolver) -> str:
    """Fetch a Google Photos share page and return the video-variant direct URL
    (``base + "=dv"``) for its single embedded item — WITHOUT downloading the
    media. Reuses :func:`_run_download` (SSRF-guarded, size-capped) and the same
    :data:`_PHOTOS_MEDIA_RE` as :func:`_fetch_photos`. Raises :class:`LinkError`
    (422) when the page embeds no / multiple items, or its (unofficial) format is
    no longer recognised."""
    page, _resp, _parsed = _run_download(
        client, _photos_page_url(original_url), resolver=resolver,
        max_bytes=PHOTOS_PAGE_MAX_BYTES,
    )
    html = page.decode("utf-8", "replace")
    candidates = list(dict.fromkeys(_PHOTOS_MEDIA_RE.findall(html)))
    if not candidates:
        raise LinkError(422, _PHOTOS_NO_MEDIA)
    if len(candidates) > 1:
        raise LinkError(422, _PHOTOS_MULTIPLE)
    # "=dv" is the full-res video stream variant (verified against a real link).
    return candidates[0] + "=dv"


def resolve_media_url(
    url: str,
    *,
    resolver=None,
    transport: "httpx.BaseTransport | None" = None,
    timeout: float = LINK_TIMEOUT_S,
) -> tuple[str, str | None]:
    """Resolve a durable share/link URL to its CURRENT direct media URL WITHOUT
    downloading the bytes, returning ``(direct_media_url, content_type_hint)``.

    Used by the HD-replay endpoint: the client streams the returned URL straight
    from the source CDN (we never proxy the media). Three shapes:

    * Google Photos share pages (``photos.app.goo.gl`` / ``photos.google.com``)
      are re-fetched and re-parsed to their embedded
      ``lh3.googleusercontent.com/pw/…`` base; the ``=dv`` video variant is
      returned (hint ``video/mp4``).
    * Google Drive share links are rewritten to their ``uc?export=download`` form
      (no download, so no content-type is known → hint ``None``).
    * Any other http(s) URL passes through unchanged; the hint is guessed from
      the URL's extension when possible, else ``None``.

    The full SSRF guard applies — every hop of the Photos page fetch is checked,
    and the host of a direct/Drive URL is resolved-and-checked before it is
    handed back, so a link that resolves to an internal address can never be
    surfaced to the client. Raises :class:`LinkError` (422) on a non-http(s)
    scheme, a blocked host, or an unparseable/empty Photos page — the endpoint
    surfaces that honestly and the client falls back to the stored derivative.
    ``resolver`` and ``transport`` are injectable for tests."""
    resolver = resolver or _default_resolver
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise LinkError(422, "only http(s) links are supported")
    host = (parsed.hostname or "").lower()

    if host in PHOTOS_HOSTS:
        client = httpx.Client(
            follow_redirects=False,
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": _DESKTOP_UA},
        )
        try:
            media_url = _resolve_photos_media_url(client, url, resolver=resolver)
        finally:
            client.close()
        return media_url, "video/mp4"

    # Drive share → direct-download; every other URL passes through unchanged.
    direct = rewrite_url(url)
    _guard_ssrf((urlparse(direct).hostname or "").lower(), resolver)
    # Best-effort hint from the ORIGINAL url's extension (a Drive uc? url has
    # none → None, which is honest: we haven't fetched the bytes to learn it).
    content_type_hint = mimetypes.guess_type(parsed.path)[0]
    return direct, content_type_hint
