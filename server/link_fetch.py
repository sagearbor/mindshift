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

Everything here is synchronous (httpx.Client) — the caller runs it in a worker
thread, matching the Deepgram/ffmpeg house pattern.
"""

from __future__ import annotations

import ipaddress
import logging
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

_HTML_HINT = (
    "That link isn't a direct file link. Google Photos links aren't supported "
    "yet — share the file to Drive or use a direct link."
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


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def fetch_link(
    url: str,
    *,
    resolver=None,
    transport: "httpx.BaseTransport | None" = None,
    timeout: float = LINK_TIMEOUT_S,
    max_bytes: int = MAX_LINK_BYTES,
) -> tuple[bytes, str | None, str | None]:
    """Download the media at ``url`` and return ``(data, filename, content_type)``.

    Rewrites Drive share links, enforces http(s), SSRF-guards every redirect hop,
    streams the body with a hard size cap, and rejects an HTML response. Raises
    :class:`LinkError` with the appropriate status on any of these. ``resolver``
    and ``transport`` are injectable for tests (real DNS / real sockets are used
    by default)."""
    resolver = resolver or _default_resolver
    current = rewrite_url(url)

    client = httpx.Client(
        follow_redirects=False,
        timeout=timeout,
        transport=transport,
        headers={"User-Agent": "MindShift/1.0 (+link-ingest)"},
    )
    try:
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

            # Terminal response — read it (bounded), then hand back the bytes.
            try:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise LinkError(
                        422, f"could not fetch link (HTTP {resp.status_code})",
                    ) from exc
                content_type = (
                    resp.headers.get("content-type", "").split(";", 1)[0]
                    .strip().lower()
                )
                if content_type == "text/html":
                    raise LinkError(422, _HTML_HINT)
                total = 0
                chunks: list[bytes] = []
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
                except httpx.HTTPError as exc:
                    raise LinkError(
                        422, f"could not read link body: {exc}",
                    ) from exc
                data = b"".join(chunks)
                filename = _filename_from(resp, parsed)
                return data, filename, content_type or None
            finally:
                resp.close()

        raise LinkError(422, "too many redirects")
    finally:
        client.close()
