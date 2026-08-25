"""In-app calls (2026-08-25) — the call model, the process-local registry,
WebRTC signaling relay, the shared (merged) transcript, and end-of-call
persistence.

Why MindShift places the call itself
------------------------------------
Android (10+) and iOS hand third-party apps silence while the phone is on a
cellular/VoIP call, so "phone Mom and get coached on the same Pixel" cannot
work — unless MindShift IS the call. Then the app owns the mic, each
participant's audio is its own stream, and every side can be coached.

Design (owner-approved)
-----------------------
* Audio flows PEER-TO-PEER over WebRTC (a full mesh — with three people
  every client holds two connections); this server only relays the
  signaling (SDP offers/answers, ICE candidates) between the members'
  existing ``/ws/session/{id}`` sockets — no second auth path, no media
  through the server.
* Roles. A call has up to two ``participant``s (the people being coached —
  the host is always one) and up to one ``therapist`` (an observer: she
  sees the merged transcript and both participants' suggestions/nudges
  read-only, her own turns are transcribed and merged, and she is never
  coached). ``max_participants`` (2 or 3, default 3) caps the total.
* Each member transcribes THEMSELVES on-device (the phone's fast loop or
  Safari Web Speech) and reports ``turn_local`` exactly as a solo session
  does. The server merges every member's turns into ONE shared call
  transcript, ordered by server arrival, and pushes each turn to every
  OTHER socket as a ``transcript`` event so each screen shows the whole
  conversation without ever hearing the remote audio through its own STT.
* Attribution is STRUCTURAL: member = speaker. In a call every turn is
  labelled by SLOT — the host ``"Speaker A"``, the second participant
  ``"Speaker B"``, the therapist ``"Speaker C"`` (``SLOT_LABELS``) —
  whatever label the phone's own diarizer used, so all the existing
  "Speaker X" machinery (side-aware coaching, the episode label ladder,
  mid-call naming) works unchanged. A member's OWN turns are ``is_self``
  for them; everyone else's are OTHER turns. Coaching runs PER PARTICIPANT
  with the merged context: nudges on your own turns, suggestions about
  BOTH other people's turns. The therapist's socket additionally receives
  every ``suggestion`` / ``tone_flag`` / ``speaker_identity`` the
  participants get, tagged ``for_uid`` (absent on your own events).
* Time base: each member's ``turn_local`` times are on ITS phone's capture
  clock. The merged transcript orders by server arrival (``seq``) and
  keeps BOTH clocks: ``local_start_time``/``local_end_time`` are the
  sender's, ``start_time``/``end_time`` are re-based onto the CALL timeline
  by a per-member offset fixed at that member's first turn (server seconds
  since the call started minus the turn's local end).
* On end, ONE EPISODE PER PARTICIPANT (never for the therapist) through
  the existing live-session ingest (``routers.sessions.ingest_live``,
  mode ``"call"``): the full merged turn list (the therapist's turns
  included, as "Mom (therapist)"), ``self_speaker`` = that participant's
  slot label, the others named as the participant knows them, auto-share
  via the therapist LINK exactly as a solo session — and, when a therapist
  is IN the call, a direct share grant to her as well, whether or not a
  link exists yet (she was on the call; the participant chose her by
  letting her join). The phone must NOT also POST /sessions/live for a
  call session (it would store its own half twice).

Registry scope — PROCESS-LOCAL, like the watch relay (server/watch/relay.py):
the sockets of a call must land on the same process. Production runs
Cloud Run with ``--max-instances 1`` for exactly this reason (the watch
relay already needs it); a multi-instance deployment would need this map
in a shared store (a flagged, later decision).

Names — who the OTHERS are called, for viewer X looking at Y (in order):
X's own mid-call naming of Y (a ``speaker_label`` frame on Y's slot label,
persisted call-wide), Y's self-declared ``display_name`` (from ``POST
/calls``/``join`` or ``call_join``), Y's account email (resolved once,
best-effort — the same thing the therapist link shows both ways), else Y's
slot label; a therapist's name carries the " (therapist)" suffix for
everyone but herself. Enrolled people (``/voice/people``) are voiceprints,
not accounts, so they cannot be mapped to a member automatically; naming
someone mid-call is the way to attach one.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import logging
import os
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

# How long a call with NO socket bound (nobody connected yet, or joined over
# REST and never connected) stays joinable. A call with a live socket never
# expires by clock — it ends when its last socket leaves or a participant
# ends it.
CALL_TTL_MINUTES = int(os.getenv("MINDSHIFT_CALL_TTL_MINUTES", "180"))
CALL_TTL_MAX_MINUTES = 24 * 60
# Ended calls are kept this long so GET /calls/{id} can still hand back the
# episode ids to a phone that reconnects after the sockets closed.
CALL_RETENTION_MINUTES = int(os.getenv("MINDSHIFT_CALL_RETENTION_MINUTES", "60"))
# Bounds mirror routers/sessions.py's LIVE_MAX_TURNS / LIVE_MAX_TRANSCRIPT_CHARS
# so a persisted call episode can never exceed what ingest accepts.
CALL_MAX_TURNS = 400
CALL_MAX_TRANSCRIPT_CHARS = 60_000
# An SDP offer is a few KB; an ICE candidate a few hundred bytes. 64 KiB is
# far past any real signaling message and stops a client using the relay as
# a free data channel through the server.
RTC_PAYLOAD_MAX_BYTES = 64 * 1024
# Per-socket bound on signaling frames: ICE gathering is bursty (a few
# dozen candidates in the first second, one connection per peer), then
# nearly silent. Past the burst a member's frames are refused with a
# reason — one phone can't use the relay to flood the others' sockets.
RTC_SIGNAL_RATE_PER_S = 20.0
RTC_SIGNAL_BURST = 60
# Bound on calls held in this process (abuse guard; a real deployment sees
# a handful at a time). Retained ENDED calls are evicted early before a new
# call is refused, so they can never crowd out live ones.
MAX_CALLS = int(os.getenv("MINDSHIFT_MAX_CALLS", "500"))
# Un-ended (open or active) calls ONE account may host at a time — a single
# tenant creating calls in a loop hits 429 long before MAX_CALLS.
MAX_OPEN_CALLS_PER_HOST = int(os.getenv("MINDSHIFT_MAX_OPEN_CALLS_PER_HOST", "3"))
# How long a remote-turn delivery may block the sender's receive loop.
DELIVERY_TIMEOUT_S = 2.0
# A phone's turn_local whose padded range contains the midpoint of a CLOUD
# row this member pushed earlier (the server's transcriber finalized the
# span before the phone did) is the same words: it replaces that row in
# place instead of appending. Same pad + midpoint rule as the pipeline's
# local-range suppression (audio_pipeline.LOCAL_RANGE_PAD_S); only the last
# few rows are candidates (the race is always between adjacent turns).
CLOUD_DUP_PAD_S = 0.25
CLOUD_DUP_LOOKBACK = 8
# Whether the persisted call episodes get the batch analysis + "what you
# could have said" reflection scheduled (the same two LLM passes a solo
# session gets). Env-overridable like the other knobs; tests turn them off.
ANALYZE_ON_END = os.getenv("MINDSHIFT_CALL_ANALYZE", "1") != "0"
REFLECT_ON_END = os.getenv("MINDSHIFT_CALL_REFLECT", "1") != "0"

DEFAULT_STUN_URL = "stun:stun.l.google.com:19302"
# How long an ephemeral TURN credential stays valid (override with
# MINDSHIFT_TURN_TTL_SECONDS, read per handout by turn_ttl_seconds()). Must
# outlive the longest call a phone can be on (CALL_TTL_MINUTES defaults to
# 180) because a peer connection keeps using the credential it was created
# with — a relay allocation is refreshed against the SAME username/password
# for the life of the connection. Default 4h; floor a minute, cap a day.
TURN_TTL_SECONDS = 4 * 3600
TURN_TTL_MIN_SECONDS = 60
TURN_TTL_MAX_SECONDS = 24 * 3600
# The per-member half of the TURN REST username. Anything a coturn username
# can't hold (a colon would split the timestamp, whitespace breaks the STUN
# attribute) is stripped; an empty result falls back to "member".
_TURN_KEY_STRIP = re.compile(r"[^A-Za-z0-9._@=+-]")
TURN_KEY_MAX = 64
DISPLAY_NAME_MAX = 60
JOIN_CODE_LEN = 6
# No 0/O/1/I — the code is read out loud or typed from a text.
JOIN_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
# Brute-force guards on the code (32^6 ≈ 1e9 codes; REST is IP-rate-limited,
# the WebSocket `call_join` frame is not). Per SOCKET: wrong codes before
# that socket's further call_join frames are refused (a new socket costs a
# fresh token handshake). Per CALL: wrong codes from anyone before the code
# is burned for good — the named invitee never needed it and is unaffected;
# the host starts a new call. A mistyped code is a handful, never fifty.
JOIN_ATTEMPTS_MAX = 8
JOIN_CODE_FAILURES_MAX = 50

ROLE_PARTICIPANT = "participant"
ROLE_THERAPIST = "therapist"
ROLES = (ROLE_PARTICIPANT, ROLE_THERAPIST)
MAX_PARTICIPANT_ROLE = 2      # coached people per call
MAX_THERAPIST_ROLE = 1        # observers per call
DEFAULT_MAX_PARTICIPANTS = 3  # total members (2 participants + 1 therapist)
THERAPIST_SUFFIX = " (therapist)"

# Slot → the label every turn of that member carries. Participants take A
# then B (the host is always A); the therapist is always C, whenever she
# joins — so labels never depend on join order across roles.
SLOT_LABELS = {"A": "Speaker A", "B": "Speaker B", "C": "Speaker C"}
PARTICIPANT_SLOTS = ("A", "B")
THERAPIST_SLOT = "C"
SELF_PERSON_ID = "self"

STATUS_OPEN = "open"        # created; fewer than two members joined
STATUS_ACTIVE = "active"    # at least two members joined
STATUS_ENDED = "ended"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_ms(received_at: Any) -> int | None:
    """Milliseconds since a row's ``received_at`` stamp (None if unparseable)
    — how far the phone trailed the server's transcriber on the same span."""
    if not isinstance(received_at, str):
        return None
    try:
        then = datetime.fromisoformat(received_at)
    except ValueError:
        return None
    return int((datetime.now(timezone.utc) - then).total_seconds() * 1000)


def turn_urls() -> list[str]:
    """``MINDSHIFT_TURN_URLS`` split on commas. Empty = STUN only."""
    return [u.strip() for u in os.getenv("MINDSHIFT_TURN_URLS", "").split(",") if u.strip()]


def _bounded_ttl(ttl: int) -> int:
    return max(TURN_TTL_MIN_SECONDS, min(ttl, TURN_TTL_MAX_SECONDS))


def turn_ttl_seconds() -> int:
    """The configured ephemeral-credential lifetime, bounded to [1min, 24h].
    Garbage in the env falls back to the default rather than raising — a
    typo'd TTL must never take the call server down."""
    raw = os.getenv("MINDSHIFT_TURN_TTL_SECONDS", "").strip()
    try:
        ttl = int(raw) if raw else TURN_TTL_SECONDS
    except ValueError:
        ttl = TURN_TTL_SECONDS
    return _bounded_ttl(ttl)


def turn_key(user_key: str | None) -> str:
    """The per-member half of a TURN REST username, made safe to embed."""
    cleaned = _TURN_KEY_STRIP.sub("", (user_key or "").strip())[:TURN_KEY_MAX]
    return cleaned or "member"


def turn_rest_credentials(
    user_key: str | None,
    *,
    secret: str,
    ttl_seconds: int | None = None,
    now: float | None = None,
) -> tuple[str, str, int]:
    """One short-lived TURN credential in the standard *TURN REST API* shape
    (draft-uberti-behave-turn-rest-00, what coturn's ``use-auth-secret`` /
    ``static-auth-secret`` implements, and what Cloudflare/Metered/coturn all
    accept):

        username   = "<unix-expiry>:<user_key>"
        credential = base64( HMAC-SHA1( secret, username ) )

    The TURN server recomputes the same HMAC from its own copy of the secret,
    so nothing has to be provisioned per user and a leaked credential dies at
    ``expiry``. Returns ``(username, credential, expires_at_unix)``.

    Each call member gets its OWN username (their uid), so one member's
    credential can't be handed around as everyone's shared password, and a
    stolen one is traceable and short-lived.
    """
    ttl = turn_ttl_seconds() if ttl_seconds is None else _bounded_ttl(int(ttl_seconds))
    expiry = int(now if now is not None else time.time()) + ttl
    username = f"{expiry}:{turn_key(user_key)}"
    digest = hmac.new(secret.encode("utf-8"), username.encode("utf-8"), hashlib.sha1).digest()
    return username, base64.b64encode(digest).decode("ascii"), expiry


# --- Cloudflare Realtime TURN (vendor-minted credentials) -------------------
#
# Cloudflare's TURN service mints credentials through its own REST API; the
# shared-secret (RFC 5766 REST) scheme above does not apply to it, nor to
# Twilio or Metered. The account-scoped API token is NOT what lives here: a
# TURN key token can only mint relay credentials for that one key.
#
# Set MINDSHIFT_TURN_KEY_ID + MINDSHIFT_TURN_KEY_TOKEN to enable. A failed or
# slow mint degrades to STUN-only rather than failing the call; the result is
# cached until shortly before expiry so a busy call does not mint per member.

CLOUDFLARE_TURN_MINT_URL = (
    "https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers"
)
CLOUDFLARE_MINT_TIMEOUT_S = 4.0
_CF_CACHE: dict[str, Any] = {"expires_at": 0.0, "servers": None}
_CF_LOCK = threading.Lock()


def cloudflare_turn_key() -> tuple[str, str] | None:
    """``(key_id, key_token)`` when Cloudflare TURN is configured, else None."""
    key_id = os.getenv("MINDSHIFT_TURN_KEY_ID", "").strip()
    token = os.getenv("MINDSHIFT_TURN_KEY_TOKEN", "").strip()
    return (key_id, token) if key_id and token else None


def cloudflare_ice_servers() -> list[dict] | None:
    """Mint (or reuse) a Cloudflare ICE server list, or None when Cloudflare
    TURN is not configured. Never raises: on any failure the caller falls
    back to the STUN/static path so a call still tries a direct connection."""
    key = cloudflare_turn_key()
    if key is None:
        return None
    key_id, token = key
    now = time.time()
    with _CF_LOCK:
        cached = _CF_CACHE.get("servers")
        if cached is not None and now < float(_CF_CACHE.get("expires_at") or 0.0):
            return cached
    ttl = turn_ttl_seconds()
    try:
        import httpx

        resp = httpx.post(
            CLOUDFLARE_TURN_MINT_URL.format(key_id=key_id),
            headers={"Authorization": f"Bearer {token}"},
            json={"ttl": ttl},
            timeout=CLOUDFLARE_MINT_TIMEOUT_S,
        )
        resp.raise_for_status()
        servers = resp.json().get("iceServers")
    except Exception as exc:  # noqa: BLE001 — a relay is an optimisation
        logger.warning("cloudflare TURN mint failed (%s); falling back to STUN", exc)
        return None
    if isinstance(servers, dict):
        servers = [servers]
    if not isinstance(servers, list) or not servers:
        logger.warning("cloudflare TURN mint returned no iceServers; falling back")
        return None
    if not any(s.get("username") for s in servers if isinstance(s, dict)):
        logger.warning("cloudflare TURN mint returned no credentials; falling back")
        return None
    with _CF_LOCK:
        # Re-mint a minute before expiry so no client is handed a stale pair.
        _CF_CACHE["servers"] = servers
        _CF_CACHE["expires_at"] = now + max(60.0, ttl - 60.0)
    return servers


def _reset_cloudflare_cache() -> None:
    """Test seam: drop the minted-credential cache."""
    with _CF_LOCK:
        _CF_CACHE["servers"] = None
        _CF_CACHE["expires_at"] = 0.0


def ice_servers(user_key: str | None = None) -> list[dict]:
    """The ICE server list handed to one member: Google's public STUN by
    default, plus a TURN relay when the deployment configured one
    (``MINDSHIFT_TURN_URLS``, comma-separated).

    Two ways to authenticate to that relay, in order of preference:

    * VENDOR-MINTED (``MINDSHIFT_TURN_KEY_ID`` + ``MINDSHIFT_TURN_KEY_TOKEN``)
      — Cloudflare Realtime TURN. The vendor mints a short-lived
      username/credential per handout over its REST API; Cloudflare (like
      Twilio and Metered) does NOT support the shared-secret scheme below,
      so this is the path for those providers. The key token stays on the
      server; only the minted pair reaches a client. A mint failure degrades
      to the STUN-only list rather than sinking the call — the client's
      preflight then says a relay is unavailable.
    * EPHEMERAL (``MINDSHIFT_TURN_SECRET``, optionally
      ``MINDSHIFT_TURN_REALM``) — a TURN REST credential minted per member
      per handout, valid ``MINDSHIFT_TURN_TTL_SECONDS`` (default 4h). Nothing
      long-lived ever leaves the server. Works with self-hosted coturn.
    * STATIC (``MINDSHIFT_TURN_USERNAME`` / ``MINDSHIFT_TURN_CREDENTIAL``) —
      one password shared by everyone who ever joins a call. Still supported
      (some vendors only issue static creds), but it is the weaker path: it
      does not expire and every client holds it.

    Read per call (not at import) so the owner can add TURN with a config
    change, and so each handout gets a fresh expiry. Without TURN, two phones
    on carrier-grade NAT may fail to connect — the client's ICE preflight
    (live/call/iceProbe.ts) says so out loud rather than failing silently.
    """
    vendor = cloudflare_ice_servers()
    if vendor is not None:
        return vendor
    servers: list[dict] = [{"urls": [DEFAULT_STUN_URL]}]
    urls = turn_urls()
    if urls:
        entry: dict[str, Any] = {"urls": urls}
        secret = os.getenv("MINDSHIFT_TURN_SECRET", "").strip()
        if secret:
            username, credential, _expiry = turn_rest_credentials(user_key, secret=secret)
            entry["username"] = username
            entry["credential"] = credential
        else:
            username = os.getenv("MINDSHIFT_TURN_USERNAME", "").strip()
            credential = os.getenv("MINDSHIFT_TURN_CREDENTIAL", "").strip()
            if username:
                entry["username"] = username
            if credential:
                entry["credential"] = credential
        realm = os.getenv("MINDSHIFT_TURN_REALM", "").strip()
        if realm:
            # Not part of RTCIceServer — carried so a failing relay can be
            # diagnosed from the client ("which realm did the server mean?").
            entry["realm"] = realm
        servers.append(entry)
    return servers


def turn_status() -> dict[str, Any]:
    """How TURN is configured right now, for ``GET /calls/ice`` and the
    client preflight line. Never leaks the secret or the credential."""
    if cloudflare_turn_key():
        return {
            "turn_configured": True,
            "turn_credential_mode": "cloudflare",
            "ttl_seconds": turn_ttl_seconds(),
        }
    urls = turn_urls()
    if not urls:
        mode = "none"
    elif os.getenv("MINDSHIFT_TURN_SECRET", "").strip():
        mode = "ephemeral"
    elif os.getenv("MINDSHIFT_TURN_USERNAME", "").strip():
        mode = "static"
    else:
        mode = "open"  # URLs with no auth at all (a test relay)
    return {
        "turn_configured": bool(urls),
        "turn_credential_mode": mode,
        "ttl_seconds": turn_ttl_seconds() if mode == "ephemeral" else None,
    }


def join_url(join_code: str) -> str:
    """Where a text-message invite points. Defaults to the web app (any
    browser with a mic can be the far end — Mom on her iPhone in Safari);
    override with ``MINDSHIFT_CALL_JOIN_BASE`` (e.g. ``mindshift://call``)."""
    base = os.getenv("MINDSHIFT_CALL_JOIN_BASE", "https://arborfam-hub.web.app/call").rstrip("/")
    return f"{base}/{join_code}"


def new_join_code() -> str:
    return "".join(secrets.choice(JOIN_CODE_ALPHABET) for _ in range(JOIN_CODE_LEN))


def normalize_join_code(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    code = raw.strip().upper().replace("-", "").replace(" ", "")
    if len(code) != JOIN_CODE_LEN or any(c not in JOIN_CODE_ALPHABET for c in code):
        return None
    return code


def clean_display_name(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    name = " ".join(raw.split())[:DISPLAY_NAME_MAX]
    return name or None


def clean_role(raw: object) -> str:
    """``participant`` unless the caller says ``therapist``; anything else
    (a typo, a wrong type) raises so a role is never silently guessed."""
    if raw is None:
        return ROLE_PARTICIPANT
    if isinstance(raw, str) and raw.strip().lower() in ROLES:
        return raw.strip().lower()
    raise CallError(422, "role must be 'participant' or 'therapist'")


class TokenBucket:
    """A plain token bucket: ``burst`` tokens to start, ``rate_per_s`` back
    per second, never more than ``burst``. ``allow()`` spends one."""

    def __init__(self, *, rate_per_s: float, burst: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.rate = float(rate_per_s)
        self.burst = float(burst)
        self._clock = clock
        self._tokens = self.burst
        self._last = clock()

    def allow(self) -> bool:
        now = self._clock()
        self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class CallError(Exception):
    """A rejected call operation: ``status`` is the HTTP code the REST
    router answers with, ``detail`` the short, value-free reason (the WS
    handler puts the same text in its ``{"error": ...}`` frame)."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


# ---------------------------------------------------------------------------
# Members + endpoints
# ---------------------------------------------------------------------------

class CallEndpoint:
    """What a bound WebSocket session exposes to the call (implemented by
    audio_pipeline's per-session endpoint). Every method is best-effort on
    the socket side — a dead peer socket must never raise into the sender.

    * ``send_json(payload)`` — a frame to this member's phone.
    * ``on_remote_turn(turn, display_name=)`` — someone else said
      something: render it (``transcript``) and, for a participant, coach
      on it. A ``turn`` carrying ``replaces_seq`` is the corrected copy of
      a row already delivered: render it in place, do not coach again.
    * ``set_peer_name(label, display_name)`` — another member's name (as
      this member sees it) changed; update the running session's naming.
    * ``detach()`` — the call ended; the session keeps coaching solo.
    """

    uid: str
    session_id: str

    async def send_json(self, payload: dict) -> None:  # pragma: no cover — protocol
        raise NotImplementedError

    async def on_remote_turn(self, turn: dict, *, display_name: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def set_peer_name(self, label: str, display_name: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def detach(self) -> None:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Participant:
    """One member of the call — a coached ``participant`` or the observing
    ``therapist`` (the class name predates roles; every member is one)."""

    uid: str
    slot: str
    role: str = ROLE_PARTICIPANT
    email: str | None = None
    declared_name: str | None = None
    joined_at: str = field(default_factory=now_iso)
    endpoint: CallEndpoint | None = None
    # Sender clock → call timeline (see the module docstring). None until
    # this member's first turn.
    offset_s: float | None = None
    episode_id: str | None = None
    # Therapist emails the episode was shared with at persist time (the
    # linked therapist via auto-share, the in-call therapist directly).
    shared_with: list[str] = field(default_factory=list)
    turn_count: int = 0

    @property
    def label(self) -> str:
        return SLOT_LABELS[self.slot]

    @property
    def connected(self) -> bool:
        return self.endpoint is not None

    @property
    def is_therapist(self) -> bool:
        return self.role == ROLE_THERAPIST


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

@dataclass
class Call:
    call_id: str
    host_uid: str
    join_code: str
    created_at: str
    expires_at: str
    invitee_uid: str | None = None
    invitee_email: str | None = None
    max_participants: int = DEFAULT_MAX_PARTICIPANTS
    status: str = STATUS_OPEN
    participants: dict[str, Participant] = field(default_factory=dict)  # insertion order = host first
    # viewer uid → {target uid → name the viewer gave them} (call-wide naming).
    names: dict[str, dict[str, str]] = field(default_factory=dict)
    turns: list[dict] = field(default_factory=list)
    seq: int = 0
    started_at: str | None = None
    ended_at: str | None = None
    ended_by: str | None = None
    end_reason: str | None = None
    # Wrong join codes presented so far (see JOIN_CODE_FAILURES_MAX).
    code_failures: int = 0
    store: Any = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    _t0: float = field(default_factory=time.monotonic, repr=False, compare=False)
    _ended_wall: float | None = field(default=None, repr=False, compare=False)

    # -- lookups ------------------------------------------------------------

    def participant(self, uid: str) -> Participant | None:
        return self.participants.get(uid)

    def by_label(self, label: str) -> Participant | None:
        for p in self.participants.values():
            if p.label == label:
                return p
        return None

    def others_of(self, uid: str) -> list[Participant]:
        return [p for p in self.participants.values() if p.uid != uid]

    def peer_of(self, uid: str) -> Participant | None:
        """The OTHER coached participant (None for a call that has none yet;
        for the therapist, the host)."""
        me = self.participants.get(uid)
        for p in self.participants.values():
            if p.uid != uid and not p.is_therapist:
                if me is None or me.is_therapist or not p.is_therapist:
                    return p
        return None

    def therapist(self) -> Participant | None:
        for p in self.participants.values():
            if p.is_therapist:
                return p
        return None

    def coached(self) -> list[Participant]:
        return [p for p in self.participants.values() if not p.is_therapist]

    @property
    def is_full(self) -> bool:
        return len(self.participants) >= self.max_participants

    @property
    def ended(self) -> bool:
        return self.status == STATUS_ENDED

    def connected_participants(self) -> list[Participant]:
        return [p for p in self.participants.values() if p.connected]

    def can_see(self, uid: str) -> bool:
        return uid in self.participants or uid == self.invitee_uid

    def expired(self, now: datetime | None = None) -> bool:
        """A call with NO socket bound past its TTL — open, or active only
        through REST joins nobody ever connected to (a call with a live
        socket never expires by clock; the last socket out ends it)."""
        if self.ended or self.connected_participants():
            return False
        now = now or datetime.now(timezone.utc)
        return datetime.fromisoformat(self.expires_at) <= now

    def elapsed_s(self) -> float:
        return time.monotonic() - self._t0

    # -- names --------------------------------------------------------------

    def display_name_for(self, viewer_uid: str, target_uid: str) -> str:
        """What ``viewer_uid`` calls ``target_uid`` (module docstring order)."""
        target = self.participants.get(target_uid)
        if viewer_uid == target_uid:
            return "You"
        suffix = THERAPIST_SUFFIX if target is not None and target.is_therapist else ""
        given = (self.names.get(viewer_uid) or {}).get(target_uid)
        if given:
            return given + suffix
        if target is not None:
            if target.declared_name:
                return target.declared_name + suffix
            if target.email:
                return target.email + suffix
            return target.label + suffix
        if target_uid == self.invitee_uid and self.invitee_email:
            return self.invitee_email
        return "Caller"

    def participant_rows(self, viewer_uid: str) -> list[dict]:
        return [
            {
                "uid": p.uid,
                "slot": p.slot,
                "label": p.label,
                "role": p.role,
                "display_name": self.display_name_for(viewer_uid, p.uid),
                "is_self": p.uid == viewer_uid,
                "connected": p.connected,
                "joined_at": p.joined_at,
            }
            for p in self.participants.values()
        ]

    def state_for(self, viewer_uid: str) -> dict:
        """The ``call_state`` body as ``viewer_uid`` sees it (``is_self`` and
        ``display_name`` are relative to the viewer)."""
        me = self.participants.get(viewer_uid)
        peer_label = None
        if me is not None:
            peer = self.peer_of(viewer_uid)
            if peer is not None:
                peer_label = peer.label
            elif not me.is_therapist:
                # The other participant's label is fixed by slot before they join.
                peer_label = SLOT_LABELS["B" if me.slot == "A" else "A"]
        therapist = self.therapist()
        return {
            "call_id": self.call_id,
            "status": self.status,
            "host_uid": self.host_uid,
            "max_participants": self.max_participants,
            "self_uid": viewer_uid,
            "self_role": me.role if me else None,
            "self_label": me.label if me else None,
            "peer_label": peer_label,
            "therapist_label": SLOT_LABELS[THERAPIST_SLOT],
            "therapist_uid": therapist.uid if therapist else None,
            "participants": self.participant_rows(viewer_uid),
            "invitee": (
                {"uid": self.invitee_uid, "email": self.invitee_email}
                if (self.invitee_uid or self.invitee_email) else None
            ),
            # Per-member ephemeral TURN creds when a secret is configured:
            # the viewer's uid is the username, so nobody shares a password.
            "ice_servers": ice_servers(viewer_uid),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "end_reason": self.end_reason,
            "turn_count": len(self.turns),
            "episode_id": me.episode_id if me else None,
            "shared_with": list(me.shared_with) if me else [],
        }

    def rest_view(self, viewer_uid: str) -> dict:
        """``GET /calls/{id}`` — the state plus the join code/url, which only
        a member (or the invitee) ever sees."""
        return {
            **self.state_for(viewer_uid),
            "join_code": self.join_code,
            "join_url": join_url(self.join_code),
            "invitee_uid": self.invitee_uid,
            "invitee_email": self.invitee_email,
        }

    # -- wire helpers -------------------------------------------------------

    async def _send(self, p: Participant, payload: dict) -> None:
        ep = p.endpoint
        if ep is None:
            return
        try:
            await asyncio.wait_for(ep.send_json(payload), timeout=DELIVERY_TIMEOUT_S)
        except Exception:  # noqa: BLE001 — a dead peer socket must never raise into the sender
            logger.debug("call %s: send to %s failed", self.call_id, p.uid, exc_info=True)

    async def broadcast_state(self) -> None:
        for p in list(self.participants.values()):
            if p.connected:
                await self._send(p, {"type": "call_state", **self.state_for(p.uid)})

    async def fan_out(self, from_uid: str, payload: dict) -> None:
        """A coaching event a participant just received (suggestion, tone
        flag, identity verdict), copied to every connected THERAPIST socket
        tagged ``for_uid`` — the observer's read-only view of the coaching.
        Never to another participant (their coaching is their own)."""
        if self.ended:
            return
        for p in list(self.participants.values()):
            if p.is_therapist and p.uid != from_uid and p.connected:
                await self._send(p, {**payload, "for_uid": from_uid})

    def _seed_peer_names(self) -> None:
        for p in self.participants.values():
            if p.endpoint is None:
                continue
            for other in self.others_of(p.uid):
                with contextlib.suppress(Exception):
                    p.endpoint.set_peer_name(other.label, self.display_name_for(p.uid, other.uid))

    # -- membership ---------------------------------------------------------

    def _next_slot(self, role: str) -> str:
        if role == ROLE_THERAPIST:
            if self.therapist() is not None:
                raise CallError(409, "call already has a therapist")
            return THERAPIST_SLOT
        taken = {p.slot for p in self.coached()}
        for slot in PARTICIPANT_SLOTS:
            if slot not in taken:
                return slot
        raise CallError(409, "call already has two participants")

    def add_participant(
        self, uid: str, *, email: str | None, display_name: str | None, role: str = ROLE_PARTICIPANT,
    ) -> Participant:
        """Join (REST or WS). Idempotent for an existing member (the
        name/email refresh; the role cannot change); raises CallError when
        the call is over, full, or the role's seat is taken."""
        existing = self.participants.get(uid)
        if existing is not None:
            if display_name:
                existing.declared_name = display_name
            if email and not existing.email:
                existing.email = email
            return existing
        if self.ended:
            raise CallError(410, "call has ended")
        if self.expired():
            raise CallError(410, "call has expired")
        if self.is_full:
            raise CallError(409, "call is full")
        slot = self._next_slot(role)
        p = Participant(uid=uid, slot=slot, role=role, email=email, declared_name=display_name)
        self.participants[uid] = p
        if len(self.participants) >= 2:
            self.status = STATUS_ACTIVE
        return p

    async def bind(self, uid: str, endpoint: CallEndpoint, *, store: Any = None, display_name: str | None = None) -> Participant:
        """Attach a live WS session to this member. A second socket for the
        same uid (reconnect) replaces the first; the old one is detached."""
        async with self.lock:
            if self.ended:
                raise CallError(410, "call has ended")
            p = self.participants.get(uid)
            if p is None:
                raise CallError(403, "not a participant of this call")
            if display_name:
                p.declared_name = display_name
            if p.endpoint is not None and p.endpoint is not endpoint:
                with contextlib.suppress(Exception):
                    p.endpoint.detach()
            if p.endpoint is not endpoint:
                # A new socket is a new capture clock (the phone's session
                # restarted at 0): re-fix the sender→call-timeline offset at
                # its next turn, or its turns would land before the ones
                # already merged.
                p.offset_s = None
            p.endpoint = endpoint
            if store is not None:
                self.store = store
            if self.started_at is None:
                self.started_at = now_iso()
            self._seed_peer_names()
            await self.broadcast_state()
            return p

    async def leave(self, uid: str, endpoint: CallEndpoint | None = None) -> bool:
        """This member's socket is going away (graceful stop or a
        disconnect). Idempotent. When it was the LAST connected socket the
        call ends (persisting the episodes) while this endpoint is still
        attached, so it receives its ``call_ended`` (episode id) before its
        own ``session_complete``. Returns True when the call ended here."""
        async with self.lock:
            p = self.participants.get(uid)
            if p is None or p.endpoint is None or (endpoint is not None and p.endpoint is not endpoint):
                return False
            if self.ended:
                p.endpoint = None
                return False
            others = [q for q in self.participants.values() if q.uid != uid and q.connected]
            if not others:
                await self._end_locked(reason="all participants left", ended_by=uid)
                return True
            p.endpoint = None
            await self.broadcast_state()
            return False

    # -- names --------------------------------------------------------------

    async def set_viewer_name(self, viewer_uid: str, target_uid: str, name: str) -> None:
        async with self.lock:
            self.names.setdefault(viewer_uid, {})[target_uid] = name
            self._seed_peer_names()
            await self.broadcast_state()

    async def set_declared_name(self, uid: str, name: str) -> None:
        async with self.lock:
            p = self.participants.get(uid)
            if p is None:
                return
            p.declared_name = name
            self._seed_peer_names()
            await self.broadcast_state()

    # -- signaling ----------------------------------------------------------

    async def relay_signal(self, from_uid: str, payload: dict, *, to_uid: str | None = None) -> None:
        """Forward one signaling payload to the addressed (connected) member
        verbatim, stamped ``from``. ``to`` may be omitted only while the
        call has exactly two members (it means "the other one"); with three
        every client holds a connection per peer, so the address is
        required. Raises CallError when there is nobody to deliver to — the
        client waits for ``call_state`` to show the peer connected before
        (re)offering."""
        if from_uid not in self.participants:
            raise CallError(403, "not a participant of this call")
        if to_uid is None:
            if len(self.participants) > 2:
                raise CallError(400, "'to' is required in a call with more than two members")
            others = self.others_of(from_uid)
            target = others[0] if others else None
        else:
            target = self.participants.get(to_uid)
        if target is None or target.uid == from_uid:
            raise CallError(404, "peer has not joined")
        if not target.connected:
            raise CallError(409, "peer not connected")
        await self._send(target, {
            "type": "rtc_signal", "call_id": self.call_id, "from": from_uid, "payload": payload,
        })

    # -- the merged transcript ----------------------------------------------

    def _timeline(self, p: Participant, start: float, end: float) -> tuple[float, float]:
        if p.offset_s is None:
            p.offset_s = max(0.0, self.elapsed_s() - end)
        return round(start + p.offset_s, 3), round(end + p.offset_s, 3)

    def _cloud_duplicate(self, uid: str, local_start: float, local_end: float) -> dict | None:
        """The recent CLOUD row of ``uid`` that a phone turn spanning
        ``[local_start, local_end]`` (the member's own capture clock — the
        same clock the server's transcriber stamps) duplicates, if any."""
        lo, hi = local_start - CLOUD_DUP_PAD_S, local_end + CLOUD_DUP_PAD_S
        for t in reversed(self.turns[-CLOUD_DUP_LOOKBACK:]):
            if t.get("participant_uid") != uid or t.get("transcript_source") != "cloud":
                continue
            ls, le = t.get("local_start_time"), t.get("local_end_time")
            if ls is None or le is None:
                continue
            if lo <= (float(ls) + float(le)) / 2.0 <= hi:
                return t
        return None

    async def _deliver(self, uid: str, row: dict, *, replaces_seq: int | None = None) -> None:
        """Push one merged row to every OTHER connected member. ``replaces_seq``
        marks the row as the corrected copy of a row those members already
        rendered (see push_turn); it rides the wire as
        ``transcript.replaces_seq`` so a client can swap it in place by
        ``seq`` instead of appending a second line. A dead peer socket must
        never raise into the sender."""
        payload = row if replaces_seq is None else {**row, "replaces_seq": replaces_seq}
        for other in self.others_of(uid):
            if other.endpoint is None:
                continue
            try:
                await asyncio.wait_for(
                    other.endpoint.on_remote_turn(payload, display_name=self.display_name_for(other.uid, uid)),
                    timeout=DELIVERY_TIMEOUT_S,
                )
            except Exception:  # noqa: BLE001 — the sender's session must not sink on a peer
                logger.warning(
                    "call %s: delivering turn %d to %s failed", self.call_id, row["seq"], other.uid,
                    exc_info=True,
                )

    async def push_turn(self, uid: str, event: Any) -> dict | None:
        """A member's own finalized turn (a validated ``TurnLocalEvent`` or
        an equivalent dict from the server-STT fallback): append it to the
        merged transcript and deliver it to every OTHER connected member as
        an OTHER turn. Returns the merged row (None when the call is over).

        A phone turn that duplicates a CLOUD row this member pushed just
        before (the server's transcriber won the race for the phone's first
        span) REPLACES that row in place — same ``seq``, same position, the
        phone's text/tone/prosody — and is delivered AGAIN, tagged
        ``replaces_seq``: the others already rendered the transcriber's
        wording (no ``text_tone``, no sender clock), and only a second
        delivery can correct the line on their screens. They are not coached
        on it twice — same words, and the cloud copy already went through the
        coach (see audio_pipeline's ``on_remote_turn``)."""
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        async with self.lock:
            if self.ended:
                return None
            p = self.participants.get(uid)
            if p is None:
                raise CallError(403, "not a participant of this call")
            local_start, local_end = float(data.get("start_time") or 0.0), float(data.get("end_time") or 0.0)
            start, end = self._timeline(p, local_start, local_end)
            source = data.get("transcript_source") or "on-device"
            dup = self._cloud_duplicate(uid, local_start, local_end) if source != "cloud" else None
            if dup is None:
                self.seq += 1
            row = {
                "seq": dup["seq"] if dup is not None else self.seq,
                "participant_uid": uid,
                "slot": p.slot,
                "role": p.role,
                "speaker": p.label,
                "text": str(data.get("text") or ""),
                "start_time": start,
                "end_time": end,
                "local_start_time": data.get("start_time"),
                "local_end_time": data.get("end_time"),
                "transcript_source": source,
                "speaker_match_score": data.get("speaker_match_score"),
                "prosody": data.get("prosody"),
                "text_tone": data.get("text_tone"),
                "suggestion": data.get("suggestion"),
                "suggestion_source": data.get("suggestion_source"),
                "tts_source": data.get("tts_source"),
                "received_at": now_iso(),
            }
            if dup is not None:
                # How far the phone trailed the transcriber on this span — the
                # width of the race a hold-back would have to cover. Logged so
                # production tells us whether that stays sub-second.
                logger.info(
                    "call %s: %s's phone turn replaces its cloud duplicate seq %d (%s ms later)",
                    self.call_id, uid, dup["seq"], _age_ms(dup.get("received_at")),
                )
                replaced_seq = int(dup["seq"])
                dup.clear()
                dup.update(row)
                await self._deliver(uid, dup, replaces_seq=replaced_seq)
                return dup
            self.turns.append(row)
            p.turn_count += 1
            if len(self.turns) > CALL_MAX_TURNS:
                del self.turns[:-CALL_MAX_TURNS]
            await self._deliver(uid, row)
            return row

    def turns_for(self, viewer_uid: str, session_id: str) -> list[dict]:
        """The merged transcript as ``viewer_uid``'s episode stores it: own
        turns ``is_self`` (the reserved ``self`` person id), everyone else's
        OTHER (their phones' ``self`` verdicts must not leak into this
        viewer's episode), bounded like ingest."""
        out: list[dict] = []
        for t in self.turns:
            own = t["participant_uid"] == viewer_uid
            out.append({
                "type": "turn_local",
                "session_id": session_id,
                "speaker": t["speaker"],
                "text": t["text"],
                "start_time": t["start_time"],
                "end_time": t["end_time"],
                "transcript_source": t["transcript_source"],
                "is_self": own,
                "speaker_person_id": SELF_PERSON_ID if own else None,
                "speaker_match_score": t.get("speaker_match_score") if own else None,
                "prosody": t.get("prosody"),
                "text_tone": t.get("text_tone"),
                "suggestion": t.get("suggestion"),
                "suggestion_source": t.get("suggestion_source"),
                "tts_source": t.get("tts_source"),
                # Both clocks + provenance (kept by live_sessions.storage_turns).
                "call_seq": t["seq"],
                "participant_uid": t["participant_uid"],
                "local_start_time": t.get("local_start_time"),
                "local_end_time": t.get("local_end_time"),
            })
        out = out[-CALL_MAX_TURNS:]
        while out and sum(len(t["text"]) for t in out) > CALL_MAX_TRANSCRIPT_CHARS:
            out.pop(0)
        return out

    # -- ending -------------------------------------------------------------

    async def end(self, *, reason: str, ended_by: str | None, store: Any = None) -> None:
        async with self.lock:
            if store is not None:
                self.store = store
            await self._end_locked(reason=reason, ended_by=ended_by)

    async def _end_locked(self, *, reason: str, ended_by: str | None) -> None:
        if self.ended:
            return
        self.status = STATUS_ENDED
        self.ended_at = now_iso()
        self.ended_by = ended_by
        self.end_reason = reason
        self._ended_wall = time.monotonic()
        await self._persist_episodes()
        for p in list(self.participants.values()):
            if p.connected:
                frame = {
                    "type": "call_ended",
                    "call_id": self.call_id,
                    "reason": reason,
                    "ended_by": ended_by,
                    "episode_id": p.episode_id,
                    "recording_id": p.episode_id,
                    "shared_with": list(p.shared_with),
                    "turn_count": len(self.turns),
                }
                if p.is_therapist:
                    # The observer's view: every participant's episode (she
                    # was granted each). A participant learns only its own.
                    frame["episodes"] = {q.uid: q.episode_id for q in self.coached()}
                await self._send(p, frame)
        for p in list(self.participants.values()):
            ep, p.endpoint = p.endpoint, None
            if ep is not None:
                with contextlib.suppress(Exception):
                    ep.detach()
        logger.info(
            "call %s ended (%s) by %s: %d turns, episodes %s",
            self.call_id, reason, ended_by, len(self.turns),
            {p.uid: p.episode_id for p in self.participants.values()},
        )

    def _title_for(self, p: Participant) -> str:
        names = [self.display_name_for(p.uid, o.uid) for o in self.others_of(p.uid)]
        if not names:
            return "Call"
        if len(names) == 1:
            return f"Call with {names[0]}"
        return "Call with " + ", ".join(names[:-1]) + f" and {names[-1]}"

    async def _persist_episodes(self) -> None:
        """One live-session episode per coached PARTICIPANT (mode ``call``)
        through the existing ingest — best-effort per participant: a store
        failure for one never blocks another's episode or the
        ``call_ended``. The in-call therapist is granted each episode
        directly (see the module docstring) on top of the link's auto-share."""
        if self.store is None:
            if self.turns:
                logger.warning("call %s: recording storage is not enabled — nothing persisted", self.call_id)
            return
        from routers.sessions import ingest_live

        session_id = f"call-{self.call_id}"
        therapist = self.therapist()
        for p in self.coached():
            turns = self.turns_for(p.uid, session_id)
            if not turns:
                continue
            labels: dict[str, dict] = {
                p.label: {"display_name": "You", "person_id": SELF_PERSON_ID, "is_self": True},
            }
            for other in self.others_of(p.uid):
                labels[other.label] = {
                    "display_name": self.display_name_for(p.uid, other.uid),
                    "person_id": None, "is_self": False,
                }
            try:
                out = await ingest_live(
                    self.store, p.uid,
                    session_id=session_id,
                    started_at=self.started_at or self.created_at,
                    ended_at=self.ended_at or now_iso(),
                    mode="call",
                    turn_events=turns,
                    tone_flags=[],
                    identities=[],
                    speaker_labels=labels,
                    title=self._title_for(p),
                    context="",
                    analyze=ANALYZE_ON_END,
                    reflect=REFLECT_ON_END,
                )
            except Exception:  # noqa: BLE001 — recorded, never raised into the socket path
                logger.warning(
                    "call %s: persisting %s's episode failed", self.call_id, p.uid, exc_info=True,
                )
                continue
            p.episode_id = out.episode_id
            p.shared_with = list(out.shared_with)
            if therapist is not None and therapist.uid != p.uid:
                await self._share_with_therapist(p, therapist, out.episode_id)

    async def _share_with_therapist(self, p: Participant, therapist: Participant, recording_id: str) -> None:
        """The same per-episode grant Replay's "Share with…" and the link's
        auto-share make (``store.add_share``) — for the therapist who was ON
        the call. Skipped when the link already granted her; best-effort."""
        who = therapist.email or therapist.uid
        if who in p.shared_with:
            return
        add_share = getattr(self.store, "add_share", None)
        if not callable(add_share):
            return
        try:
            shares = await add_share(
                p.uid, recording_id,
                recipient_uid=therapist.uid,
                recipient_email=therapist.email or "",
                owner_email=p.email,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "call %s: sharing %s's episode with the in-call therapist failed",
                self.call_id, p.uid, exc_info=True,
            )
            return
        if shares is not None:
            p.shared_with.append(who)


# ---------------------------------------------------------------------------
# Registry (process-local)
# ---------------------------------------------------------------------------

class CallRegistry:
    def __init__(self) -> None:
        self._calls: dict[str, Call] = {}
        self._by_code: dict[str, str] = {}

    def reset(self) -> None:
        self._calls.clear()
        self._by_code.clear()

    def __len__(self) -> int:
        return len(self._calls)

    def sweep(self) -> None:
        """Expire OPEN calls past their TTL and forget ENDED calls past the
        retention window. Cheap; run on every registry access."""
        now = datetime.now(timezone.utc)
        for call in list(self._calls.values()):
            if call.expired(now):
                call.status = STATUS_ENDED
                call.ended_at = now_iso()
                call.end_reason = "expired"
                call._ended_wall = time.monotonic()
                logger.info("call %s expired unjoined", call.call_id)
            if call.ended and call._ended_wall is not None and (
                time.monotonic() - call._ended_wall > CALL_RETENTION_MINUTES * 60
            ):
                self._forget(call)

    def _forget(self, call: Call) -> None:
        self._calls.pop(call.call_id, None)
        self._by_code.pop(call.join_code, None)

    def create(
        self,
        host_uid: str,
        *,
        host_email: str | None = None,
        display_name: str | None = None,
        invitee_uid: str | None = None,
        invitee_email: str | None = None,
        ttl_minutes: int | None = None,
        max_participants: int | None = None,
    ) -> Call:
        self.sweep()
        hosting = sum(1 for c in self._calls.values() if c.host_uid == host_uid and not c.ended)
        if hosting >= MAX_OPEN_CALLS_PER_HOST:
            raise CallError(429, "too many open calls for this account — end one first")
        if len(self._calls) >= MAX_CALLS:
            # Retained ended calls go first (oldest ended first), so one
            # tenant's finished calls never crowd out another's live one.
            for old in sorted(
                (c for c in self._calls.values() if c.ended),
                key=lambda c: c._ended_wall if c._ended_wall is not None else 0.0,
            ):
                if len(self._calls) < MAX_CALLS:
                    break
                self._forget(old)
        if len(self._calls) >= MAX_CALLS:
            raise CallError(503, "too many open calls")
        ttl = min(max(1, int(ttl_minutes or CALL_TTL_MINUTES)), CALL_TTL_MAX_MINUTES)
        cap = int(max_participants or DEFAULT_MAX_PARTICIPANTS)
        if cap < 2 or cap > MAX_PARTICIPANT_ROLE + MAX_THERAPIST_ROLE:
            raise CallError(422, f"max_participants must be 2..{MAX_PARTICIPANT_ROLE + MAX_THERAPIST_ROLE}")
        code = new_join_code()
        while code in self._by_code:
            code = new_join_code()
        now = datetime.now(timezone.utc)
        call = Call(
            call_id=str(uuid.uuid4()),
            host_uid=host_uid,
            join_code=code,
            created_at=now.isoformat(),
            expires_at=(now + timedelta(minutes=ttl)).isoformat(),
            invitee_uid=invitee_uid,
            invitee_email=invitee_email,
            max_participants=cap,
        )
        call.add_participant(host_uid, email=host_email, display_name=display_name, role=ROLE_PARTICIPANT)
        self._calls[call.call_id] = call
        self._by_code[code] = call.call_id
        logger.info(
            "call %s created by %s (invitee=%s, ttl=%dm, max=%d)", call.call_id, host_uid,
            invitee_uid or invitee_email or "open", ttl, cap,
        )
        return call

    def get(self, call_id: str) -> Call | None:
        self.sweep()
        return self._calls.get(call_id)

    def by_code(self, join_code: str) -> Call | None:
        self.sweep()
        cid = self._by_code.get(join_code)
        return self._calls.get(cid) if cid else None

    def join(
        self,
        call: Call,
        uid: str,
        *,
        join_code: str | None = None,
        email: str | None = None,
        display_name: str | None = None,
        role: str | None = None,
    ) -> Participant:
        """Admit ``uid`` in ``role`` (participant unless told therapist):
        the named invitee needs no code; anyone else must present the join
        code. Idempotent for a member (the role stays what it was)."""
        if uid in call.participants:
            return call.add_participant(uid, email=email, display_name=display_name)
        role = self.authorize_join(call, uid, join_code=join_code, role=role)
        return call.add_participant(uid, email=email, display_name=display_name, role=role)

    def authorize_join(
        self, call: Call, uid: str, *, join_code: str | None = None, role: str | None = None,
    ) -> str:
        """The cheap checks of :meth:`join` (role, ended, the code) with no
        side effect but the wrong-code tally — so a handler can refuse a
        guess BEFORE paying for an account lookup. Returns the clean role;
        a member needs no authorization (returns its current role)."""
        member = call.participants.get(uid)
        if member is not None:
            return member.role
        role = clean_role(role)
        if call.ended:
            raise CallError(410, "call has expired" if call.end_reason == "expired" else "call has ended")
        if uid != call.invitee_uid:
            code = normalize_join_code(join_code)
            if code is None or code != call.join_code or call.code_failures >= JOIN_CODE_FAILURES_MAX:
                call.code_failures += 1
                if call.code_failures == JOIN_CODE_FAILURES_MAX:
                    logger.warning(
                        "call %s: %d wrong join codes — the code is burned", call.call_id, call.code_failures,
                    )
                raise CallError(403, "join code does not match")
        return role


registry = CallRegistry()


async def resolve_email(uid: str, resolver: Callable[[str], str | None] | None = None) -> str | None:
    """The account's email for naming, best-effort (never raises: no
    Firebase in tests/local dev → None → the slot label is used)."""
    if resolver is None:
        try:
            import main
            resolver = main.resolve_email_by_uid
        except Exception:  # noqa: BLE001
            return None
    try:
        return await asyncio.to_thread(resolver, uid)
    except Exception:  # noqa: BLE001
        return None
