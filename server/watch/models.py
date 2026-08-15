# Ported from gauge@2157433 server/models.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
from typing import Literal
from pydantic import BaseModel, Field, model_validator


Channel = Literal["A", "B"]
VectorName = Literal["yelling", "aggressive_tone", "interrupting", "airtime", "hr_spike"]


class VectorSubscription(BaseModel):
    vector: VectorName
    sensitivity: float = 1.0
    haptics: bool = True
    channel: Channel = "A"

    @model_validator(mode="before")
    @classmethod
    def set_default_channel(cls, data):
        if isinstance(data, dict) and "channel" not in data:
            vector = data.get("vector")
            data["channel"] = "B" if vector == "hr_spike" else "A"
        return data


class VectorEvent(BaseModel):
    vector: VectorName
    level: int = Field(ge=0, le=3)
    t: float
    value: float
    detail: str = ""
    # Who this event's behavior is measured about — always SELF_PARTICIPANT_ID
    # for v1's self-coaching vectors (interrupting/airtime); the OTHER party
    # involved goes in `detail`, never here (keeps the bias guard intact:
    # Gauge measures its own wearer). Additive/defaulted LAST field so the
    # Kotlin mirror (`ignoreUnknownKeys = true`) and every existing
    # constructor call are unaffected.
    participant_id: str | None = None


class NudgeEvent(BaseModel):
    channel: Channel
    level: int = Field(ge=0, le=3)
    t: float
    vectors: list[VectorName]


# The wearer's own participant id convention: the wearer is always
# represented as this fixed id, both in live capture (ws_ingest.py) and REST
# handlers (rest_api.py, e.g. sharing consent, which is about the owner's
# recording rather than a specific other-participant).
SELF_PARTICIPANT_ID = "self"


class Participant(BaseModel):
    id: str
    role: Literal["self", "other"]
    speaker_label: str
    display_name: str | None = None
    account_id: str | None = None


class ConsentRecord(BaseModel):
    id: str
    # Participant-scoped (a LiveSession's own Participant.id) for "labeling"
    # and "sharing"; account-scoped (the consenting account's id) for
    # "mutual_visibility"; the fixed SELF_PARTICIPANT_ID constant for
    # "capture" (a Capture has no Participant of its own — this is minted by
    # the API layer, Task 15) — a documented four-way meaning, not an
    # accident. This field is the ONE source of truth for mutual-visibility
    # consent: GroupMember carries no redundant boolean.
    participant_id: str
    kind: Literal["labeling", "sharing", "mutual_visibility", "capture"]
    attested_by: str
    confirmed: bool = False
    ts: str


class LiveSession(BaseModel):
    id: str
    owner_account: str
    started_at: str
    ended_at: str | None
    status: Literal["live", "captured", "analyzed", "not_analyzed", "transcription_unavailable"]
    participants: list[Participant]
    vector_events: list[VectorEvent]
    nudge_events: list[NudgeEvent]
    series: dict[str, list[float]] = {}
    summary: str | None = None
    shared_with: list[str] = []
    consents: list[ConsentRecord] = []
    # Raw captured audio (base64), kept server-side only for Task 8's
    # post-session analysis. Never serialized to the wire.
    pcm_b64: str = Field(default="", exclude=True)


class EnrollmentBaseline(BaseModel):
    account_id: str
    rms_db: float
    f0_median: float
    updated_at: str


class Account(BaseModel):
    id: str                                     # verified token subject (Firebase uid)
    provider: Literal["google", "legacy"] = "google"
    email: str | None = None
    display_name: str | None = None
    created_at: str
    updated_at: str


class SpeakerProfile(BaseModel):
    # Field names deliberately mirror server/engine/speaker_id.new_profile()'s
    # returned dict, so SpeakerProfile(account_id=..., **new_profile(...))
    # constructs directly with no translation layer.
    account_id: str
    version: int = 1
    embedding: list[float]
    dim: int
    enroll_count: int
    model: str
    created_at: str
    updated_at: str
    sources: list[dict] = []


GroupKind = Literal["pair", "team"]


class GroupMember(BaseModel):
    account_id: str
    joined_at: str
    # No visibility flag by design: mutual-visibility consent lives ONLY in
    # Group.consents (ConsentRecord), so there is exactly one source of truth.


class GroupInvite(BaseModel):
    code: str
    email: str | None = None
    invited_by: str
    created_at: str
    accepted_by: str | None = None
    accepted_at: str | None = None


class Group(BaseModel):
    id: str
    kind: GroupKind
    name: str = ""
    created_by: str
    created_at: str
    members: list[GroupMember] = []
    invites: list[GroupInvite] = []
    consents: list[ConsentRecord] = []


class PeriodStats(BaseModel):
    episodes: int
    calm: float | None          # mean episode calm score, 1dp; None when episodes == 0
    nudges: int                 # nudge events delivered in the period
    escalations: int            # self-attributed vector events at level >= 2


class MemberStanding(BaseModel):
    account_id: str
    display_name: str | None = None
    current: PeriodStats
    prior: PeriodStats
    delta_vs_self: float | None = None   # PRIMARY: current.calm - prior.calm, 1dp
    improving: bool | None = None        # delta_vs_self > 0; None when delta is None


class GroupStanding(BaseModel):
    group_id: str
    period_days: int
    period_start: str            # ISO, inclusive
    period_end: str              # ISO, exclusive
    members: list[MemberStanding]
    both_improving: bool = False # headline win-win: every member beat their own prior period
    ahead: str | None = None     # SECONDARY: account_id with the higher current.calm; None on tie/missing


class TelemetryEvent(BaseModel):
    id: str
    device: str
    app_version: str
    level: str          # "debug" | "info" | "warn" | "error" | "crash"
    tag: str
    message: str
    stack: str | None = None
    ts: str             # device-side timestamp (ISO string as sent; opaque)
    received_at: str    # server-generated ISO-8601 UTC


class Capture(BaseModel):
    """A consent-attested retro-capture: a short clip of the wearer's own
    audio, saved on-device and uploaded on request. ``consents`` carries the
    ConsentRecord(s) attesting the wearer opted in at save time (kind
    "capture") — load-bearing product requirement, not optional metadata.
    """

    id: str
    account_id: str
    device: str | None = None
    captured_at: str                  # device-side ISO string (opaque, as sent)
    received_at: str                  # server-generated ISO-8601 UTC
    duration_s: float
    trigger: str = ""                 # free-text trigger context ("volume", "manual", ...)
    sample_rate: int = 16000
    status: Literal["awaiting_audio", "stored"] = "awaiting_audio"
    audio_uri: str | None = None      # e.g. gs://<bucket>/captures/<account>/<id>.pcm
    audio_bytes: int | None = None    # decompressed size actually stored
    upload_encoding: str | None = None  # "gzip" when the wire body was compressed; provenance only
    labels: dict = {}                 # opaque ground-truth payload written by the dashboard
    labels_updated_at: str | None = None
    consents: list[ConsentRecord] = []


# The shipped watch's unauthenticated account id (server/auth.py's legacy
# `?account=` ladder step). Defined once here — not in store.py or
# rest_api.py — because both already import from watch.models and
# models.py has no dependency on either, so this is the only spot that
# can't create an import cycle.
LEGACY_ACCOUNT_ID = "default"


class Pairing(BaseModel):
    """Ephemeral device-pairing handshake record (OAuth-device-code-style
    short-code flow — see docs/superpowers/plans/2026-08-04-gauge-wave-c-
    couples-wrist.md's Open Question 1). The human-typeable ``code`` is
    NEVER stored except as its SHA-256 hash (``code_hash``); the long-lived
    device token is likewise never stored except as its hash
    (``device_token_hash``, mirrored onto its own ``DeviceToken`` doc below).

    ``device_token`` IS held here in plaintext, but only as a deliberate,
    narrow exception: the raw token must reach the watch at least once (it
    polls ``GET /me/pair/status``), and this field is the one hop that
    carries it — see ``server/pairing_store.py``'s module docstring for why
    that exposure window is bounded to this record's own short TTL
    (``expires_at``) AND to ``token_reads`` (FIX ROUND 1 hardening: capped
    fetch count — see ``server/pairing_api.py``'s ``pair_status``), and how
    the status handler refuses to return it once either bound is hit, even
    if the record itself lingers in storage.

    FIX ROUND 2: the brute-force circuit breaker moved OFF this model
    entirely — it's now a per-CALLING-ACCOUNT counter (see
    ``FailedClaimRecord`` below), not a per-pairing one. See
    ``server/pairing_api.py``'s module docstring for why the earlier
    per-pairing ``"invalidated"`` status (FIX ROUND 1) was replaced rather
    than kept: it was defeatable at zero cost by minting a free decoy
    ``POST /me/pair/start`` pairing to keep alongside a real brute-force
    target.
    """

    id: str                                   # pairing_id
    code_hash: str
    status: Literal["pending", "claimed"] = "pending"   # "expired" is computed from expires_at, never persisted
    created_at: str
    expires_at: str
    claimed_account_id: str | None = None
    claimed_at: str | None = None
    device_token: str | None = None           # raw token; see docstring above for the bounded exposure window
    device_token_hash: str | None = None
    token_reads: int = 0                      # FIX ROUND 1: successful "claimed" status reads that returned device_token


class DeviceToken(BaseModel):
    """Long-lived, full-auth-grade credential minted by ``POST /me/pair/claim``.
    Sent by the paired watch as ``Authorization: Bearer <raw token>``;
    verified by ``server/auth.py``'s ``DeviceTokenVerifier``, which hashes
    the presented token and looks it up here by ``token_hash`` — the raw
    token itself is NEVER persisted in this record."""

    token_hash: str                            # primary lookup key (SHA-256 hex of the raw token)
    account_id: str
    created_at: str
    pairing_id: str                            # audit trail only — which pairing minted this token


class FailedClaimRecord(BaseModel):
    """Per-CALLING-ACCOUNT brute-force circuit-breaker state for
    ``POST /me/pair/claim`` (server/pairing_api.py's per-account lockout —
    see its module docstring, FIX ROUND 2 then FIX ROUND 3). Raw record
    only: whether ``count`` currently means "locked out" is a time-windowed
    policy decision (measured from ``last_failed_at`` against an injected
    clock) that lives in server/pairing_api.py, not here — mirrors how
    ``Pairing.expires_at`` is a raw field interpreted only by that module's
    ``_is_expired``.

    ``last_failed_at`` is stamped on EVERY recorded failure (not just the
    first) — FIX ROUND 3: the lockout window is measured from the MOST
    RECENT failure, so a string of failures keeps sliding the window forward
    for as long as they keep landing within it.
    """

    account_id: str
    count: int
    last_failed_at: str


class LegacyClaim(BaseModel):
    """Audit record for the one-shot legacy (LEGACY_ACCOUNT_ID) account claim
    (D2). Exactly one of these ever exists (fixed doc id LEGACY_ACCOUNT_ID);
    it is both the idempotency gate (same uid may re-claim, another uid is a
    409) and the audit log entry the spec requires."""

    account_id: str            # the uid that claimed the legacy history
    first_claimed_at: str      # ISO-8601 UTC
    last_claimed_at: str
    # Bumped one episode at a time as each move commits (rest_api.py's
    # claim_legacy) -- audit-accuracy only, never gates access. Can UNDER-
    # count by at most one episode per crash (never over-counts, never
    # fabricates a moved episode) -- see claim_legacy's Phase 2 comment.
    episodes_moved_total: int = 0
