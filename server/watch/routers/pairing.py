# Ported from gauge@2157433 server/pairing_api.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B8): `make_pairing_router`'s closure-factory signature is now
# `(store, full_auth_dep)` -- gauge's own param was already named `full_auth`
# (this router required non-legacy auth for `/me/pair/claim` from the start,
# via `server.auth.require_full_auth`), so this is the same B5/B6/B7 naming
# convention (`full_auth_dep`) made explicit rather than a behavior change.
# Imports repoint to `watch.auth`/`watch.models`/`watch.pairing_store`
# (Task B3/B1/B4's flat modules) instead of gauge's vendored `server.*`.
# Nothing else changes: constants, the per-account lockout math, the
# always-200 status contract, and the token-read budget are preserved
# VERBATIM per the task brief -- this is security-critical pairing
# machinery, not a "clean up while you're here" port.
"""Device-pairing API: the OAuth-device-code-style short-code flow that lets
an unclaimed watch (no identity yet) get a real, full-auth-grade account
identity without on-wrist interactive OAuth or a Wearable Data Layer relay —
see docs/superpowers/plans/2026-08-04-gauge-wave-c-couples-wrist.md's Open
Question 1 and Task 5 KDoc for the full design rationale. This server-side
companion was written against that plan's documented wire contract (the
watch-side ``PairingStart``/``PairingStatus`` models, transcribed verbatim
in gauge-watch's Task 2 brief), not re-derived.

Flow:
  1. ``POST /me/pair/start`` (no auth — the watch has no identity yet) mints
     a short human-typeable code + an opaque ``pairing_id``, TTL ~10 minutes.
  2. The watch displays the code; an ALREADY SIGNED-IN phone/web principal
     types it into a "pair a watch" screen and calls
     ``POST /me/pair/claim`` (full-auth only — see
     ``watch.auth.require_full_auth``), which mints a long-lived, opaque
     device token for the watch's account.
  3. ``GET /me/pair/status?pairing_id=`` (no auth — ``pairing_id`` itself,
     an unguessable random id, is the capability that scopes this to one
     attempt) is polled by the watch; once claimed, it hands back the raw
     device token while still inside the pairing's TTL window (see
     ``watch/pairing_store.py``'s module docstring for the bounded exposure
     window this relies on).

CONTRACT RULING (coordinator, post-plan review of the watch-side
``DevicePairingClient``): ``GET /me/pair/status`` ALWAYS returns HTTP 200
with a decodable ``{"status": ...}`` body — including for an unknown OR
expired ``pairing_id``, both of which report ``"status": "expired"``. Never
404/410 here. Reason: the watch's ``DevicePairingClient.poll()`` treats any
non-200/undecodable response as a transport failure (``null``), which is
indistinguishable from "server unreachable" — a 404/410 for a legitimately
expired code would make the watch hang on "Getting a code…" forever instead
of showing "Code expired". Non-200s on this route are reserved for real
authn/transport failures (there are none today — this route takes no auth
at all) or 5xx. ``POST /me/pair/claim`` is unaffected by this ruling: it's
called by an already-signed-in phone/web client (never the watch), which
has no such poll-and-treat-as-null constraint, so its 401/404/409 responses
are ordinary REST semantics.

Every secret at rest is hashed (``watch/pairing_store.py``'s ``hash_secret``)
— codes and tokens are never logged or stored in plaintext beyond the
single, short-lived, narrowly-scoped exception documented on
``Pairing.device_token``.

FIX ROUND 1 (review hardenings, both applied):

1. **Fetch-count cap on the plaintext token read.** ``pair_status`` used to
   return the raw ``device_token`` on every poll within the pairing's TTL —
   unbounded within that 10-minute window. Now capped at
   ``MAX_TOKEN_READS`` successful reads (via the same atomic
   ``update_pairing_atomically`` seam, so two concurrent polls can't each
   read past the cap): the read that reaches the cap still returns the
   token (preserving the watch's retry resilience for a flaky connection),
   every read after it gets ``device_token: null`` while ``status`` stays
   ``"claimed"``. Matters specifically because ``pairing_id`` travels as a
   ``GET`` query param — a proxy/access-log leak vector the wire contract
   itself can't avoid (it's locked cross-repo against the already-built
   watch client).

2. **Failed-claim-attempt counter — REPLACED in FIX ROUND 2, see below.**
   The original per-pairing version (attribute a wrong-code miss to
   whichever pairing was the SOLE currently-pending one) is gone.

FIX ROUND 2 (review finding: the FIX ROUND 1 per-pairing counter was
materially bypassable — ``POST /me/pair/start`` takes no auth and no rate
limit, so an attacker could keep one free decoy pairing perpetually
pending, permanently forcing ``len(pending) != 1`` and no-opping the
counter for a real target pairing elsewhere, at zero incremental cost.
Empirically demonstrated by FIX ROUND 1's own bystander-safety test, which
is mechanically identical to the bypass):

**Per-CALLING-ACCOUNT failed-claim-attempt counter**, tracked by
``principal.account_id`` instead of by target pairing. ``POST
/me/pair/claim`` requires ``full_auth``, so the caller's identity is
*always* known and *always* costs a real, verified account to obtain —
unlike a pairing_id, an account_id cannot be minted for free. This
sidesteps the SHA-256 attribution problem FIX ROUND 1 ran into entirely (no
need to guess which pairing a wrong code was "aimed at"), is immune to the
decoy-pairing bypass (the count doesn't depend on how many pairings happen
to be pending), and isolates bystanders even better than FIX ROUND 1's
design: a different account's claim attempts are never touched, full stop
— no ambiguity-driven no-op branch needed at all.

Every failed claim (code doesn't match anything) is recorded via
``get_failed_claim_record``/``set_failed_claim_record`` (watch/pairing_store.py)
for the CALLING account. Once ``MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT`` is
reached, ``pair_claim`` rejects further attempts from that account with
``429``, checked FIRST, before even hashing/looking up the submitted code,
so a locked-out account is rejected even with the objectively correct code.

FIX ROUND 3 (review ruling: FIX ROUND 2's PERMANENT lockout was ruled
not-ship-as-is — the anti-laundering reasoning was valid but the binary
framing missed a middle ground):

**Time-windowed lockout, measured from the LAST failure, never reset on
success.** The count now expires ``FAILED_CLAIM_WINDOW_HOURS`` after the
most recent failure (``FailedClaimRecord.last_failed_at``, injected clock —
see ``_is_account_locked``/``_record_failed_claim`` below, neither calls
``datetime.now()`` internally). This closes the account-laundering exploit
FIX ROUND 2 was defending against IDENTICALLY to a permanent lockout —
expiry never depends on a SUCCESSFUL claim, only on time elapsed since a
FAILURE, so an attacker still cannot reset their budget by claiming an
unrelated pairing they legitimately started — while giving a legitimately
locked-out user (e.g., one who mistyped a code many times) a real, bounded,
self-service recovery path instead of requiring manual/support
intervention forever.

The window SLIDES with each new failure: ``last_failed_at`` is stamped on
every recorded failure, not just the first, so a string of failures within
the window keeps pushing its expiry forward; only a full
``FAILED_CLAIM_WINDOW_HOURS`` with ZERO failures fully resets the count (to
0, via the next failure restarting it at 1 — see
``_record_failed_claim``).

Non-blocking, reviewer-accepted note: the lockout CHECK
(``get_failed_claim_record``) and the eventual increment
(``set_failed_claim_record``) are not one atomic transaction, so there is a
narrow, bounded check-then-act race at the exact threshold (two concurrent
failures could both read the same pre-increment count and both proceed).
Left as-is by explicit reviewer ruling — worst case the lockout triggers on
attempt N+1 instead of exactly N, not a materially different guarantee.
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from watch.auth import AuthDep, Principal
from watch.models import DeviceToken, FailedClaimRecord, Pairing
from watch.pairing_store import PairingStore, hash_secret

RateLimitDep = Callable[[Request], Awaitable[None]]


async def _noop_rate_limit(request: Request) -> None:
    """Default ``rate_limit_dep`` (Task H1): no limiting at all. Keeps
    ``create_watch_test_app`` and every existing ported test unaffected —
    only ``build_watch_routers()`` (server/watch/app.py) supplies the real
    per-IP limiter, applied to the two unauthenticated routes below
    (``/me/pair/start``, ``/me/pair/status``). ``/me/pair/claim`` is
    ALREADY behind ``full_auth_dep`` (an authenticated caller, unlike the
    other two) so it is deliberately not additionally rate-limited here."""
    return None

# ~10 min: long enough for a human to read a code off a watch face and type
# it into a phone, short enough to bound the raw-device-token exposure
# window documented on Pairing.device_token (watch/pairing_store.py).
PAIRING_TTL_MINUTES = 10
PAIRING_CODE_LENGTH = 6
# Excludes 0/O, 1/I/L — ambiguous on a small watch face or read aloud.
PAIRING_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
# 256 bits of entropy, base64url-encoded (secrets.token_urlsafe) — an
# opaque, unguessable, full-auth-grade bearer credential, not a human-facing
# value, so no alphabet/length constraints beyond "large enough to never
# collide and never be brute-forced."
DEVICE_TOKEN_BYTES = 32
# pairing_id itself is a capability (GET /me/pair/status takes no auth) —
# generous entropy so it can never be guessed within its TTL.
PAIRING_ID_BYTES = 24
# FIX ROUND 1 hardening: successful "claimed" status reads that still
# return the raw device_token, before it's redacted for good — see module
# docstring. A legitimate watch typically needs 1-2 polls to notice
# "claimed"; 5 leaves generous headroom for a flaky connection's retries.
MAX_TOKEN_READS = 5
# FIX ROUND 2 hardening: failed POST /me/pair/claim attempts from one
# calling account (see module docstring) before that account is locked out
# of the endpoint entirely, even with a correct code. A legitimate user
# needs at most 1-2 tries (typo correction); 15 is a generous margin above
# that while still being a small, bounded number of guesses out of the ~1
# billion possible codes.
MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT = 15
# FIX ROUND 3: the failed-claim count expires this long after the LAST
# failure (never on success) — see module docstring. Long enough that a
# legitimate user who trips the cap doesn't need to babysit a short cooldown
# to retry; short enough that a legitimately-locked-out user isn't stuck for
# more than a day without a manual/support workaround.
FAILED_CLAIM_WINDOW_HOURS = 24


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _is_account_locked(record: FailedClaimRecord | None, now: datetime) -> bool:
    """Window-expiry policy for the per-account lockout — clock INJECTED
    (``now``), never read internally, so this is deterministically testable.
    See module docstring's FIX ROUND 3 section."""
    if record is None or record.count < MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT:
        return False
    last_failed = datetime.fromisoformat(record.last_failed_at)
    return now - last_failed < timedelta(hours=FAILED_CLAIM_WINDOW_HOURS)


async def _record_failed_claim(store: PairingStore, account_id: str, now: datetime) -> None:
    """Reset-or-increment the per-account failed-claim record — clock
    INJECTED (``now``), never read internally. If the prior failure (if any)
    fell outside the window, this restarts the count at 1 (a fresh window);
    otherwise it increments and, either way, stamps ``last_failed_at = now``
    so the window keeps sliding with each new failure — see module
    docstring's FIX ROUND 3 section for why the window is anchored to the
    MOST RECENT failure, not the first.

    Plain read-then-write, not an atomic mutator — see
    ``PairingStore.set_failed_claim_record``'s docstring for the accepted,
    bounded check-then-act race this implies.
    """
    record = await store.get_failed_claim_record(account_id)
    window_expired = record is not None and (
        now - datetime.fromisoformat(record.last_failed_at) >= timedelta(hours=FAILED_CLAIM_WINDOW_HOURS)
    )
    new_count = 1 if (record is None or window_expired) else record.count + 1
    await store.set_failed_claim_record(account_id, new_count, now.isoformat())


def _mint_code() -> str:
    return "".join(secrets.choice(PAIRING_CODE_ALPHABET) for _ in range(PAIRING_CODE_LENGTH))


def _mint_device_token() -> str:
    return secrets.token_urlsafe(DEVICE_TOKEN_BYTES)


def _is_expired(p: Pairing, now: datetime) -> bool:
    return now >= datetime.fromisoformat(p.expires_at)


class PairingStartResponse(BaseModel):
    code: str
    pairing_id: str
    expires_at: str


class PairingStatusResponse(BaseModel):
    status: Literal["pending", "claimed", "expired"]
    account_id: str | None = None
    device_token: str | None = None


class PairingClaimRequest(BaseModel):
    code: str


class PairingClaimResponse(BaseModel):
    status: Literal["claimed"]
    pairing_id: str
    account_id: str


def make_pairing_router(
    store: PairingStore, full_auth_dep: AuthDep, rate_limit_dep: RateLimitDep = _noop_rate_limit,
) -> APIRouter:
    router = APIRouter()

    @router.post("/me/pair/start", response_model=PairingStartResponse)
    async def pair_start(_rl: None = Depends(rate_limit_dep)) -> PairingStartResponse:
        # No auth: an unclaimed watch has no identity yet to authenticate as
        # — there is nothing for watch.auth's resolve_principal to check.
        code = _mint_code()
        pairing_id = secrets.token_urlsafe(PAIRING_ID_BYTES)
        now = _now()
        expires_at = (now + timedelta(minutes=PAIRING_TTL_MINUTES)).isoformat()
        await store.create_pairing(Pairing(
            id=pairing_id,
            code_hash=hash_secret(code),
            status="pending",
            created_at=now.isoformat(),
            expires_at=expires_at,
        ))
        return PairingStartResponse(code=code, pairing_id=pairing_id, expires_at=expires_at)

    @router.get("/me/pair/status", response_model=PairingStatusResponse)
    async def pair_status(
        pairing_id: str, _rl: None = Depends(rate_limit_dep),
    ) -> PairingStatusResponse:
        # CONTRACT RULING (see module docstring): always 200. An unknown
        # pairing_id and an expired one are reported identically as
        # "expired" — the watch's poll loop only needs "this attempt is
        # dead, restart," not which specific reason, and NOT distinguishing
        # them also avoids leaking "did this pairing_id ever exist" to an
        # unauthenticated caller.
        pairing = await store.get_pairing(pairing_id)
        if pairing is None:
            return PairingStatusResponse(status="expired")

        now = _now()
        if _is_expired(pairing, now):
            return PairingStatusResponse(status="expired")

        if pairing.status == "pending":
            return PairingStatusResponse(status="pending")

        # status == "claimed": still inside the delivery window (we already
        # returned "expired" above otherwise). FIX ROUND 1 hardening:
        # consume one read of the plaintext token via the atomic seam — the
        # read-then-maybe-redact must happen together, or two concurrent
        # polls could each read the token past MAX_TOKEN_READS before
        # either write commits.
        token_to_return: str | None = None

        def consume_read(current: Pairing | None) -> Pairing:
            nonlocal token_to_return
            if current is None:
                raise HTTPException(status_code=404, detail="pairing vanished mid-read")
            token_to_return = current.device_token
            if current.device_token is not None:
                current.token_reads += 1
                if current.token_reads >= MAX_TOKEN_READS:
                    current.device_token = None
            return current

        updated = await store.update_pairing_atomically(pairing_id, consume_read)
        return PairingStatusResponse(
            status="claimed", account_id=updated.claimed_account_id, device_token=token_to_return,
        )

    @router.post("/me/pair/claim", response_model=PairingClaimResponse)
    async def pair_claim(
        body: PairingClaimRequest, principal: Principal = Depends(full_auth_dep)
    ) -> PairingClaimResponse:
        # full_auth_dep (not the plain auth dep): a legacy `?account=` caller
        # must never be able to mint a device token for ANY account — see
        # watch/auth.py's require_full_auth. This is the one pairing route
        # that requires identity at all (the already-signed-in phone/web
        # side redeeming a code the watch displayed). Ordinary REST error
        # semantics here (401/404/409) are unaffected by the status-poll
        # ruling above — this endpoint is never called by the watch's
        # poll-and-treat-non-200-as-null client, only by a normal signed-in
        # phone/web caller.
        account = principal.account_id
        now = _now()

        # FIX ROUND 2 hardening (window semantics added FIX ROUND 3):
        # checked FIRST, before even hashing/looking up the submitted code,
        # so a locked-out account is rejected even with the objectively
        # correct code — see module docstring for why this replaced FIX
        # ROUND 1's per-pairing attribution heuristic, and for the
        # time-windowed-vs-permanent lockout ruling.
        existing_record = await store.get_failed_claim_record(account)
        if _is_account_locked(existing_record, now):
            raise HTTPException(
                status_code=429, detail="too many failed pairing attempts on this account",
            )

        code_hash = hash_secret(body.code)
        lookup = await store.get_pairing_by_code_hash(code_hash)
        if lookup is None:
            await _record_failed_claim(store, account, now)
            raise HTTPException(status_code=404, detail="pairing code not found or already used")

        raw_token = _mint_device_token()
        token_hash = hash_secret(raw_token)

        def mutate(current: Pairing | None) -> Pairing:
            # Re-validated against a FRESH read inside the atomic seam —
            # same TOCTOU-safety pattern as groups.py's make_join_mutator:
            # `lookup` above only resolves WHICH pairing_id the code maps
            # to, never the authoritative check.
            if current is None:
                raise HTTPException(status_code=404, detail="pairing code not found or already used")
            if _is_expired(current, now):
                raise HTTPException(status_code=404, detail="pairing code not found or already used")
            if current.status == "claimed":
                raise HTTPException(status_code=409, detail="this code was already claimed")
            current.status = "claimed"
            current.claimed_account_id = account
            current.claimed_at = now.isoformat()
            current.device_token = raw_token
            current.device_token_hash = token_hash
            return current

        claimed = await store.update_pairing_atomically(lookup.id, mutate)
        # Only reached by whichever concurrent claim (if any) won the atomic
        # race above — a race loser raises out of `mutate` and never gets
        # here, so exactly one DeviceToken is ever minted per pairing.
        store.put_device_token(DeviceToken(
            token_hash=token_hash, account_id=account, created_at=now.isoformat(), pairing_id=claimed.id,
        ))
        return PairingClaimResponse(status="claimed", pairing_id=claimed.id, account_id=account)

    return router
