"""The patient ↔ therapist link and its one side effect: auto-sharing.

A patient names ONE therapist account by email (``PUT /therapist/link``,
routers/therapist.py). The link is a small document the store keeps in
two places (patient side + the therapist's reverse index, see
``RecordingsStore.write_therapist_link``). It carries:

    {patient_uid, patient_email, therapist_uid, therapist_email,
     status: "pending" | "accepted", auto_share: bool,
     created_at, accepted_at}

Decisions (recorded here so nobody re-litigates them in the UI):

* The link GRANTS NOTHING by itself. Every episode the therapist can read
  is a normal per-episode share grant (``store.add_share`` — the mechanism
  ReplayScreen's "Share with…" already uses). The link only tells ingest
  whom to grant to, so revoking a single episode, listing who can see
  what, and the therapist's ``GET /sessions`` all keep working unchanged.
* ``auto_share`` defaults ON the moment the patient links. Auto-share fires
  at INGEST (a finished live session, a stored upload) while the link
  exists with ``auto_share`` on — regardless of whether the therapist has
  tapped Accept yet. The patient owns the data and chose the recipient by
  email, exactly as a manual share does; Accept is the therapist's own
  acknowledgement (it moves the patient from "wants to share with you" to
  the patient list) and Decline removes the link so nothing further is
  shared. Earlier episodes are never back-shared: "from now on" is the
  honest promise the settings row makes; older ones can still be shared
  by hand from Replay.
* Auto-share is best-effort and never fails ingest: a store hiccup is
  logged and the episode is still stored; the patient can share it by
  hand.
* CONSENT (2026-08-25, see server/consent.py). The link now also carries
  ``consents: {scope: record}`` — what the patient was TOLD and when.
  ``PUT /therapist/link`` grants the ``episodes`` scope (the client shows
  the disclosure above the email field; naming a therapist with auto-share
  on IS that agreement), and revoking it stops auto-share while leaving the
  link and the already-granted episodes alone. Links written by an older
  server carry no ``consents`` key at all and keep sharing exactly as
  before — a release must not silently cut a real patient off — while a
  link that HAS the key and lost ``episodes`` is a deliberate revoke and is
  honoured. Every auto-shared episode is stamped with the record that was
  in force (``consent.episode_disclosure``), so the therapist's dashboard
  can say "shared automatically" and a note in a patient file can name the
  disclosure version the patient saw that day.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import consent as consent_mod

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_ACCEPTED = "accepted"

NOTE_MAX_CHARS = 5000


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_link(
    *,
    patient_uid: str,
    patient_email: str | None,
    therapist_uid: str,
    therapist_email: str,
) -> dict:
    return {
        "patient_uid": patient_uid,
        "patient_email": patient_email,
        "therapist_uid": therapist_uid,
        "therapist_email": therapist_email,
        "status": STATUS_PENDING,
        "auto_share": True,
        "created_at": now_iso(),
        "accepted_at": None,
        # Creating a link IS the patient agreeing that this therapist sees
        # their sessions — the client shows consent.DISCLOSURES["episodes"]
        # above the email field, and only the patient's own request reaches
        # here. Recorded with the disclosure version they were shown. The
        # "live" scope (being observed during a call) stays a separate tap.
        # An absent "consents" key therefore always means "written by a
        # server that predates consent" — see should_auto_share.
        "consents": {
            consent_mod.SCOPE_EPISODES: consent_mod.new_consent(
                granted_by=patient_uid, scope=consent_mod.SCOPE_EPISODES,
            ),
        },
    }


def patient_view(link: dict | None) -> dict:
    """What the PATIENT's settings screen sees. Never leaks the therapist's
    uid — the client addresses the link by nothing but its own account.

    The ``consent`` block rides along even with NO link, every scope
    ungranted. That is not decoration: ``PUT /therapist/link`` records an
    ``episodes`` consent the moment the patient names someone, so the
    patient must be able to READ the sentence they are agreeing to BEFORE
    they submit — and the client must never keep its own copy of the
    wording (a client copy would drift from the ``text_version`` the record
    cites). Unlinked, this is the only place that sentence comes from."""
    if not link:
        return {"linked": False, "consent": consent_mod.view(None)}
    return {
        "linked": True,
        "therapist_email": link.get("therapist_email"),
        "status": link.get("status") or STATUS_PENDING,
        "auto_share": bool(link.get("auto_share", True)),
        "created_at": link.get("created_at"),
        "accepted_at": link.get("accepted_at"),
        # What this patient agreed to, and the exact sentence in force —
        # the client renders the server's wording, never its own copy.
        "consent": consent_mod.view(link),
    }


def therapist_view(link: dict) -> dict:
    """One row of the THERAPIST's patient list. The patient's uid is what
    accept/decline address, so it is exposed (to the linked therapist only)."""
    return {
        "patient_uid": link.get("patient_uid"),
        "patient_email": link.get("patient_email"),
        "status": link.get("status") or STATUS_PENDING,
        "auto_share": bool(link.get("auto_share", True)),
        "created_at": link.get("created_at"),
        "accepted_at": link.get("accepted_at"),
        # Which scopes this patient has consented to (the therapist must be
        # able to see that a patient revoked live observation), and when this
        # therapist last opened their sessions — the dashboard's unread mark.
        "consent_scopes": sorted(consent_mod.consents_of(link).keys()),
        "last_seen_at": link.get("therapist_last_seen_at"),
    }


def consent_ok(link: dict | None) -> bool:
    """Whether the patient has agreed to their therapist seeing episodes.

    A link with NO ``consents`` key was written by a server that predates
    consent: it keeps working (grandfathered — a release must not silently
    stop sharing for a patient already in treatment). A link that HAS the
    key without ``episodes`` is a deliberate revoke and stops sharing."""
    if not isinstance(link, dict):
        return False
    if "consents" not in link:
        return True
    return consent_mod.has_consent(link, consent_mod.SCOPE_EPISODES)


def should_auto_share(link: dict | None) -> bool:
    """Whether ingest should grant this patient's linked therapist."""
    if not link:
        return False
    if not link.get("therapist_uid"):
        return False
    if not bool(link.get("auto_share", True)):
        return False
    return consent_ok(link)


async def stamp_episode_disclosure(
    store, owner_uid: str, recording_id: str, disclosure: dict,
) -> None:
    """Record on the episode HOW it came to be shared with a therapist and
    under which consent (``consent.episode_disclosure``). Best-effort and
    additive: a store that predates the method, or a write failure, leaves
    the grant itself untouched — the share is the access, this is the
    paperwork."""
    write = getattr(store, "write_episode_disclosure", None)
    if not callable(write):
        return
    try:
        await write(owner_uid, recording_id, disclosure)
    except Exception:  # noqa: BLE001 — paperwork must never fail a share
        logger.warning(
            "Episode disclosure write failed for uid=%s rid=%s", owner_uid, recording_id,
            exc_info=True,
        )


async def auto_share_recording(store, owner_uid: str, recording_id: str) -> list[str]:
    """Grant the owner's linked therapist read access to ``recording_id`` when
    the link says so. Returns the therapist emails granted (``[]`` when no
    link / auto-share off / consent revoked / any failure). Never raises —
    see the module docstring."""
    read_link = getattr(store, "read_therapist_link", None)
    if not callable(read_link):
        return []
    try:
        link = await read_link(owner_uid)
    except Exception:  # noqa: BLE001 — a link read failure must not fail ingest
        logger.warning("Therapist link read failed for uid=%s", owner_uid, exc_info=True)
        return []
    if not should_auto_share(link):
        return []
    therapist_uid = link["therapist_uid"]
    if therapist_uid == owner_uid:
        return []
    try:
        shares = await store.add_share(
            owner_uid, recording_id,
            recipient_uid=therapist_uid,
            recipient_email=link.get("therapist_email") or "",
            owner_email=link.get("patient_email"),
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "Auto-share to therapist failed for uid=%s rid=%s", owner_uid, recording_id,
            exc_info=True,
        )
        return []
    if shares is None:
        return []
    therapist_email = link.get("therapist_email") or therapist_uid
    await stamp_episode_disclosure(store, owner_uid, recording_id, consent_mod.episode_disclosure(
        origin=consent_mod.ORIGIN_AUTO,
        consent=consent_mod.consent_for(link, consent_mod.SCOPE_EPISODES),
        therapist_email=link.get("therapist_email"),
    ))
    return [therapist_email]
