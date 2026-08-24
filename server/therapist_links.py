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
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

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
    }


def patient_view(link: dict | None) -> dict:
    """What the PATIENT's settings screen sees. Never leaks the therapist's
    uid — the client addresses the link by nothing but its own account."""
    if not link:
        return {"linked": False}
    return {
        "linked": True,
        "therapist_email": link.get("therapist_email"),
        "status": link.get("status") or STATUS_PENDING,
        "auto_share": bool(link.get("auto_share", True)),
        "created_at": link.get("created_at"),
        "accepted_at": link.get("accepted_at"),
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
    }


def should_auto_share(link: dict | None) -> bool:
    """Whether ingest should grant this patient's linked therapist."""
    if not link:
        return False
    if not link.get("therapist_uid"):
        return False
    return bool(link.get("auto_share", True))


async def auto_share_recording(store, owner_uid: str, recording_id: str) -> list[str]:
    """Grant the owner's linked therapist read access to ``recording_id`` when
    the link says so. Returns the therapist emails granted (``[]`` when no
    link / auto-share off / any failure). Never raises — see the module
    docstring."""
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
    return [link.get("therapist_email") or therapist_uid]
