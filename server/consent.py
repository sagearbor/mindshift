"""Patient consent — the disclosure a patient sees BEFORE anything of theirs
reaches a therapist, and the record that says they saw it.

Why this module exists
----------------------
Everything the therapist half of MindShift does was already *authorized*: the
patient names the therapist (``PUT /therapist/link``), the patient hands out
a call's join code. Nothing, however, was ever *disclosed* — no screen told
the patient in plain words WHAT the therapist would see, and nothing was
stored to say they were told. A practising therapist using this with a real
(non-family) patient needs both: the sentence, and the record.

The shape
---------
One consent record, the same everywhere::

    {"granted_by": <uid>,          # the PATIENT's uid — never the therapist's
     "at": "2026-08-25T…+00:00",   # when they tapped
     "scope": "episodes" | "live", # what they agreed to
     "text_version": "2026-08-25"} # WHICH disclosure they were shown

``text_version`` is the point of the record: the sentence will get reworded,
and a note in a patient file must say which wording was on screen that day.
:data:`DISCLOSURES` keeps the current text per scope; old versions live in
git, which is where an auditor would look.

Where records live
------------------
* **Per patient**, on the therapist link (``link["consents"][scope]``) — the
  standing agreement. ``PUT /therapist/link`` grants ``episodes`` (naming a
  therapist with auto-share on IS that agreement, and the client shows the
  disclosure above the field); ``live`` is a separate, explicit tap
  (``POST /therapist/consent``) because being listened to AS IT HAPPENS is a
  different thing from a transcript read later.
* **Per episode**, on the recording's meta (``consent`` + ``share_origin``)
  — a snapshot of the record that was in force WHEN this session was shared,
  so revoking consent later never rewrites what the file already says, and
  the therapist's dashboard can show "shared automatically" vs "shared by
  hand" honestly.

Revoking (``POST /therapist/consent`` with ``granted: false``) drops the
record. Without an ``episodes`` record, ingest does not auto-share — see
``therapist_links.should_auto_share``.
"""

from __future__ import annotations

from datetime import datetime, timezone

# Bump when the wording below changes. A stored record keeps the version it
# was granted under; nothing rewrites an old record.
TEXT_VERSION = "2026-08-25"

SCOPE_EPISODES = "episodes"
SCOPE_LIVE = "live"
SCOPES = (SCOPE_EPISODES, SCOPE_LIVE)

DISCLOSURES = {
    SCOPE_EPISODES: (
        "Your therapist will see this session's transcript, tone and "
        "suggestions — including what you could have said. You can turn "
        "sharing off, or un-share any single session, at any time."
    ),
    SCOPE_LIVE: (
        "Your therapist can join your calls and watch the transcript, tone "
        "and coaching as it happens. Everyone on the call is asked before "
        "she can listen."
    ),
}

# How an episode came to be shared with the therapist — the dashboard banner.
ORIGIN_AUTO = "auto"        # the link's auto-share fired at ingest
ORIGIN_MANUAL = "manual"    # the patient tapped "Share with…" themselves
ORIGIN_IN_CALL = "in_call"  # the therapist was ON the call (a direct grant)

ORIGIN_LABELS = {
    ORIGIN_AUTO: "shared automatically by the patient's therapist link",
    ORIGIN_MANUAL: "shared by hand by the patient",
    ORIGIN_IN_CALL: "shared because you were on the call",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_scope(raw: object) -> str:
    """The scope, or raise ``ValueError`` — a scope is never guessed."""
    if isinstance(raw, str) and raw.strip().lower() in SCOPES:
        return raw.strip().lower()
    raise ValueError(f"scope must be one of {', '.join(SCOPES)}")


def disclosure(scope: str) -> str:
    return DISCLOSURES.get(scope, "")


def new_consent(*, granted_by: str, scope: str, at: str | None = None) -> dict:
    """One consent record (see the module docstring)."""
    return {
        "granted_by": granted_by,
        "at": at or now_iso(),
        "scope": scope,
        "text_version": TEXT_VERSION,
    }


def consents_of(link: dict | None) -> dict[str, dict]:
    """The consent records on a therapist link, defensively typed (a link
    written before this module has none)."""
    if not isinstance(link, dict):
        return {}
    stored = link.get("consents")
    if not isinstance(stored, dict):
        return {}
    return {
        scope: record
        for scope, record in stored.items()
        if scope in SCOPES and isinstance(record, dict)
    }


def consent_for(link: dict | None, scope: str) -> dict | None:
    return consents_of(link).get(scope)


def has_consent(link: dict | None, scope: str) -> bool:
    return consent_for(link, scope) is not None


def grant(link: dict, *, granted_by: str, scope: str) -> dict:
    """``link`` with ``scope`` granted (idempotent — an existing record keeps
    its original timestamp, so "consented since" stays true)."""
    consents = dict(consents_of(link))
    if scope not in consents:
        consents[scope] = new_consent(granted_by=granted_by, scope=scope)
    return {**link, "consents": consents}


def revoke(link: dict, scope: str) -> dict:
    consents = dict(consents_of(link))
    consents.pop(scope, None)
    return {**link, "consents": consents}


def view(link: dict | None) -> dict:
    """What both sides' clients read: which scopes are granted, when, and
    the exact sentence in force today (so a client never hard-codes it)."""
    consents = consents_of(link)
    return {
        "text_version": TEXT_VERSION,
        "scopes": {
            scope: {
                "granted": scope in consents,
                "at": (consents.get(scope) or {}).get("at"),
                "granted_text_version": (consents.get(scope) or {}).get("text_version"),
                "disclosure": DISCLOSURES[scope],
            }
            for scope in SCOPES
        },
    }


def episode_disclosure(*, origin: str, consent: dict | None, therapist_email: str | None) -> dict:
    """The block stamped onto an episode's meta when it is shared with a
    therapist. Pure; ``consent`` is the record that was in force (``None``
    for a manual share, where the patient's own tap is the consent)."""
    return {
        "share_origin": origin,
        "shared_with_therapist": therapist_email or None,
        "shared_at": now_iso(),
        "consent": dict(consent) if isinstance(consent, dict) else None,
    }
