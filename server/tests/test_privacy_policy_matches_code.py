"""Cheap tripwire: the privacy policy must at least NAME the things the code
does that Play's Data safety pack (docs/play/play-answers-mindshift.yaml,
docs/play/play-answers-mindshift-wear.yaml) declares.

docs/play/README.md's rule is that the privacy policy and the Data safety
matrix must say the same thing, and that the policy is the half that rots
first because nothing tests it (2026-09-20 found exactly that: the page still
said a live session "keeps no audio at all" months after DEFAULT_KEEP_AUDIO
flipped to true, and it never mentioned heart rate at all).

This test does not — cannot — check that the policy's prose is *accurate*;
that still has to be verified against the code by hand, in the same sitting
that changes the pack. What it CAN check cheaply, forever, is that the words
naming each declared data type or processor never silently disappear from the
page again. If this test starts failing, either the policy lost a section it
still needs, or the code genuinely stopped doing the thing — and in the
latter case, the pack's `changed_<date>:` marker and this test should be
updated in the same commit.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVACY_HTML = REPO_ROOT / "apps" / "mobile" / "public" / "privacy" / "index.html"
DELETE_ACCOUNT_HTML = REPO_ROOT / "apps" / "mobile" / "public" / "delete-account" / "index.html"


def _text(path: Path) -> str:
    assert path.is_file(), f"expected privacy page at {path}"
    return path.read_text(encoding="utf-8")


def test_privacy_policy_file_exists() -> None:
    _text(PRIVACY_HTML)


def test_privacy_policy_mentions_heart_rate() -> None:
    body = _text(PRIVACY_HTML).lower()
    assert "heart rate" in body, (
        "The privacy policy must name heart rate (Wear OS Health info, "
        "collected per docs/play/play-answers-mindshift.yaml data_safety."
        "data_types['Health and fitness -> Health info'])."
    )


def test_privacy_policy_mentions_wear_os() -> None:
    body = _text(PRIVACY_HTML).lower()
    assert "wear os" in body, (
        "The privacy policy must name the Wear OS companion app "
        "(com.sagearbor.gauge.wear) — it has no privacy policy of its own, "
        "per docs/play/README.md."
    )


def test_privacy_policy_mentions_tone_inference() -> None:
    body = _text(PRIVACY_HTML).lower()
    assert "arousal" in body or "tone" in body, (
        "The privacy policy must name the server-side tone/arousal-valence "
        "inference (server/watch/heat_judge.py, server/tone_id.py), declared "
        "under Personal info -> Other info in the Play pack."
    )


def test_privacy_policy_mentions_deepgram() -> None:
    body = _text(PRIVACY_HTML)
    assert "Deepgram" in body, (
        "Deepgram (speech-to-text) is a named third-party processor in "
        "docs/play/play-answers-mindshift.yaml's third_parties_to_name_in_the_"
        "privacy_policy list and must stay named here word for word."
    )


def test_privacy_policy_last_updated_line_present() -> None:
    body = _text(PRIVACY_HTML)
    assert re.search(r"Last updated:\s*\d", body), (
        "The policy must carry a dated 'Last updated:' line so a stale page "
        "is at least detectable by eye."
    )


def test_delete_account_page_mentions_heart_rate_and_audio() -> None:
    # The delete-account page lists data types it erases; if it lists any at
    # all, heart rate and audio (both now real, deletable data types) must be
    # among them so the two legal pages do not disagree with each other.
    body = _text(DELETE_ACCOUNT_HTML).lower()
    assert "heart" in body, "delete-account page must mention heart-rate deletion"
    assert "audio" in body, "delete-account page must mention audio deletion"
