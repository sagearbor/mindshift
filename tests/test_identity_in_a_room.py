"""Does the voiceprint find its owner in a real room, on one microphone?

Every nudge rests on one claim: the voice it just heard is yours. That claim had
only ever been tested on two-speaker synthetic scenes and a 30-second family
recording — neither of which has crosstalk, four people, or a real room.

Scored by scripts/identity_audit.py over AMI: enrol from the wearer's own clean
headset (what phone voice training produces), then match against the summed
single-microphone mix (what one device actually hears).

SKIPPED without the corpus or without ECAPA, which includes CI — the audio is
CC BY 4.0 and lives in gitignored tmp/, and torch/speechbrain are deliberately
out of the base requirements. Build with:

    python scripts/ami_corpus.py --keep-headsets
    pip install -r requirements-voice.txt
    python scripts/identity_audit.py --meetings 4
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "tmp/ami-corpus/identity.json"

pytestmark = pytest.mark.skipif(
    not RESULTS.exists(),
    reason="identity audit not run — see this module's docstring",
)

#: server/speaker_id.py's MATCH_THRESHOLD. Restated rather than imported so the
#: test still means something if the import path moves; the mismatch assertion
#: below catches drift between the two.
SHIPPED_THRESHOLD = 0.65


@pytest.fixture(scope="module")
def rows():
    data = json.loads(RESULTS.read_text())
    assert len(data) > 500, f"only {len(data)} scored turns — run more meetings"
    return data


def test_the_threshold_matches_the_shipped_one(rows):
    import sys
    sys.path.insert(0, str(REPO / "server"))
    import speaker_id as sid
    assert sid.MATCH_THRESHOLD == SHIPPED_THRESHOLD, (
        "speaker_id.MATCH_THRESHOLD moved; the numbers in "
        "docs/plans/2026-09-18-real-conversation-audit.md describe the old value"
    )


def test_the_voiceprint_does_separate_its_owner_from_the_room(rows):
    """The model is not the problem, and that matters: it means the miss rate
    below is a CALIBRATION defect, not a reason to replace ECAPA."""
    own = np.array([r["score"] for r in rows if r["is_self"]])
    other = np.array([r["score"] for r in rows if not r["is_self"]])
    auc = (own[:, None] > other[None, :]).mean()
    assert auc > 0.85, f"AUC {auc:.3f} — the print no longer separates its owner from the room"
    assert np.median(own) - np.median(other) > 0.3


def test_it_almost_never_mistakes_someone_else_for_you(rows):
    """The expensive failure: the wrist telling you off for what a colleague
    just did. This is the one the shipped threshold is genuinely good at, and
    any loosening has to keep it that way."""
    other = np.array([r["score"] for r in rows if not r["is_self"]])
    assert (other >= SHIPPED_THRESHOLD).mean() < 0.01


def test_but_it_misses_most_of_your_own_turns_in_a_real_room(rows):
    """The finding. Documented as a FAILING-BY-DESIGN assertion: it asserts the
    defect still exists, so that fixing it fails the test and forces the number
    and the docs to be updated together.

    The bar sits at 0.65 while the wearer's own CLEAN turns have a median of
    about 0.63 — so roughly half of them are missed by construction.
    """
    own = np.array([r["score"] for r in rows if r["is_self"]])
    found = (own >= SHIPPED_THRESHOLD).mean()
    assert found < 0.55, (
        f"self-recall is now {found:.1%} — if the threshold or the enrolment changed, "
        "update docs/plans/2026-09-18-real-conversation-audit.md and re-point this test"
    )


def test_overlapped_turns_are_unmatchable_and_should_be_excluded_not_thresholded(rows):
    """A single mic hands ECAPA a blend of two voices, and the embedding is
    worthless — not marginal, worthless. No threshold can rescue that, so the
    fix is to skip those spans."""
    messy = np.array([r["score"] for r in rows if r["is_self"] and r["overlap_fraction"] >= 0.5])
    clean = np.array([r["score"] for r in rows if r["is_self"] and r["overlap_fraction"] < 0.1])
    assert len(messy) >= 10 and len(clean) >= 50
    assert np.median(messy) < 0.3, "heavily overlapped turns used to score as noise"
    assert np.median(clean) - np.median(messy) > 0.3


def test_lowering_the_bar_would_roughly_double_recall_for_little_cost(rows):
    """The evidence behind the recommendation, pinned so it cannot quietly rot."""
    own = np.array([r["score"] for r in rows if r["is_self"]])
    other = np.array([r["score"] for r in rows if not r["is_self"]])
    at_050 = (own >= 0.50).mean()
    fa_050 = (other >= 0.50).mean()
    assert at_050 > (own >= SHIPPED_THRESHOLD).mean() * 1.6, "0.50 no longer buys most of the recall"
    assert fa_050 < 0.02, f"0.50 now costs {fa_050:.1%} false accepts — the trade has changed"
