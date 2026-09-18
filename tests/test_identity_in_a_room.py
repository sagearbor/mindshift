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
#:
#: Lowered 0.65 -> 0.60 on 2026-09-18 by this audit. The guard below did its job:
#: changing the constant failed this file, which forced the number and the
#: write-up to move together instead of the docs quietly going stale.
SHIPPED_THRESHOLD = 0.60


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
    the 2026-09-18 loosening had to keep it that way — it did."""
    other = np.array([r["score"] for r in rows if not r["is_self"]])
    assert (other >= SHIPPED_THRESHOLD).mean() < 0.01


def test_the_residual_false_accepts_are_overlap_not_confusion(rows):
    """Why loosening the bar was safe, and the distinction that made the call.

    The few impostor turns that clear the bar are not the model confusing two
    people — they are turns where the wearer was genuinely talking over someone,
    so their voice really is in that audio. On turns with no overlap at all the
    false-accept rate is zero. Independently, 91 CREMA-D speakers / 8,190 pairs
    put the highest cross-speaker score anywhere at 0.567, below this bar.
    """
    clean_impostors = [r["score"] for r in rows
                       if not r["is_self"] and r["overlap_fraction"] < 0.05]
    assert len(clean_impostors) > 300
    assert max(clean_impostors) < SHIPPED_THRESHOLD, (
        "a NON-overlapped turn of someone else's now clears the bar — that would be "
        "genuine voice confusion, and the 0.60 decision rested on it not happening"
    )


def test_but_it_misses_most_of_your_own_turns_in_a_real_room(rows):
    """The finding. Documented as a FAILING-BY-DESIGN assertion: it asserts the
    defect still exists, so that fixing it fails the test and forces the number
    and the docs to be updated together.

    The bar sits at 0.65 while the wearer's own CLEAN turns have a median of
    about 0.63 — so roughly half of them are missed by construction.
    """
    own = np.array([r["score"] for r in rows if r["is_self"]])
    found = (own >= SHIPPED_THRESHOLD).mean()
    # 0.65 -> 0.60 moved this from ~35% to ~48%. Better, and still a defect: the
    # threshold change bought back the turns sitting just under the old bar, but
    # the median clean turn is ~0.63, so half the distribution is still close to
    # the line. The known-defect framing stays until recall clears 55%.
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


def test_why_the_bar_stopped_at_060_and_not_lower(rows):
    """Records the decision, not the proposal it replaced.

    Lower bars keep buying recall — 0.55 and 0.50 find more of the wearer's own
    turns than 0.60 does. The reason to stop at 0.60 is not recall, it is the
    ceiling of "not the same voice":

        this module's own corpus (AMI, non-overlapped turns) .. below 0.60
        91 CREMA-D speakers, 8,190 pairs, 24,570 comparisons .. max 0.567
        the original calibration table's merged artifacts ..... max 0.558

    Three independent measurements agree that genuine cross-speaker similarity
    tops out around 0.56-0.57. 0.60 sits just above all of them; 0.55 does not,
    and would start accepting scores that have been observed between DIFFERENT
    people. That is the cardinal sin this matcher is built to avoid.
    """
    own = np.array([r["score"] for r in rows if r["is_self"]])
    assert (own >= 0.50).mean() > (own >= SHIPPED_THRESHOLD).mean(), (
        "a lower bar no longer buys recall — the distribution has moved and the "
        "0.60 decision should be re-derived rather than assumed"
    )
    # The headroom that made 0.60 the stopping point rather than 0.55.
    CONFUSION_CEILING = 0.567
    assert SHIPPED_THRESHOLD > CONFUSION_CEILING, (
        f"the bar ({SHIPPED_THRESHOLD}) is at or below the highest score ever observed "
        f"between two DIFFERENT speakers ({CONFUSION_CEILING})"
    )
