"""How often would the wrist buzz during a REAL conversation, and about whom?

The gate for the failure the owner actually hit. Every haptic test in this repo
asks "is this the right cue"; none asked "how often", and that blind spot let a
build ship that buzzed 36 times a minute on a poker game while ~1,250 tests
stayed green.

Runs the ported nudge chain (scripts/conversation_audit.py) over the AMI
meeting corpus — real multi-party conversation, with per-speaker headsets
giving ground truth for who was talking, so the numbers are not opinions.

SKIPPED when the corpus is absent, which includes CI: the audio is ~2 GB, lives
in gitignored tmp/, and is CC BY 4.0 rather than ours to vendor. Build it with
`python scripts/ami_corpus.py`. The dose ARITHMETIC is pinned unconditionally
elsewhere, against committed fixtures, by
apps/watch/.../PulseDoseTest.kt and ReminderDoseTest — so a logic regression
still fails in CI even though this corpus-scale check cannot run there.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "tmp/ami-corpus"
sys.path.insert(0, str(REPO / "scripts"))

pytestmark = pytest.mark.skipif(
    not (CORPUS / "manifest.json").exists(),
    reason="AMI corpus not built — run `python scripts/ami_corpus.py`",
)

#: Ceiling for the SHIPPED configuration, in buzzes per hour of ordinary
#: meeting. Deliberately a BUDGET, not a pinned value: it must survive the
#: corpus growing without being re-blessed, and its job is to catch a
#: regression, not to bless the current number.
#:
#: It is NOT a target. As of 2026-09-18 the shipped chain sits around 50/hour —
#: roughly a buzz a minute during a meeting where nobody is angry — which is
#: still too many to ship. The budget is set just above that so it fails loudly
#: if anything makes it worse, and the real number is reported by
#: `scripts/conversation_audit.py` for the record.
MAX_BUZZES_PER_HOUR = 70.0

#: The pre-2026-09-10 configuration must stay clearly worse, or the two fixes
#: (pulse train off, reminder back-off) have stopped doing anything.
MIN_IMPROVEMENT_FACTOR = 2.0


@pytest.fixture(scope="module")
def audit_rows():
    import conversation_audit as ca

    manifest = json.loads((CORPUS / "manifest.json").read_text())
    assert manifest["meetings"], "manifest is empty"
    return ca, manifest


def _total(ca, manifest, **kw):
    rows = [ca.audit(m, CORPUS, **kw) for m in manifest["meetings"]]
    minutes = sum(r["duration_min"] for r in rows)
    buzzes = sum(r["buzzes"] for r in rows)
    return rows, buzzes / (minutes / 60.0)


def test_shipped_configuration_stays_under_the_dose_budget(audit_rows):
    ca, manifest = audit_rows
    rows, per_hour = _total(ca, manifest, pulse_on=False, reminder_backoff=True)
    assert per_hour <= MAX_BUZZES_PER_HOUR, (
        f"{per_hour:.1f} buzzes/hour across {len(rows)} real meetings, over the "
        f"{MAX_BUZZES_PER_HOUR} budget. Nobody in these recordings is angry."
    )


def test_the_two_fixes_are_still_doing_their_job(audit_rows):
    ca, manifest = audit_rows
    _, shipped = _total(ca, manifest, pulse_on=False, reminder_backoff=True)
    _, before = _total(ca, manifest, pulse_on=True, reminder_backoff=False)
    assert before >= shipped * MIN_IMPROVEMENT_FACTOR, (
        f"the pre-2026-09-10 configuration ({before:.1f}/h) is no longer clearly worse than "
        f"the shipped one ({shipped:.1f}/h) — has the pulse train or the reminder back-off regressed?"
    )


def test_the_pulse_train_is_the_dominant_source_when_it_is_on(audit_rows):
    """Pins WHY the old build was unbearable, so the reason survives the fix."""
    ca, manifest = audit_rows
    rows, _ = _total(ca, manifest, pulse_on=True, reminder_backoff=False)
    pulse = sum(r["by_kind"]["pulse"] for r in rows)
    total = sum(r["buzzes"] for r in rows)
    assert pulse / total > 0.5, (
        f"the pulse train was {pulse/total:.0%} of all buzzes; if that has changed, the "
        "story recorded in docs/plans/2026-09-10-heat-rubric-and-buzz-dose.md is stale"
    )


def test_a_buzz_still_carries_no_information_about_who_was_speaking(audit_rows):
    """The finding this corpus exists to make measurable — and a guard against
    quietly believing it has been fixed.

    A single microphone gives loudness no identity, so the wrist cannot tell
    "you got loud" from "someone near you got loud". Comparing against the base
    rate (each person's share of the talking) is what makes that rigorous: in a
    four-way meeting almost any buzz lands while someone else is talking, so
    the raw percentage alone would prove nothing.

    If this ever FAILS, that is good news — it means identity reached the wrist
    — and the assertion should be inverted rather than deleted.
    """
    ca, manifest = audit_rows
    rows, _ = _total(ca, manifest, pulse_on=False, reminder_backoff=True)
    on_self = sum(w["on_own_turn"] for r in rows for w in r["per_wearer"].values())
    expected = sum(w["expected_on_own_turn"] for r in rows for w in r["per_wearer"].values())
    total = sum(w["buzzes"] for r in rows for w in r["per_wearer"].values())
    lift = (on_self - expected) / total * 100
    assert lift < 15.0, (
        f"buzzes now land on the wearer's own turns {lift:+.1f} points above chance. "
        "If identity has genuinely reached the wrist, invert this test."
    )
