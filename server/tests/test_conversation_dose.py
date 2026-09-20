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

REPO = Path(__file__).resolve().parent.parent.parent
CORPUS = REPO / "tmp/ami-corpus"
CORPORA = REPO / "tmp/corpora"
HEAT_MAP = REPO / "tmp/heat-map/heat_map.json"
sys.path.insert(0, str(REPO / "scripts"))

#: Applied per-test rather than as a module-level `pytestmark`: the hold-3s
#: parity check below needs no corpus at all and must run in CI, because it is
#: the one assertion tying the SHIPPED policy to the numbers measured here.
requires_ami = pytest.mark.skipif(
    not (CORPUS / "manifest.json").exists(),
    reason="AMI corpus not built — run `python scripts/ami_corpus.py`",
)

#: REGRESSION CEILING for the shipped configuration, in buzzes per hour of
#: ordinary meeting. Not a target, and emphatically not an endorsement.
#:
#: Set from the full 3.8-hour, 12-meeting corpus. It was 95, against a measured
#: 88.8 buzzes/hour — about one every forty seconds, in meetings where nobody is
#: angry. hold-3s (2026-09-20) took the same corpus to **38.3**, so the ceiling
#: comes down with it: a regression ceiling that stays above a fix it no longer
#: describes is not a gate. An earlier version was 70, calibrated on a
#: two-meeting sample that happened to be quiet; the full corpus failed it
#: immediately. That is the gate working, and the honest response is to record
#: the real number rather than to keep a comfortable one.
#:
#: The product target is still lower — see
#: docs/plans/2026-09-18-real-conversation-audit.md. `test_the_dose_is_still_
#: too_high_to_ship` below asserts the defect is still present, so that fixing
#: it fails and forces this number and the docs to move together.
MAX_BUZZES_PER_HOUR = 45.0

#: What a coaching cue could plausibly cost without becoming wallpaper. Not
#: measured from anything — a stated product judgement, recorded so the gap is
#: explicit rather than implied.
SHIPPABLE_TARGET_PER_HOUR = 12.0

#: The pre-2026-09-10 configuration must stay clearly worse, or the two fixes
#: (pulse train off, reminder back-off) have stopped doing anything. Both arms
#: run at the SAME hold, so this isolates those two fixes rather than measuring
#: hold-3s a second time. Measured at 6.6x over the full corpus before hold-3s
#: (585/h before, 88.8/h after) and 13.5x after it (518/h before, 38.3/h after).
#: Held at 3.0 so normal corpus variation cannot flap the gate, while a real
#: regression in either fix still trips it.
MIN_IMPROVEMENT_FACTOR = 3.0

# ------------------------------------------------------------ hold-3s --
#
# The 2026-09-20 change: the ladder's first rung (+6 dB over the speaker's own
# baseline) must HOLD for three consecutive 1 s windows before the ladder may
# climb. Shipped on all three runtimes (server MINDSHIFT_HEAT_HOLD_S, phone and
# watch HEAT_HOLD_S), reference implementation `conversation_audit.sustained`.
#
# The numbers below are MEDIANS PER RECORDING, not corpus aggregates: the
# aggregate above is dominated by the loudest meetings, and the question the
# hold was chosen against is "what does a typical conversation cost the wearer".

#: Measured 2026-09-20: AMI 46.4, SBCSAE 36.1. Ceilings a couple of points
#: above, so re-encoding or a corpus refresh cannot flap the gate.
HOLD_MEDIAN_CEILING = {"AMI": 50.0, "SBCSAE": 40.0}

#: The ladder hold-3s replaced, same corpora, same chain: AMI 97.0/h and
#: SBCSAE 80.2/h per median recording. Asserted EXACTLY (to 0.1) at hold 0 and
#: hold 1, which is how "N=0 or 1 reproduces today's behaviour" stays a fact
#: rather than a claim in a comment.
PRE_HOLD_MEDIAN = {"AMI": 97.0, "SBCSAE": 80.2}

#: Where each corpus's per-recording manifest lives, and the key its entries
#: use for the audio path. AMI's manifest predates the shared heat-map shape.
_SBCSAE = CORPORA / "sbcsae"


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


@requires_ami
def test_shipped_configuration_stays_under_the_dose_budget(audit_rows):
    ca, manifest = audit_rows
    rows, per_hour = _total(ca, manifest, pulse_on=False, reminder_backoff=True)
    assert per_hour <= MAX_BUZZES_PER_HOUR, (
        f"{per_hour:.1f} buzzes/hour across {len(rows)} real meetings, over the "
        f"{MAX_BUZZES_PER_HOUR} budget. Nobody in these recordings is angry."
    )


@requires_ami
def test_the_dose_is_still_too_high_to_ship(audit_rows):
    """A known-defect assertion: it asserts the problem STILL EXISTS.

    Written this way on purpose. A number that is bad but stable is easy to
    stop seeing, and a green suite would quietly imply this was finished. When
    someone gets the dose under the target this test fails, which is the moment
    to update the target, the ceiling and the write-up together.
    """
    ca, manifest = audit_rows
    _, per_hour = _total(ca, manifest, pulse_on=False, reminder_backoff=True)
    assert per_hour > SHIPPABLE_TARGET_PER_HOUR, (
        f"dose is now {per_hour:.1f}/h, at or under the {SHIPPABLE_TARGET_PER_HOUR}/h target — "
        "good news. Update MAX_BUZZES_PER_HOUR, this test, and "
        "docs/plans/2026-09-18-real-conversation-audit.md together."
    )


@requires_ami
def test_the_two_fixes_are_still_doing_their_job(audit_rows):
    ca, manifest = audit_rows
    _, shipped = _total(ca, manifest, pulse_on=False, reminder_backoff=True)
    _, before = _total(ca, manifest, pulse_on=True, reminder_backoff=False)
    assert before >= shipped * MIN_IMPROVEMENT_FACTOR, (
        f"the pre-2026-09-10 configuration ({before:.1f}/h) is no longer clearly worse than "
        f"the shipped one ({shipped:.1f}/h) — has the pulse train or the reminder back-off regressed?"
    )


@requires_ami
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


@requires_ami
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


# ------------------------------------------------ the other direction --
#
# Everything above measures OVER-firing on calm conversation. CONFER measures
# the opposite: real televised arguments, rated heated second-by-second by ten
# annotators, where the shipped ladder almost never fires — because broadcast
# audio is level-controlled (crest factor ~13 dB vs ~23 dB for a home
# recording) and the loudness dynamics the ladder depends on are compressed
# away. A phone call through a carrier codec or an earbud with automatic gain
# has the same property. Known-defect assertion, like the dose one.

CONFER_MANIFEST = REPO / "tmp/corpora/confer/heatmap_manifest.json"


@pytest.mark.skipif(not CONFER_MANIFEST.exists(), reason="CONFER not ingested — scripts/confer_corpus.py")
def test_the_ladder_is_blind_to_level_controlled_arguments():
    import conversation_audit as ca
    import heat_map as hm

    entries = json.loads(CONFER_MANIFEST.read_text())
    heated = [e for e in entries if e.get("heated_spans")]
    assert len(heated) >= 8, "need a handful of human-rated heated clips"
    expected = hits = 0
    for e in heated:
        pcm, sr = hm.load_pcm(str(CONFER_MANIFEST.parent / e["audio"]))
        over = ca.db_over_baseline(ca.windows_dbfs(pcm, sr))
        # The SHIPPED chain, hold-3s included — the point of this case is what
        # the wearer would actually feel on level-controlled audio.
        felt = ca.replay(over, False, True, gate=ca.sustained(over, 6.0, ca.HEAT_HOLD_S))
        m = hm.nudge_metrics(felt, e["heated_spans"], e["duration_s"])
        expected += m["nudge_expected"]
        hits += m["nudge_hits"]
    assert expected >= 20
    caught = hits / expected
    assert caught < 0.25, (
        f"the loudness ladder now catches {caught:.0%} of human-rated heated spans on "
        "level-controlled audio — if a detector that does not depend on loudness dynamics "
        "has shipped, invert this test and update docs/plans/2026-09-19-heat-map-rounds.md"
    )


# ---------------------------------------- hold-3s, measured per recording --
#
# The gate for the 2026-09-20 change. Everything above is the AMI aggregate;
# these run the SAME ported chain over every recording in both real-conversation
# corpora and take the MEDIAN, which is the number the hold was chosen against:
# what an ordinary conversation costs the wearer, not what the loudest meeting
# in the set does.
#
# ~5 s for 32 hours of audio (the WAVs are 16 kHz mono and the chain is pure
# numpy), so this is a normal-speed test wherever the corpora exist. Where they
# do not — CI, a fresh clone — `test_the_heat_map_agrees_with_the_replay` pins
# the same two medians from the committed heat-map run, and
# `test_the_shipped_policy_reproduces_the_measured_gate` needs no files at all.


def _hold_s() -> int:
    import conversation_audit as ca

    return ca.HEAT_HOLD_S


def _corpus_entries() -> dict:
    """``{corpus: [{audio, duration_s}]}`` for the two real-conversation
    corpora, omitting either one that is not on disk."""
    out: dict[str, list[dict]] = {}
    if (CORPUS / "manifest.json").exists():
        manifest = json.loads((CORPUS / "manifest.json").read_text())
        out["AMI"] = [
            {"audio": str(CORPUS / m["mix"]), "duration_s": m["duration_s"]}
            for m in manifest["meetings"]
        ]
    sb = _SBCSAE / "heatmap_manifest.json"
    if sb.exists():
        out["SBCSAE"] = [
            {"audio": str(_SBCSAE / e["audio"]), "duration_s": e["duration_s"]}
            for e in json.loads(sb.read_text())
        ]
    return out


def _median_dose_per_hour(entries: list[dict], hold_s: int) -> float:
    """Median buzzes/hour across recordings, through the shipped chain at the
    given hold. ``hold_s`` 0 or 1 is the pre-2026-09-20 ladder."""
    import statistics

    import conversation_audit as ca
    import heat_map as hm

    doses = []
    for e in entries:
        pcm, sr = hm.load_pcm(e["audio"])
        over = ca.db_over_baseline(ca.windows_dbfs(pcm, sr))
        felt = ca.replay(over, False, True, gate=ca.sustained(over, 6.0, hold_s))
        doses.append(len(felt) / (e["duration_s"] / 3600.0))
    return statistics.median(doses)


_CORPORA_REASON = (
    "neither real-conversation corpus is on disk — build them with "
    "`python scripts/ami_corpus.py` / `python scripts/sbcsae_corpus.py`"
)
requires_corpora = pytest.mark.skipif(not _corpus_entries(), reason=_CORPORA_REASON)


@requires_corpora
def test_hold3_roughly_halves_the_dose_on_real_conversation():
    """The measurement hold-3s was chosen on, re-run from the audio.

    AMI is four people in an office meeting; SBCSAE is Americans talking to
    each other at home, at work and on the phone. Nobody in either is angry,
    and the ladder this replaced buzzed about once every forty seconds.
    """
    for corpus, entries in _corpus_entries().items():
        got = _median_dose_per_hour(entries, _hold_s())
        assert got <= HOLD_MEDIAN_CEILING[corpus], (
            f"{corpus}: median {got:.1f} buzzes/hour with the first rung holding "
            f"{_hold_s()} s, over the {HOLD_MEDIAN_CEILING[corpus]} ceiling across "
            f"{len(entries)} recordings"
        )
        assert got < PRE_HOLD_MEDIAN[corpus] * 0.75, (
            f"{corpus}: hold-{_hold_s()}s only took the median from "
            f"{PRE_HOLD_MEDIAN[corpus]}/h to {got:.1f}/h. It was adopted for roughly halving "
            "the dose; if it no longer does, the reason to carry it is gone."
        )


@requires_corpora
def test_a_hold_of_zero_or_one_reproduces_the_ladder_it_replaced():
    """The escape hatch has to be exact, not approximate: it is what every
    pre-2026-09-20 expectation in this repo is pinned with."""
    for corpus, entries in _corpus_entries().items():
        at_zero = _median_dose_per_hour(entries, 0)
        at_one = _median_dose_per_hour(entries, 1)
        assert round(at_zero, 6) == round(at_one, 6), (
            f"{corpus}: hold 0 ({at_zero:.1f}/h) and hold 1 ({at_one:.1f}/h) must be the "
            "same ladder — one qualifying window is enough in both"
        )
        assert abs(at_zero - PRE_HOLD_MEDIAN[corpus]) < 0.1, (
            f"{corpus}: the pre-hold ladder now measures {at_zero:.1f}/h, not the "
            f"{PRE_HOLD_MEDIAN[corpus]}/h hold-3s was compared against. The corpus or the "
            "chain has moved; re-measure BOTH numbers before touching the ceiling."
        )


@pytest.mark.skipif(
    not HEAT_MAP.exists(),
    reason="no tmp/heat-map/heat_map.json — run `python scripts/heat_map.py`",
)
def test_the_heat_map_agrees_with_the_replay():
    """The same two medians, read off the committed heat-map run instead of
    recomputed from audio — so the gate still has teeth on a machine that holds
    the heat map but not the 32 GB of WAVs, and so heat_map.py and this file
    cannot drift apart. They are the two places the owner reads these numbers.
    """
    import statistics

    rows = json.loads(HEAT_MAP.read_text())
    checked = 0
    for corpus, ceiling in HOLD_MEDIAN_CEILING.items():
        held = [r["dose_sustained3s"] for r in rows if r["corpus"] == corpus and "dose_sustained3s" in r]
        before = [r["dose_per_hour"] for r in rows if r["corpus"] == corpus and r.get("dose_per_hour") is not None]
        if not held or not before:
            continue
        checked += 1
        assert statistics.median(held) <= ceiling, (
            f"{corpus}: the heat map's hold-3s median is {statistics.median(held):.1f}/h, over "
            f"the {ceiling} ceiling — re-run scripts/heat_map.py and re-measure before moving it"
        )
        assert abs(statistics.median(before) - PRE_HOLD_MEDIAN[corpus]) < 0.5, (
            f"{corpus}: the heat map's pre-hold median is {statistics.median(before):.1f}/h, not "
            f"{PRE_HOLD_MEDIAN[corpus]}/h — the heat map is stale or the chain moved"
        )
    if not checked:
        pytest.skip("heat_map.json has no AMI/SBCSAE rows with dose columns")


def test_the_shipped_policy_reproduces_the_measured_gate():
    """THE link between this file and the product, and the only case here that
    needs no files at all.

    Everything above measures ``conversation_audit.py``'s port. This asserts
    that the ``NudgePolicy`` that actually ships — fed one 1 s window at a time,
    exactly as ``watch/routers/ws.py`` feeds it — produces the identical
    escalation stream over the same windows, at every hold. Without it the dose
    numbers would be a statement about a script in scripts/, not about what the
    wearer feels.
    """
    import numpy as np

    import conversation_audit as ca
    from nudge_policy import NudgePolicy
    from watch.models import VectorEvent, VectorSubscription

    # A deterministic stand-in for a conversation: mostly near baseline, with
    # bursts of raised voice of every length from one window upward — precisely
    # what the hold has to discriminate between.
    rng = np.random.default_rng(20260920)
    over = rng.normal(0.0, 2.5, 600)
    at = 10
    for burst in (1, 2, 3, 1, 5, 2, 8, 1, 4, 3, 1, 12, 2, 6):
        over[at:at + burst] = rng.uniform(6.5, 18.0, burst)
        at += burst + int(rng.integers(3, 40))

    for hold_s in (0, 1, 3, 5):
        felt = ca.replay(over, False, True, gate=ca.sustained(over, 6.0, hold_s))
        want = [(t, level) for t, kind, level in felt if kind == "escalation"]

        policy = NudgePolicy(
            [VectorSubscription(vector="yelling")],
            cooldown_s=ca.COOLDOWN_S,
            channels=("A",),
            hold_s=float(hold_s),
        )
        got = []
        for i, o in enumerate(over):
            t = i * ca.WINDOW_S
            level = ca.level_for(o)
            # push_pcm emits nothing below the first rung, so neither does this.
            events = [VectorEvent(vector="yelling", level=level, t=t, value=float(o))] if level else []
            for n in policy.on_events(events, t):
                if n.vectors:                     # a cooldown decay carries none
                    got.append((t, n.level))
        assert got == want, f"hold {hold_s}s: the shipped policy and the replay disagree"

    # ...and the gate is doing something on this signal, so the equality above
    # is not four copies of the same trivially-empty list.
    at_three = len(ca.replay(over, False, True, gate=ca.sustained(over, 6.0, 3)))
    at_one = len(ca.replay(over, False, True, gate=ca.sustained(over, 6.0, 1)))
    assert 0 < at_three < at_one
