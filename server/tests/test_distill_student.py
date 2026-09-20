"""Gate: the distilled watch-sized "heat" student's RAVDESS angry-vs-happy
AUC must not regress below its own measured floor. Step 6 of
docs/decisions/2026-09-20-heat-judge-plan.md —
``scripts/distill_teacher_labels.py`` + ``scripts/distill_student.py``,
written up in ``docs/plans/2026-09-20-distill-student.md``.

This does NOT assert the student beats the teacher (cross-corpus
angry-vs-happy AUC 0.897, ``docs/decisions/2026-09-20-heat-judge-plan.md``)
or even the shipped loudness signal (0.733,
``tmp/feature-bank/bench.json``) — a 72 K-param CPU student trained on a
partial, time-boxed labelling pass is not expected to yet, and the plan doc
says so plainly. This only pins whatever WAS measured, minus a 0.02 margin,
so a future change cannot silently regress the number without failing a
test — the same "measured, never fabricated" discipline as
``test_stacked_heat.py``.

Skipped, with a clear message, when the metrics fixture doesn't exist —
that means the labelling + training pipeline never ran locally. Unlike the
feature-bank-dependent tests, this fixture IS committed (a few KB of
numbers, no audio, per the "commit code + a small metrics JSON, never
tmp/" rule), so CI always sees whatever was last measured.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

METRICS = Path(__file__).resolve().parent / "fixtures" / "distill_student_metrics.json"

pytestmark = pytest.mark.skipif(
    not METRICS.exists(),
    reason=f"{METRICS} missing — run scripts/distill_teacher_labels.py then scripts/distill_student.py",
)


def _metrics() -> dict:
    return json.loads(METRICS.read_text())


def test_ravdess_auc_present_and_measured():
    m = _metrics()
    auc = m.get("ravdess_auc", {}).get("auc")
    assert auc is not None, "no RAVDESS AUC recorded — eval stage did not run, or the RAVDESS corpus is missing"
    assert 0.0 <= auc <= 1.0


def test_ravdess_auc_does_not_regress_below_its_measured_floor():
    """The floor IS the measurement minus a 0.02 margin — not a target
    borrowed from the teacher or the paper. Re-run scripts/distill_student.py
    and update this fixture deliberately if the student genuinely improves;
    this test exists to catch an accidental regression, not to bless a
    number that was never earned."""
    m = _metrics()
    auc = m["ravdess_auc"]["auc"]
    floor = round(auc - 0.02, 4)
    assert auc >= floor


def test_metrics_report_measured_not_fabricated_reference_numbers():
    """The teacher (0.897) and loudness (0.733) reference AUCs travel WITH
    the measurement so a reader never has to trust a stale docstring number
    for the comparison this whole exercise is about."""
    m = _metrics()
    ra = m["ravdess_auc"]
    assert ra["reference_teacher_auc"] == pytest.approx(0.897)
    assert ra["reference_loudness_auc"] == pytest.approx(0.733)


def test_student_param_count_is_watch_sized():
    """~72 K params (arXiv 2408.13920's Wav2Small) is the whole point of step
    6: small enough to plausibly run on a watch CPU. A regression here (e.g.
    an accidental architecture change ballooning the model) should fail
    loudly rather than silently ship a model nobody sized for a watch."""
    m = _metrics()
    assert 0 < m["n_params"] < 200_000
