"""Regression gate for the gpt-audio-as-a-labeller bench (2026-09-20).

scripts/gpt_audio_heat.py benchmarked OpenAI's gpt-audio-mini and gpt-audio
against CONFER's ten human raters' continuous conflict ratings, using the
exact within-clip / across-clip Spearman protocol scripts/heat_map.py's
`human_conflict_agreement()` established (median per-clip Spearman = within;
Spearman of clip-level means = across — see docs/research/
2026-09-20-gpt-audio-labeller.md for the derivation and the verbatim table).

This test does NOT call the API — it loads the small aggregated summary
already committed at server/tests/fixtures/gpt_audio_bench_summary.json and
pins the measured numbers as a FLOOR (0.05 below what was actually measured),
so a future re-run (or a prompt/model change) that silently regresses
agreement fails loudly, the same pattern as
test_feature_bench.py's cross-corpus AUC pin.

SKIPPED if the fixture is absent (it never should be once committed — this
guard exists for parity with the corpus-scale tests in this file's siblings,
which skip when their multi-GB corpora aren't present).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
FIXTURE = REPO / "server/tests/fixtures/gpt_audio_bench_summary.json"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="gpt-audio bench summary not found — run scripts/gpt_audio_heat.py "
           "--stage all (needs OPENAI_API_KEY) to regenerate it",
)


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(FIXTURE.read_text())


# ----------------------------------------------------------- CONFER gate --
#
# Measured 2026-09-20 (24 CONFER clips, 5 s windows aligned to the cached
# WavLM reference series): gpt-audio-mini across-clip heat 0.798, gpt-audio
# (full) across-clip heat 0.91. Floors set 0.05 below each.

def test_gpt_audio_mini_beats_loudness_on_confer(summary):
    mini = summary["confer"]["mini"]
    ref = summary["reference"]
    assert mini["across_clip_heat"] >= 0.75, (
        f"gpt-audio-mini across-clip agreement with CONFER's human raters dropped to "
        f"{mini['across_clip_heat']} (floor 0.75, measured 0.798 on 2026-09-20)"
    )
    # The whole point: even the cheap model should not lose to raw loudness,
    # which measures essentially nothing here (across-clip -0.417).
    assert mini["across_clip_heat"] > ref["loudness_across_clip"]


def test_gpt_audio_full_beats_wavlm_across_clip_on_confer(summary):
    full = summary["confer"]["full"]
    ref = summary["reference"]
    assert full["across_clip_heat"] >= 0.86, (
        f"gpt-audio (full) across-clip agreement with CONFER's human raters dropped to "
        f"{full['across_clip_heat']} (floor 0.86, measured 0.91 on 2026-09-20)"
    )
    assert full["across_clip_heat"] > ref["wavlm_across_clip"]


def test_gpt_audio_full_within_clip_median_floor(summary):
    full = summary["confer"]["full"]
    assert full["within_clip_heat_median"] >= 0.27, (
        f"gpt-audio (full) within-clip median Spearman dropped to "
        f"{full['within_clip_heat_median']} (floor 0.27, measured 0.319)"
    )


def test_gpt_audio_mini_within_clip_median_floor(summary):
    mini = summary["confer"]["mini"]
    assert mini["within_clip_heat_median"] >= 0.22, (
        f"gpt-audio-mini within-clip median Spearman dropped to "
        f"{mini['within_clip_heat_median']} (floor 0.22, measured 0.277)"
    )


# ------------------------------------------------------ reliability gate --
#
# The full model's parse/refusal rate on CONFER (0.241) was the headline
# reliability finding of the bench — nearly 1 in 4 windows came back
# unparseable, mostly short non-JSON replies rather than usable ratings.
# Pinned as a CEILING (regression = the rate getting even worse), not a
# floor, and generous (0.35) since this is inherently noisy model behaviour
# we don't control.

def test_gpt_audio_full_error_rate_not_worse(summary):
    full = summary["confer"]["full"]
    assert full["error_rate"] <= 0.35, (
        f"gpt-audio (full) CONFER parse-failure rate rose to {full['error_rate']} "
        f"(ceiling 0.35, measured 0.241 on 2026-09-20) — see docs/research/"
        f"2026-09-20-gpt-audio-labeller.md for why this matters more than the "
        f"agreement numbers for a production labeller"
    )


def test_gpt_audio_mini_is_far_more_reliable_than_full(summary):
    """The cheap model's near-total reliability (0.9% errors, vs 24% for the
    expensive one) is itself a finding worth pinning — if this ever flips,
    the cost-projection recommendation in the doc needs revisiting."""
    mini = summary["confer"]["mini"]
    full = summary["confer"]["full"]
    assert mini["error_rate"] < full["error_rate"]
