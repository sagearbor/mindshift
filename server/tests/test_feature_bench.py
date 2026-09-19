"""Which signal tells anger from happiness — pinned, so the answer cannot quietly rot.

Runs scripts/feature_bench.py's evaluation over the cached feature bank
(tmp/feature-bank/*.parquet, gitignored; rebuild with scripts/feature_bank.py)
and asserts the ORDERING the 2026-09-20 bench established, cross-corpus:

    WavLM arousal/valence/dominance  >  eGeMAPS  >  loudness over own baseline

on the angry-vs-HAPPY question — the failure the shipped ladder has. The
numbers themselves live in docs/plans/2026-09-19-heat-map-rounds.md; this test
guards the ranking with a margin, not the decimals.

Skipped when the bank is absent, which includes CI (the corpora are CC BY-NC /
ODbL and live in tmp/).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
BANK = REPO / "tmp/feature-bank"
sys.path.insert(0, str(REPO / "scripts"))

pytestmark = pytest.mark.skipif(
    not all((BANK / f"{g}.parquet").exists() for g in ("index", "prosody", "egemaps", "tone")),
    reason="feature bank not built — scripts/feature_bank.py --group prosody|egemaps|tone",
)


@pytest.fixture(scope="module")
def bench():
    import feature_bench as fb
    df, groups = fb.load()
    df = df.dropna(subset=groups["tone"])          # apples-to-apples on the WavLM subset
    return fb, df, groups


def _xc_happy(fb, df, cols) -> float:
    r = fb.evaluate(df, cols, gbm=False)
    return r["cremad->ravdess_auc_happy"]


def test_the_tone_models_three_numbers_beat_loudness_cross_corpus(bench):
    fb, df, groups = bench
    loud = _xc_happy(fb, df, ["prosody_db_over_own_neutral"])
    tone = _xc_happy(fb, df, groups["tone"])
    assert tone - loud > 0.10, f"WavLM dims {tone:.3f} vs loudness {loud:.3f} — the 2026-09-20 gap was 0.16"


def test_egemaps_beats_loudness_and_loses_to_the_tone_model(bench):
    fb, df, groups = bench
    loud = _xc_happy(fb, df, ["prosody_db_over_own_neutral"])
    ege = _xc_happy(fb, df, groups["egemaps"])
    tone = _xc_happy(fb, df, groups["tone"])
    assert ege > loud + 0.05, f"eGeMAPS {ege:.3f} should clear loudness {loud:.3f}"
    assert tone > ege, f"three dimensional numbers ({tone:.3f}) should still beat 88 hand-crafted ones ({ege:.3f})"


def test_our_own_prosody_features_hurt_egemaps_cross_corpus(bench):
    """The recording-setup lesson from activation v1, again: adding features
    that encode the microphone makes generalisation WORSE."""
    fb, df, groups = bench
    ege = _xc_happy(fb, df, groups["egemaps"])
    both = _xc_happy(fb, df, groups["egemaps"] + groups["prosody"])
    assert both < ege, f"eGeMAPS+prosody {both:.3f} vs eGeMAPS alone {ege:.3f} — if adding them now helps, the features changed"
