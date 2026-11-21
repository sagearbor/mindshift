"""Quiet coach quality gate (plan 2026-11-05, Step 1).

Pins what the SHIPPED quiet coach does on the 31 split replay fixtures:

* deterministic ($0): how many turns per fixture pass the trigger gate, and
  that the LLM call rate stays <= 15 calls/h;
* measured (needs the saved real-LLM + judge run, tmp/nudge-eval/
  quiet-baseline-20261106.json from scripts/recreplay/quiet_coach_live_eval.py):
  spoken lines per fixture, leaks == 0, calls/h, and the baseline U2/h the
  prompt-v3 work is measured against.

Both parts skip when the recordings / the saved run are not on this machine
(like the AMI tests); they never fabricate numbers.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from recreplay.quiet_sim import simulate

WT = Path(__file__).resolve().parents[2]


def _main_repo() -> Path:
    try:
        out = subprocess.run(
            ["git", "-C", str(WT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return Path(out).parent if out else WT
    except Exception:  # noqa: BLE001
        return WT


def _recordings() -> Path | None:
    for c in (os.environ.get("MINDSHIFT_RECORDINGS"), WT / "tmp" / "recordings", _main_repo() / "tmp" / "recordings"):
        if c and (Path(c) / "landscape" / "splits.json").exists():
            return Path(c)
    return None


REC = _recordings()
BASELINE = (WT / "tmp" / "nudge-eval" / "quiet-baseline-20261106.json")
if not BASELINE.exists():
    BASELINE = _main_repo() / "tmp" / "nudge-eval" / "quiet-baseline-20261106.json"

# Turns per fixture that pass skip_reason + coach_trigger (server-side triggers:
# heat lexicon, wearer floor >= 45 s, direct question). Fixtures not listed: 0.
TRIGGERED_TURNS = {
    "ami_TS3004a": 1, "sbcsae_SBC033": 1, "confer_20111031_seq1": 1, "confer_20111111_seq2": 1,
    "confer_20111212_seq11": 1, "confer_20120514_seq1": 1, "confer_20111003_seq10": 1,
    "chime6_S02_t975": 2, "chime6_S09_t3975": 1, "confer_20111031_seq2": 1, "confer_20111212_seq12": 1,
    "confer_20120123_seq2": 1, "confer_20120123_seq4": 1, "sbcsae_SBC035": 1, "sbcsae_SBC042": 2,
}
# Lines the production speak gate voiced (importance >= 80) with the real Haiku
# calls of the saved run. Everything else was below the speak threshold.
SPOKEN_LINES = {"sbcsae_SBC035": 1, "sbcsae_SBC042": 1}
SESSIONS, HOURS = 30, 1.5712


@pytest.fixture(autouse=True)
def _production_gate_env(monkeypatch):
    # server/conftest.py turns the turn-skip and the trigger gate OFF for the
    # other tests; this gate measures the production settings (all default ON).
    monkeypatch.setenv("MINDSHIFT_TURN_SKIP", "1")
    for k in ("MINDSHIFT_TRIGGER_GATE", "MINDSHIFT_TRIGGER_HEAT", "MINDSHIFT_TRIGGER_FLOOR",
              "MINDSHIFT_TRIGGER_QUESTION", "MINDSHIFT_TRIGGER_LIBRARY", "MINDSHIFT_TRIGGER_HEAT_MIN",
              "MINDSHIFT_TRIGGER_FLOOR_S"):
        monkeypatch.delenv(k, raising=False)


def _runs():
    splits = json.loads((REC / "landscape" / "splits.json").read_text())
    for split in ("dev", "held_out"):
        for it in splits[split]:
            h1 = REC / "ot" / "X" / "X-H1" / it / "run.json"
            h0 = REC / "work" / it / "run.json"
            p = h1 if (split == "held_out" and h1.exists()) else h0
            if p.exists():
                b = json.loads(p.read_text())
                if b.get("phone"):
                    yield it, b


@pytest.mark.skipif(REC is None, reason="replay recordings not on this machine")
def test_trigger_gate_call_rate_and_per_fixture_counts():
    counts, hours, sessions = {}, 0.0, 0
    for it, b in _runs():
        sessions += 1
        hours += float(b["audio"]["duration_s"]) / 3600
        n = sum(1 for r in simulate(b["phone"]["sent"], {}) if r["triggers"])
        if n:
            counts[it] = n
    assert counts == TRIGGERED_TURNS
    assert sessions == SESSIONS
    assert hours == pytest.approx(HOURS, abs=0.001)
    calls_per_h = sum(counts.values()) / hours
    assert calls_per_h <= 15, calls_per_h          # plan: <= 15 cloud calls/h
    assert calls_per_h == pytest.approx(10.8, abs=0.1)


@pytest.mark.skipif(not BASELINE.exists(), reason="saved real-LLM baseline run not on this machine")
def test_saved_baseline_pins_spoken_lines_leaks_and_calls():
    run = json.loads(BASELINE.read_text())["all"]
    spoken = {k: v for k, v in run["spoken_per_item"].items() if v}
    assert spoken == SPOKEN_LINES
    assert run["label_both"]["leaks_n"] == 0          # no raw labels / unseen names in spoken lines
    assert run["llm_calls_per_h"] <= 15
    assert run["llm_calls"] == sum(TRIGGERED_TURNS.values())
    assert run["label_both"]["n_spoken"] == sum(SPOKEN_LINES.values())
    # Known state of the shipped coach (judged by claude-sonnet-5-5 + gpt-5-mini,
    # both must say helpful): both spoken lines were judged unhelpful. This pins
    # the measured baseline so a change has to beat it, not drift past it.
    assert run["label_both"]["helpful"] == 0
    assert run["label_both"]["U_per_h"] == pytest.approx(-2 * 2 / HOURS, abs=0.05)
