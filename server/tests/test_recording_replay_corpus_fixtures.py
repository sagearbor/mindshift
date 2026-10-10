"""Frozen CORPUS fixtures (tmp/recordings/fixtures/<corpus>_<id>/, from
scripts/corpus_to_inbox.py + recording_replay.py): fast integrity checks that
need no network, no LLM and no real-time streaming.

* the fixture carries everything an offline re-run needs (audio, the
  ground-truth annotation, Deepgram cache, LLM cache, the held-out voiceprint);
* the notes are marked as corpus ground truth (never mistaken for the owner's);
* the voiceprint was enrolled OUTSIDE the tested window;
* re-scoring the frozen run reproduces the baseline metrics exactly.

The full offline re-run (phone loop + local server in real time, ~1.6 h for
the default batch) is ``recording_replay.py --fixtures`` or
``MINDSHIFT_CORPUS_FIXTURES_FULL=1 pytest server/tests/test_recording_replay_fixtures.py``.
Skipped cleanly when there are no corpus fixtures (CI, fresh clones).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recreplay import fixture
from recreplay import score as score_mod

CORPUS = [fx for fx in fixture.list_fixtures() if fixture.is_corpus_fixture(fx)]
pytestmark = pytest.mark.skipif(not CORPUS, reason=f"no corpus fixtures under {fixture.FIXTURES}")
IDS = [p.name for p in CORPUS]


@pytest.mark.parametrize("fx", CORPUS, ids=IDS)
def test_fixture_is_complete_for_an_offline_rerun(fx: Path):
    for f in ("audio.wav", "notes.txt", "deepgram.json", "baseline.json", "run.json", "phone.json", "phone_meta.json"):
        assert (fx / f).exists(), f"{fx.name}: missing {f}"
    gts = list((fx / "annotations").glob("*.annotation.groundtruth.json"))
    assert gts, "no ground-truth annotation"
    obj = json.loads(gts[0].read_text())
    assert obj["retime"] is False and obj["annotator"]["model"].startswith("corpus-ground-truth")
    base = json.loads((fx / "baseline.json").read_text())
    if base["settings"].get("enroll") == "profile":
        assert (fx / "voiceprint.json").exists(), "recorded with a held-out print that was not frozen"
    if base["metrics"].get("server_lines"):
        assert any((fx / "llm_cache").rglob("*")), "server lines but no LLM cache: an offline re-run would miss"


@pytest.mark.parametrize("fx", CORPUS, ids=IDS)
def test_voiceprint_was_enrolled_outside_the_window(fx: Path):
    vp = fx / "voiceprint.json"
    if not vp.exists():
        pytest.skip("no speaker truth for this corpus (heat-only item)")
    gt = json.loads(next((fx / "annotations").glob("*.annotation.groundtruth.json")).read_text())
    note = json.loads(vp.read_text())["samples"][0]["note"]
    a, b = gt["source"]["window_s"]
    assert "held-out" in note and f"{a:.0f}-{b:.0f}" in note


@pytest.mark.parametrize("fx", CORPUS, ids=IDS)
def test_rescoring_the_frozen_run_reproduces_the_baseline(fx: Path):
    bundle = json.loads((fx / "run.json").read_text())
    base = json.loads((fx / "baseline.json").read_text())
    bundle["score"] = score_mod.score(bundle, moment_window_s=float(base["settings"].get("moment_window_s") or 6.0),
                                      moment_anchor=base["settings"].get("moment_anchor") or "turn_start")
    cur = fixture.metrics(bundle)
    for k in ("moments_hits", "moments_total", "fires", "identity_phone_accuracy", "violations", "server_lines"):
        assert cur[k] == base["metrics"][k], f"{fx.name}: {k} {cur[k]} != baseline {base['metrics'][k]}"
