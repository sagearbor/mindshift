"""Opt-in regression over the owner's PRIVATE real-conversation fixtures
(tmp/recordings/fixtures/<name>/, frozen by scripts/recording_replay.py).

Each fixture is re-run fully OFFLINE: Deepgram from its cached transcript,
the phone's real on-device loop re-executed (tsx + ECAPA), the server's real
coaching code on a local uvicorn in real time with every LLM reply from the
fixture's llm_cache (recorded latency replayed; a prompt that changed is a
cache MISS and an honest error, never a live call). The run must not regress
against the fixture's baseline.json on identity accuracy, first "that's you",
latency budget, moment hits, violations or errors (recreplay.fixture.compare).

Skipped cleanly when the private fixtures are absent (CI has none), or when
this machine can't run the phone loop (no tsx / ECAPA export) or the voice
deps. Each fixture takes about as long as its recording (real-time streaming).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from recreplay import fixture, phone, pipeline

# Corpus fixtures (scripts/corpus_to_inbox.py; ~1.6 h of real-time streaming
# for the default batch) re-run in full only on request; their fast integrity
# check lives in test_recording_replay_corpus_fixtures.py.
FIXTURES = [fx for fx in fixture.list_fixtures()
            if os.getenv("MINDSHIFT_CORPUS_FIXTURES_FULL") or not fixture.is_corpus_fixture(fx)]
_SKIP_REASON = None
if not FIXTURES:
    _SKIP_REASON = f"no private recording fixtures under {fixture.FIXTURES}"
elif phone.find_tsx() is None:
    _SKIP_REASON = "tsx not installed: the phone loop cannot run"
elif os.getenv("MINDSHIFT_SKIP_RECORDING_FIXTURES"):
    _SKIP_REASON = "MINDSHIFT_SKIP_RECORDING_FIXTURES set"

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


def _ecapa_present() -> bool:
    root = Path(phone.REPO_ROOT)
    for d in [root, *root.parents][:5]:
        if any((d / "server" / ".ecapa_cache").glob("ecapa_*.onnx")):
            return True
    return bool(os.getenv("MINDSHIFT_ECAPA_ONNX_PATH"))


@pytest.mark.parametrize("fx", FIXTURES, ids=[p.name for p in FIXTURES])
def test_fixture_does_not_regress(fx: Path, tmp_path: Path):
    if not _ecapa_present():
        pytest.skip("no ECAPA ONNX export: phone speaker-ID would be off, not comparable")
    inp, baseline = fixture.inputs_from_fixture(fx, tmp_path)
    if inp.profile_path is None and (baseline.get("settings") or {}).get("enroll") == "profile":
        pytest.skip("the fixture was recorded with the owner's voiceprint, which is not on this machine")
    bundle = pipeline.run(inp, fixture.options_from_baseline(baseline))
    assert bundle["phone_error"] is None, bundle["phone_error"]
    assert bundle["server_error"] is None, bundle["server_error"]
    current = fixture.metrics(bundle)
    regressions, notes = fixture.compare(baseline, current)
    print(f"{fx.name}: {current}\nnotes: {notes}")
    assert regressions == [], "\n".join(regressions + notes)
    # offline means offline: the re-run never asked a real LLM
    assert (bundle["server"]["llm"] or {}).get("misses", 0) == (bundle["server"]["llm"] or {}).get("offline_misses", 0)
