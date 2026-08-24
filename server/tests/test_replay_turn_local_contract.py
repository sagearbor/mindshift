"""Cross-language contract check for the phone's ``turn_local`` events.

The mobile replay harness (apps/mobile/src/live/replay/) drives the REAL
on-device fast loop over the checked-in scene / real recordings and dumps
every ``TurnLocalEvent`` the loop would have sent, exactly as serialized
(``{"scene", "mode", "generated_by", "options", "events": [...]}``). This
test validates those dumps with the server's pydantic model — the same
model ``audio_pipeline.py`` parses on the WebSocket — so a field the
TypeScript side renames, drops, or types differently fails here, not in a
live session.

Two sources, both validated when present:

- ``server/tests/fixtures/replay/*.json`` — committed dumps (one per
  scene/mode), regenerated with
  ``cd apps/mobile && npx tsx src/live/replay/cli.ts <scene> --dump
  ../../server/tests/fixtures/replay``. Always present, so the check never
  silently passes.
- ``apps/mobile/.replay-out/*.json`` — whatever the Jest replay suites
  wrote most recently (gitignored); checked too when the directory exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from models.audio import TurnLocalEvent

_REPO = Path(__file__).resolve().parents[2]
_COMMITTED = Path(__file__).resolve().parent / "fixtures" / "replay"
_JEST_OUT = _REPO / "apps" / "mobile" / ".replay-out"


def _dump_files() -> list[Path]:
    files = sorted(_COMMITTED.glob("turn_local_*.json"))
    if _JEST_OUT.is_dir():
        files += sorted(_JEST_OUT.glob("turn_local_*.json"))
    return files


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_committed_dumps_exist_for_every_scene_and_a_silent_mode():
    names = {p.name for p in _COMMITTED.glob("turn_local_*.json")}
    for scene in ("scene_couple_escalation", "scene_family3", "scene_meeting4", "family_real", "poker6_real"):
        assert f"turn_local_{scene}_earpiece.json" in names, f"missing committed dump for {scene}"
    assert "turn_local_scene_couple_escalation_therapist.json" in names


@pytest.mark.parametrize("path", _dump_files(), ids=lambda p: p.name)
def test_every_dumped_turn_local_event_validates(path: Path):
    dump = _load(path)
    assert dump["generated_by"] == "apps/mobile/src/live/replay/sceneReplay.ts"
    assert dump["mode"] in {"earpiece", "speaker", "therapist"}
    events = dump["events"]
    assert events, f"{path.name}: no events"
    session_id = f"replay-{dump['scene']}-{dump['mode']}"
    prev_end = -1.0
    for raw in events:
        try:
            ev = TurnLocalEvent.model_validate(raw)
        except ValidationError as exc:  # pragma: no cover - the assertion message is the point
            pytest.fail(f"{path.name}: {exc}")
        # The TS side must send exactly the model's fields — no extras that
        # the server would silently drop, no missing optionals.
        assert set(raw) == set(TurnLocalEvent.model_fields), sorted(set(raw) ^ set(TurnLocalEvent.model_fields))
        assert ev.type == "turn_local"
        assert ev.session_id == session_id
        assert ev.transcript_source == "on-device"
        assert ev.tts_source == "on-device"
        assert ev.end_time >= ev.start_time >= 0
        # Turns are finalized in order.
        assert ev.start_time >= prev_end - 0.05
        prev_end = ev.end_time
        # Coaching fields travel together.
        assert (ev.suggestion is None) == (ev.suggestion_source is None)
        if ev.suggestion is not None:
            assert ev.suggestion_source == "on-device"
        # Identity honesty: is_self only with a matched person; never a
        # bare guess.
        if ev.is_self is True:
            assert ev.speaker_person_id is not None
            assert ev.speaker_match_score is not None
        if ev.speaker_person_id is None:
            assert ev.is_self is not True
        # Prosody / tone are best-effort but never fabricated zeros.
        assert ev.prosody is not None
        if ev.prosody.pitch_hz is not None:
            assert 60 <= ev.prosody.pitch_hz <= 400
        if ev.text_tone is not None:
            for v in (ev.text_tone.warmth, ev.text_tone.frustration, ev.text_tone.defensiveness):
                assert v is None or 0 <= v <= 100
        # Text-less turns (no on-device STT text) carry no suggestion.
        if not ev.text:
            assert ev.suggestion is None


def test_dumps_carry_the_options_that_shaped_them():
    for path in _COMMITTED.glob("turn_local_*.json"):
        opts = _load(path)["options"]
        assert set(opts) == {"stt_final_latency_ms", "os_latency_ms", "enroll", "speaker_id"}
        assert opts["enroll"] in {"self", "all", "none"}
        assert isinstance(opts["speaker_id"], bool)


def test_speaking_modes_coach_and_therapist_sends_the_same_turns():
    couple = _load(_COMMITTED / "turn_local_scene_couple_escalation_earpiece.json")
    therapist = _load(_COMMITTED / "turn_local_scene_couple_escalation_therapist.json")
    assert len(couple["events"]) == len(therapist["events"])
    assert any(e["suggestion"] for e in couple["events"])
    # Therapist mode still produces the on-screen suggestion; only speech is suppressed.
    assert sum(1 for e in therapist["events"] if e["suggestion"]) == sum(1 for e in couple["events"] if e["suggestion"])
    # The self voiceprint was enrolled: at least one turn is is_self=True.
    assert any(e["is_self"] is True for e in couple["events"])
