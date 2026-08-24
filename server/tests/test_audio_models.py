"""Wire-shape tests for server/models/audio.py's Foundation A additions.

These are pure pydantic tests — no WebSocket, no pipeline. audio_pipeline.py
doesn't emit or consume the new events yet (later track); what matters here
is that (a) an OLD client's view of the protocol is untouched — a
SuggestionEvent serialized before `suggestion_source` existed still parses,
and a new one still carries every old field — and (b) the new client→server
TurnLocalEvent rejects malformed reports at the door rather than letting a
typo'd source or an out-of-range tone score reach the pipeline.
"""

import json

import pytest
from pydantic import ValidationError

from models.audio import (
    SpeakerIdentityEvent,
    SuggestionEvent,
    ToneFlagEvent,
    TranscriptEvent,
    TurnLocalEvent,
    TurnProsody,
    TurnTextTone,
)

# A SuggestionEvent exactly as the pre-Foundation-A server serialized it
# (every field that existed then, nothing more). If this stops parsing, an
# old client's recorded/cached events break.
LEGACY_SUGGESTION_WIRE = {
    "type": "suggestion",
    "kind": "response",
    "session_id": "s1",
    "utterance_text": "you never listen",
    "speaker": "Speaker B",
    "suggestions": ["I hear that you feel unheard."],
    "empathy_slider": 70,
    "audio_b64": None,
    "importance": 80,
    "speak": True,
}


class TestSuggestionEventBackCompat:
    def test_legacy_wire_parses_and_defaults_to_cloud(self):
        ev = SuggestionEvent.model_validate(LEGACY_SUGGESTION_WIRE)
        assert ev.suggestion_source == "cloud"
        assert ev.kind == "response"
        assert ev.suggestions == ["I hear that you feel unheard."]

    def test_minimal_constructor_still_works(self):
        """The pipeline's existing SuggestionEvent(...) call sites pass no
        suggestion_source — the default must keep them compiling AND keep the
        wire meaning 'cloud'."""
        ev = SuggestionEvent(session_id="s1", utterance_text="t", speaker="Speaker A",
                             suggestions=[], empathy_slider=50)
        assert ev.suggestion_source == "cloud"
        assert ev.type == "suggestion"

    def test_json_round_trip_keeps_every_legacy_field(self):
        ev = SuggestionEvent.model_validate(LEGACY_SUGGESTION_WIRE)
        wire = json.loads(ev.model_dump_json())
        for key, value in LEGACY_SUGGESTION_WIRE.items():
            assert wire[key] == value, key
        assert wire["suggestion_source"] == "cloud"
        assert SuggestionEvent.model_validate(wire) == ev

    def test_on_device_source_round_trips(self):
        ev = SuggestionEvent(session_id="s1", utterance_text="t", speaker="Speaker A",
                             suggestions=["x"], empathy_slider=50, suggestion_source="on-device")
        assert SuggestionEvent.model_validate_json(ev.model_dump_json()).suggestion_source == "on-device"

    def test_transcript_event_untouched(self):
        """Sanity anchor: the sibling event's shape didn't move."""
        ev = TranscriptEvent(session_id="s1", speaker="Speaker A", text="hi", start_time=0.0, end_time=1.0)
        assert set(json.loads(ev.model_dump_json())) == {"type", "session_id", "speaker", "text", "start_time", "end_time"}


def _turn_local(**overrides) -> dict:
    base = {
        "session_id": "s1",
        "speaker": "Speaker A",
        "text": "I said I'm fine.",
        "start_time": 12.5,
        "end_time": 14.25,
        "transcript_source": "on-device",
    }
    base.update(overrides)
    return base


class TestTurnLocalEvent:
    def test_minimal_report_parses_with_nulls_everywhere_optional(self):
        ev = TurnLocalEvent.model_validate(_turn_local())
        assert ev.type == "turn_local"
        assert ev.speaker_person_id is None
        assert ev.speaker_match_score is None
        assert ev.is_self is None
        assert ev.prosody is None
        assert ev.text_tone is None
        assert ev.suggestion is None
        assert ev.suggestion_source is None
        assert ev.tts_source is None

    def test_full_report_round_trips(self):
        wire = _turn_local(
            speaker_person_id="person-42",
            speaker_match_score=0.83,
            is_self=True,
            prosody={"rms_dbfs": -18.2, "pitch_hz": 142.0, "speech_rate": 4.1},
            text_tone={"warmth": 20, "defensiveness": 75, "sarcasm": 10, "sadness": 30, "frustration": 60,
                       "label": "defensive"},
            suggestion="Try: 'I'm not fine, and I want to talk about it.'",
            suggestion_source="on-device",
            tts_source="on-device",
        )
        ev = TurnLocalEvent.model_validate(wire)
        assert ev.prosody == TurnProsody(rms_dbfs=-18.2, pitch_hz=142.0, speech_rate=4.1)
        assert ev.text_tone == TurnTextTone(warmth=20, defensiveness=75, sarcasm=10, sadness=30,
                                            frustration=60, label="defensive")
        again = TurnLocalEvent.model_validate_json(ev.model_dump_json())
        assert again == ev
        assert json.loads(again.model_dump_json())["type"] == "turn_local"

    def test_partial_prosody_keeps_missing_measurements_null(self):
        """Best-effort measurement: an unvoiced turn reports pitch_hz null,
        and that null must survive (never coerced to 0)."""
        ev = TurnLocalEvent.model_validate(_turn_local(prosody={"rms_dbfs": -30.0, "pitch_hz": None}))
        assert ev.prosody.rms_dbfs == -30.0
        assert ev.prosody.pitch_hz is None
        assert ev.prosody.speech_rate is None

    @pytest.mark.parametrize("missing", ["session_id", "speaker", "text", "start_time", "end_time", "transcript_source"])
    def test_required_fields(self, missing):
        wire = _turn_local()
        del wire[missing]
        with pytest.raises(ValidationError):
            TurnLocalEvent.model_validate(wire)

    @pytest.mark.parametrize("field,bad", [
        ("transcript_source", "server"),
        ("transcript_source", "ondevice"),
        ("suggestion_source", "server"),
        ("tts_source", "cloud"),   # TTS is "server", not "cloud" — the two vocabularies differ on purpose
    ])
    def test_source_literals_reject_typos(self, field, bad):
        with pytest.raises(ValidationError):
            TurnLocalEvent.model_validate(_turn_local(**{field: bad}))

    @pytest.mark.parametrize("field,bad", [("warmth", 101), ("frustration", -1), ("sarcasm", 250)])
    def test_text_tone_scores_are_0_to_100(self, field, bad):
        with pytest.raises(ValidationError):
            TurnLocalEvent.model_validate(_turn_local(text_tone={field: bad}))

    def test_text_tone_label_is_free_text(self):
        ev = TurnLocalEvent.model_validate(_turn_local(text_tone={"label": "wistful"}))
        assert ev.text_tone.label == "wistful"
        assert ev.text_tone.warmth is None

    def test_negative_times_rejected(self):
        with pytest.raises(ValidationError):
            TurnLocalEvent.model_validate(_turn_local(start_time=-0.1))


class TestToneFlagEvent:
    def test_round_trip_and_discriminator(self):
        ev = ToneFlagEvent(session_id="s1", speaker="Speaker B", start_time=3.0, end_time=5.5,
                           source="audio", scores={"frustration": 78.0, "warmth": 12.0},
                           label="frustrated", confidence=0.7)
        wire = json.loads(ev.model_dump_json())
        assert wire["type"] == "tone_flag"
        assert wire["scores"] == {"frustration": 78.0, "warmth": 12.0}
        assert ToneFlagEvent.model_validate(wire) == ev

    def test_scores_default_empty(self):
        ev = ToneFlagEvent(session_id="s1", speaker="Speaker A", start_time=0.0, end_time=1.0,
                           source="text", label="neutral", confidence=0.2)
        assert ev.scores == {}

    @pytest.mark.parametrize("bad", [{"source": "llm"}, {"confidence": 1.5}, {"confidence": -0.1}])
    def test_validation(self, bad):
        base = dict(session_id="s1", speaker="Speaker A", start_time=0.0, end_time=1.0,
                    source="text", label="neutral", confidence=0.5)
        base.update(bad)
        with pytest.raises(ValidationError):
            ToneFlagEvent.model_validate(base)


class TestSpeakerIdentityEvent:
    def test_known_person_round_trips(self):
        ev = SpeakerIdentityEvent(session_id="s1", speaker="Speaker A", person_id="person-7",
                                  display_name="Sam", is_self=False, score=0.91)
        wire = json.loads(ev.model_dump_json())
        assert wire["type"] == "speaker_identity"
        assert SpeakerIdentityEvent.model_validate(wire) == ev

    def test_unknown_speaker_is_nulls_not_self(self):
        ev = SpeakerIdentityEvent(session_id="s1", speaker="Speaker B", is_self=False, score=0.12)
        assert ev.person_id is None and ev.display_name is None

    def test_is_self_is_required_here(self):
        """Unlike TurnLocalEvent (the phone may not know), this is the server's
        verdict and must be a definite bool."""
        with pytest.raises(ValidationError):
            SpeakerIdentityEvent.model_validate({"session_id": "s1", "speaker": "Speaker A", "score": 0.5})


def test_type_discriminators_are_distinct():
    """The client switches on `type`; a collision would be silently wrong."""
    types = {
        TranscriptEvent(session_id="s", speaker="a", text="t", start_time=0, end_time=0).type,
        SuggestionEvent(session_id="s", utterance_text="t", speaker="a", suggestions=[], empathy_slider=0).type,
        TurnLocalEvent.model_validate(_turn_local()).type,
        ToneFlagEvent(session_id="s", speaker="a", start_time=0, end_time=0, source="text", label="l", confidence=0).type,
        SpeakerIdentityEvent(session_id="s", speaker="a", is_self=False, score=0).type,
    }
    assert types == {"transcript", "suggestion", "turn_local", "tone_flag", "speaker_identity"}
