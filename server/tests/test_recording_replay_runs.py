"""Recording-replay pipeline: the phone meta / turn_local plumbing and the
real-time SERVER run against a local uvicorn serving main.app (an LLM
double here, so it needs no key, no private audio and no ECAPA)."""

from __future__ import annotations

import json

import numpy as np
import pytest

import main
from recreplay import annotation as ann
from recreplay import phone
from recreplay import server_run


WORDS = [
    {"word": "How", "start": 1.0, "end": 1.2, "speaker": 0},
    {"word": "was", "start": 1.2, "end": 1.4, "speaker": 0},
    {"word": "school?", "start": 1.4, "end": 1.8, "speaker": 0},
    {"word": "Fine.", "start": 2.4, "end": 2.8, "speaker": 1},
]
DG_TURNS = [
    {"speaker": "Speaker A", "text": "How was school?", "start_time": 1.0, "end_time": 1.8},
    {"speaker": "Speaker B", "text": "Fine.", "start_time": 2.4, "end_time": 2.8},
]
ANN = {
    "format": "mindshift-annotation/v1",
    "speakers": [{"id": "S1", "voice_description": "man"}, {"id": "S2", "voice_description": "boy"}],
    "segments": [
        {"start": 0.5, "end": 1.3, "speaker": "S1", "text": "how was school", "vocal": {"emotion": "frustrated"}},
        {"start": 1.9, "end": 2.3, "speaker": "S2", "text": "fine", "vocal": {"emotion": "tired"}},
    ],
}


def test_build_meta_from_aligned_annotation_labels_the_owner_you():
    a = ann.parse_annotation_text(json.dumps(ANN))
    al = ann.align(a, WORDS)
    meta, source = phone.build_meta("x", DG_TURNS, WORDS, wearer_label="Speaker A", wearer_ann_id="S1",
                                    alignment=al, annotation=a, phone_tone="annotation")
    assert source == "annotation"
    assert meta["self_speaker"] == "You"
    assert [(t["speaker"], t["text"], t.get("emotion_coarse")) for t in meta["turns"]] == [
        ("You", "How was school?", "angry"), ("S2", "Fine.", "sad"),
    ]
    assert meta["turns"][0]["start_time"] == 1.0          # re-timed, Deepgram's clock


def test_build_meta_without_annotation_uses_deepgram_and_no_tone():
    meta, source = phone.build_meta("x", DG_TURNS, WORDS, wearer_label="Speaker B", wearer_ann_id=None,
                                    alignment=None, annotation=None, phone_tone="annotation")
    assert source == "deepgram"
    assert [t["speaker"] for t in meta["turns"]] == ["Speaker A", "You"]
    assert all("emotion_coarse" not in t for t in meta["turns"])


def test_turn_locals_strip_the_scripted_phone_suggestion():
    out = {"sent": [{"type": "turn_local", "session_id": "replay-x", "turn_uid": "u-0", "speaker": "You",
                     "is_self": True, "text": "hi", "start_time": 1.0, "end_time": 1.8,
                     "transcript_source": "on-device", "suggestion": "keep going", "suggestion_source": "on-device",
                     "sent_at_audio_s": 2.6}]}
    evs, rel = phone.turn_locals_for_server(out, "sess-1")
    assert rel == [2.6]
    assert evs[0]["session_id"] == "sess-1"
    assert evs[0]["suggestion"] is None and evs[0]["suggestion_source"] is None
    assert "sent_at_audio_s" not in evs[0]


class _CoachLLM:
    """Coaching prompts -> fixed JSON (streamed in pieces, so partials flow)."""

    model = "fake-model"

    def __init__(self):
        self.systems = []

    def _answer(self, system):
        if "Produce ONE nudge" in system:
            return json.dumps({"nudge": "slow down", "importance": 80})
        return json.dumps({"suggestions": ["Tell me more about it."], "importance": 90})

    def complete(self, system, user, max_tokens=512, **_):
        self.systems.append(system[:30])
        return self._answer(system)

    def stream_complete(self, system, user, **_):
        self.systems.append("stream:" + system[:30])
        text = self._answer(system)
        for i in range(0, len(text), 7):
            yield text[i:i + 7]


def test_server_run_streams_real_time_and_times_every_event():
    sr = 16000
    rng = np.random.default_rng(0)
    pcm = (rng.standard_normal(sr * 4) * 800).astype("<i2")      # 4 s of noise
    turns = [
        {"type": "turn_local", "session_id": "x", "turn_uid": "t-0", "speaker": "S2", "speaker_person_id": None,
         "speaker_match_score": None, "is_self": False, "text": "I failed the quiz today.", "start_time": 0.5,
         "end_time": 1.6, "transcript_source": "on-device", "prosody": None, "text_tone": None,
         "suggestion": None, "suggestion_source": None, "tts_source": "on-device"},
        {"type": "turn_local", "session_id": "x", "turn_uid": "t-1", "speaker": "You", "speaker_person_id": "self",
         "speaker_match_score": 0.8, "is_self": True, "text": "Well you should have studied harder.",
         "start_time": 2.0, "end_time": 3.2, "transcript_source": "on-device", "prosody": None, "text_tone": None,
         "suggestion": None, "suggestion_source": None, "tts_source": "on-device"},
    ]
    llm = _CoachLLM()
    cfg = server_run.ServerConfig(speed=4.0, llm_client=llm, session_context="dinner at home",
                                  relationship="child", stop_timeout_s=30)
    run = server_run.run_server(pcm, turns, [2.0, 3.6], cfg)
    assert run.error is None, run.error
    assert run.config_ack and run.config_ack.get("type") == "config_ack"
    assert run.session_complete is not None
    kinds = [e["event"].get("type") for e in run.events]
    assert "suggestion" in kinds
    finals = [e for e in run.events if e["event"].get("type") == "suggestion" and not e["event"].get("partial")]
    assert {f["event"]["utterance_text"] for f in finals} >= {"I failed the quiz today."}
    # each turn went out at its release time on the audio clock (speed 4)
    assert [s["turn"]["turn_uid"] for s in run.sent] == ["t-0", "t-1"]
    assert run.sent[0]["at_s"] == pytest.approx(2.0 / 4, abs=0.15)
    # events are timed on the same clock and arrive after their turn went out
    for f in finals:
        sent_at = next(s["at_s"] for s in run.sent if s["turn"]["text"] == f["event"]["utterance_text"])
        assert f["at_s"] >= sent_at
    # the session context reached the coaching prompt path, and the local
    # server's seams were restored afterwards
    assert llm.systems
    assert getattr(main.app.state, "llm_client", None) is not llm
