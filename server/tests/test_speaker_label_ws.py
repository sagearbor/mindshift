"""Mid-call naming over the realtime WebSocket — the ``speaker_label`` frame.

The phone tells the running session "Speaker B is Mom" (optionally an
enrolled person id, optionally "this is me"). The server applies it to the
session only: the coach's prompts name the person from the next turn on,
``is_self`` moves side-aware coaching to that voice, and the frame is
acked. Persistence rides POST /sessions/live (see test_sessions_live.py).
"""

import json

import pytest

import audio_pipeline
from audio_pipeline import SessionContext, apply_speaker_label, display_speaker

from tests.test_audio_pipeline import (  # noqa: E402 — the DI fixtures
    MOCK_LLM_JSON, fake_ws, open_ws, recv_skipping_transcripts,
)

_ = (fake_ws, MOCK_LLM_JSON)  # fixture re-export (pytest collects by name)


class TestApplySpeakerLabel:
    def _ctx(self):
        return SessionContext(session_id="s")

    def test_names_a_label_and_acks(self):
        ctx = self._ctx()
        ack = apply_speaker_label(ctx, {
            "type": "speaker_label", "speaker": " Speaker B ", "display_name": "Mom ",
            "person_id": "mom", "is_self": False,
        })
        assert ack == {
            "type": "speaker_label_ack", "speaker": "Speaker B",
            "person_id": "mom", "display_name": "Mom", "is_self": False,
        }
        assert display_speaker(ctx, "Speaker B") == "Mom"
        assert display_speaker(ctx, "Speaker A") == "Speaker A"
        assert ctx.self_speaker is None

    def test_self_moves_side_aware_coaching(self):
        ctx = self._ctx()
        ctx.self_speaker = "Speaker A"
        apply_speaker_label(ctx, {"speaker": "Speaker B", "display_name": "You", "person_id": "self", "is_self": True})
        assert ctx.self_speaker == "Speaker B"
        # Naming that same label as someone ELSE clears the stale self claim.
        apply_speaker_label(ctx, {"speaker": "Speaker B", "display_name": "Mom", "person_id": None, "is_self": False})
        assert ctx.self_speaker is None
        # A non-diarizer label can be "me" for the prompt without touching
        # the Speaker-X-shaped self_speaker comparison.
        apply_speaker_label(ctx, {"speaker": "Unknown", "display_name": "You", "is_self": True})
        assert ctx.self_speaker is None
        assert display_speaker(ctx, "Unknown") == "You"

    def test_one_label_per_person(self):
        ctx = self._ctx()
        apply_speaker_label(ctx, {"speaker": "Speaker A", "display_name": "Mom", "person_id": "mom"})
        apply_speaker_label(ctx, {"speaker": "Speaker B", "display_name": "Mom", "person_id": "mom"})
        assert "Speaker A" not in ctx.speaker_labels
        assert ctx.speaker_labels["Speaker B"]["person_id"] == "mom"

    @pytest.mark.parametrize("bad", [
        {"display_name": "Mom"},                                   # no speaker
        {"speaker": "", "display_name": "Mom"},
        {"speaker": "Speaker B"},                                  # no name
        {"speaker": "Speaker B", "display_name": "   "},
        {"speaker": "Speaker B", "display_name": "x" * 61},
        {"speaker": "x" * 65, "display_name": "Mom"},
        {"speaker": "Speaker B", "display_name": "Mom", "person_id": "Not A Slug"},
        {"speaker": "Speaker B", "display_name": "Mom", "person_id": 7},
        {"speaker": "Speaker B", "display_name": "Mom", "is_self": "yes"},
        {"speaker": 3, "display_name": "Mom"},
    ])
    def test_rejects_malformed(self, bad):
        ctx = self._ctx()
        assert apply_speaker_label(ctx, bad) is None
        assert ctx.speaker_labels == {}


class TestSpeakerLabelFrame:
    SID = "3d6d1d60-5a7e-5a2e-9f4a-4b2c1d0e9f11"

    def test_ack_then_prompt_names_the_person(self, fake_ws):
        llm = fake_ws.app.state.llm_client
        with open_ws(fake_ws, f"/ws/session/{self.SID}") as ws:
            ws.send_text(json.dumps({
                "type": "speaker_label", "speaker": "Speaker A",
                "display_name": "Mom", "person_id": "mom", "is_self": False,
            }))
            ack = json.loads(ws.receive_text())
            assert ack == {
                "type": "speaker_label_ack", "speaker": "Speaker A",
                "person_id": "mom", "display_name": "Mom", "is_self": False,
            }
            ws.send_bytes(b"\x00" * 50)
            resp = recv_skipping_transcripts(ws)
            assert resp["type"] == "suggestion"
            # The wire keeps the raw label …
            assert resp["speaker"] == "Speaker A"
        # … while the coach was told who spoke.
        user_prompts = [c.kwargs.get("user") for c in llm.complete.call_args_list]
        assert any('Transcript turn from Mom: "' in (p or "") for p in user_prompts)

    def test_unlabeled_prompt_is_byte_identical(self, fake_ws):
        llm = fake_ws.app.state.llm_client
        with open_ws(fake_ws, f"/ws/session/{self.SID}") as ws:
            ws.send_bytes(b"\x00" * 50)
            assert recv_skipping_transcripts(ws)["type"] == "suggestion"
        user_prompts = [c.kwargs.get("user") for c in llm.complete.call_args_list]
        assert all((p or "").startswith('Transcript turn: "') for p in user_prompts)

    def test_invalid_frame_is_an_error_not_a_close(self, fake_ws):
        with open_ws(fake_ws, f"/ws/session/{self.SID}") as ws:
            ws.send_text(json.dumps({"type": "speaker_label", "speaker": "Speaker A"}))
            assert json.loads(ws.receive_text()) == {"error": "invalid speaker_label"}
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 40}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"

    def test_is_self_switches_the_coached_voice(self, fake_ws, monkeypatch):
        seen = {}
        original = audio_pipeline._apply_config

        async def spy(ctx, payload):
            await original(ctx, payload)
            seen["ctx"] = ctx
        monkeypatch.setattr(audio_pipeline, "_apply_config", spy)
        with open_ws(fake_ws, f"/ws/session/{self.SID}") as ws:
            ws.send_text(json.dumps({"type": "config", "self_speaker": "Speaker A"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            assert seen["ctx"].self_speaker == "Speaker A"
            ws.send_text(json.dumps({
                "type": "speaker_label", "speaker": "Speaker B", "display_name": "You",
                "person_id": "self", "is_self": True,
            }))
            assert json.loads(ws.receive_text())["type"] == "speaker_label_ack"
            assert seen["ctx"].self_speaker == "Speaker B"
