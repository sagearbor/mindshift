"""The coach prompt is GENERIC (owner requirements A + B, 2026-10-07).

Background: at a family dinner the live coach treated the owner's son as the
wearer and scripted "I think I did well on the quiz" in the son's voice. One
cause was the prompt itself: it hard-coded ``role="Husband"`` as the default,
never said who the wearer was, and never forbade invented facts.

These tests build the REAL prompts (no LLM call — the assertions are on the
exact system text the LLM would receive):

* the core job is one generic sentence — help the wearer be a more positive
  presence — with no relationship assumed;
* never state facts the wearer hasn't said, never put words in another
  speaker's mouth;
* live output prefers short cues about the wearer's manner / next move;
* an OPTIONAL ``relationship`` (child / partner / parent / coworker / friend /
  other) adds a light hint and never changes the core job;
* an older client that sends a free-text ``role`` (or nothing) still works.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import audio_pipeline
from main import (
    COACH_CORE_JOB,
    COACH_GROUND_RULES,
    app,
    empathy_system_prompt,
    normalize_relationship,
    self_feedback_prompt,
)

# Words that would mean the prompt assumes a relationship nobody stated.
RELATIONSHIP_WORDS = (
    "husband", "wife", "spouse", "partner", "son", "daughter", "child",
    "parent", "boss", "coworker", "marriage",
)


def _assert_no_relationship(prompt: str) -> None:
    low = prompt.lower()
    for word in RELATIONSHIP_WORDS:
        assert f" {word}" not in low, f"prompt assumes a relationship: {word!r}"
    assert "role in this conversation" not in low


class TestCorePrompt:
    @pytest.mark.parametrize("slider", [0, 35, 65, 100])
    @pytest.mark.parametrize("live", [True, False])
    def test_default_prompt_assumes_no_relationship(self, slider, live):
        prompt = empathy_system_prompt(slider, live=live)
        _assert_no_relationship(prompt)
        assert COACH_CORE_JOB in prompt
        assert COACH_GROUND_RULES in prompt

    def test_core_job_and_ground_rules_wording(self):
        assert "more positive presence" in COACH_CORE_JOB
        assert "wearer" in COACH_CORE_JOB
        low = COACH_GROUND_RULES.lower()
        assert "never state facts the wearer has not said" in low
        assert "never put words in another speaker's mouth" in low

    def test_live_prompt_prefers_short_cues_over_scripts(self):
        live = empathy_system_prompt(50, live=True)
        assert "ask what part was hardest" in live
        assert "acknowledge before answering" in live
        assert "at most 10 words" in live
        assert "at most 15 words" not in live

    @pytest.mark.parametrize("slider", [0, 50, 100])
    def test_self_feedback_prompt_is_generic_too(self, slider):
        prompt = self_feedback_prompt(slider)
        _assert_no_relationship(prompt)
        assert COACH_CORE_JOB in prompt
        assert COACH_GROUND_RULES in prompt


class TestOptionalRelationship:
    @pytest.mark.parametrize("rel", ["child", "partner", "parent", "coworker", "friend"])
    def test_relationship_adds_a_light_hint_only(self, rel):
        base = empathy_system_prompt(50, live=True)
        hinted = empathy_system_prompt(50, live=True, relationship=rel)
        assert f"the wearer's {rel}" in hinted
        assert "does not change your job" in hinted
        # The core paragraph (job + stance) is byte-identical with or without.
        assert hinted.split("\n\n")[0] == base.split("\n\n")[0]
        assert COACH_GROUND_RULES in hinted

    def test_other_and_unknown_relationships_add_nothing(self):
        base = empathy_system_prompt(50, live=True)
        assert empathy_system_prompt(50, live=True, relationship="other") == base
        assert empathy_system_prompt(50, live=True, relationship="wizard") == base
        assert empathy_system_prompt(50, live=True, relationship=None) == base

    def test_normalize_relationship(self):
        assert normalize_relationship("  Child ") == "child"
        assert normalize_relationship("wizard") is None
        assert normalize_relationship(7) is None
        assert normalize_relationship(None) is None

    def test_legacy_role_is_only_a_hint(self):
        """An older client's free-text role still works, as context — never
        as 'the user's role is …' and never replacing the generic job."""
        base = empathy_system_prompt(50)
        legacy = empathy_system_prompt(50, "Husband / Wife")
        assert "Husband / Wife" in legacy
        assert legacy.split("\n\n")[0] == base.split("\n\n")[0]
        assert "role in this conversation" not in legacy.lower()


class TestSessionDefaults:
    def test_session_context_has_no_role_default(self):
        ctx = audio_pipeline.SessionContext(session_id="s")
        assert ctx.role is None
        assert ctx.relationship is None

    def test_config_relationship_is_validated(self):
        ctx = audio_pipeline.SessionContext(session_id="s")
        asyncio.run(audio_pipeline._apply_config(ctx, {"relationship": "Coworker"}))
        assert ctx.relationship == "coworker"
        asyncio.run(audio_pipeline._apply_config(ctx, {"relationship": "wizard"}))
        assert ctx.relationship == "coworker"  # bad value ignored
        asyncio.run(audio_pipeline._apply_config(ctx, {"relationship": None}))
        assert ctx.relationship is None  # explicit null resets


class TestRestRespondBackwardCompatible:
    def _client(self, captured):
        class LLM:
            def complete(self, system, user, **kw):
                captured.append(system)
                return json.dumps({
                    "suggestions": ["a", "b", "c"],
                    "tone_score": {"warmth": 1, "defensiveness": 1, "sarcasm": 1,
                                   "constructiveness": 1, "overall": 1},
                })
        app.state.llm_client = LLM()
        return TestClient(app)

    def test_respond_without_role(self):
        captured: list[str] = []
        client = self._client(captured)
        r = client.post("/respond", json={"transcript_turn": "hi", "empathy_slider": 50})
        assert r.status_code == 200, r.text
        _assert_no_relationship(captured[-1])

    def test_respond_with_legacy_role_and_relationship(self):
        captured: list[str] = []
        client = self._client(captured)
        r = client.post("/respond", json={
            "transcript_turn": "hi", "empathy_slider": 50,
            "role": "Parent / Child", "relationship": "child",
        })
        assert r.status_code == 200, r.text
        assert "the wearer's child" in captured[-1]
        assert "Parent / Child" in captured[-1]
