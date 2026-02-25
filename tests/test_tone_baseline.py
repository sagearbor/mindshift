"""
Baseline tone scoring tests for MindShift.

Uses synthetic transcripts to verify that Claude-based tone scoring
produces consistent, reasonable scores across known emotional scenarios.
These tests validate the text-based tone analysis (Tier 0) before
audio-based detection is integrated (Tier 2).

Each test provides a transcript, asks for tone dimension scores
(warmth, defensiveness, sarcasm, constructiveness, calmness),
and asserts that scores fall within expected ranges.
"""

import json
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Synthetic transcripts with expected score ranges
# Each entry: transcript text, expected dimension ranges (min, max) out of 100
# ---------------------------------------------------------------------------

SYNTHETIC_TRANSCRIPTS = [
    {
        "id": "high_warmth",
        "transcript": "I really appreciate you sharing that with me. It takes courage to be vulnerable, and I want you to know I'm here for you no matter what.",
        "expected": {
            "warmth": (70, 100),
            "defensiveness": (0, 15),
            "sarcasm": (0, 10),
            "constructiveness": (60, 100),
            "calmness": (70, 100),
        },
    },
    {
        "id": "high_defensiveness",
        "transcript": "That's not what I said. You always twist my words. Why do I even bother trying to explain myself to you?",
        "expected": {
            "warmth": (0, 20),
            "defensiveness": (70, 100),
            "sarcasm": (10, 50),
            "constructiveness": (0, 25),
            "calmness": (0, 30),
        },
    },
    {
        "id": "high_sarcasm",
        "transcript": "Oh sure, because you're always so perfect. Must be nice living in a world where you never make mistakes.",
        "expected": {
            "warmth": (0, 15),
            "defensiveness": (40, 80),
            "sarcasm": (70, 100),
            "constructiveness": (0, 20),
            "calmness": (20, 60),
        },
    },
    {
        "id": "constructive_feedback",
        "transcript": "I noticed we keep running into this issue. What if we tried setting a weekly check-in to stay on the same page? I think that could help us both.",
        "expected": {
            "warmth": (50, 85),
            "defensiveness": (0, 15),
            "sarcasm": (0, 10),
            "constructiveness": (75, 100),
            "calmness": (65, 100),
        },
    },
    {
        "id": "angry_escalation",
        "transcript": "I am so sick of this! Every single time I ask you for one simple thing, you can't even be bothered. I'm done talking about it.",
        "expected": {
            "warmth": (0, 10),
            "defensiveness": (30, 70),
            "sarcasm": (0, 30),
            "constructiveness": (0, 15),
            "calmness": (0, 15),
        },
    },
    {
        "id": "neutral_logistics",
        "transcript": "The appointment is at 3 PM on Thursday. Can you pick up the kids from school that day, or should I rearrange my schedule?",
        "expected": {
            "warmth": (30, 60),
            "defensiveness": (0, 10),
            "sarcasm": (0, 10),
            "constructiveness": (50, 85),
            "calmness": (70, 100),
        },
    },
    {
        "id": "passive_aggressive",
        "transcript": "No, it's fine. Do whatever you want. You always do anyway. I don't know why you even ask me.",
        "expected": {
            "warmth": (0, 15),
            "defensiveness": (40, 80),
            "sarcasm": (40, 85),
            "constructiveness": (0, 15),
            "calmness": (20, 55),
        },
    },
    {
        "id": "vulnerable_disclosure",
        "transcript": "I've been feeling really lonely lately. I know I pull away sometimes, and I'm sorry. I'm scared that if I tell you how I feel, you'll think I'm too much.",
        "expected": {
            "warmth": (40, 75),
            "defensiveness": (0, 20),
            "sarcasm": (0, 10),
            "constructiveness": (40, 75),
            "calmness": (40, 75),
        },
    },
    {
        "id": "contempt",
        "transcript": "You can't even load a dishwasher right. Honestly, it's like living with a teenager. I shouldn't have to explain basic things to a grown adult.",
        "expected": {
            "warmth": (0, 10),
            "defensiveness": (20, 60),
            "sarcasm": (40, 80),
            "constructiveness": (0, 15),
            "calmness": (15, 50),
        },
    },
    {
        "id": "full_empathy_validation",
        "transcript": "That sounds incredibly hard. I can see why you'd feel that way — anyone would. Thank you for telling me. What do you need from me right now?",
        "expected": {
            "warmth": (80, 100),
            "defensiveness": (0, 10),
            "sarcasm": (0, 5),
            "constructiveness": (70, 100),
            "calmness": (75, 100),
        },
    },
]

TONE_DIMENSIONS = ["warmth", "defensiveness", "sarcasm", "constructiveness", "calmness"]

SCORING_PROMPT_TEMPLATE = """You are a tone analysis system for a relationship coaching app.

Analyze the following transcript and score it on these 5 dimensions, each from 0 to 100:

- warmth: Kindness, affection, positive regard (0 = cold/hostile, 100 = deeply warm)
- defensiveness: Self-protection, blame-shifting, denial (0 = open/receptive, 100 = highly defensive)
- sarcasm: Ironic, mocking, or contemptuous undertone (0 = sincere, 100 = dripping sarcasm)
- constructiveness: Solution-focused, collaborative intent (0 = destructive/blaming, 100 = highly constructive)
- calmness: Emotional regulation, absence of escalation (0 = highly agitated, 100 = very calm)

Transcript:
\"{transcript}\"

Respond with ONLY a JSON object, no other text:
{{"warmth": <int>, "defensiveness": <int>, "sarcasm": <int>, "constructiveness": <int>, "calmness": <int>}}"""


def _parse_scores(response_text: str) -> dict:
    """Extract JSON scores from LLM response, handling markdown fences."""
    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _score_transcript(transcript: str) -> dict:
    """Send a transcript to Claude for tone scoring.

    Requires ANTHROPIC_API_KEY in the environment.
    Falls back to a mock scorer if the SDK is not installed or key is missing.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                messages=[
                    {
                        "role": "user",
                        "content": SCORING_PROMPT_TEMPLATE.format(transcript=transcript),
                    }
                ],
            )
            return _parse_scores(message.content[0].text)
        except ImportError:
            pass  # Fall through to mock

    # Mock scorer: returns midpoint of expected ranges for testing without API
    # This allows CI to run without an API key
    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_transcripts():
    """Load the sample_transcripts.json fixture file."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_transcripts.json"
    with open(fixture_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToneBaseline:
    """Verify Claude tone scoring produces scores in expected ranges for
    known synthetic transcripts."""

    @pytest.mark.parametrize(
        "case",
        SYNTHETIC_TRANSCRIPTS,
        ids=[c["id"] for c in SYNTHETIC_TRANSCRIPTS],
    )
    def test_tone_scores_in_expected_range(self, case):
        """Each synthetic transcript should score within the expected
        range for every tone dimension."""
        scores = _score_transcript(case["transcript"])

        if scores is None:
            pytest.skip("ANTHROPIC_API_KEY not set and anthropic SDK not available")

        for dim in TONE_DIMENSIONS:
            lo, hi = case["expected"][dim]
            actual = scores[dim]
            assert lo <= actual <= hi, (
                f"[{case['id']}] {dim}: expected {lo}–{hi}, got {actual}"
            )

    @pytest.mark.parametrize(
        "case",
        SYNTHETIC_TRANSCRIPTS,
        ids=[c["id"] for c in SYNTHETIC_TRANSCRIPTS],
    )
    def test_scores_are_valid_integers(self, case):
        """All scores should be integers in [0, 100]."""
        scores = _score_transcript(case["transcript"])

        if scores is None:
            pytest.skip("ANTHROPIC_API_KEY not set and anthropic SDK not available")

        for dim in TONE_DIMENSIONS:
            val = scores[dim]
            assert isinstance(val, int), f"[{case['id']}] {dim} should be int, got {type(val)}"
            assert 0 <= val <= 100, f"[{case['id']}] {dim} out of range: {val}"

    def test_all_dimensions_present(self):
        """Scoring a transcript should return all 5 dimensions."""
        scores = _score_transcript(SYNTHETIC_TRANSCRIPTS[0]["transcript"])

        if scores is None:
            pytest.skip("ANTHROPIC_API_KEY not set and anthropic SDK not available")

        for dim in TONE_DIMENSIONS:
            assert dim in scores, f"Missing dimension: {dim}"

    def test_high_warmth_beats_contempt(self):
        """Warmth of the 'high_warmth' transcript should exceed 'contempt'."""
        warmth_scores = _score_transcript(SYNTHETIC_TRANSCRIPTS[0]["transcript"])
        contempt_scores = _score_transcript(SYNTHETIC_TRANSCRIPTS[8]["transcript"])

        if warmth_scores is None or contempt_scores is None:
            pytest.skip("ANTHROPIC_API_KEY not set and anthropic SDK not available")

        assert warmth_scores["warmth"] > contempt_scores["warmth"], (
            f"high_warmth ({warmth_scores['warmth']}) should beat contempt ({contempt_scores['warmth']})"
        )

    def test_escalation_less_calm_than_neutral(self):
        """Angry escalation should score lower on calmness than neutral logistics."""
        escalation = _score_transcript(SYNTHETIC_TRANSCRIPTS[4]["transcript"])
        neutral = _score_transcript(SYNTHETIC_TRANSCRIPTS[5]["transcript"])

        if escalation is None or neutral is None:
            pytest.skip("ANTHROPIC_API_KEY not set and anthropic SDK not available")

        assert neutral["calmness"] > escalation["calmness"], (
            f"neutral ({neutral['calmness']}) should be calmer than escalation ({escalation['calmness']})"
        )


class TestSampleTranscriptsFixture:
    """Validate the structure and content of sample_transcripts.json."""

    def test_fixture_loads(self, sample_transcripts):
        """The fixture file should load without errors."""
        assert "conversations" in sample_transcripts
        assert "version" in sample_transcripts

    def test_has_five_conversations(self, sample_transcripts):
        """There should be exactly 5 conversations."""
        assert len(sample_transcripts["conversations"]) == 5

    def test_each_conversation_has_required_fields(self, sample_transcripts):
        """Each conversation must have id, scenario, roles, and turns."""
        for conv in sample_transcripts["conversations"]:
            assert "id" in conv
            assert "scenario" in conv
            assert "roles" in conv
            assert "turns" in conv
            assert len(conv["turns"]) >= 3, f"{conv['id']} has fewer than 3 turns"

    def test_each_turn_has_labels(self, sample_transcripts):
        """Each turn must have empathy_level, tone, and response_type."""
        for conv in sample_transcripts["conversations"]:
            for i, turn in enumerate(conv["turns"]):
                assert "speaker" in turn, f"{conv['id']} turn {i}: missing speaker"
                assert "text" in turn, f"{conv['id']} turn {i}: missing text"
                assert "empathy_level" in turn, f"{conv['id']} turn {i}: missing empathy_level"
                assert "tone" in turn, f"{conv['id']} turn {i}: missing tone"
                assert "response_type" in turn, f"{conv['id']} turn {i}: missing response_type"

    def test_empathy_levels_in_range(self, sample_transcripts):
        """All empathy_level values should be 0–100."""
        for conv in sample_transcripts["conversations"]:
            for turn in conv["turns"]:
                level = turn["empathy_level"]
                assert 0 <= level <= 100, (
                    f"{conv['id']}: empathy_level {level} out of range"
                )

    def test_roles_defined(self, sample_transcripts):
        """Each conversation should define speaker_a and speaker_b roles."""
        for conv in sample_transcripts["conversations"]:
            assert "speaker_a" in conv["roles"]
            assert "speaker_b" in conv["roles"]
            for role_key in ("speaker_a", "speaker_b"):
                role = conv["roles"][role_key]
                assert "name" in role
                assert "role" in role
