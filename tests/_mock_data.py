"""Shared mock LLM payloads and constants for the integration test suite.

Kept in a uniquely-named module (not conftest.py) so test files can import it
unambiguously even when pytest collects both ``server/`` and ``tests/`` in one
run — two ``conftest.py`` modules would otherwise shadow each other.
"""

import json

MOCK_RESPOND_JSON = json.dumps({
    "suggestions": [
        "I hear what you're saying.",
        "That sounds really frustrating.",
        "Can you tell me more about how that made you feel?",
    ],
    "tone_score": {
        "warmth": 60,
        "defensiveness": 30,
        "sarcasm": 10,
        "constructiveness": 55,
        "overall": 65,
    },
})

MOCK_ASSERTIVE_JSON = json.dumps({
    "suggestions": [
        "Set a clear boundary here.",
        "Be direct about your needs.",
        "State your position firmly.",
    ],
    "tone_score": {
        "warmth": 20,
        "defensiveness": 60,
        "sarcasm": 15,
        "constructiveness": 40,
        "overall": 35,
    },
})

MOCK_FULL_EMPATHY_JSON = json.dumps({
    "suggestions": [
        "That must be so hard for you.",
        "Your feelings are completely valid.",
        "I'm here for you no matter what.",
    ],
    "tone_score": {
        "warmth": 90,
        "defensiveness": 5,
        "sarcasm": 2,
        "constructiveness": 70,
        "overall": 85,
    },
})

MOCK_SCORE_JSON = json.dumps({
    "warmth": 70,
    "defensiveness": 20,
    "sarcasm": 5,
    "constructiveness": 80,
    "overall": 75,
})

TONE_SCORE_KEYS = {"warmth", "defensiveness", "sarcasm", "constructiveness", "overall"}
