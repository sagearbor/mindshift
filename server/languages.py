"""Coach / conversation language settings (config keys `coach_language`,
`conversation_language`).

Single allowlist, mirrored by apps/mobile/src/live/languagePrefs.ts. Unknown
or unset values mean "auto": prompts stay byte-identical and Deepgram keeps
its default language, exactly as before these settings existed.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# BCP-47 -> English name (used in the prompt instruction).
LANGUAGES: dict[str, str] = {
    "en-US": "English",
    "es-ES": "Spanish",
    "fr-FR": "French",
    "de-DE": "German",
    "pt-BR": "Portuguese",
    "it-IT": "Italian",
    "hi-IN": "Hindi",
    "ja-JP": "Japanese",
    "zh-CN": "Mandarin Chinese",
}

# Deepgram nova-3 `language` values for the allowlist. A code absent here
# falls back to the default (and is logged) rather than failing the stream.
DEEPGRAM_LANGUAGE: dict[str, str] = {
    "en-US": "en-US",
    "es-ES": "es",
    "fr-FR": "fr",
    "de-DE": "de",
    "pt-BR": "pt-BR",
    "it-IT": "it",
    "hi-IN": "hi",
    "ja-JP": "ja",
    "zh-CN": "zh-CN",
}


def validate_language(value: object) -> str | None:
    """The code if on the allowlist, else None ("auto"/unknown/non-string)."""
    return value if isinstance(value, str) and value in LANGUAGES else None


def language_instruction(code: str | None, noun: str = "suggestion") -> str | None:
    name = LANGUAGES.get(code) if code else None
    return f"Write the {noun} in {name}." if name else None


def with_language(system: str, code: str | None, noun: str = "suggestion") -> str:
    """`system` with the reply-language line appended; unchanged when unset."""
    line = language_instruction(code, noun)
    return f"{system}\n\n{line}" if line else system


def deepgram_language(code: str | None) -> str | None:
    if not code:
        return None
    dg = DEEPGRAM_LANGUAGE.get(code)
    if dg is None:
        logger.warning("No Deepgram language mapping for %s; using default", code)
    return dg
