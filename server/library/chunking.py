"""Token estimate + overlapping chunker for library text.

Tokens are ESTIMATED as ``ceil(chars / 4)`` — the usual English rule of thumb
for Claude/Gemini tokenizers, and no tokenizer dependency on the server. It
runs slightly high for plain prose, which is the safe direction for a prompt
budget. Chunks target ~800 tokens with ~100 tokens of overlap, cut at the
nearest paragraph / line / sentence / word boundary so a fact is rarely split.
"""

from __future__ import annotations

import math

CHARS_PER_TOKEN = 4
DEFAULT_TARGET_TOKENS = 800
DEFAULT_OVERLAP_TOKENS = 100

_BOUNDARIES = ("\n\n", "\n", ". ", "? ", "! ", "; ", " ")


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0


def chunk_text(
    text: str,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    max_chars = target_tokens * CHARS_PER_TOKEN
    overlap = min(overlap_tokens * CHARS_PER_TOKEN, max_chars // 2)
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            floor = start + max_chars // 2
            for sep in _BOUNDARIES:
                cut = text.rfind(sep, floor, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        # Start the overlap on a word boundary rather than mid-word.
        space = text.find(" ", nxt, end)
        if space != -1:
            nxt = space + 1
        start = nxt
    return chunks
