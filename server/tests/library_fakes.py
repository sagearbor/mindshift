"""Test-only doubles for the coach knowledge library.

``FakeEmbedder`` is DETERMINISTIC bag-of-words hashing: texts sharing words
get similar vectors, so "retrieval picks the relevant chunk" is a real
assertion about ranking, not a canned answer. It exists ONLY under tests —
production code never falls back to it (no embedding credentials means
retrieval reports itself unavailable).
"""

from __future__ import annotations

import hashlib
import math
import re

from library.embeddings import EmbeddingUnavailable

DIM = 64
_WORD = re.compile(r"[a-z0-9]+")


def bow_vector(text: str, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    for w in _WORD.findall(text.lower()):
        if len(w) < 3:
            continue
        h = int(hashlib.sha1(w.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class FakeEmbedder:
    name = "fake-bow"
    dimension = DIM

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, int]] = []

    async def embed(self, texts, *, task):
        self.calls.append((task, len(texts)))
        if self.fail:
            raise EmbeddingUnavailable("fake embedder told to fail")
        return [bow_vector(t) for t in texts]
