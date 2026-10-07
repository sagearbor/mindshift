"""Opt-in live check of the library's Vertex AI embedder.

SKIPPED unless ``MINDSHIFT_LIBRARY_LIVE=1`` (Application Default Credentials
must also resolve). It embeds three short texts with the REAL
``text-embedding-005`` model on the configured project
(``MINDSHIFT_VERTEX_PROJECT`` / ``MINDSHIFT_FIRESTORE_PROJECT`` /
``GOOGLE_CLOUD_PROJECT``, default ``arborfam-hub``) and asserts the vectors
are 768-d and that a salary question lands nearer the salary note than the
parking note. It touches no Firestore or GCS data.

    MINDSHIFT_LIBRARY_LIVE=1 pytest server/tests/test_library_live.py -q
"""

import math
import os

import pytest

from library.embeddings import VertexEmbedder, resolve_project

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        os.getenv("MINDSHIFT_LIBRARY_LIVE") != "1",
        reason="MINDSHIFT_LIBRARY_LIVE!=1 — live Vertex embedding test skipped",
    ),
]


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


async def test_vertex_embeddings_rank_the_relevant_note_first():
    emb = VertexEmbedder(resolve_project() or "arborfam-hub")
    docs = await emb.embed(
        [
            "Market data: senior engineers in Denver earn a median salary of $168k.",
            "The parking garage closes at 9pm and visitors use level 2.",
        ],
        task="document",
    )
    [query] = await emb.embed(["How much should I ask for in my raise?"], task="query")
    assert all(len(v) == 768 for v in docs + [query])
    assert _cos(query, docs[0]) > _cos(query, docs[1])
