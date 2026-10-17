"""VertexEmbedder request shape + credential gating, with the HTTP layer
replaced by an httpx.MockTransport (no network, no Google credentials)."""

import json

import httpx
import pytest

from library import embeddings
from library.embeddings import EmbeddingUnavailable, VertexEmbedder

pytestmark = pytest.mark.anyio


class _Creds:
    valid = True
    token = "tok"


def _patch_transport(monkeypatch, handler):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


async def test_batches_requests_and_parses_vectors(monkeypatch):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((str(request.url), request.headers["authorization"], body))
        return httpx.Response(200, json={"predictions": [
            {"embeddings": {"values": [float(i)] * 3}} for i, _ in enumerate(body["instances"])
        ]})

    _patch_transport(monkeypatch, handler)
    emb = VertexEmbedder("proj-x", credentials=_Creds(), dimension=3)
    out = await emb.embed([f"t{i}" for i in range(embeddings.BATCH_SIZE + 2)], task="document")
    assert len(out) == embeddings.BATCH_SIZE + 2
    assert len(seen) == 2
    url, auth, body = seen[0]
    assert url == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/proj-x/locations/"
        "us-central1/publishers/google/models/text-embedding-005:predict"
    )
    assert auth == "Bearer tok"
    assert body["instances"][0] == {"content": "t0", "task_type": "RETRIEVAL_DOCUMENT"}
    assert body["parameters"] == {"outputDimensionality": 3}

    await emb.embed(["q"], task="query")
    assert seen[-1][2]["instances"][0]["task_type"] == "RETRIEVAL_QUERY"


async def test_api_errors_raise_unavailable_with_the_google_reason(monkeypatch):
    def handler(request):
        return httpx.Response(403, json={"error": {
            "status": "PERMISSION_DENIED",
            "details": [{"reason": "SERVICE_DISABLED"}],
        }})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(EmbeddingUnavailable, match="403 \\(SERVICE_DISABLED\\)"):
        await VertexEmbedder("p", credentials=_Creds()).embed(["x"], task="query")


def test_no_project_means_no_embedder(monkeypatch):
    for var in ("MINDSHIFT_VERTEX_PROJECT", "MINDSHIFT_FIRESTORE_PROJECT", "GOOGLE_CLOUD_PROJECT",
                "MINDSHIFT_LIBRARY_EMBEDDINGS"):
        monkeypatch.delenv(var, raising=False)
    emb, reason = embeddings.get_embedder()
    assert emb is None and "project" in reason


def test_embeddings_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("MINDSHIFT_VERTEX_PROJECT", "p")
    monkeypatch.setenv("MINDSHIFT_LIBRARY_EMBEDDINGS", "off")
    emb, reason = embeddings.get_embedder()
    assert emb is None and "disabled" in reason


def test_unresolvable_credentials_mean_no_embedder(monkeypatch):
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    def _no_creds(**kwargs):
        raise DefaultCredentialsError("none here")

    monkeypatch.setenv("MINDSHIFT_VERTEX_PROJECT", "p")
    monkeypatch.delenv("MINDSHIFT_LIBRARY_EMBEDDINGS", raising=False)
    monkeypatch.setattr(google.auth, "default", _no_creds)
    emb, reason = embeddings.get_embedder()
    assert emb is None and "credentials" in reason


def test_firestore_vector_query_matches_the_documented_index():
    """Offline (anonymous credentials, nothing sent): the retrieval query is
    uid == AND item_id == pre-filters + COSINE find_nearest on ``embedding`` —
    exactly the composite index in server/library/firestore.indexes.json."""
    from pathlib import Path

    from google.auth.credentials import AnonymousCredentials
    from google.cloud import firestore
    from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
    from google.cloud.firestore_v1.vector import Vector

    from library.store import CHUNKS, FirestoreLibraryStore

    store = FirestoreLibraryStore("p")
    store._db = firestore.Client(project="p", credentials=AnonymousCredentials())
    pb = store._chunks_for("u", "i").find_nearest(
        vector_field="embedding", query_vector=Vector([0.1, 0.2]),
        distance_measure=DistanceMeasure.COSINE, limit=3,
        distance_result_field="vector_distance",
    )._to_protobuf()
    filters = [f.field_filter.field.field_path for f in pb.where.composite_filter.filters]
    assert filters == ["uid", "item_id"]
    assert pb.find_nearest.vector_field.field_path == "embedding"

    index = json.loads(
        (Path(__file__).resolve().parents[1] / "library" / "firestore.indexes.json").read_text()
    )["indexes"][0]
    assert index["collectionGroup"] == CHUNKS
    assert [f["fieldPath"] for f in index["fields"]] == ["uid", "item_id", "embedding"]
    assert index["fields"][2]["vectorConfig"]["dimension"] == embeddings.DEFAULT_DIMENSION
