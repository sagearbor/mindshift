"""Text embeddings for library retrieval — Vertex AI, on the EXISTING project
credentials (Application Default Credentials; on Cloud Run the service
account). No new API key.

Configuration (all optional):

* ``MINDSHIFT_LIBRARY_EMBEDDINGS``  ``off`` disables embeddings entirely.
* ``MINDSHIFT_VERTEX_PROJECT``      GCP project; falls back to
  ``MINDSHIFT_FIRESTORE_PROJECT``, then ``GOOGLE_CLOUD_PROJECT``.
* ``MINDSHIFT_VERTEX_LOCATION``     default ``us-central1``.
* ``MINDSHIFT_EMBEDDING_MODEL``     default ``text-embedding-005`` (768-d).

The Cloud Run service account needs ``roles/aiplatform.user`` and the project
needs ``aiplatform.googleapis.com`` enabled.

Honest degradation: with no project or no resolvable credentials,
:func:`get_embedder` returns ``None`` and the library reports retrieval as
unavailable. There is no fallback embedder outside the test suite.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

Task = Literal["document", "query"]

DEFAULT_MODEL = "text-embedding-005"
DEFAULT_LOCATION = "us-central1"
DEFAULT_DIMENSION = 768
# text-embedding-005 accepts up to 250 instances / 20k tokens per request; a
# batch of 10 ~800-token chunks stays well inside the token cap.
BATCH_SIZE = 10

_TASK_TYPES = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}


class EmbeddingUnavailable(RuntimeError):
    """Embedding could not be produced (no credentials, API error, ...)."""


class Embedder(Protocol):
    name: str
    dimension: int

    async def embed(self, texts: list[str], *, task: Task) -> list[list[float]]: ...


class VertexEmbedder:
    """Vertex AI ``publishers/google/models/{model}:predict`` over httpx."""

    def __init__(
        self, project: str, location: str = DEFAULT_LOCATION,
        model: str = DEFAULT_MODEL, dimension: int = DEFAULT_DIMENSION,
        credentials=None,
    ) -> None:
        self.project = project
        self.location = location
        self.model = model
        self.dimension = dimension
        self.name = f"vertex:{model}"
        self._credentials = credentials
        self._lock = asyncio.Lock()

    @property
    def url(self) -> str:
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/"
            f"{self.project}/locations/{self.location}/publishers/google/models/"
            f"{self.model}:predict"
        )

    async def _token(self) -> str:
        async with self._lock:
            if self._credentials is None:
                import google.auth

                self._credentials, _ = await asyncio.to_thread(
                    google.auth.default,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            if not self._credentials.valid:
                from google.auth.transport.requests import Request

                await asyncio.to_thread(self._credentials.refresh, Request())
            return self._credentials.token

    async def embed(self, texts: list[str], *, task: Task) -> list[list[float]]:
        import httpx

        out: list[list[float]] = []
        try:
            token = await self._token()
            async with httpx.AsyncClient(timeout=60) as client:
                for i in range(0, len(texts), BATCH_SIZE):
                    batch = texts[i:i + BATCH_SIZE]
                    resp = await client.post(
                        self.url,
                        headers={"Authorization": f"Bearer {token}"},
                        json={
                            "instances": [
                                {"content": t, "task_type": _TASK_TYPES[task]}
                                for t in batch
                            ],
                            "parameters": {"outputDimensionality": self.dimension},
                        },
                    )
                    if resp.status_code != 200:
                        raise EmbeddingUnavailable(
                            f"Vertex embeddings HTTP {resp.status_code}"
                        )
                    preds = resp.json().get("predictions") or []
                    if len(preds) != len(batch):
                        raise EmbeddingUnavailable("Vertex returned a short batch")
                    out.extend([p["embeddings"]["values"] for p in preds])
        except EmbeddingUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
            logger.warning("Vertex embedding failed", exc_info=True)
            raise EmbeddingUnavailable(f"embedding failed ({type(exc).__name__})") from exc
        return out


def resolve_project() -> str | None:
    for var in ("MINDSHIFT_VERTEX_PROJECT", "MINDSHIFT_FIRESTORE_PROJECT", "GOOGLE_CLOUD_PROJECT"):
        value = (os.getenv(var) or "").strip()
        if value:
            return value
    return None


def get_embedder() -> tuple["Embedder | None", str | None]:
    """Return ``(embedder, None)`` or ``(None, reason it is unavailable)``."""
    if (os.getenv("MINDSHIFT_LIBRARY_EMBEDDINGS") or "").strip().lower() in {"off", "0", "false", "no"}:
        return None, "embeddings disabled (MINDSHIFT_LIBRARY_EMBEDDINGS=off)"
    project = resolve_project()
    if not project:
        return None, "no GCP project configured for Vertex embeddings"
    try:
        import google.auth

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    except Exception as exc:  # noqa: BLE001 — DefaultCredentialsError and friends
        logger.info("Library embeddings unavailable: %s", exc)
        return None, f"no Google credentials for Vertex embeddings ({type(exc).__name__})"
    return VertexEmbedder(
        project,
        location=os.getenv("MINDSHIFT_VERTEX_LOCATION") or DEFAULT_LOCATION,
        model=os.getenv("MINDSHIFT_EMBEDDING_MODEL") or DEFAULT_MODEL,
        credentials=credentials,
    ), None
