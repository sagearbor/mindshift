"""Metadata + vector tier for the library: Firestore in production, memory in
keyless dev/CI (same ``MINDSHIFT_FIRESTORE_PROJECT`` switch as watch/store.py).

Firestore layout (top-level collections, every doc carries ``uid``)::

    library_items/{item_id}              ItemRecord fields
    library_chunks/{item_id}_{ord:05d}   uid, item_id, ord, text,
                                         embedding (Vector, 768-d)

Retrieval is native Firestore vector search, pre-filtered (equality only) to
the caller's uid and ONE selected item per query, merged by distance::

    for item in selected:
        library_chunks.where(uid == U).where(item_id == item)
            .find_nearest("embedding", q, COSINE, limit=k)

That query needs ONE composite vector index (not created by code; create it
at deploy — also in server/library/firestore.indexes.json)::

    gcloud firestore indexes composite create \\
      --project=arborfam-hub \\
      --collection-group=library_chunks \\
      --query-scope=COLLECTION \\
      --field-config=field-path=uid,order=ASCENDING \\
      --field-config=field-path=item_id,order=ASCENDING \\
      --field-config='field-path=embedding,vector-config={"dimension":"768","flat":"{}"}'

Listing items (``uid ==``) and deleting chunks (``uid == AND item_id ==``)
use Firestore's automatic single-field indexes and need nothing extra.
"""

from __future__ import annotations

import asyncio
import math
import os
from typing import Protocol

from library.models import ChunkRecord, ItemRecord

ITEMS = "library_items"
CHUNKS = "library_chunks"
_BATCH_LIMIT = 400      # under Firestore's 500-writes-per-batch cap


class LibraryStore(Protocol):
    async def put_item(self, rec: ItemRecord) -> None: ...
    async def get_item(self, uid: str, item_id: str) -> ItemRecord | None: ...
    async def list_items(self, uid: str) -> list[ItemRecord]: ...
    async def delete_item(self, uid: str, item_id: str) -> bool: ...
    async def put_chunks(self, chunks: list[ChunkRecord]) -> None: ...
    async def delete_chunks(self, uid: str, item_id: str) -> int: ...
    async def delete_all_chunks(self, uid: str) -> int: ...
    async def nearest(
        self, uid: str, item_ids: list[str], vector: list[float], limit: int,
    ) -> list[tuple[ChunkRecord, float]]: ...


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return 1.0 - dot / (na * nb)


class MemoryLibraryStore:
    def __init__(self) -> None:
        self._items: dict[str, ItemRecord] = {}
        self._chunks: dict[str, ChunkRecord] = {}

    async def put_item(self, rec: ItemRecord) -> None:
        self._items[rec.id] = ItemRecord.from_doc(rec.to_doc())

    async def get_item(self, uid: str, item_id: str) -> ItemRecord | None:
        rec = self._items.get(item_id)
        if rec is None or rec.uid != uid:
            return None
        return ItemRecord.from_doc(rec.to_doc())

    async def list_items(self, uid: str) -> list[ItemRecord]:
        return [ItemRecord.from_doc(r.to_doc()) for r in self._items.values() if r.uid == uid]

    async def delete_item(self, uid: str, item_id: str) -> bool:
        rec = self._items.get(item_id)
        if rec is None or rec.uid != uid:
            return False
        del self._items[item_id]
        return True

    async def put_chunks(self, chunks: list[ChunkRecord]) -> None:
        for c in chunks:
            self._chunks[c.id] = c

    async def delete_chunks(self, uid: str, item_id: str) -> int:
        ids = [k for k, c in self._chunks.items() if c.uid == uid and c.item_id == item_id]
        for k in ids:
            del self._chunks[k]
        return len(ids)

    async def delete_all_chunks(self, uid: str) -> int:
        ids = [k for k, c in self._chunks.items() if c.uid == uid]
        for k in ids:
            del self._chunks[k]
        return len(ids)

    async def nearest(self, uid, item_ids, vector, limit):
        wanted = set(item_ids)
        scored = [
            (c, _cosine_distance(vector, c.embedding))
            for c in self._chunks.values()
            if c.uid == uid and c.item_id in wanted and c.embedding
        ]
        scored.sort(key=lambda t: (t[1], t[0].id))
        return scored[:limit]


class FirestoreLibraryStore:
    """Lazily imports google-cloud-firestore (same pattern as watch/store.py)."""

    def __init__(self, project: str) -> None:
        self.project = project
        self._db = None

    def _get_db(self):
        if self._db is None:
            from google.cloud import firestore

            self._db = firestore.Client(project=self.project)
        return self._db

    # -- items ---------------------------------------------------------------

    def _put_item_sync(self, rec):
        self._get_db().collection(ITEMS).document(rec.id).set(rec.to_doc())

    def _get_item_sync(self, uid, item_id):
        doc = self._get_db().collection(ITEMS).document(item_id).get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        if data.get("uid") != uid:
            return None
        return ItemRecord.from_doc(data)

    def _list_items_sync(self, uid):
        from google.cloud.firestore_v1.base_query import FieldFilter

        q = self._get_db().collection(ITEMS).where(filter=FieldFilter("uid", "==", uid))
        return [ItemRecord.from_doc(d.to_dict()) for d in q.stream()]

    def _delete_item_sync(self, uid, item_id):
        ref = self._get_db().collection(ITEMS).document(item_id)
        doc = ref.get()
        if not doc.exists or doc.to_dict().get("uid") != uid:
            return False
        ref.delete()
        return True

    async def put_item(self, rec):
        await asyncio.to_thread(self._put_item_sync, rec)

    async def get_item(self, uid, item_id):
        return await asyncio.to_thread(self._get_item_sync, uid, item_id)

    async def list_items(self, uid):
        return await asyncio.to_thread(self._list_items_sync, uid)

    async def delete_item(self, uid, item_id):
        return await asyncio.to_thread(self._delete_item_sync, uid, item_id)

    # -- chunks --------------------------------------------------------------

    def _put_chunks_sync(self, chunks):
        from google.cloud.firestore_v1.vector import Vector

        db = self._get_db()
        for i in range(0, len(chunks), _BATCH_LIMIT):
            batch = db.batch()
            for c in chunks[i:i + _BATCH_LIMIT]:
                batch.set(db.collection(CHUNKS).document(c.id), {
                    "uid": c.uid, "item_id": c.item_id, "ord": c.ord,
                    "text": c.text, "embedding": Vector(c.embedding),
                })
            batch.commit()

    def _delete_query_sync(self, query):
        db = self._get_db()
        n = 0
        while True:
            docs = list(query.limit(_BATCH_LIMIT).stream())
            if not docs:
                return n
            batch = db.batch()
            for d in docs:
                batch.delete(d.reference)
            batch.commit()
            n += len(docs)

    def _chunks_for(self, uid, item_id=None):
        from google.cloud.firestore_v1.base_query import FieldFilter

        q = self._get_db().collection(CHUNKS).where(filter=FieldFilter("uid", "==", uid))
        if item_id is not None:
            q = q.where(filter=FieldFilter("item_id", "==", item_id))
        return q

    async def put_chunks(self, chunks):
        await asyncio.to_thread(self._put_chunks_sync, chunks)

    async def delete_chunks(self, uid, item_id):
        return await asyncio.to_thread(self._delete_query_sync, self._chunks_for(uid, item_id))

    async def delete_all_chunks(self, uid):
        return await asyncio.to_thread(self._delete_query_sync, self._chunks_for(uid))

    def _nearest_sync(self, uid, item_ids, vector, limit):
        from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
        from google.cloud.firestore_v1.vector import Vector

        results: list[tuple[ChunkRecord, float]] = []
        for item_id in item_ids:
            q = (
                self._chunks_for(uid, item_id)
                .find_nearest(
                    vector_field="embedding",
                    query_vector=Vector(vector),
                    distance_measure=DistanceMeasure.COSINE,
                    limit=limit,
                    distance_result_field="vector_distance",
                )
            )
            for d in q.stream():
                data = d.to_dict()
                results.append((
                    ChunkRecord(
                        id=d.id, uid=data["uid"], item_id=data["item_id"],
                        ord=int(data["ord"]), text=data["text"],
                    ),
                    float(data.get("vector_distance", 1.0)),
                ))
        results.sort(key=lambda t: (t[1], t[0].id))
        return results[:limit]

    async def nearest(self, uid, item_ids, vector, limit):
        if not item_ids:
            return []
        return await asyncio.to_thread(self._nearest_sync, uid, item_ids, vector, limit)


def get_store() -> "LibraryStore":
    project = os.environ.get("MINDSHIFT_FIRESTORE_PROJECT")
    if project:
        return FirestoreLibraryStore(project)
    return MemoryLibraryStore()
