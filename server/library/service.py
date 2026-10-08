"""LibraryService — the one object the router, DELETE /me and the coach use.

Lifecycle of an item:

1. **Validate + extract inline** (so limits answer 413/422 immediately and
   nothing over a limit is ever stored, let alone truncated).
2. **Store** the original upload and the extracted text in the blob tier and
   the metadata in the store, ``status="processing"``.
3. **Index in the background**: chunk (~800 tokens, overlap), embed, write
   vectors, then ``status="ready"``. With no embedder configured the item is
   ``ready`` immediately with ``indexed=False`` (full mode still works;
   retrieval reports itself unavailable). An embedding error marks it
   ``failed`` with the reason — its text still serves full mode.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

from library import blobs as blobs_mod
from library import context as ctx_mod
from library.chunking import chunk_text, estimate_tokens
from library.embeddings import EmbeddingUnavailable
from library.extract import CONTENT_TYPES, ExtractionError, UnsupportedType, extract_text
from library.models import (
    MAX_FILE_BYTES,
    MAX_ITEMS_PER_USER,
    MAX_TEXT_CHARS,
    MAX_TITLE_CHARS,
    ChunkRecord,
    ItemRecord,
    LibraryContext,
)

logger = logging.getLogger(__name__)

# How much of recent_text is embedded as the retrieval query (its tail — the
# latest turns matter most).
QUERY_TAIL_CHARS = 4000
MAX_RETRIEVED_CANDIDATES = 50


class LibraryError(Exception):
    status_code = 422

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class TooLarge(LibraryError):
    status_code = 413


class Invalid(LibraryError):
    status_code = 422


class LimitReached(LibraryError):
    status_code = 422


class NotFound(LibraryError):
    status_code = 404


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_title(title: str | None) -> str:
    t = " ".join((title or "").split())
    if not t:
        raise Invalid("title must not be empty")
    if len(t) > MAX_TITLE_CHARS:
        raise Invalid(f"title is longer than {MAX_TITLE_CHARS} characters")
    return t


def _check_text_size(text: str) -> None:
    if len(text) > MAX_TEXT_CHARS:
        raise TooLarge(
            f"text is {len(text):,} characters; the limit is {MAX_TEXT_CHARS:,} "
            "per item (nothing was saved — split it into smaller items)"
        )


class LibraryService:
    def __init__(self, store, blobs, embedder=None, *, embedder_reason: str | None = None) -> None:
        self.store = store
        self.blobs = blobs
        self.embedder = embedder
        self.unavailable_reason = None if embedder else (
            embedder_reason or "no embedding model configured"
        )
        self._tasks: set[asyncio.Task] = set()
        self._index_locks: dict[str, asyncio.Lock] = {}
        self._text_cache: OrderedDict[tuple[str, int], str] = OrderedDict()

    @property
    def retrieval_available(self) -> bool:
        return self.embedder is not None

    # -- create ----------------------------------------------------------------

    async def _check_capacity(self, uid: str) -> None:
        if len(await self.store.list_items(uid)) >= MAX_ITEMS_PER_USER:
            raise LimitReached(
                f"library is full ({MAX_ITEMS_PER_USER} items); delete one to add another"
            )

    def _new_record(self, uid, title, kind, text, **extra) -> ItemRecord:
        item_id = str(uuid.uuid4())
        now = _now()
        return ItemRecord(
            id=item_id, uid=uid, title=title, kind=kind, chars=len(text),
            status="processing", created_at=now, updated_at=now,
            text_key=blobs_mod.item_prefix(uid, item_id) + "text.txt", **extra,
        )

    async def create_note(self, uid: str, title: str, text: str) -> ItemRecord:
        title = _clean_title(title)
        text = (text or "").strip()
        if not text:
            raise Invalid("note text must not be empty")
        _check_text_size(text)
        await self._check_capacity(uid)
        rec = self._new_record(uid, title, "note", text)
        return await self._save_and_index(rec, text)

    async def create_document(
        self, uid: str, filename: str | None, content_type: str | None,
        data: bytes, title: str | None = None,
    ) -> ItemRecord:
        if len(data) > MAX_FILE_BYTES:
            raise TooLarge(
                f"file is {len(data):,} bytes; the limit is {MAX_FILE_BYTES:,} (10 MB)"
            )
        if not data:
            raise Invalid("file is empty")
        try:
            extracted = await asyncio.to_thread(extract_text, filename, content_type, data)
        except (UnsupportedType, ExtractionError) as exc:
            raise Invalid(str(exc)) from exc
        _check_text_size(extracted.text)
        await self._check_capacity(uid)
        stem = (filename or "").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        title = _clean_title(title or stem or "Untitled")
        rec = self._new_record(
            uid, title, extracted.kind, extracted.text,
            filename=(filename or None), content_type=CONTENT_TYPES.get(extracted.ext),
        )
        rec.original_key = blobs_mod.item_prefix(uid, rec.id) + "original" + extracted.ext
        await self.blobs.put(rec.original_key, data, rec.content_type or "application/octet-stream")
        return await self._save_and_index(rec, extracted.text)

    async def _save_and_index(self, rec: ItemRecord, text: str) -> ItemRecord:
        await self.blobs.put(rec.text_key, text.encode("utf-8"), "text/plain; charset=utf-8")
        if self.embedder is None:
            rec.status = "ready"
            rec.indexed = False
        await self.store.put_item(rec)
        self._cache_text(rec, text)
        if self.embedder is not None:
            self._spawn(self._index(rec.uid, rec.id, rec.version, text))
        return rec

    # -- indexing ----------------------------------------------------------------

    def _spawn(self, coro) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Await every in-flight indexing task (tests; graceful shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _index(self, uid: str, item_id: str, version: int, text: str) -> None:
        lock = self._index_locks.setdefault(item_id, asyncio.Lock())
        async with lock:
            try:
                rec = await self.store.get_item(uid, item_id)
                if rec is None or rec.version != version:
                    return  # deleted, or a newer edit has its own task queued
                pieces = chunk_text(text)
                try:
                    vectors = await self.embedder.embed(pieces, task="document")
                except EmbeddingUnavailable as exc:
                    rec.status, rec.error, rec.indexed = "failed", f"indexing failed: {exc}", False
                    rec.updated_at = _now()
                    await self.store.put_item(rec)
                    return
                await self.store.delete_chunks(uid, item_id)
                await self.store.put_chunks([
                    ChunkRecord(
                        id=f"{item_id}_{i:05d}", uid=uid, item_id=item_id, ord=i,
                        text=piece, embedding=list(vec),
                    )
                    for i, (piece, vec) in enumerate(zip(pieces, vectors))
                ])
                current = await self.store.get_item(uid, item_id)
                if current is None:
                    # Deleted while we embedded — don't leave orphan vectors.
                    await self.store.delete_chunks(uid, item_id)
                    return
                if current.version != version:
                    return
                current.status, current.error = "ready", None
                current.indexed, current.chunk_count = True, len(pieces)
                current.embedding_model = getattr(self.embedder, "name", None)
                current.updated_at = _now()
                await self.store.put_item(current)
            except Exception as exc:  # noqa: BLE001 — record, never hide
                logger.exception("library indexing failed uid=%s item=%s", uid, item_id)
                rec = await self.store.get_item(uid, item_id)
                if rec is not None and rec.version == version:
                    rec.status, rec.error = "failed", f"indexing failed ({type(exc).__name__})"
                    await self.store.put_item(rec)

    # -- read --------------------------------------------------------------------

    async def list_items(self, uid: str) -> list[ItemRecord]:
        items = await self.store.list_items(uid)
        items.sort(key=lambda r: (r.created_at, r.id))
        return items

    async def get_item(self, uid: str, item_id: str) -> ItemRecord | None:
        return await self.store.get_item(uid, item_id)

    def _cache_text(self, rec: ItemRecord, text: str) -> None:
        key = (rec.id, rec.version)
        self._text_cache[key] = text
        self._text_cache.move_to_end(key)
        while len(self._text_cache) > 64:
            self._text_cache.popitem(last=False)

    async def get_text(self, rec: ItemRecord) -> str:
        key = (rec.id, rec.version)
        if key in self._text_cache:
            self._text_cache.move_to_end(key)
            return self._text_cache[key]
        raw = await self.blobs.get(rec.text_key) if rec.text_key else None
        if raw is None:
            raise NotFound(f"text for item {rec.id} is missing from storage")
        text = raw.decode("utf-8")
        self._cache_text(rec, text)
        return text

    # -- update / delete -------------------------------------------------------------

    async def update_item(
        self, uid: str, item_id: str, *, title: str | None = None, text: str | None = None,
    ) -> ItemRecord:
        rec = await self.store.get_item(uid, item_id)
        if rec is None:
            raise NotFound("item not found")
        if title is not None:
            rec.title = _clean_title(title)
        if text is not None:
            if rec.kind != "note":
                raise Invalid("only a note's text can be edited; re-upload a document instead")
            text = text.strip()
            if not text:
                raise Invalid("note text must not be empty")
            _check_text_size(text)
            rec.version += 1
            rec.chars = len(text)
            rec.indexed, rec.chunk_count, rec.error = False, 0, None
            rec.status = "processing" if self.embedder else "ready"
            await self.blobs.put(rec.text_key, text.encode("utf-8"), "text/plain; charset=utf-8")
            if self.embedder is None:
                await self.store.delete_chunks(uid, item_id)
        rec.updated_at = _now()
        await self.store.put_item(rec)
        if text is not None:
            self._cache_text(rec, text)
            if self.embedder is not None:
                self._spawn(self._index(uid, item_id, rec.version, text))
        return rec

    async def delete_item(self, uid: str, item_id: str) -> bool:
        rec = await self.store.get_item(uid, item_id)
        if rec is None:
            return False
        # Blobs and vectors first, the metadata doc last: a failure part-way
        # leaves a visible item the user can delete again, never invisible data.
        await self.blobs.delete_prefix(blobs_mod.item_prefix(uid, item_id))
        await self.store.delete_chunks(uid, item_id)
        await self.store.delete_item(uid, item_id)
        for key in [k for k in self._text_cache if k[0] == item_id]:
            del self._text_cache[key]
        return True

    async def delete_all_for_user(self, uid: str) -> int:
        """Account deletion: every item, blob and vector for ``uid``. Returns
        the number of items removed. Sweeps by prefix / uid afterwards so
        anything orphaned by an earlier partial failure goes too."""
        items = await self.store.list_items(uid)
        for rec in items:
            await self.delete_item(uid, rec.id)
        await self.blobs.delete_prefix(blobs_mod.user_prefix(uid))
        await self.store.delete_all_chunks(uid)
        return len(items)

    # -- the coach hook ----------------------------------------------------------------

    async def build_context(
        self, uid: str, item_ids: list[str], recent_text: str, budget_tokens: int,
    ) -> LibraryContext:
        wanted = list(dict.fromkeys(i for i in item_ids if i))
        found: list[ItemRecord] = []
        missing: list[str] = []
        for item_id in wanted:
            rec = await self.store.get_item(uid, item_id)
            (found.append(rec) if rec is not None else missing.append(item_id))
        if not found:
            return LibraryContext(text="", mode="full", missing_item_ids=missing)
        # Stable order regardless of selection order → prompt-cache friendly.
        found.sort(key=lambda r: (r.created_at, r.id))

        with_text = [(rec, await self.get_text(rec)) for rec in found]
        full = ctx_mod.render_full(with_text)
        if estimate_tokens(full) <= budget_tokens:
            return LibraryContext(
                text=full, mode="full", item_ids=[r.id for r in found],
                est_tokens=estimate_tokens(full), missing_item_ids=missing,
            )

        if self.embedder is None:
            return LibraryContext(
                text="", mode="unavailable", missing_item_ids=missing,
                reason=f"selected material exceeds the budget and retrieval is unavailable: {self.unavailable_reason}",
            )
        indexed = [r for r in found if r.indexed]
        skipped = [r.id for r in found if not r.indexed]
        if not indexed:
            return LibraryContext(
                text="", mode="unavailable", missing_item_ids=missing,
                skipped_item_ids=skipped,
                reason="selected material exceeds the budget and none of it is indexed yet",
            )
        query = (recent_text or "").strip()[-QUERY_TAIL_CHARS:] or " ".join(r.title for r in found)
        try:
            [qvec] = await self.embedder.embed([query], task="query")
        except EmbeddingUnavailable as exc:
            return LibraryContext(
                text="", mode="unavailable", missing_item_ids=missing,
                skipped_item_ids=skipped, reason=f"retrieval failed: {exc}",
            )
        k = min(MAX_RETRIEVED_CANDIDATES, max(4, 2 * budget_tokens // 800 + 4))
        candidates = await self.store.nearest(uid, [r.id for r in indexed], qvec, k)

        by_id = {r.id: r for r in indexed}
        order = {r.id: n for n, r in enumerate(indexed)}
        chosen: list = []
        text = ""
        for chunk, _dist in candidates:
            trial = chosen + [chunk]
            rendered = self._render_chunks(trial, by_id, order)
            if estimate_tokens(rendered) <= budget_tokens:
                chosen, text = trial, rendered
        chosen.sort(key=lambda c: (order[c.item_id], c.ord))
        return LibraryContext(
            text=text, mode="retrieved",
            item_ids=list(dict.fromkeys(c.item_id for c in chosen)),
            chunk_ids=[c.id for c in chosen], est_tokens=estimate_tokens(text),
            missing_item_ids=missing, skipped_item_ids=skipped,
            reason=None if chosen else "no excerpt fits the token budget",
        )

    @staticmethod
    def _render_chunks(chunks, by_id, order) -> str:
        groups: dict[str, list] = {}
        for c in sorted(chunks, key=lambda c: (order[c.item_id], c.ord)):
            groups.setdefault(c.item_id, []).append((c.ord, c.text))
        return ctx_mod.render_excerpts([(by_id[i], ex) for i, ex in groups.items()])


def build_from_env() -> LibraryService:
    from library.embeddings import get_embedder
    from library.store import get_store

    embedder, reason = get_embedder()
    return LibraryService(get_store(), blobs_mod.get_blobs(), embedder, embedder_reason=reason)
