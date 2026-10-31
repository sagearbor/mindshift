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
import json
import logging
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

from library import blobs as blobs_mod
from library import context as ctx_mod
from library import tables as tables_mod
from library.chunking import chunk_text, estimate_tokens
from library.embeddings import EmbeddingUnavailable
from library.extract import CONTENT_TYPES, ExtractionError, UnsupportedType, extract_text
from library.models import (
    MAX_FILE_BYTES,
    MAX_PROJECT_DEPTH,
    MAX_PROJECT_NAME_CHARS,
    MAX_PROJECTS_PER_USER,
    MAX_ITEMS_PER_USER,
    MAX_TEXT_CHARS,
    MAX_TITLE_CHARS,
    ChunkRecord,
    ItemRecord,
    LibraryContext,
    LibraryFacts,
    ProjectRecord,
)

logger = logging.getLogger(__name__)

# How much of recent_text is embedded as the retrieval query (its tail — the
# latest turns matter most).
QUERY_TAIL_CHARS = 4000
MAX_RETRIEVED_CANDIDATES = 50
# The typed table JSON stays within the same 2 MB per-item limit as the text
# (it is usually SMALLER than the labelled row text). Over it, the item is
# still saved and usable as text; only exact lookups are skipped.
MAX_TABLE_JSON_BYTES = MAX_TEXT_CHARS
# How long build_facts reuses the selected items' metadata (one store read
# per item otherwise, every turn). Same window as live.FULL_CACHE_TTL_S.
FACTS_RECORDS_TTL_S = 60.0


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


class Conflict(LibraryError):
    status_code = 409


_UNSET = object()


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
        self._table_cache: OrderedDict[tuple[str, int], tables_mod.TableSet] = OrderedDict()
        self._facts_recs: dict[tuple[str, str], tuple[float, ItemRecord | None]] = {}

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

    async def create_note(
        self, uid: str, title: str, text: str, project_id: str | None = None,
    ) -> ItemRecord:
        title = _clean_title(title)
        text = (text or "").strip()
        if not text:
            raise Invalid("note text must not be empty")
        _check_text_size(text)
        await self._check_capacity(uid)
        await self._require_project(uid, project_id)
        rec = self._new_record(uid, title, "note", text, project_id=project_id or None)
        return await self._save_and_index(rec, text)

    async def create_document(
        self, uid: str, filename: str | None, content_type: str | None,
        data: bytes, title: str | None = None, project_id: str | None = None,
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
        await self._require_project(uid, project_id)
        stem = (filename or "").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        title = _clean_title(title or stem or "Untitled")
        rec = self._new_record(
            uid, title, extracted.kind, extracted.text,
            filename=(filename or None), content_type=CONTENT_TYPES.get(extracted.ext),
            project_id=project_id or None,
        )
        rec.original_key = blobs_mod.item_prefix(uid, rec.id) + "original" + extracted.ext
        await self.blobs.put(rec.original_key, data, rec.content_type or "application/octet-stream")
        if extracted.table is not None:
            await self._save_table(rec, extracted.table)
        return await self._save_and_index(rec, extracted.text)

    async def _save_table(self, rec: ItemRecord, table: dict) -> None:
        blob = json.dumps(table, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(blob) > MAX_TABLE_JSON_BYTES:
            logger.warning(
                "library table form skipped (%d bytes > %d) item=%s; text still saved",
                len(blob), MAX_TABLE_JSON_BYTES, rec.id,
            )
            return
        rec.table_key = blobs_mod.item_prefix(rec.uid, rec.id) + "table.json"
        await self.blobs.put(rec.table_key, blob, "application/json")

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

    # -- projects (folders) ---------------------------------------------------------

    async def _require_project(self, uid: str, project_id: str | None) -> ProjectRecord | None:
        if not project_id:
            return None
        proj = await self.store.get_project(uid, project_id)
        if proj is None:
            raise NotFound("project not found")
        return proj

    @staticmethod
    def _depth(by_id: dict[str, ProjectRecord], project_id: str | None) -> int:
        """Depth of ``project_id`` (top level = 1; None = 0). Tolerates a
        corrupt parent loop by stopping at the cap."""
        depth, cur = 0, project_id
        while cur is not None and depth <= MAX_PROJECT_DEPTH + 1:
            rec = by_id.get(cur)
            if rec is None:
                break
            depth += 1
            cur = rec.parent_id
        return depth

    @staticmethod
    def _height(children: dict[str | None, list[str]], project_id: str) -> int:
        """Levels in the subtree rooted at ``project_id`` (a leaf = 1)."""
        best, stack, seen = 1, [(project_id, 1)], {project_id}
        while stack:
            pid, h = stack.pop()
            best = max(best, h)
            for c in children.get(pid, []):
                if c not in seen:
                    seen.add(c)
                    stack.append((c, h + 1))
        return best

    @staticmethod
    def _clean_project_name(name: str | None) -> str:
        n = " ".join((name or "").split())
        if not n:
            raise Invalid("project name must not be empty")
        if len(n) > MAX_PROJECT_NAME_CHARS:
            raise Invalid(f"project name is longer than {MAX_PROJECT_NAME_CHARS} characters")
        return n

    async def list_projects(self, uid: str) -> list[ProjectRecord]:
        projects = await self.store.list_projects(uid)
        projects.sort(key=lambda p: (p.created_at, p.id))
        return projects

    async def create_project(
        self, uid: str, name: str, parent_id: str | None = None,
    ) -> ProjectRecord:
        name = self._clean_project_name(name)
        projects = await self.store.list_projects(uid)
        if len(projects) >= MAX_PROJECTS_PER_USER:
            raise LimitReached(f"you have {MAX_PROJECTS_PER_USER} projects; delete one to add another")
        by_id = {p.id: p for p in projects}
        if parent_id and parent_id not in by_id:
            raise NotFound("parent project not found")
        if self._depth(by_id, parent_id) + 1 > MAX_PROJECT_DEPTH:
            raise Invalid(f"projects can be nested at most {MAX_PROJECT_DEPTH} levels deep")
        now = _now()
        rec = ProjectRecord(
            id=str(uuid.uuid4()), uid=uid, name=name, parent_id=parent_id or None,
            created_at=now, updated_at=now,
        )
        await self.store.put_project(rec)
        return rec

    async def update_project(
        self, uid: str, project_id: str, *, name: str | None = None, parent_id=_UNSET,
    ) -> ProjectRecord:
        projects = await self.store.list_projects(uid)
        by_id = {p.id: p for p in projects}
        rec = by_id.get(project_id)
        if rec is None:
            raise NotFound("project not found")
        if name is not None:
            rec.name = self._clean_project_name(name)
        if parent_id is not _UNSET:
            new_parent = parent_id or None
            if new_parent is not None and new_parent not in by_id:
                raise NotFound("parent project not found")
            # Cycle: the new parent is this project or anywhere beneath it.
            cur, hops = new_parent, 0
            while cur is not None and hops <= MAX_PROJECT_DEPTH + len(by_id):
                if cur == project_id:
                    raise Invalid(
                        "cannot move a project into itself or one of its own sub-projects (cycle)"
                    )
                cur = by_id[cur].parent_id if cur in by_id else None
                hops += 1
            children: dict[str | None, list[str]] = {}
            for p in projects:
                children.setdefault(p.parent_id, []).append(p.id)
            if self._depth(by_id, new_parent) + self._height(children, project_id) > MAX_PROJECT_DEPTH:
                raise Invalid(
                    f"moving it there would nest projects deeper than {MAX_PROJECT_DEPTH} levels"
                )
            rec.parent_id = new_parent
        rec.updated_at = _now()
        await self.store.put_project(rec)
        return rec

    async def delete_project(
        self, uid: str, project_id: str, *, contents: str | None = None,
    ) -> bool:
        """Delete a project. SAFE by default: a project that still holds files
        or sub-projects is refused (409). Pass ``contents="move_to_parent"``
        to move them up one level (to top level when this project is top
        level) and then delete it. Files are never deleted along with a
        project."""
        if contents not in (None, "move_to_parent"):
            raise Invalid("contents must be 'move_to_parent' or omitted")
        projects = await self.store.list_projects(uid)
        rec = next((p for p in projects if p.id == project_id), None)
        if rec is None:
            return False
        subs = [p for p in projects if p.parent_id == project_id]
        items = [i for i in await self.store.list_items(uid) if i.project_id == project_id]
        if (subs or items) and contents != "move_to_parent":
            raise Conflict(
                f"project is not empty ({len(items)} files, {len(subs)} sub-projects); "
                "retry with contents=move_to_parent to move them up one level"
            )
        now = _now()
        for p in subs:
            p.parent_id, p.updated_at = rec.parent_id, now
            await self.store.put_project(p)
        for i in items:
            i.project_id, i.updated_at = rec.parent_id, now
            await self.store.put_item(i)
        await self.store.delete_project(uid, project_id)
        return True

    # -- update / delete -------------------------------------------------------------

    async def update_item(
        self, uid: str, item_id: str, *, title: str | None = None, text: str | None = None,
        project_id=_UNSET,
    ) -> ItemRecord:
        rec = await self.store.get_item(uid, item_id)
        if rec is None:
            raise NotFound("item not found")
        if project_id is not _UNSET:
            await self._require_project(uid, project_id)
            rec.project_id = project_id or None
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
        for key in [k for k in self._table_cache if k[0] == item_id]:
            del self._table_cache[key]
        self._facts_recs.pop((uid, item_id), None)
        return True

    async def delete_all_for_user(self, uid: str) -> int:
        """Account deletion: every item, blob and vector for ``uid``. Returns
        the number of items removed. Sweeps by prefix / uid afterwards so
        anything orphaned by an earlier partial failure goes too."""
        items = await self.store.list_items(uid)
        for rec in items:
            await self.delete_item(uid, rec.id)
        for proj in await self.store.list_projects(uid):
            await self.store.delete_project(uid, proj.id)
        await self.blobs.delete_prefix(blobs_mod.user_prefix(uid))
        await self.store.delete_all_chunks(uid)
        return len(items)

    # -- exact table lookups ----------------------------------------------------------

    async def _facts_record(self, uid: str, item_id: str) -> ItemRecord | None:
        key = (uid, item_id)
        hit = self._facts_recs.get(key)
        if hit is not None and time.monotonic() - hit[0] < FACTS_RECORDS_TTL_S:
            return hit[1]
        rec = await self.store.get_item(uid, item_id)
        if len(self._facts_recs) > 512:
            self._facts_recs.clear()
        self._facts_recs[key] = (time.monotonic(), rec)
        return rec

    async def get_table(self, rec: ItemRecord) -> tables_mod.TableSet | None:
        """The loaded, indexed table of a table item (None when it has no
        stored table form). Cached by (id, version)."""
        if not rec.table_key:
            return None
        key = (rec.id, rec.version)
        if key in self._table_cache:
            self._table_cache.move_to_end(key)
            return self._table_cache[key]
        raw = await self.blobs.get(rec.table_key)
        if raw is None:
            return None
        ts = await asyncio.to_thread(lambda: tables_mod.TableSet.from_json(json.loads(raw)))
        self._table_cache[key] = ts
        while len(self._table_cache) > 32:
            self._table_cache.popitem(last=False)
        return ts

    async def build_facts(self, uid: str, item_ids: list[str], recent_text: str) -> LibraryFacts:
        wanted = list(dict.fromkeys(i for i in item_ids if i))
        recs = await asyncio.gather(*(self._facts_record(uid, i) for i in wanted))
        table_recs = sorted(
            (r for r in recs if r is not None and r.kind == "table" and r.table_key),
            key=lambda r: (r.created_at, r.id),
        )
        if not table_recs:
            return LibraryFacts(has_tables=False)
        sources = []
        for rec in table_recs:
            ts = await self.get_table(rec)
            if ts is not None and ts.sheets:
                sources.append((tables_mod.Source(rec.id, rec.title), ts))
        if not sources or not (recent_text or "").strip():
            return LibraryFacts(has_tables=bool(sources))
        facts = await asyncio.to_thread(tables_mod.answer, sources, recent_text)
        return LibraryFacts(
            text=tables_mod.render_facts(facts),
            item_ids=list(dict.fromkeys(f.item_id for f in facts)),
            has_tables=True,
        )

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
