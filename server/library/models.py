"""Records and limits for the coach knowledge library."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Kind = Literal["note", "document", "table"]
Status = Literal["processing", "ready", "failed"]
Mode = Literal["full", "retrieved", "unavailable"]

MAX_FILE_BYTES = 10 * 1024 * 1024        # 10 MB per uploaded file
MAX_TEXT_CHARS = 2 * 1024 * 1024         # 2 MB of extracted text per item
MAX_ITEMS_PER_USER = 50
MAX_TITLE_CHARS = 200
PREVIEW_CHARS = 2000


@dataclass
class ItemRecord:
    """One library item (Firestore ``library_items/{id}``). The text itself is
    in the blob tier (see blobs.py); this is the metadata only."""

    id: str
    uid: str
    title: str
    kind: str
    chars: int
    status: str
    created_at: str
    updated_at: str
    version: int = 1
    error: str | None = None
    filename: str | None = None
    content_type: str | None = None
    original_key: str | None = None
    text_key: str | None = None
    indexed: bool = False
    chunk_count: int = 0
    embedding_model: str | None = None

    def to_doc(self) -> dict:
        return asdict(self)

    @classmethod
    def from_doc(cls, doc: dict) -> "ItemRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in doc.items() if k in known})


@dataclass
class ChunkRecord:
    """One embedded chunk (Firestore ``library_chunks/{item_id}_{ord}``)."""

    id: str
    uid: str
    item_id: str
    ord: int
    text: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class LibraryContext:
    """What :func:`library.build_library_context` hands the coach prompt.

    ``text``      the delimited block to place in the prompt ("" when nothing
                  fits or retrieval is unavailable).
    ``mode``      ``full`` (every selected item verbatim, stable order — safe
                  to prompt-cache), ``retrieved`` (top chunks for
                  ``recent_text`` within budget), or ``unavailable`` (too big
                  for full and retrieval cannot run; ``reason`` says why).
    ``item_ids``  items that contributed text, in output order.
    ``chunk_ids`` chunks used (retrieved mode only).
    ``missing_item_ids`` requested ids that are not this user's / do not exist.
    ``skipped_item_ids`` selected items that could not contribute in
                  retrieved mode because they are not (yet) indexed.
    """

    text: str
    mode: str
    item_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    est_tokens: int = 0
    reason: str | None = None
    missing_item_ids: list[str] = field(default_factory=list)
    skipped_item_ids: list[str] = field(default_factory=list)
