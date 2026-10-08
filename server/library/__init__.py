"""Per-user coach knowledge library: saved notes, uploaded documents (used
whole when small, prompt-cache friendly) and vector retrieval when the
selected material is too big for the prompt.

The coach's single entry point is :func:`build_library_context`. The REST
surface is ``server/routers/library.py``; account deletion calls
:meth:`LibraryService.delete_all_for_user` via ``account_deletion``.
"""

from __future__ import annotations

from library.models import LibraryContext

__all__ = ["LibraryContext", "build_library_context", "get_service", "set_service"]

_service = None


def get_service():
    """The process-wide :class:`library.service.LibraryService`, built from env
    on first use (Firestore + GCS + Vertex in prod; memory and no embedder in
    keyless dev/CI)."""
    global _service
    if _service is None:
        from library.service import build_from_env

        _service = build_from_env()
    return _service


def set_service(service) -> None:
    """Install a service (tests) or ``None`` to rebuild from env next time."""
    global _service
    _service = service


async def build_library_context(
    uid: str, item_ids: list[str], recent_text: str, budget_tokens: int,
) -> LibraryContext:
    """The library block for one coach prompt.

    If the selected items' full text (wrapped) fits ``budget_tokens`` it is
    returned whole, in a stable order (``mode="full"`` — byte-identical across
    calls for the same selection, so a cached prompt prefix keeps hitting).
    Otherwise ``recent_text`` is embedded and the nearest chunks that fit are
    returned (``mode="retrieved"``). When retrieval cannot run,
    ``mode="unavailable"`` with ``text=""`` and a ``reason`` — never a
    silently truncated document. Items not owned by ``uid`` are ignored and
    listed in ``missing_item_ids``.
    """
    return await get_service().build_context(uid, item_ids, recent_text, budget_tokens)
