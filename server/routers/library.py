"""Coach knowledge library — ``/library/items``.

A user saves material once (typed notes, uploaded PDF / .txt / .md / .docx /
.csv / .xlsx) and selects it for sessions so the earpiece coach can draw on
it. The logic lives in ``server/library/``; this file is the HTTP shape.

* ``POST   /library/items``       JSON ``{kind:"note", title, text}`` OR a
  multipart upload (``file`` + optional ``title``) → 201 ``LibraryItem``
* ``GET    /library/items``       ``{items, retrieval_available, limits}``
* ``GET    /library/items/{id}``  ``LibraryItemDetail`` (adds a text preview;
  ``?full=1`` also returns the whole text, capped at the 2 MB item limit, so
  the phone can edit a note longer than the preview)
* ``PATCH  /library/items/{id}``  ``{title?, text?}`` (text: notes only)
* ``DELETE /library/items/{id}``  removes the GCS blobs, the Firestore doc
  and every chunk/vector

Projects (folders, nestable to 10 levels; no project_id = top level):

* ``POST   /library/projects``        ``{name, parent_id?}`` -> 201 project
* ``PATCH  /library/projects/{id}``   ``{name?, parent_id?}`` rename / move
  (``parent_id: null`` = top level; cycles and >10 levels -> 422)
* ``DELETE /library/projects/{id}``   a non-empty project is REFUSED (409);
  ``?contents=move_to_parent`` moves its files and sub-projects up one level
  first. Files are never deleted with a project.
* ``PATCH /library/items/{id}`` ``{project_id}`` moves a file (null = top level);
  uploads/notes accept an optional ``project_id``.

Strictly per-user: the uid comes from the verified token only, and another
user's id is a 404 (no existence oracle). Limits answer 413 (10 MB file,
2 MB extracted text) or 422 (50 items, unsupported/unreadable/empty input);
nothing over a limit is stored or truncated.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

import library
import plans
from auth import get_current_uid
from library import models
from library.service import _UNSET, LibraryError, NotFound

router = APIRouter(prefix="/library", tags=["library"])

_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
# Multipart framing overhead allowed on top of the 10 MB file before the
# Content-Length precheck refuses the request outright.
_MULTIPART_SLACK = 64 * 1024


async def _rate_limit(request: Request) -> None:
    """main's per-IP limiter, imported lazily (main includes this router)."""
    import main

    await main._rate_limit(request)


class LibraryItem(BaseModel):
    id: str
    title: str
    kind: Literal["note", "document", "table"]
    chars: int
    status: Literal["processing", "ready", "failed"]
    error: Optional[str] = None
    created_at: str
    updated_at: str
    indexed: bool = Field(description="Chunks + vectors exist (retrieval can use this item).")
    project_id: Optional[str] = None


class LibraryItemDetail(LibraryItem):
    filename: Optional[str] = None
    content_type: Optional[str] = None
    chunk_count: int = 0
    preview: str
    preview_truncated: bool
    text: Optional[str] = Field(
        default=None,
        description="The whole text (only with ?full=1; capped at max_text_chars).",
    )


class LibraryLimits(BaseModel):
    max_file_bytes: int
    max_text_chars: int
    max_items: int


class LibraryProject(BaseModel):
    id: str
    name: str
    parent_id: Optional[str] = None
    created_at: str
    updated_at: str


class ProjectCreate(BaseModel):
    name: str
    parent_id: Optional[str] = None


class ProjectPatch(BaseModel):
    name: Optional[str] = None
    parent_id: Optional[str] = None


class LibraryList(BaseModel):
    items: list[LibraryItem]
    projects: list[LibraryProject] = []
    retrieval_available: bool
    limits: LibraryLimits


class NoteCreate(BaseModel):
    kind: Literal["note"]
    title: str
    text: str
    project_id: Optional[str] = None


class ItemPatch(BaseModel):
    title: Optional[str] = None
    text: Optional[str] = None
    project_id: Optional[str] = None  # explicit null = move to top level


class Deleted(BaseModel):
    deleted: bool
    id: str


def _declared_size(request: Request) -> int:
    length = request.headers.get("content-length")
    return int(length) if length and length.isdigit() else 0


def _item(rec) -> LibraryItem:
    return LibraryItem(
        id=rec.id, title=rec.title, kind=rec.kind, chars=rec.chars, status=rec.status,
        error=rec.error, created_at=rec.created_at, updated_at=rec.updated_at,
        indexed=rec.indexed, project_id=rec.project_id,
    )


def _project(rec) -> LibraryProject:
    return LibraryProject(
        id=rec.id, name=rec.name, parent_id=rec.parent_id,
        created_at=rec.created_at, updated_at=rec.updated_at,
    )


def _raise(exc: LibraryError):
    raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


def _check_id(item_id: str, detail: str = "item not found") -> None:
    if not _ID.match(item_id):
        raise HTTPException(status_code=404, detail=detail)


@router.post("/items", status_code=201, response_model=LibraryItem)
async def create_item(
    request: Request,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
) -> LibraryItem:
    svc = library.get_service()
    ctype = (request.headers.get("content-type") or "").lower()
    # Plan gate (server/plans.py): blocks creating NEW items over the user's
    # plan caps; never touches existing ones. Raises HTTP 429 `plan_limit`.
    await plans.check_library_create(uid, svc, _declared_size(request))
    try:
        if ctype.startswith("multipart/form-data"):
            length = request.headers.get("content-length")
            if length and length.isdigit() and int(length) > models.MAX_FILE_BYTES + _MULTIPART_SLACK:
                raise HTTPException(
                    status_code=413,
                    detail=f"file is larger than the {models.MAX_FILE_BYTES:,}-byte (10 MB) limit",
                )
            form = await request.form()
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(status_code=422, detail="multipart upload needs a 'file' field")
            data = await upload.read(models.MAX_FILE_BYTES + 1)
            title = form.get("title")
            project_id = form.get("project_id")
            if isinstance(project_id, str) and project_id.strip():
                _check_id(project_id.strip(), "project not found")
            rec = await svc.create_document(
                uid, upload.filename, upload.content_type, data,
                title=title if isinstance(title, str) and title.strip() else None,
                project_id=project_id.strip() if isinstance(project_id, str) and project_id.strip() else None,
            )
        else:
            try:
                note = NoteCreate.model_validate(await request.json())
            except (ValidationError, ValueError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail='expected JSON {"kind": "note", "title": ..., "text": ...} or a multipart file upload',
                ) from exc
            if note.project_id:
                _check_id(note.project_id, "project not found")
            rec = await svc.create_note(uid, note.title, note.text, note.project_id)
    except LibraryError as exc:
        _raise(exc)
    return _item(rec)


@router.get("/items", response_model=LibraryList)
async def list_items(uid: str = Depends(get_current_uid)) -> LibraryList:
    svc = library.get_service()
    return LibraryList(
        items=[_item(r) for r in await svc.list_items(uid)],
        projects=[_project(p) for p in await svc.list_projects(uid)],
        retrieval_available=svc.retrieval_available,
        limits=LibraryLimits(
            max_file_bytes=models.MAX_FILE_BYTES,
            max_text_chars=models.MAX_TEXT_CHARS,
            max_items=models.MAX_ITEMS_PER_USER,
        ),
    )


@router.get("/items/{item_id}", response_model=LibraryItemDetail)
async def get_item(
    item_id: str, full: bool = False, uid: str = Depends(get_current_uid),
) -> LibraryItemDetail:
    _check_id(item_id)
    svc = library.get_service()
    rec = await svc.get_item(uid, item_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="item not found")
    try:
        text = await svc.get_text(rec)
    except NotFound as exc:
        _raise(exc)
    base = _item(rec).model_dump()
    return LibraryItemDetail(
        **base, filename=rec.filename, content_type=rec.content_type,
        chunk_count=rec.chunk_count, preview=text[: models.PREVIEW_CHARS],
        preview_truncated=len(text) > models.PREVIEW_CHARS,
        # Owner-only by construction: rec was looked up by the caller's uid.
        text=text[: models.MAX_TEXT_CHARS] if full else None,
    )


@router.patch("/items/{item_id}", response_model=LibraryItem)
async def patch_item(
    item_id: str, body: ItemPatch,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
) -> LibraryItem:
    _check_id(item_id)
    move = "project_id" in body.model_fields_set
    if body.title is None and body.text is None and not move:
        raise HTTPException(status_code=422, detail="nothing to update: send title, text and/or project_id")
    if move and body.project_id:
        _check_id(body.project_id, "project not found")
    try:
        rec = await library.get_service().update_item(
            uid, item_id, title=body.title, text=body.text,
            project_id=body.project_id if move else _UNSET,
        )
    except LibraryError as exc:
        _raise(exc)
    return _item(rec)


@router.delete("/items/{item_id}", response_model=Deleted)
async def delete_item(item_id: str, uid: str = Depends(get_current_uid)) -> Deleted:
    _check_id(item_id)
    if not await library.get_service().delete_item(uid, item_id):
        raise HTTPException(status_code=404, detail="item not found")
    return Deleted(deleted=True, id=item_id)


# -- projects (folders) -------------------------------------------------------

@router.post("/projects", status_code=201, response_model=LibraryProject)
async def create_project(
    body: ProjectCreate,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
) -> LibraryProject:
    if body.parent_id:
        _check_id(body.parent_id, "parent project not found")
    try:
        return _project(await library.get_service().create_project(uid, body.name, body.parent_id))
    except LibraryError as exc:
        _raise(exc)


@router.patch("/projects/{project_id}", response_model=LibraryProject)
async def patch_project(
    project_id: str, body: ProjectPatch,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
) -> LibraryProject:
    _check_id(project_id, "project not found")
    move = "parent_id" in body.model_fields_set
    if body.name is None and not move:
        raise HTTPException(status_code=422, detail="nothing to update: send name and/or parent_id")
    if move and body.parent_id:
        _check_id(body.parent_id, "parent project not found")
    try:
        rec = await library.get_service().update_project(
            uid, project_id, name=body.name,
            parent_id=body.parent_id if move else _UNSET,
        )
    except LibraryError as exc:
        _raise(exc)
    return _project(rec)


@router.delete("/projects/{project_id}", response_model=Deleted)
async def delete_project(
    project_id: str, contents: Optional[str] = None,
    uid: str = Depends(get_current_uid),
) -> Deleted:
    _check_id(project_id, "project not found")
    try:
        ok = await library.get_service().delete_project(uid, project_id, contents=contents)
    except LibraryError as exc:
        _raise(exc)
    if not ok:
        raise HTTPException(status_code=404, detail="project not found")
    return Deleted(deleted=True, id=project_id)
