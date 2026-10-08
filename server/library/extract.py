"""Turn an uploaded file into plain text for the coach knowledge library.

Supported: PDF (``pypdf``), .txt / .md (UTF-8, latin-1 fallback), .docx (the
OOXML package read with the stdlib — no python-docx dependency), .csv and
.xlsx (``openpyxl``). Spreadsheets become HEADER-LABELLED ROW TEXT
("Plan: Growth; Seats: 25; Price: 199") and are marked ``kind="table"`` so the
coach can read a row without knowing column positions.

Tables ALSO come back in a typed, queryable form (``Extracted.table``, see
library/tables.py) built from the very same rows, so the coach can be handed
exact computed answers ("Q3 Northeast revenue = $225,500, rows 5-7") instead
of guessing from retrieved row text.

Nothing here truncates. A file that yields no text, or that its parser cannot
read, raises :class:`ExtractionError`; a file type we do not handle raises
:class:`UnsupportedType`. The router turns both into 422s.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from xml.etree import ElementTree

DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".docx"}
TABLE_EXTENSIONS = {".csv", ".xlsx"}
SUPPORTED_EXTENSIONS = DOCUMENT_EXTENSIONS | TABLE_EXTENSIONS

# Fallback when a client sends no useful extension.
_MIME_TO_EXT = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/csv": ".csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}

CONTENT_TYPES = {ext: mime for mime, ext in _MIME_TO_EXT.items()}
CONTENT_TYPES[".markdown"] = "text/markdown"


class UnsupportedType(ValueError):
    """The file is not one of the supported formats."""


class ExtractionError(ValueError):
    """The file is a supported type but no usable text could be read from it."""


@dataclass(frozen=True)
class Extracted:
    kind: str  # "document" | "table"
    text: str
    ext: str
    table: dict | None = None  # typed rows + schema (tables only)


def resolve_extension(filename: str | None, content_type: str | None) -> str:
    ext = PurePosixPath(filename or "").suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return ext
    mime = (content_type or "").split(";")[0].strip().lower()
    if not ext and mime in _MIME_TO_EXT:
        return _MIME_TO_EXT[mime]
    raise UnsupportedType(
        f"unsupported file type {ext or mime or 'unknown'!r}; supported: "
        + ", ".join(sorted(SUPPORTED_EXTENSIONS))
    )


def extract_text(filename: str | None, content_type: str | None, data: bytes) -> Extracted:
    ext = resolve_extension(filename, content_type)
    table = None
    try:
        if ext == ".pdf":
            text, kind = _pdf(data), "document"
        elif ext == ".docx":
            text, kind = _docx(data), "document"
        elif ext == ".csv":
            (text, table), kind = _csv(data), "table"
        elif ext == ".xlsx":
            (text, table), kind = _xlsx(data), "table"
        else:
            text, kind = _decode(data), "document"
    except (UnsupportedType, ExtractionError):
        raise
    except Exception as exc:  # noqa: BLE001 — any parser failure is a 422, not a 500
        raise ExtractionError(f"could not read this {ext} file ({type(exc).__name__})") from exc
    text = _normalize(text)
    if not text.strip():
        raise ExtractionError(f"no readable text found in this {ext} file")
    return Extracted(kind=kind, text=text, ext=ext, table=table)


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return text.strip()


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    return "\n\n".join(p for p in pages if p)


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    paragraphs = []
    for p in root.iter(f"{_W}p"):
        parts = []
        for node in p.iter():
            if node.tag == f"{_W}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{_W}tab":
                parts.append("\t")
            elif node.tag in (f"{_W}br", f"{_W}cr"):
                parts.append("\n")
        line = "".join(parts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def rows_to_text(rows: list[list[str]]) -> str:
    """Header-labelled row text: the first non-empty row is the header."""
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    header = rows[0]
    out = []
    for n, row in enumerate(rows[1:], start=1):
        pairs = []
        for i, value in enumerate(row):
            if not value:
                continue
            label = header[i] if i < len(header) and header[i] else f"Column {i + 1}"
            pairs.append(f"{label}: {value}")
        if pairs:
            out.append(f"Row {n}: " + "; ".join(pairs))
    if not out:  # header only — still say what the columns are
        out.append("Columns: " + "; ".join(h for h in header if h))
    return "\n".join(out)


def _csv(data: bytes) -> tuple[str, dict | None]:
    text = _decode(data)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [[_cell(c) for c in row] for row in csv.reader(io.StringIO(text), dialect)]
    return rows_to_text(rows), _table([(None, rows, {})])


def _table(sheets) -> dict | None:
    """The typed table (library/tables.py); a failure here never fails the
    upload — the item still works as text."""
    from library import tables

    try:
        return tables.build_table(sheets)
    except Exception:  # noqa: BLE001
        return None


def _currency_of(number_format: str | None) -> str | None:
    for sym in "$€£¥":
        if number_format and sym in number_format:
            return sym
    return None


def _xlsx(data: bytes) -> tuple[str, dict | None]:
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        sheets = []
        parsed = []
        for ws in wb.worksheets:
            rows: list[list[str]] = []
            units: dict[int, str] = {}
            for row in ws.iter_rows():
                vals = []
                for i, c in enumerate(row):
                    value = getattr(c, "value", None)
                    vals.append(_cell(value))
                    if isinstance(value, (int, float)) and i not in units:
                        sym = _currency_of(getattr(c, "number_format", None))
                        if sym:
                            units[i] = sym
                rows.append(vals)
            body = rows_to_text(rows)
            if body:
                sheets.append(f"Sheet: {ws.title}\n{body}")
                parsed.append((ws.title, rows, units))
        return "\n\n".join(sheets), _table(parsed)
    finally:
        wb.close()
