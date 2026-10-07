"""Text extraction + chunking for the coach knowledge library (server/library/).

Every fixture document is BUILT in the test (reportlab for the PDF, openpyxl
for the workbook, a hand-zipped OOXML package for the .docx) so nothing binary
is checked in and every format is exercised through its real parser.
"""

import io
import zipfile

import pytest

from library import chunking, extract


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _pdf_bytes(lines):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 720
    for line in lines:
        c.drawString(72, y, line)
        y -= 18
    c.showPage()
    c.drawString(72, 720, "Second page closing line")
    c.save()
    return buf.getvalue()


def _docx_bytes(paragraphs):
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(
        f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>' for p in paragraphs
    )
    # A table cell, to prove text inside tables is not dropped.
    body += (
        "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cell alpha</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>Cell beta</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
    )
    document = f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{ns}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", document)
    return buf.getvalue()


def _xlsx_bytes():
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Pricing"
    ws.append(["Plan", "Seats", "Price"])
    ws.append(["Starter", 5, 49])
    ws.append(["Growth", 25, 199])
    ws.append([None, None, None])  # blank row is skipped, not emitted
    other = wb.create_sheet("Contacts")
    other.append(["Name", "Role"])
    other.append(["Dana", "CFO"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

def test_plain_text_and_markdown_are_documents():
    r = extract.extract_text("notes.txt", "text/plain", "Hello wörld\nline two".encode())
    assert r.kind == "document"
    assert r.text == "Hello wörld\nline two"
    r = extract.extract_text("pitch.md", None, b"# Pitch\n\n- point one")
    assert r.kind == "document" and "point one" in r.text


def test_text_in_a_legacy_encoding_still_decodes():
    r = extract.extract_text("old.txt", "text/plain", "café".encode("latin-1"))
    assert r.text == "café"


def test_pdf_text_is_extracted_from_every_page():
    data = _pdf_bytes(["Quarterly revenue grew 12 percent", "Churn fell to 3 percent"])
    r = extract.extract_text("deck.pdf", "application/pdf", data)
    assert r.kind == "document"
    assert "Quarterly revenue grew 12 percent" in r.text
    assert "Second page closing line" in r.text


def test_docx_paragraphs_and_table_cells_are_extracted():
    data = _docx_bytes(["First paragraph.", "Second &amp; last."])
    r = extract.extract_text("brief.docx", None, data)
    assert r.kind == "document"
    assert "First paragraph." in r.text
    assert "Cell alpha" in r.text and "Cell beta" in r.text


def test_csv_becomes_header_labelled_rows_marked_table():
    data = b"Plan,Seats,Price\nStarter,5,49\nGrowth,25,199\n"
    r = extract.extract_text("pricing.csv", "text/csv", data)
    assert r.kind == "table"
    assert "Plan: Starter; Seats: 5; Price: 49" in r.text
    assert "Plan: Growth; Seats: 25; Price: 199" in r.text


def test_xlsx_every_sheet_becomes_header_labelled_rows():
    r = extract.extract_text("book.xlsx", None, _xlsx_bytes())
    assert r.kind == "table"
    assert "Sheet: Pricing" in r.text
    assert "Plan: Growth; Seats: 25; Price: 199" in r.text
    assert "Sheet: Contacts" in r.text
    assert "Name: Dana; Role: CFO" in r.text


@pytest.mark.parametrize("name", ["virus.exe", "photo.png", "noext"])
def test_unsupported_types_are_refused(name):
    with pytest.raises(extract.UnsupportedType):
        extract.extract_text(name, "application/octet-stream", b"\x00\x01")


def test_corrupt_file_is_an_extraction_error_not_a_crash():
    with pytest.raises(extract.ExtractionError):
        extract.extract_text("broken.pdf", "application/pdf", b"%PDF-not really")
    with pytest.raises(extract.ExtractionError):
        extract.extract_text("broken.xlsx", None, b"PK not a zip")


def test_a_file_with_no_text_is_an_extraction_error():
    with pytest.raises(extract.ExtractionError):
        extract.extract_text("blank.txt", "text/plain", b"   \n\n ")


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------

def test_token_estimate_is_conservative_and_monotonic():
    assert chunking.estimate_tokens("") == 0
    assert chunking.estimate_tokens("abcd") == 1
    assert chunking.estimate_tokens("a" * 4000) == 1000


def test_short_text_is_one_chunk():
    chunks = chunking.chunk_text("Just one short note.")
    assert chunks == ["Just one short note."]


def test_long_text_chunks_near_target_with_overlap_and_full_coverage():
    paras = [f"Paragraph {i}. " + ("word " * 120).strip() for i in range(60)]
    text = "\n\n".join(paras)
    chunks = chunking.chunk_text(text, target_tokens=800, overlap_tokens=100)
    assert len(chunks) > 1
    for c in chunks:
        assert chunking.estimate_tokens(c) <= 800
    # Every paragraph marker survives somewhere — nothing is silently dropped.
    joined = "\n".join(chunks)
    for i in range(60):
        assert f"Paragraph {i}." in joined
    # Consecutive chunks overlap (the tail of one starts the next).
    for a, b in zip(chunks, chunks[1:]):
        assert a[-200:].split()[-1] in b[:1200]


def test_one_giant_unbroken_line_is_still_split():
    text = "x" * 20000
    chunks = chunking.chunk_text(text, target_tokens=800, overlap_tokens=100)
    assert len(chunks) >= 6
    assert all(len(c) <= 3200 for c in chunks)


def test_chunking_is_deterministic():
    text = "\n\n".join(f"Section {i} " + "lorem ipsum " * 200 for i in range(10))
    assert chunking.chunk_text(text) == chunking.chunk_text(text)
