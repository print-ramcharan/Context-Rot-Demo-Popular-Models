from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def extract_document(path: str | Path) -> str:
    """Extract structured Markdown from supported files.

    Docling is preferred for PDF/DOCX because it preserves layout, headings,
    tables and OCR-capable document structure. Lightweight fallbacks are kept
    for text-native files and installations where Docling is unavailable.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8-sig")

    if suffix not in {".pdf", ".docx"}:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        document = result.document
        markdown = document.export_to_markdown()
        if markdown and markdown.strip():
            return markdown
        raise ValueError("Document parser returned no readable text")
    except ImportError:
        logger.warning("Docling is not installed; using basic text extraction fallback")
    except Exception as exc:
        logger.exception("Docling extraction failed for %s", file_path.name)
        if suffix == ".pdf":
            raise ValueError(
                f"Docling could not extract '{file_path.name}'. "
                "If this is a scanned PDF, install the declared Docling dependency "
                "and ensure its OCR models are available."
            ) from exc
        raise

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"\n\n## Page {page_number}\n\n{page_text}")
        extracted = "".join(pages)
        if not extracted.strip():
            raise ValueError("No selectable PDF text found; OCR-capable Docling is required for this scan.")
        return extracted

    from docx import Document
    document = Document(str(file_path))
    parts = []
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            parts.append(paragraph.text)
    for table_index, table in enumerate(document.tables, start=1):
        parts.append(f"\n### Table {table_index}")
        for row in table.rows:
            parts.append(" | ".join(cell.text.replace("\n", " ").strip() for cell in row.cells))
    return "\n\n".join(parts)
