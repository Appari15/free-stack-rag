"""
Text extractors for each supported file type.
Each extractor reads raw bytes or file paths and returns plain text.
"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import structlog

from core.models import FileType

logger = structlog.get_logger()


def extract_text(source: str | Path | BytesIO, file_type: FileType) -> str:
    """
    Route to the correct extractor based on file type.
    """
    extractors = {
        FileType.TXT: _extract_plain,
        FileType.MARKDOWN: _extract_markdown,
        FileType.PDF: _extract_pdf,
        FileType.HTML: _extract_html,
    }

    extractor = extractors.get(file_type, _extract_plain)
    text = extractor(source)

    # Clean up: normalize whitespace, remove null bytes
    text = text.replace("\x00", "")
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    logger.debug(
        "text_extracted",
        file_type=file_type.value,
        chars=len(text),
    )
    return text


def _read_bytes(source: str | Path | BytesIO) -> bytes:
    if isinstance(source, BytesIO):
        return source.read()
    with open(source, "rb") as f:
        return f.read()


def _extract_plain(source: str | Path | BytesIO) -> str:
    raw = _read_bytes(source)
    return raw.decode("utf-8", errors="replace")


def _extract_markdown(source: str | Path | BytesIO) -> str:
    text = _extract_plain(source)
    # Keep structure but flag code blocks so they aren't confused
    # with prose during retrieval
    text = re.sub(
        r"```(\w*)\n([\s\S]*?)```",
        lambda m: (
            f"\n[CODE BLOCK ({m.group(1) or 'plain'})]\n{m.group(2)}\n[END CODE BLOCK]\n"
        ),
        text,
    )
    return text


def _extract_pdf(source: str | Path | BytesIO) -> str:
    try:
        import pymupdf
    except ImportError:
        raise ImportError("Install pymupdf: pip install pymupdf")

    raw = _read_bytes(source)
    doc = pymupdf.open(stream=raw, filetype="pdf")

    pages: list[str] = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text}")

    doc.close()
    return "\n\n".join(pages)


def _extract_html(source: str | Path | BytesIO) -> str:
    raw = _extract_plain(source)
    # Strip HTML tags (basic; production would use BeautifulSoup)
    clean = re.sub(r"<script[\s\S]*?</script>", "", raw, flags=re.IGNORECASE)
    clean = re.sub(r"<style[\s\S]*?</style>", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = re.sub(r"&\w+;", " ", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()
