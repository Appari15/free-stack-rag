"""
Document loader: reads files, extracts text, chunks, returns DocumentChunks.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import structlog

from config.settings import settings
from core.chunking import ChunkConfig, chunk_text
from core.models import DocumentChunk, DocumentMetadata, FileType
from ingestion.extractors import extract_text

logger = structlog.get_logger()

EXTENSION_MAP: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".txt": FileType.TXT,
    ".html": FileType.HTML,
    ".htm": FileType.HTML,
}


class DocumentLoader:
    """
    Loads documents from files or raw content, producing
    ready-to-embed DocumentChunks.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.config = ChunkConfig(
            chunk_size=chunk_size or settings.default_chunk_size,
            chunk_overlap=chunk_overlap or settings.default_chunk_overlap,
        )

    def load_file(self, path: str | Path) -> list[DocumentChunk]:
        """Load a single file and return its chunks."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_type = EXTENSION_MAP.get(path.suffix.lower())
        if file_type is None:
            logger.warning("unsupported_file", path=str(path), suffix=path.suffix)
            return []

        metadata = DocumentMetadata(
            source=path.name,
            file_type=file_type,
            title=path.stem,
        )

        text = extract_text(path, file_type)
        chunks = chunk_text(text, metadata, config=self.config)

        logger.info(
            "file_loaded",
            path=str(path),
            chars=len(text),
            chunks=len(chunks),
        )
        return chunks

    def load_bytes(
        self,
        data: bytes,
        filename: str,
        metadata: DocumentMetadata,
    ) -> list[DocumentChunk]:
        """Load from raw bytes (file upload)."""
        text = extract_text(BytesIO(data), metadata.file_type)
        return chunk_text(text, metadata, config=self.config)

    def load_text(
        self,
        content: str,
        metadata: DocumentMetadata,
    ) -> list[DocumentChunk]:
        """Load from a raw string."""
        return chunk_text(content, metadata, config=self.config)

    def load_directory(self, dir_path: str | Path) -> list[DocumentChunk]:
        """Recursively load all supported files in a directory."""
        dir_path = Path(dir_path)
        all_chunks: list[DocumentChunk] = []

        for ext in EXTENSION_MAP:
            for file_path in sorted(dir_path.rglob(f"*{ext}")):
                try:
                    all_chunks.extend(self.load_file(file_path))
                except Exception as e:
                    logger.error(
                        "file_load_error",
                        path=str(file_path),
                        error=str(e),
                    )

        logger.info(
            "directory_loaded",
            path=str(dir_path),
            files=len(set(c.document_id for c in all_chunks)),
            chunks=len(all_chunks),
        )
        return all_chunks
