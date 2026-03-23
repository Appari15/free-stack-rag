"""
Background file watcher — automatically ingests new or changed
documents from a watched directory.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog
from watchfiles import Change, awatch

from config.settings import settings
from ingestion.loader import DocumentLoader

logger = structlog.get_logger()


class DocumentWatcher:
    """
    Watches settings.document_watch_dir for file changes.
    On startup, does a full initial load of existing files.
    Then watches for adds/modifications.
    """

    def __init__(self, on_chunks_ready):
        """
        Args:
            on_chunks_ready: async callable(chunks) — stores the chunks.
                             Injected by app lifespan to avoid circular imports.
        """
        self._on_chunks_ready = on_chunks_ready
        self._loader = DocumentLoader()
        self._watch_dir = Path(settings.document_watch_dir)

    async def start(self):
        """Initial load + file watching loop."""
        self._watch_dir.mkdir(parents=True, exist_ok=True)
        await self._initial_load()
        await self._watch_loop()

    async def _initial_load(self):
        chunks = self._loader.load_directory(self._watch_dir)
        if chunks:
            await self._on_chunks_ready(chunks)
            logger.info("initial_load_done", chunks=len(chunks))
        else:
            logger.info("initial_load_empty")

    async def _watch_loop(self):
        try:
            async for changes in awatch(self._watch_dir):
                for change_type, path_str in changes:
                    if change_type in (Change.added, Change.modified):
                        await self._handle_change(path_str)
                    elif change_type == Change.deleted:
                        logger.info("file_deleted", path=path_str)
        except asyncio.CancelledError:
            logger.info("watcher_cancelled")
        except Exception as e:
            logger.error("watcher_error", error=str(e))

    async def _handle_change(self, path_str: str):
        try:
            chunks = self._loader.load_file(path_str)
            if chunks:
                await self._on_chunks_ready(chunks)
                logger.info(
                    "file_auto_ingested",
                    path=path_str,
                    chunks=len(chunks),
                )
        except Exception as e:
            logger.error("auto_ingest_error", path=path_str, error=str(e))
