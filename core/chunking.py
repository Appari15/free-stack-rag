"""
Text chunking engine with recursive splitting.

Why recursive?  It tries the largest separators first (paragraphs),
then falls back to sentences, then words, then characters.  This
preserves semantic coherence within chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import tiktoken

from core.models import DocumentChunk, DocumentMetadata

_ENCODER = tiktoken.get_encoding("cl100k_base")

# ── Helpers ──────────────────────────────────────────────────


def count_tokens(text: str) -> int:
    """Count tokens using the same tokenizer as GPT-4 / Titan."""
    return len(_ENCODER.encode(text))


def decode_tokens(tokens: list[int]) -> str:
    return _ENCODER.decode(tokens)


def encode_tokens(text: str) -> list[int]:
    return _ENCODER.encode(text)


# ── Config ───────────────────────────────────────────────────


@dataclass
class ChunkConfig:
    chunk_size: int = 512  # max tokens per chunk
    chunk_overlap: int = 50  # token overlap between consecutive chunks
    min_chunk_size: int = 30  # discard chunks smaller than this


# ── Recursive Chunker ───────────────────────────────────────

# Separators ordered from coarsest to finest
SEPARATORS = [
    "\n\n\n",  # triple newline (section breaks)
    "\n\n",  # double newline (paragraphs)
    "\n",  # single newline
    ". ",  # sentence boundary
    "! ",
    "? ",
    "; ",
    ", ",
    " ",  # word boundary
    "",  # character-level (last resort)
]


def recursive_chunk(
    text: str,
    config: ChunkConfig,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Recursively split text so every chunk is <= config.chunk_size tokens.
    """
    if not text or text.strip() == "":
        return []

    if separators is None:
        separators = SEPARATORS

    # Base case: text already fits
    if count_tokens(text) <= config.chunk_size:
        return [text]

    # Pick the first (coarsest) separator that exists in the text
    sep = ""
    remaining_seps = [""]
    for i, s in enumerate(separators):
        if s == "" or s in text:
            sep = s
            remaining_seps = separators[i + 1 :]
            break

    # Hard token split as absolute last resort
    if sep == "":
        return _hard_split(text, config)

    parts = text.split(sep)
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part

        if count_tokens(candidate) <= config.chunk_size:
            current = candidate
        else:
            # Flush current
            if current:
                if count_tokens(current) > config.chunk_size:
                    # Still too big → recurse with finer separators
                    chunks.extend(recursive_chunk(current, config, remaining_seps))
                elif count_tokens(current) >= config.min_chunk_size:
                    chunks.append(current)
            current = part

    # Flush remainder
    if current:
        if count_tokens(current) > config.chunk_size:
            chunks.extend(recursive_chunk(current, config, remaining_seps))
        elif count_tokens(current) >= config.min_chunk_size:
            chunks.append(current)

    return chunks


def _hard_split(text: str, config: ChunkConfig) -> list[str]:
    """Character-level token split when no separators work."""
    tokens = encode_tokens(text)
    step = config.chunk_size - config.chunk_overlap
    if step <= 0:
        step = config.chunk_size

    chunks: list[str] = []
    for i in range(0, len(tokens), step):
        window = tokens[i : i + config.chunk_size]
        decoded = decode_tokens(window)
        if count_tokens(decoded) >= config.min_chunk_size:
            chunks.append(decoded)
    return chunks


def apply_overlap(chunks: list[str], config: ChunkConfig) -> list[str]:
    """
    Prepend the last N tokens of the previous chunk to the current one.
    This prevents information loss at boundaries.
    """
    if config.chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    result: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tokens = encode_tokens(chunks[i - 1])
        overlap_tokens = prev_tokens[-config.chunk_overlap :]
        overlap_text = decode_tokens(overlap_tokens)
        result.append(f"{overlap_text} {chunks[i]}")
    return result


# ── Public API ───────────────────────────────────────────────


def chunk_text(
    text: str,
    metadata: DocumentMetadata,
    document_id: str | None = None,
    config: ChunkConfig | None = None,
) -> list[DocumentChunk]:
    """
    Full pipeline: split → overlap → wrap in DocumentChunk models.
    """
    cfg = config or ChunkConfig()
    doc_id = document_id or str(uuid4())

    raw_chunks = recursive_chunk(text, cfg)
    raw_chunks = apply_overlap(raw_chunks, cfg)

    chunks: list[DocumentChunk] = []
    for i, content in enumerate(raw_chunks):
        chunks.append(
            DocumentChunk(
                document_id=doc_id,
                content=content.strip(),
                metadata=metadata,
                chunk_index=i,
                total_chunks=len(raw_chunks),
                token_count=count_tokens(content),
            )
        )
    return chunks
