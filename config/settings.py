"""
Single source of truth for every configurable value.
Loaded from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── LLM ──────────────────────────────────────────
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.1:8b"

    # ── Embeddings ───────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── ChromaDB ─────────────────────────────────────
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection: str = "rag_documents"

    # ── Application ──────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8080
    log_level: str = "INFO"
    api_key: str = "CHANGE_ME_IN_PRODUCTION"

    # ── Ingestion ────────────────────────────────────
    document_watch_dir: str = "/data/documents"
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50

    # ── Hybrid Search ────────────────────────────────
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3


settings = Settings()
