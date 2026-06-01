"""Application configuration loaded from environment variables.

Uses pydantic-settings so values can come from a `.env` file or the OS env.
Centralising configuration here keeps the rest of the codebase free of
`os.getenv` calls and makes the app easy to test and containerise.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM provider ---
    llm_provider: Literal["ollama", "dummy"] = "dummy"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b-instruct"
    ollama_embedding_model: str = "nomic-embed-text"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # --- RAG ---
    chroma_persist_dir: Path = Field(default=Path("./data/chroma"))
    chroma_collection: str = "compliance_docs"
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_top_k: int = 5

    # --- Agent ---
    max_hallucination_retries: int = 2

    # --- Data ---
    documents_dir: Path = Field(default=Path("./data/documents"))

    # --- UI ---
    streamlit_server_port: int = 8501

    # --- Logging ---
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
