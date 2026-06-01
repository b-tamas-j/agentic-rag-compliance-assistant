"""Sanity test to ensure config loads without errors."""

from app.config import get_settings


def test_settings_load() -> None:
    settings = get_settings()
    assert settings.rag_chunk_size > 0
    assert settings.rag_top_k > 0
    assert settings.llm_provider in {"ollama", "dummy"}
