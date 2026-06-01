"""LLM and embedding provider factory.

Single entry point used everywhere in the codebase so the rest of the
application is agnostic to whether we run against Ollama or the dummy
in-memory provider used in tests and load benchmarks.

The selected backend is driven by :data:`Settings.llm_provider`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings, get_settings
from app.llm.dummy import DummyChatModel, DummyEmbeddings

ModelRole = Literal["main", "judge", "fast"]


def _resolve_ollama_model(settings: Settings, role: ModelRole) -> str:
    if role == "judge":
        return settings.ollama_judge_model
    if role == "fast":
        return settings.ollama_fast_model
    return settings.ollama_model


@lru_cache(maxsize=4)
def get_chat_model(role: ModelRole = "main") -> BaseChatModel:
    """Return a chat model instance for the given role.

    ``role`` lets callers pick between the main agent model, the
    independent judge model used in evaluation, and a smaller "fast"
    model used in load tests.
    """
    settings = get_settings()
    if settings.llm_provider == "dummy":
        return DummyChatModel(model_name=f"dummy-{role}")

    # Local import keeps the dummy path free of the optional dependency.
    from langchain_ollama import ChatOllama

    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=_resolve_ollama_model(settings, role),
        temperature=settings.llm_temperature,
        num_predict=settings.llm_max_tokens,
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """Return an embedding model based on the configured provider."""
    settings = get_settings()
    if settings.llm_provider == "dummy":
        return DummyEmbeddings()

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )


def reset_provider_cache() -> None:
    """Clear cached model instances (useful in tests after env changes)."""
    get_chat_model.cache_clear()
    get_embedding_model.cache_clear()

