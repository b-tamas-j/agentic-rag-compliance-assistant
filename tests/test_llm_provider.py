"""Tests for the LLM provider abstraction."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.llm import get_chat_model, get_embedding_model, reset_provider_cache
from app.llm.dummy import DummyChatModel, DummyEmbeddings


def _force_dummy(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "dummy")
    get_settings.cache_clear()
    reset_provider_cache()


def test_dummy_chat_is_deterministic(monkeypatch) -> None:
    _force_dummy(monkeypatch)
    chat = get_chat_model()
    assert isinstance(chat, DummyChatModel)

    msg = [HumanMessage(content="Mennyi az általános ÁFA kulcs?")]
    a = chat.invoke(msg)
    b = chat.invoke(msg)
    assert a.content == b.content
    assert "Mennyi az általános ÁFA kulcs?" in a.content


def test_chat_model_role_is_distinct(monkeypatch) -> None:
    _force_dummy(monkeypatch)
    main = get_chat_model("main")
    judge = get_chat_model("judge")
    fast = get_chat_model("fast")
    assert main.model_name == "dummy-main"
    assert judge.model_name == "dummy-judge"
    assert fast.model_name == "dummy-fast"


def test_dummy_embeddings_shape_and_determinism(monkeypatch) -> None:
    _force_dummy(monkeypatch)
    emb = get_embedding_model()
    assert isinstance(emb, DummyEmbeddings)

    docs = ["első chunk", "második chunk"]
    vectors = emb.embed_documents(docs)
    assert len(vectors) == 2
    assert all(len(v) == emb.dim for v in vectors)

    # Determinism
    assert emb.embed_query("kérdés") == emb.embed_query("kérdés")
    # Different inputs -> different vectors
    assert emb.embed_query("a") != emb.embed_query("b")
    # Values in [-1, 1)
    for v in vectors:
        assert all(-1.0 <= x < 1.0 for x in v)


def test_ollama_provider_is_selected(monkeypatch) -> None:
    """When provider=ollama, the factory must return ChatOllama / OllamaEmbeddings."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    get_settings.cache_clear()
    reset_provider_cache()

    from langchain_ollama import ChatOllama, OllamaEmbeddings

    chat = get_chat_model()
    emb = get_embedding_model()
    assert isinstance(chat, ChatOllama)
    assert isinstance(emb, OllamaEmbeddings)
