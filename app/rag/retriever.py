"""Chroma-backed retriever factory.

Wraps :class:`langchain_chroma.Chroma` so the rest of the code does not
have to know about the persistence directory, the collection name, or
the embedding function — those all come from :mod:`app.config`.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.vectorstores import VectorStoreRetriever

from app.config import get_settings


def get_vector_store(
    *,
    persist_dir: Path | None = None,
    collection_name: str | None = None,
):
    """Return the configured Chroma vector store instance."""
    from langchain_chroma import Chroma

    from app.llm import get_embedding_model

    settings = get_settings()
    persist_dir = persist_dir or settings.chroma_persist_dir
    collection_name = collection_name or settings.chroma_collection
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=str(persist_dir),
    )


def get_retriever(
    *,
    top_k: int | None = None,
    persist_dir: Path | None = None,
    collection_name: str | None = None,
    where: dict | None = None,
) -> VectorStoreRetriever:
    """Return a similarity retriever bound to the project's Chroma store.

    ``where`` is forwarded as a Chroma metadata filter (e.g.
    ``{"source": {"$in": ["13+A+nonprofit+...pdf"]}}``) so callers can
    restrict the search to specific documents.
    """
    settings = get_settings()
    store = get_vector_store(persist_dir=persist_dir, collection_name=collection_name)
    search_kwargs: dict = {"k": top_k or settings.rag_top_k}
    if where:
        search_kwargs["filter"] = where
    return store.as_retriever(search_kwargs=search_kwargs)

