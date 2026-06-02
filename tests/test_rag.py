"""Tests for the RAG subsystem (splitter, ingestion, retriever, subgraph)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.config import get_settings
from app.llm import reset_provider_cache
from app.rag.ingestion import ingest_documents, load_and_split
from app.rag.retriever import get_vector_store
from app.rag.splitter import SplitterConfig, split_legal_text
from app.rag.subgraph import build_rag_subgraph


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Force dummy provider + a throw-away Chroma dir for every test."""
    monkeypatch.setenv("LLM_PROVIDER", "dummy")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "rag_tests")
    get_settings.cache_clear()
    reset_provider_cache()
    yield


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------
def test_splitter_keeps_section_header_on_every_chunk() -> None:
    text = (
        "19. § (1) A társasági adó mértéke 9 százalék. "
        + ("Lorem ipsum dolor sit amet. " * 80)  # force multiple chunks
        + "\n\n7. § (1) Az adózás előtti eredményt csökkenti az elhatárolt veszteség."
    )
    cfg = SplitterConfig(chunk_size=200, chunk_overlap=20)
    docs = split_legal_text(text, source="t.md", page=1, config=cfg)

    # We must have produced multiple chunks for §19 and at least one for §7.
    sections = {d.metadata["section"] for d in docs}
    assert "19. §" in sections
    assert "7. §" in sections

    # Every chunk that came from §19 must start with its header.
    section_19_chunks = [d for d in docs if d.metadata["section"] == "19. §"]
    assert len(section_19_chunks) >= 2
    assert all(d.page_content.startswith("19. §") for d in section_19_chunks)


def test_splitter_handles_text_without_section_marker() -> None:
    docs = split_legal_text("Csak egy kis szöveg, paragrafus nélkül.", source="t.md")
    assert len(docs) == 1
    assert docs[0].metadata["section"] == ""


# ---------------------------------------------------------------------------
# Ingestion -> retrieval round-trip
# ---------------------------------------------------------------------------
def test_ingest_and_retrieve_round_trip(tmp_path: Path) -> None:
    sample = tmp_path / "sample.md"
    sample.write_text(
        "19. § (1) A társasági adó mértéke 9 százalék.\n\n"
        "7. § (1) Az adózás előtti eredményt csökkenti az elhatárolt veszteség.\n",
        encoding="utf-8",
    )
    chunks = load_and_split(sample)
    assert chunks, "Expected at least one chunk from the sample document."

    indexed = ingest_documents(chunks)
    assert indexed == len(chunks)

    store = get_vector_store()
    results = store.similarity_search("elhatárolt veszteség", k=3)
    assert results
    # The §7 chunk should be the best match for this query.
    assert any("7. §" in d.page_content for d in results)


def test_ingestion_is_idempotent(tmp_path: Path) -> None:
    sample = tmp_path / "sample.md"
    sample.write_text("19. § (1) A társasági adó mértéke 9 százalék.\n", encoding="utf-8")
    chunks = load_and_split(sample)

    ingest_documents(chunks)
    ingest_documents(chunks)  # second run must not duplicate

    store = get_vector_store()
    # We added the same stable IDs twice -> count must stay equal to len(chunks).
    assert store._collection.count() == len(chunks)


# ---------------------------------------------------------------------------
# Subgraph
# ---------------------------------------------------------------------------
def test_rag_subgraph_returns_relevant_docs(tmp_path: Path) -> None:
    # Seed the store with two distinct chunks.
    docs = [
        Document(page_content="19. § (1) A társasági adó mértéke 9 százalék.",
                 metadata={"source": "s.md", "section": "19. §", "chunk_id": 0}),
        Document(page_content="7. § (1) Az adózás előtti eredményt csökkenti az elhatárolt veszteség.",
                 metadata={"source": "s.md", "section": "7. §", "chunk_id": 1}),
    ]
    ingest_documents(docs)

    graph = build_rag_subgraph()
    out = graph.invoke({"query": "elhatárolt veszteség"})

    assert "retrieved_docs" in out and out["retrieved_docs"]
    assert "relevant_docs" in out and out["relevant_docs"]
    # Dummy grader keeps everything; the §7 chunk should be among the docs.
    assert any("7. §" in d.page_content for d in out["relevant_docs"])
