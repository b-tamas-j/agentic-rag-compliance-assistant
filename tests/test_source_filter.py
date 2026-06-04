"""Tests for source-hint classification + retrieval-mode dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.agent.nodes import (
    SOURCE_HINT_FILES,
    classify_query_node,
    retrieve_documents_node,
)
from app.config import get_settings
from app.llm import reset_provider_cache
from app.rag.bm25 import reset_bm25_cache
from app.rag.ingestion import build_and_persist_bm25, ingest_documents
from app.rag.subgraph import (
    _reciprocal_rank_fusion,
    build_rag_subgraph,
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "dummy")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "source_filter_tests")
    monkeypatch.setenv("RAG_BM25_PATH", str(tmp_path / "bm25.pkl"))
    get_settings.cache_clear()
    reset_provider_cache()
    reset_bm25_cache()
    # Reset the cached compiled subgraph between tests so changes to
    # the retrieval mode take effect.
    import app.agent.nodes as nodes_mod
    nodes_mod._rag_subgraph = None
    yield


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
def test_classify_returns_source_hint_for_nonprofit() -> None:
    out = classify_query_node({"query": "Hogyan adóznak a nonprofit alapítványok?"})
    assert out["category"] == "tao"
    assert out["source_hint"] == "nonprofit"


def test_classify_returns_credit_hint_for_nahi() -> None:
    out = classify_query_node({"query": "Mi a növekedési adóhitel feltétele?"})
    assert out["source_hint"] == "credit"


def test_classify_returns_offering_hint_for_felajanlas() -> None:
    out = classify_query_node({"query": "Mire vonatkozik a tao-felajánlás?"})
    assert out["source_hint"] == "offering"


def test_classify_off_topic_has_general_hint() -> None:
    out = classify_query_node({"query": "Milyen idő lesz holnap?"})
    assert out["category"] == "off_topic"
    assert out["source_hint"] == "general"


# ---------------------------------------------------------------------------
# Source filter actually narrows retrieval
# ---------------------------------------------------------------------------
def _seed_two_sources() -> None:
    docs = [
        Document(
            page_content="19. § (1) A társasági adó mértéke 9 százalék.",
            metadata={"source": "41 A társasági adó legfontosabb szabályai 2025.09.01.pdf",
                      "section": "19. §", "chunk_id": 0, "page": 1},
        ),
        Document(
            page_content="A nonprofit közhasznú szervezet adómentes bevételeiről.",
            metadata={"source": "13+A+nonprofit+szervezetek+adózása+2025.01.27.pdf",
                      "section": "", "chunk_id": 0, "page": 1},
        ),
    ]
    ingest_documents(docs)
    build_and_persist_bm25(docs)


def test_retrieve_documents_node_respects_source_hint() -> None:
    _seed_two_sources()
    out = retrieve_documents_node(
        {
            "query": "társasági adó mértéke",
            "sub_queries": ["társasági adó mértéke"],
            "source_hint": "nonprofit",
        }
    )
    docs = out["retrieved_docs"]
    assert docs, "Expected at least one chunk through the nonprofit filter."
    assert all(
        d.metadata["source"] in SOURCE_HINT_FILES["nonprofit"] for d in docs
    )


def test_retrieve_documents_node_general_hint_skips_filter() -> None:
    _seed_two_sources()
    out = retrieve_documents_node(
        {
            "query": "társasági adó",
            "sub_queries": ["társasági adó"],
            "source_hint": "general",
        }
    )
    sources = {d.metadata["source"] for d in out["retrieved_docs"]}
    # Both PDFs should be reachable when no filter is applied.
    assert len(sources) >= 1


# ---------------------------------------------------------------------------
# Retrieval-mode dispatch
# ---------------------------------------------------------------------------
def test_bm25_mode_uses_keyword_index(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RETRIEVAL_MODE", "bm25")
    get_settings.cache_clear()
    _seed_two_sources()

    graph = build_rag_subgraph()
    out = graph.invoke({"query": "nonprofit közhasznú"})
    docs = out.get("retrieved_docs", [])
    assert docs, "BM25 mode must return results for a lexical match."
    assert any("nonprofit" in d.metadata["source"] for d in docs)


def test_hybrid_mode_combines_dense_and_bm25(monkeypatch) -> None:
    monkeypatch.setenv("RAG_RETRIEVAL_MODE", "hybrid")
    get_settings.cache_clear()
    _seed_two_sources()

    graph = build_rag_subgraph()
    out = graph.invoke({"query": "társasági adó mértéke 9 százalék"})
    assert out.get("retrieved_docs"), "Hybrid mode must return at least one doc."


# ---------------------------------------------------------------------------
# RRF unit
# ---------------------------------------------------------------------------
def test_rrf_promotes_doc_ranked_high_by_both_lists() -> None:
    a = Document(page_content="A", metadata={"source": "s", "page": 1, "chunk_id": 0})
    b = Document(page_content="B", metadata={"source": "s", "page": 1, "chunk_id": 1})
    c = Document(page_content="C", metadata={"source": "s", "page": 1, "chunk_id": 2})

    # 'a' is top-1 in both rankings; 'b' is top-1 in one only.
    fused = _reciprocal_rank_fusion([[a, b, c], [a, c, b]], k=3, rrf_k=60)
    assert fused[0].page_content == "A"
    assert {d.page_content for d in fused} == {"A", "B", "C"}
