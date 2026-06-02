"""End-to-end tests for the agentic workflow (dummy provider)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.agent import build_agent_graph
from app.agent.nodes import (
    _TAO_KEYWORDS,  # noqa: F401  (sanity check it stays exported)
    classify_query_node,
    tool_executor_node,
)
from app.config import get_settings
from app.llm import reset_provider_cache
from app.rag.ingestion import ingest_documents


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "dummy")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "agent_tests")
    get_settings.cache_clear()
    reset_provider_cache()
    # Force the cached subgraph to be rebuilt against the fresh Chroma dir.
    from app.agent import nodes as nodes_mod
    nodes_mod._rag_subgraph = None
    yield


def _seed_corpus() -> None:
    ingest_documents(
        [
            Document(
                page_content="19. § (1) A társasági adó mértéke 9 százalék.",
                metadata={"source": "tao.md", "section": "19. §", "chunk_id": 0},
            ),
            Document(
                page_content="17. § (2) Az elhatárolt veszteség legfeljebb az adóalap 50%-áig vehető figyelembe.",
                metadata={"source": "tao.md", "section": "17. §", "chunk_id": 1},
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Node-level
# ---------------------------------------------------------------------------
def test_classify_query_tao_keyword() -> None:
    out = classify_query_node({"query": "Mennyi a társasági adó mértéke?"})
    assert out["category"] == "tao"


def test_classify_query_off_topic() -> None:
    out = classify_query_node({"query": "Milyen az időjárás Budapesten?"})
    assert out["category"] == "off_topic"


def test_tool_executor_runs_tao_calculator_on_amount() -> None:
    _seed_corpus()
    out = tool_executor_node({"query": "Mennyi a TAO 1 000 000 Ft adóalap után?"})
    tools = [r["tool"] for r in out["tool_results"]]
    assert "tao_calculator" in tools
    calc = next(r for r in out["tool_results"] if r["tool"] == "tao_calculator")
    assert calc["output"]["tax_huf"] == 90_000


def test_tool_executor_validates_citation_in_query() -> None:
    _seed_corpus()
    out = tool_executor_node({"query": "Mit mond a 19. § a társasági adó mértékéről?"})
    tools = [r["tool"] for r in out["tool_results"]]
    assert "legal_reference_validator" in tools
    validator = next(r for r in out["tool_results"] if r["tool"] == "legal_reference_validator")
    assert validator["output"]["found"] is True


# ---------------------------------------------------------------------------
# End-to-end through the compiled graph
# ---------------------------------------------------------------------------
def test_graph_handles_tao_question_end_to_end() -> None:
    _seed_corpus()
    graph = build_agent_graph()
    result = graph.invoke({"query": "Mennyi a társasági adó mértéke?"})

    assert result["category"] == "tao"
    assert result.get("retrieved_docs"), "Expected RAG subgraph to surface docs."
    assert result.get("final_answer"), "Expected a final answer."
    assert result["grounded"] is True


def test_graph_short_circuits_for_off_topic() -> None:
    graph = build_agent_graph()
    result = graph.invoke({"query": "Milyen az időjárás Budapesten?"})

    assert result["category"] == "off_topic"
    assert "társasági adó" in result["final_answer"]
    # Off-topic path must skip retrieval entirely.
    assert "retrieved_docs" not in result or not result["retrieved_docs"]


def test_graph_runs_tool_when_amount_and_citation_present() -> None:
    _seed_corpus()
    graph = build_agent_graph()
    result = graph.invoke(
        {"query": "A 19. § szerint mennyi TAO jár 1 000 000 Ft adóalapra?"}
    )

    tool_names = {r["tool"] for r in result.get("tool_results", [])}
    assert "tao_calculator" in tool_names
    assert "legal_reference_validator" in tool_names
    assert result.get("final_answer")
