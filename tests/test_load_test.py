"""Tests for the load-test harness (dummy provider, throwaway Chroma)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.config import get_settings
from app.llm import reset_provider_cache
from app.load_test import ScenarioQuery, build_query_mix, run
from app.load_test.timing import NodeTimings, trace_node_timings
from app.rag.ingestion import ingest_documents


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "dummy")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "load_tests")
    get_settings.cache_clear()
    reset_provider_cache()
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
                page_content="17. § (2) Az elhatárolt veszteség az adóalap 50%-áig vehető figyelembe.",
                metadata={"source": "tao.md", "section": "17. §", "chunk_id": 1},
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def test_build_query_mix_cycles_to_requested_length() -> None:
    queries = build_query_mix(20)
    assert len(queries) == 20
    assert all(isinstance(q, ScenarioQuery) for q in queries)
    assert any(q.expected_category == "off_topic" for q in queries)


# ---------------------------------------------------------------------------
# Percentiles + tracer
# ---------------------------------------------------------------------------
def test_node_timings_percentiles_are_monotonic() -> None:
    t = NodeTimings()
    for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        t.record("dummy_node", s)
    percs = t.percentiles("dummy_node")
    assert percs["p50"] <= percs["p95"] <= percs["p99"]
    assert percs["p99"] == 1.0


def test_node_timings_handles_empty_node() -> None:
    t = NodeTimings()
    percs = t.percentiles("never_called")
    assert percs == {"p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_tracer_records_each_node_at_least_once() -> None:
    _seed_corpus()
    with trace_node_timings() as timings:
        from app.agent import build_agent_graph
        graph = build_agent_graph()
        graph.invoke({"query": "Mennyi a társasági adó mértéke?"})

    expected_nodes = {
        "classify_query",
        "query_decomposer",
        "retrieve_documents",
        "tool_executor",
        "answer_generator",
        "hallucination_checker",
    }
    assert expected_nodes.issubset(set(timings.node_names))
    # Every recorded node must have at least one positive sample.
    for node in expected_nodes:
        samples = timings.samples(node)
        assert samples and all(s >= 0 for s in samples)


def test_tracer_restores_originals_on_exit() -> None:
    from app.agent import graph as graph_mod
    from app.agent import nodes as nodes_mod

    before_nodes = nodes_mod.classify_query_node
    before_graph = graph_mod.classify_query_node
    with trace_node_timings():
        assert nodes_mod.classify_query_node is not before_nodes
        assert graph_mod.classify_query_node is not before_graph
    assert nodes_mod.classify_query_node is before_nodes
    assert graph_mod.classify_query_node is before_graph


# ---------------------------------------------------------------------------
# End-to-end runner with small n + concurrency
# ---------------------------------------------------------------------------
def test_runner_produces_csvs_and_chart(tmp_path: Path) -> None:
    _seed_corpus()
    rows, timings = run(n=4, concurrency=2, report_dir=tmp_path, render_chart=True)

    assert len(rows) == 4
    per_query = tmp_path / "load_test_per_query.csv"
    per_node = tmp_path / "load_test_per_node.csv"
    chart = tmp_path / "load_test_per_node.png"
    assert per_query.exists()
    assert per_node.exists()
    assert chart.exists() and chart.stat().st_size > 0

    # Tracer must have seen calls during the run.
    assert any(timings.samples(n) for n in timings.node_names)


def test_runner_without_chart(tmp_path: Path) -> None:
    _seed_corpus()
    rows, _ = run(n=3, concurrency=1, report_dir=tmp_path, render_chart=False)
    assert len(rows) == 3
    assert not (tmp_path / "load_test_per_node.png").exists()
