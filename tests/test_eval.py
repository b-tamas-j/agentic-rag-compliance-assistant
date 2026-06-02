"""Tests for the eval harness (dummy provider, throwaway Chroma)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.config import get_settings
from app.eval import (
    EvalQuestion,
    aggregate,
    category_correct,
    citation_accuracy,
    expected_terms_coverage,
    judge_answer,
    load_dataset,
    retrieval_recall_at_k,
)
from app.eval.runner import run as run_eval
from app.llm import reset_provider_cache
from app.rag.ingestion import ingest_documents


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "dummy")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "eval_tests")
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
# Dataset
# ---------------------------------------------------------------------------
def test_default_dataset_loads_and_has_15_items() -> None:
    questions = load_dataset()
    assert len(questions) == 15
    assert all(isinstance(q, EvalQuestion) for q in questions)
    assert {q.category for q in questions} == {"tao", "off_topic"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_recall_at_k_full_hit() -> None:
    q = EvalQuestion(id="x", category="tao", question="?", expected_sections=["19. §"])
    docs = [Document(page_content="A 19. § kimondja, hogy...", metadata={"section": "19. §"})]
    assert retrieval_recall_at_k(q, docs) == 1.0


def test_recall_at_k_partial() -> None:
    q = EvalQuestion(
        id="x", category="tao", question="?", expected_sections=["19. §", "17. §"]
    )
    docs = [Document(page_content="19. §", metadata={})]
    assert retrieval_recall_at_k(q, docs) == 0.5


def test_recall_vacuously_true_for_off_topic() -> None:
    q = EvalQuestion(id="x", category="off_topic", question="?", expected_sections=[])
    assert retrieval_recall_at_k(q, []) == 1.0


def test_citation_accuracy_rewards_grounded_citations() -> None:
    q = EvalQuestion(id="x", category="tao", question="?")
    docs = [Document(page_content="19. § (1)", metadata={"section": "19. §"})]
    assert citation_accuracy(q, "A 19. § szerint 9%.", docs) == 1.0


def test_citation_accuracy_penalises_invented_citation() -> None:
    q = EvalQuestion(id="x", category="tao", question="?")
    docs = [Document(page_content="19. §", metadata={"section": "19. §"})]
    # Cites 19. § (grounded) AND 999. § (invented) -> 1/2
    assert citation_accuracy(q, "A 19. § és a 999. § alapján.", docs) == 0.5


def test_terms_coverage() -> None:
    q = EvalQuestion(id="x", category="tao", question="?", expected_terms=["9", "százalék"])
    assert expected_terms_coverage(q, "Az adó 9 százalék.") == 1.0
    assert expected_terms_coverage(q, "Az adó 9%-os.") == 0.5


def test_category_correct() -> None:
    q = EvalQuestion(id="x", category="tao", question="?")
    assert category_correct(q, "tao") is True
    assert category_correct(q, "off_topic") is False


def test_aggregate_averages_numeric_and_bool_columns() -> None:
    rows = [
        {"recall_at_k": 1.0, "grounded": True},
        {"recall_at_k": 0.5, "grounded": False},
    ]
    summary = aggregate(rows)
    assert summary["recall_at_k"] == 0.75
    assert summary["grounded"] == 0.5


# ---------------------------------------------------------------------------
# Judge (dummy fallback)
# ---------------------------------------------------------------------------
def test_dummy_judge_returns_valid_scores() -> None:
    verdict = judge_answer(
        "Mennyi a TAO?",
        "A társasági adó mértéke 9 százalék a Tao. tv. 19. § alapján.",
        [Document(page_content="19. § ...", metadata={"section": "19. §"})],
    )
    for k in ("groundedness", "relevance", "completeness"):
        assert 1 <= verdict[k] <= 5


# ---------------------------------------------------------------------------
# Runner end-to-end with a 2-question mini-dataset
# ---------------------------------------------------------------------------
def test_runner_writes_csv_and_summary(tmp_path: Path) -> None:
    _seed_corpus()

    mini = tmp_path / "mini.json"
    mini.write_text(
        json.dumps(
            [
                {
                    "id": "m1",
                    "category": "tao",
                    "question": "Mennyi a társasági adó mértéke?",
                    "expected_sections": ["19. §"],
                    "expected_terms": ["9"],
                    "topic": "adómérték",
                },
                {
                    "id": "m2",
                    "category": "off_topic",
                    "question": "Milyen az időjárás?",
                    "expected_sections": [],
                    "expected_terms": [],
                    "topic": "off-topic",
                },
            ]
        ),
        encoding="utf-8",
    )

    out = tmp_path / "out.csv"
    rows = run_eval(mini, out, with_judge=True)

    assert len(rows) == 2
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "category_predicted" in content
    # The TAO row should have all judge columns; the off-topic row should not be judged.
    tao_row = next(r for r in rows if r["id"] == "m1")
    assert tao_row["category_predicted"] == "tao"
    assert "judge_groundedness" in tao_row
    off_row = next(r for r in rows if r["id"] == "m2")
    assert "judge_groundedness" not in off_row
