"""Tests for the BM25 keyword index."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.rag.bm25 import (
    build_bm25_index,
    get_bm25_index,
    load_bm25_index,
    persist_bm25_index,
    reset_bm25_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_bm25_cache()
    yield
    reset_bm25_cache()


def _sample_docs() -> list[Document]:
    return [
        Document(
            page_content=(
                "19. § (1) A társasági adó mértéke 9 százalék az adóalap "
                "után. Ez az általános adómérték."
            ),
            metadata={"source": "tao.pdf", "section": "19. §", "chunk_id": 0},
        ),
        Document(
            page_content=(
                "17. § (1) Az adózás előtti eredményt csökkenti az elhatárolt "
                "veszteség legfeljebb 50 százalékáig."
            ),
            metadata={"source": "tao.pdf", "section": "17. §", "chunk_id": 1},
        ),
        Document(
            page_content=(
                "A nonprofit szervezetek közhasznú besorolása külön szabályok "
                "szerint történik."
            ),
            metadata={"source": "nonprofit.pdf", "section": "", "chunk_id": 0},
        ),
    ]


def test_build_and_search_returns_top_hit() -> None:
    index = build_bm25_index(_sample_docs())
    hits = index.search("elhatárolt veszteség", k=2)
    assert hits, "BM25 must return at least one hit for a vocabulary match."
    assert "17. §" in hits[0].page_content


def test_search_returns_empty_on_no_overlap() -> None:
    index = build_bm25_index(_sample_docs())
    # A query with no shared tokens against the corpus -> all scores 0.
    assert index.search("kvantum-szupremácia hipotézis", k=3) == []


def test_source_filter_restricts_results() -> None:
    index = build_bm25_index(_sample_docs())
    hits = index.search("közhasznú", k=5, source_filter=["nonprofit.pdf"])
    assert hits
    assert all(d.metadata["source"] == "nonprofit.pdf" for d in hits)


def test_persist_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "bm25.pkl"
    index = build_bm25_index(_sample_docs())
    persist_bm25_index(index, path)
    assert path.exists()

    loaded = load_bm25_index(path)
    assert loaded is not None
    hits = loaded.search("társasági adó mértéke", k=1)
    assert hits and "19. §" in hits[0].page_content


def test_get_bm25_index_caches_and_invalidates(tmp_path: Path) -> None:
    path = tmp_path / "bm25.pkl"
    persist_bm25_index(build_bm25_index(_sample_docs()), path)

    first = get_bm25_index(path)
    second = get_bm25_index(path)
    assert first is second, "Repeated reads with unchanged mtime should hit the cache."

    # Rebuild with a different corpus and persist -> mtime changes ->
    # cache must reload.
    other = [
        Document(page_content="Teljesen más szöveg.", metadata={"source": "x.pdf"})
    ]
    persist_bm25_index(build_bm25_index(other), path)
    third = get_bm25_index(path)
    assert third is not None and third is not first


def test_get_bm25_index_missing_file_returns_none(tmp_path: Path) -> None:
    assert get_bm25_index(tmp_path / "nope.pkl") is None


def test_empty_corpus_rejected() -> None:
    with pytest.raises(ValueError):
        build_bm25_index([])
