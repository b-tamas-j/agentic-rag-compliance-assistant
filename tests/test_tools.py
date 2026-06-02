"""Tests for the deterministic agent tools (tao_calculator, legal_reference_validator)."""

from __future__ import annotations

import math

import pytest
from langchain_core.documents import Document

from app.agent.tools import (
    LOSS_OFFSET_CAP,
    TAO_RATE,
    legal_reference_validator,
    tao_calculator,
)
from app.config import get_settings
from app.llm import reset_provider_cache
from app.rag.ingestion import ingest_documents


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Force dummy provider + a throw-away Chroma dir for every test."""
    monkeypatch.setenv("LLM_PROVIDER", "dummy")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "tools_tests")
    get_settings.cache_clear()
    reset_provider_cache()
    yield


# ---------------------------------------------------------------------------
# tao_calculator
# ---------------------------------------------------------------------------
def test_tao_calculator_basic_rate() -> None:
    out = tao_calculator.invoke({"tax_base_huf": 1_000_000})
    assert out["rate"] == TAO_RATE
    assert out["tax_huf"] == 90_000
    assert out["loss_applied_huf"] == 0
    assert math.isclose(out["effective_rate"], TAO_RATE)
    assert "9%" in out["explanation"]


def test_tao_calculator_loss_capped_at_half_of_base() -> None:
    # Loss is bigger than 50% of the base -> only half can be applied.
    out = tao_calculator.invoke(
        {"tax_base_huf": 1_000_000, "loss_carried_forward_huf": 800_000}
    )
    expected_offset = 1_000_000 * LOSS_OFFSET_CAP
    assert out["loss_applied_huf"] == expected_offset
    assert out["adjusted_base_huf"] == 1_000_000 - expected_offset
    assert out["tax_huf"] == 45_000


def test_tao_calculator_loss_below_cap_applied_fully() -> None:
    out = tao_calculator.invoke(
        {"tax_base_huf": 1_000_000, "loss_carried_forward_huf": 200_000}
    )
    assert out["loss_applied_huf"] == 200_000
    assert out["tax_huf"] == 72_000


def test_tao_calculator_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError):
        tao_calculator.invoke({"tax_base_huf": -1})
    with pytest.raises(ValueError):
        tao_calculator.invoke(
            {"tax_base_huf": 100, "loss_carried_forward_huf": -1}
        )


# ---------------------------------------------------------------------------
# legal_reference_validator
# ---------------------------------------------------------------------------
def _seed_corpus() -> None:
    """Populate the (per-test) Chroma store with a couple of §-tagged chunks."""
    ingest_documents(
        [
            Document(
                page_content="19. § (1) A társasági adó mértéke 9 százalék.",
                metadata={"source": "tao.md", "section": "19. §", "chunk_id": 0},
            ),
            Document(
                page_content="17. § (2) Az elhatárolt veszteség az adóalap 50%-áig...",
                metadata={"source": "tao.md", "section": "17. §", "chunk_id": 1},
            ),
        ]
    )


def test_validator_finds_existing_section() -> None:
    _seed_corpus()
    out = legal_reference_validator.invoke({"citation": "Tao. tv. 19. §"})
    assert out["found"] is True
    assert out["section"] == "19. §"
    assert out["match_count"] >= 1
    assert "tao.md" in out["sources"]


def test_validator_reports_missing_section() -> None:
    _seed_corpus()
    out = legal_reference_validator.invoke({"citation": "999. §"})
    assert out["found"] is False
    assert out["section"] == "999. §"
    assert out["sources"] == []


def test_validator_handles_input_without_section_token() -> None:
    _seed_corpus()
    out = legal_reference_validator.invoke({"citation": "no reference here"})
    assert out["found"] is False
    assert out["section"] is None
    assert "reason" in out


def test_validator_parses_letter_suffix_sections() -> None:
    ingest_documents(
        [
            Document(
                page_content="17/A. § (1) Speciális rendelkezés.",
                metadata={"source": "tao.md", "section": "17/A. §", "chunk_id": 0},
            )
        ]
    )
    out = legal_reference_validator.invoke({"citation": "lásd a 17/A. § szerint"})
    assert out["section"] == "17/A. §"
    assert out["found"] is True
