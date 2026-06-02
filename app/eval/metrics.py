"""Deterministic evaluation metrics — no LLM involved."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from app.eval.dataset import EvalQuestion

_SECTION_RE = re.compile(r"\d+(?:/[A-ZÁÉÍÓÖŐÚÜŰ])?\.\s*§")


def _normalize_section(token: str) -> str:
    """Strip whitespace differences so '19. §' and '19.§' compare equal."""
    return token.replace(" ", "").lower()


def retrieval_recall_at_k(
    question: EvalQuestion,
    retrieved_docs: list[Document],
) -> float:
    """Fraction of `expected_sections` that appear in any retrieved chunk.

    A section is considered hit if it shows up either in a chunk's
    ``metadata.section`` or anywhere in its text.
    """
    if not question.expected_sections:
        return 1.0  # vacuously true for off-topic questions

    found_sections: set[str] = set()
    for doc in retrieved_docs:
        meta_section = (doc.metadata or {}).get("section", "")
        text = doc.page_content or ""
        for expected in question.expected_sections:
            key = _normalize_section(expected)
            if _normalize_section(meta_section) == key or key in _normalize_section(text):
                found_sections.add(expected)

    return len(found_sections) / len(question.expected_sections)


def citation_accuracy(
    question: EvalQuestion,
    answer: str,
    retrieved_docs: list[Document],
) -> float:
    """Of the ``§`` citations the answer mentions, how many are backed by retrieved docs.

    Returns 1.0 if the answer cites no sections (no false positives possible);
    returns 0.0 if the answer cites only invented sections.
    """
    cited = {_normalize_section(t) for t in _SECTION_RE.findall(answer or "")}
    if not cited:
        return 1.0

    available: set[str] = set()
    for doc in retrieved_docs:
        text = doc.page_content or ""
        meta_section = (doc.metadata or {}).get("section", "")
        if meta_section:
            available.add(_normalize_section(meta_section))
        for tok in _SECTION_RE.findall(text):
            available.add(_normalize_section(tok))

    hits = sum(1 for c in cited if c in available)
    return hits / len(cited)


def expected_terms_coverage(question: EvalQuestion, answer: str) -> float:
    """Fraction of `expected_terms` (lowercased substring match) present in the answer."""
    if not question.expected_terms:
        return 1.0
    lowered = (answer or "").lower()
    hits = sum(1 for term in question.expected_terms if term.lower() in lowered)
    return hits / len(question.expected_terms)


def category_correct(question: EvalQuestion, predicted_category: str | None) -> bool:
    return (predicted_category or "") == question.category


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average numeric metric columns across rows."""
    if not rows:
        return {}
    numeric_keys = [
        k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    out: dict[str, float] = {}
    for key in numeric_keys:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        out[key] = sum(values) / len(values) if values else 0.0
    # Boolean accuracies (e.g. category_correct, grounded)
    bool_keys = [k for k, v in rows[0].items() if isinstance(v, bool)]
    for key in bool_keys:
        values = [bool(r.get(key)) for r in rows]
        out[key] = sum(values) / len(values) if values else 0.0
    return out


__all__ = [
    "retrieval_recall_at_k",
    "citation_accuracy",
    "expected_terms_coverage",
    "category_correct",
    "aggregate",
]
