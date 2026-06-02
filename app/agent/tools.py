"""Deterministic tools used by the agentic workflow.

Two tools are exposed:

* :func:`tao_calculator` — computes Hungarian corporate income tax
  (társasági adó / TAO) on a given tax base, optionally accounting for
  carried-forward losses (Tao. tv. §17).
* :func:`legal_reference_validator` — verifies that a section reference
  written by the agent (e.g. ``"19. §"``) actually appears in the
  indexed corpus, so the chat answer can be flagged as grounded.

Both tools are pure / deterministic — no LLM call inside — and are
declared with :func:`langchain_core.tools.tool` so they can be bound to
a LangGraph node or a ``ChatModel.bind_tools`` call later on.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import tool

# Standard Hungarian corporate income tax rate (Tao. tv. §19 (1)).
TAO_RATE: float = 0.09

# A carried-forward loss can only offset up to 50% of the tax base
# (Tao. tv. §17 (2)). Without this cap, "elhatárolt veszteség" abuse
# could zero out the base entirely.
LOSS_OFFSET_CAP: float = 0.50

# Matches "19. §", "19.§", "19/A. §", "17. § (2)" etc.
_SECTION_RE = re.compile(
    r"(\d+(?:/[A-ZÁÉÍÓÖŐÚÜŰ])?)\s*\.\s*§"
)


# ---------------------------------------------------------------------------
# 1. TAO calculator
# ---------------------------------------------------------------------------
@tool("tao_calculator", return_direct=False)
def tao_calculator(
    tax_base_huf: float,
    loss_carried_forward_huf: float = 0.0,
) -> dict[str, Any]:
    """Compute Hungarian corporate income tax (TAO) on a given tax base.

    Use this whenever the user asks for a numeric tax amount or wants to
    know the effect of carrying forward a prior-year loss. The standard
    rate is 9%. A carried-forward loss reduces the tax base, but only up
    to 50% of it (Tao. tv. §17).

    Args:
        tax_base_huf: Pre-tax positive tax base in HUF (``adóalap``).
        loss_carried_forward_huf: Optional prior-year carried-forward
            loss in HUF. Capped at 50% of the tax base.

    Returns:
        Dict with the adjusted base, the computed tax, the effective
        rate, the loss actually applied and a short human-readable
        explanation (Hungarian) suitable for chat output.
    """
    if tax_base_huf < 0:
        raise ValueError("tax_base_huf must be non-negative")
    if loss_carried_forward_huf < 0:
        raise ValueError("loss_carried_forward_huf must be non-negative")

    max_offset = tax_base_huf * LOSS_OFFSET_CAP
    applied_loss = min(loss_carried_forward_huf, max_offset)
    adjusted_base = tax_base_huf - applied_loss
    tax = round(adjusted_base * TAO_RATE, 2)
    effective_rate = (tax / tax_base_huf) if tax_base_huf > 0 else 0.0

    explanation = (
        f"Adóalap: {tax_base_huf:,.0f} Ft. "
        f"Elhatárolt veszteségből levonható: {applied_loss:,.0f} Ft "
        f"(max. az adóalap 50%-a, Tao. tv. 17. §). "
        f"Módosított adóalap: {adjusted_base:,.0f} Ft. "
        f"Társasági adó (9%): {tax:,.0f} Ft."
    ).replace(",", " ")  # Hungarian thousands separator

    return {
        "tax_base_huf": tax_base_huf,
        "loss_applied_huf": applied_loss,
        "adjusted_base_huf": adjusted_base,
        "tax_huf": tax,
        "effective_rate": round(effective_rate, 4),
        "rate": TAO_RATE,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# 2. Legal reference validator
# ---------------------------------------------------------------------------
@tool("legal_reference_validator", return_direct=False)
def legal_reference_validator(citation: str) -> dict[str, Any]:
    """Verify that a Hungarian legal section reference exists in the corpus.

    Use this to fact-check any ``"N. §"`` style citation the model
    produces, before showing it to the user. The check is a literal
    substring lookup against the indexed Chroma chunks; no LLM is
    invoked.

    Args:
        citation: Any string that contains a section reference, e.g.
            ``"Tao. tv. 19. §"`` or just ``"17/A. §"``. Only the
            ``N. §`` (or ``N/L. §``) token is used for the lookup;
            surrounding text is ignored.

    Returns:
        Dict with the parsed section, whether it was found in any
        indexed chunk, how many chunks matched, and the list of source
        documents the section appears in (deduplicated).
    """
    match = _SECTION_RE.search(citation)
    if not match:
        return {
            "citation": citation,
            "section": None,
            "found": False,
            "match_count": 0,
            "sources": [],
            "reason": "No '§' reference parsed from input.",
        }

    section_token = f"{match.group(1)}. §"

    # Lazy import to keep the tool importable even when Chroma deps are
    # missing in some minimal environments.
    from app.rag.retriever import get_vector_store

    store = get_vector_store()
    # ``get(where=...)`` lets us scan the collection without an embedding
    # call. We accept either an exact metadata.section match or a
    # substring hit in the chunk text (covers chunks where the splitter
    # could not tag the section header — e.g. NAV explanatory leaflets).
    raw = store.get(where={"section": section_token})
    docs_meta = raw.get("metadatas") or []

    if not docs_meta:
        full = store.get()
        all_meta = full.get("metadatas") or []
        all_text = full.get("documents") or []
        docs_meta = [m for m, t in zip(all_meta, all_text) if section_token in (t or "")]

    sources = sorted({(m or {}).get("source", "?") for m in docs_meta})
    return {
        "citation": citation,
        "section": section_token,
        "found": bool(docs_meta),
        "match_count": len(docs_meta),
        "sources": sources,
    }


__all__ = ["tao_calculator", "legal_reference_validator", "TAO_RATE", "LOSS_OFFSET_CAP"]
