"""Legal reference validator — verifies that an ``N. §`` citation exists in the corpus."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Matches "19. §", "19.§", "19/A. §", "17. § (2)" etc.
_SECTION_RE = re.compile(r"(\d+(?:/[A-ZÁÉÍÓÖŐÚÜŰ])?)\s*\.\s*§")


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
    logger.debug("legal_reference_validator called: citation=%r", citation)
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


__all__ = ["legal_reference_validator"]
