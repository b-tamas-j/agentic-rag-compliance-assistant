"""State definition for the main agentic workflow.

A single :class:`AgentState` carries all data between LangGraph nodes.
Every field is optional (``total=False``) so individual nodes can
contribute incrementally without re-declaring the full state.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.documents import Document


QueryCategory = Literal["tao", "off_topic"]

# Coarse topical hint used to narrow Chroma/BM25 retrieval to one or two
# source PDFs. ``None`` (or ``"general"``) means no filter.
SourceHint = Literal[
    "nonprofit",     # 13+A+nonprofit+szervezetek+adózása+2025.01.27.pdf
    "calculation",   # 41 A társasági adó legfontosabb szabályai 2025.09.01.pdf
    "offering",      # 55 Tao-felajánlás 2025.07.21.pdf
    "credit",        # 93 Növekedési adóhitel 2025.01.17.pdf
    "general",
]


class AgentState(TypedDict, total=False):
    """State carried through the compliance-assistant workflow."""

    # --- Input ---
    query: str

    # --- Classification + decomposition ---
    category: QueryCategory
    source_hint: SourceHint
    sub_queries: list[str]

    # --- Retrieval ---
    retrieved_docs: list[Document]

    # --- Tool execution ---
    tool_results: list[dict]

    # --- Answer generation + verification ---
    draft_answer: str
    hallucination_retries: int
    grounded: bool
    final_answer: str
