"""State definition for the main agentic workflow.

A single :class:`AgentState` carries all data between LangGraph nodes.
Every field is optional (``total=False``) so individual nodes can
contribute incrementally without re-declaring the full state.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.documents import Document


QueryCategory = Literal["tao", "off_topic"]


class AgentState(TypedDict, total=False):
    """State carried through the compliance-assistant workflow."""

    # --- Input ---
    query: str

    # --- Classification + decomposition ---
    category: QueryCategory
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
