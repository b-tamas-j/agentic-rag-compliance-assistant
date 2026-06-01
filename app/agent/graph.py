"""LangGraph workflow definition (skeleton).

Implemented in `feat/agent-workflow` branch. This module will assemble the
agent's StateGraph: classify -> decompose -> RAG subgraph -> tools ->
answer -> hallucination check (with retry).
"""

from __future__ import annotations


def build_agent_graph():  # pragma: no cover - skeleton
    """Construct the LangGraph StateGraph for the compliance assistant."""
    raise NotImplementedError("Agent graph is implemented in a later branch.")
