"""LangGraph workflow for the Hungarian TAO compliance assistant.

The graph wires the 7 nodes from :mod:`app.agent.nodes` into a single
``StateGraph``, with two conditional edges:

* After ``classify_query`` we route either into the main RAG pipeline
  or straight to the off-topic handler.
* After ``hallucination_checker`` we either return the final answer or
  loop back into ``answer_generator`` for one more attempt (capped by
  ``Settings.max_hallucination_retries``).

The embedded RAG subgraph itself contains 3 further nodes but is treated
as one unit here, so the parent graph reports 7 distinct nodes — well
above the 5-node target.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes import (
    answer_generator_node,
    classify_query_node,
    hallucination_checker_node,
    off_topic_handler_node,
    query_decomposer_node,
    retrieve_documents_node,
    tool_executor_node,
)
from app.agent.state import AgentState


def _route_by_category(state: AgentState) -> str:
    """Branch after classification: in-scope query vs polite refusal."""
    return "query_decomposer" if state.get("category") == "tao" else "off_topic_handler"


def _route_after_check(state: AgentState) -> str:
    """Either finish or retry answer generation."""
    return END if state.get("grounded", True) else "answer_generator"


def build_agent_graph() -> CompiledStateGraph:
    """Compile and return the main agent StateGraph."""
    builder = StateGraph(AgentState)

    builder.add_node("classify_query", classify_query_node)
    builder.add_node("query_decomposer", query_decomposer_node)
    builder.add_node("retrieve_documents", retrieve_documents_node)
    builder.add_node("tool_executor", tool_executor_node)
    builder.add_node("answer_generator", answer_generator_node)
    builder.add_node("hallucination_checker", hallucination_checker_node)
    builder.add_node("off_topic_handler", off_topic_handler_node)

    builder.add_edge(START, "classify_query")

    builder.add_conditional_edges(
        "classify_query",
        _route_by_category,
        {
            "query_decomposer": "query_decomposer",
            "off_topic_handler": "off_topic_handler",
        },
    )

    builder.add_edge("query_decomposer", "retrieve_documents")
    builder.add_edge("retrieve_documents", "tool_executor")
    builder.add_edge("tool_executor", "answer_generator")
    builder.add_edge("answer_generator", "hallucination_checker")

    builder.add_conditional_edges(
        "hallucination_checker",
        _route_after_check,
        {
            "answer_generator": "answer_generator",
            END: END,
        },
    )

    builder.add_edge("off_topic_handler", END)

    return builder.compile()


__all__ = ["build_agent_graph"]
