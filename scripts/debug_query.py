"""Run a single query through the agent graph, no UI.

Convenient entry point for the VS Code debugger: set breakpoints in
`app/agent/nodes.py`, hit F5 with the "Debug: Single query (no UI)"
configuration, and step through every node.

Usage:
    uv run python scripts/debug_query.py
    uv run python scripts/debug_query.py "Mennyi a TAO 10 000 000 Ft adoalapra?"
"""

from __future__ import annotations

import logging
import sys

from app.agent import build_agent_graph
from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Verbose logs for our own modules; keep third-party (httpx, chromadb) quieter.
logging.getLogger("app").setLevel(logging.DEBUG)

DEFAULT_QUERY = "Mennyi a társasági adó mértéke?"


def _read_query() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    try:
        entered = input(f"Kérdés (Enter = default: {DEFAULT_QUERY!r}): ").strip()
    except EOFError:
        entered = ""
    return entered or DEFAULT_QUERY


def main() -> None:
    query = _read_query()
    settings = get_settings()
    print(f"Provider:    {settings.llm_provider}")
    print(f"Main model:  {settings.ollama_model}")
    print(f"Fast model:  {settings.ollama_fast_model}")
    print(f"Judge model: {settings.ollama_judge_model}")
    print(f"Query:       {query}")
    print("-" * 60)

    graph = build_agent_graph()
    final_state = graph.invoke({"query": query})

    print("-" * 60)
    print(f"Category:    {final_state.get('category')}")
    print(f"Sub-queries: {final_state.get('sub_queries')}")
    print(f"Docs:        {len(final_state.get('retrieved_docs', []))}")
    print(f"Tools:       {len(final_state.get('tool_results', []))}")
    print(f"Grounded:    {final_state.get('grounded')}")
    print(f"Retries:     {final_state.get('hallucination_retries')}")
    print()
    print("Answer:")
    print(final_state.get("final_answer") or "(empty)")


if __name__ == "__main__":
    main()
