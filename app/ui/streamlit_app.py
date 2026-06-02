"""Streamlit chat UI for the Hungarian TAO compliance assistant.

Run locally:
    uv run streamlit run app/ui/streamlit_app.py

The UI streams each agent node as it executes, surfaces the
deterministic tool calls (`tao_calculator`, `legal_reference_validator`)
in dedicated cards, and lists the source PDF chunks the answer was
grounded on.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import streamlit as st

from app.agent import build_agent_graph
from app.config import get_settings

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="TAO Compliance Assistant",
    page_icon="📑",
    layout="wide",
)

# Human-readable labels for the seven nodes (in pipeline order).
_NODE_LABELS: dict[str, str] = {
    "classify_query": "1. Kérdés osztályozása",
    "query_decomposer": "2. Alkérdésekre bontás",
    "retrieve_documents": "3. Forrásdokumentumok keresése",
    "tool_executor": "4. Eszközhívások",
    "answer_generator": "5. Válasz generálása",
    "hallucination_checker": "6. Forrásalapúság ellenőrzése",
    "off_topic_handler": "Témán kívüli kérdés kezelése",
}


@st.cache_resource(show_spinner="Agent graph fordítása…")
def _get_graph():
    """Compile the agent graph once per Streamlit process (with in-memory checkpointer)."""
    return build_agent_graph(with_memory=True)


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[dict(role, content, meta?)]
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


def _render_sidebar() -> None:
    settings = get_settings()
    with st.sidebar:
        st.header("Beállítások")
        st.markdown(f"**LLM provider:** `{settings.llm_provider}`")
        if settings.llm_provider == "ollama":
            st.markdown(f"**Fő modell:** `{settings.ollama_model}`")
            st.markdown(f"**Gyors modell:** `{settings.ollama_fast_model}`")
            st.markdown(f"**Bíró modell:** `{settings.ollama_judge_model}`")
            st.markdown(f"**Embedding:** `{settings.ollama_embedding_model}`")
        st.markdown(f"**Top-K:** {settings.rag_top_k}")
        st.markdown(f"**Max retry:** {settings.max_hallucination_retries}")
        st.divider()
        st.caption(f"Session: `{st.session_state.thread_id[:8]}…`")
        if st.button("Új beszélgetés", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        st.divider()
        st.caption(
            "A válasz kizárólag az indexelt NAV TAO tájékoztatókra "
            "épül; jogi tanácsadásnak nem minősül."
        )


def _render_tool_card(tool_result: dict[str, Any]) -> None:
    tool = tool_result.get("tool", "?")
    if "error" in tool_result:
        st.error(f"**{tool}** hiba: {tool_result['error']}")
        return

    output = tool_result.get("output", {})
    if tool == "tao_calculator":
        cols = st.columns(3)
        cols[0].metric("Adóalap", f"{output.get('tax_base_huf', 0):,.0f} Ft".replace(",", " "))
        cols[1].metric(
            "Levonható veszteség",
            f"{output.get('loss_applied_huf', 0):,.0f} Ft".replace(",", " "),
        )
        cols[2].metric("Számított TAO", f"{output.get('tax_huf', 0):,.0f} Ft".replace(",", " "))
        st.caption(output.get("explanation", ""))
    elif tool == "legal_reference_validator":
        section = output.get("section") or output.get("citation")
        if output.get("found"):
            st.success(
                f"**{section}** megtalálva — {output.get('match_count', 0)} chunk; "
                f"források: {', '.join(output.get('sources', []) or ['?'])}"
            )
        else:
            st.warning(f"**{section}** nem található az indexelt korpuszban.")
    else:
        st.json(tool_result)


def _render_sources(docs: list[Any]) -> None:
    if not docs:
        return
    st.markdown("**Felhasznált források:**")
    for i, doc in enumerate(docs, 1):
        meta = getattr(doc, "metadata", {}) or {}
        src = meta.get("source", "?")
        section = meta.get("section") or "?"
        page = meta.get("page", "?")
        title = f"[{i}] {src} — {section} (oldal {page})"
        with st.expander(title):
            st.write(doc.page_content)


def _render_message(msg: dict[str, Any]) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta") or {}
        for tool_result in meta.get("tool_results", []) or []:
            _render_tool_card(tool_result)
        _render_sources(meta.get("retrieved_docs") or [])


def _process_query(query: str) -> dict[str, Any]:
    """Stream the graph for one user query, return the final state."""
    graph = _get_graph()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        status = st.status("Agent dolgozik…", expanded=True)
        final_state: dict[str, Any] = {}
        # graph.stream yields {node_name: update_dict} for each node completion.
        for event in graph.stream({"query": query}, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                label = _NODE_LABELS.get(node_name, node_name)
                with status:
                    st.markdown(f"✅ **{label}**")
                final_state.update(update or {})

        status.update(label="Kész", state="complete", expanded=False)

        # Render the final answer + tool cards + sources in the same bubble.
        final_answer = final_state.get("final_answer") or final_state.get("draft_answer", "")
        st.markdown(final_answer)
        for tool_result in final_state.get("tool_results", []) or []:
            _render_tool_card(tool_result)
        _render_sources(final_state.get("retrieved_docs") or [])

    return final_state


def main() -> None:
    _init_session()
    _render_sidebar()

    st.title("📑 TAO Compliance Assistant")
    st.caption(
        "Magyar társasági adó (Tao. tv.) kérdés-válasz asszisztens "
        "NAV tájékoztatók alapján."
    )

    for msg in st.session_state.messages:
        _render_message(msg)

    user_query = st.chat_input("Tedd fel a TAO-val kapcsolatos kérdésed…")
    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    final_state = _process_query(user_query)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_state.get("final_answer")
            or final_state.get("draft_answer", "(üres válasz)"),
            "meta": {
                "tool_results": final_state.get("tool_results", []),
                "retrieved_docs": final_state.get("retrieved_docs", []),
            },
        }
    )


if __name__ == "__main__":
    main()
