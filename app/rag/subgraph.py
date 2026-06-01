"""Modular RAG subgraph used by the main agentic workflow.

The subgraph implements the textbook *transform -> retrieve -> grade*
flow as three LangGraph nodes:

1. ``query_transform`` rewrites the raw user question into a search-
   friendly Hungarian formulation (legal vocabulary, no pronouns).
2. ``retrieve`` performs a Chroma top-k similarity search.
3. ``grade_documents`` asks an LLM to keep only the chunks that are
   actually relevant to the (transformed) question. With the dummy
   provider this step short-circuits and keeps everything, so the whole
   pipeline can run offline in CI and load tests.

This subgraph is callable from the main workflow as a single unit and
**does not** count towards the 5+ node target of the parent graph.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm import get_chat_model
from app.llm.dummy import DummyChatModel
from app.rag.retriever import get_retriever


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class RAGState(TypedDict, total=False):
    """State carried through the RAG subgraph."""

    query: str
    rewritten_query: str
    retrieved_docs: list[Document]
    relevant_docs: list[Document]


# ---------------------------------------------------------------------------
# Structured output for the grading step
# ---------------------------------------------------------------------------
class RelevanceVerdict(BaseModel):
    """LLM-as-grader output for a single retrieved chunk."""

    is_relevant: bool = Field(
        description=(
            "True if the chunk contains information that helps answer the "
            "question. False if it is off-topic or only tangentially related."
        )
    )
    reason: str = Field(
        default="",
        description="One short sentence explaining the decision (Hungarian).",
    )


# ---------------------------------------------------------------------------
# Prompts (kept short and Hungarian-first for the target domain)
# ---------------------------------------------------------------------------
_QUERY_TRANSFORM_PROMPT = (
    "Fogalmazd át az alábbi kérdést jogszabály-keresésre optimalizált, "
    "tömör formára magyarul. Tartsd meg a szakkifejezéseket (pl. ÁFA, TAO, "
    "fordított adózás), de hagyd el a kérdőszavakat és a felesleges szavakat. "
    "Csak az átfogalmazott kérdést add vissza, semmi mást.\n\n"
    "Kérdés: {query}"
)

_GRADE_PROMPT = (
    "Te egy magyar számviteli/adójogi szakértő vagy. Döntsd el, hogy az "
    "alábbi forrásrészlet RELEVÁNS-e a feltett kérdés megválaszolásához.\n\n"
    "Kérdés: {query}\n\n"
    "Forrásrészlet:\n{chunk}\n\n"
    "Add vissza a döntést a megadott szerkezetben."
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def _query_transform_node(state: RAGState) -> RAGState:
    query = state["query"]
    chat = get_chat_model("main")
    if isinstance(chat, DummyChatModel):
        # Skip LLM rewriting under the dummy provider for determinism.
        return {"rewritten_query": query}
    response = chat.invoke([HumanMessage(content=_QUERY_TRANSFORM_PROMPT.format(query=query))])
    rewritten = (response.content or "").strip() or query
    return {"rewritten_query": rewritten}


def _retrieve_node(state: RAGState) -> RAGState:
    query = state.get("rewritten_query") or state["query"]
    settings = get_settings()
    retriever = get_retriever(top_k=settings.rag_top_k)
    docs = retriever.invoke(query)
    return {"retrieved_docs": docs}


def _grade_documents_node(state: RAGState) -> RAGState:
    docs = state.get("retrieved_docs", [])
    query = state.get("rewritten_query") or state["query"]
    chat = get_chat_model("main")

    # Dummy provider can't honour structured output; keep everything.
    if isinstance(chat, DummyChatModel) or not docs:
        return {"relevant_docs": docs}

    grader = chat.with_structured_output(RelevanceVerdict)
    kept: list[Document] = []
    for doc in docs:
        prompt = _GRADE_PROMPT.format(query=query, chunk=doc.page_content)
        try:
            verdict: RelevanceVerdict = grader.invoke(prompt)  # type: ignore[assignment]
        except Exception:
            # If the model fails to produce valid structured output, be
            # conservative and keep the chunk so we don't lose recall.
            kept.append(doc)
            continue
        if verdict.is_relevant:
            kept.append(doc)
    # Always return at least one doc to give the answer generator a chance.
    return {"relevant_docs": kept or docs}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_rag_subgraph() -> CompiledStateGraph:
    """Compile and return the RAG subgraph."""
    builder = StateGraph(RAGState)
    builder.add_node("query_transform", _query_transform_node)
    builder.add_node("retrieve", _retrieve_node)
    builder.add_node("grade_documents", _grade_documents_node)

    builder.add_edge(START, "query_transform")
    builder.add_edge("query_transform", "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    builder.add_edge("grade_documents", END)
    return builder.compile()

