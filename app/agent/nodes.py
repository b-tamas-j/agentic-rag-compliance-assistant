"""Node implementations for the compliance-assistant workflow.

Each function takes an :class:`AgentState` slice and returns a partial
update dict. All nodes are LLM-aware but degrade gracefully under the
dummy provider so the whole graph runs offline in tests and CI.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agent.state import AgentState
from app.agent.tools import legal_reference_validator, tao_calculator
from app.config import get_settings
from app.llm import get_chat_model
from app.llm.dummy import DummyChatModel
from app.rag.subgraph import build_rag_subgraph

# Cached compiled RAG subgraph (built lazily on first call).
_rag_subgraph = None


def _get_rag_subgraph():
    global _rag_subgraph
    if _rag_subgraph is None:
        _rag_subgraph = build_rag_subgraph()
    return _rag_subgraph


# Keywords that mark a query as in-scope (Hungarian corporate income tax).
# Used as a deterministic fallback and by the dummy provider.
_TAO_KEYWORDS = (
    "társasági adó", "tao", "adóalap", "elhatárolt veszteség",
    "növekedési adóhitel", "nahi", "tao-felajánlás", "nonprofit",
    "vagyonkezelő alapítvány", "adómérték", "adókedvezmény",
    "vesztesegelhatarolas",
)


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------
class ClassificationVerdict(BaseModel):
    category: str = Field(
        description="Either 'tao' (Hungarian corporate income tax) or 'off_topic'."
    )


class GroundednessVerdict(BaseModel):
    grounded: bool = Field(
        description="True iff every factual claim in the draft is supported by the sources."
    )
    reason: str = Field(default="", description="Short Hungarian explanation.")


# ---------------------------------------------------------------------------
# 1. classify_query
# ---------------------------------------------------------------------------
_CLASSIFY_PROMPT = (
    "Te egy magyar adójogi asszisztens vagy. Döntsd el, hogy az alábbi "
    "kérdés a TÁRSASÁGI ADÓ (Tao. tv.) témakörébe tartozik-e.\n\n"
    "Kérdés: {query}\n\n"
    "Válaszolj a megadott szerkezetben: 'tao' vagy 'off_topic'."
)


def classify_query_node(state: AgentState) -> dict[str, Any]:
    """Decide whether the question is in scope (TAO) or off-topic."""
    query = state["query"].lower()
    chat = get_chat_model("fast")

    if isinstance(chat, DummyChatModel):
        # Deterministic keyword match for the offline pipeline.
        category = "tao" if any(kw in query for kw in _TAO_KEYWORDS) else "off_topic"
        return {"category": category}

    classifier = chat.with_structured_output(ClassificationVerdict)
    try:
        verdict: ClassificationVerdict = classifier.invoke(  # type: ignore[assignment]
            _CLASSIFY_PROMPT.format(query=state["query"])
        )
        category = "tao" if verdict.category.lower().strip() == "tao" else "off_topic"
    except Exception:
        category = "tao" if any(kw in query for kw in _TAO_KEYWORDS) else "off_topic"
    return {"category": category}


# ---------------------------------------------------------------------------
# 2. query_decomposer
# ---------------------------------------------------------------------------
_DECOMPOSE_PROMPT = (
    "Bontsd az alábbi magyar adójogi kérdést 1-3 önállóan kereshető, "
    "rövid alkérdésre. Minden alkérdés külön sorba kerüljön, számozás nélkül. "
    "Ha a kérdés már önmagában elég egyszerű, add vissza egyetlen sorként.\n\n"
    "Kérdés: {query}"
)


def query_decomposer_node(state: AgentState) -> dict[str, Any]:
    """Split a complex query into 1-3 retrievable sub-questions."""
    query = state["query"]
    chat = get_chat_model("main")

    if isinstance(chat, DummyChatModel):
        return {"sub_queries": [query]}

    try:
        response = chat.invoke([HumanMessage(content=_DECOMPOSE_PROMPT.format(query=query))])
        lines = [ln.strip(" -•\t") for ln in (response.content or "").splitlines() if ln.strip()]
        sub_queries = lines[:3] or [query]
    except Exception:
        sub_queries = [query]
    return {"sub_queries": sub_queries}


# ---------------------------------------------------------------------------
# 3. retrieve_documents  (delegates to the RAG subgraph for each sub-query)
# ---------------------------------------------------------------------------
def retrieve_documents_node(state: AgentState) -> dict[str, Any]:
    """Run the RAG subgraph for every sub-query and merge the results."""
    sub_queries = state.get("sub_queries") or [state["query"]]
    rag = _get_rag_subgraph()

    seen: set[str] = set()
    merged: list[Document] = []
    for sq in sub_queries:
        result = rag.invoke({"query": sq})
        for doc in result.get("relevant_docs", []) or result.get("retrieved_docs", []):
            key = f"{doc.metadata.get('source', '')}::{doc.metadata.get('chunk_id', '')}::{doc.page_content[:40]}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
    return {"retrieved_docs": merged}


# ---------------------------------------------------------------------------
# 4. tool_executor
# ---------------------------------------------------------------------------
# Pattern: "1 000 000 Ft adóalap" / "1000000 Ft" / "1,000,000 HUF".
_NUMBER_RE = re.compile(r"(\d[\d\s.,]*)\s*(?:Ft|HUF|forint)", re.IGNORECASE)
_SECTION_RE = re.compile(r"\d+(?:/[A-ZÁÉÍÓÖŐÚÜŰ])?\.\s*§")


def _parse_huf(text: str) -> float | None:
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    raw = match.group(1).replace(" ", "").replace(",", "").replace(".", "")
    try:
        return float(raw)
    except ValueError:
        return None


def tool_executor_node(state: AgentState) -> dict[str, Any]:
    """Invoke deterministic tools whose pre-conditions are visible in the query.

    * Calls ``tao_calculator`` whenever the query mentions a HUF amount
      alongside tax/adóalap wording.
    * Calls ``legal_reference_validator`` for every ``"N. §"`` token in
      the query.
    """
    query = state["query"]
    results: list[dict] = []

    amount = _parse_huf(query)
    if amount is not None and ("adó" in query.lower() or "tao" in query.lower()):
        try:
            results.append(
                {
                    "tool": "tao_calculator",
                    "output": tao_calculator.invoke({"tax_base_huf": amount}),
                }
            )
        except Exception as exc:  # defensive; tool already validates inputs
            results.append({"tool": "tao_calculator", "error": str(exc)})

    for citation in _SECTION_RE.findall(query):
        try:
            results.append(
                {
                    "tool": "legal_reference_validator",
                    "output": legal_reference_validator.invoke({"citation": citation}),
                }
            )
        except Exception as exc:
            results.append({"tool": "legal_reference_validator", "error": str(exc)})

    return {"tool_results": results}


# ---------------------------------------------------------------------------
# 5. answer_generator
# ---------------------------------------------------------------------------
_ANSWER_PROMPT = (
    "Te egy magyar társasági adó szakértő vagy. Válaszolj az alábbi kérdésre "
    "kizárólag a megadott források alapján. Hivatkozz a paragrafusokra (pl. "
    "'Tao. tv. 19. §'), ahol releváns. Ha a források nem fedik le a kérdést, "
    "ezt írd le őszintén.\n\n"
    "Kérdés: {query}\n\n"
    "Források:\n{context}\n\n"
    "Eszközök eredménye (ha van): {tools}\n\n"
    "Válasz magyarul:"
)


def _format_context(docs: list[Document]) -> str:
    if not docs:
        return "(nincs releváns forrás)"
    lines = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "?")
        section = doc.metadata.get("section") or "?"
        lines.append(f"[{i}] ({src}, {section}) {doc.page_content[:400]}")
    return "\n".join(lines)


def answer_generator_node(state: AgentState) -> dict[str, Any]:
    """Draft an answer using retrieved docs and tool outputs."""
    docs = state.get("retrieved_docs", [])
    tools = state.get("tool_results", [])
    query = state["query"]
    chat = get_chat_model("main")

    if isinstance(chat, DummyChatModel):
        # Deterministic offline draft: stitch together first chunk + tool data.
        first = docs[0].page_content[:200] if docs else "(nincs forrás)"
        tool_text = "; ".join(
            t.get("output", {}).get("explanation", "") or t.get("tool", "")
            for t in tools
        )
        draft = f"[dummy] Forrás: {first} | Eszközök: {tool_text}".strip()
        return {"draft_answer": draft}

    prompt = _ANSWER_PROMPT.format(
        query=query,
        context=_format_context(docs),
        tools=tools or "(nincs)",
    )
    response = chat.invoke([HumanMessage(content=prompt)])
    return {"draft_answer": (response.content or "").strip()}


# ---------------------------------------------------------------------------
# 6. hallucination_checker
# ---------------------------------------------------------------------------
_GROUNDEDNESS_PROMPT = (
    "Ellenőrizd, hogy az alábbi VÁLASZ minden ténymegállapítása alá van-e "
    "támasztva a megadott FORRÁSOK-kal. Ha bármi nem szerepel a forrásokban, "
    "akkor 'grounded=false'. Ha minden állítás visszavezethető a forrásokra, "
    "akkor 'grounded=true'.\n\n"
    "FORRÁSOK:\n{context}\n\n"
    "VÁLASZ:\n{answer}"
)


def hallucination_checker_node(state: AgentState) -> dict[str, Any]:
    """Judge whether the draft answer is grounded in the retrieved sources."""
    draft = state.get("draft_answer", "")
    docs = state.get("retrieved_docs", [])
    chat = get_chat_model("judge")

    retries = state.get("hallucination_retries", 0)

    if isinstance(chat, DummyChatModel):
        # Dummy: trust the draft (load tests / CI shouldn't loop forever).
        return {
            "grounded": True,
            "final_answer": draft,
            "hallucination_retries": retries,
        }

    judge = chat.with_structured_output(GroundednessVerdict)
    try:
        verdict: GroundednessVerdict = judge.invoke(  # type: ignore[assignment]
            _GROUNDEDNESS_PROMPT.format(context=_format_context(docs), answer=draft)
        )
        grounded = bool(verdict.grounded)
    except Exception:
        # On judge failure, accept the draft to avoid infinite retries.
        grounded = True

    settings = get_settings()
    if not grounded and retries < settings.max_hallucination_retries:
        return {"grounded": False, "hallucination_retries": retries + 1}
    # Either grounded, or out of retries -> accept current draft.
    return {"grounded": True, "final_answer": draft, "hallucination_retries": retries}


# ---------------------------------------------------------------------------
# 7. off_topic_handler
# ---------------------------------------------------------------------------
_OFF_TOPIC_MESSAGE = (
    "Sajnálom, ez az asszisztens csak magyar társasági adó (Tao. tv.) "
    "kérdésekben tud segíteni. Kérlek, tedd fel a kérdést TAO témában!"
)


def off_topic_handler_node(state: AgentState) -> dict[str, Any]:
    """Return a polite refusal for off-topic questions."""
    return {"final_answer": _OFF_TOPIC_MESSAGE, "grounded": True}


__all__ = [
    "classify_query_node",
    "query_decomposer_node",
    "retrieve_documents_node",
    "tool_executor_node",
    "answer_generator_node",
    "hallucination_checker_node",
    "off_topic_handler_node",
]
